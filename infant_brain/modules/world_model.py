"""
World Model V3 — BrainEncoder + Transformer Dynamics.

Perception pipeline (mimics human vision):
  1. BIG PICTURE — ViT sees whole scene, all patches relate to each other
  2. DETAILS    — CNN refines fine-grained features within patches
  3. ATTENTION  — focus on what matters right now
  4. MEMORY     — GRU remembers across time (motion, history)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectSlots(nn.Module):
    """Slot Attention with reconstruction — discovers objects from patches.

    K slots compete to claim patches. A decoder reconstructs the original
    patches from the slots, forcing each slot to capture a different part
    of the scene. Without reconstruction, slots collapse to the same thing.
    """

    def __init__(self, embed_dim=128, n_slots=4, slot_dim=64, n_iters=3):
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.n_iters = n_iters
        self.embed_dim = embed_dim

        self.slot_mu = nn.Parameter(torch.randn(1, n_slots, slot_dim) * (slot_dim ** -0.5))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, n_slots, slot_dim))

        self.project_k = nn.Linear(embed_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(embed_dim, slot_dim, bias=False)

        self.norm_input = nn.LayerNorm(embed_dim)
        self.norm_slot = nn.LayerNorm(slot_dim)

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2), nn.GELU(),
            nn.Linear(slot_dim * 2, slot_dim),
        )
        self.norm_mlp = nn.LayerNorm(slot_dim)

        self.project_out = nn.Linear(n_slots * slot_dim, embed_dim)

        self.slot_decoder = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2), nn.GELU(),
            nn.Linear(slot_dim * 2, embed_dim),
        )

        self._scale = slot_dim ** -0.5

    def forward(self, features):
        """
        Args:
            features: (B, N, embed_dim) patch features from ViT
        Returns:
            combined: (B, embed_dim) slot-structured summary
            slots: (B, n_slots, slot_dim) raw slot representations
            recon_loss: scalar reconstruction loss (forces slots apart)
        """
        B, N, _ = features.shape
        inputs = self.norm_input(features)
        k = self.project_k(inputs)
        v = self.project_v(inputs)

        slots = self.slot_mu + self.slot_log_sigma.exp() * torch.randn(
            B, self.n_slots, self.slot_dim, device=features.device)

        for _ in range(self.n_iters):
            q = self.norm_slot(slots)
            attn_logits = torch.bmm(q, k.transpose(1, 2)) * self._scale
            attn = F.softmax(attn_logits, dim=1)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            updates = torch.bmm(attn, v)
            slots = self.gru(
                updates.reshape(B * self.n_slots, self.slot_dim),
                slots.reshape(B * self.n_slots, self.slot_dim),
            ).reshape(B, self.n_slots, self.slot_dim)
            slots = slots + self.mlp(self.norm_mlp(slots))

        combined = self.project_out(slots.reshape(B, -1))

        decoded = self.slot_decoder(slots)
        attn_masks = F.softmax(attn_logits, dim=1).transpose(1, 2)
        recon = torch.bmm(attn_masks, decoded)
        recon_loss = F.mse_loss(recon, features.detach())

        return combined, slots, recon_loss


class _ConvBlock(nn.Module):
    """3x3 conv with GroupNorm + GELU + residual, operating on (B, C, H, W)."""

    def __init__(self, channels):
        super().__init__()
        # GroupNorm with 8 groups is a safe choice for embed_dim 64-256.
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        self.norm = nn.GroupNorm(groups, channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.conv(self.norm(x)))


class BrainEncoder(nn.Module):
    """See everything → Find details → Focus → Remember.

    Step 1 (Vision front-end): produce per-patch features. Two backends:
        * "vit": patch embed + Transformer (every patch attends to every
                 other patch from layer 1).
        * "cnn": patch embed + small conv stack (local receptive fields,
                 progressive abstraction). Closer to mammalian V1→IT and
                 cheaper at small image sizes.
    Step 2 (CNN refine): residual MLP polishes the patch features.
    Step 3 (Attn):   Cross-attention with learnable query → focus on what matters.
    Step 4 (GRU):    Combine with temporal memory → track motion over time.
    """

    def __init__(self, img_size=64, patch_size=8, in_channels=3,
                 embed_dim=128, latent_dim=64, depth=4, n_heads=4, mlp_ratio=4,
                 n_slots=4, slot_dim=64, slot_iters=3,
                 encoder_type: str = "vit"):
        super().__init__()
        assert img_size % patch_size == 0
        assert encoder_type in ("vit", "cnn"), f"unknown encoder_type {encoder_type!r}"
        self.n_patches = (img_size // patch_size) ** 2
        self.latent_dim = latent_dim
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.encoder_type = encoder_type

        # --- 1. BIG PICTURE: patch embedding (shared by both backends) ---
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches, embed_dim) * 0.02
        )

        if encoder_type == "vit":
            # Global self-attention from layer 1.
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=n_heads,
                dim_feedforward=embed_dim * mlp_ratio,
                activation="gelu", batch_first=True,
                norm_first=True, dropout=0.0,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
            self.cnn_blocks = None
        else:
            # Local conv stack on the (B, embed, H', W') feature map.
            # `depth` conv blocks, each: Conv3x3 -> GroupNorm -> GELU + residual.
            self.transformer = None
            self.cnn_blocks = nn.ModuleList([
                _ConvBlock(embed_dim) for _ in range(depth)
            ])
        self.global_norm = nn.LayerNorm(embed_dim)

        # --- 2. DETAILS: small MLP refines patch features ---
        self.detail_refine = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # --- 2b. OBJECT SLOTS: discover entities from patches ---
        self.object_slots = ObjectSlots(embed_dim, n_slots, slot_dim, n_iters=slot_iters)

        # --- 3. ATTENTION: learnable query focuses on what matters ---
        self.focus_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.focus_attn = nn.MultiheadAttention(
            embed_dim, n_heads, batch_first=True, dropout=0.0,
        )
        self.focus_norm = nn.LayerNorm(embed_dim)

        # --- 4. MEMORY: GRU tracks motion and history across time ---
        self.gru = nn.GRUCell(embed_dim, latent_dim)
        self.output_norm = nn.LayerNorm(latent_dim)

        self._hidden = None
        self._last_slots = None
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.focus_query, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def reset_memory(self, batch_size=1, device='cpu'):
        self._hidden = torch.zeros(batch_size, self.latent_dim, device=device)

    def forward(self, x, use_memory=True):
        B = x.shape[0]

        # 1. BIG PICTURE — see the whole scene
        feat_map = self.patch_embed(x)                      # (B, embed, H/p, W/p)

        if self.encoder_type == "vit":
            patches = feat_map.flatten(2).transpose(1, 2)   # (B, n_patches, embed)
            patches = patches + self.pos_embed
            global_feats = self.transformer(patches)        # global self-attention
        else:
            # CNN backend: keep the (B, embed, H', W') feature map and apply
            # local conv blocks. Receptive field grows progressively, like V1→IT.
            for block in self.cnn_blocks:
                feat_map = block(feat_map)
            patches = feat_map.flatten(2).transpose(1, 2)   # (B, n_patches, embed)
            patches = patches + self.pos_embed              # still useful: gives slot attention positional info
            global_feats = patches
        global_feats = self.global_norm(global_feats)

        # 2. DETAILS — refine fine-grained features
        refined = global_feats + self.detail_refine(global_feats)  # residual refinement

        # 2b. OBJECT SLOTS — discover entities from patches
        slot_combined, raw_slots, slot_recon_loss = self.object_slots(refined)
        self._last_slots = raw_slots.detach().float()
        self._last_recon_loss = slot_recon_loss

        # 3. ATTENTION — focus on what matters
        query = self.focus_query.expand(B, -1, -1)         # (B, 1, embed)
        focused, _ = self.focus_attn(query, refined, refined)  # (B, 1, embed)
        summary = self.focus_norm(focused.squeeze(1) + slot_combined)

        # 4. MEMORY — combine with temporal history
        summary = summary.float()
        if use_memory:
            if self._hidden is None or self._hidden.shape[0] != B:
                self._hidden = torch.zeros(B, self.latent_dim, device=x.device)
            self._hidden = self.gru(summary, self._hidden)
            return self.output_norm(self._hidden)
        else:
            h0 = torch.zeros(B, self.latent_dim, device=x.device)
            out = self.gru(summary, h0)
            return self.output_norm(out)


# Backward-compatible alias
ViTEncoder = BrainEncoder
Encoder = BrainEncoder


class DreamerBlock(nn.Module):
    """Pre-norm residual FFN block (DreamerV3-style)."""

    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        return x + self.ff(self.norm(x))


class RecurrentPredictor(nn.Module):
    """RSSM-lite: GRU-based recurrent predictor that maintains context across
    imagination steps. The one-shot TransformerPredictor treated each dream step
    independently — step 12 knew nothing about steps 1-11. This GRU carries a
    hidden state through the entire dream, giving later steps context from earlier ones.

    Two modes:
      - Single-step (h=None): for WM training on real transitions
      - Multi-step (h carried): for dream imagination, accumulates context
    """

    def __init__(self, latent_dim=256, num_actions=6, h_dim=256):
        super().__init__()
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        self.action_embed = nn.Embedding(num_actions, latent_dim)
        self.gru = nn.GRUCell(latent_dim * 2, h_dim)
        self.output = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.SiLU(),
            nn.Linear(h_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, z, action, h=None):
        """Predict next latent state.
        Args:
            z: current latent (B, latent_dim)
            action: action taken (B,) long
            h: optional GRU hidden state (B, h_dim). None = start fresh.
        Returns:
            z_next: predicted next latent (B, latent_dim)
            h_next: updated GRU hidden state (B, h_dim)
        """
        B = z.shape[0]
        if h is None:
            h = torch.zeros(B, self.h_dim, device=z.device, dtype=z.dtype)
        a = self.action_embed(action)
        h = self.gru(torch.cat([z, a], dim=-1), h)
        z_next = self.output(h)
        return z_next, h


class TransformerPredictor(nn.Module):
    """Legacy one-shot predictor. Kept for backward compatibility with checkpoints."""

    def __init__(self, latent_dim=256, num_actions=6, hidden_dim=512,
                 n_layers=4, mlp_ratio=4):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, latent_dim)
        self.input_proj = nn.Linear(latent_dim * 2, hidden_dim)
        self.blocks = nn.ModuleList([
            DreamerBlock(hidden_dim, mlp_ratio) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, z, action, h=None):
        a = self.action_embed(action)
        x = self.input_proj(torch.cat([z, a], dim=-1))
        for block in self.blocks:
            x = block(x)
        return self.output(self.norm(x)), None


Predictor = RecurrentPredictor


def symlog(x):
    """DreamerV3's symlog: compresses large values while preserving sign.
    Makes reward prediction work across games without manual tuning."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    """Inverse of symlog."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def sigreg(z, eps=1e-4):
    """SIGReg anti-collapse regularization (VICReg-style)."""
    B, D = z.shape
    z_c = z - z.mean(dim=0)
    std = z_c.std(dim=0)
    var_loss = F.relu(1.0 - std).mean()
    cov = (z_c.T @ z_c) / (B - 1)
    denom = (std.unsqueeze(0) * std.unsqueeze(1)).clamp(min=eps)
    corr = cov / denom
    mask = ~torch.eye(D, dtype=torch.bool, device=z.device)
    cov_loss = corr[mask].pow(2).mean()
    return var_loss + cov_loss


class WorldModel(nn.Module):
    """V4 World Model: BrainEncoder + RecurrentPredictor (RSSM-lite) + SIGReg.

    Key upgrade from V3: the predictor now carries a GRU hidden state across
    imagination steps. Multi-step dreams no longer degrade because each step
    has context from all previous steps — like remembering where you are in
    a thought experiment.
    """

    def __init__(self, latent_dim=256, num_actions=6, hidden_dim=512,
                 sigreg_weight=0.1, in_channels=3, encoder_type: str = "vit",
                 encoder_depth: int | None = None):
        super().__init__()
        import torch as _torch
        # Architecture autosize:
        #   "GPU big" only when CUDA is available. We deliberately do NOT
        #   include MPS in this check because torch.backends.mps.is_available()
        #   flips between sandboxed and non-sandboxed processes on macOS,
        #   silently changing img_size, patch_size and encoder depth and
        #   producing checkpoints that are not state-dict compatible across
        #   runs (which breaks multi-seed comparability for the paper). MPS
        #   users who actually want the larger model can opt in by passing
        #   encoder_depth=4 explicitly.
        has_gpu = _torch.cuda.is_available()
        img_size = 64 if has_gpu else 32
        patch_size = 8 if has_gpu else 4
        depth = encoder_depth if encoder_depth is not None else (4 if has_gpu else 2)
        embed = min(latent_dim, 256 if has_gpu else 128)
        mlp_r = 4 if has_gpu else 2
        slot_iters = 3 if has_gpu else 2

        self.encoder = BrainEncoder(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels,
            embed_dim=embed, latent_dim=latent_dim, depth=depth,
            n_heads=4, mlp_ratio=mlp_r, n_slots=4, slot_dim=64,
            slot_iters=slot_iters, encoder_type=encoder_type,
        )
        self.predictor = RecurrentPredictor(
            latent_dim=latent_dim, num_actions=num_actions, h_dim=latent_dim,
        )
        self._img_size = img_size
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.sigreg_weight = sigreg_weight

    def encode(self, obs, use_memory=True):
        return self.encoder(obs, use_memory=use_memory)

    @property
    def last_slots(self):
        """Raw slot representations from the most recent encode() call."""
        return self.encoder._last_slots

    @property
    def slot_recon_loss(self):
        """Reconstruction loss from the most recent encode() call."""
        return getattr(self.encoder, '_last_recon_loss', None)

    def reset_memory(self, batch_size=1, device='cpu'):
        self.encoder.reset_memory(batch_size, device)

    def predict_next(self, z, action, h=None):
        """Predict next latent. Returns (z_next, h_next).
        For backward compatibility, callers that ignore h still work."""
        z_next, h_next = self.predictor(z, action, h=h)
        return z_next, h_next

    def imagine_sequence(self, z_start, actions, h=None):
        """Imagine a full sequence of steps with recurrent context.
        Args:
            z_start: initial latent (B, latent_dim)
            actions: sequence of actions (T, B) long
            h: optional initial hidden (B, h_dim)
        Returns:
            z_seq: predicted latents (T+1, B, latent_dim) including z_start
            h_final: final hidden state
        """
        z_seq = [z_start]
        z = z_start
        for t in range(actions.shape[0]):
            z, h = self.predictor(z, actions[t], h=h)
            z_seq.append(z)
        return torch.stack(z_seq), h

    def compute_loss(self, obs, action, next_obs, temporal_weight=0.0, return_z=False):
        z = self.encode(obs, use_memory=False)
        recon1 = self.encoder._last_recon_loss
        z_next_target = self.encode(next_obs, use_memory=False)
        recon2 = self.encoder._last_recon_loss
        z_next_pred, _ = self.predict_next(z, action)
        pred_loss = F.mse_loss(z_next_pred, z_next_target)
        reg_loss = (sigreg(z) + sigreg(z_next_target)) / 2
        slot_recon = (recon1 + recon2) / 2
        total = pred_loss + self.sigreg_weight * reg_loss + 0.5 * slot_recon
        metrics = {
            "loss": total.item(),
            "pred_loss": pred_loss.item(),
            "sigreg_loss": reg_loss.item(),
            "slot_recon": slot_recon.item(),
        }
        if temporal_weight > 0:
            tc_loss = F.mse_loss(z, z_next_target)
            total = total + temporal_weight * tc_loss
            metrics["tc_loss"] = tc_loss.item()
        if return_z:
            return total, metrics, z
        return total, metrics

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
