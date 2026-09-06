"""
The Infant Brain — all 7 modules wired together.

Usage:
    from infant_brain.brain import Brain
    from infant_brain.envs import AtariEnv
    brain = Brain(AtariEnv("Pong"))
    brain.train(episodes=2000)
    brain.save("checkpoints/pong.pt")
"""

import hashlib
import os
import random as _stdlib_random
import struct
import time
from collections import deque
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler


def _seed_everything(seed: int):
    """Make a run reproducible across stdlib random, NumPy, and PyTorch (CPU + CUDA).

    Note: we deliberately do NOT set cudnn.deterministic=True here because it
    halves throughput on conv workloads. Set it manually if you need bitwise
    reproducibility across machines."""
    _stdlib_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stable_concept_id(z_cpu: torch.Tensor, n_concepts: int) -> int:
    """Deterministic pre-cluster concept assignment.

    Python's built-in hash() is salted per-process, so the previous code
    (`hash(tuple(z[:4].tolist())) % n`) produced different concept IDs for
    the same latent on repeated runs, silently breaking reproducibility
    before clustering kicked in. blake2b on the raw float32 bytes is stable."""
    head = z_cpu[:4].detach().float().contiguous().numpy().tobytes()
    digest = hashlib.blake2b(head, digest_size=8).digest()
    return struct.unpack("<Q", digest)[0] % max(n_concepts, 1)

from .modules import (WorldModel, LanguageModule, MetacognitiveMonitor, ValueSystem,
                      EpisodicMemory, SemanticMemory, SleepCycle, KnowledgeTree,
                      SelfCorrectingBrain, IntuitionGate, symlog, symexp)
from .envs.base import BrainEnv


class SignalBus:
    """Modular signal registry — modules register their signals, bus builds the vector.
    Adding a new module = one line: bus.register("my_signal", 1). No SIGNAL_DIM constant."""

    def __init__(self):
        self._signals = {}
        self._order = []

    def register(self, name, dim=1):
        if name not in self._signals:
            self._signals[name] = dim
            self._order.append(name)

    @property
    def total_dim(self):
        return sum(self._signals[k] for k in self._order)

    def build(self, values, device):
        if not hasattr(self, '_buffer') or self._buffer.device != device:
            self._buffer = torch.zeros(self.total_dim, device=device)
        buf = self._buffer
        offset = 0
        for name in self._order:
            v = values.get(name)
            dim = self._signals[name]
            if v is None:
                buf[offset:offset+dim].zero_()
            elif isinstance(v, (int, float)):
                buf[offset] = v
            elif isinstance(v, np.ndarray):
                buf[offset:offset+dim].copy_(torch.from_numpy(v).float())
            elif isinstance(v, torch.Tensor):
                buf[offset:offset+dim].copy_(v.float())
            else:
                buf[offset:offset+dim].zero_()
            offset += dim
        return buf.clone()


N_OBJECT_SLOTS = 4
SLOT_DIM = 64

def _make_signal_bus(num_actions):
    bus = SignalBus()
    bus.register("fear", 1)
    bus.register("curiosity", 1)
    bus.register("competence", 1)
    bus.register("doing_well", 1)
    bus.register("getting_better", 1)
    bus.register("surprised", 1)
    bus.register("last_reward", 1)
    bus.register("mem_reward", 1)
    bus.register("belief_trust", 1)
    bus.register("concept_depth", 1)
    bus.register("has_word", 1)
    bus.register("action_prior", num_actions)
    bus.register("object_slots", N_OBJECT_SLOTS * 2)
    return bus


class Actor(nn.Module):
    """Unified actor: concat(z, signals, gated_ear) -> action logits.
    ear_input = [inner_voice, instruction] — language flows INTO decisions."""

    def __init__(self, latent_dim=256, hidden_dim=256, num_actions=4,
                 signal_dim=15, ear_dim=512):
        super().__init__()
        self.signal_dim = signal_dim
        self.ear_dim = ear_dim
        self.ear_gate = nn.Sequential(
            nn.Linear(ear_dim, 64), nn.SiLU(),
            nn.Linear(64, ear_dim), nn.Sigmoid(),
        )
        self.net = nn.Sequential(
            nn.Linear(latent_dim + signal_dim + ear_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.num_actions = num_actions
        self._zero_sig = None
        self._zero_ear = None

    def _get_zero_sig(self, device, batch=None):
        if self._zero_sig is None or self._zero_sig.device != device:
            self._zero_sig = torch.zeros(self.signal_dim, device=device)
        if batch is not None:
            return self._zero_sig.unsqueeze(0).expand(batch, -1)
        return self._zero_sig

    def _get_zero_ear(self, device, batch=None):
        if self._zero_ear is None or self._zero_ear.device != device:
            self._zero_ear = torch.zeros(self.ear_dim, device=device)
        if batch is not None:
            return self._zero_ear.unsqueeze(0).expand(batch, -1)
        return self._zero_ear

    def forward(self, z, signals=None, ear_input=None):
        if signals is None:
            signals = self._get_zero_sig(z.device, z.shape[0] if z.dim() > 1 else None)
        if ear_input is None:
            ear_input = self._get_zero_ear(z.device, z.shape[0] if z.dim() > 1 else None)
        gated_ear = self.ear_gate(ear_input) * ear_input
        if z.dim() == 1:
            x = torch.cat([z, signals, gated_ear])
        else:
            x = torch.cat([z, signals, gated_ear], dim=-1)
        return self.net(x)

    def act(self, z, signals=None, ear_input=None):
        with torch.no_grad():
            logits = self.forward(
                z.unsqueeze(0),
                signals.unsqueeze(0) if signals is not None else None,
                ear_input.unsqueeze(0) if ear_input is not None else None,
            ).squeeze(0)
            return torch.multinomial(F.softmax(logits, dim=-1), 1).item()


class Critic(nn.Module):
    """Unified critic: concat(z, signals) -> scalar V(z)."""

    def __init__(self, latent_dim=256, hidden_dim=256, signal_dim=15):
        super().__init__()
        self.signal_dim = signal_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + signal_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._zero_sig = None

    def _get_zero_sig(self, device, batch=None):
        if self._zero_sig is None or self._zero_sig.device != device:
            self._zero_sig = torch.zeros(self.signal_dim, device=device)
        if batch is not None:
            return self._zero_sig.unsqueeze(0).expand(batch, -1)
        return self._zero_sig

    def forward(self, z, signals=None):
        if signals is None:
            signals = self._get_zero_sig(z.device, z.shape[0] if z.dim() > 1 else None)
        if z.dim() == 1:
            x = torch.cat([z, signals])
        else:
            x = torch.cat([z, signals], dim=-1)
        return self.net(x).squeeze(-1)


class DistributionalCritic(nn.Module):
    """DreamerV3-style categorical value head.

    Outputs logits over `n_bins` evenly spaced bins in [v_min, v_max]. The
    loss is cross-entropy against a two-hot encoding of the scalar target,
    which is more robust than MSE when the return distribution is wide or
    multi-modal (e.g., sparse-reward Atari). Forward returns the expected
    scalar value E[V] = sum(softmax(logits) * bin_centers), so the rest of
    the codebase (which expects a scalar critic) keeps working unchanged.
    Use `.logits(...)` and `.two_hot(target)` for training.
    """

    def __init__(self, latent_dim=256, hidden_dim=256, signal_dim=15,
                 n_bins=51, v_min=-20.0, v_max=20.0):
        super().__init__()
        assert n_bins >= 2 and v_max > v_min
        self.signal_dim = signal_dim
        self.n_bins = n_bins
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + signal_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, n_bins),
        )
        self.register_buffer(
            "bin_centers",
            torch.linspace(self.v_min, self.v_max, n_bins),
            persistent=False,
        )
        self._zero_sig = None

    def _get_zero_sig(self, device, batch=None):
        if self._zero_sig is None or self._zero_sig.device != device:
            self._zero_sig = torch.zeros(self.signal_dim, device=device)
        if batch is not None:
            return self._zero_sig.unsqueeze(0).expand(batch, -1)
        return self._zero_sig

    def _prep_input(self, z, signals):
        if signals is None:
            signals = self._get_zero_sig(z.device, z.shape[0] if z.dim() > 1 else None)
        if z.dim() == 1:
            return torch.cat([z, signals])
        return torch.cat([z, signals], dim=-1)

    def logits(self, z, signals=None):
        x = self._prep_input(z, signals)
        out = self.net(x)
        if z.dim() == 1:
            return out
        return out

    def forward(self, z, signals=None):
        """Returns the *scalar* expected value (so the call site is unchanged)."""
        logits = self.logits(z, signals)
        probs = F.softmax(logits, dim=-1)
        bins = self.bin_centers.to(probs.device)
        if probs.dim() == 1:
            return (probs * bins).sum(dim=-1)
        return (probs * bins).sum(dim=-1)

    def two_hot(self, target):
        """Encode scalar `target` as a two-hot probability vector over bins.

        Linearly distribute mass between the two adjacent bins straddling
        `target`. Targets outside [v_min, v_max] are clamped.
        """
        bins = self.bin_centers.to(target.device)
        t = target.clamp(self.v_min, self.v_max)
        # find right bin index (>= target)
        idx_hi = torch.bucketize(t, bins, right=False).clamp(1, self.n_bins - 1)
        idx_lo = idx_hi - 1
        b_hi = bins[idx_hi]
        b_lo = bins[idx_lo]
        denom = (b_hi - b_lo).clamp(min=1e-8)
        w_hi = ((t - b_lo) / denom).clamp(0.0, 1.0)
        w_lo = 1.0 - w_hi
        out = torch.zeros(*t.shape, self.n_bins, device=t.device, dtype=w_hi.dtype)
        out.scatter_(-1, idx_lo.unsqueeze(-1), w_lo.unsqueeze(-1))
        out.scatter_(-1, idx_hi.unsqueeze(-1), w_hi.unsqueeze(-1))
        return out

    def loss(self, z, signals, target):
        """Cross-entropy between predicted logits and two-hot(target)."""
        logits = self.logits(z, signals)
        target_dist = self.two_hot(target.detach())
        log_probs = F.log_softmax(logits, dim=-1)
        return -(target_dist * log_probs).sum(dim=-1).mean()


class RewardPredictor(nn.Module):
    """Predicts reward from (latent_z, action). Essential for dream training."""

    def __init__(self, latent_dim=256, num_actions=4, hidden_dim=128):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, latent_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z, action):
        a = self.action_embed(action)
        return self.net(torch.cat([z, a], dim=-1)).squeeze(-1)


class DonePredictor(nn.Module):
    """Learns from experience: which (state, action) pairs lead to death?
    The brain discovers on its own that dying is bad — no hardcoded reward shaping."""

    def __init__(self, latent_dim=256, num_actions=4, hidden_dim=128):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, latent_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z, action):
        a = self.action_embed(action)
        return self.net(torch.cat([z, a], dim=-1)).squeeze(-1)



def _derive_word(reward, action, frame, prev_frame, num_actions):
    if reward > 0: return "score", 1.0
    if reward < 0: return "miss", 1.0
    diff = (frame - prev_frame).abs()
    change = diff.mean().item()
    h_mid = frame.shape[1] // 2
    w_mid = frame.shape[2] // 2
    vert = diff[:, :h_mid, :].mean().item() - diff[:, h_mid:, :].mean().item()
    horiz = diff[:, :, w_mid:].mean().item() - diff[:, :, :w_mid].mean().item()
    if change > 0.03:
        if abs(vert) > abs(horiz):
            return ("up", 0.8) if vert > 0 else ("down", 0.8)
        else:
            return ("right", 0.8) if horiz > 0 else ("left", 0.8)
    if change > 0.015: return "fast", 0.6
    if change > 0.008: return "new", 0.5
    return None, 0.0


def _discover_concepts(latent_buffer, n_clusters=6, min_samples=50):
    if len(latent_buffer) < min_samples:
        return None, None
    z_all = torch.stack(list(latent_buffer))
    k = min(n_clusters, len(z_all))
    centroids = z_all[torch.randperm(len(z_all))[:k]].clone()
    for _ in range(20):
        labels = torch.cdist(z_all, centroids).argmin(dim=1)
        new_c = []
        for i in range(k):
            mask = labels == i
            new_c.append(z_all[mask].mean(dim=0) if mask.sum() >= 2 else centroids[i])
        centroids = torch.stack(new_c)
    return labels, centroids


class Brain:
    def __init__(self, env: BrainEnv, latent_dim=256, hidden_dim=512,
                 n_concepts=6, device="auto", staged=True,
                 encoder_type: str = "vit", encoder_depth: int | None = None,
                 actor_hidden_dim: int = 256, critic_hidden_dim: int = 256,
                 critic_distributional: bool = False,
                 critic_n_bins: int = 51, critic_v_min: float = -20.0,
                 critic_v_max: float = 20.0):
        """
        Args:
            encoder_type: "vit" (default) or "cnn". CNN is faster and more
                biologically plausible for the early visual front-end.
                See `WorldModel` and `BrainEncoder` for details.
            encoder_depth: number of transformer/conv blocks in the encoder.
                None = WorldModel default (2 on CPU, 4 on GPU). Higher depth
                grows the receptive field for CNN and the reasoning capacity
                for ViT.
            actor_hidden_dim / critic_hidden_dim: width of the MLP heads.
                Default 256 matches the legacy hard-coded sizes.
            critic_distributional: if True, replaces the scalar critic with a
                categorical (DreamerV3-style) value head over `critic_n_bins`
                bins spanning [critic_v_min, critic_v_max], trained with
                cross-entropy on a two-hot target. The expected value is
                used wherever a scalar V(s) is needed (advantage, target).
        """
        self.env = env
        self.latent_dim = latent_dim
        self.n_concepts = n_concepts
        self._staged = staged
        self.encoder_type = encoder_type
        self.encoder_depth = encoder_depth
        self.actor_hidden_dim = actor_hidden_dim
        self.critic_hidden_dim = critic_hidden_dim
        self.critic_distributional = critic_distributional

        if device == "auto":
            if torch.cuda.is_available(): self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else: self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        na = env.num_actions
        in_ch = getattr(env, 'frame_channels', 3)
        self.world_model = WorldModel(latent_dim, na, hidden_dim, in_channels=in_ch,
                                      encoder_type=encoder_type,
                                      encoder_depth=encoder_depth).to(self.device)
        self.lang_module = LanguageModule(latent_dim, 128, env.vocab_words).to(self.device)

        self.signal_bus = _make_signal_bus(na)
        sd = self.signal_bus.total_dim
        ear_dim = latent_dim * 2
        self.actor = Actor(latent_dim, actor_hidden_dim, na,
                           signal_dim=sd, ear_dim=ear_dim).to(self.device)
        if critic_distributional:
            self.critic = DistributionalCritic(
                latent_dim, critic_hidden_dim, signal_dim=sd,
                n_bins=critic_n_bins, v_min=critic_v_min, v_max=critic_v_max,
            ).to(self.device)
            self.slow_critic = DistributionalCritic(
                latent_dim, critic_hidden_dim, signal_dim=sd,
                n_bins=critic_n_bins, v_min=critic_v_min, v_max=critic_v_max,
            ).to(self.device)
        else:
            self.critic = Critic(latent_dim, critic_hidden_dim, signal_dim=sd).to(self.device)
            self.slow_critic = Critic(latent_dim, critic_hidden_dim, signal_dim=sd).to(self.device)
        self.slow_critic.load_state_dict(self.critic.state_dict())
        self.slow_critic.requires_grad_(False)
        self.reward_pred = RewardPredictor(latent_dim, na, 256).to(self.device)
        self.done_pred = DonePredictor(latent_dim, na, 256).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.gamma = 0.99
        self.lam = 0.95
        self.slow_ema = 0.98
        self.entropy_coeff = 5e-3

        # Optional bias for the eps-greedy exploration sample. None means
        # uniform random (default). When set, the random kick in
        # _curiosity_action samples from Categorical(prior) instead. Shapes
        # which actions get *explored*, which shapes what gets into replay,
        # which shapes what dream-training sees. Useful when we know a
        # priori that some action is more likely to be productive (e.g.,
        # in Breakout with auto_fire, action 3 = RIGHT consistently produces
        # the highest scores in our seed-7 run).
        self._exploration_prior = None
        # Multiplicative decay applied to the *deviation* from uniform every
        # episode, so the prior fades and the policy converges to its own
        # learned exploration after enough episodes.
        self._exploration_prior_decay = 1.0  # 1.0 = no decay
        self._uniform_action_prob = 1.0 / na

        # Trust-region anchor on the actor: a frozen snapshot of the policy at
        # the time of the most recent best-eval checkpoint. KL(current||ref)
        # is added to the actor loss to dampen destructive late-training
        # drift (we observed eval reward peak at ep 25 then degrade through
        # ep 500 — the actor over-commits once dream-training pulls it toward
        # a single high-confidence action). Coefficient is configurable; set
        # to 0 to disable. The reference is refreshed in train() whenever a
        # new best checkpoint is saved, so the actor can still improve — it
        # just can't sprint away from the last known-good policy.
        self.reference_actor = None
        self.actor_kl_coef = 0.0  # set via train(actor_kl_coef=...) 

        self.reward_map = deque(maxlen=2000)

        self._reward_running_mean = 0.0
        self._reward_running_var = 1.0
        self._reward_count = 0
        self.loss_aversion = 2.5
        self._mem_cache = 0.0
        self._mem_step = 0
        self._concept_words = {}
        self._action_prior_cache = np.zeros(na)
        self._wm_frozen = False
        self._wm_freeze_ep = 0
        self._wm_stable_count = 0
        self._total_actor_actions = 0
        self._reward_stable_count = 0
        self._peak_reward = -float('inf')
        self._prev_reward_trend = 0.0

        concepts = [f"state_{i}" for i in range(n_concepts)]
        self.monitor = MetacognitiveMonitor(concepts, window=200)
        self.value_sys = ValueSystem(concepts, history_len=200)
        self.episodic = EpisodicMemory(capacity=10_000)
        self.semantic = SemanticMemory(concepts, latent_dim)
        self.sleep_cycle = SleepCycle(replay_batch=64, replay_steps=5)
        self.knowledge_tree = KnowledgeTree(concepts)
        self.self_brain = SelfCorrectingBrain(concepts, self.monitor, self.semantic, self.knowledge_tree)
        self.intuition = IntuitionGate(latent_dim, hidden_dim=64)

        self._dev_stage = 0
        self._dev_stages = {
            0: "perceive",
            1: "act",
            2: "remember",
            3: "speak",
            4: "reflect",
            5: "organize",
        }
        self._dev_stage_ep = {0: 0}

        # Split optimizers so the soft-freeze / online updates don't accidentally
        # touch unrelated heads. Previously a single "wm_optimizer" owned WM +
        # lang + reward + done, which meant the soft-freeze dropped the LR on
        # the language and reward/done heads as well. It also caused Adam's
        # moment estimates to be corrupted by the manual SGD step on the
        # predictor / lang_module that runs inside the inner loop.
        self.wm_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=3e-4)
        self.lang_optimizer = torch.optim.Adam(self.lang_module.parameters(), lr=1e-4)
        self.head_optimizer = torch.optim.Adam(
            list(self.reward_pred.parameters()) + list(self.done_pred.parameters()),
            lr=3e-4,
        )
        self.total_params = (self.world_model.param_count() + self.lang_module.param_count() +
                             sum(p.numel() for p in self.actor.parameters()) +
                             sum(p.numel() for p in self.critic.parameters()) +
                             sum(p.numel() for p in self.reward_pred.parameters()) +
                             sum(p.numel() for p in self.done_pred.parameters()) +
                             sum(p.numel() for p in self.intuition.parameters()))

        self.use_amp = self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self._amp_ctx = lambda: autocast(device_type="cuda", dtype=torch.float16) if self.use_amp else nullcontext()

        if self.device.type == "cuda":
            try:
                self.world_model = torch.compile(self.world_model, mode="reduce-overhead")
            except Exception:
                pass

    def _advance_stage(self, ep, ep_wm):
        """Competence-based developmental progression.
        Uses WM freeze as the anchor point — once the brain can predict,
        higher stages activate based on experience since that milestone.

        Stage 0 → 1 (act):      world model can predict (wm_loss < 0.3)
        Stage 1 → 2 (remember): actor has taken 500+ actions
        Stage 2 → 3 (speak):    WM frozen (can predict well) + 20 eps of experience
        Stage 3 → 4 (reflect):  30 eps since speak (enough language/memory context)
        Stage 4 → 5 (organize): self-correction found at least 1 real issue
        """
        if not self._staged:
            if self._dev_stage < 5:
                self._dev_stage = 5
                for s in range(6):
                    self._dev_stage_ep[s] = 0
                print("  [STAGED=OFF] All modules active from ep 1")
            return

        prev = self._dev_stage

        if self._dev_stage == 0 and ep_wm < 0.3:
            self._dev_stage = 1

        if self._dev_stage == 1 and self._total_actor_actions >= 500:
            self._dev_stage = 2

        if self._dev_stage == 2:
            eps_since_freeze = ep - self._wm_freeze_ep if self._wm_frozen else 0
            if self._wm_frozen and eps_since_freeze >= 20:
                self._dev_stage = 3

        if self._dev_stage == 3:
            eps_since_speak = ep - self._dev_stage_ep.get(3, ep)
            if eps_since_speak >= 30:
                self._dev_stage = 4

        if self._dev_stage == 4:
            total_corrections = len(self.self_brain.updater.corrections)
            if total_corrections >= 1:
                self._dev_stage = 5

        if self._dev_stage != prev:
            self._dev_stage_ep[self._dev_stage] = ep
            stage_name = self._dev_stages[self._dev_stage]
            print(f"  [STAGE {self._dev_stage}] -> {stage_name} at ep {ep+1}")

    @property
    def can_act(self):
        return self._dev_stage >= 1

    @property
    def can_remember(self):
        return self._dev_stage >= 2

    @property
    def can_speak(self):
        return self._dev_stage >= 3

    @property
    def can_reflect(self):
        return self._dev_stage >= 4

    @property
    def can_organize(self):
        return self._dev_stage >= 5

    def _build_signals(self, comp=0.0, last_reward=0.0, concept=None, z_cpu=None):
        """Build signal vector via the signal bus — each module contributes."""
        na = self.env.num_actions

        fear = 1.0 if self.value_sys._doing_well < -0.3 else 0.0
        curiosity = self.value_sys.get_curiosity_drive()
        doing_well = self.value_sys._doing_well
        getting_better = self.value_sys._getting_better
        surprised = self.value_sys._surprised

        mem_reward = 0.0
        action_prior = np.zeros(na)
        if self.can_remember and z_cpu is not None and len(self.episodic) > 50 and self._mem_step % 8 == 0:
            similar = self.episodic.recall_similar(z_cpu, top_k=1)
            if similar:
                mem_reward = float(similar[0].get("reward", 0.0))
            scores = self.episodic.recall_best_action(z_cpu, na, top_k=10)
            if scores is not None:
                action_prior = scores
            self._mem_cache = mem_reward
            self._action_prior_cache = action_prior
        else:
            mem_reward = self._mem_cache
            action_prior = self._action_prior_cache

        belief_trust = 1.0
        if concept and concept in self.self_brain.detector.error_history:
            errs = list(self.self_brain.detector.error_history[concept])
            if len(errs) >= 10:
                belief_trust = max(0.0, 1.0 - np.mean(errs[-10:]) * 10)

        concept_depth = 0.0
        if concept and concept in self.knowledge_tree.nodes:
            node = self.knowledge_tree.nodes[concept]
            concept_depth = min(1.0, node.interaction_count / 100.0)

        has_word = 0.0
        if concept and concept in getattr(self, '_concept_words', {}):
            has_word = 1.0

        slot_signal = np.zeros(N_OBJECT_SLOTS * 2)
        slots = self.world_model.last_slots
        if slots is not None:
            s = slots[0] if slots.dim() == 3 else slots
            for i in range(min(N_OBJECT_SLOTS, s.shape[0])):
                slot_signal[i * 2] = s[i, 0].item()
                slot_signal[i * 2 + 1] = s[i, 1].item()

        return self.signal_bus.build({
            "fear": fear,
            "curiosity": curiosity,
            "competence": comp,
            "doing_well": doing_well,
            "getting_better": getting_better,
            "surprised": surprised,
            "last_reward": last_reward,
            "mem_reward": mem_reward,
            "belief_trust": belief_trust,
            "concept_depth": concept_depth,
            "has_word": has_word,
            "action_prior": action_prior,
            "object_slots": slot_signal,
        }, self.device)

    def train(self, episodes=2000, max_steps=500, batch_size=None, buffer_size=100_000,
              plan_len=12, cluster_every=20, sleep_every=50, check_every=10, seed=42,
              log_every=None, callback=None,
              eval_every: int = 0, eval_episodes: int = 3,
              eval_max_steps: int | None = None,
              eval_auto_fire: bool = False,
              eval_epsilon: float = 0.05,
              best_ckpt_path: str | None = None,
              eval_env_factory=None,
              actor_kl_coef: float = 0.0,
              early_stop_patience: int = 0):
        """
        Train the brain end-to-end.

        Args (eval-related):
            eval_every: if > 0, run a quick greedy evaluation every N episodes
                and remember the best score.
            eval_episodes: how many episodes per evaluation pass.
            eval_max_steps: per-eval-episode step cap (defaults to max_steps*4).
            eval_auto_fire: force FIRE on the first 5 steps of each eval episode
                (needed for Breakout-style games where the ball must be launched).
            best_ckpt_path: if set, save a checkpoint here every time the eval
                score beats the previous best. After training, the brain is
                reloaded from this path so callers get the BEST policy seen,
                not the (often worse) final policy.
            eval_env_factory: zero-arg callable returning a fresh env for
                evaluation. Required when eval_every>0 because the training
                env is being mutated each step. If None, tries to clone
                self.env via a few common constructors; falls back to
                copy.deepcopy.
            actor_kl_coef: weight on the KL penalty between the current
                actor and a frozen snapshot taken at the last best-eval
                checkpoint. 0 disables. Typical useful range 0.01 - 0.1.
                Helps prevent late-training policy drift away from a known
                good policy without forbidding all change (the snapshot is
                refreshed each time eval beats the previous best).
            early_stop_patience: if > 0, stop training after this many
                consecutive in-training evals without improving the best
                eval score. Saves compute when the policy has plateaued.
                Requires eval_every > 0 to take effect.
        """
        if batch_size is None:
            batch_size = 256 if self.device.type == "cuda" else 64
        if log_every is None:
            log_every = max(1, episodes // 20)
        if eval_max_steps is None:
            eval_max_steps = max_steps * 4

        best_eval_score = float("-inf")
        best_eval_ep = -1
        evals_without_improvement = 0
        # Activate KL anchor for this training run. Reference is set lazily
        # on the first eval that beats the (initial -inf) baseline.
        self.actor_kl_coef = float(actor_kl_coef)
        if actor_kl_coef > 0:
            print(f"  Actor KL anchor enabled: coef={actor_kl_coef}, "
                  f"reference will refresh on each new best checkpoint.")
        if early_stop_patience > 0 and eval_every > 0:
            print(f"  Early stopping enabled: patience={early_stop_patience} "
                  f"evals without improvement.")

        # Seed every RNG (stdlib, NumPy, PyTorch CPU+CUDA) so two runs with
        # the same `seed` actually produce comparable curves. Without this,
        # the actor's torch.multinomial sampling and any nn.Dropout-style
        # ops used a global RNG that was never seeded.
        _seed_everything(seed)

        rng = np.random.RandomState(seed)
        na = self.env.num_actions
        replay = deque(maxlen=buffer_size)
        frame_replay = deque(maxlen=5_000)
        latent_buf = deque(maxlen=5000)
        cluster_centroids = None
        concepts = [f"state_{i}" for i in range(self.n_concepts)]
        word_counts = {w: 0 for w in self.env.vocab_words}
        state_visit_counts = {}
        intrinsic_scale = 0.5

        history = {k: [] for k in ["episode", "wm_loss", "lang_loss", "reward",
                                    "dream_actor_loss", "curiosity_actions", "plan_actions",
                                    "corrections", "tree_size", "wellbeing",
                                    "competence", "word_diversity"]}

        print(f"Device: {self.device}")
        print(f"Parameters: {self.total_params:,}")
        print(f"\n{'='*60}\n  Training {episodes} episodes\n{'='*60}")

        t0 = time.time()
        total_frames = total_corr = total_cur = total_plan = 0

        for ep in range(episodes):
            frame = self.env.reset(seed=seed + ep)
            prev_frame = frame.clone()
            self.world_model.reset_memory(batch_size=1, device=self.device)
            ep_rew = ep_cur = ep_plan = ep_corr = 0
            ep_wm = ep_lang = ep_dream_actor = 0.0
            ep_words = set()
            intrinsic_scale = max(0.05, 0.5 * (1.0 - ep / max(episodes * 0.8, 1)))

            z_cached = None
            ear_input = torch.zeros(self.latent_dim * 2, device=self.device)
            self.world_model.eval()

            for step in range(max_steps):

                # === 1. SEE — encode current frame ===
                with torch.inference_mode(), self._amp_ctx():
                    z_now = self.world_model.encode(frame.unsqueeze(0).to(self.device)).squeeze(0)
                z_cached = z_now.detach().float()
                z_cpu = z_cached.cpu()
                latent_buf.append(z_cpu)

                # === 2. RECOGNIZE — which concept? ===
                if cluster_centroids is not None:
                    concept = f"state_{(cluster_centroids - z_cpu).pow(2).sum(dim=1).argmin().item()}"
                else:
                    concept = f"state_{_stable_concept_id(z_cpu, self.n_concepts)}"

                # === 3. LEARN WORD — only every 10 steps, only if can speak ===
                word = None; widx = train_word = -1
                if self.can_speak and step % 10 == 0:
                    word, wconf = _derive_word(0, 0, frame, prev_frame, na)
                    if word is not None:
                        widx = self.lang_module.vocab.encode(word)
                        word_counts[word] = word_counts.get(word, 0) + 1; ep_words.add(word)
                        train_word = widx if wconf >= 0.5 else -1
                    if train_word >= 0:
                        # Use the proper Adam optimizer (with urgency-scaled LR)
                        # instead of `p.data -= lr * p.grad`. The previous code
                        # mixed manual SGD with Adam (because lang was also in
                        # wm_optimizer), corrupting Adam's running moments.
                        lang_lr = 1e-4 * self.value_sys.get_learning_urgency()
                        for pg in self.lang_optimizer.param_groups:
                            pg['lr'] = lang_lr
                        self.lang_module.train()
                        self.lang_optimizer.zero_grad(set_to_none=True)
                        lang_loss_now, _ = self.lang_module.compute_loss(
                            z_cached.unsqueeze(0).detach(), torch.tensor([train_word], device=self.device))
                        lang_loss_now.backward()
                        torch.nn.utils.clip_grad_norm_(self.lang_module.parameters(), 1.0)
                        self.lang_optimizer.step()
                        ep_lang = lang_loss_now.item()

                # === 4. THINK — signals stay fresh every step ===
                comp = self.monitor.competence(concept) if concept else 0.0
                curiosity_drive = self.value_sys.get_curiosity_drive()
                last_rew = ep_rew / max(step, 1)
                self._mem_step = step
                if word is not None and concept:
                    self._concept_words[concept] = word
                signals = self._build_signals(comp, last_rew, concept, z_cpu)

                # === 4b. EARS — inner voice + teacher instruction (every 5 steps) ===
                if step % 5 == 0:
                    teacher_word = self._teacher_instruction(last_rew)
                    ear_input = self._compute_ear_input(z_cached, teacher_word)

                # === 5. DECIDE — random before stage 1, then actor ===
                if not self.can_act:
                    action = rng.randint(0, na); ep_cur += 1
                else:
                    explore_prob = curiosity_drive * (1.0 - comp)
                    if rng.random() < explore_prob:
                        action = self._curiosity_action(z_cached, signals, ear_input); ep_cur += 1
                    else:
                        with torch.no_grad():
                            logits = self.actor(z_cached.unsqueeze(0), signals.unsqueeze(0),
                                                ear_input.unsqueeze(0)).squeeze(0)
                            self.intuition.self_monitor.measure_actor_confidence(logits.unsqueeze(0))
                            action = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
                        ep_plan += 1
                        self._total_actor_actions += 1

                # === 7. ACT ===
                next_frame, reward, done = self.env.step(action)
                extrinsic_reward = reward
                ep_rew += reward; total_frames += 1

                # === 7b. EVALUATE next state ===
                with torch.no_grad(), self._amp_ctx():
                    z_next = self.world_model.encode(next_frame.unsqueeze(0).to(self.device)).squeeze(0).float()

                # === 7b2. INTRINSIC CURIOSITY — count-based exploration bonus ===
                z_key = tuple((z_next.cpu()[:8] * 4).round().int().tolist())
                state_visit_counts[z_key] = state_visit_counts.get(z_key, 0) + 1
                visit_n = state_visit_counts[z_key]
                intrinsic_bonus = intrinsic_scale / (visit_n ** 0.5)
                reward = reward + intrinsic_bonus

                # === 7c. NORMALIZE REWARD for critic training ===
                if self.can_act:
                    norm_reward = self._normalize_reward(reward)
                else:
                    norm_reward = reward

                # === 7d. REWARD MAP — remember WHERE important things happened ===
                if extrinsic_reward != 0:
                    self.reward_map.append((z_cpu.clone(), action, reward, signals.detach().cpu()))

                # === 8. EVALUATE — prediction error ===
                with torch.no_grad(), self._amp_ctx():
                    z_pred, _ = self.world_model.predict_next(z_cached.unsqueeze(0),
                        torch.tensor([action], device=self.device))
                    pred_error = F.mse_loss(z_pred.squeeze(0).float(), z_next.float()).item()

                tw_post = -1
                if self.can_speak and (reward != 0 or step % 20 == 0):
                    word_post, wconf_post = _derive_word(reward, action, next_frame, frame, na)
                    if word_post is not None:
                        widx_post = self.lang_module.vocab.encode(word_post)
                        word_counts[word_post] = word_counts.get(word_post, 0) + 1
                        ep_words.add(word_post)
                        tw_post = widx_post if wconf_post >= 0.5 else -1
                        if tw_post >= 0:
                            lang_lr_post = 1e-4 * self.value_sys.get_learning_urgency()
                            for pg in self.lang_optimizer.param_groups:
                                pg['lr'] = lang_lr_post
                            self.lang_module.train()
                            self.lang_optimizer.zero_grad(set_to_none=True)
                            ll_post, _ = self.lang_module.compute_loss(
                                z_next.unsqueeze(0).detach(), torch.tensor([tw_post], device=self.device))
                            ll_post.backward()
                            torch.nn.utils.clip_grad_norm_(self.lang_module.parameters(), 1.0)
                            self.lang_optimizer.step()
                            ep_lang = ll_post.item()

                if concept:
                    self.monitor.record(concept, pred_error)
                    comp = self.monitor.competence(concept)
                    lp = self.monitor.learning_progress(concept)
                    self.value_sys.process(pred_error, concept, comp, lp, reward=reward)
                    self.value_sys.record_action(concept, action)

                # === 8b. SELF-AWARE INTUITION — brain looks inward ===
                self.intuition.self_monitor.measure_loss_trend(pred_error)
                self.intuition.self_monitor.measure_z_diversity(z_cpu)
                force = (extrinsic_reward != 0) or (step < 20)
                should_learn, gate_conf = self.intuition.decide(force_learn=force)

                # === 9. LEARN WORLD — only if intuition says YES ===
                if should_learn and step % 10 == 0 and not self._wm_frozen:
                    self.intuition.self_monitor.snapshot_weights(self.world_model)
                    # Drive a small online update through the proper Adam
                    # optimizer (urgency-scaled LR) instead of overwriting
                    # weights with raw gradient steps. The previous
                    # `p.data -= wm_lr * p.grad` corrupted Adam's running
                    # moments because the same params were also stepped by
                    # the batch wm_optimizer at episode end.
                    wm_lr = 5e-5 * self.value_sys.get_learning_urgency()
                    saved_lr = [pg['lr'] for pg in self.wm_optimizer.param_groups]
                    for pg in self.wm_optimizer.param_groups:
                        pg['lr'] = wm_lr
                    self.world_model.train()
                    self.wm_optimizer.zero_grad(set_to_none=True)
                    z_p, _ = self.world_model.predict_next(z_cached.unsqueeze(0).detach(),
                        torch.tensor([action], device=self.device))
                    online_loss = F.mse_loss(z_p, z_next.unsqueeze(0).detach())
                    online_loss.backward()
                    self.intuition.self_monitor.measure_gradient_health(self.world_model)
                    torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
                    self.wm_optimizer.step()
                    for pg, lr in zip(self.wm_optimizer.param_groups, saved_lr):
                        pg['lr'] = lr
                    self.intuition.self_monitor.measure_weight_change(self.world_model)
                    ep_wm = online_loss.item()
                    was_useful = pred_error > 0.01
                    self.intuition.record_outcome(self.intuition.read_self(), was_useful)

                # === 10. STORE — always observe, but deep-store only if intuition says YES ===
                quality = self.value_sys.assess_quality(reward, pred_error)
                if should_learn:
                    self.episodic.store(frame, action, next_frame, tw_post,
                                        concept or "unknown", pred_error, ep,
                                        z=z_cpu, reward=reward, quality=quality)
                if concept and self.can_reflect and should_learn:
                    c = self.self_brain.observe(concept, pred_error, z_cpu)
                    if c: ep_corr += 1

                frame_replay.append((frame, action, next_frame, tw_post, reward, done))
                replay.append((z_cpu.clone(), action, z_next.cpu().clone(),
                               reward, done, signals.detach().cpu(),
                               ear_input.detach().cpu()))

                # === 10b. ELIGIBILITY TRACE — reward credit flows backward ===
                if extrinsic_reward != 0 and len(replay) > 1:
                    trace_len = min(10, len(replay) - 1)
                    for k in range(1, trace_len + 1):
                        idx = len(replay) - 1 - k
                        if idx < 0:
                            break
                        old = replay[idx]
                        if old[4]:
                            break
                        trace_credit = extrinsic_reward * (self.gamma ** k) * 0.3
                        boosted = old[3] + trace_credit
                        replay[idx] = (old[0], old[1], old[2], boosted, old[4], old[5], old[6])

                prev_frame = frame; frame = next_frame
                if done: break

            total_corr += ep_corr; total_cur += ep_cur; total_plan += ep_plan

            # Intuition Gate — self-check from internals + train the gate
            self.intuition.update_exploration(self._dev_stage, ep, episodes)
            self.intuition.check_and_reopen()
            if ep % 10 == 0:
                self.intuition.train_gate()

            # Volume scaling — grow training capacity as the brain matures
            progress = min(1.0, ep / max(episodes * 0.5, 1))
            scaled_bs = int(batch_size * (1.0 + 3.0 * progress))
            scaled_bs = min(scaled_bs, len(replay), batch_size * 4)

            if len(frame_replay) >= batch_size:
                self.world_model.train(); self.lang_module.train()
                fr_bs = min(scaled_bs, len(frame_replay))
                fr_priorities = np.array([1.0 + abs(frame_replay[i][4]) * 5.0
                                          for i in range(len(frame_replay))])
                fr_priorities /= fr_priorities.sum()
                idxs = rng.choice(len(frame_replay), fr_bs, replace=False, p=fr_priorities)
                fbatch = [frame_replay[i] for i in idxs]
                b_obs = torch.stack([b[0] for b in fbatch]).to(self.device)
                b_act = torch.tensor([b[1] for b in fbatch], dtype=torch.long, device=self.device)
                b_nobs = torch.stack([b[2] for b in fbatch]).to(self.device)

                # Zero grads on every owning optimizer so the joint backward
                # below distributes gradients correctly.
                self.wm_optimizer.zero_grad(set_to_none=True)
                self.lang_optimizer.zero_grad(set_to_none=True)
                self.head_optimizer.zero_grad(set_to_none=True)
                with self._amp_ctx():
                    wm_loss, wm_m, z_enc = self.world_model.compute_loss(b_obs, b_act, b_nobs, return_z=True)

                    labeled = [i for i, r in enumerate(frame_replay) if r[3] >= 0]
                    lang_loss = torch.tensor(0.0, device=self.device)
                    if self.can_speak and len(labeled) >= 8:
                        lbs = min(32, len(labeled))
                        li = rng.choice(labeled, lbs, replace=len(labeled) < lbs)
                        lb = [frame_replay[i] for i in li]
                        lo = torch.stack([b[0] for b in lb]).to(self.device)
                        lw = torch.tensor([b[3] for b in lb], dtype=torch.long, device=self.device)
                        lang_loss, _ = self.lang_module.compute_loss(self.world_model.encode(lo, use_memory=False), lw)

                    b_rew = torch.tensor([b[4] for b in fbatch],
                                          dtype=torch.float32, device=self.device)
                    b_done = torch.tensor([1.0 if b[5] else 0.0 for b in fbatch],
                                           dtype=torch.float32, device=self.device)
                    rew_pred = self.reward_pred(z_enc, b_act)
                    rew_loss = F.mse_loss(rew_pred, symlog(b_rew))
                    done_logits = self.done_pred(z_enc, b_act)
                    done_loss = F.binary_cross_entropy_with_logits(done_logits, b_done)

                    total_loss = wm_loss + 0.2 * lang_loss + 1.0 * rew_loss + 1.0 * done_loss

                    self.scaler.scale(total_loss).backward()
                    # Unscale every optimizer that has gradients to clip safely.
                    self.scaler.unscale_(self.wm_optimizer)
                    self.scaler.unscale_(self.lang_optimizer)
                    self.scaler.unscale_(self.head_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.lang_module.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.reward_pred.parameters()) + list(self.done_pred.parameters()), 1.0)
                    self.scaler.step(self.wm_optimizer)
                    self.scaler.step(self.lang_optimizer)
                    self.scaler.step(self.head_optimizer)
                    self.scaler.update()
                    ep_wm = wm_m["pred_loss"]; ep_lang = lang_loss.item()

                    if ep > 50 and ep_wm < 0.05:
                        self._wm_stable_count += 1
                        if self._wm_stable_count >= 20 and not self._wm_frozen:
                            self._wm_frozen = True
                            self._wm_freeze_ep = ep
                            # Soft-freeze ONLY the world model; lang_optimizer
                            # and head_optimizer keep their original LRs so the
                            # heads can still adapt to the (now-stable) latents.
                            for pg in self.wm_optimizer.param_groups:
                                pg['lr'] = 3e-5
                            print(f"  [SOFT-FREEZE] WM LR reduced 10x at ep {ep+1}, wm_loss={ep_wm:.5f}")
                    else:
                        self._wm_stable_count = 0

            # Critic learns from REAL transitions — gated by dev stage
            if self.can_act and len(replay) >= batch_size:
                self._train_critic_from_replay(replay, rng, scaled_bs)
                # Also train the actor on real trajectories. Pure dream
                # training caused mode collapse: the actor would over-fit
                # to whichever action the WM imagined as best, regardless
                # of whether it actually paid off in the real environment.
                self._train_actor_from_replay(replay, rng, scaled_bs)

            # DreamerV3 dream training — ramp up when WM is reliable
            if self.can_act and len(replay) >= batch_size:
                if self._wm_frozen:
                    n_dream_rounds = 3
                    dream_starts = 32 if self.device.type == "cpu" else 64
                else:
                    n_dream_rounds = 2
                    dream_starts = 32
                for _ in range(n_dream_rounds):
                    ep_dream_actor = self._dream_train(replay, rng, horizon=plan_len,
                                                       n_starts=dream_starts)

            self._advance_stage(ep, ep_wm)
            self.value_sys.update_wm_loss(ep_wm)
            self.value_sys.end_episode(ep_rew)
            self.monitor.record_episode_reward(ep_rew)
            self._decay_exploration_prior()

            if ep_rew > self._peak_reward:
                self._peak_reward = ep_rew
            # NOTE: We used to unfreeze the WM when reward dropped below 50% of
            # peak. Empirically (cnn_long, breakout_fixed runs) this CAUSED the
            # regression rather than curing it: bumping the WM LR back up shifts
            # the latent space, invalidating everything the actor learned on top
            # of those latents. The right response to a reward drop is to train
            # the actor harder — which we already do via _train_actor_from_replay
            # every step — not to thrash the world model. So we leave the WM
            # frozen once it has converged. If you want to revisit, reintroduce
            # carefully (e.g., gradual LR warmup over many episodes).

            if (ep + 1) % cluster_every == 0 and len(latent_buf) >= 100:
                _, cluster_centroids = _discover_concepts(latent_buf, self.n_concepts)
                if cluster_centroids is not None:
                    for i in range(len(cluster_centroids)):
                        sc = f"state_{i}"
                        if sc not in self.monitor.error_history:
                            self.monitor.error_history[sc] = deque(maxlen=200)
                            self.monitor.interaction_count[sc] = 0
                        if sc not in self.value_sys.error_history:
                            self.value_sys.error_history[sc] = deque(maxlen=200)
                            self.value_sys.attempt_history[sc] = deque(maxlen=200)
                            self.value_sys.action_history[sc] = deque(maxlen=50)
                            self.value_sys.strategy_noise[sc] = 0.0

            if self.can_remember and (ep + 1) % sleep_every == 0 and len(self.episodic) > 64:
                if self._wm_frozen:
                    self.sleep_cycle.run_replay_only(
                        self.world_model, self.episodic, self.semantic, self.device)
                else:
                    self.world_model.train(); self.lang_module.train()
                    self.sleep_cycle.run(self.world_model, self.lang_module, self.episodic,
                                         self.semantic, self.wm_optimizer, self.device,
                                         lang_weight=0.15, lang_optimizer=self.lang_optimizer)

            if self.can_reflect and (ep + 1) % check_every == 0:
                r = self.self_brain.periodic_check()
                total_corr += len(r.get("changes", []))

            avg_comp = np.mean([self.monitor.competence(c) for c in concepts])
            for k, v in [("episode", ep), ("wm_loss", ep_wm), ("lang_loss", ep_lang),
                         ("reward", float(ep_rew)), ("dream_actor_loss", ep_dream_actor),
                         ("curiosity_actions", total_cur), ("plan_actions", total_plan),
                         ("corrections", total_corr),
                         ("tree_size", len(self.knowledge_tree.nodes) - 1),
                         ("wellbeing", self.value_sys.get_wellbeing()),
                         ("competence", float(avg_comp)),
                         ("word_diversity", len(ep_words))]:
                history[k].append(v)

            if callback: callback(ep, history)

            if (ep + 1) % log_every == 0:
                elapsed = time.time() - t0
                fps = total_frames / max(elapsed, 1)
                cd = self.value_sys.get_curiosity_drive()
                mt = self.value_sys.get_memory_trust()
                lu = self.value_sys.get_learning_urgency()
                wb = self.value_sys.get_wellbeing()
                dw = self.value_sys._doing_well
                gb = self.value_sys._getting_better
                wm_tag = "FROZEN" if self._wm_frozen else f"{np.mean(history['wm_loss'][-50:]):.5f}"
                print(f"  ep {ep+1:5d}/{episodes}  "
                      f"wm={wm_tag}  "
                      f"lang={np.mean(history['lang_loss'][-50:]):.3f}  "
                      f"rew={np.mean(history['reward'][-50:]):+.1f}  "
                      f"actor={ep_plan}  cur={ep_cur}  "
                      f"corr={total_corr}  "
                      f"cur_d={cd:.2f} well={dw:+.2f} trend={gb:+.2f} wb={wb:+.2f} "
                      f"gate={self.intuition.stats['yes_rate']:.0%}  "
                      f"[{elapsed:.0f}s {fps:.0f}fps]")

            # === In-training evaluation + best-checkpoint saving ===
            # We periodically evaluate the *current* greedy policy on a fresh
            # copy of the env (different seeds than training) and save the
            # checkpoint whenever it beats the best score we've seen. This
            # protects against the late-training reward drift that bit
            # cnn_breakout / cnn_long: the policy peaks somewhere in the
            # middle, then drifts down. Saving on peak keeps the prototype.
            if eval_every > 0 and (ep + 1) % eval_every == 0:
                eval_score = self._evaluate_policy(
                    n_episodes=eval_episodes,
                    max_steps=eval_max_steps,
                    seed_base=20_000,            # disjoint from train seeds
                    auto_fire=eval_auto_fire,
                    epsilon=eval_epsilon,
                    env_factory=eval_env_factory,
                )
                marker = ""
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    best_eval_ep = ep + 1
                    evals_without_improvement = 0
                    if best_ckpt_path is not None:
                        self.save(best_ckpt_path)
                        marker = "  [BEST -> saved]"
                    else:
                        marker = "  [BEST]"
                    # Refresh trust-region anchor: new best policy = new
                    # reference. Subsequent actor updates will be regularized
                    # toward THIS policy until a better one is found.
                    if self.actor_kl_coef > 0:
                        self._refresh_reference_actor()
                        marker = marker + " [ref refreshed]"
                else:
                    evals_without_improvement += 1
                print(f"  [EVAL] ep {ep+1}: greedy mean={eval_score:+.2f} "
                      f"(best={best_eval_score:+.2f} @ ep {best_eval_ep}){marker}")
                if (early_stop_patience > 0
                        and evals_without_improvement >= early_stop_patience):
                    print(f"  [EARLY STOP] No improvement for "
                          f"{evals_without_improvement} consecutive evals; "
                          f"stopping at ep {ep+1}/{episodes}.")
                    break

        elapsed = time.time() - t0
        self.env.close()
        self._print_report(history, word_counts, elapsed, total_frames,
                           total_cur, total_plan, total_corr)

        # Reload best checkpoint so the returned brain is the best one seen,
        # not the (potentially regressed) final one.
        if best_ckpt_path is not None and best_eval_score > float("-inf"):
            print(f"  Reloading best checkpoint from {best_ckpt_path} "
                  f"(eval={best_eval_score:+.2f} @ ep {best_eval_ep})")
            self.load(best_ckpt_path)

        history["best_eval_score"] = [best_eval_score] if best_eval_score > float("-inf") else []
        history["best_eval_ep"] = [best_eval_ep] if best_eval_ep >= 0 else []
        return history

    def set_exploration_prior(self, prior, decay: float = 1.0):
        """Set a probability distribution to bias exploration sampling.

        Args:
            prior: array-like of length num_actions. Must be non-negative
                and will be re-normalized to sum to 1. Set None to clear.
            decay: per-episode multiplicative decay applied to the
                *deviation from uniform*. 1.0 = no decay (prior stays);
                0.99 ≈ converges to uniform after ~500 eps; 0.95 ≈ ~100 eps.
                Use decay < 1 if you want the prior to fade as the actor
                develops its own exploration policy.
        """
        if prior is None:
            self._exploration_prior = None
            return
        p = np.asarray(prior, dtype=np.float64)
        assert len(p) == self.env.num_actions, (
            f"prior length {len(p)} != num_actions {self.env.num_actions}")
        assert (p >= 0).all(), "prior must be non-negative"
        s = p.sum()
        assert s > 0, "prior must have positive mass"
        self._exploration_prior = (p / s).astype(np.float32)
        self._exploration_prior_decay = float(decay)

    def _decay_exploration_prior(self):
        """Apply one episode of decay toward the uniform distribution."""
        if self._exploration_prior is None or self._exploration_prior_decay >= 1.0:
            return
        u = self._uniform_action_prob
        # mix toward uniform: prior <- decay * prior + (1-decay) * uniform
        d = self._exploration_prior_decay
        self._exploration_prior = (d * self._exploration_prior + (1 - d) * u)
        # re-normalize defensively (FP drift)
        self._exploration_prior /= self._exploration_prior.sum()

    def load_reference_actor(self, ckpt_path: str):
        """Load just the actor parameters from a checkpoint and freeze them
        as the KL trust-region reference.

        Use case: warm-start the KL anchor with a known-good policy from a
        previous run, instead of waiting for the first in-training eval to
        define the reference. Call this BEFORE train(actor_kl_coef=...).
        """
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if "actor" not in ckpt:
            raise KeyError(f"checkpoint {ckpt_path} has no 'actor' state_dict")
        # Rebuild the reference actor at the *checkpoint's* original sizes.
        # This lets us anchor a bigger / smaller current actor against a
        # reference trained at a different width — the KL only needs the
        # reference's logits over the same action set, not matching weights.
        ref_state = ckpt["actor"]
        ref_latent_plus_extras = ref_state["net.0.weight"].shape[1]
        ref_hidden_dim = ref_state["net.0.weight"].shape[0]
        ref_num_actions = ref_state["net.6.weight"].shape[0]
        if ref_num_actions != self.actor.num_actions:
            raise ValueError(
                f"reference actor has {ref_num_actions} actions but current "
                f"env has {self.actor.num_actions}; KL anchor incompatible."
            )
        ref_signal_dim = self.actor.signal_dim
        ref_ear_dim = self.actor.ear_dim
        ref_latent_dim = ref_latent_plus_extras - ref_signal_dim - ref_ear_dim
        ref = Actor(
            latent_dim=ref_latent_dim,
            hidden_dim=ref_hidden_dim,
            num_actions=ref_num_actions,
            signal_dim=ref_signal_dim,
            ear_dim=ref_ear_dim,
        ).to(self.device)
        ref.load_state_dict(ref_state)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        self.reference_actor = ref
        print(f"  Loaded KL reference actor from {ckpt_path} "
              f"(hidden_dim={ref_hidden_dim})")

    def _refresh_reference_actor(self):
        """Snapshot the current actor as the new trust-region anchor.

        Called whenever a new best-eval checkpoint is saved. The snapshot
        is detached from the graph and put in eval mode — it's used only
        to compute reference logits for the KL penalty, never trained.
        """
        import copy
        self.reference_actor = copy.deepcopy(self.actor)
        self.reference_actor.eval()
        for p in self.reference_actor.parameters():
            p.requires_grad_(False)

    def _kl_to_reference(self, logits, z, sig, ear):
        """KL(current || reference) at the same input states.

        Returns 0 if no reference has been set yet (i.e. before the first
        best-eval checkpoint). Tensors must already be on self.device.
        """
        if self.reference_actor is None or self.actor_kl_coef <= 0:
            return torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            ref_logits = self.reference_actor(z, sig, ear)
        log_p = F.log_softmax(logits, dim=-1)
        log_q = F.log_softmax(ref_logits, dim=-1)
        p = log_p.exp()
        # KL(p || q) = sum p * (log p - log q); per-state then mean over batch.
        return (p * (log_p - log_q)).sum(dim=-1).mean()

    def _evaluate_policy(self, n_episodes: int, max_steps: int,
                         seed_base: int, auto_fire: bool,
                         epsilon: float = 0.05,
                         env_factory=None) -> float:
        """Run eval episodes on a fresh env and return mean reward.

        Uses the same `act()` interface as final evaluation so what we measure
        in training matches what we report at the end. Default epsilon=0.05
        matches standard Atari evaluation convention (5% random) — pure greedy
        (epsilon=0) tends to lock the policy into NOOP-trap states and
        underestimates the true policy quality.
        """
        if env_factory is not None:
            eval_env = env_factory()
        else:
            # Best-effort clone for the two envs we ship.
            game = getattr(self.env, "_game", None)
            if game is not None:
                eval_env = type(self.env)(game)
            else:
                import copy as _copy
                eval_env = _copy.deepcopy(self.env)
        rewards = []
        try:
            for ep in range(n_episodes):
                frame = eval_env.reset(seed=seed_base + ep)
                self.world_model.reset_memory(batch_size=1, device=self.device)
                ep_r = 0.0
                for t in range(max_steps):
                    force = 1 if (auto_fire and t < 5 and eval_env.num_actions > 1) else -1
                    a = self.act(frame, deterministic=True, epsilon=epsilon,
                                 force_action=force)
                    frame, r, done = eval_env.step(a)
                    ep_r += r
                    if done:
                        break
                rewards.append(ep_r)
        finally:
            eval_env.close()
        # Restore training-time hidden state (the real env's hidden state was
        # left intact; we used a separate eval_env). Reset memory back to
        # batch=1 just in case downstream code assumes it.
        self.world_model.reset_memory(batch_size=1, device=self.device)
        return float(np.mean(rewards)) if rewards else float("-inf")

    def _teacher_instruction(self, reward):
        """Simple teacher: maps game events to instruction words."""
        if reward > 0:
            return "score"
        elif reward < 0:
            return "dodge"
        return "explore"

    def _compute_ear_input(self, z, instruction_word=None):
        """Build ear_input = [inner_voice | instruction_embed].
        Inner voice: scene narration + slot object descriptions.
        Instruction: what the teacher says to do."""
        with torch.no_grad():
            z_norm = F.normalize(z.unsqueeze(0) if z.dim() == 1 else z, dim=-1)
            inner_voice = self.lang_module.bridge.forward_v2t(z_norm)
            if z.dim() == 1:
                inner_voice = inner_voice.squeeze(0)

            slots = self.world_model.last_slots
            if slots is not None:
                if slots.dim() == 2:
                    slots = slots.unsqueeze(0)
                _, slot_embeds = self.lang_module.name_slots(slots)
                slot_avg = slot_embeds.mean(dim=1)
                if z.dim() == 1:
                    slot_avg = slot_avg.squeeze(0)
                inner_voice = inner_voice + slot_avg

            if instruction_word is not None:
                widx = self.lang_module.vocab.encode(instruction_word)
                if widx < 0:
                    widx = 0
                device = z.device
                if z.dim() == 1:
                    idx_t = torch.tensor([widx], device=device)
                    instr_embed = self.lang_module.text_encoder(idx_t).squeeze(0)
                else:
                    idx_t = torch.full((z.shape[0],), widx, dtype=torch.long, device=device)
                    instr_embed = self.lang_module.text_encoder(idx_t)
            else:
                if z.dim() == 1:
                    instr_embed = torch.zeros(self.latent_dim, device=z.device)
                else:
                    instr_embed = torch.zeros(z.shape[0], self.latent_dim, device=z.device)

            return torch.cat([inner_voice, instr_embed], dim=-1)

    def _curiosity_action(self, z_cached, signals, ear_input=None):
        """Curiosity explores using the SAME unified actor.

        Uses BOTH temperature (smooths the policy distribution) AND
        epsilon-greedy (forces uniform-random action with probability eps).
        The previous version only used temperature, which collapsed when
        the actor became confident — softmax(huge_logits / 2) is still very
        peaky. epsilon-greedy guarantees state coverage independent of how
        confident the actor has become.
        """
        well = self.value_sys._doing_well
        if well <= 0:
            temp, eps = 2.0, 0.30
        elif well < 0.3:
            temp, eps = 1.2, 0.15
        else:
            temp, eps = 0.7, 0.05

        if torch.rand(1).item() < eps:
            # Biased exploration: when an exploration prior is set, sample
            # the random kick from that distribution instead of uniform.
            if self._exploration_prior is not None:
                p = torch.tensor(self._exploration_prior,
                                 dtype=torch.float32, device=self.device)
                return int(torch.multinomial(p, 1).item())
            return int(torch.randint(0, self.env.num_actions, (1,)).item())
        with torch.no_grad():
            logits = self.actor(z_cached.unsqueeze(0),
                                signals.unsqueeze(0),
                                ear_input.unsqueeze(0) if ear_input is not None else None).squeeze(0)
            probs = F.softmax(logits / temp, dim=-1)
            return int(torch.multinomial(probs, 1).item())

    def _normalize_reward(self, reward):
        """Welford's online normalization with loss aversion.
        Only ACTUAL negative rewards (dying, losing) are amplified —
        not neutral steps that happen to normalize below zero."""
        feared = reward * self.loss_aversion if reward < 0 else reward
        self._reward_count += 1
        delta = feared - self._reward_running_mean
        self._reward_running_mean += delta / self._reward_count
        delta2 = feared - self._reward_running_mean
        self._reward_running_var += (delta * delta2 - self._reward_running_var) / max(self._reward_count, 2)
        std = max(self._reward_running_var ** 0.5, 0.1)
        return (feared - self._reward_running_mean) / std

    def _normalize_reward_readonly(self, reward):
        """Normalize using current stats WITHOUT updating them."""
        feared = reward * self.loss_aversion if reward < 0 else reward
        std = max(self._reward_running_var ** 0.5, 0.1)
        return (feared - self._reward_running_mean) / std

    def _train_critic_from_replay(self, replay, rng, batch_size, n_step=5):
        """Train critic on REAL transitions using n-step returns.
        Instead of 1-step TD (noisy), accumulates n real rewards before bootstrapping.
        Replay stores consecutive transitions within episodes, so replay[i:i+n]
        gives a valid sequence as long as no done=True appears in between."""
        max_start = len(replay) - n_step
        if max_start < 1:
            return
        bs = min(batch_size, max_start)
        start_idxs = rng.choice(max_start, bs, replace=False)

        z_starts, z_ends, sigs, targets = [], [], [], []
        for idx in start_idxs:
            G = 0.0
            gamma_k = 1.0
            final_z = replay[idx][2]
            hit_done = False
            for k in range(n_step):
                entry = replay[idx + k]
                G += gamma_k * entry[3]
                gamma_k *= self.gamma
                final_z = entry[2]
                if entry[4]:
                    hit_done = True
                    break

            z_starts.append(replay[idx][0])
            z_ends.append(final_z)
            sigs.append(replay[idx][5])
            targets.append((G, gamma_k, hit_done))

        with torch.no_grad():
            z_now = torch.stack(z_starts).to(self.device)
            z_end = torch.stack(z_ends).to(self.device)
            b_sig = torch.stack(sigs).to(self.device)
            v_end = self.slow_critic(z_end, b_sig).squeeze(-1)
            td_target = torch.tensor(
                [g + (0.0 if done else gk * v.item())
                 for (g, gk, done), v in zip(targets, v_end)],
                dtype=torch.float32, device=self.device)

        self.critic_optimizer.zero_grad()
        self.critic.train()
        if self.critic_distributional:
            critic_loss = self.critic.loss(z_now, b_sig, td_target)
        else:
            v_now = self.critic(z_now, b_sig).squeeze(-1)
            critic_loss = F.mse_loss(v_now, td_target)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        with torch.no_grad():
            for sp, tp in zip(self.critic.parameters(), self.slow_critic.parameters()):
                tp.data.mul_(self.slow_ema).add_(sp.data, alpha=1 - self.slow_ema)

    def _train_actor_from_replay(self, replay, rng, batch_size, n_step=5):
        """REINFORCE-with-baseline on REAL transitions.

        Complements `_dream_train`. Dream-only training causes the actor to
        over-fit to the world model's imagination — it finds the single
        action that maximizes predicted reward in dream and exploits it,
        which manifests as mode collapse at evaluation. Real-trajectory
        updates anchor the policy to actions that actually paid off in the
        real environment, not just in dreams.

        Uses n-step Monte Carlo returns minus a critic baseline for
        variance reduction. Adds a per-batch entropy bonus.
        """
        if len(replay) < batch_size + n_step:
            return 0.0
        max_start = len(replay) - n_step
        bs = min(batch_size, max_start)
        start_idxs = rng.choice(max_start, bs, replace=False)

        z_starts, actions, sigs, ears, returns = [], [], [], [], []
        for idx in start_idxs:
            G = 0.0
            gamma_k = 1.0
            for k in range(n_step):
                entry = replay[idx + k]
                G += gamma_k * entry[3]
                gamma_k *= self.gamma
                if entry[4]:
                    break
            z_starts.append(replay[idx][0])
            actions.append(replay[idx][1])
            sigs.append(replay[idx][5])
            ears.append(replay[idx][6])
            returns.append(G)

        z = torch.stack(z_starts).to(self.device)
        a = torch.tensor(actions, dtype=torch.long, device=self.device)
        sig = torch.stack(sigs).to(self.device)
        ear = torch.stack(ears).to(self.device)
        ret = torch.tensor(returns, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            v = self.critic(z, sig)
        advantage = (ret - v).detach()
        adv_mean = advantage.mean()
        adv_std = advantage.std().clamp(min=1e-6)
        advantage = (advantage - adv_mean) / adv_std

        self.actor.train()
        logits = self.actor(z, sig, ear)
        log_probs = F.log_softmax(logits, dim=-1)
        chosen_log_prob = log_probs.gather(-1, a.unsqueeze(-1)).squeeze(-1)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)

        actor_loss = -(advantage * chosen_log_prob).mean()
        actor_loss = actor_loss - self.entropy_coeff * entropy.mean()
        # Trust-region anchor: pull current actor toward the last best policy
        # at the SAME real states we just sampled. Has no effect until the
        # first best checkpoint is saved (reference_actor stays None).
        kl = self._kl_to_reference(logits, z, sig, ear)
        actor_loss = actor_loss + self.actor_kl_coef * kl

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()
        return float(actor_loss.item())

    def _dream_train(self, replay, rng, horizon=15, n_starts=None):
        """DreamerV3-style: imagine trajectories, compute lambda-returns,
        train actor with REINFORCE + entropy, critic with MSE.
        Uses prioritized replay: 50% starts from rewarding transitions so
        the actor practices scoring, not just idle paddle movement."""
        if n_starts is None:
            n_starts = 64 if self.device.type == "cuda" else 16
        if len(replay) < n_starts:
            return 0.0

        na = self.env.num_actions

        n_reward = max(n_starts // 2, 1)
        reward_idxs = [i for i in range(len(replay)) if abs(replay[i][3]) > 0]
        if len(reward_idxs) >= n_reward:
            chosen_r = rng.choice(reward_idxs, n_reward, replace=False).tolist()
            remaining = [i for i in range(len(replay)) if i not in set(chosen_r)]
            chosen_u = rng.choice(remaining, n_starts - n_reward, replace=False).tolist()
            idxs = chosen_r + chosen_u
        else:
            idxs = rng.choice(len(replay), n_starts, replace=False)

        z = torch.stack([replay[i][0] for i in idxs]).to(self.device)
        dream_signals = torch.stack([replay[i][5] for i in idxs]).to(self.device)
        dream_ear = torch.stack([replay[i][6] for i in idxs]).to(self.device)
        # Extract stored instruction half (second latent_dim) for reuse in dreams
        dream_instr = dream_ear[:, self.latent_dim:]

        imagined_z = [z]
        imagined_logits = []
        imagined_actions = []
        imagined_rewards = []
        imagined_continue = []

        # Per-state entropy floor (probability mixing).
        # Without this, the actor's softmax can become peaked enough that
        # one action absorbs ~100% of mass in every state, and
        # `dist.sample()` always picks it — collapsing the policy. We mix
        # the actor's distribution with a uniform distribution at floor
        # `mix_floor`, guaranteeing every action keeps at least
        # `mix_floor / num_actions` probability everywhere.
        mix_floor = 0.10
        uniform_p = mix_floor / na

        dream_h = None
        # Track per-step actor INPUTS so we can compute KL to the reference
        # actor at the same imagined states (and only those).
        imagined_z_inputs = []
        imagined_ears = []
        for t in range(horizon):
            with torch.no_grad():
                z_norm = F.normalize(z, dim=-1)
                inner_voice = self.lang_module.bridge.forward_v2t(z_norm)
            dream_ear_t = torch.cat([inner_voice, dream_instr], dim=-1)
            imagined_z_inputs.append(z)
            imagined_ears.append(dream_ear_t)
            logits = self.actor(z, dream_signals, ear_input=dream_ear_t)
            probs = F.softmax(logits, dim=-1)
            mixed_probs = (1.0 - mix_floor) * probs + uniform_p
            dist = torch.distributions.Categorical(probs=mixed_probs)
            action = dist.sample()
            imagined_logits.append(logits)
            imagined_actions.append(action)
            with torch.no_grad():
                pred_rew = symexp(self.reward_pred(z, action))
                pred_done = torch.sigmoid(self.done_pred(z, action))
                cont = 1.0 - pred_done
                z, dream_h = self.world_model.predict_next(z, action, h=dream_h)
            imagined_rewards.append(pred_rew)
            imagined_continue.append(cont)
            imagined_z.append(z)

        all_z = torch.stack(imagined_z, dim=0)
        imagined_rewards = torch.stack(imagined_rewards, dim=0)
        imagined_continue = torch.stack(imagined_continue, dim=0)
        sig_for_critic = dream_signals.unsqueeze(0).expand(horizon + 1, -1, -1)
        with torch.no_grad():
            all_values = self.slow_critic(
                all_z.reshape(-1, all_z.shape[-1]),
                sig_for_critic.reshape(-1, self.actor.signal_dim))
            all_values = all_values.reshape(horizon + 1, n_starts)

        returns = torch.zeros(horizon, n_starts, device=self.device)
        last_val = all_values[-1]
        for t in reversed(range(horizon)):
            cont = imagined_continue[t]
            returns[t] = imagined_rewards[t] + self.gamma * cont * (
                self.lam * last_val + (1 - self.lam) * all_values[t + 1])
            last_val = returns[t]

        z_flat = all_z[:-1].reshape(-1, all_z.shape[-1])
        sig_flat = sig_for_critic[:-1].reshape(-1, self.actor.signal_dim)
        critic_values = self.critic(z_flat, sig_flat).reshape(horizon, n_starts)
        if self.critic_distributional:
            critic_loss = self.critic.loss(z_flat, sig_flat,
                                           returns.detach().reshape(-1))
        else:
            critic_loss = F.mse_loss(critic_values, returns.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        advantages = (returns - critic_values).detach()
        adv_mean = advantages.mean()
        adv_std = advantages.std().clamp(min=1e-6)
        advantages = (advantages - adv_mean) / adv_std

        momentum = self.value_sys.get_actor_momentum()
        actor_loss = torch.tensor(0.0, device=self.device)
        kl_acc = torch.tensor(0.0, device=self.device)
        for t in range(horizon):
            # Match the sampling distribution used during rollout
            # (mixed probs, not raw logits). REINFORCE is only unbiased
            # when log_prob comes from the actual sampling distribution.
            probs_t = F.softmax(imagined_logits[t], dim=-1)
            mixed_t = (1.0 - mix_floor) * probs_t + uniform_p
            dist = torch.distributions.Categorical(probs=mixed_t)
            log_prob = dist.log_prob(imagined_actions[t])
            entropy = dist.entropy()
            actor_loss -= momentum * (advantages[t] * log_prob).mean()
            actor_loss -= self.entropy_coeff * entropy.mean()
            # KL anchor at imagined states. Reference actor is None until the
            # first best checkpoint is saved, in which case _kl_to_reference
            # returns 0 and this is a no-op.
            kl_acc = kl_acc + self._kl_to_reference(
                imagined_logits[t], imagined_z_inputs[t],
                dream_signals, imagined_ears[t])

        actor_loss /= horizon
        actor_loss = actor_loss + self.actor_kl_coef * (kl_acc / horizon)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()

        with torch.no_grad():
            for sp, tp in zip(self.critic.parameters(), self.slow_critic.parameters()):
                tp.data.mul_(self.slow_ema).add_(sp.data, alpha=1 - self.slow_ema)

        return actor_loss.item()

    def _print_report(self, h, wc, elapsed, frames, cur, plan, corr):
        print(f"\n{'='*60}\n  TRAINING COMPLETE\n{'='*60}")
        print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min) | Frames: {frames:,} | Device: {self.device}")
        first_wm = np.mean(h["wm_loss"][:50]) if len(h["wm_loss"]) > 50 else h["wm_loss"][0]
        final_wm = np.mean(h["wm_loss"][-50:])
        print(f"\n  1. World Model:     {first_wm:.5f} -> {final_wm:.5f} ({first_wm/max(final_wm,1e-8):.0f}x)")
        print(f"  2. Curiosity:       {cur:,} actions")
        print(f"  3. Metacognition:   {len(self.knowledge_tree.nodes)-1} concepts")
        first_l = np.mean(h["lang_loss"][:50]) if len(h["lang_loss"]) > 50 else 0
        print(f"  4. Language:        {first_l:.3f} -> {np.mean(h['lang_loss'][-50:]):.3f}")
        print(f"  5. Values:          wellbeing={self.value_sys.get_wellbeing():.3f}")
        print(f"  6. Memory+Sleep:    {len(self.episodic)} memories, {self.sleep_cycle.cycle_count} cycles")
        print(f"  7. Self-correction: {corr} corrections | Actor: {plan:,} actions")
        gs = self.intuition.stats
        gh = gs.get('health', {})
        print(f"  8. Intuition Gate:  yes={gs['yes_rate']:.0%} no={gs['no_rate']:.0%} "
              f"explore={gs['exploration']:.0%} reopens={gs['reopens']}")
        print(f"     Self-diagnosis:  grads={'OK' if gh.get('grad_healthy') else 'SICK'} "
              f"weights={'moving' if gh.get('weights_moving') else 'stuck'} "
              f"actor={'confused' if gh.get('actor_confused') else 'confident'} "
              f"loss={'improving' if gh.get('loss_improving') else 'plateau'}")
        with torch.no_grad():
            gate_vals = self.actor.ear_gate(torch.zeros(1, self.actor.ear_dim, device=self.device))
            gate_mean = gate_vals.mean().item()
        print(f"  9. Ears:            gate_openness={gate_mean:.3f} "
              f"(0=deaf, 0.5=listening)")
        slots = self.world_model.last_slots
        if slots is not None:
            slot_names, _ = self.lang_module.name_slots(
                slots[:1] if slots.dim() == 3 else slots.unsqueeze(0))
            print(f" 10. Object Slots:    {N_OBJECT_SLOTS} slots -> {slot_names}")
        used = [w for w, c in wc.items() if c > 0]
        print(f"\n  Words learned: {len(used)}/{len(self.env.vocab_words)} {used}")

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "world_model": self.world_model.state_dict(),
            "lang_module": self.lang_module.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "slow_critic": self.slow_critic.state_dict(),
            "reward_pred": self.reward_pred.state_dict(),
            "done_pred": self.done_pred.state_dict(),
            "num_actions": self.env.num_actions,
            "latent_dim": self.latent_dim,
            "vocab_words": self.env.vocab_words,
            "wm_frozen": self._wm_frozen,
            "reward_stats": {
                "mean": self._reward_running_mean,
                "var": self._reward_running_var,
                "count": self._reward_count,
            },
            "value_sys": {
                "doing_well": self.value_sys._doing_well,
                "getting_better": self.value_sys._getting_better,
                "reward_baseline": self.value_sys._reward_baseline,
                "episode_rewards": list(self.value_sys.episode_rewards),
            },
        }, path)
        print(f"  Saved to {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.world_model.load_state_dict(ckpt["world_model"])
        self.lang_module.load_state_dict(ckpt["lang_module"])
        if "actor" in ckpt:
            self.actor.load_state_dict(ckpt["actor"])
            self.critic.load_state_dict(ckpt["critic"])
            self.slow_critic.load_state_dict(ckpt.get("slow_critic", ckpt["critic"]))
        if "reward_pred" in ckpt:
            self.reward_pred.load_state_dict(ckpt["reward_pred"])
        if "done_pred" in ckpt:
            self.done_pred.load_state_dict(ckpt["done_pred"])
        if ckpt.get("wm_frozen"):
            self._wm_frozen = True
            for pg in self.wm_optimizer.param_groups:
                pg['lr'] = 3e-5
        if "reward_stats" in ckpt:
            rs = ckpt["reward_stats"]
            self._reward_running_mean = rs["mean"]
            self._reward_running_var = rs["var"]
            self._reward_count = rs["count"]
        if "value_sys" in ckpt:
            vs = ckpt["value_sys"]
            self.value_sys._doing_well = vs["doing_well"]
            self.value_sys._getting_better = vs["getting_better"]
            self.value_sys._reward_baseline = vs["reward_baseline"]
            for r in vs.get("episode_rewards", []):
                self.value_sys.episode_rewards.append(r)
        print(f"  Loaded from {path}")

    def describe(self, frame):
        self.world_model.eval(); self.lang_module.eval()
        with torch.no_grad():
            z = self.world_model.encode(frame.unsqueeze(0).to(self.device))
            _, sims = self.lang_module.name_scene(z)
            top = sims[0].argsort(descending=True)[:3]
            return [(self.lang_module.vocab.decode(i.item()), round(sims[0][i].item(), 2)) for i in top]

    def encode(self, frame):
        self.world_model.eval()
        with torch.no_grad():
            return self.world_model.encode(frame.unsqueeze(0).to(self.device)).squeeze(0)

    def predict(self, frame, action):
        self.world_model.eval()
        with torch.no_grad():
            z = self.world_model.encode(frame.unsqueeze(0).to(self.device))
            z_next, _ = self.world_model.predict_next(z, torch.tensor([action], device=self.device))
            return z_next.squeeze(0)

    def act(self, frame, deterministic: bool = False, epsilon: float = 0.0,
            force_action: int = -1) -> int:
        """Run a single inference step using the trained actor.

        Args:
            frame: env observation tensor of shape (C, H, W) in [0, 1].
            deterministic: if True, argmax over the policy logits.
                Otherwise sample from the categorical distribution.
            epsilon: probability of taking a uniformly random action instead
                of consulting the actor. Standard Atari practice (e.g. DQN
                evals use eps=0.05). Helps when the policy mode-collapses.
            force_action: if >= 0, return this action regardless of policy.
                Useful for FIRE-on-reset hacks in games like Breakout where
                the ball must be launched manually.
        Returns:
            int action id.
        """
        if force_action >= 0:
            return int(force_action)
        if epsilon > 0.0 and torch.rand(1).item() < epsilon:
            return int(torch.randint(0, self.env.num_actions, (1,)).item())

        self.world_model.eval()
        self.actor.eval()
        with torch.no_grad():
            z = self.world_model.encode(frame.unsqueeze(0).to(self.device)).squeeze(0)
            signals = torch.zeros(self.signal_bus.total_dim, device=self.device)
            ear_input = self._compute_ear_input(z, instruction_word=None)
            logits = self.actor(z.unsqueeze(0), signals.unsqueeze(0),
                                ear_input.unsqueeze(0)).squeeze(0)
            if deterministic:
                return int(logits.argmax(dim=-1).item())
            probs = F.softmax(logits, dim=-1)
            return int(torch.multinomial(probs, 1).item())
