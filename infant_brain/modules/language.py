"""
Language Module — cross-modal prediction between visual latent space and words.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


VOCAB_WORDS = [
    "red", "green", "yellow",
    "normal", "slippery", "heavy",
    "up", "down", "left", "right",
    "push", "slide", "stuck",
]


class Vocabulary:
    def __init__(self, words=None):
        words = words or VOCAB_WORDS
        self.word2idx = {w: i for i, w in enumerate(words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.size = len(words)

    def encode(self, word):
        return self.word2idx.get(word, -1)

    def decode(self, idx):
        return self.idx2word.get(idx, "<unk>")

    def __len__(self):
        return self.size

    def __contains__(self, word):
        return word in self.word2idx


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, word_idx):
        return self.embedding(word_idx)


class CrossModalBridge(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=64):
        super().__init__()
        self.visual_to_text = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.text_to_visual = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward_v2t(self, z):
        return self.visual_to_text(z)

    def forward_t2v(self, e):
        return self.text_to_visual(e)


class LanguageModule(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=64, vocab_words=None, slot_dim=64):
        super().__init__()
        self.vocab = Vocabulary(vocab_words)
        self.text_encoder = TextEncoder(self.vocab.size, latent_dim)
        self.bridge = CrossModalBridge(latent_dim, hidden_dim)
        self.latent_dim = latent_dim
        self.slot_proj = nn.Linear(slot_dim, latent_dim)

    def compute_loss(self, z, word_idx, temperature=0.1):
        e = self.text_encoder(word_idx)
        z_n = F.normalize(z, dim=-1)
        e_n = F.normalize(e, dim=-1)
        all_e = F.normalize(self.text_encoder.embedding.weight, dim=-1)

        e_pred = self.bridge.forward_v2t(z_n)
        e_pred_n = F.normalize(e_pred, dim=-1)
        logits = e_pred_n @ all_e.T / temperature
        loss_v2t = F.cross_entropy(logits, word_idx)

        z_pred = self.bridge.forward_t2v(e_n)
        loss_t2v = F.mse_loss(z_pred, z_n.detach())

        total = loss_v2t + loss_t2v
        metrics = {"lang_loss": total.item(), "v2t_loss": loss_v2t.item(), "t2v_loss": loss_t2v.item()}
        return total, metrics

    def name_scene(self, z):
        with torch.no_grad():
            z_n = F.normalize(z, dim=-1)
            e_pred = self.bridge.forward_v2t(z_n)
            all_embeds = F.normalize(self.text_encoder.embedding.weight, dim=-1)
            sims = F.cosine_similarity(e_pred.unsqueeze(1), all_embeds.unsqueeze(0), dim=-1)
            best_idx = sims.argmax(dim=-1)
            words = [self.vocab.decode(i.item()) for i in best_idx]
        return words, sims

    def name_slots(self, slots):
        """Name each object slot using the language bridge.
        Args:
            slots: (B, K, slot_dim) raw slot representations
        Returns:
            slot_words: list of K words (best match per slot)
            slot_embeds: (B, K, latent_dim) language embeddings per slot
        """
        with torch.no_grad():
            B, K, _ = slots.shape
            s_proj = self.slot_proj(slots)
            s_norm = F.normalize(s_proj, dim=-1)
            e_pred = self.bridge.forward_v2t(s_norm.reshape(B * K, -1))
            all_embeds = F.normalize(self.text_encoder.embedding.weight, dim=-1)
            sims = F.cosine_similarity(
                e_pred.unsqueeze(1), all_embeds.unsqueeze(0), dim=-1)
            best_idx = sims.argmax(dim=-1)
            words = [self.vocab.decode(i.item()) for i in best_idx[:K]]
            return words, e_pred.reshape(B, K, -1)

    def imagine_word(self, word, device="cpu"):
        idx = self.vocab.encode(word)
        if idx < 0:
            return None
        with torch.no_grad():
            word_t = torch.tensor([idx], device=device)
            e = self.text_encoder(word_t)
            e_n = F.normalize(e, dim=-1)
            z_pred = self.bridge.forward_t2v(e_n)
        return z_pred

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
