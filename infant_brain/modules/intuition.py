"""
Intuition Gate — self-aware learning decision.

Instead of checking external results ("did reward go up?"), the brain
inspects its OWN internals to decide whether learning is needed:

  - Gradient health:  "Are my gradients flowing or stuck?"
  - Weight dynamics:  "Are my weights actually changing?"
  - Actor confidence: "Am I confused or certain?"
  - Loss trajectory:  "Is my internal loss improving?"
  - Representation:   "Am I seeing diverse states or blind?"

Like a student who FEELS they don't understand — before failing the test.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np


class SelfMonitor:
    """Watches the brain's internal health signals — no external reward needed."""

    def __init__(self, window=50):
        self._grad_norms = deque(maxlen=window)
        self._weight_deltas = deque(maxlen=window)
        self._wm_losses = deque(maxlen=window)
        self._actor_entropies = deque(maxlen=window)
        self._z_diversity = deque(maxlen=window)
        self._prev_weights = None

    def snapshot_weights(self, model):
        """Save current weights for delta comparison next time."""
        self._prev_weights = {
            name: p.data.clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    def measure_weight_change(self, model):
        """How much did the weights actually move since last snapshot?"""
        if self._prev_weights is None:
            return 0.0
        total_delta = 0.0
        count = 0
        for name, p in model.named_parameters():
            if name in self._prev_weights and p.requires_grad:
                delta = (p.data - self._prev_weights[name]).abs().mean().item()
                total_delta += delta
                count += 1
        avg_delta = total_delta / max(count, 1)
        self._weight_deltas.append(avg_delta)
        return avg_delta

    def measure_gradient_health(self, model):
        """Are gradients flowing (healthy) or vanishing/exploding (sick)?"""
        total_norm = 0.0
        count = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item()
                count += 1
        avg_norm = total_norm / max(count, 1)
        self._grad_norms.append(avg_norm)
        return avg_norm

    def measure_actor_confidence(self, logits):
        """Is the actor sure about its actions or completely confused?
        High entropy = confused = needs more learning.
        Low entropy = confident = maybe stop learning."""
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean().item()
            max_entropy = np.log(logits.shape[-1])
            normalized = entropy / max(max_entropy, 1e-8)
        self._actor_entropies.append(normalized)
        return normalized

    def measure_loss_trend(self, loss_value):
        """Is the internal loss going down (learning) or stagnant (stuck)?"""
        self._wm_losses.append(loss_value)
        if len(self._wm_losses) < 10:
            return 0.0
        recent = list(self._wm_losses)
        first_half = np.mean(recent[:len(recent)//2])
        second_half = np.mean(recent[len(recent)//2:])
        return (first_half - second_half) / max(abs(first_half), 1e-8)

    def measure_z_diversity(self, z):
        """Are the latent representations diverse or all the same?
        If the brain encodes everything the same → it's blind."""
        self._z_diversity.append(z.detach().cpu())
        if len(self._z_diversity) < 10:
            return 1.0
        recent = torch.stack(list(self._z_diversity)[-20:])
        variance = recent.var(dim=0).mean().item()
        return min(1.0, variance * 10)

    @property
    def health_report(self):
        """Full self-diagnosis in one dict."""
        def _safe_mean(d):
            return float(np.mean(list(d))) if d else 0.0
        def _safe_trend(d):
            if len(d) < 6:
                return 0.0
            vals = list(d)
            first = np.mean(vals[:len(vals)//2])
            second = np.mean(vals[len(vals)//2:])
            return float(second - first)

        return {
            'grad_norm': _safe_mean(self._grad_norms),
            'grad_healthy': 1e-6 < _safe_mean(self._grad_norms) < 10.0,
            'weight_change': _safe_mean(self._weight_deltas),
            'weights_moving': _safe_mean(self._weight_deltas) > 1e-7,
            'actor_entropy': _safe_mean(self._actor_entropies),
            'actor_confused': _safe_mean(self._actor_entropies) > 0.7,
            'loss_improving': _safe_trend(self._wm_losses) < 0,
            'z_diverse': _safe_mean(self._z_diversity) > 0.1 if self._z_diversity else True,
        }


class IntuitionGate(nn.Module):
    """Self-aware learning gate.

    Instead of "did reward go up?" it asks:
      - "Are my gradients healthy?"    → vanishing = stuck, need to learn differently
      - "Are my weights changing?"     → no change = wasted compute, stop
      - "Am I confused?"               → high entropy = keep learning
      - "Is my loss improving?"        → plateau = try something new
      - "Am I seeing diverse states?"  → all same = blind, retrain encoder

    The gate learns to map these self-signals to a yes/no decision.
    """

    def __init__(self, latent_dim=256, hidden_dim=64):
        super().__init__()
        self.self_monitor = SelfMonitor(window=50)
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.exploration_rate = 1.0
        self._total_decisions = 0
        self._yes_count = 0
        self._no_count = 0
        self._reopen_count = 0
        self._skip_streak = 0
        self._forced_count = 0
        self._outcome_buffer = []

    def read_self(self):
        """Gather all self-monitoring signals into a tensor."""
        report = self.self_monitor.health_report
        return torch.tensor([
            report['grad_norm'],
            report['weight_change'],
            report['actor_entropy'],
            1.0 if report['loss_improving'] else 0.0,
            1.0 if report.get('z_diverse', True) else 0.0,
        ], dtype=torch.float32)

    def decide(self, force_learn=False):
        """Should I learn from this experience?

        Looks INWARD (own health signals), not outward (reward).
        """
        self._total_decisions += 1

        if force_learn:
            self._yes_count += 1
            self._forced_count += 1
            self._skip_streak = 0
            return True, 0.0

        if torch.rand(1).item() < self.exploration_rate:
            coin = torch.rand(1).item() > 0.5
            if coin:
                self._yes_count += 1
                self._skip_streak = 0
            else:
                self._no_count += 1
                self._skip_streak += 1
            return coin, 0.0

        self_signals = self.read_self()
        with torch.no_grad():
            logit = self.net(self_signals).item()
            prob = torch.sigmoid(torch.tensor(logit)).item()

        should_learn = prob > 0.5
        confidence = abs(prob - 0.5) * 2

        if should_learn:
            self._yes_count += 1
            self._skip_streak = 0
        else:
            self._no_count += 1
            self._skip_streak += 1

        return should_learn, confidence

    def check_and_reopen(self):
        """Self-correction: if skipping too much and internals look sick, reopen."""
        report = self.self_monitor.health_report
        sick = (not report['grad_healthy'] or
                report['actor_confused'] or
                not report['loss_improving'])

        if self._skip_streak > 30 and sick:
            self._skip_streak = 0
            self._reopen_count += 1
            self.exploration_rate = min(1.0, self.exploration_rate + 0.3)
            return True
        return False

    def record_outcome(self, self_signals, was_useful):
        """Remember: did learning from this state actually help?"""
        self._outcome_buffer.append((self_signals.clone(), float(was_useful)))
        if len(self._outcome_buffer) > 200:
            self._outcome_buffer = self._outcome_buffer[-200:]

    def train_gate(self, lr=1e-4):
        """Train the gate on past outcomes: did my yes/no decisions pay off?"""
        if len(self._outcome_buffer) < 16:
            return 0.0

        signals = torch.stack([s for s, _ in self._outcome_buffer[-64:]])
        outcomes = torch.tensor([o for _, o in self._outcome_buffer[-64:]])
        targets = (outcomes > 0).float()

        self.train()
        logits = self.net(signals).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, targets)

        loss.backward()
        with torch.no_grad():
            for p in self.parameters():
                if p.grad is not None:
                    p.data -= lr * p.grad
        self.zero_grad(set_to_none=True)

        return loss.item()

    def update_exploration(self, dev_stage, episode, total_episodes):
        """Decay exploration as the brain matures."""
        stage_factor = {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2, 5: 0.1}
        progress = min(1.0, episode / max(total_episodes * 0.7, 1))
        self.exploration_rate = stage_factor.get(dev_stage, 0.1) * (1.0 - 0.5 * progress)
        self.exploration_rate = max(0.05, self.exploration_rate)

    @property
    def stats(self):
        total = max(self._total_decisions, 1)
        return {
            'total': self._total_decisions,
            'yes_rate': self._yes_count / total,
            'no_rate': self._no_count / total,
            'forced': self._forced_count,
            'exploration': self.exploration_rate,
            'reopens': self._reopen_count,
            'health': self.self_monitor.health_report,
        }
