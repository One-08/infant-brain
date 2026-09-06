"""
Value System — the brain's emotional core. Simple, clean, correct timescale.

3 core signals (episode-level):
  doing_well    — am I scoring?
  getting_better — am I improving?
  surprised     — did something unexpected happen?

All behavioral variables derive from these 3.
Backup of previous version: value_system_backup.py
"""

from collections import deque
import numpy as np


class ValueSystem:
    def __init__(self, concepts=None, history_len=100):
        self.episode_rewards = deque(maxlen=50)
        self.step_errors = deque(maxlen=500)
        self.error_baseline = 0.01
        self.baseline_momentum = 0.99

        self.concepts = list(concepts) if concepts else []
        self.error_history = {c: deque(maxlen=history_len) for c in self.concepts}
        self.attempt_history = {c: deque(maxlen=history_len) for c in self.concepts}
        self.action_history = {c: deque(maxlen=50) for c in self.concepts}
        self.strategy_noise = {c: 0.0 for c in self.concepts}
        self.recent_concepts = deque(maxlen=50)

        self._doing_well = 0.0
        self._getting_better = 0.0
        self._surprised = 0.0
        self.wellbeing = 0.0
        self.wellbeing_history = deque(maxlen=500)

        self._reward_baseline = 0.0
        self._plateau_counter = 0
        self._improving_streak = 0
        self._wm_loss = 1.0

        self.signals = {
            "doing_well": 0.0,
            "getting_better": 0.0,
            "surprised": 0.0,
        }

    # ---- step-level: only prediction tracking ----

    def process(self, prediction_error, concept, competence, learning_progress,
                reward=0.0):
        """Called every step. Tracks prediction quality only."""
        self.step_errors.append(prediction_error)
        self.error_baseline = (self.baseline_momentum * self.error_baseline +
                               (1 - self.baseline_momentum) * prediction_error)

        if concept:
            self.error_history.setdefault(concept, deque(maxlen=100)).append(prediction_error)
            self.attempt_history.setdefault(concept, deque(maxlen=100)).append(1)
            self.recent_concepts.append(concept)

        self._surprised = float(prediction_error > self.error_baseline * 2.0)
        self.signals["surprised"] = self._surprised

        self.wellbeing = self._doing_well + 0.5 * self._getting_better
        self.wellbeing_history.append(self.wellbeing)
        return self.signals.copy()

    # ---- episode-level: reward reflection ----

    def end_episode(self, episode_reward):
        """Called once per episode — the brain reflects on how it did.
        Uses adaptive baseline: 'good' is relative to own recent history."""
        self.episode_rewards.append(episode_reward)

        if len(self.episode_rewards) >= 3:
            recent = list(self.episode_rewards)[-10:]
            avg = np.mean(recent)
            std = max(np.std(list(self.episode_rewards)), 0.5)
            self._doing_well = float(np.clip((avg - self._reward_baseline) / std, -1.0, 1.0))
            self._reward_baseline = 0.95 * self._reward_baseline + 0.05 * avg
        else:
            self._doing_well = 0.0

        if len(self.episode_rewards) >= 10:
            eps = list(self.episode_rewards)
            old_avg = np.mean(eps[-20:-10]) if len(eps) >= 20 else np.mean(eps[:len(eps)//2])
            new_avg = np.mean(eps[-10:])
            diff = new_avg - old_avg
            self._getting_better = float(np.clip(
                diff / max(abs(old_avg), 0.5), -1.0, 1.0))
            if abs(diff) < 0.1:
                self._plateau_counter += 1
            else:
                self._plateau_counter = 0
            if self._getting_better > 0:
                self._improving_streak += 1
            else:
                self._improving_streak = 0
        else:
            self._getting_better = 0.0

        self.signals["doing_well"] = self._doing_well
        self.signals["getting_better"] = self._getting_better

    # ---- output methods: drive all behavioral variables ----

    def get_curiosity_drive(self):
        """How curious? Adaptive: never fully satisfied, plateau raises curiosity."""
        well = self._doing_well
        better = self._getting_better
        plateau_boost = min(0.15, self._plateau_counter * 0.02)
        if well > 0.3 and better >= 0:
            base = 0.10
        elif well > 0.1 and better >= 0:
            base = 0.20
        elif well > 0 and better < 0:
            base = 0.35
        elif well <= 0 and better < -0.2:
            base = 0.60
        else:
            base = 0.35
        return min(0.60, base + plateau_boost)

    def get_memory_trust(self):
        """Trust past experience? Good results → trust. Bad → distrust."""
        well = self._doing_well
        if well > 0.3:
            return 0.7
        if well > 0:
            return 0.4
        return 0.1

    def get_learning_urgency(self):
        """How fast to learn world model? Getting worse → push harder. Improving → careful."""
        better = self._getting_better
        if better < -0.2:
            return 2.0
        if better > 0.2:
            return 0.7
        return 1.0

    def update_wm_loss(self, wm_loss):
        """Called each episode so momentum can verify world model quality."""
        self._wm_loss = wm_loss

    def get_actor_momentum(self):
        """Smooth momentum — avoids the 8x swings (0.5→4.0) that destabilize learning.
        Range clamped to [0.5, 2.0] based on ablation showing 4.0x causes oscillation."""
        better = self._getting_better
        streak = self._improving_streak
        wm_reliable = self._wm_loss < 0.1
        has_enough_data = len(self.episode_rewards) >= 15

        if better < -0.2:
            return 0.5
        if better < 0:
            return 0.7

        if not has_enough_data:
            return 1.0

        if streak >= 3 and wm_reliable and better > 0.2:
            return 2.0
        if streak >= 2 and wm_reliable and better > 0.1:
            return 1.5
        if better > 0:
            return 1.2

        return 1.0

    def assess_quality(self, reward, pred_error):
        """Simple quality label for memory storage."""
        if reward > 0:
            return "good"
        if reward < 0:
            return "bad"
        if pred_error > self.error_baseline * 2:
            return "surprising"
        return "neutral"

    def record_action(self, concept, action):
        if concept in self.action_history:
            self.action_history[concept].append(action)

    def get_signals(self):
        return self.signals.copy()

    def get_wellbeing(self):
        return self.wellbeing
