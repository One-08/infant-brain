"""
Metacognitive Monitor — tracks per-concept competence and learning progress.

Competence blends prediction accuracy WITH reward performance.
Predicting well but not scoring = not truly competent.
"""

from collections import deque
import numpy as np


ACCURACY_THRESHOLD = 0.01


class MetacognitiveMonitor:
    def __init__(self, concepts, window=200):
        self.concepts = list(concepts)
        self.window = window
        self.error_history = {c: deque(maxlen=window) for c in self.concepts}
        self.correct_history = {c: deque(maxlen=window) for c in self.concepts}
        self.interaction_count = {c: 0 for c in self.concepts}
        self.best_error = {c: float("inf") for c in self.concepts}
        self.reward_competence = 0.0
        self.episode_rewards = deque(maxlen=30)

    def record(self, concept, prediction_error):
        if concept not in self.error_history:
            self.error_history[concept] = deque(maxlen=self.window)
            self.correct_history[concept] = deque(maxlen=self.window)
            self.best_error[concept] = float("inf")
        self.error_history[concept].append(prediction_error)
        self.interaction_count[concept] = self.interaction_count.get(concept, 0) + 1

        is_correct = prediction_error < ACCURACY_THRESHOLD
        self.correct_history[concept].append(1.0 if is_correct else 0.0)

        if prediction_error < self.best_error.get(concept, float("inf")):
            self.best_error[concept] = prediction_error

    def record_episode_reward(self, episode_reward):
        """Track how well the brain performs (reward = action competence)."""
        self.episode_rewards.append(episode_reward)
        if len(self.episode_rewards) >= 3:
            avg = np.mean(list(self.episode_rewards)[-10:])
            self.reward_competence = float(np.clip(avg / 10.0, 0.0, 1.0))

    def competence(self, concept):
        """Blend of prediction accuracy (can I predict?) and reward (can I act?).
        50% prediction + 50% reward. Not competent until BOTH are good."""
        correct = self.correct_history.get(concept, [])
        if len(correct) < 5:
            return 0.0
        recent = list(correct)[-min(50, len(correct)):]
        pred_comp = float(np.clip(np.mean(recent), 0.0, 1.0))
        return float(np.clip(0.5 * pred_comp + 0.5 * self.reward_competence, 0.0, 1.0))

    def learning_progress(self, concept):
        """How fast accuracy is improving."""
        correct = list(self.correct_history.get(concept, []))
        if len(correct) < 20:
            return 0.0
        mid = len(correct) // 2
        old_acc = np.mean(correct[:mid])
        new_acc = np.mean(correct[mid:])
        return float(new_acc - old_acc)

    def get_focus(self):
        lps = {c: self.learning_progress(c) for c in self.concepts}
        max_lp = max(lps.values())
        if max_lp < 1e-6:
            return None
        return max(lps, key=lps.get)

    def get_report(self):
        return {c: {
            "competence": round(self.competence(c), 3),
            "accuracy": round(self.competence(c), 3),
            "learning_progress": round(self.learning_progress(c), 6),
            "interactions": self.interaction_count.get(c, 0),
            "best_error": round(self.best_error.get(c, float("inf")), 6),
        } for c in self.concepts}

    def exploration_weights(self, temperature=2.0):
        lps = np.array([self.learning_progress(c) for c in self.concepts])
        lps = lps + 0.05
        weights = np.exp(lps * temperature)
        weights /= weights.sum()
        return {c: float(w) for c, w in zip(self.concepts, weights)}
