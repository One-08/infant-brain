"""
Self-Correction — detects changes, updates beliefs, fills knowledge gaps.
"""

import numpy as np
import torch
from collections import deque


class ChangeDetector:
    def __init__(self, concepts, window=50, spike_threshold=2.0):
        self.concepts = list(concepts)
        self.window = window
        self.spike_threshold = spike_threshold
        self.error_history = {c: deque(maxlen=window * 2) for c in concepts}
        self.detected_changes = []
        self.change_count = {c: 0 for c in concepts}
        self.baseline_error = {}
        self.obs_count = {c: 0 for c in concepts}
        self.last_correction_at = {c: -999 for c in concepts}

    def record(self, concept, error):
        if concept not in self.error_history:
            self.error_history[concept] = deque(maxlen=self.window * 2)
            self.change_count[concept] = 0
            self.obs_count[concept] = 0
            self.last_correction_at[concept] = -999
        self.error_history[concept].append(error)
        self.obs_count[concept] = self.obs_count.get(concept, 0) + 1
        if concept not in self.baseline_error:
            self.baseline_error[concept] = error

    def check(self, concept):
        errs = list(self.error_history.get(concept, []))
        if len(errs) < 100:
            return False

        obs = self.obs_count.get(concept, 0)
        last = self.last_correction_at.get(concept, -999)
        if obs - last < 3000:
            return False

        old = np.mean(errs[:len(errs) // 2])
        new = np.mean(errs[len(errs) // 2:])

        # Spike detection: error suddenly gets worse
        if old > 1e-8:
            ratio = new / old
            if ratio > self.spike_threshold:
                self._record_change(concept, "spike", ratio)
                return True

        # Plateau detection: error high and not improving
        if len(errs) >= self.window:
            recent = np.mean(errs[-self.window // 2:])
            older = np.mean(errs[-self.window:-self.window // 2])
            if older > 1e-8 and abs(recent - older) / older < 0.02 and recent > 0.1:
                self._record_change(concept, "plateau", recent / max(older, 1e-8))
                return True

        # Inconsistency: high variance in recent errors
        if len(errs) >= 20:
            recent_std = np.std(errs[-20:])
            recent_mean = np.mean(errs[-20:])
            if recent_mean > 1e-6 and recent_std / recent_mean > 3.0:
                self._record_change(concept, "inconsistent", recent_std / recent_mean)
                return True

        return False

    def _record_change(self, concept, change_type, value):
        self.change_count[concept] = self.change_count.get(concept, 0) + 1
        self.last_correction_at[concept] = self.obs_count.get(concept, 0)
        self.detected_changes.append({
            "concept": concept, "type": change_type, "value": round(value, 3)
        })

    def check_all(self):
        return [c for c in self.concepts if self.check(c)]


class BeliefUpdater:
    def __init__(self):
        self.corrections = []

    def correct(self, concept, monitor, semantic_mem, knowledge_tree):
        """Soft correction: decay competence instead of wiping it.
        Keeps recent history so the brain doesn't forget everything."""
        actions = []
        if concept in monitor.error_history:
            hist = monitor.error_history[concept]
            keep = max(10, len(hist) // 2)
            recent = list(hist)[-keep:]
            hist.clear()
            for v in recent:
                hist.append(v)
            if concept in monitor.correct_history:
                ch = monitor.correct_history[concept]
                recent_c = list(ch)[-keep:]
                ch.clear()
                for v in recent_c:
                    ch.append(v)
            monitor.interaction_count[concept] = max(
                keep, monitor.interaction_count.get(concept, 0) // 2)
            actions.append("decay_competence")
        if concept in semantic_mem.prototypes:
            semantic_mem.confidence[concept] = max(
                0, semantic_mem.confidence.get(concept, 0) // 2)
            actions.append("halve_prototype_confidence")
        if concept in knowledge_tree.nodes:
            node = knowledge_tree.nodes[concept]
            keep = max(10, len(node.error_history) // 2)
            recent = list(node.error_history)[-keep:]
            node.error_history.clear()
            for v in recent:
                node.error_history.append(v)
            node.competence *= 0.5
            node.learning_progress *= 0.5
            actions.append("decay_tree_node")
        correction = {"concept": concept, "actions": actions}
        self.corrections.append(correction)
        return correction


class GapFiller:
    def __init__(self, knowledge_tree):
        self.tree = knowledge_tree
        self.fill_history = deque(maxlen=200)

    def get_target_concept(self):
        gaps = self.tree.get_gaps()
        if gaps:
            return gaps[0]
        frontiers = self.tree.get_frontiers()
        if frontiers:
            return frontiers[0]
        return None

    def record_fill_attempt(self, concept, error):
        self.fill_history.append({"concept": concept, "error": error})


class SelfCorrectingBrain:
    def __init__(self, concepts, monitor, semantic_mem, knowledge_tree):
        self.detector = ChangeDetector(concepts)
        self.updater = BeliefUpdater()
        self.filler = GapFiller(knowledge_tree)
        self.monitor = monitor
        self.semantic_mem = semantic_mem
        self.knowledge_tree = knowledge_tree

    def observe(self, concept, error, z=None):
        if concept is None:
            return None
        self.detector.record(concept, error)
        self.knowledge_tree.record(concept, error, z)
        if self.detector.check(concept):
            return self.updater.correct(concept, self.monitor, self.semantic_mem, self.knowledge_tree)
        self.filler.record_fill_attempt(concept, error)
        return None

    def periodic_check(self):
        changed = self.detector.check_all()
        for c in changed:
            self.updater.correct(c, self.monitor, self.semantic_mem, self.knowledge_tree)
        leaf_concepts = [n for n, nd in self.knowledge_tree.nodes.items()
                         if n != "root" and not nd.children]
        splits = []
        for c in leaf_concepts:
            new = self.knowledge_tree.try_split(c)
            if new:
                splits.append((c, new))
        return {"changes": changed, "splits": splits}
