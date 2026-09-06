"""
Knowledge Tree — self-organizing concept hierarchy.
Splits concepts when latent variance is high, detects knowledge gaps.
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import deque


class KnowledgeNode:
    def __init__(self, name, parent=None, depth=0):
        self.name = name
        self.parent = parent
        self.children = []
        self.depth = depth
        self.error_history = deque(maxlen=200)
        self.interaction_count = 0
        self.z_samples = deque(maxlen=100)
        self.competence = 0.0
        self.learning_progress = 0.0

    def record(self, prediction_error, z=None):
        self.error_history.append(prediction_error)
        self.interaction_count += 1
        if z is not None:
            self.z_samples.append(z.detach().cpu())
        self._update_metrics()

    def _update_metrics(self):
        if len(self.error_history) < 5:
            return
        recent = list(self.error_history)[-min(50, len(self.error_history)):]
        self.competence = float(np.clip(1.0 - np.mean(recent) * 50, 0.0, 1.0))
        if len(self.error_history) >= 20:
            errs = list(self.error_history)
            mid = len(errs) // 2
            self.learning_progress = float(abs(np.mean(errs[:mid]) - np.mean(errs[mid:])))

    def should_split(self, min_samples=30, variance_threshold=0.3):
        if len(self.z_samples) < min_samples or self.children:
            return False
        z_stack = torch.stack(list(self.z_samples))
        return z_stack.var(dim=0).mean().item() > variance_threshold

    def is_gap(self):
        return self.competence < 0.5 and self.learning_progress > 0.001

    def is_frontier(self):
        return 0.3 < self.competence < 0.8 and self.learning_progress > 0.005


class KnowledgeTree:
    def __init__(self, initial_concepts):
        self.root = KnowledgeNode("root")
        self.nodes = {"root": self.root}
        for concept in initial_concepts:
            node = KnowledgeNode(concept, parent=self.root, depth=1)
            self.root.children.append(node)
            self.nodes[concept] = node

    def record(self, concept, prediction_error, z=None):
        if concept not in self.nodes:
            node = KnowledgeNode(concept, parent=self.root, depth=1)
            self.root.children.append(node)
            self.nodes[concept] = node
        self.nodes[concept].record(prediction_error, z)

    def try_split(self, concept, latent_dim=32):
        node = self.nodes.get(concept)
        if node is None or not node.should_split():
            return []
        z_stack = torch.stack(list(node.z_samples))
        n = len(z_stack)
        idx = torch.randperm(n)
        c1, c2 = z_stack[idx[0]], z_stack[idx[1]]
        for _ in range(10):
            d1 = (z_stack - c1.unsqueeze(0)).pow(2).sum(dim=1)
            d2 = (z_stack - c2.unsqueeze(0)).pow(2).sum(dim=1)
            mask1, mask2 = d1 <= d2, d1 > d2
            if mask1.sum() < 3 or mask2.sum() < 3:
                return []
            c1, c2 = z_stack[mask1].mean(dim=0), z_stack[mask2].mean(dim=0)
        if F.cosine_similarity(c1.unsqueeze(0), c2.unsqueeze(0)).item() > 0.9:
            return []
        child_names = []
        for i, (mask, _) in enumerate([(mask1, c1), (mask2, c2)]):
            name = f"{concept}_sub{i}"
            child = KnowledgeNode(name, parent=node, depth=node.depth + 1)
            for z in z_stack[mask]:
                child.z_samples.append(z)
            child.interaction_count = mask.sum().item()
            node.children.append(child)
            self.nodes[name] = child
            child_names.append(name)
        return child_names

    def get_gaps(self):
        return [n for n, nd in self.nodes.items() if n != "root" and nd.is_gap()]

    def get_frontiers(self):
        return [n for n, nd in self.nodes.items() if n != "root" and nd.is_frontier()]

    def get_report(self):
        return {
            "total_concepts": len(self.nodes) - 1,
            "gaps": self.get_gaps(),
            "frontiers": self.get_frontiers(),
        }
