"""
Memory System — Episodic Memory, Semantic Memory, and Sleep Cycle.
"""

import numpy as np
import torch
import torch.nn.functional as F


class EpisodicMemory:
    def __init__(self, capacity=10_000):
        self.capacity = capacity
        self.episodes = []

    def store(self, obs, action, next_obs, word_idx, concept, surprise, episode_num,
              z=None, reward=0.0, quality="neutral"):
        emotion = abs(reward) * 3.0 + surprise
        keep_frames = (reward != 0 or surprise > 0.5)
        entry = {"action": action, "word_idx": word_idx, "concept": concept,
                 "surprise": surprise, "episode_num": episode_num,
                 "reward": reward, "quality": quality, "emotion": emotion}
        if keep_frames:
            entry["obs"] = obs
            entry["next_obs"] = next_obs
        if z is not None:
            entry["z"] = z
        self.episodes.append(entry)
        if len(self.episodes) > self.capacity:
            self._prune_least_surprising()

    def _prune_least_surprising(self):
        emotional = [e for e in self.episodes if e.get("reward", 0) != 0]
        rest = [e for e in self.episodes if e.get("reward", 0) == 0]
        rest.sort(key=lambda e: e.get("emotion", e["surprise"]), reverse=True)
        keep_rest = self.capacity - len(emotional)
        self.episodes = emotional + rest[:max(0, keep_rest)]

    def sample_surprising(self, n, temperature=2.0):
        """Sample memories weighted by emotional intensity.
        Emotionally charged memories (scoring, dying, surprising events)
        are replayed more often — just like humans remember intense moments."""
        if len(self.episodes) == 0:
            return []
        emotions = np.array([e.get("emotion", e["surprise"]) for e in self.episodes], dtype=np.float64)
        emotions = np.clip(emotions, -20, 20)
        log_w = emotions * temperature
        log_w -= log_w.max()
        weights = np.exp(log_w)
        weights = np.maximum(weights, 1e-10)
        weights /= weights.sum()
        n = min(n, len(self.episodes))
        idxs = np.random.choice(len(self.episodes), size=n, replace=False, p=weights)
        return [self.episodes[i] for i in idxs]

    def recall_similar(self, z, top_k=3):
        """Find memories with the most similar latent vector."""
        with_z = [e for e in self.episodes if "z" in e]
        if not with_z:
            return []
        stored_z = torch.stack([e["z"] for e in with_z])
        dists = (stored_z - z).pow(2).sum(dim=-1)
        k = min(top_k, len(with_z))
        top_idx = dists.argsort()[:k]
        return [with_z[i] for i in top_idx]

    def recall_best_action(self, z, num_actions, top_k=10):
        """What action worked best in states similar to this one?
        Returns a score per action based on past experience.
        Caps search to 2000 most recent memories for speed."""
        with_z = [e for e in self.episodes if "z" in e]
        if len(with_z) < 20:
            return None
        if len(with_z) > 2000:
            with_z = with_z[-2000:]
        stored_z = torch.stack([e["z"] for e in with_z])
        dists = (stored_z - z).pow(2).sum(dim=-1)
        k = min(top_k, len(with_z))
        top_idx = dists.argsort()[:k]

        action_value = np.zeros(num_actions)
        action_count = np.zeros(num_actions)
        for i in top_idx:
            mem = with_z[i]
            a = mem["action"]
            r = mem.get("reward", 0.0)
            dist_weight = 1.0 / (1.0 + dists[i].item())
            action_value[a] += r * dist_weight
            action_count[a] += dist_weight

        mask = action_count > 0
        if not mask.any():
            return None
        action_score = np.where(mask, action_value / np.maximum(action_count, 1e-8), 0.0)
        return action_score

    def get_concept_episodes(self, concept, max_n=500):
        matching = [e for e in self.episodes if e["concept"] == concept]
        if len(matching) > max_n:
            matching.sort(key=lambda e: e["surprise"], reverse=True)
            matching = matching[:max_n]
        return matching

    def prune(self, keep_fraction=0.7):
        if len(self.episodes) == 0:
            return 0
        emotional = [e for e in self.episodes if e.get("reward", 0) != 0]
        rest = [e for e in self.episodes if e.get("reward", 0) == 0]
        n_keep = max(1, int(len(self.episodes) * keep_fraction))
        rest.sort(key=lambda e: e.get("emotion", e["surprise"]), reverse=True)
        keep_rest = max(0, n_keep - len(emotional))
        new_eps = emotional + rest[:keep_rest]
        n_pruned = len(self.episodes) - len(new_eps)
        self.episodes = new_eps
        return n_pruned

    def __len__(self):
        return len(self.episodes)


class SemanticMemory:
    def __init__(self, concepts, latent_dim=32):
        self.concepts = list(concepts)
        self.latent_dim = latent_dim
        self.prototypes = {}
        self.confidence = {}
        self.update_count = {c: 0 for c in concepts}

    def update_prototype(self, concept, z_vectors):
        if len(z_vectors) == 0:
            return
        new_centroid = torch.mean(z_vectors, dim=0)
        if concept in self.prototypes:
            momentum = min(0.9, 0.5 + self.update_count.get(concept, 0) * 0.05)
            self.prototypes[concept] = momentum * self.prototypes[concept] + (1 - momentum) * new_centroid
        else:
            self.prototypes[concept] = new_centroid
        self.confidence[concept] = len(z_vectors)
        self.update_count[concept] = self.update_count.get(concept, 0) + 1

    def query(self, concept):
        if concept not in self.prototypes:
            return None, 0
        return self.prototypes[concept], self.confidence.get(concept, 0)


class SleepCycle:
    def __init__(self, replay_batch=128, replay_steps=10, prune_fraction=0.7):
        self.replay_batch = replay_batch
        self.replay_steps = replay_steps
        self.prune_fraction = prune_fraction
        self.cycle_count = 0

    def run(self, world_model, lang_module, episodic_mem, semantic_mem,
            optimizer, device, temporal_weight=0.05, lang_weight=0.1,
            lang_optimizer=None):
        """Run a sleep cycle.

        Args:
            optimizer:      optimizer for `world_model` parameters.
            lang_optimizer: optional optimizer for `lang_module` parameters.
                If omitted, language gradients are dropped on the floor — pass
                this in once you've split optimizers per module.
        """
        self.cycle_count += 1
        metrics = {"cycle": self.cycle_count}

        replay_loss = self._replay_phase(world_model, lang_module, episodic_mem,
                                         optimizer, device, temporal_weight, lang_weight,
                                         lang_optimizer=lang_optimizer)
        metrics["replay_loss"] = replay_loss

        n_updated = self._consolidate_phase(world_model, episodic_mem, semantic_mem, device)
        metrics["prototypes_updated"] = n_updated

        n_pruned = episodic_mem.prune(keep_fraction=self.prune_fraction)
        metrics["memories_pruned"] = n_pruned
        return metrics

    def run_replay_only(self, world_model, episodic_mem, semantic_mem, device):
        """Sleep when WM is frozen: consolidate memories + prune, no WM training."""
        self.cycle_count += 1
        n_updated = self._consolidate_phase(world_model, episodic_mem, semantic_mem, device)
        n_pruned = episodic_mem.prune(keep_fraction=self.prune_fraction)
        return {"cycle": self.cycle_count, "replay_loss": 0.0,
                "prototypes_updated": n_updated, "memories_pruned": n_pruned}

    def _replay_phase(self, world_model, lang_module, episodic_mem,
                      optimizer, device, temporal_weight, lang_weight,
                      lang_optimizer=None):
        if len(episodic_mem) < self.replay_batch:
            return 0.0
        total_loss = 0.0
        for _ in range(self.replay_steps):
            batch = episodic_mem.sample_surprising(self.replay_batch)
            batch = [e for e in batch if "obs" in e and "next_obs" in e]
            if len(batch) < 8:
                continue
            obs = torch.stack([e["obs"] for e in batch]).to(device)
            acts = torch.tensor([e["action"] for e in batch], dtype=torch.long, device=device)
            nobs = torch.stack([e["next_obs"] for e in batch]).to(device)
            words = torch.tensor([e["word_idx"] for e in batch], dtype=torch.long, device=device)

            world_model.train()
            lang_module.train()
            optimizer.zero_grad(set_to_none=True)
            if lang_optimizer is not None:
                lang_optimizer.zero_grad(set_to_none=True)
            wm_loss, _ = world_model.compute_loss(obs, acts, nobs, temporal_weight=temporal_weight)

            word_mask = words >= 0
            lang_loss = torch.tensor(0.0, device=device)
            if word_mask.sum() > 1:
                with torch.no_grad():
                    z_lang = world_model.encode(obs[word_mask], use_memory=False)
                lang_loss, _ = lang_module.compute_loss(z_lang.detach(), words[word_mask])

            loss = wm_loss + lang_weight * lang_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), 1.0)
            if lang_optimizer is not None:
                torch.nn.utils.clip_grad_norm_(lang_module.parameters(), 1.0)
            optimizer.step()
            if lang_optimizer is not None:
                lang_optimizer.step()
            total_loss += loss.item()
        return total_loss / self.replay_steps

    def _consolidate_phase(self, world_model, episodic_mem, semantic_mem, device):
        world_model.eval()
        n_updated = 0
        for concept in semantic_mem.concepts:
            episodes = episodic_mem.get_concept_episodes(concept)
            if len(episodes) < 5:
                continue
            z_vectors = []
            with torch.no_grad():
                for ep in episodes:
                    if "z" in ep:
                        z_vectors.append(ep["z"])
                    elif "obs" in ep:
                        obs = ep["obs"].unsqueeze(0).to(device)
                        z = world_model.encode(obs)
                        z_vectors.append(z.squeeze(0).cpu())
            if z_vectors:
                semantic_mem.update_prototype(concept, torch.stack(z_vectors))
                n_updated += 1
        return n_updated
