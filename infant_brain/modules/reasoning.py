"""
Reasoning Engine — action planning with GRU + step annotation + confidence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_GOAL_TYPES = 4


class ActionPlanner(nn.Module):
    def __init__(self, latent_dim=32, goal_dim=16, hidden_dim=64,
                 num_actions=4, num_goal_types=NUM_GOAL_TYPES):
        super().__init__()
        self.goal_embed = nn.Embedding(num_goal_types, goal_dim)
        self.gru = nn.GRUCell(latent_dim + goal_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

    def forward(self, z_seq, goal_idx):
        B, T, D = z_seq.shape
        g = self.goal_embed(goal_idx)
        h = torch.zeros(B, self.hidden_dim, device=z_seq.device)
        all_logits = []
        for t in range(T):
            inp = torch.cat([z_seq[:, t], g], dim=-1)
            h = self.gru(inp, h)
            all_logits.append(self.action_head(h))
        return torch.stack(all_logits, dim=1)

    def plan(self, z0, goal_idx, world_model, plan_len=3, temperature=1.0):
        device = z0.device
        z = z0.unsqueeze(0)
        g = self.goal_embed(goal_idx.unsqueeze(0)).squeeze(0)
        h = torch.zeros(self.hidden_dim, device=device)
        actions, z_traj, logits_list = [], [z.squeeze(0)], []
        for _ in range(plan_len):
            inp = torch.cat([z.squeeze(0), g], dim=-1)
            h = self.gru(inp, h)
            logits = self.action_head(h)
            logits_list.append(logits)
            probs = F.softmax(logits / temperature, dim=-1)
            action = torch.multinomial(probs, 1).item()
            actions.append(action)
            action_t = torch.tensor([action], device=device)
            with torch.no_grad():
                z, _ = world_model.predict_next(z, action_t)
                z = z.detach()
            z_traj.append(z.squeeze(0))
        return torch.tensor(actions, device=device), torch.stack(z_traj), torch.stack(logits_list)


class StepAnnotator(nn.Module):
    def __init__(self, latent_dim=32, num_actions=4, vocab_size=13, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2 + num_actions, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
        )
        self.num_actions = num_actions

    def forward(self, z_t, action, z_next):
        a_oh = F.one_hot(action, self.num_actions).float()
        return self.net(torch.cat([z_t, a_oh, z_next], dim=-1))


class ConfidenceHead(nn.Module):
    def __init__(self, latent_dim=32, plan_len=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * (plan_len + 1), 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, z_trajectory):
        B = z_trajectory.shape[0]
        return self.net(z_trajectory.reshape(B, -1)).squeeze(-1)


class Plan:
    def __init__(self, actions, z_trajectory, action_logits, annotations, confidence):
        self.actions = actions
        self.z_trajectory = z_trajectory
        self.action_logits = action_logits
        self.annotations = annotations
        self.confidence = confidence


class ReasoningEngine(nn.Module):
    def __init__(self, latent_dim=32, num_actions=4, vocab_size=13,
                 plan_len=3, goal_dim=16):
        super().__init__()
        self.planner = ActionPlanner(latent_dim, goal_dim, 64, num_actions)
        self.annotator = StepAnnotator(latent_dim, num_actions, vocab_size, 64)
        self.confidence_head = ConfidenceHead(latent_dim, plan_len)
        self.plan_len = plan_len
        self.num_actions = num_actions

    def reason(self, z0, goal_idx, world_model, temperature=1.0):
        actions, z_traj, action_logits = self.planner.plan(
            z0, goal_idx, world_model, self.plan_len, temperature)
        annotations = []
        for t in range(self.plan_len):
            logits = self.annotator(z_traj[t:t+1], actions[t:t+1], z_traj[t+1:t+2])
            annotations.append(logits.argmax(dim=-1).item())
        conf = self.confidence_head(z_traj.unsqueeze(0))
        return Plan(actions, z_traj, action_logits, annotations, conf.item())

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
