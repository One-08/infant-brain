"""Support-ticket wrapper for a first shippable non-game workflow.

This module provides:
1) A BrainEnv-compatible ticket triage environment for training.
2) A safe action executor for production-like dry-run/guarded execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from infant_brain.envs.base import BrainEnv


ACTION_IGNORE = 0
ACTION_ASSIGN_BILLING = 1
ACTION_ASSIGN_TECH = 2
ACTION_ESCALATE = 3

ACTION_NAMES = {
    ACTION_IGNORE: "ignore",
    ACTION_ASSIGN_BILLING: "assign_billing",
    ACTION_ASSIGN_TECH: "assign_tech",
    ACTION_ESCALATE: "escalate",
}


@dataclass
class Ticket:
    id: str
    text: str
    category: str
    priority: str
    customer_tier: str = "standard"


def _brain_obs_size() -> int:
    """Match `WorldModel` image size: 32 on CPU, 64 on CUDA.

    The encoder patch grid must match `pos_embed`; using the wrong H×W
    raises a patch-count mismatch (e.g. 256 vs 64 tokens on CPU).
    """
    return 64 if torch.cuda.is_available() else 32


def _default_tickets() -> List[Ticket]:
    return [
        Ticket("T-001", "Payment deducted twice from card", "billing", "high", "enterprise"),
        Ticket("T-002", "Login page spins forever after update", "technical", "high", "enterprise"),
        Ticket("T-003", "Need invoice PDF for last month", "billing", "medium"),
        Ticket("T-004", "Feature request: dark mode", "general", "low"),
        Ticket("T-005", "Service unavailable in production", "technical", "urgent", "enterprise"),
        Ticket("T-006", "Refund request for wrong plan", "billing", "medium"),
        Ticket("T-007", "How to change profile photo?", "general", "low"),
        Ticket("T-008", "Data export endpoint returns 500", "technical", "high"),
    ]


class SupportTicketEnv(BrainEnv):
    """Simple RL environment for support-ticket triage decisions.

    State is a 3×H×H tensor (H=32 on CPU, 64 on CUDA) matching `WorldModel`.
    Each step handles exactly one ticket; an episode runs through all tickets.
    """

    VOCAB = [
        "ticket",
        "billing",
        "technical",
        "general",
        "urgent",
        "high",
        "medium",
        "low",
        "enterprise",
        "standard",
        "escalate",
        "assign",
        "ignore",
    ]

    def __init__(self, tickets: Optional[List[Ticket]] = None):
        self._base_tickets = list(tickets) if tickets is not None else _default_tickets()
        self._tickets: List[Ticket] = []
        self._idx = 0
        self._done = False
        self._rng = np.random.RandomState(42)

    def reset(self, seed=None) -> torch.Tensor:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._tickets = list(self._base_tickets)
        self._rng.shuffle(self._tickets)
        self._idx = 0
        self._done = False
        return self._encode_ticket(self._tickets[self._idx])

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        h = _brain_obs_size()
        if self._done:
            return torch.zeros(3, h, h), 0.0, True

        ticket = self._tickets[self._idx]
        reward = self._score_action(ticket, int(action))

        self._idx += 1
        if self._idx >= len(self._tickets):
            self._done = True
            return torch.zeros(3, h, h), reward, True
        return self._encode_ticket(self._tickets[self._idx]), reward, False

    @property
    def num_actions(self) -> int:
        return 4

    @property
    def vocab_words(self) -> list:
        return self.VOCAB

    @property
    def current_ticket(self) -> Optional[Ticket]:
        if self._done or not self._tickets or self._idx >= len(self._tickets):
            return None
        return self._tickets[self._idx]

    def close(self):
        pass

    def _score_action(self, ticket: Ticket, action: int) -> float:
        if ticket.priority == "urgent":
            return 1.0 if action == ACTION_ESCALATE else -1.0
        if ticket.category == "billing":
            if action == ACTION_ASSIGN_BILLING:
                return 0.8
            if action == ACTION_ESCALATE and ticket.customer_tier == "enterprise":
                return 0.6
            return -0.4
        if ticket.category == "technical":
            if action == ACTION_ASSIGN_TECH:
                return 0.8
            if action == ACTION_ESCALATE and ticket.priority in {"high", "urgent"}:
                return 0.7
            return -0.4
        if ticket.category == "general":
            return 0.3 if action == ACTION_IGNORE else -0.2
        return -0.1

    @staticmethod
    def _encode_ticket(ticket: Ticket) -> torch.Tensor:
        h = _brain_obs_size()
        arr = np.zeros((3, h, h), dtype=np.float32)
        cat_map = {"billing": 0.2, "technical": 0.6, "general": 0.9}
        pri_map = {"low": 0.2, "medium": 0.5, "high": 0.75, "urgent": 1.0}
        tier_map = {"standard": 0.35, "enterprise": 0.9}

        c = cat_map.get(ticket.category, 0.1)
        p = pri_map.get(ticket.priority, 0.1)
        t = tier_map.get(ticket.customer_tier, 0.35)

        arr[0, :, :] = c
        arr[1, :, :] = p
        arr[2, :, :] = t
        return torch.from_numpy(arr)


class SafeTicketExecutor:
    """Executes ticket actions with explicit safety constraints.

    Default behavior is dry-run to avoid accidental side effects.
    """

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run

    def execute(self, ticket: Ticket, action: int) -> Dict[str, str]:
        action_name = ACTION_NAMES.get(int(action), "unknown")

        # Safety rail: only allow known actions.
        if action_name == "unknown":
            return {"ticket_id": ticket.id, "status": "blocked", "reason": "unknown_action"}

        # Safety rail: auto-escalate only for urgent or enterprise high-priority.
        if action == ACTION_ESCALATE and not (
            ticket.priority == "urgent" or (ticket.priority == "high" and ticket.customer_tier == "enterprise")
        ):
            return {
                "ticket_id": ticket.id,
                "status": "blocked",
                "reason": "escalation_not_allowed_without_review",
            }

        if self.dry_run:
            return {
                "ticket_id": ticket.id,
                "status": "dry_run",
                "action": action_name,
                "note": "no external API call executed",
            }

        # In production this is where API calls would go.
        return {"ticket_id": ticket.id, "status": "executed", "action": action_name}
