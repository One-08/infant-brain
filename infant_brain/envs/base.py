"""Abstract base environment interface for the Infant Brain."""

from abc import ABC, abstractmethod
import torch


class BrainEnv(ABC):
    @abstractmethod
    def reset(self, seed=None) -> torch.Tensor:
        """Reset and return initial frame as (3, 64, 64) float tensor in [0, 1]."""
        ...

    @abstractmethod
    def step(self, action: int) -> tuple:
        """Execute action. Returns (frame, reward, done)."""
        ...

    @property
    @abstractmethod
    def num_actions(self) -> int: ...

    @property
    @abstractmethod
    def vocab_words(self) -> list: ...

    @abstractmethod
    def close(self): ...
