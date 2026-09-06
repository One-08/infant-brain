"""
Infant Brain — A self-learning AI that grows like a child.

7 modules working together:
  1. World Model    — learns physics from pixels (JEPA + SIGReg)
  2. Curiosity      — explores what's surprising
  3. Metacognition  — knows what it knows and doesn't know
  4. Language        — grounds words in visual experience
  5. Values         — internal feedback signals guide learning
  6. Memory + Sleep — stores experiences, consolidates during sleep
  7. Reasoning      — plans with imagined rollouts and actor-critic learning
"""

from .brain import Brain
from .envs import AtariEnv, GridWorldEnv

__version__ = "0.1.0"
__all__ = ["Brain", "AtariEnv", "GridWorldEnv", "__version__"]
