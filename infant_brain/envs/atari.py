"""Atari environment wrapper — any ALE game via Gymnasium."""

import numpy as np
import torch
from collections import deque
from PIL import Image
from .base import BrainEnv


class AtariEnv(BrainEnv):
    VOCAB = ["score", "miss", "hit", "idle", "up", "down", "left", "right",
             "fast", "slow", "new", "same", "dodge", "explore", "wait"]

    def __init__(self, game="Pong", frameskip=4, frame_stack=3, img_size=None):
        try:
            import gymnasium as gym
            import ale_py  # noqa: F401
        except ImportError:
            raise ImportError("Atari requires: pip install gymnasium[atari] ale-py autorom")
        self._game = game
        self._env = gym.make(f"ALE/{game}-v5", render_mode="rgb_array", frameskip=frameskip)
        self._num_actions = self._env.action_space.n
        self._frame_stack = frame_stack
        self._frame_buffer = deque(maxlen=frame_stack)
        # Image size autoselect:
        #   - explicit arg always wins
        #   - else env var INFANT_BRAIN_IMG_SIZE (so tests/CI can pin it)
        #   - else 64 only when CUDA is available (MPS is intentionally
        #     excluded: its availability flips between sandboxed/non-
        #     sandboxed processes on macOS, which silently changes the
        #     architecture and breaks multi-seed comparability)
        #   - else 32 (the value used for all reported paper experiments)
        import os as _os
        if img_size is not None:
            self._img_size = int(img_size)
        elif _os.environ.get("INFANT_BRAIN_IMG_SIZE"):
            self._img_size = int(_os.environ["INFANT_BRAIN_IMG_SIZE"])
        elif torch.cuda.is_available():
            self._img_size = 64
        else:
            self._img_size = 32

    def reset(self, seed=None):
        obs, _ = self._env.reset(seed=seed)
        frame = self._preprocess(obs, self._img_size)
        for _ in range(self._frame_stack):
            self._frame_buffer.append(frame)
        return self._stack()

    def step(self, action):
        obs, reward, done, truncated, info = self._env.step(action)
        frame = self._preprocess(obs, self._img_size)
        self._frame_buffer.append(frame)
        return self._stack(), float(reward), done or truncated

    def _stack(self):
        return torch.cat(list(self._frame_buffer), dim=0)

    @property
    def num_actions(self): return self._num_actions

    @property
    def frame_channels(self):
        return 3 * self._frame_stack

    @property
    def vocab_words(self): return self.VOCAB

    def close(self): self._env.close()

    @staticmethod
    def _preprocess(frame, size=64):
        img = Image.fromarray(frame).resize((size, size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)
