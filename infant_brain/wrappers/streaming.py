"""Webcam / video-file streaming as a BrainEnv (RGB frames → brain-sized tensor).

Install: ``pip install opencv-python-headless`` (or add optional extra ``[streaming]``).

Frame size matches ``WorldModel`` expectations: 32×32 on CPU, 64×64 on CUDA.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from infant_brain.envs.base import BrainEnv


def brain_obs_size() -> int:
    """Match `WorldModel` / support-ticket wrapper: 32 CPU, 64 CUDA."""
    return 64 if torch.cuda.is_available() else 32


def bgr_to_chw_float(bgr: np.ndarray, size: int) -> torch.Tensor:
    """Resize BGR uint8 frame to (3, size, size) float [0, 1]."""
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "Streaming requires OpenCV. Install: pip install opencv-python-headless"
        ) from e
    h, w = bgr.shape[:2]
    if h != size or w != size:
        bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    return t


class OpenCVSource:
    """Thin wrapper around cv2.VideoCapture (camera index or video file)."""

    def __init__(self, *, camera: Optional[int] = None, video_path: Optional[str] = None):
        try:
            import cv2
        except ImportError as e:
            raise ImportError(
                "Streaming requires OpenCV. Install: pip install opencv-python-headless"
            ) from e
        self._cv2 = cv2
        if video_path is not None:
            self.cap = cv2.VideoCapture(video_path)
            self._is_file = True
        else:
            idx = 0 if camera is None else int(camera)
            self.cap = cv2.VideoCapture(idx)
            self._is_file = False
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open video source "
                f"({'file: ' + video_path if video_path else 'camera ' + str(camera)})"
            )

    def rewind(self):
        """Seek to start (video files only)."""
        if self._is_file:
            self.cap.set(self._cv2.CAP_PROP_POS_FRAMES, 0)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ok, frame = self.cap.read()
        return ok, frame if ok else None

    def release(self):
        self.cap.release()


class StreamVisualEnv(BrainEnv):
    """One frame per step from a live camera or video file.

    Observation: RGB tensor (3, H, H) with H = ``brain_obs_size()``.

    Default reward is a small motion signal (mean |Δ| across pixels) so short
    online training gets a non-zero signal. Set ``motion_reward_scale=0`` to
    disable.

    Actions are abstract labels for your prototype (e.g. drive overlays, modes).
    Map them in your app; this env does not move the camera.
    """

    VOCAB = [
        "still",
        "motion",
        "bright",
        "dark",
        "up",
        "down",
        "left",
        "right",
        "focus",
        "scene",
    ]

    def __init__(
        self,
        source: OpenCVSource,
        num_actions: int = 4,
        motion_reward_scale: float = 10.0,
    ):
        self._src = source
        self._motion_scale = motion_reward_scale
        self._n_act = num_actions
        self._size = brain_obs_size()
        self._prev: Optional[torch.Tensor] = None

    @property
    def num_actions(self) -> int:
        return self._n_act

    @property
    def vocab_words(self) -> list:
        return self.VOCAB

    @property
    def frame_channels(self) -> int:
        return 3

    def reset(self, seed=None) -> torch.Tensor:
        self._src.rewind()
        ok, bgr = self._src.read()
        if not ok or bgr is None:
            raise RuntimeError("StreamVisualEnv.reset: no frame (check camera / file path)")
        frame = bgr_to_chw_float(bgr, self._size)
        self._prev = frame.clone()
        return frame

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        ok, bgr = self._src.read()
        if not ok or bgr is None:
            z = torch.zeros(3, self._size, self._size)
            return z, 0.0, True
        frame = bgr_to_chw_float(bgr, self._size)
        reward = 0.0
        if self._prev is not None and self._motion_scale != 0.0:
            reward = float((frame - self._prev).abs().mean().item() * self._motion_scale)
        self._prev = frame.clone()
        return frame, reward, False

    def close(self):
        self._src.release()
