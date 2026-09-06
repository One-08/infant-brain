"""GridWorld environment — 2D grid with colored shapes, agent pushes objects."""

import numpy as np
import torch
from .base import BrainEnv


class GridWorld:
    COLORS = {
        "red": np.array([220, 50, 50], dtype=np.uint8),
        "blue": np.array([50, 80, 220], dtype=np.uint8),
        "green": np.array([50, 180, 50], dtype=np.uint8),
        "yellow": np.array([220, 200, 50], dtype=np.uint8),
        "purple": np.array([160, 50, 220], dtype=np.uint8),
    }
    SHAPES = ["square", "circle"]
    ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    NUM_ACTIONS = 4

    def __init__(self, grid_size=8, num_objects=5, cell_pixels=4, seed=None):
        self.grid_size = grid_size
        self.num_objects = num_objects
        self.cell_pixels = cell_pixels
        self.image_size = grid_size * cell_pixels
        self.rng = np.random.RandomState(seed)
        self.agent_pos = None
        self.objects = []

    def reset(self):
        taken = set()
        r, c = self.rng.randint(self.grid_size), self.rng.randint(self.grid_size)
        self.agent_pos = (r, c)
        taken.add((r, c))
        color_names = list(self.COLORS.keys())
        self.objects = []
        for _ in range(self.num_objects):
            while True:
                pos = (self.rng.randint(self.grid_size), self.rng.randint(self.grid_size))
                if pos not in taken:
                    break
            taken.add(pos)
            self.objects.append({"pos": pos, "color": color_names[self.rng.randint(len(color_names))],
                                 "shape": self.SHAPES[self.rng.randint(len(self.SHAPES))]})
        return self._render()

    def step(self, action):
        dr, dc = self.ACTION_DELTAS[action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
        if not self._in_bounds(nr, nc):
            return self._render()
        obj_idx = self._object_at(nr, nc)
        if obj_idx is not None:
            pr, pc = nr + dr, nc + dc
            if self._in_bounds(pr, pc) and self._object_at(pr, pc) is None:
                self.objects[obj_idx]["pos"] = (pr, pc)
                self.agent_pos = (nr, nc)
        else:
            self.agent_pos = (nr, nc)
        return self._render()

    def _in_bounds(self, r, c):
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def _object_at(self, r, c):
        for i, obj in enumerate(self.objects):
            if obj["pos"] == (r, c):
                return i
        return None

    def _render(self):
        cp = self.cell_pixels
        img = np.full((self.image_size, self.image_size, 3), 30, dtype=np.uint8)
        for i in range(1, self.grid_size):
            px = i * cp
            img[px, :] = 45
            img[:, px] = 45
        for obj in self.objects:
            r, c = obj["pos"]
            y0, x0 = r * cp + 1, c * cp + 1
            size = cp - 2
            color = self.COLORS[obj["color"]]
            if obj["shape"] == "square":
                img[y0:y0+size, x0:x0+size] = color
            else:
                cy, cx = y0 + size//2, x0 + size//2
                rad = size // 2
                for dy in range(-rad, rad+1):
                    for dx in range(-rad, rad+1):
                        if dy*dy + dx*dx <= rad*rad:
                            py, px2 = cy+dy, cx+dx
                            if 0 <= py < self.image_size and 0 <= px2 < self.image_size:
                                img[py, px2] = color
        r, c = self.agent_pos
        y0, x0 = r*cp+2, c*cp+2
        s = max(1, cp-4)
        img[y0:y0+s, x0:x0+s] = [255, 255, 255]
        return img


class GridWorldEnv(BrainEnv):
    VOCAB = ["red", "green", "yellow", "blue", "purple", "normal", "slippery",
             "heavy", "up", "down", "left", "right", "push", "slide", "stuck",
             "dodge", "explore", "wait", "score"]

    def __init__(self, grid_size=8, num_objects=5, cell_pixels=4):
        self._env = GridWorld(grid_size, num_objects, cell_pixels)

    def reset(self, seed=None):
        if seed is not None:
            self._env.rng = np.random.RandomState(seed)
        return self._to_tensor(self._env.reset())

    def step(self, action):
        return self._to_tensor(self._env.step(action)), 0.0, False

    @property
    def num_actions(self): return GridWorld.NUM_ACTIONS

    @property
    def vocab_words(self): return self.VOCAB

    def close(self): pass

    @staticmethod
    def _to_tensor(img):
        return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
