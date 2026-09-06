#!/usr/bin/env python3
"""PPO baseline at matched compute, for fair comparison with the brain.

Trains stable-baselines3 PPO on the same set of Atari games as
multi_game_test.py, with:
  - The SAME observation that the brain sees: 32x32 RGB, 3-frame stack,
    frameskip=4 (CHW float32 in [0,1]).
  - The SAME interaction budget: episodes * max_steps env-steps total
    (default 500 * 400 = 200_000 env-steps).
  - SB3's default `CnnPolicy` (a small NatureCNN). NOTE: PPO's CnnPolicy
    has slightly more parameters than our brain (~700k vs ~580k); we
    accept this overshoot to use the SB3 default policy unmodified, which
    is a stronger and more standard baseline.

After training, evaluates the deterministic policy on 50 episodes.

Outputs:
  checkpoints/ppo_<game>.zip   — SB3 model
  checkpoints/ppo_baseline_summary.json
  checkpoints/ppo_baseline_summary.txt
"""

import argparse
import json
import os
import sys
import time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image
import torch as th
import torch.nn as nn

import ale_py  # noqa: F401
gym.register_envs(ale_py)


GAMES = [
    "Breakout",
    "SpaceInvaders",
    "Freeway",
    "MsPacman",
    "Boxing",
    "Asterix",
]


class BrainMatchedAtariEnv(gym.Env):
    """Gym env that produces the SAME observation as our `AtariEnv`.

    32x32 RGB resized BILINEAR, 3-frame stack (channel-first, uint8 [0,255]).
    We deliberately match our brain's preprocessing so the PPO baseline
    sees the same input the brain does. SB3's `CnnPolicy` expects uint8
    images and normalizes them to [0,1] internally.
    """

    metadata = {"render_modes": []}

    def __init__(self, game, frameskip=4, frame_stack=3, img_size=32,
                 max_episode_steps=400):
        self._env = gym.make(f"ALE/{game}-v5", render_mode="rgb_array",
                             frameskip=frameskip)
        self.action_space = self._env.action_space
        self._frame_stack = frame_stack
        self._img_size = img_size
        self._max_episode_steps = max_episode_steps
        self._step_count = 0
        c = 3 * frame_stack
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(c, img_size, img_size), dtype=np.uint8)
        self._buf = deque(maxlen=frame_stack)

    def _preprocess(self, frame):
        img = Image.fromarray(frame).resize(
            (self._img_size, self._img_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        return arr.transpose(2, 0, 1)  # CHW

    def _stack(self):
        return np.concatenate(list(self._buf), axis=0)

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed)
        frame = self._preprocess(obs)
        for _ in range(self._frame_stack):
            self._buf.append(frame)
        self._step_count = 0
        return self._stack(), info

    def step(self, action):
        obs, reward, done, truncated, info = self._env.step(action)
        self._buf.append(self._preprocess(obs))
        self._step_count += 1
        if self._step_count >= self._max_episode_steps:
            truncated = True
        return self._stack(), float(reward), bool(done), bool(truncated), info

    def close(self):
        self._env.close()


class TinyCnn(nn.Module):
    """Compact CNN feature extractor for 32x32 RGB stacks.

    Roughly matches the brain's CNN encoder (~few conv layers, ~256-d
    features) so the PPO baseline has a comparable visual front-end.
    """

    def __init__(self, observation_space, features_dim=256):
        super().__init__()
        self.features_dim = features_dim
        c = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float() / 255.0
            n_flat = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, obs):
        return self.linear(self.cnn(obs.float() / 255.0))


def train_ppo(game, total_timesteps, save_dir, seed):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env_fn = lambda s=seed: BrainMatchedAtariEnv(game)
    vec_env = DummyVecEnv([env_fn])
    vec_env.seed(seed)

    policy_kwargs = dict(
        features_extractor_class=TinyCnn,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        normalize_images=False,  # extractor already normalizes
    )

    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        seed=seed,
        verbose=0,
        policy_kwargs=policy_kwargs,
        n_steps=128,
        batch_size=128,
        learning_rate=2.5e-4,
        n_epochs=4,
        clip_range=0.1,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  PPO params: {n_params:,}")

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    elapsed = time.time() - t0

    save_path = os.path.join(save_dir, f"ppo_{game.lower()}_seed{seed}.zip")
    model.save(save_path)
    vec_env.close()
    print(f"  Trained PPO on {game} (seed={seed}) in {elapsed/60:.1f} min  -> {save_path}")
    return save_path, n_params, elapsed


def eval_ppo(game, save_path, n_episodes=50, max_steps=2000):
    from stable_baselines3 import PPO
    env = BrainMatchedAtariEnv(game, max_episode_steps=max_steps)
    model = PPO.load(save_path)

    out = {}
    for mode in ["argmax", "sample"]:
        det = (mode == "argmax")
        rewards = []
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=50_000 + ep)
            ep_r = 0.0
            for t in range(max_steps):
                action, _ = model.predict(obs, deterministic=det)
                obs, r, done, trunc, _ = env.step(int(action))
                ep_r += r
                if done or trunc:
                    break
            rewards.append(ep_r)
        rs = np.array(rewards)
        out[mode] = dict(mean=float(rs.mean()), std=float(rs.std()),
                         minv=float(rs.min()), maxv=float(rs.max()),
                         median=float(np.median(rs)))
        print(f"  [{mode:6s}] mean={rs.mean():+.2f}  std={rs.std():.2f}  "
              f"min={rs.min():+.1f}  max={rs.max():+.1f}  med={np.median(rs):+.1f}")
    env.close()
    return out


def write_summary(rows, path_json, path_txt):
    lines = []
    lines.append("=" * 100)
    lines.append("  PPO BASELINE  (matched input to brain, 50-ep eval)")
    lines.append("=" * 100)
    lines.append(f"{'game':14s}  {'seed':>5s}  {'argmax':>10s}  {'sample':>10s}  "
                 f"{'max':>5s}  {'params':>9s}  {'frames':>9s}  {'min':>5s}")
    lines.append("-" * 100)
    for r in rows:
        lines.append(
            f"{r['game']:14s}  "
            f"{r.get('seed', '-'):>5}  "
            f"{r['argmax']['mean']:+10.2f}  "
            f"{r['sample']['mean']:+10.2f}  "
            f"{r['argmax']['maxv']:+5.0f}  "
            f"{r['n_params']:9,d}  "
            f"{r['total_timesteps']:9,d}  "
            f"{r['elapsed_min']:5.1f}"
        )
    lines.append("-" * 100)
    text = "\n".join(lines)
    with open(path_txt, "w") as f:
        f.write(text)
    with open(path_json, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print("\n" + text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=500,
                   help="Used only to compute total_timesteps = episodes * max_steps.")
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=666)
    p.add_argument("--seeds", default=None,
                   help="Comma-separated list of seeds; sweeps each.")
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--games", default=None,
                   help="Comma-separated subset. Default: all.")
    p.add_argument("--n_eval_episodes", type=int, default=50)
    p.add_argument("--skip_train", action="store_true")
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    total_timesteps = args.episodes * args.max_steps

    games = GAMES
    if args.games:
        wanted = {s.strip() for s in args.games.split(",")}
        games = [g for g in GAMES if g in wanted]

    seeds = ([int(s) for s in args.seeds.split(",")]
             if args.seeds else [args.seed])

    rows = []
    summary_json = os.path.join(args.save_dir, "ppo_baseline_summary.json")
    summary_txt = os.path.join(args.save_dir, "ppo_baseline_summary.txt")

    prior = {}  # key: (game, seed) -> row
    if os.path.exists(summary_json):
        try:
            with open(summary_json) as f:
                prior_rows = json.load(f).get("rows", [])
            for r in prior_rows:
                key = (r["game"], int(r.get("seed", args.seed)))
                prior[key] = r
        except Exception:
            pass

    for seed in seeds:
        for g in games:
            save_path = os.path.join(args.save_dir, f"ppo_{g.lower()}_seed{seed}.zip")
            print(f"\n{'='*70}\n  PPO  {g}  seed={seed}  total_timesteps={total_timesteps:,}\n{'='*70}")
            if args.skip_train:
                if not os.path.exists(save_path):
                    legacy = os.path.join(args.save_dir, f"ppo_{g.lower()}.zip")
                    if os.path.exists(legacy) and seed == args.seed:
                        save_path = legacy
                    else:
                        print(f"  no model at {save_path}; skipping.")
                        continue
                n_params = prior.get((g, seed), {}).get("n_params", -1)
                elapsed = prior.get((g, seed), {}).get("elapsed_min", 0.0)
            else:
                if (g, seed) in prior and "argmax" in prior[(g, seed)]:
                    print(f"  [{g} seed={seed}] already evaluated; skipping.")
                    continue
                # Reuse the legacy single-seed checkpoint when seed matches.
                legacy = os.path.join(args.save_dir, f"ppo_{g.lower()}.zip")
                if seed == 666 and os.path.exists(legacy) and not os.path.exists(save_path):
                    print(f"  [{g} seed={seed}] reusing legacy {legacy}")
                    save_path = legacy
                    # Param count is in the prior summary if we've seen it.
                    n_params = prior.get((g, seed), {}).get("n_params", -1)
                    elapsed = prior.get((g, seed), {}).get("elapsed_min", 0.0)
                else:
                    save_path, n_params, secs = train_ppo(
                        g, total_timesteps=total_timesteps,
                        save_dir=args.save_dir, seed=seed)
                    # Re-tag with seed if train_ppo wrote the legacy filename
                    if not os.path.exists(save_path) or save_path == legacy:
                        os.rename(legacy, os.path.join(
                            args.save_dir, f"ppo_{g.lower()}_seed{seed}.zip"))
                        save_path = os.path.join(
                            args.save_dir, f"ppo_{g.lower()}_seed{seed}.zip")
                    elapsed = secs / 60.0

            scores = eval_ppo(g, save_path, n_episodes=args.n_eval_episodes,
                              max_steps=2000)
            rows.append({
                "game": g,
                "seed": seed,
                "n_params": n_params,
                "total_timesteps": total_timesteps,
                "elapsed_min": elapsed,
                "argmax": scores["argmax"],
                "sample": scores["sample"],
            })
            merged = {(r["game"], int(r.get("seed", args.seed))): r for r in rows}
            for k, v in prior.items():
                merged.setdefault(k, v)
            ordered = []
            for gname in [x for x in GAMES]:
                for s in sorted({k[1] for k in merged if k[0] == gname}):
                    ordered.append(merged[(gname, s)])
            write_summary(ordered, summary_json, summary_txt)

    print(f"\nSummary: {summary_json} and {summary_txt}")


if __name__ == "__main__":
    main()
