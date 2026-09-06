#!/usr/bin/env python3
"""
Train the Infant Brain on any environment.

Usage:
    python scripts/train.py --config configs/atari_pong.yaml
    python scripts/train.py --env atari --game Pong --episodes 5000 --device cuda
    python scripts/train.py --env gridworld --episodes 1000
"""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from infant_brain.brain import Brain
from infant_brain.envs import GridWorldEnv, AtariEnv


def make_env(cfg):
    t = cfg["env"]["type"]
    if t == "gridworld":
        return GridWorldEnv(cfg["env"].get("grid_size", 8), cfg["env"].get("num_objects", 5))
    elif t == "atari":
        return AtariEnv(cfg["env"].get("game", "Pong"), cfg["env"].get("frameskip", 4))
    raise ValueError(f"Unknown env: {t}")


def main():
    p = argparse.ArgumentParser(description="Train the Infant Brain")
    p.add_argument("--config", type=str)
    p.add_argument("--env", choices=["gridworld", "atari"])
    p.add_argument("--game", default="Pong")
    p.add_argument("--episodes", type=int)
    p.add_argument("--device", default="auto")
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--seed", type=int)
    args = p.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    else:
        env_type = args.env or "gridworld"
        env_cfg = {"type": env_type}
        if env_type == "atari":
            env_cfg["game"] = args.game
        cfg = {"env": env_cfg,
               "brain": {"latent_dim": 32, "hidden_dim": 256, "n_concepts": 6},
               "training": {"episodes": 2000, "max_steps": 500, "batch_size": 64,
                             "buffer_size": 100000,
                             "plan_len": 5,
                             "cluster_every": 100, "sleep_every": 200,
                             "check_every": 50, "seed": 42}}

    if args.env: cfg["env"]["type"] = args.env
    if args.game and cfg["env"].get("type") == "atari": cfg["env"]["game"] = args.game
    if args.episodes: cfg["training"]["episodes"] = args.episodes
    if args.seed: cfg["training"]["seed"] = args.seed

    env = make_env(cfg)
    bc = cfg.get("brain", {})
    brain = Brain(env, bc.get("latent_dim", 32), bc.get("hidden_dim", 256),
                  bc.get("n_concepts", 6), args.device)

    tc = cfg.get("training", {})
    history = brain.train(**{k: tc[k] for k in tc if k in [
        "episodes", "max_steps", "batch_size", "buffer_size",
        "plan_len", "cluster_every", "sleep_every", "check_every", "seed"]})

    name = cfg["env"].get("game", cfg["env"]["type"])
    os.makedirs(args.save_dir, exist_ok=True)
    brain.save(os.path.join(args.save_dir, f"brain_{name.lower()}.pt"))
    with open(os.path.join(args.save_dir, f"history_{name.lower()}.json"), "w") as f:
        json.dump(history, f)
    print(f"\n  Done. Saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
