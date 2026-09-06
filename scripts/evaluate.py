#!/usr/bin/env python3
"""
Evaluate a trained Infant Brain.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/brain_pong.pt --env atari --game Pong
    python scripts/evaluate.py --checkpoint checkpoints/brain_pong.pt --env atari --game Pong --deterministic

Note: previous versions of this script sampled RANDOM actions during eval
(`env.step(rng.randint(env.num_actions))`), so the reported "evaluation"
was random-policy reward, not the trained policy. This version uses
`brain.act(frame)` and reports the actor's own performance.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from infant_brain.brain import Brain
from infant_brain.envs import GridWorldEnv, AtariEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--env", required=True, choices=["gridworld", "atari"])
    p.add_argument("--game", default="Pong")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=2000,
                   help="Per-episode step cap (Atari episodes can be long).")
    p.add_argument("--device", default="auto")
    p.add_argument("--deterministic", action="store_true",
                   help="Use argmax instead of sampling from the policy.")
    p.add_argument("--seed", type=int, default=999)
    args = p.parse_args()

    env = AtariEnv(args.game) if args.env == "atari" else GridWorldEnv()
    brain = Brain(env, device=args.device)
    brain.load(args.checkpoint)

    print(f"\n{'='*60}\n  Evaluating on {args.env} ({args.game})  "
          f"[mode={'argmax' if args.deterministic else 'sample'}]\n{'='*60}\n")

    rewards = []
    seen_words = set()
    for ep in range(args.episodes):
        frame = env.reset(seed=args.seed + ep)
        brain.world_model.reset_memory(batch_size=1, device=brain.device)
        ep_r = 0.0
        ep_words = set()
        for _ in range(args.max_steps):
            top = brain.describe(frame)
            ep_words.add(top[0][0])
            action = brain.act(frame, deterministic=args.deterministic)
            frame, r, done = env.step(action)
            ep_r += r
            if done:
                break
        rewards.append(ep_r)
        seen_words |= ep_words
        print(f"  Episode {ep+1:3d}: reward={ep_r:+8.1f}  words_seen={sorted(ep_words)}")

    env.close()

    rewards = np.array(rewards)
    print(f"\n  Mean reward over {args.episodes} eps: {rewards.mean():+.2f}  "
          f"(std={rewards.std():.2f}, min={rewards.min():+.1f}, max={rewards.max():+.1f})")
    print(f"  Vocabulary touched: {sorted(seen_words)}")

    print(f"\n  Language test (15 scenes):")
    env2 = AtariEnv(args.game) if args.env == "atari" else GridWorldEnv()
    rng = np.random.RandomState(args.seed)
    frame = env2.reset(seed=args.seed)
    brain.world_model.reset_memory(batch_size=1, device=brain.device)
    for i in range(15):
        for _ in range(rng.randint(5, 30)):
            action = brain.act(frame, deterministic=args.deterministic)
            frame, r, d = env2.step(action)
            if d:
                frame = env2.reset()
                brain.world_model.reset_memory(batch_size=1, device=brain.device)
        top = brain.describe(frame)
        print(f"    Scene {i+1:2d}: {top[0][0]:8s}({top[0][1]:.2f})  "
              f"{top[1][0]:8s}({top[1][1]:.2f})  {top[2][0]:8s}({top[2][1]:.2f})")
    env2.close()


if __name__ == "__main__":
    main()
