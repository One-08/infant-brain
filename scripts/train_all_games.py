#!/usr/bin/env python3
"""
Train the brain on multiple Atari games sequentially.
Designed for GPU — runs 10 games, saves results for each.

Usage:
    python scripts/train_all_games.py --device cuda --episodes 5000
    python scripts/train_all_games.py --device cuda --episodes 10000 --latent_dim 64
"""

import argparse
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from infant_brain.brain import Brain
from infant_brain.envs import AtariEnv

GAMES = [
    "Pong",
    "Breakout",
    "SpaceInvaders",
    "MsPacman",
    "Freeway",
    "Qbert",
    "Seaquest",
    "BeamRider",
    "Enduro",
    "Asteroids",
]


def main():
    p = argparse.ArgumentParser(description="Train on multiple Atari games")
    p.add_argument("--device", default="auto")
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--buffer_size", type=int, default=500_000)
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_concepts", type=int, default=8)
    p.add_argument("--games", nargs="+", default=None,
                   help="Specific games to train on (default: all 10)")
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    games = args.games or GAMES
    os.makedirs(args.save_dir, exist_ok=True)

    all_results = {}
    total_start = time.time()

    print(f"\n{'='*60}")
    print(f"  INFANT BRAIN — Multi-Game Training")
    print(f"  Games: {len(games)} | Episodes: {args.episodes} | Device: {args.device}")
    print(f"  Latent: {args.latent_dim} | Hidden: {args.hidden_dim}")
    print(f"{'='*60}\n")

    for i, game in enumerate(games):
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(games)}] Starting {game}")
        print(f"{'='*60}")

        try:
            env = AtariEnv(game=game)
            brain = Brain(
                env,
                latent_dim=args.latent_dim,
                hidden_dim=args.hidden_dim,
                n_concepts=args.n_concepts,
                device=args.device,
            )

            t0 = time.time()
            history = brain.train(
                episodes=args.episodes,
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size,
                seed=args.seed,
            )
            elapsed = time.time() - t0

            brain.save(os.path.join(args.save_dir, f"brain_{game.lower()}.pt"))
            with open(os.path.join(args.save_dir, f"history_{game.lower()}.json"), "w") as f:
                json.dump(history, f)

            import numpy as np
            first_wm = np.mean(history["wm_loss"][:50])
            final_wm = np.mean(history["wm_loss"][-50:])
            improvement = first_wm / max(final_wm, 1e-8)

            all_results[game] = {
                "status": "OK",
                "wm_first": round(first_wm, 5),
                "wm_final": round(final_wm, 5),
                "improvement": round(improvement, 1),
                "time_min": round(elapsed / 60, 1),
                "corrections": history["corrections"][-1] if history["corrections"] else 0,
                "concepts": history["tree_size"][-1] if history["tree_size"] else 0,
            }
            print(f"\n  {game}: {improvement:.0f}x improvement in {elapsed/60:.1f} min")

        except Exception as e:
            all_results[game] = {"status": "FAILED", "error": str(e)}
            print(f"\n  {game}: FAILED — {e}")

    total_elapsed = time.time() - total_start

    print(f"\n\n{'='*60}")
    print(f"  ALL GAMES COMPLETE — {total_elapsed/60:.1f} min total")
    print(f"{'='*60}")
    print(f"\n  {'Game':<16s} {'Status':<8s} {'First':>8s} {'Final':>8s} {'Improv':>8s} {'Time':>6s}")
    print(f"  {'-'*54}")
    for game, r in all_results.items():
        if r["status"] == "OK":
            print(f"  {game:<16s} {'OK':<8s} {r['wm_first']:>8.5f} {r['wm_final']:>8.5f} "
                  f"{r['improvement']:>7.0f}x {r['time_min']:>5.1f}m")
        else:
            print(f"  {game:<16s} {'FAIL':<8s} {r.get('error', '')[:40]}")

    with open(os.path.join(args.save_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {args.save_dir}/all_results.json")


if __name__ == "__main__":
    main()
