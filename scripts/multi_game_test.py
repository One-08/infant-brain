#!/usr/bin/env python3
"""Multi-game generalization test for the infant-brain prototype.

Runs the *same* recipe on a list of Atari games to validate whether the
architecture generalizes beyond the one game (Breakout) we tuned on.

Recipe (locked across all games — no per-game hyperparameter tuning):
  - CNN encoder, latent_dim=32, hidden_dim=256
  - Actor MLP width=384, Critic MLP width=384
  - Scalar critic (no distributional)
  - 500 episodes, max_steps=400 per episode
  - In-training eval every 25 eps, 5 eps each, eps=0.05
  - Early stop after 12 evals without improvement
  - auto_fire ON for games that need it (Breakout, SpaceInvaders)
  - NO action prior, NO KL anchor (would be per-game tuning)

After all training runs finish, runs a 50-episode argmax+sample eval per
game and writes a comparison table to:
  checkpoints/multi_game_summary.json
  checkpoints/multi_game_summary.txt

Random-policy baseline is computed for each game so the architecture's
contribution is visible.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from infant_brain.brain import Brain
from infant_brain.envs import AtariEnv


# Games to test, with their per-game flags. The flags are NOT tuned —
# auto_fire is a well-known requirement for these specific games regardless
# of the agent (the env literally won't start without a FIRE action) and
# applying it is standard practice in every Atari benchmark.
GAMES = [
    {"name": "Breakout",      "auto_fire": True},
    {"name": "SpaceInvaders", "auto_fire": True},
    {"name": "Freeway",       "auto_fire": False},
    {"name": "MsPacman",      "auto_fire": False},
    {"name": "Boxing",        "auto_fire": False},
    {"name": "Asterix",       "auto_fire": False},
]

RECIPE = dict(
    latent_dim=32,
    hidden_dim=256,
    actor_hidden_dim=384,
    critic_hidden_dim=384,
    encoder_type="cnn",
    n_concepts=6,
    staged=True,
)

# Expected total parameter count for the locked architecture on
# Breakout (4 actions). If a code change silently grows the model
# (e.g. encoder depth flips because of the MPS/CUDA autosize bug we
# fixed before), the assertion below trips at training start, so we
# never burn 4 hours of compute on architecturally-incompatible seeds
# that would silently break the multi-seed mean/std reporting.
EXPECTED_PARAMS_BREAKOUT = 580_616
PARAM_TOLERANCE = 0  # architecture must be byte-identical across seeds.

TRAIN_KWARGS = dict(
    max_steps=400,
    batch_size=32,
    buffer_size=10_000,
    cluster_every=10,
    sleep_every=25,
    check_every=10,
    log_every=50,
)

EVAL_PROTOCOL = dict(
    eval_episodes=50,
    eval_max_steps=2000,
    eval_epsilon=0.05,
)


def make_env(name):
    return AtariEnv(name)


def train_one(game_cfg, episodes, seed, save_dir, device):
    name = game_cfg["name"]
    tag = f"multi_{name.lower()}_seed{seed}"
    ckpt_path = os.path.join(save_dir, f"brain_{name.lower()}_{tag}.pt")
    best_ckpt_path = os.path.join(save_dir, f"brain_{name.lower()}_{tag}_best.pt")

    print(f"\n{'='*70}\n  TRAINING {name} (seed={seed})  -> {best_ckpt_path}\n{'='*70}")

    env = make_env(name)
    brain = Brain(env, device=device, **RECIPE)
    print(f"  params={brain.total_params:,}  num_actions={env.num_actions}")

    if name == "Breakout":
        delta = abs(brain.total_params - EXPECTED_PARAMS_BREAKOUT)
        if delta > PARAM_TOLERANCE:
            raise SystemExit(
                f"\n  ARCHITECTURE DRIFT DETECTED on {name}: "
                f"got {brain.total_params:,} params, expected "
                f"{EXPECTED_PARAMS_BREAKOUT:,} ({delta:+,} delta).\n"
                f"  Refusing to train -- this would produce checkpoints "
                f"that are not state-dict-compatible with seed 666 and "
                f"break the multi-seed mean/std reporting in the paper.\n"
                f"  Likely cause: torch.backends.mps.is_available() flipped, "
                f"or RECIPE changed. Inspect "
                f"infant_brain/modules/world_model.py and "
                f"infant_brain/envs/atari.py.")

    eval_kwargs = dict(
        eval_every=25,
        eval_episodes=5,
        eval_max_steps=TRAIN_KWARGS["max_steps"] * 4,
        eval_auto_fire=game_cfg["auto_fire"],
        eval_epsilon=0.05,
        best_ckpt_path=best_ckpt_path,
        eval_env_factory=lambda n=name: make_env(n),
        early_stop_patience=12,
    )

    t0 = time.time()
    history = brain.train(episodes=episodes, seed=seed, **TRAIN_KWARGS, **eval_kwargs)
    elapsed = time.time() - t0
    brain.save(ckpt_path)
    env.close()

    # Pull last-N training reward from history (history["reward"] is per-ep).
    rewards = history["reward"]
    n = max(1, len(rewards) // 5)
    train_summary = {
        "episodes_done": len(rewards),
        "reward_first_n": float(np.mean(rewards[:n])),
        "reward_last_n": float(np.mean(rewards[-n:])),
        "reward_best_train": float(max(rewards)),
        "elapsed_min": elapsed / 60.0,
    }
    print(f"  done in {elapsed/60:.1f} min  best_train={train_summary['reward_best_train']:+.1f}")
    return best_ckpt_path, train_summary


def eval_policy(env_factory, brain, n_episodes, max_steps, deterministic,
                epsilon, auto_fire):
    eval_env = env_factory()
    rewards = []
    for ep in range(n_episodes):
        frame = eval_env.reset(seed=50_000 + ep)
        brain.world_model.reset_memory(batch_size=1, device=brain.device)
        ep_r = 0.0
        for t in range(max_steps):
            force = 1 if (auto_fire and t < 5 and eval_env.num_actions > 1) else -1
            a = brain.act(frame, deterministic=deterministic,
                          epsilon=epsilon, force_action=force)
            frame, r, done = eval_env.step(a)
            ep_r += r
            if done:
                break
        rewards.append(ep_r)
    eval_env.close()
    return np.array(rewards)


def random_policy_score(name, n_episodes, max_steps, auto_fire, seed=12345):
    env = make_env(name)
    rng = np.random.default_rng(seed)
    rewards = []
    for ep in range(n_episodes):
        env.reset(seed=50_000 + ep)
        ep_r = 0.0
        for t in range(max_steps):
            if auto_fire and t < 5 and env.num_actions > 1:
                a = 1
            else:
                a = int(rng.integers(env.num_actions))
            _, r, done = env.step(a)
            ep_r += r
            if done:
                break
        rewards.append(ep_r)
    env.close()
    return np.array(rewards)


def eval_one(game_cfg, ckpt_path, n_eval_episodes, device):
    name = game_cfg["name"]
    print(f"\n{'-'*70}\n  EVAL {name} on {ckpt_path}\n{'-'*70}")
    env = make_env(name)
    brain = Brain(env, device=device, **RECIPE)
    brain.load(ckpt_path)

    out = {"game": name}
    for mode in ["argmax", "sample"]:
        det = (mode == "argmax")
        r = eval_policy(
            env_factory=lambda n=name: make_env(n),
            brain=brain,
            n_episodes=n_eval_episodes,
            max_steps=EVAL_PROTOCOL["eval_max_steps"],
            deterministic=det,
            epsilon=EVAL_PROTOCOL["eval_epsilon"],
            auto_fire=game_cfg["auto_fire"],
        )
        out[mode] = dict(mean=float(r.mean()), std=float(r.std()),
                         minv=float(r.min()), maxv=float(r.max()),
                         median=float(np.median(r)))
        print(f"  [{mode:6s}] mean={r.mean():+.2f}  std={r.std():.2f}  "
              f"min={r.min():+.1f}  max={r.max():+.1f}  median={np.median(r):+.1f}")
    env.close()

    # Random policy baseline (cheap; same protocol)
    rand_r = random_policy_score(name, n_episodes=20,
                                 max_steps=EVAL_PROTOCOL["eval_max_steps"],
                                 auto_fire=game_cfg["auto_fire"])
    out["random"] = dict(mean=float(rand_r.mean()), std=float(rand_r.std()),
                         maxv=float(rand_r.max()))
    print(f"  [random] mean={rand_r.mean():+.2f}  std={rand_r.std():.2f}  "
          f"max={rand_r.max():+.1f}")
    out["delta_argmax_vs_random"] = out["argmax"]["mean"] - out["random"]["mean"]
    out["ratio_argmax_vs_random"] = (
        out["argmax"]["mean"] / out["random"]["mean"]
        if abs(out["random"]["mean"]) > 1e-3 else float("nan")
    )
    return out


def write_summary_txt(rows, path):
    lines = []
    lines.append("=" * 110)
    lines.append("  MULTI-GAME GENERALIZATION TEST  (50-ep eval, eps=0.05)")
    lines.append("=" * 110)
    lines.append(f"{'game':14s}  {'seed':>5s}  {'argmax':>10s}  {'sample':>10s}  "
                 f"{'random':>10s}  {'delta':>8s}  {'ratio':>6s}  {'max':>5s}  "
                 f"{'train_best':>10s}  {'eps':>5s}  {'min':>5s}")
    lines.append("-" * 110)
    for r in rows:
        lines.append(
            f"{r['game']:14s}  "
            f"{r.get('seed', '-'):>5}  "
            f"{r['argmax']['mean']:+10.2f}  "
            f"{r['sample']['mean']:+10.2f}  "
            f"{r['random']['mean']:+10.2f}  "
            f"{r['delta_argmax_vs_random']:+8.2f}  "
            f"{r['ratio_argmax_vs_random']:+6.2f}  "
            f"{r['argmax']['maxv']:+5.0f}  "
            f"{r['train']['reward_best_train']:+10.1f}  "
            f"{r['train']['episodes_done']:5d}  "
            f"{r['train']['elapsed_min']:5.1f}"
        )
    lines.append("-" * 110)
    lines.append(
        f"  Recipe: encoder={RECIPE['encoder_type']}, "
        f"hidden={RECIPE['hidden_dim']}, actor={RECIPE['actor_hidden_dim']}, "
        f"critic={RECIPE['critic_hidden_dim']}. NO per-game KL anchor or action prior.")
    lines.append("")
    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text)
    print("\n" + text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--seed", type=int, default=666,
                   help="Single seed (used when --seeds is not provided).")
    p.add_argument("--seeds", default=None,
                   help="Comma-separated list of seeds to sweep. When set, "
                        "each (game, seed) pair is trained + evaluated. "
                        "Per-seed checkpoints and a multi-seed summary are "
                        "written.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--games", default=None,
                   help="Comma-separated subset of games to run. Default: all.")
    p.add_argument("--skip_train", action="store_true",
                   help="Skip training; just re-evaluate existing checkpoints.")
    p.add_argument("--n_eval_episodes", type=int, default=50)
    args = p.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() // 2))
    os.makedirs(args.save_dir, exist_ok=True)

    games_to_run = GAMES
    if args.games:
        wanted = {s.strip() for s in args.games.split(",")}
        games_to_run = [g for g in GAMES if g["name"] in wanted]
        unknown = wanted - {g["name"] for g in games_to_run}
        if unknown:
            print(f"  WARNING: unknown games requested: {unknown}")

    seeds = ([int(s) for s in args.seeds.split(",")]
             if args.seeds else [args.seed])

    rows = []
    summary_path_json = os.path.join(args.save_dir, "multi_game_summary.json")
    summary_path_txt = os.path.join(args.save_dir, "multi_game_summary.txt")

    # Resume support: load any rows we already have (keyed by (game, seed))
    # so re-runs only train missing combos.
    prior = {}  # key: (game, seed) -> row
    if os.path.exists(summary_path_json):
        try:
            with open(summary_path_json) as f:
                prior_data = json.load(f)
            for r in prior_data.get("rows", []):
                key = (r["game"], int(r.get("seed", args.seed)))
                prior[key] = r
            print(f"  Loaded {len(prior)} prior rows.")
        except Exception as e:
            print(f"  Could not parse prior summary ({e}); starting fresh.")

    for seed in seeds:
        for cfg in games_to_run:
            name = cfg["name"]
            tag = f"multi_{name.lower()}_seed{seed}"
            best_ckpt_path = os.path.join(args.save_dir, f"brain_{name.lower()}_{tag}_best.pt")

            if args.skip_train:
                if not os.path.exists(best_ckpt_path):
                    fallback = os.path.join(
                        args.save_dir, f"brain_{name.lower()}_{tag}.pt")
                    if os.path.exists(fallback):
                        print(f"  [{name} seed={seed}] no _best.pt; using final ckpt {fallback}")
                        best_ckpt_path = fallback
                    else:
                        print(f"  [{name} seed={seed}] no checkpoint; skipping.")
                        continue
                train_summary = prior.get((name, seed), {}).get("train", {
                    "episodes_done": 0, "reward_first_n": 0.0,
                    "reward_last_n": 0.0, "reward_best_train": 0.0,
                    "elapsed_min": 0.0,
                })
            else:
                # Skip if we already have this (game, seed) combo evaluated.
                if (name, seed) in prior and "argmax" in prior[(name, seed)]:
                    print(f"  [{name} seed={seed}] already evaluated; skipping.")
                    continue
                best_ckpt_path, train_summary = train_one(
                    cfg, episodes=args.episodes, seed=seed,
                    save_dir=args.save_dir, device=args.device)
                if not os.path.exists(best_ckpt_path):
                    fallback = os.path.join(
                        args.save_dir, f"brain_{name.lower()}_{tag}.pt")
                    if os.path.exists(fallback):
                        print(f"  [{name} seed={seed}] no _best.pt; using {fallback}")
                        best_ckpt_path = fallback
                    else:
                        print(f"  [{name} seed={seed}] SKIPPING eval; no ckpt.")
                        continue

            result = eval_one(cfg, best_ckpt_path,
                              n_eval_episodes=args.n_eval_episodes,
                              device=args.device)
            result["seed"] = seed
            result["train"] = train_summary
            result["ckpt"] = best_ckpt_path
            rows.append(result)

            merged = {(r["game"], int(r.get("seed", args.seed))): r for r in rows}
            for k, v in prior.items():
                merged.setdefault(k, v)
            ordered = []
            for g in GAMES:
                for s in sorted({k[1] for k in merged if k[0] == g["name"]}):
                    ordered.append(merged[(g["name"], s)])
            all_rows = ordered
            with open(summary_path_json, "w") as f:
                json.dump({"rows": all_rows, "recipe": RECIPE,
                           "eval_protocol": EVAL_PROTOCOL,
                           "train_kwargs": TRAIN_KWARGS,
                           "seeds": seeds}, f, indent=2)
            write_summary_txt(all_rows, summary_path_txt)

    print(f"\nSummary written to {summary_path_json} and {summary_path_txt}")


if __name__ == "__main__":
    main()
