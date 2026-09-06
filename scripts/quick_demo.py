#!/usr/bin/env python3
"""
Quick local demo on a single Atari game.

Designed to fit a CPU laptop (Apple Silicon / x86) in ~10-20 minutes:
  * Small brain (latent=32, hidden=128).
  * Short episodes (max_steps=300).
  * Modest count (default 50 episodes).
  * Logs every 5 episodes so you can watch progress live.
  * After training, evaluates the trained policy with brain.act() over 5
    episodes (greedy) and prints a clean summary.

Usage:
    python scripts/quick_demo.py --game Breakout --episodes 50
    python scripts/quick_demo.py --game Pong --episodes 50 --device cpu
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="Breakout")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--device", default="cpu",
                   help="cpu / cuda / mps / auto. CPU is safest on macOS.")
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--actor_hidden_dim", type=int, default=256,
                   help="Width of the actor MLP. Default 256 matches the "
                        "legacy hard-coded size; bump (e.g., 384/512) when "
                        "scaling up capacity.")
    p.add_argument("--critic_hidden_dim", type=int, default=256,
                   help="Width of the critic MLP. Default 256 matches the "
                        "legacy hard-coded size.")
    p.add_argument("--distributional_critic", action="store_true",
                   help="Use a DreamerV3-style categorical value head "
                        "(cross-entropy on two-hot targets) instead of a "
                        "scalar MSE critic. More robust on sparse-reward / "
                        "wide-return games.")
    p.add_argument("--critic_n_bins", type=int, default=51,
                   help="Number of value bins for the distributional critic.")
    p.add_argument("--critic_v_min", type=float, default=-20.0,
                   help="Lower bound of the distributional critic value range.")
    p.add_argument("--critic_v_max", type=float, default=20.0,
                   help="Upper bound of the distributional critic value range.")
    p.add_argument("--encoder", choices=["vit", "cnn"], default="vit",
                   help="Visual front-end: ViT (transformer) or CNN. "
                        "CNN is faster on small Atari frames and biologically "
                        "closer to V1->IT.")
    p.add_argument("--encoder_depth", type=int, default=None,
                   help="Number of transformer/conv blocks in the encoder. "
                        "Default: 2 on CPU, 4 on GPU. Bigger = wider "
                        "receptive field (CNN) or more reasoning (ViT).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--eval_episodes", type=int, default=5)
    p.add_argument("--no_train", action="store_true",
                   help="Skip training and just evaluate an existing checkpoint.")
    p.add_argument("--eval_mode", choices=["argmax", "sample", "both"], default="both",
                   help="argmax = greedy. sample = follow the policy distribution. "
                        "Some games (e.g., Breakout) need stochasticity to launch the ball.")
    p.add_argument("--entropy_coeff", type=float, default=None,
                   help="Override the actor's entropy regularization (default 5e-3). "
                        "Higher = more exploration / less mode collapse.")
    p.add_argument("--epsilon", type=float, default=0.05,
                   help="Epsilon for epsilon-greedy at eval. Standard Atari "
                        "practice uses 0.05. Set to 0 for pure policy.")
    p.add_argument("--auto_fire", action="store_true",
                   help="Force action=1 (FIRE) on the first 5 steps of each "
                        "eval episode. Required for Breakout-style games "
                        "where the ball must be launched manually.")
    p.add_argument("--run_tag", default="demo",
                   help="Suffix on checkpoint/history filenames to keep runs separate.")
    p.add_argument("--eval_every", type=int, default=0,
                   help="If > 0, run a quick greedy eval every N episodes "
                        "during training and save the best checkpoint by "
                        "eval reward. Strongly recommended for long runs to "
                        "protect against late-training reward drift.")
    p.add_argument("--best_eval_episodes", type=int, default=3,
                   help="Episodes per in-training eval pass.")
    p.add_argument("--actor_kl_coef", type=float, default=0.0,
                   help="Coefficient on KL(current_actor || best_checkpoint_actor) "
                        "added to the actor loss. 0 disables. Useful range "
                        "0.01 - 0.1. Prevents destructive late-training drift.")
    p.add_argument("--patience", type=int, default=0,
                   help="Early stopping: stop after N consecutive in-training "
                        "evals without improvement. 0 disables. Requires "
                        "eval_every > 0.")
    p.add_argument("--action_prior", default=None,
                   help="Comma-separated weights to bias eps-greedy "
                        "exploration toward specific actions (Step 2). "
                        "Length must equal num_actions. "
                        "Example: '0.1,0.1,0.1,0.7' biases toward action 3.")
    p.add_argument("--action_prior_decay", type=float, default=0.99,
                   help="Per-episode decay of the exploration prior toward "
                        "uniform. 1.0 = no decay. 0.99 ~ fades over 500 eps.")
    p.add_argument("--ref_ckpt", default=None,
                   help="Path to a checkpoint whose actor will be loaded as "
                        "the initial KL trust-region reference (Step 3). "
                        "Only useful with --actor_kl_coef > 0.")
    args = p.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() // 2))

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, f"brain_{args.game.lower()}_{args.run_tag}.pt")
    best_ckpt_path = os.path.join(args.save_dir, f"brain_{args.game.lower()}_{args.run_tag}_best.pt")
    hist_path = os.path.join(args.save_dir, f"history_{args.game.lower()}_{args.run_tag}.json")

    print(f"\n{'='*64}")
    print(f"  QUICK DEMO  game={args.game}  episodes={args.episodes}  "
          f"device={args.device}")
    print(f"{'='*64}\n")

    env = AtariEnv(args.game)
    brain = Brain(env, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
                  n_concepts=6, device=args.device, staged=True,
                  encoder_type=args.encoder,
                  encoder_depth=args.encoder_depth,
                  actor_hidden_dim=args.actor_hidden_dim,
                  critic_hidden_dim=args.critic_hidden_dim,
                  critic_distributional=args.distributional_critic,
                  critic_n_bins=args.critic_n_bins,
                  critic_v_min=args.critic_v_min,
                  critic_v_max=args.critic_v_max)
    if args.distributional_critic:
        print(f"  Distributional critic: {args.critic_n_bins} bins in "
              f"[{args.critic_v_min}, {args.critic_v_max}]")
    if args.entropy_coeff is not None:
        brain.entropy_coeff = args.entropy_coeff
        print(f"  Overriding entropy_coeff={args.entropy_coeff}")
    if args.action_prior is not None:
        weights = [float(x) for x in args.action_prior.split(",")]
        brain.set_exploration_prior(weights, decay=args.action_prior_decay)
        print(f"  Action prior: {brain._exploration_prior.tolist()} "
              f"(decay={args.action_prior_decay})")
    if args.ref_ckpt is not None and args.actor_kl_coef > 0:
        brain.load_reference_actor(args.ref_ckpt)
    depth_str = f"depth={args.encoder_depth}" if args.encoder_depth else "depth=default"
    print(f"  Built. encoder={args.encoder} {depth_str}  "
          f"params={brain.total_params:,}  num_actions={env.num_actions}")

    if not args.no_train:
        t0 = time.time()
        eval_kwargs = {}
        if args.eval_every > 0:
            eval_kwargs.update(
                eval_every=args.eval_every,
                eval_episodes=args.best_eval_episodes,
                eval_max_steps=args.max_steps * 4,
                eval_auto_fire=args.auto_fire,
                eval_epsilon=args.epsilon,
                best_ckpt_path=best_ckpt_path,
                eval_env_factory=lambda g=args.game: AtariEnv(g),
            )
            print(f"  In-training eval: every {args.eval_every} eps, "
                  f"{args.best_eval_episodes} eps each, eps={args.epsilon}, "
                  f"best -> {best_ckpt_path}")
        if args.actor_kl_coef > 0:
            eval_kwargs["actor_kl_coef"] = args.actor_kl_coef
        if args.patience > 0:
            eval_kwargs["early_stop_patience"] = args.patience
        history = brain.train(
            episodes=args.episodes,
            max_steps=args.max_steps,
            batch_size=32,
            buffer_size=10_000,
            cluster_every=10,
            sleep_every=25,
            check_every=10,
            seed=args.seed,
            log_every=max(1, args.episodes // 10),
            **eval_kwargs,
        )
        elapsed = time.time() - t0
        print(f"\n  Training done in {elapsed/60:.1f} min "
              f"({elapsed/args.episodes:.1f}s/episode).")

        brain.save(ckpt_path)
        with open(hist_path, "w") as f:
            json.dump({k: list(map(float, v)) if v else [] for k, v in history.items()},
                      f, indent=2)

        rewards = history["reward"]
        wm = history["wm_loss"]
        n = max(1, len(rewards) // 5)
        print(f"\n  TRAINING SUMMARY")
        print(f"  {'-'*40}")
        print(f"  episodes:           {len(rewards)}")
        print(f"  reward first {n}:    {np.mean(rewards[:n]):+.2f}")
        print(f"  reward last  {n}:    {np.mean(rewards[-n:]):+.2f}")
        print(f"  reward best:        {max(rewards):+.1f}")
        print(f"  wm_loss first {n}:   {np.mean(wm[:n]):.4f}")
        print(f"  wm_loss last  {n}:   {np.mean(wm[-n:]):.4f}")
        print(f"  improvement (wm):   "
              f"{np.mean(wm[:n]) / max(np.mean(wm[-n:]), 1e-8):.1f}x")
        print(f"  intuition gate:     yes_rate="
              f"{brain.intuition.stats['yes_rate']:.0%}, "
              f"explore_rate={brain.intuition.stats['exploration']:.0%}")
        print(f"  dev stage reached:  "
              f"{brain._dev_stage} ({brain._dev_stages.get(brain._dev_stage, '?')})")
    else:
        brain.load(ckpt_path)

    # ---- Evaluate trained policy with brain.act() ----
    if args.eval_episodes <= 0:
        env.close()
        return

    modes = ["argmax", "sample"] if args.eval_mode == "both" else [args.eval_mode]
    eval_env = AtariEnv(args.game)
    summary = {}
    for mode in modes:
        deterministic = (mode == "argmax")
        # epsilon-greedy is most useful when sampling (deterministic argmax
        # policies often collapse to one action — let eps inject diversity).
        eps = args.epsilon
        print(f"\n  EVALUATING (mode={mode}, eps={eps}, "
              f"auto_fire={args.auto_fire}, {args.eval_episodes} episodes)")
        print(f"  {'-'*40}")
        eval_rewards = []
        action_hist_total = np.zeros(env.num_actions, dtype=np.int64)
        for ep in range(args.eval_episodes):
            frame = eval_env.reset(seed=10_000 + ep)
            brain.world_model.reset_memory(batch_size=1, device=brain.device)
            ep_r = 0.0
            steps = 0
            action_counts = np.zeros(env.num_actions, dtype=np.int64)
            for t in range(args.max_steps * 4):
                # Force FIRE for the first few steps so games like Breakout
                # (where the ball must be launched) actually start playing.
                force = 1 if (args.auto_fire and t < 5 and env.num_actions > 1) else -1
                action = brain.act(frame, deterministic=deterministic,
                                   epsilon=eps, force_action=force)
                action_counts[action] += 1
                frame, r, done = eval_env.step(action)
                ep_r += r
                steps += 1
                if done:
                    break
            eval_rewards.append(ep_r)
            action_hist_total += action_counts
            print(f"    eval ep {ep+1}: reward={ep_r:+6.1f}  steps={steps}  "
                  f"actions={action_counts.tolist()}")
        er = np.array(eval_rewards)
        summary[mode] = (er, action_hist_total)
    eval_env.close()
    env.close()

    print(f"\n  EVAL SUMMARY")
    print(f"  {'-'*40}")
    for mode, (er, ahist) in summary.items():
        print(f"  [{mode:6s}] mean={er.mean():+.2f} std={er.std():.2f} "
              f"min={er.min():+.1f} max={er.max():+.1f}  "
              f"action_dist={ahist.tolist()}")
    print(f"  checkpoint:   {ckpt_path}")
    print(f"  history:      {hist_path}\n")


if __name__ == "__main__":
    main()
