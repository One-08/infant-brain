#!/usr/bin/env python3
"""
Prototype: camera or video file → Infant Brain → discrete actions (behaviors).

Install streaming dependency:
    pip install opencv-python-headless
    # or: pip install -e ".[streaming]"

Examples:
    # Inference only (no training): show actions from current policy
    python scripts/stream_act_demo.py --camera 0 --frames 120 --no-train

    # Short online training on motion reward, then run
    python scripts/stream_act_demo.py --video /path/to/clip.mp4 --episodes 20 --frames 200

    # Verbose: print action every frame
    python scripts/stream_act_demo.py --camera 0 --frames 60 --no-train -v
"""

import argparse
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from infant_brain.brain import Brain
from infant_brain.wrappers.streaming import OpenCVSource, StreamVisualEnv, brain_obs_size

# Map abstract actions to prototype “behaviors” (wire these to your robot/UI).
ACTION_HINTS = {
    0: "behavior_0 (e.g. idle / observe)",
    1: "behavior_1 (e.g. track motion)",
    2: "behavior_2 (e.g. alert / flag frame)",
    3: "behavior_3 (e.g. secondary mode)",
}


def _open_source(args) -> OpenCVSource:
    if args.video:
        return OpenCVSource(video_path=args.video)
    return OpenCVSource(camera=0 if args.camera is None else args.camera)


def main():
    p = argparse.ArgumentParser(description="Stream video → Brain → actions")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", type=int, default=None, help="Webcam index (e.g. 0)")
    src.add_argument("--video", type=str, default=None, help="Path to video file")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-train", action="store_true", help="Skip training; run policy as-is")
    p.add_argument("--episodes", type=int, default=15, help="Training episodes (if not --no-train)")
    p.add_argument("--max-train-steps", type=int, default=32, help="Steps per training episode")
    p.add_argument("--frames", type=int, default=150, help="Inference frames after training")
    p.add_argument("--every", type=int, default=5, help="Print action every N frames (unless -v)")
    p.add_argument("-v", "--verbose", action="store_true", help="Print action every frame")
    p.add_argument(
        "--motion-scale",
        type=float,
        default=10.0,
        help="Scale for motion-based reward during training (0 = disable)",
    )
    args = p.parse_args()

    try:
        source = _open_source(args)
    except ImportError as e:
        print(e)
        sys.exit(1)

    env = StreamVisualEnv(
        source,
        num_actions=4,
        motion_reward_scale=args.motion_scale,
    )
    brain = Brain(env, latent_dim=64, hidden_dim=256, n_concepts=6, device=args.device)

    h = brain_obs_size()
    print(f"Observation size: {h}x{h} (matches WorldModel on this device)")
    print("Action mapping (customize in script):", ACTION_HINTS)

    if not args.no_train:
        print(f"\nOnline training: {args.episodes} episodes × up to {args.max_train_steps} steps ...")
        brain.train(
            episodes=args.episodes,
            max_steps=args.max_train_steps,
            batch_size=32,
            buffer_size=10_000,
            cluster_every=50,
            sleep_every=100,
            check_every=50,
            seed=42,
        )
        # train() closes env; reopen stream and reload weights for inference
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            ckpt_path = tmp.name
        brain.save(ckpt_path)
        try:
            source2 = _open_source(args)
            env = StreamVisualEnv(
                source2,
                num_actions=4,
                motion_reward_scale=args.motion_scale,
            )
            brain = Brain(env, latent_dim=64, hidden_dim=256, n_concepts=6, device=args.device)
            brain.load(ckpt_path)
        finally:
            os.unlink(ckpt_path)
    else:
        print("\nSkipping training (--no-train). Actions reflect an untrained / prior policy.")

    print(f"\nStreaming inference (~{args.frames} frames). Ctrl+C to stop.\n")
    frame = env.reset(seed=0)
    t0 = time.time()
    try:
        for i in range(args.frames):
            action = brain.act(frame, deterministic=True, epsilon=0.05)
            if args.verbose or (i % max(1, args.every) == 0):
                hint = ACTION_HINTS.get(action, str(action))
                print(f"  frame {i:5d}  action={action}  {hint}")
            frame, _rew, done = env.step(action)
            if done:
                print("  End of stream or read failure; resetting env.")
                frame = env.reset(seed=i)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
