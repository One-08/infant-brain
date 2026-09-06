#!/usr/bin/env python3
"""Train and run Infant Brain on support-ticket triage."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from infant_brain.brain import Brain
from infant_brain.wrappers.support_tickets import ACTION_NAMES, SafeTicketExecutor, SupportTicketEnv


def run_demo(episodes: int, dry_run: bool, device: str, verbose: bool):
    env = SupportTicketEnv()
    brain = Brain(env, latent_dim=64, hidden_dim=256, n_concepts=6, device=device)
    history = brain.train(
        episodes=episodes,
        max_steps=16,
        batch_size=32,
        buffer_size=5000,
        cluster_every=10,
        sleep_every=20,
        check_every=10,
        seed=42,
    )

    # Use the trained policy in deterministic mode for repeatable triage.
    exec_env = SupportTicketEnv()
    frame = exec_env.reset(seed=123)
    executor = SafeTicketExecutor(dry_run=dry_run)
    if verbose:
        print(
            "\nNote: the policy is trained on a compact encoding of category / priority / "
            "customer tier (rendered as a small tensor). Ticket text is shown here for you; "
            "to learn from wording, add text embeddings and feed them into the env state.\n"
        )
    print("\nTriage run:")
    for _ in range(64):
        ticket = exec_env.current_ticket
        if ticket is None:
            break
        action = brain.act(frame, deterministic=True, epsilon=0.0)
        result = executor.execute(ticket, action)
        line = (
            f"- {ticket.id} [{ticket.category}/{ticket.priority}] "
            f"-> {ACTION_NAMES.get(action, 'unknown')} | {result}"
        )
        if verbose:
            text = ticket.text.strip().replace("\n", " ")
            if len(text) > 120:
                text = text[:117] + "..."
            print(f"  Text: {text}")
        print(line)
        frame, _, done = exec_env.step(action)
        if done:
            break

    print("\nDone. Last training reward:", history["reward"][-1] if history.get("reward") else "n/a")


def main():
    parser = argparse.ArgumentParser(description="Support ticket triage demo with Infant Brain")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--execute", action="store_true", help="Execute actions (still simulated in this demo)")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print each ticket's subject/body text before the decision line",
    )
    args = parser.parse_args()

    run_demo(
        episodes=args.episodes,
        dry_run=not args.execute,
        device=args.device,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
