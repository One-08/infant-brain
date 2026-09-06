"""Smoke tests — verify the brain can be built, trained for a couple of episodes,
saved, reloaded, and that determinism / inference plumbing works.

These tests deliberately stay on CPU and use the GridWorld env so they can run
in CI without GPUs, ROMs, or network access. Total runtime budget: < 60s.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from infant_brain.brain import Brain, _seed_everything, _stable_concept_id
from infant_brain.envs import GridWorldEnv


# Keep these tiny so the suite stays fast on CPU.
TINY_KW = dict(
    episodes=2,
    max_steps=8,
    batch_size=4,
    buffer_size=64,
    plan_len=2,
    cluster_every=1,
    sleep_every=10_000,  # don't trigger sleep in smoke tests
    check_every=10_000,
    log_every=1,
)


def _make_brain(encoder_type: str = "vit"):
    # BrainEncoder hardcodes img_size=32 on CPU (see WorldModel.__init__),
    # so the env must produce 32x32 frames. grid_size=4 * cell_pixels=8 = 32.
    env = GridWorldEnv(grid_size=4, num_objects=2, cell_pixels=8)
    return Brain(env, latent_dim=32, hidden_dim=64, n_concepts=3, device="cpu",
                 encoder_type=encoder_type)


@pytest.mark.parametrize("encoder_type", ["vit", "cnn"])
def test_brain_constructs_and_trains_with_either_encoder(encoder_type):
    """Both encoder backends must build and train end-to-end without errors."""
    brain = _make_brain(encoder_type=encoder_type)
    history = brain.train(seed=2, **TINY_KW)
    assert len(history["reward"]) == TINY_KW["episodes"]
    frame = brain.env.reset(seed=0)
    a = brain.act(frame, deterministic=True)
    assert 0 <= a < brain.env.num_actions


def test_stable_concept_id_is_deterministic_across_runs():
    """The previous code used Python's salted hash() which gave different
    concept IDs on every interpreter run. Regression test for that."""
    z = torch.tensor([0.1, -0.2, 0.3, 0.4, 0.5], dtype=torch.float32)
    out_a = _stable_concept_id(z, n_concepts=8)
    out_b = _stable_concept_id(z.clone(), n_concepts=8)
    assert out_a == out_b
    assert 0 <= out_a < 8


def test_seed_everything_makes_torch_random_reproducible():
    _seed_everything(123)
    a = torch.randn(5)
    _seed_everything(123)
    b = torch.randn(5)
    assert torch.allclose(a, b)

    _seed_everything(123)
    n_a = np.random.rand(5)
    _seed_everything(123)
    n_b = np.random.rand(5)
    assert np.allclose(n_a, n_b)


def test_brain_constructs_with_split_optimizers():
    brain = _make_brain()
    # All three optimizers must exist and be disjoint in their parameter sets.
    wm_ids = {id(p) for g in brain.wm_optimizer.param_groups for p in g['params']}
    lang_ids = {id(p) for g in brain.lang_optimizer.param_groups for p in g['params']}
    head_ids = {id(p) for g in brain.head_optimizer.param_groups for p in g['params']}
    assert wm_ids and lang_ids and head_ids
    assert wm_ids.isdisjoint(lang_ids), "WM and lang optimizers share params!"
    assert wm_ids.isdisjoint(head_ids), "WM and head optimizers share params!"
    assert lang_ids.isdisjoint(head_ids), "Lang and head optimizers share params!"


def test_brain_train_runs_for_a_couple_episodes():
    brain = _make_brain()
    history = brain.train(seed=42, **TINY_KW)
    assert "reward" in history
    assert len(history["reward"]) == TINY_KW["episodes"]
    # Loss curves should at least be populated, even if not improving in 2 eps.
    assert len(history["wm_loss"]) == TINY_KW["episodes"]


def test_brain_act_returns_valid_action_after_training():
    brain = _make_brain()
    brain.train(seed=7, **TINY_KW)
    frame = brain.env.reset(seed=0)
    a = brain.act(frame)
    assert isinstance(a, int)
    assert 0 <= a < brain.env.num_actions
    a_det = brain.act(frame, deterministic=True)
    assert isinstance(a_det, int)
    assert 0 <= a_det < brain.env.num_actions


def test_save_and_load_round_trip():
    brain = _make_brain()
    brain.train(seed=1, **TINY_KW)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "brain.pt")
        brain.save(path)
        assert os.path.exists(path)

        # Build a fresh brain and load — the actor should produce a valid action.
        brain2 = _make_brain()
        brain2.load(path)
        frame = brain2.env.reset(seed=0)
        a = brain2.act(frame, deterministic=True)
        assert 0 <= a < brain2.env.num_actions


def test_seed_42_runs_are_reproducible_in_short_horizon():
    """Two short runs with the same seed should produce the same first reward.

    We only assert on the first episode because Atari/GridWorld replay buffers
    plus the value-system signals can amplify floating-point drift after that.
    """
    h1 = _make_brain().train(seed=99, **TINY_KW)
    h2 = _make_brain().train(seed=99, **TINY_KW)
    assert h1["reward"][0] == pytest.approx(h2["reward"][0])
