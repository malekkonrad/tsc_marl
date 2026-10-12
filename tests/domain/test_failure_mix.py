"""Failure randomization in training (`domain.sensors.FailureMix`).

The schedule of failures must depend only on (mix seed, episode number): not on
the policy's `--seed` and not on global RNG state. Otherwise two training arms
would see different failures and their difference would mix the mechanism with
another draw, silently.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from tsc_marl.domain.sensors import FailureMix, FailureSpec

MIX = {
    "seed": 0,
    "scenarios": [
        {"weight": 4, "kind": "none"},
        {"weight": 2, "kind": "mask", "level": 0.25},
        {"weight": 1, "kind": "gauss", "level": 2.0},
    ],
}


def mix() -> FailureMix:
    return FailureMix.from_config(MIX)


# ------------------------------------------------------------------- sampling


def test_scenario_depends_only_on_the_episode_number():
    """A run resumed at episode 137 gets the same scenario as a continuous one,
    and two arms get the same schedule."""
    m = mix()
    assert [m.sample(e) for e in range(30)] == [m.sample(e) for e in range(30)]
    assert m.sample(137) == m.sample(137)


def test_sampling_ignores_the_global_rng():
    """`--seed` drives network initialisation through `set_global_seed`; the
    failure schedule must not come from the same source."""
    m = mix()
    random.seed(1234)
    np.random.seed(1234)
    first = [m.sample(e) for e in range(20)]
    random.seed(9)
    np.random.seed(9)
    assert [m.sample(e) for e in range(20)] == first


def test_mix_seed_changes_the_schedule():
    a = FailureMix.from_config(MIX)
    b = FailureMix.from_config({**MIX, "seed": 1})
    assert [a.sample(e) for e in range(40)] != [b.sample(e) for e in range(40)]


def test_weights_are_respected():
    """4:2:1, checked to an order of magnitude: weights work at all."""
    m = mix()
    samples = [m.sample(e) for e in range(3000)]
    clean = sum(1 for s in samples if not s.active)
    assert m.probabilities.tolist() == pytest.approx([4 / 7, 2 / 7, 1 / 7])
    assert 0.5 < clean / len(samples) < 0.65


def test_mix_contains_clean_and_failed_episodes():
    m = mix()
    samples = {m.sample(e) for e in range(200)}
    assert any(not s.active for s in samples)
    assert any(s.active for s in samples)


# ------------------------------------------------------------------- parsing


def test_no_mix_is_none():
    """A run without `failure_mix` is exactly the plain core: None, not a mix with
    one clean scenario (which would rebuild the `StateBuilder`)."""
    assert FailureMix.from_config(None) is None
    assert FailureMix.from_config({}) is None


def test_typo_in_a_scenario_fails_at_construction():
    with pytest.raises(TypeError):
        FailureMix.from_config({"scenarios": [{"kind": "mask", "lvl": 0.25}]})
    with pytest.raises(ValueError, match="kind"):
        FailureMix.from_config({"scenarios": [{"kind": "maska", "level": 0.25}]})


def test_rejects_nonsense_weights():
    with pytest.raises(ValueError, match="non-negative"):
        FailureMix((FailureSpec(),), (-1.0,))
    with pytest.raises(ValueError, match="positive"):
        FailureMix((FailureSpec(),), (0.0,))
    with pytest.raises(ValueError, match="empty"):
        FailureMix((), ())


def test_as_dict_describes_the_whole_schedule():
    """Otherwise a run directory would not show what the policy trained on."""
    d = mix().as_dict()
    assert d["seed"] == 0
    assert [s["weight"] for s in d["scenarios"]] == [4.0, 2.0, 1.0]
    assert [s["kind"] for s in d["scenarios"]] == ["none", "mask", "gauss"]


def test_describe_uses_scenario_slugs():
    assert mix().describe() == "clean=57%, mask0p25=29%, gauss2=14%"
