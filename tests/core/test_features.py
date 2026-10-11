"""State and reward of the core (`features.StateBuilder`).

No bug in this module raises: a shifted block, pressure confused with EP, a
reward over the wrong lane set all give a network that learns, only worse.
Hence independent recomputation and comparison with upstream where possible.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsc_marl.core.features import REWARD_VARIANTS, STATE_VARIANTS, StateBuilder
from tsc_marl.domain import ats

ACTION_INTERVAL = 15


def test_rejects_unknown_variant():
    """Validated before touching the world: a typo in YAML fails at construction."""
    with pytest.raises(ValueError, match="state="):
        StateBuilder(None, [], state="ATS")  # case-sensitive
    with pytest.raises(ValueError, match="reward="):
        StateBuilder(None, [], reward="delay")


@pytest.mark.slow
def test_row_width_per_variant(advanced_colight_world):
    """Only ATS has two feature blocks (pressure and demand); the rest have one."""
    world, _ = advanced_colight_world
    for variant in STATE_VARIANTS:
        b = StateBuilder.from_registry(world, world.intersections, state=variant)
        expected = (2 if variant == "ats" else 1) * b.n_mov + b.n_phases
        assert b.ob_length == expected, variant
        assert b.observe().shape == (len(world.intersections), expected), variant
        assert b.observe().dtype == np.float32, variant


@pytest.mark.slow
def test_one_hot_is_always_at_the_end_of_the_row(advanced_colight_world):
    """The phase block is the last `n_phases` columns in every variant: blocks are
    zero-padded separately, so padding never shifts the one-hot."""
    world, _ = advanced_colight_world
    actions = [i % len(inter.phases) for i, inter in enumerate(world.intersections)]
    for _ in range(ACTION_INTERVAL):
        world.step(actions)

    for variant in STATE_VARIANTS:
        b = StateBuilder.from_registry(world, world.intersections, state=variant)
        phase_block = b.observe()[:, -b.n_phases :]
        assert np.array_equal(phase_block.sum(axis=1), np.ones(b.n_nodes)), variant
        assert len(set(np.argmax(phase_block, axis=1).tolist())) > 1, "all phases identical"
        for idx, inter in enumerate(world.intersections):
            assert np.argmax(phase_block[idx]) == inter.current_phase, (variant, inter.id)


@pytest.mark.slow
def test_ep_is_the_first_block_of_ats(advanced_colight_world):
    """ATS = EP + demand, so the first block must be identical; otherwise one
    variant computes pressure differently from the other."""
    world, _ = advanced_colight_world
    for _ in range(ACTION_INTERVAL):
        world.step([0] * len(world.intersections))

    ep = StateBuilder.from_registry(world, world.intersections, state="ep")
    full = StateBuilder.from_registry(world, world.intersections, state="ats")
    assert ep.observe()[:, : ep.n_mov] == pytest.approx(full.observe()[:, : full.n_mov])


@pytest.mark.slow
def test_pressure_is_unnormalised_ep(advanced_colight_world):
    """The only difference between pressure and EP is normalisation by lane
    count (Wu et al., 2021). Recomputed from `ats`, not from the other variant."""
    world, _ = advanced_colight_world
    for _ in range(ACTION_INTERVAL):
        world.step([0] * len(world.intersections))

    b = StateBuilder.from_registry(world, world.intersections, state="pressure")
    rows = b.observe()
    q = world.get_info("lane_waiting_count")
    for idx, s in enumerate(b.structs):
        # `obs_length` is set in the config, so the queue is windowed; it is taken
        # from the builder so the test checks the variant, not the window.
        expected = ats.movement_pressure(
            s.movements, b._queue(s, b._snapshot(), q), normalize=False
        )
        assert rows[idx, : len(expected)] == pytest.approx(expected), idx


@pytest.mark.slow
def test_queue_reward_matches_colight(advanced_colight_world):
    """The strongest test here: our reward must equal upstream's
    `LaneVehicleGenerator(average="all", negative=True) * 12`, otherwise the core
    optimises something else than the baseline it is compared with. 12 is the
    number of in lanes (4 roads x 3 lanes), so mean * 12 is the sum."""
    world, model = advanced_colight_world
    for _ in range(ACTION_INTERVAL):
        world.step([2] * len(world.intersections))

    b = StateBuilder.from_registry(world, world.intersections, reward="queue")
    assert all(len(s.reward_lanes) == 12 for s in b.structs), "upstream constant 12 does not fit"
    assert b.rewards() == pytest.approx(model.get_reward(), abs=1e-4)


@pytest.mark.slow
def test_pressure_reward_is_non_positive(advanced_colight_world):
    """The PressLight variant penalises imbalance (abs), so it never rewards."""
    world, _ = advanced_colight_world
    b = StateBuilder.from_registry(world, world.intersections, reward="pressure")
    for _ in range(3):
        for _ in range(ACTION_INTERVAL):
            world.step([1] * len(world.intersections))
        assert (b.rewards() <= 0).all()
    assert set(REWARD_VARIANTS) == {"queue", "pressure"}


@pytest.mark.slow
def test_row_order_follows_the_given_list(advanced_colight_world):
    """Order is an input of the builder: the core passes
    `graph.order_intersections()`, the agent `world.intersections`."""
    world, _ = advanced_colight_world
    for _ in range(ACTION_INTERVAL):
        world.step([0] * len(world.intersections))

    forward = StateBuilder.from_registry(world, world.intersections)
    backward = StateBuilder.from_registry(world, list(reversed(world.intersections)))
    assert backward.observe() == pytest.approx(forward.observe()[::-1])
    assert backward.rewards() == pytest.approx(forward.rewards()[::-1])


@pytest.mark.slow
def test_demand_non_negative_and_pressure_sometimes_negative(advanced_colight_world):
    """Vehicle counts cannot be negative, while pressure is a difference: if it
    never goes below zero, the downstream term is lost and spillback is invisible."""
    world, _ = advanced_colight_world
    b = StateBuilder.from_registry(world, world.intersections, state="ats")

    seen_negative = False
    for _ in range(20):
        for _ in range(ACTION_INTERVAL):
            world.step([1] * len(world.intersections))
        rows = b.observe()
        assert (rows[:, b.n_mov : 2 * b.n_mov] >= 0).all()
        seen_negative |= bool((rows[:, : b.n_mov] < 0).any())
    assert seen_negative, "pressure never negative, check the downstream term"
