"""Advanced-CoLight agent; needs a running CityFlow, hence `slow`.

Guards the contract between `get_ob()` and `_build_model()`: row layout and
width. A width mismatch fails loudly (`nn.Linear` shape error), but blocks
swapped INSIDE the row do not: the network learns anyway, only worse.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsc_marl.domain import ats

pytestmark = pytest.mark.slow

ACTION_INTERVAL = 15


def test_ob_length_matches_the_network(advanced_colight_world):
    """Row width == input of the model's first layer."""
    _, model = advanced_colight_world
    structs = model._ats_structs()
    n_mov = max(len(s.movements) for s in structs)

    assert model.ob_length == 2 * n_mov + model.action_space.n
    assert model.ob_length == 20  # JiNan: 8 pressure + 8 running + 4 phases
    assert model.model.embedding_MLP.embedding_node[0].in_features == model.ob_length
    assert model.target_model.embedding_MLP.embedding_node[0].in_features == model.ob_length


def test_observation_shape_and_dtype(advanced_colight_world):
    world, model = advanced_colight_world
    ob = model.get_ob()
    assert ob.shape == (len(world.intersections), model.ob_length)
    assert ob.dtype == np.float32


def test_one_hot_shows_the_current_phase_of_the_right_intersection(advanced_colight_world):
    """Two contracts: (1) exactly one 1 in the phase block, (2) row i describes
    `world.intersections[i]`, the order the neighbourhood graph is indexed in."""
    world, model = advanced_colight_world
    n_mov = max(len(s.movements) for s in model._ats_structs())

    # different phases on different intersections, so the test cannot pass by chance
    actions = [i % len(inter.phases) for i, inter in enumerate(world.intersections)]
    for _ in range(ACTION_INTERVAL):
        world.step(actions)

    ob = model.get_ob()
    phase_block = ob[:, 2 * n_mov :]
    assert np.array_equal(phase_block.sum(axis=1), np.ones(len(world.intersections)))
    assert len(set(np.argmax(phase_block, axis=1).tolist())) > 1, "all phases identical"
    for idx, inter in enumerate(world.intersections):
        assert np.argmax(phase_block[idx]) == inter.current_phase, inter.id


def test_row_blocks_match_the_ats_primitives(advanced_colight_world):
    """Independent recomputation of e(l,m) and r(l,m) from `ats`: catches swapped
    blocks, off-by-one shifts and pressure confused with demand."""
    world, model = advanced_colight_world
    n_mov = max(len(s.movements) for s in model._ats_structs())
    obs_length = 167  # as in the config

    for _ in range(ACTION_INTERVAL):
        world.step([0] * len(world.intersections))
    ob = model.get_ob()

    eng = world.eng
    lane_vehicles, distances, speeds = (
        eng.get_lane_vehicles(),
        eng.get_vehicle_distance(),
        eng.get_vehicle_speed(),
    )
    for idx, s in enumerate(model._ats_structs()):
        q = ats.windowed_queue(
            s.all_lanes, lane_vehicles, distances, speeds, world.lane_length, obs_length
        )
        running = ats.effective_running(
            s.in_lanes, lane_vehicles, distances, speeds, world.lane_length, s.effective_range
        )
        expected_p = ats.movement_pressure(s.movements, q)
        expected_r = ats.movement_running(s.movements, running)

        assert ob[idx, : len(expected_p)] == pytest.approx(expected_p), idx
        assert ob[idx, n_mov : n_mov + len(expected_r)] == pytest.approx(expected_r), idx


def test_demand_non_negative_and_pressure_sometimes_negative(advanced_colight_world):
    """Vehicle counts cannot be negative, pressure is a difference: if it never
    goes below zero, the downstream term is lost and spillback is invisible."""
    world, model = advanced_colight_world
    n_mov = max(len(s.movements) for s in model._ats_structs())

    seen_negative = False
    for _ in range(20):
        for _ in range(ACTION_INTERVAL):
            world.step([1] * len(world.intersections))
        ob = model.get_ob()
        assert (ob[:, n_mov : 2 * n_mov] >= 0).all()
        seen_negative |= bool((ob[:, :n_mov] < 0).any())
    assert seen_negative, "pressure never negative, check the downstream term"
