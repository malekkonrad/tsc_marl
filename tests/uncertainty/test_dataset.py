"""Behaviour-cloning dataset (state -> expert action).

The key check is `expert_check`, run on EVERY dump: one number checks feature
order, phase incidence, the `rank -> node_idx` renumbering and that labels come
from the clean state. These tests make sure the number means what it should and
that the failure layer does not leak into the expert's trajectory.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsc_marl.domain import ats
from tsc_marl.domain.sensors import FailureSpec
from tsc_marl.uncertainty import dataset

W1 = 1.0


def make_x(ep, demand, phase, n_phases=2):
    """A state row in `StateBuilder` layout: EP block, demand block, phase one-hot."""
    row = np.zeros(len(ep) + len(demand) + n_phases, dtype=np.float32)
    row[: len(ep)] = ep
    row[len(ep) : len(ep) + len(demand)] = demand
    row[len(ep) + len(demand) + phase] = 1.0
    return row


IDENTITY = np.eye(2, dtype=bool)[None]  # one intersection, phase k serves movement k


def check(rows, y):
    x = np.stack(rows)[:, None, :]  # [T, 1, F]
    return dataset.expert_check(x, np.array(y, dtype=np.int64)[:, None], IDENTITY, W1, 2)


# ------------------------------------------------------------- expert_check


def test_reproduces_argmax_pressure():
    """Without the demand latch Alg 1 is a plain argmax of p(s)."""
    r = check([make_x([1.0, 5.0], [0.0, 0.0], 0), make_x([7.0, 2.0], [0.0, 0.0], 1)], [1, 0])
    assert r["agreement"] == 1.0
    assert r["tie_rate"] == 0.0


def test_reproduces_the_demand_latch():
    """`d(a_cur) * W1 > max p(s)` keeps the CURRENT phase though it does not have
    the largest pressure: the whole difference between Advanced-MP and Efficient-MP."""
    row = make_x([1.0, 5.0], [9.0, 0.0], phase=0)
    assert check([row], [0])["agreement"] == 1.0  # keeps phase 0
    assert check([row], [1])["agreement"] == 0.0  # argmax would give 1


def test_a_tie_does_not_count_towards_agreement():
    """With equal p(s) summation order decides, not the state: information nobody
    has. It goes to `tie_rate` and out of the agreement."""
    r = check([make_x([3.0, 3.0], [0.0, 0.0], 0)], [1])
    assert r["tie_rate"] == 1.0
    assert np.isnan(r["agreement_decidable"])  # no decidable samples
    assert r["margin"][0, 0] == pytest.approx(0.0)


def test_the_latch_boundary_is_excluded_too():
    """`d(a_cur) * W1 == max p(s)`: the condition is STRICT, so the last bit decides."""
    r = check([make_x([4.0, 1.0], [4.0, 0.0], 0)], [0])
    assert r["boundary_rate"] == 1.0


def test_margin_measures_ambivalence():
    """`tie_margin` is a measured variable: DECISION uncertainty can be mistaken
    for genuine phase ambivalence."""
    r = check([make_x([5.0, 1.0], [0.0, 0.0], 0), make_x([5.0, 4.9], [0.0, 0.0], 0)], [0, 0])
    assert r["margin"][0, 0] == pytest.approx(4.0)
    assert r["margin"][1, 0] == pytest.approx(0.1, abs=1e-5)


# ------------------------------------------------------------------ incidence


def test_incidence_covers_exactly_the_phase_movements(roadnet):
    """Def 11 -> Def 5/10. Indices point into `controlled_movements`, the list that
    fixes the column order of the state vector."""
    for inter, _ in roadnet:
        flat = ats.controlled_movements(inter)
        per_phase = ats.phase_movements(inter)
        incidence = ats.phase_incidence(inter)
        assert len(incidence) == len(per_phase)
        for phase, idx in enumerate(incidence):
            assert len(idx) == len(per_phase[phase])
            assert len(set(idx)) == len(idx), "movement counted twice in a phase"
            assert all(0 <= i < len(flat) for i in idx)
            # the same movements, as indices into the flat list
            assert {tuple(map(tuple, flat[i])) for i in idx} == {
                tuple(map(tuple, m)) for m in per_phase[phase]
            }


def test_incidence_matches_phase_pressure(roadnet):
    """p(s) via the incidence from the EP block equals `ats.phase_pressure` from the
    queues; otherwise the expert reconstruction would measure its own error."""
    rng = np.random.default_rng(0)
    for inter, _ in roadnet:
        flat = ats.controlled_movements(inter)
        lanes = sorted(ats.all_lanes(flat))
        q = dict(zip(lanes, rng.integers(0, 8, len(lanes)).tolist(), strict=True))
        ep = np.array(ats.movement_pressure(flat, q))
        direct = ats.phase_pressure(ats.phase_movements(inter), q)
        via_incidence = [ep[idx].sum() for idx in ats.phase_incidence(inter)]
        assert np.allclose(direct, via_incidence)


# --------------------------------------------------------------- integration


@pytest.fixture(scope="module")
def dumped():
    """A short dump on a real CityFlow: 2 episodes x 30 decisions (~20 s); two,
    because the key integration test compares them."""
    return dataset.collect(episodes=2, decisions=30)


@pytest.mark.slow
def test_the_bc_task_is_realizable(dumped):
    """Advanced-MP decides from EXACTLY the information in the ATS state; agreement
    below 1.0 on decidable samples would mean a shifted column or a wrong renumbering."""
    assert dumped.meta["agreement_decidable"] == 1.0


@pytest.mark.slow
def test_failure_touches_neither_the_expert_nor_the_trajectory(dumped):
    """The expert reads the engine and the failure layer lives in `StateBuilder`,
    so a scenario CANNOT change the simulation; episodes differ only in the failure
    draw, so identical `y` and `x_clean` prove the separation."""
    d = dumped.decisions if hasattr(dumped, "decisions") else dumped.meta["decisions_per_episode"]
    assert np.array_equal(dumped.y[:d], dumped.y[d : 2 * d])
    assert np.array_equal(dumped.x_clean[:d], dumped.x_clean[d : 2 * d])
    assert dumped.meta["att_per_episode"][0] == dumped.meta["att_per_episode"][1]


@pytest.mark.slow
def test_scenarios_differ_between_episodes(dumped):
    """Otherwise replication over failure draws would have nothing to average."""
    d = dumped.meta["decisions_per_episode"]
    dead = dumped.dead["mask0p25"]
    assert not np.array_equal(dead[0], dead[d]), "the same set of dead detectors"


@pytest.mark.slow
def test_the_clean_scenario_is_clean(dumped):
    """Reference row: `FailureSpec()` must not touch anything."""
    assert np.array_equal(dumped.x[dataset.CLEAN], dumped.x_clean)
    assert dumped.meta["changed_cells"][dataset.CLEAN] == 0.0
    assert dumped.meta["dead_movements"][dataset.CLEAN] == 0.0


@pytest.mark.slow
def test_dead_movements_are_not_changed_features(dumped):
    """A silent failure: a masked detector on an empty lane reports the TRUE 0, it
    is dead and the feature does not change; localisation ground truth comes from
    the layer, not from feature differences."""
    assert dumped.meta["dead_movements"]["mask0p25"] > 3 * dumped.meta["changed_cells"]["mask0p25"]
    assert dumped.meta["dead_movements"]["gauss3p5"] == 0.0  # noise kills nothing
    assert dumped.meta["changed_cells"]["gauss3p5"] > 0.5


@pytest.mark.slow
def test_shapes_and_saving(dumped, tmp_path):
    n, f = dumped.meta["n_nodes"], dumped.x_clean.shape[-1]
    t = dumped.meta["decisions_per_episode"] * dumped.meta["episodes"]
    assert dumped.x_clean.shape == (t, n, f)
    assert dumped.y.shape == dumped.phase_time.shape == dumped.tie_margin.shape == (t, n)
    assert dumped.incidence.shape[:2] == (n, dumped.incidence.shape[1])
    assert dumped.edge_index.shape[0] == 2

    loaded = np.load(dumped.save(tmp_path / "d.npz"))
    assert set(loaded.files) >= {"x_clean", "y", "incidence", "tie_margin", "edge_index"}
    assert np.array_equal(loaded["x__mask0p25"], dumped.x["mask0p25"])
    assert np.array_equal(loaded["dead__mask0p25"], dumped.dead["mask0p25"])


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (FailureSpec(), "clean"),
        (FailureSpec(kind="gauss", level=3.5), "gauss3p5"),
        (FailureSpec(kind="mask", level=0.25, unit="intersection"), "mask0p25_intersection"),
    ],
)
def test_slug_works_as_an_npz_key(spec, expected):
    """npz keys become file names inside the archive, so no dots."""
    assert spec.slug() == expected
    assert "." not in spec.slug()
