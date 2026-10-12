"""Sensor failure layer (`domain.sensors`).

This layer is a measuring instrument: a bug in it shows up as a shifted
degradation curve, not an exception. The tests target what results cannot
show: (1) a disabled layer changes NOTHING, (2) the reward stays clean, (3) one
physical detector shows every reader the same value, (4) the ground truth
matches what was actually broken.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsc_marl.core.features import StateBuilder
from tsc_marl.domain.sensors import FailureSpec, SensorModel, failure_units

# 4 roads x 3 lanes, as on the JiNan/HangZhou intersections.
LANES = [f"road_{r}_{i}" for r in "NESW" for i in range(3)]
CLEAN = {lane: float(10 + i) for i, lane in enumerate(LANES)}


def model(**spec) -> SensorModel:
    s = FailureSpec(**spec)
    return SensorModel(s, failure_units([], LANES, s.unit))


def read(m: SensorModel, channel: str = "queue", lanes=None) -> dict[str, float]:
    return m.read(CLEAN, lanes or LANES, channel=channel)


# --------------------------------------------------------------------- spec


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"kind": "Gauss"}, "kind="),  # case-sensitive
        ({"unit": "detector"}, "unit="),
        ({"substitution": "nan"}, "substitution="),
        ({"kind": "gauss", "level": -1}, "level="),
        ({"kind": "mask", "level": 1.5}, "FRACTION"),
        ({"kind": "gauss", "level": 1, "unit": "road"}, "per reading"),
        ({"onset": -5}, "onset="),
    ],
)
def test_bad_spec_fails_at_construction(kwargs, match):
    """A YAML typo fails at once. `unit` with measurement noise would be ignored,
    so it is an error, not a silent no-op."""
    with pytest.raises(ValueError, match=match):
        FailureSpec(**kwargs)


def test_suffix_distinguishes_scenarios():
    """Otherwise two scenarios would share a run directory: the rest of the
    `run_id` (agent, network, seed, yellow) is identical."""
    seen = {
        FailureSpec(kind=k, level=v, unit=u, substitution=s, onset=t).suffix()
        for k, v, u, s, t in [
            ("gauss", 3.5, "lane", "zero", 0),
            ("gauss", 4.0, "lane", "zero", 0),
            ("urand", 3.5, "lane", "zero", 0),
            ("mask", 0.25, "lane", "zero", 0),
            ("mask", 0.25, "intersection", "zero", 0),
            ("mask", 0.25, "lane", "hold", 120),
            ("drift", 0.2, "road", "zero", 0),
        ]
    }
    assert len(seen) == 7
    assert FailureSpec().suffix() == ""  # a clean run has no suffix


def test_slug_names_clean_and_failed_scenarios():
    assert FailureSpec().slug() == "clean"
    assert FailureSpec(kind="gauss", level=0.0).slug() == "clean"  # inactive
    assert FailureSpec(kind="mask", level=0.5, substitution="hold", onset=60).slug() == (
        "mask0p5_hold_t60"
    )


# ------------------------------------------------------------ persistent damage


@pytest.mark.parametrize("level", [0.25, 0.5])
def test_mask_is_an_exact_fraction(level):
    """25% means 25%, not "25% on average"; otherwise severity would vary across
    draws and mix method effects with sampling noise."""
    m = model(kind="mask", level=level)
    assert len(m.broken) == round(level * len(LANES))


def test_mask_groups_by_failure_unit():
    """`unit='road'` is one camera per approach: either the whole road fails or
    none of its lanes does."""
    m = model(kind="mask", level=0.5, unit="road")
    roads = {lane.rsplit("_", 1)[0] for lane in m.broken}
    assert len(roads) == 2
    for road in roads:
        assert sum(1 for lane in m.broken if lane.startswith(road)) == 3


def test_dead_detector_is_dead_on_both_channels():
    """One physical sensor serves queue and moving vehicles; killing one channel
    only would be physically impossible and a free hint for the uncertainty head."""
    m = model(kind="mask", level=0.5)
    m.begin_decision()
    q, r = read(m, "queue"), read(m, "running")
    for lane in m.broken:
        assert q[lane] == 0.0 and r[lane] == 0.0
    for lane in set(LANES) - m.broken:
        assert q[lane] == CLEAN[lane] and r[lane] == CLEAN[lane]


def test_broken_detector_does_not_flicker():
    """Persistent damage lasts the whole episode."""
    m = model(kind="mask", level=0.25)
    broken = set(m.broken)
    for _ in range(240):  # decisions per episode at 15 s
        m.begin_decision()
        out = read(m)
        assert set(m.broken) == broken
        assert all(out[lane] == 0.0 for lane in broken)


def test_reset_redraws_failures_for_a_new_episode():
    """Replication under failure is over failure draws (traffic is fixed)."""
    m = model(kind="mask", level=0.25)
    patterns = set()
    for episode in range(10):
        m.reset(episode)
        patterns.add(m.broken)
    assert len(patterns) > 1


def test_failure_pattern_ignores_global_seeds():
    """Otherwise seeds would differ in policy initialisation AND in which
    detectors failed."""
    np.random.seed(0)
    a = model(kind="mask", level=0.25).broken
    np.random.seed(12345)
    np.random.random(1000)
    b = model(kind="mask", level=0.25).broken
    assert a == b


def test_drift_is_multiplicative_and_leaves_no_holes():
    """The unobservable scenario: values stay plausible at any load, no gaps and
    no variance jumps."""
    m = model(kind="drift", level=0.2)
    m.begin_decision()
    out = read(m)
    assert m.broken == frozenset()
    assert all(m.valid(LANES).values())
    ratios = {lane: out[lane] / CLEAN[lane] for lane in LANES}
    assert len(set(np.round(list(ratios.values()), 6))) > 1  # different detectors, different bias
    assert any(abs(r - 1.0) > 0.05 for r in ratios.values())
    m.begin_decision()  # the bias is persistent, not redrawn per reading
    assert read(m) == out


# ------------------------------------------------------------ transient damage


def test_noise_is_one_per_decision_and_shared_by_readers():
    """A's out lane is B's in lane, so one detector is read twice per decision.
    Two independent draws would let the network average the noise away."""
    m = model(kind="gauss", level=3.5)
    m.begin_decision()
    first = read(m)
    assert read(m) == first, "same detector, same decision, two values"
    assert read(m, lanes=LANES[:3]) == {k: first[k] for k in LANES[:3]}

    m.begin_decision()
    assert read(m) != first, "new decision, new reading"


def test_noise_is_independent_between_lanes():
    """One draw for the whole network would be a common shift that EP (a
    difference of means) would largely cancel."""
    m = model(kind="gauss", level=3.5)
    m.begin_decision()
    deltas = {lane: read(m)[lane] - CLEAN[lane] for lane in LANES}
    assert len(set(np.round(list(deltas.values()), 6))) > len(LANES) // 2


@pytest.mark.parametrize("kind", ["gauss", "urand"])
def test_severity_grows_with_level(kind):
    severity = []
    for level in (0.5, 1.0, 2.0, 4.0):
        m = model(kind=kind, level=level)
        diffs = []
        for _ in range(50):
            m.begin_decision()
            out = read(m)
            diffs += [abs(out[lane] - CLEAN[lane]) for lane in LANES]
        severity.append(float(np.mean(diffs)))
    assert severity == sorted(severity)


def test_count_is_never_negative():
    """A detector cannot report minus two cars."""
    m = model(kind="gauss", level=8.0)
    for _ in range(50):
        m.begin_decision()
        out = m.read(dict.fromkeys(LANES, 1.0), LANES, channel="queue")
        assert all(v >= 0.0 for v in out.values())


# --------------------------------------------------------------- substitution


def test_read_does_not_modify_its_input():
    """`values` may be the world's cache, from which the REWARD is computed."""
    m = model(kind="mask", level=0.5)
    m.begin_decision()
    before = dict(CLEAN)
    read(m)
    assert CLEAN == before


def test_hold_keeps_the_last_value_before_the_failure():
    """Harder than `zero`: no trace (an exact 0 for many decisions) a rule could catch."""
    m = model(kind="mask", level=0.5, substitution="hold", onset=3)
    for _ in range(3):  # decisions 0, 1, 2: before onset
        m.begin_decision()
        assert read(m) == CLEAN
    m.begin_decision()  # decision 3: failure
    out = read(m)
    assert m.broken
    for lane in m.broken:
        assert out[lane] == CLEAN[lane], "hold must keep the last healthy reading"


def test_hold_without_onset_degenerates_to_zero():
    """A detector dead from the first decision has nothing to hold."""
    m = model(kind="mask", level=0.5, substitution="hold")
    m.begin_decision()
    out = read(m)
    assert all(out[lane] == 0.0 for lane in m.broken)


def test_mean_substitutes_the_mean_of_live_detectors():
    m = model(kind="mask", level=0.5, substitution="mean")
    m.begin_decision()
    out = read(m)
    live = [out[lane] for lane in LANES if lane not in m.broken]
    assert out[next(iter(m.broken))] == pytest.approx(float(np.mean(live)))


def test_onset_starts_the_failure_at_runtime():
    m = model(kind="mask", level=0.5, onset=2)
    m.begin_decision()
    assert read(m) == CLEAN, "detectors are healthy before onset"
    assert not m.live
    m.begin_decision()
    assert not m.live
    m.begin_decision()
    assert m.live
    assert any(read(m)[lane] == 0.0 for lane in m.broken)


# --------------------------------------------------------------- ground truth


def test_report_records_what_actually_failed():
    """The spec alone is not enough: on another network it breaks other lanes."""
    m = model(kind="mask", level=0.25, unit="road")
    r = m.report()
    assert r["kind"] == "mask" and r["unit"] == "road"
    assert r["n_units"] == 4  # four roads, not twelve lanes
    assert r["n_broken"] == len(m.broken) == 3
    assert sorted(m.broken) == r["broken"]


def test_valid_sees_exactly_the_broken_lanes():
    """Read by oracles, never by the state vector, so `ob_length` stays the same."""
    m = model(kind="mask", level=0.25)
    valid = m.valid(LANES)
    assert {lane for lane, ok in valid.items() if not ok} == set(m.broken)


@pytest.mark.slow
@pytest.mark.parametrize("unit", ["lane", "road", "intersection"])
def test_failure_units_on_a_real_roadnet(advanced_colight_world, unit):
    """`intersection` assigns a lane to the intersection it leads into, which
    synthetic names cannot check. Slow, because `Intersection.in_roads` is filled
    only by `World.__init__`."""
    world, _ = advanced_colight_world
    intersections = list(world.intersections)
    lanes = sorted(
        f"{road['id']}_{i}"
        for inter in intersections
        for road in inter.in_roads
        for i in range(len(road["lanes"]))
    )
    groups = failure_units(intersections, lanes, unit)
    assert set(groups) == set(lanes)
    n_units = len(set(groups.values()))
    if unit == "lane":
        assert n_units == len(lanes)
    elif unit == "intersection":
        assert n_units == len(intersections)
    else:
        assert len(intersections) < n_units < len(lanes)


# ------------------------------------------------------- integration with the state


def warm_up(world, steps: int = 300) -> None:
    """CityFlow starts EMPTY, where every count is 0 and every failure is
    indistinguishable from a clean run; without warm-up these tests check nothing."""
    zeros = [0] * len(world.intersections)
    for _ in range(steps):
        world.step(zeros)


@pytest.mark.slow
def test_disabled_layer_changes_nothing(advanced_colight_world):
    """Bit-identical: the failure layer must not touch a clean run."""
    world, _ = advanced_colight_world
    warm_up(world)
    base = StateBuilder.from_registry(world, world.intersections, failure=None)
    spec = FailureSpec(kind="gauss", level=3.5)
    with_layer = StateBuilder.from_registry(world, world.intersections, failure=spec)
    assert np.array_equal(base.observe(), with_layer.observe(corrupt=False))


@pytest.mark.slow
def test_reward_stays_clean_under_every_failure(advanced_colight_world):
    """Failures affect only the policy input; `rewards()` reads the world's cache,
    not the observation path, and this guards it against refactoring."""
    world, _ = advanced_colight_world
    warm_up(world)
    clean = StateBuilder.from_registry(world, world.intersections, failure=None).rewards()
    assert clean.sum() < 0, "an empty network would make this test pointless"
    for spec in (
        FailureSpec(kind="gauss", level=4.0),
        FailureSpec(kind="mask", level=0.5),
        FailureSpec(kind="mask", level=1.0, unit="intersection"),
        FailureSpec(kind="drift", level=0.3, unit="road"),
    ):
        b = StateBuilder.from_registry(world, world.intersections, failure=spec)
        b.observe()  # the corrupted observation must not leak into the reward
        assert np.array_equal(b.rewards(), clean), spec


@pytest.mark.slow
@pytest.mark.parametrize(
    "spec",
    [
        FailureSpec(kind="gauss", level=3.5),
        FailureSpec(kind="mask", level=0.5),
        FailureSpec(kind="drift", level=0.3),
    ],
)
def test_failure_actually_changes_the_observation(advanced_colight_world, spec):
    """The other side of the previous test: the layer must DO something. Row width
    stays the same, because the validity flag is not part of the state."""
    world, _ = advanced_colight_world
    warm_up(world)
    b = StateBuilder.from_registry(world, world.intersections, failure=spec)
    dirty, clean = b.observe(), b.observe(corrupt=False)
    assert dirty.shape == clean.shape
    assert np.abs(clean[:, : -b.n_phases]).sum() > 0, "all zeros, nothing to corrupt"
    assert not np.array_equal(dirty, clean), spec
