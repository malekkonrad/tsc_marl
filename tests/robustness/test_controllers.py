"""Controllers under failure and evaluation plans.

The max-pressure controllers compute their decision from the STATE VECTOR rather
than the engine, which is the only reason their degradation curves mean anything.
The tests check the formulas (pressure, Advanced-MP hold rule, t_min latch) on
hand-built states, without CityFlow.
"""

from __future__ import annotations

import textwrap
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from runlog import params_hash
from tsc_marl.analysis.degradation import _sweep_section, mechanism_pairs, shared_checkpoint
from tsc_marl.core.features import StateBuilder, phase_mask
from tsc_marl.core.loop import CoreTrainer, LoopConfig
from tsc_marl.domain.sensors import FailureSpec
from tsc_marl.paths import DATA
from tsc_marl.robustness import controllers as c
from tsc_marl.robustness import degrade as d
from tsc_marl.runner.degrade import checkpoint, fixed_time_program, unit_params

# 2 intersections, 2 phases, 3 movements; phase 0 serves movements {0,1}, phase 1 {2}
INC = np.zeros((2, 2, 3), dtype=bool)
INC[:, 0, [0, 1]] = True
INC[:, 1, [2]] = True


def inter(phase=0, time=100, n_phases=2):
    return SimpleNamespace(
        current_phase=phase, current_phase_time=time, phases=list(range(n_phases))
    )


def obs(ep, demand=None, n_phases=2):
    """A state row: [EP (3) | demand (3) | phase one-hot]."""
    ep = np.asarray(ep, dtype=np.float32)
    demand = np.zeros_like(ep) if demand is None else np.asarray(demand, dtype=np.float32)
    return np.concatenate([ep, demand, np.zeros((ep.shape[0], n_phases), np.float32)], axis=1)


def test_efficient_mp_picks_the_phase_with_the_highest_pressure():
    ctrl = c.PressureController(INC, [inter(), inter()], 3, t_min=15)
    # node 0: p = [1+1, 5] -> phase 1; node 1: p = [3+3, 1] -> phase 0
    x = obs([[1, 1, 5], [3, 3, 1]])
    assert list(ctrl.act(x)) == [1, 0]


def test_minimum_green_latch():
    """A phase younger than t_min stays, whatever the state, as in the agent."""
    ctrl = c.PressureController(INC, [inter(phase=0, time=5), inter(phase=0, time=15)], 3, t_min=15)
    x = obs([[0, 0, 9], [0, 0, 9]])
    assert list(ctrl.act(x)) == [0, 1]


def test_advanced_mp_defends_the_current_phase_with_demand():
    """d(a_cur) * W1 > max p(s) -> hold; the inequality is STRICT."""
    ctrl = c.PressureController(INC, [inter(phase=0), inter(phase=0)], 3, t_min=15, w1=1.0)
    x = obs([[0, 0, 4], [0, 0, 4]], demand=[[3, 2, 0], [2, 2, 0]])
    # node 0: d(0)=5 > max p=4 -> holds 0; node 1: d(0)=4, NOT > 4 -> argmax p = 1
    assert list(ctrl.act(x)) == [0, 1]


def test_advanced_mp_requires_the_demand_block():
    ctrl = c.PressureController(INC, [inter(), inter()], 3, t_min=15, w1=1.0)
    with pytest.raises(ValueError, match="demand"):
        ctrl.act(np.zeros((2, 4), np.float32))


def test_a_broken_reading_changes_the_mp_decision():
    """Max-pressure under failure reads THE SAME broken state as the policy: a masked
    (zeroed) movement stops pulling green towards itself."""
    ctrl = c.PressureController(INC, [inter(), inter()], 3, t_min=15)
    truth = obs([[0, 0, 5], [0, 0, 5]])
    masked_obs = truth.copy()
    masked_obs[0, 2] = 0.0  # dead detector on movement 2 of intersection 0
    assert list(ctrl.act(truth)) == [1, 1]
    assert list(ctrl.act(masked_obs)) == [0, 1]


def test_fixedtime_cycles_after_t_fixed():
    ctrl = c.FixedTimeController([inter(phase=1, time=29), inter(phase=1, time=30)], t_fixed=30)
    assert list(ctrl.act(np.zeros((2, 8)))) == [1, 0]
    assert ctrl.state_independent


def test_colight_refuses_another_order():
    with pytest.raises(ValueError, match="order"):
        c.ColightController(object(), identity_order=False)


# ------------------------------------------------------------------ plan


def plan_file(tmp_path, extra=""):
    p = tmp_path / "trial.yml"
    p.write_text(
        textwrap.dedent(
            """
            networks: [net_a, net_b]
            draws: 3
            controllers:
              gatppo: {seeds: [0, 1]}
              efficientmp: {}
              fixedtime: {}
            scenarios:
              - {kind: gauss, level: 1.0}
              - {kind: mask, level: 0.5, substitution: hold, onset: 10}
            """
        )
        + extra
    )
    return p


def test_the_plan_prepends_the_clean_scenario(tmp_path):
    plan = d.load_plan(plan_file(tmp_path))
    assert [s.slug() for s in plan.scenarios] == ["clean", "gauss1", "mask0p5_hold_t10"]


def test_seeds_only_for_learned_controllers(tmp_path):
    """Rules are deterministic: N seeds would give N identical numbers."""
    plan = d.load_plan(plan_file(tmp_path))
    units = plan.units()
    assert sum(u.controller == "gatppo" for u in units) == 4  # 2 networks x 2 seeds
    assert sum(u.controller == "efficientmp" for u in units) == 2
    assert {u.seed for u in units if u.controller != "gatppo"} == {0}


def test_parent_and_run_id(tmp_path):
    plan = d.load_plan(plan_file(tmp_path))
    unit = d.Unit("gatppo", "net_a", 1)
    assert plan.parent_id(unit) == "gatppo__net_a__s1_y5"
    assert plan.parent_id(d.Unit("efficientmp", "net_a", 0)) is None
    assert plan.run_id(unit).endswith("__trial")


def test_fixedtime_only_on_clean(tmp_path):
    plan = d.load_plan(plan_file(tmp_path))
    specs = plan.scenarios_for(d.Unit("fixedtime", "net_a", 0))
    assert [s.slug() for s in specs] == ["clean"]


def plan_uw(tmp_path, agent: str = "gatppo", tau: float = 0.25):
    """A plan with an extra UW column, injected into the `controllers` block
    (`extra` in `plan_file` lands after `scenarios`)."""
    p = plan_file(tmp_path)
    p.write_text(
        p.read_text().replace(
            "fixedtime: {}",
            "fixedtime: {}\n"
            f"  {agent}_uw:\n"
            f"    agent: {agent}\n"
            "    seeds: [0, 1]\n"
            f"    uw: {{tau: {tau}}}",
        )
    )
    return p


def _parents():
    return [SimpleNamespace(id="parent", meta={"params_hash": "abc"})]


def test_uw_is_its_own_kind_with_the_same_parent(tmp_path):
    """The UW column uses THE SAME checkpoint as the core; the modulation has no
    parameters, so the columns differ only by the mechanism."""
    plan = d.load_plan(plan_uw(tmp_path))
    assert plan.kind("gatppo_uw") == "policy_uw" and plan.kind("gatppo") == "policy"
    assert plan.parent_id(d.Unit("gatppo_uw", "net_a", 1)) == "gatppo__net_a__s1_y5"
    assert len([u for u in plan.units() if u.controller == "gatppo_uw"]) == 4


def test_uw_looks_for_weights_where_the_core_keeps_them(tmp_path):
    """Regression: an overlay falling out of the core branch of `checkpoint()` looks
    in LibSignal's layout and the point is SKIPPED with "no weights"."""
    plan = d.load_plan(plan_uw(tmp_path))
    parent = SimpleNamespace(path=tmp_path / "parent", params={"network": "net_a"})
    paths = {
        ctrl: checkpoint(plan, d.Unit(ctrl, "net_a", 0), parent) for ctrl in ("gatppo", "gatppo_uw")
    }
    assert paths["gatppo_uw"] == paths["gatppo"] == tmp_path / "parent" / "engine" / "core.pt"


def test_weight_path_does_not_depend_on_the_controller_kind(tmp_path, monkeypatch):
    """The weight file format depends on the AGENT (who trained), not the kind (how to
    build the controller); a kind that does not exist yet still resolves to `core.pt`."""
    plan = d.load_plan(plan_uw(tmp_path))
    monkeypatch.setattr(type(plan), "kind", lambda self, ctrl: "policy_not_yet_existing")
    parent = SimpleNamespace(path=tmp_path / "parent", params={"network": "net_a"})
    unit = d.Unit("gatppo_uw", "net_a", 0)
    assert checkpoint(plan, unit, parent) == tmp_path / "parent" / "engine" / "core.pt"


def test_the_run_fingerprint_sees_mechanism_parameters(tmp_path):
    """A change of tau (or a gate threshold) MUST give another fingerprint, or a sweep
    computes the first value and skips the rest. A column WITHOUT a mechanism stays
    untouched, so computed core runs remain valid."""

    def fingerprint(plan, ctrl):
        return params_hash(unit_params(plan, d.Unit(ctrl, "net_a", 0), _parents()))

    dirs = [tmp_path / "a", tmp_path / "b"]
    for k in dirs:
        k.mkdir()
    small = d.load_plan(plan_uw(dirs[0], tau=0.25))
    large = d.load_plan(plan_uw(dirs[1], tau=0.50))
    assert fingerprint(small, "gatppo_uw") != fingerprint(large, "gatppo_uw")
    assert fingerprint(small, "gatppo") == fingerprint(large, "gatppo")


def plan_uw_side(tmp_path, side: str):
    p = plan_uw(tmp_path)
    p.write_text(p.read_text().replace("uw: {tau: 0.25}", f"uw: {{tau: 0.25, side: {side}}}"))
    return p


def test_edge_side_changes_the_fingerprint(tmp_path):
    """`side` lives in `uw`, so it enters the fingerprint."""
    hashes = []
    for side in ("target", "source"):
        (tmp_path / side).mkdir()
        plan = d.load_plan(plan_uw_side(tmp_path / side, side))
        hashes.append(params_hash(unit_params(plan, d.Unit("gatppo_uw", "net_a", 0), _parents())))
    assert hashes[0] != hashes[1]


def test_a_typo_in_the_edge_side_is_rejected(tmp_path):
    """A misspelt `side` must not silently give `target`."""
    with pytest.raises(SystemExit, match="side"):
        d.load_plan(plan_uw_side(tmp_path, "src"))


def test_uw_only_with_the_core_actor(tmp_path):
    with pytest.raises(SystemExit, match="UW-attention"):
        d.load_plan(plan_uw(tmp_path, agent="efficientmp"))


def plan_gate(tmp_path, **gate):
    spec = ", ".join(f"{k}: {v}" for k, v in ({"kind": "oracle"} | gate).items())
    p = plan_file(tmp_path)
    p.write_text(
        p.read_text().replace(
            "fixedtime: {}",
            "fixedtime: {}\n"
            "  gatppo_gate:\n"
            "    agent: gatppo\n"
            "    seeds: [0, 1]\n"
            f"    gate: {{{spec}}}",
        )
    )
    return p


def test_gate_is_its_own_kind_with_the_same_parent(tmp_path):
    """As with UW: the gate changes no weights, it only switches where a node's
    action comes from."""
    plan = d.load_plan(plan_gate(tmp_path))
    assert plan.kind("gatppo_gate") == "policy_gate"
    assert plan.parent_id(d.Unit("gatppo_gate", "net_a", 1)) == "gatppo__net_a__s1_y5"
    assert d.KINDS[plan.agent("gatppo_gate")] == "policy"  # hence weights in `core.pt`


def test_the_plan_rejects_a_gate_kind_typo(tmp_path):
    """An unknown `kind` must not silently build the oracle, the upper bound."""
    with pytest.raises(SystemExit, match="kind"):
        d.load_plan(plan_gate(tmp_path, kind="orcale"))


def test_gate_only_with_the_core_actor(tmp_path):
    p = plan_gate(tmp_path)
    p.write_text(p.read_text().replace("agent: gatppo", "agent: efficientmp"))
    with pytest.raises(SystemExit, match="gate"):
        d.load_plan(p)


def test_a_threshold_sweep_gives_distinct_fingerprints(tmp_path):
    """Four columns differ ONLY by threshold, so without `MECHANISM_KEYS` in the
    fingerprint three would be skipped as done. The core column's fingerprint stays
    independent of thresholds."""
    plans = []
    for i, threshold in enumerate((0.25, 0.5, 0.75, 1.0)):
        directory = tmp_path / f"p{i}"
        directory.mkdir()
        plans.append(d.load_plan(plan_gate(directory, threshold=threshold)))

    gated_hashes = {
        params_hash(unit_params(plan, d.Unit("gatppo_gate", "net_a", 0), _parents()))
        for plan in plans
    }
    assert len(gated_hashes) == 4

    core = {
        params_hash(unit_params(plan, d.Unit("gatppo", "net_a", 0), _parents())) for plan in plans
    }
    assert len(core) == 1


# --------------------------------------- explicit reference column (`baseline:`)


def plan_three_arms(tmp_path, baseline: str = "gatppo_fail"):
    """A plan shaped like `t2_train_uw`: TWO columns without a mechanism (clean and
    failure training) plus one with it."""
    p = plan_uw(tmp_path)
    return _write(
        p,
        p.read_text()
        .replace(
            "fixedtime: {}",
            "fixedtime: {}\n"
            "  gatppo_fail:\n"
            "    agent: gatppo\n"
            "    seeds: [0, 1]\n"
            "    parent: '{agent}__{network}__s{seed}_y{yellow}__p1_fail'",
        )
        .replace("    uw: {tau: 0.25}", f"    uw: {{tau: 0.25}}\n    baseline: {baseline}"),
    )


def _write(path, text):
    path.write_text(text)
    return path


def test_baseline_points_to_the_control_arm(tmp_path):
    """Without `baseline:` the report would pair the UW column with the FIRST `policy`
    column (clean training) and measure failure training TOGETHER with the mechanism."""
    plan = d.load_plan(plan_three_arms(tmp_path))
    assert plan.baseline("gatppo_uw") == "gatppo_fail"
    assert mechanism_pairs(plan) == [("gatppo_uw", "gatppo_fail")]

    without = d.load_plan(plan_uw(tmp_path))
    assert without.baseline("gatppo_uw") is None
    assert mechanism_pairs(without) == [("gatppo_uw", "gatppo")]  # the default stays


def test_baseline_must_exist(tmp_path):
    with pytest.raises(SystemExit, match="baseline"):
        d.load_plan(plan_three_arms(tmp_path, baseline="no_such_column"))
    with pytest.raises(SystemExit, match="itself"):
        d.load_plan(plan_three_arms(tmp_path, baseline="gatppo_uw"))


def test_plan_t2_train_uw_loads():
    """A plan from the repo: three arms, an explicit reference, parents from the
    `p1_fail` / `p1_uw` suites."""
    plan = d.load_plan("t2_train_uw")
    assert plan.kind("gatppo_fail_uw") == "policy_uw"
    assert plan.kind("gatppo_fail") == "policy"
    # BOTH readings from the plan header have a report row: arm 2 vs 1 (failure
    # training alone) and arm 3 vs 2 (the mechanism on top)
    assert mechanism_pairs(plan) == [
        ("gatppo_fail", "gatppo"),
        ("gatppo_fail_uw", "gatppo_fail"),
    ]
    unit = d.Unit("gatppo_fail_uw", "cityflow_jinan3x4_4ph", 1)
    assert plan.parent_id(unit) == "gatppo__cityflow_jinan3x4_4ph__s1_y5__p1_uw"
    other = d.Unit("gatppo_fail", "cityflow_jinan3x4_4ph", 1)
    assert plan.parent_id(other) == "gatppo__cityflow_jinan3x4_4ph__s1_y5__p1_fail"
    # arm 1 keeps the default parent, the `p1_gate` run
    assert plan.parent_id(d.Unit("gatppo", "cityflow_jinan3x4_4ph", 1)) == (
        "gatppo__cityflow_jinan3x4_4ph__s1_y5"
    )


def test_a_plain_column_with_baseline_gets_a_row(tmp_path):
    """Regression: a column WITHOUT an inference mechanism but with an explicit
    `baseline:` must reach the delta table (a plain failure-trained arm used to be
    dropped silently)."""
    plan = d.load_plan(plan_three_arms(tmp_path))
    assert plan.kind("gatppo_fail") == "policy"
    assert plan.baseline("gatppo_fail") is None
    assert ("gatppo_fail", "gatppo") not in mechanism_pairs(plan)

    with_base = d.load_plan(
        _write(
            tmp_path / "with_base.yml",
            plan_three_arms(tmp_path)
            .read_text()
            .replace(
                "    parent: '{agent}__{network}__s{seed}_y{yellow}__p1_fail'",
                "    parent: '{agent}__{network}__s{seed}_y{yellow}__p1_fail'\n"
                "    baseline: gatppo",
            ),
        )
    )
    assert ("gatppo_fail", "gatppo") in mechanism_pairs(with_base)


def test_sweep_only_with_a_single_reference(tmp_path):
    """`_sweep_section` builds ONE table around ONE reference column; a plan with
    several references (`t2_gate_dr`) would compare the gate on the DR core against
    the CLEAN core."""
    plan = d.load_plan("t2_gate_dr")
    pairs = mechanism_pairs(plan)
    assert len({base for _, base in pairs}) == 2, pairs
    with pytest.raises(ValueError, match="_sweep_section"):
        _sweep_section(plan, None, pairs)

    # single-parameter sweeps stay sweeps
    for name in ("t2_gate", "t2_gate_real", "t2_uw_tau"):
        bases = {base for _, base in mechanism_pairs(d.load_plan(name))}
        assert bases == {"gatppo"}, (name, bases)


def test_plan_t2_gate_dr_loads():
    """A 2x2 square {clean core, DR core} x {no gate, gate}; three readings: the gate
    alone, failure training alone, the gate ON TOP of failure training."""
    plan = d.load_plan("t2_gate_dr")
    assert plan.kind("gatppo_fail_gate_rule") == "policy_gate"
    assert mechanism_pairs(plan) == [
        ("gatppo_gate_rule", "gatppo"),
        ("gatppo_fail", "gatppo"),
        ("gatppo_fail_gate_rule", "gatppo_fail"),
    ]
    # both gated columns MUST read the same detector
    assert (
        plan.controllers["gatppo_gate_rule"]["gate"]
        == (plan.controllers["gatppo_fail_gate_rule"]["gate"])
    )
    # the composition and DR alone take weights from the same `p1_fail` suite
    for col in ("gatppo_fail", "gatppo_fail_gate_rule"):
        unit = d.Unit(col, "cityflow_jinan3x4_4ph", 2)
        assert plan.parent_id(unit) == "gatppo__cityflow_jinan3x4_4ph__s2_y5__p1_fail"
    # the clean core keeps the default parent (a run without a suite suffix)
    assert plan.parent_id(d.Unit("gatppo", "cityflow_jinan3x4_4ph", 2)) == (
        "gatppo__cityflow_jinan3x4_4ph__s2_y5"
    )


def test_the_plan_rejects_an_unknown_controller_and_field(tmp_path):
    p = plan_file(tmp_path)
    p.write_text(p.read_text().replace("fixedtime: {}", "fixedtime: {}\n  rocket: {}"))
    with pytest.raises(SystemExit, match="controller"):
        d.load_plan(p)
    p.write_text(p.read_text() + "typo: 1\n")
    with pytest.raises(SystemExit, match="unknown"):
        d.load_plan(p)


# ------------------------------------------------------------------ loop


class FakeEnv:
    """Counts calls; ATT = steps + draw number (distinguishable rows)."""

    def __init__(self):
        self.features = None
        self.steps = 0
        self.draw = None

    def reset(self, draw):
        self.draw, self.steps = draw, 0
        return np.zeros((2, 8), np.float32)

    def step(self, actions):
        assert actions.shape == (2,)
        self.steps += 1
        return np.zeros((2, 8), np.float32), None, False

    def metrics(self):
        att = float(self.steps + self.draw)
        return {"att": att, "throughput": 1.0, "queue": 0.0, "delay": 0.0, "reward": 0.0}


def test_evaluate_clean_once_failure_n_times():
    env = FakeEnv()
    specs = (FailureSpec(), FailureSpec(kind="gauss", level=1.0))
    ctrl = c.FixedTimeController([inter(), inter()], t_fixed=30)
    rows = d.evaluate(env, ctrl, specs, draws=3, make_builder=lambda s: None, decisions=4)
    assert [(r["scenario"], r["draw"]) for r in rows] == [
        ("clean", 0), ("gauss1", 0), ("gauss1", 1), ("gauss1", 2),
    ]  # fmt: skip
    assert all(r["att"] == 4 + r["draw"] for r in rows)
    summary = d.summarize(rows)
    assert summary["att__clean"] == summary["att"] == 4.0
    assert summary["att__gauss1"] == pytest.approx(5.0)


# -------------------------------------------------- fallback gate (oracle)

POLICY, RULE = 3, 7  # actions must DIFFER, otherwise the tests pass trivially


def features(broken=(), *, live=True, sensors=True, n=4, prefix=""):
    """A `StateBuilder` stub with only what the gate reads. `sensors=False` = clean
    scenario (the failure layer exists only for an active `FailureSpec`). `prefix`
    changes lane NAMES at the same node count, to test lane -> node cache invalidation."""
    return SimpleNamespace(
        n_nodes=n,
        structs=[SimpleNamespace(in_lanes=[f"{prefix}{i}_{c}" for c in "ab"]) for i in range(n)],
        sensors=SimpleNamespace(broken=frozenset(broken), live=live) if sensors else None,
    )


def gated(env, *, trace=False, **kwargs):
    """`kwargs` go to the ORACLE (threshold), `trace` to the wrapper."""
    policy = SimpleNamespace(act=lambda obs: np.full(env.features.n_nodes, POLICY))
    rule = SimpleNamespace(act=lambda obs: np.full(env.features.n_nodes, RULE))
    return c.GatedController(policy, rule, c.OracleGate(env, **kwargs), trace=trace)


# ------------------------------------------------- action masking


def masked(n_nodes=2, n_phases=4, legal_counts=None, k_max=3, action=0):
    """A stub policy that ALWAYS picks `action`, starving everything else.
    `legal_counts[i]` = how many phases node i really has."""
    legal_counts = legal_counts or [n_phases] * n_nodes
    legal = np.zeros((n_nodes, n_phases), dtype=bool)
    for i, k in enumerate(legal_counts):
        legal[i, :k] = True
    policy = SimpleNamespace(act=lambda obs: np.full(n_nodes, action))
    return c.MaskedController(policy, legal=legal, k_max=k_max)


def test_mask_forces_a_starved_phase():
    """The policy holds phase 0 forever; with k_max=3 the mask steps in by the 4th
    decision at the latest and gives green to the longest-waiting phase."""
    m = masked(n_nodes=1, n_phases=3, k_max=3)
    chosen = [int(m.act(None)[0]) for _ in range(8)]
    assert chosen[:3] == [0, 0, 0], "silent before the limit"
    assert set(chosen[3:]) != {0}, "after the limit it must force something else"
    assert m.episode_stats()["mask_forced"] > 0


def test_mask_keeps_the_service_age_GUARANTEE():
    """With several overdue phases only one can be served, so the bound is
    `k_max + legal_phases - 1`. The test tracks the real age step by step; choosing
    by INDEX instead of age would starve the last phase forever."""
    n_phases, k_max = 4, 3
    m = masked(n_nodes=1, n_phases=n_phases, k_max=k_max)
    age = np.zeros(n_phases, dtype=int)
    oldest = 0
    for _ in range(200):
        a = int(m.act(None)[0])
        age += 1
        age[a] = 0
        oldest = max(oldest, int(age.max()))
    assert oldest <= k_max + n_phases - 1, f"age {oldest} > bound {k_max + n_phases - 1}"


def test_mask_does_NOT_force_a_missing_phase():
    """`n_phases` is the network maximum, so a 2-phase node has dead columns; forcing
    one would be an action outside the intersection's decision space."""
    m = masked(n_nodes=1, n_phases=5, legal_counts=[2], k_max=2)
    chosen = {int(m.act(None)[0]) for _ in range(40)}
    assert chosen <= {0, 1}, f"the mask left the legal phases: {chosen}"


def test_mask_counts_only_OVERRIDES_not_agreement():
    """`mask_share` measures overrides, not activity: agreeing with the policy is a
    no-op (same principle as `gate_share`)."""
    # the policy rotates phases itself, so nothing may become overdue
    counter = {"i": -1}

    def rotate(_obs):
        counter["i"] += 1
        return np.array([counter["i"] % 2])

    m = c.MaskedController(SimpleNamespace(act=rotate), legal=np.array([[True, True]]), k_max=3)
    for _ in range(20):
        m.act(None)
    assert m.episode_stats() == {"mask_share": 0.0, "mask_forced": 0.0}


def test_mask_resets_ages_at_episode_start():
    """`reset_stats` is called at the start of EVERY episode; without it the first
    decisions would force phases overdue from the previous one."""
    m = masked(n_nodes=1, n_phases=3, k_max=3)
    for _ in range(10):
        m.act(None)
    m.reset_stats()
    assert [int(m.act(None)[0]) for _ in range(3)] == [0, 0, 0]


def test_mask_rejects_a_meaningless_k_max():
    with pytest.raises(ValueError, match="k_max"):
        masked(k_max=0)


def test_plan_t2_mask_loads():
    plan = d.load_plan("t2_mask")
    assert plan.kind("gatppo_mask_k16") == "policy_mask"
    # four limits, all against the core -> one combined table
    assert {b for _, b in mechanism_pairs(plan)} == {"gatppo"}
    assert len(mechanism_pairs(plan)) == 4


def test_the_plan_rejects_a_mask_without_k_max(tmp_path):
    """`k_max` IS the whole mechanism; a misspelt key would silently give a default."""
    p = plan_file(tmp_path)
    p.write_text(
        p.read_text().replace(
            "  fixedtime: {}",
            "  gatppo_m:\n    agent: gatppo\n    mask: {kmax: 8}\n  fixedtime: {}",
        )
    )
    with pytest.raises(SystemExit, match="k_max"):
        d.load_plan(p)


def test_the_gate_trace_explains_a_SINGLE_switch():
    """`gate_share` says HOW MANY (decision, node) pairs the rule took, not WHICH or
    WHEN. The trace keeps the RAW score next to both masks, to tell "the detector did
    not fire" from "hysteresis has not let it through yet"."""
    env = SimpleNamespace(features=features({"1_a", "1_b"}))
    controller = gated(env, trace=True)
    controller.act(None)
    trace = controller.episode_trace()

    assert len(trace) == 4, "one row per node per decision"
    assert [w[0] for w in trace] == [0] * 4  # decision number
    assert [w[1] for w in trace] == [0, 1, 2, 3]  # node number
    # node 1 has both inbound lanes dead -> raw score 1.0 and rule mode; the rest 0.0
    assert [w[2] for w in trace] == [0.0, 1.0, 0.0, 0.0]
    assert [w[4] for w in trace] == [False, True, False, False]


def test_the_trace_is_OFF_by_default():
    """Enabled by `tsc degrade run --trace` only."""
    env = SimpleNamespace(features=features({"1_a", "1_b"}))
    controller = gated(env)
    controller.act(None)
    assert controller.episode_trace() == []


def test_trace_is_NOT_part_of_the_run_fingerprint():
    """Why `--trace` is a CLI flag, not a plan key: `gate` is in `MECHANISM_KEYS`, so a
    LOGGING flag there would invalidate every computed point without changing one
    decision."""
    assert "trace" not in d.MECHANISM_KEYS
    plan = d.load_plan("t2_gate_real")
    assert "trace" not in plan.controllers["gatppo_gate_rule"]["gate"]


def test_the_gate_on_clean_leaves_the_policy_alone():
    """On the clean scenario `sensors is None` and the controller must be
    INDISTINGUISHABLE from the policy alone, as for UW-attention."""
    env = SimpleNamespace(features=features(sensors=False))
    controller = gated(env)
    assert controller.act(None).tolist() == [POLICY] * 4
    assert controller.episode_stats() == {"gate_share": 0.0, "gate_switches": 0.0}


def test_the_gate_hits_the_blinded_node_not_its_neighbour():
    """Node 1 has BOTH inbound lanes dead (2/2), node 2 one (1/2), the rest none; at
    threshold 0.5 nodes 1 and 2 switch. Policy and rule actions differ (3 vs 7), so
    an empty or inverted mask would give another vector."""
    env = SimpleNamespace(features=features({"1_a", "1_b", "2_a"}))
    assert gated(env).act(None).tolist() == [POLICY, RULE, RULE, POLICY]


def test_the_threshold_is_inclusive_and_works_both_ways():
    """`>=`: "at least half the detectors dead". With `unit: lane` and `mask0p5`
    nodes sit exactly on this boundary, so it is checked; the second assertion
    checks the threshold is really used."""
    env = SimpleNamespace(features=features({"2_a"}))  # node 2: exactly 1/2
    assert gated(env).act(None).tolist() == [POLICY, POLICY, RULE, POLICY]
    assert gated(env, threshold=0.6).act(None).tolist() == [POLICY] * 4


def test_the_gate_is_silent_before_onset():
    """`broken` is drawn at episode start, but before `onset` the readings are CLEAN;
    a gate asking only about `broken` would switch from step 0."""
    env = SimpleNamespace(features=features({"1_a", "1_b"}, live=False))
    assert gated(env).act(None).tolist() == [POLICY] * 4


def test_the_gate_sees_the_scenario_swap():
    """`degrade.evaluate` swaps `env.features` for EVERY scenario while the controller
    is built once; a gate remembering `features` would read the first (always clean)
    scenario forever. The second pass has OTHER LANE NAMES at the same node count,
    so a lane -> node cache surviving the swap would give an empty mask."""
    env = SimpleNamespace(features=features({"1_a", "1_b"}))
    controller = gated(env)
    assert controller.act(None).tolist() == [POLICY, RULE, POLICY, POLICY]

    env.features = features({"X0_a", "X0_b"}, prefix="X")
    assert controller.act(None).tolist() == [RULE, POLICY, POLICY, POLICY]


def test_reset_stats_also_resets_the_gate():
    """A no-op for the oracle, a correctness condition for windowed gates: their
    window must not enter a new episode with readings from before `world.reset()`."""
    calls = []
    gate = SimpleNamespace(mask=lambda obs: np.zeros(2, dtype=bool), reset=lambda: calls.append(1))
    policy = SimpleNamespace(act=lambda obs: np.full(2, POLICY))
    controller = c.GatedController(policy, policy, gate)
    assert calls == [1]  # the constructor calls reset_stats

    controller.act(None)
    controller.reset_stats()
    assert calls == [1, 1] and controller.episode_stats() == {}


def test_switches_are_counted_per_node():
    """`gate_switches` counts mode CHANGES (each switch resets the phase), to tell a
    stable gate from a flickering one at the same `gate_share`."""
    env = SimpleNamespace(features=features({"1_a", "1_b"}))
    controller = gated(env)
    controller.act(None)
    env.features = features({"2_a", "2_b"})  # one node enters, another leaves
    controller.act(None)
    assert controller.episode_stats() == {"gate_share": 0.25, "gate_switches": 2.0}


# ------------------------------------------- real gates: rule, UQ, hysteresis

N_MOV, WINDOW = 2, 3


def observations(steps, *, frozen=()):
    """[T, N=2, 5]: 2 movements x 2 features + one-hot. Node 0 is healthy (readings
    grow), node 1 has the movements in `frozen` stuck."""
    x = np.zeros((steps, 2, 5), np.float32)
    for t in range(steps):
        x[t, :, :4] = t + 1.0
        for m in frozen:
            x[t, 1, m] = 7.0  # EP of movement m is stuck
            x[t, 1, N_MOV + m] = 0.0  # demand of movement m is stuck (zero)
    return x


def test_the_rule_sees_a_frozen_detector_not_a_healthy_one():
    """ALL features of a movement are stuck over the whole window. Node 1 has both
    movements frozen (1.0), node 0 none (0.0); at 0.5 only node 1 switches."""
    gate = c.RuleGate(window=WINDOW, threshold=0.5, n_mov=N_MOV)
    x = observations(WINDOW, frozen=(0, 1))
    for t in range(WINDOW):
        mask = gate.mask(x[t])
    assert mask.tolist() == [False, True]


def test_the_rule_is_silent_during_warm_up():
    """With a partial buffer EVERY movement looks frozen; without the guard the gate
    would switch the whole network at each episode start."""
    gate = c.RuleGate(window=WINDOW, threshold=0.5, n_mov=N_MOV)
    x = observations(WINDOW, frozen=(0, 1))
    assert gate.mask(x[0]).tolist() == [False, False]  # 1 of 3 readings
    assert gate.mask(x[1]).tolist() == [False, False]  # 2 of 3
    assert gate.mask(x[2]).tolist() == [False, True]  # full window


def test_the_rule_clears_its_buffer_on_reset():
    gate = c.RuleGate(window=WINDOW, threshold=0.5, n_mov=N_MOV)
    x = observations(WINDOW, frozen=(0, 1))
    for t in range(WINDOW):
        gate.mask(x[t])
    gate.reset()
    assert gate.mask(x[0]).tolist() == [False, False]


def test_uq_compares_movements_with_their_own_norm():
    """The UQ gate has the same signal FORM as the rule (fraction of suspicious
    movements), so the ablation compares the criterion, not two statistics."""
    u = np.array([[0.1, 0.1], [0.9, 0.9]])  # node 1 above the norm in both movements
    online = SimpleNamespace(movements=lambda obs: u, ready=True, reset=lambda: None)
    gate = c.UQGate(online, u_mov_ref=np.full((2, 2), 0.5), threshold=0.5)
    assert gate.mask(None).tolist() == [False, True]

    online.ready = False  # warm-up, as in the rule
    assert gate.mask(None).tolist() == [False, False]


def test_hysteresis_delays_entry_and_exit():
    """`k_on` consecutive decisions to enter, `k_off` to leave; the first decision
    passes unsmoothed."""
    signal = iter(
        [
            [False],  # start: policy
            [True],  # 1st above threshold, not enough (k_on=2)
            [True],  # 2nd, enters
            [False],  # 1st below, not enough (k_off=3)
            [False],  # 2nd, still rule
            [False],  # 3rd, leaves
        ]
    )
    gate = SimpleNamespace(mask=lambda obs: np.array(next(signal)), reset=lambda: None)
    policy = SimpleNamespace(act=lambda obs: np.array([POLICY]))
    rule = SimpleNamespace(act=lambda obs: np.array([RULE]))
    controller = c.GatedController(policy, rule, gate, k_on=2, k_off=3)

    actions = [int(controller.act(None)[0]) for _ in range(6)]
    assert actions == [POLICY, POLICY, RULE, RULE, RULE, POLICY]
    assert controller.episode_stats()["gate_switches"] == 2.0


def test_detection_diagnostics_are_relative_to_the_oracle():
    """Precision and recall separate "the detector is weak" from "switching gains
    little"; the oracle is ONLY a diagnostic and never drives actions."""
    gate = SimpleNamespace(mask=lambda obs: np.array([True, True, False]), reset=lambda: None)
    oracle = SimpleNamespace(mask=lambda obs: np.array([True, False, True]), reset=lambda: None)
    policy = SimpleNamespace(act=lambda obs: np.full(3, POLICY))
    rule = SimpleNamespace(act=lambda obs: np.full(3, RULE))
    controller = c.GatedController(policy, rule, gate, oracle=oracle)

    assert controller.act(None).tolist() == [RULE, RULE, POLICY]  # the GATE's mask
    stats = controller.episode_stats()
    assert stats["gate_precision"] == pytest.approx(0.5)  # 1 hit of 2 flagged
    assert stats["gate_recall"] == pytest.approx(0.5)  # 1 of 2 really broken


def test_summarize_averages_metrics_present_in_only_some_draws():
    """`gate_precision`/`gate_recall` are emitted CONDITIONALLY (no key when a draw had
    nothing to detect); checking only the first row used to raise `KeyError` after
    computing the whole point."""
    rows = [
        {"scenario": "mask0p5", "draw": 0, "att": 700.0, "gate_recall": 0.4},
        {"scenario": "mask0p5", "draw": 1, "att": 800.0},  # no denominator
    ]
    out = d.summarize(rows)
    assert out["att__mask0p5"] == pytest.approx(750.0)
    assert out["gate_recall__mask0p5"] == pytest.approx(0.4)  # mean of DEFINED values


# ------------------------------------------------ oracle modulation (UW-attention)

GRAPH = torch.tensor([[0, 1, 2], [1, 2, 3]])  # (source, target): 0->1, 1->2, 2->3


def test_oracle_modulation_dims_the_edge_FROM_the_dead_node():
    """Source side: dead node 1 (both inbound lanes) dims edge 1->2, which carries its
    message, and its own loop, NOT edge 0->1 from healthy neighbour 0.
    log rho layout: [0->1, 1->2, 2->3, loops 0..3]."""
    env = SimpleNamespace(features=features({"1_a", "1_b"}))
    lr = c.OracleUncertainty(env, GRAPH, tau=0.5).log_rho(np.zeros((4, 3)))
    assert lr.tolist() == [0.0, -2.0, 0.0, 0.0, -2.0, 0.0, 0.0]


def test_oracle_modulation_lane_fraction_and_tau():
    """u = fraction of DEAD inbound lanes (as in `OracleGate`), log rho = -u/tau;
    half a node at tau = 0.25 is -2 nats."""
    env = SimpleNamespace(features=features({"2_a"}))
    lr = c.OracleUncertainty(env, GRAPH, tau=0.25).log_rho(np.zeros((4, 3)))
    assert lr[2].item() == pytest.approx(-2.0)  # edge 2->3
    assert lr[3 + 2].item() == pytest.approx(-2.0)  # loop of node 2


@pytest.mark.parametrize("kwargs", [{"sensors": False}, {"live": False}])
def test_oracle_modulation_is_silent_on_clean_and_before_onset(kwargs):
    """Identity without a failure layer and before `onset`: log rho is zero."""
    env = SimpleNamespace(features=features({"1_a", "1_b"}, **kwargs))
    lr = c.OracleUncertainty(env, GRAPH, tau=0.25).log_rho(np.zeros((4, 3)))
    assert (lr == 0).all()


def test_oracle_modulation_reads_the_current_scenario():
    """It holds `env`, not `env.features`: features are swapped per scenario."""
    env = SimpleNamespace(features=features(sensors=False))
    oracle = c.OracleUncertainty(env, GRAPH, tau=0.25)
    assert (oracle.log_rho(np.zeros((4, 3))) == 0).all()
    env.features = features({"0_a", "0_b"})
    assert oracle.log_rho(np.zeros((4, 3)))[0].item() == pytest.approx(-4.0)


def test_oracle_modulation_in_the_uw_controller():
    """It plugs into `UWPolicyController` unchanged and gives the `uw_spread` diagnostic."""
    seen = {}

    def actor(x, edge_index, log_rho=None):
        seen["log_rho"] = log_rho
        return torch.zeros(x.shape[0], 2)

    actor.eval = lambda: actor
    env = SimpleNamespace(features=features({"1_a", "1_b"}))
    ctrl = c.UWPolicyController(actor, GRAPH, c.OracleUncertainty(env, GRAPH, tau=0.25))
    ctrl.act(np.zeros((4, 3), np.float32))
    assert seen["log_rho"][1].item() == pytest.approx(-4.0)
    assert ctrl.episode_stats()["uw_spread"] > 0


def plan_uw_spec(tmp_path, spec: str):
    p = plan_uw(tmp_path)
    p.write_text(p.read_text().replace("uw: {tau: 0.25}", f"uw: {{{spec}}}"))
    return p


def test_oracle_modulation_requires_the_source_side(tmp_path):
    with pytest.raises(SystemExit, match="side: source"):
        d.load_plan(plan_uw_spec(tmp_path, "tau: 0.25, u: oracle"))


def test_a_typo_in_the_u_source_is_rejected(tmp_path):
    """A misspelt oracle must not silently use the head."""
    with pytest.raises(SystemExit, match="u="):
        d.load_plan(plan_uw_spec(tmp_path, "tau: 0.25, side: source, u: orcale"))


def test_oracle_modulation_has_its_own_fingerprint(tmp_path):
    hashes = []
    for i, spec in enumerate(["tau: 0.25, side: source", "tau: 0.25, side: source, u: oracle"]):
        (tmp_path / str(i)).mkdir()
        plan = d.load_plan(plan_uw_spec(tmp_path / str(i), spec))
        hashes.append(params_hash(unit_params(plan, d.Unit("gatppo_uw", "net_a", 0), _parents())))
    assert hashes[0] != hashes[1]


@pytest.mark.slow
def test_oracle_modulation_on_real_cityflow(advanced_colight_world):
    """The oracle reads the REAL `StateBuilder` swapped per scenario: silent on clean,
    with dead intersections it dims exactly the edges LEAVING dead nodes and their
    loops."""
    world, _ = advanced_colight_world
    torch.manual_seed(0)
    trainer = CoreTrainer.from_world(
        world, loop_config=LoopConfig(steps=60, test_steps=60, episodes=1)
    )
    env = trainer.env
    order = env.graph.order_intersections(world)
    oracle = c.OracleUncertainty(env, env.graph.edge_index, tau=0.25)
    ctrl = c.UWPolicyController(trainer.ppo.ac.actor, env.graph.edge_index, oracle)

    clean = FailureSpec()
    dead = FailureSpec(kind="mask", level=0.5, unit="intersection")
    rows = d.evaluate(
        env,
        ctrl,
        (clean, dead),
        draws=1,
        make_builder=lambda spec: StateBuilder.from_registry(world, order, failure=spec),
        decisions=4,
    )
    assert rows[0]["uw_spread"] == 0.0 and rows[0]["uw_active"] == 0.0
    assert rows[1]["uw_spread"] > 0.0

    lr = oracle.log_rho(None).numpy()  # state after the last episode: `dead`
    u = oracle._gate.last_score
    assert set(np.unique(u)) <= {0.0, 1.0} and u.any() and not u.all()
    src = env.graph.edge_index[0].numpy()
    e = len(src)
    assert np.array_equal(lr[:e] < 0, u[src] == 1.0)
    assert np.array_equal(lr[e:] < 0, u == 1.0)


# ------------------------------------------- rule modulation and composition with a gate

PAIR = torch.tensor([[0, 1], [1, 0]])  # 2 nodes: 0->1, 1->0; log rho = [0->1, 1->0, loop 0, loop 1]


def rule_u(threshold=0.5, tau=0.25):
    return c.RuleUncertainty(PAIR, window=WINDOW, threshold=threshold, n_mov=N_MOV, tau=tau)


def test_rule_modulation_dims_the_message_FROM_the_frozen_node():
    """Source side as in the oracle: frozen node 1 dims edge 1->0 (its message) and
    its own loop, NOT edge 0->1 from the healthy neighbour."""
    u = rule_u()
    x = observations(WINDOW, frozen=(0, 1))
    for t in range(WINDOW):
        lr = u.log_rho(x[t])
    assert lr.tolist() == [0.0, -4.0, 0.0, -4.0]


def test_rule_modulation_is_silent_below_threshold_and_graded_above():
    """Zero below the threshold (empty approaches would modulate on clean); at and
    above it u = fraction of frozen movements, as in the oracle."""
    x = observations(WINDOW, frozen=(0,))  # node 1: 1 of 2 movements, fraction 0.5
    strict, loose = rule_u(threshold=0.75), rule_u(threshold=0.5)
    for t in range(WINDOW):
        lr_strict, lr_loose = strict.log_rho(x[t]), loose.log_rho(x[t])
    assert (lr_strict == 0).all()
    assert lr_loose.tolist() == [0.0, -2.0, 0.0, -2.0]


def test_rule_modulation_is_silent_during_warm_up_and_after_reset():
    u = rule_u()
    x = observations(WINDOW, frozen=(0, 1))
    assert (u.log_rho(x[0]) == 0).all() and (u.log_rho(x[1]) == 0).all()
    assert u.log_rho(x[2])[1].item() == pytest.approx(-4.0)
    u.reset()
    assert (u.log_rho(x[0]) == 0).all()


def test_the_composition_resets_and_reports_the_stateful_policy():
    """A gate wrapping `UWPolicyController` clears its window at episode start and
    passes its `uw_spread` next to `gate_*`."""
    resets = []
    policy = SimpleNamespace(
        act=lambda obs: np.full(2, POLICY),
        reset_stats=lambda: resets.append(1),
        episode_stats=lambda: {"uw_spread": 1.5},
    )
    gate = SimpleNamespace(mask=lambda obs: np.zeros(2, dtype=bool), reset=lambda: None)
    controller = c.GatedController(policy, policy, gate)
    controller.act(None)
    controller.reset_stats()
    assert len(resets) == 2  # constructor + episode start
    controller.act(None)
    assert controller.episode_stats()["uw_spread"] == 1.5


def test_rule_modulation_requires_the_source_side(tmp_path):
    with pytest.raises(SystemExit, match="side: source"):
        d.load_plan(plan_uw_spec(tmp_path, "tau: 0.25, u: rule"))


def test_plan_t2_uw_rule_loads_and_pairs_the_composition_with_the_gate():
    """The composition is a `policy_gate` (the gate wraps UW) and pairs with the gate
    ALONE, not the core, or the delta would measure gate and UW at once."""
    plan = d.load_plan("t2_uw_rule")
    assert plan.kind("gatppo_gate_rule_uw") == "policy_gate"
    assert plan.kind("gatppo_uw_rule") == "policy_uw"
    pairs = dict(mechanism_pairs(plan))
    assert pairs["gatppo_gate_rule_uw"] == "gatppo_gate_rule"
    assert pairs["gatppo_gate_oracle_uw"] == "gatppo_gate_oracle"
    assert pairs["gatppo_uw_rule"] == "gatppo"


@pytest.mark.slow
def test_gate_and_rule_uw_composition_on_real_cityflow(advanced_colight_world):
    """Smoke test of the `t2_uw_rule` composition on a real `StateBuilder`: with dead
    intersections BOTH mechanisms fire (the gate switches, the modulation spreads
    the softmax groups)."""
    world, _ = advanced_colight_world
    torch.manual_seed(0)
    trainer = CoreTrainer.from_world(
        world, loop_config=LoopConfig(steps=60, test_steps=60, episodes=1)
    )
    env = trainer.env
    order = env.graph.order_intersections(world)
    feats = env.features
    rule = c.RuleUncertainty(
        env.graph.edge_index, window=3, threshold=1.0, n_mov=feats.n_mov, tau=0.25
    )
    ctrl = c.GatedController(
        c.UWPolicyController(trainer.ppo.ac.actor, env.graph.edge_index, rule),
        c.FixedTimeController(feats.intersections, t_fixed=30),
        c.OracleGate(env, threshold=0.75),
    )
    dead = FailureSpec(kind="mask", level=0.5, unit="intersection")
    rows = d.evaluate(
        env,
        ctrl,
        (FailureSpec(), dead),
        draws=1,
        make_builder=lambda spec: StateBuilder.from_registry(world, order, failure=spec),
        decisions=8,
    )
    assert rows[0]["gate_share"] == 0.0
    assert rows[1]["gate_share"] > 0.0 and rows[1]["uw_spread"] > 0.0


def test_pair_description_tells_a_shared_checkpoint_from_separate_training():
    """An explicit `baseline:` does not mean "separate training": gate + UW pairs with
    the gate on THE SAME checkpoint, the `gatppo_fail` arm on another."""
    composition = d.load_plan("t2_uw_rule")
    assert shared_checkpoint(composition, "gatppo_gate_rule_uw", "gatppo_gate_rule")
    training = d.load_plan("t2_train_uw")
    assert not shared_checkpoint(training, "gatppo_fail", "gatppo")


# --------------------------------------------------- channel ablation (`t2_uw_channels`)


def test_channels_are_disjoint_and_add_up_to_full_modulation():
    """The ablation decomposes the gain EXACTLY only if `self` and `edges` are disjoint
    and sum to `all`. Layout: [0->1, 1->2, 2->3, loops 0..3], dead node 1."""
    u = np.array([0.0, 1.0, 0.0, 0.0])
    src = GRAPH[0].numpy()
    full = c.source_log_rho(u, src, 0.5)
    loops = c.source_log_rho(u, src, 0.5, "self")
    edges = c.source_log_rho(u, src, 0.5, "edges")
    assert loops.tolist() == [0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0]  # only node 1's loop
    assert edges.tolist() == [0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # only its message
    assert torch.equal(loops + edges, full)


def test_channel_in_oracle_and_rule():
    env = SimpleNamespace(features=features({"1_a", "1_b"}))
    lr = c.OracleUncertainty(env, GRAPH, tau=0.5, channels="edges").log_rho(np.zeros((4, 3)))
    assert lr.tolist() == [0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="channels"):
        c.RuleUncertainty(PAIR, window=WINDOW, threshold=0.5, n_mov=N_MOV, tau=0.25, channels="x")


def test_channel_changes_the_fingerprint(tmp_path):
    """`channels` lives in `uw`, so it enters the fingerprint."""
    hashes = set()
    for i, channel in enumerate(["all", "self", "edges"]):
        (tmp_path / str(i)).mkdir()
        spec = f"tau: 0.25, side: source, u: oracle, channels: {channel}"
        plan = d.load_plan(plan_uw_spec(tmp_path / str(i), spec))
        hashes.add(params_hash(unit_params(plan, d.Unit("gatppo_uw", "net_a", 0), _parents())))
    assert len(hashes) == 3


def test_plan_t2_uw_channels_pairs_every_channel_with_the_core():
    plan = d.load_plan("t2_uw_channels")
    pairs = dict(mechanism_pairs(plan))
    for column in ("gatppo_uw_oracle", "gatppo_uw_oracle_self", "gatppo_uw_oracle_edges"):
        assert pairs[column] == "gatppo"
    assert not any(s.kind in ("gauss", "drift") for s in plan.scenarios)


def test_plan_t2_train_uw_rule_pairs_training_with_inference_only():
    """The same modulation at evaluation; only whether the policy trained with it
    differs, so the reference is ADDING it to the DR policy, not the bare DR policy."""
    plan = d.load_plan("t2_train_uw_rule")
    pairs = dict(mechanism_pairs(plan))
    assert pairs["gatppo_fail_uw_rule"] == "gatppo_fail_uwinf"
    assert pairs["gatppo_fail_uw_rule_gate"] == "gatppo_fail_gate"
    assert pairs["gatppo_fail_uwinf"] == "gatppo_fail"
    assert plan.parent_id(d.Unit("gatppo_fail_uw_rule", "cityflow_jinan3x4_4ph", 1)) == (
        "gatppo__cityflow_jinan3x4_4ph__s1_y5__p1_uw_rule"
    )
    assert not shared_checkpoint(plan, "gatppo_fail_uw_rule", "gatppo_fail_uwinf")
    assert shared_checkpoint(plan, "gatppo_fail_uwinf", "gatppo_fail")


# ----------------------------------------------- fixed time with a Webster program (Krakow)


def intersection(id_, phase, time, n_phases):
    """Green phases have STATES `S0`, `S1`, ...; the program is matched on them."""
    return SimpleNamespace(
        id=id_,
        current_phase=phase,
        current_phase_time=time,
        phases=list(range(n_phases)),
        green_phases=[SimpleNamespace(state=f"S{k}") for k in range(n_phases)],
    )


def test_webster_holds_each_phase_its_own_time_plus_yellow():
    """Threshold = yellow + the phase's green (the counter starts with the transition).
    The program is given in ANOTHER order than the phases: matching is by state."""
    a = intersection("A", 0, 24, 2)
    b = intersection("B", 1, 24, 3)
    programs = {"A": {"S1": 40, "S0": 20}, "B": {"S2": 30, "S0": 10, "S1": 20}}
    ft = c.FixedTimeController([a, b], t_fixed=30, greens=programs, yellow=5)
    assert ft.act(None).tolist() == [0, 1]  # A: 24 < 25, B: 24 < 25
    a.current_phase_time = b.current_phase_time = 25
    assert ft.act(None).tolist() == [1, 2]  # both exceeded yellow + green


def test_without_a_program_behaviour_is_unchanged():
    """JiNan/HangZhou keep `t_fixed`."""
    node = intersection("A", 0, 29, 4)
    ft = c.FixedTimeController([node], t_fixed=30)
    assert ft.act(None).tolist() == [0]
    node.current_phase_time = 30
    assert ft.act(None).tolist() == [1]


def test_the_program_must_match_the_phase_structure():
    with pytest.raises(ValueError, match="phase structure"):
        c.FixedTimeController(
            [intersection("A", 0, 0, 3)], t_fixed=30, greens={"A": {"S0": 20, "S1": 40}}
        )
    with pytest.raises(ValueError, match="without intersections"):
        c.FixedTimeController([intersection("A", 0, 0, 2)], t_fixed=30, greens={})


def test_a_fixed_time_program_only_where_the_network_declares_one():
    """JiNan/HangZhou have no `fixedTimeProgram` -> None, i.e. `t_fixed`."""
    assert fixed_time_program("cityflow_jinan3x4_4ph") is None
    assert fixed_time_program("cityflow_hangzhou4x4_4ph") is None
    if not (DATA / "raw_data" / "krakow_3w" / "3w_mid.webster.json").is_file():
        with pytest.raises(SystemExit, match="webster"):
            fixed_time_program("sumo_krakow3w_mid")
    else:
        program = fixed_time_program("sumo_krakow3w_mid")
        assert len(program) == 25 and all(program.values())


# ------------------------------------- the rule on a heterogeneous network (Krakow, 1-7 approaches)


def test_padded_movements_do_not_count_as_frozen():
    """Node 1 has ONE real movement (the second slot is zero padding); a healthy real
    movement gives 0 frozen although the padded slot is stuck. Without the mask: 0.5."""
    suspicious = np.array([[False, False], [False, True]])
    real = np.array([[True, True], [True, False]])
    assert c.frozen_fraction(suspicious, None).tolist() == [0.0, 0.5]
    assert c.frozen_fraction(suspicious, real).tolist() == [0.0, 0.0]
    suspicious[1, 0] = True  # now the only real movement is dead
    assert c.frozen_fraction(suspicious, real).tolist() == [0.0, 1.0]


def test_the_rule_with_a_movement_mask_on_a_single_approach_node():
    gate = c.RuleGate(
        window=WINDOW, threshold=0.5, n_mov=N_MOV, real=np.array([[True, True], [True, False]])
    )
    x = observations(WINDOW, frozen=(1,))  # node 1: only the padded slot is frozen
    for t in range(WINDOW):
        mask = gate.mask(x[t])
    assert mask.tolist() == [False, False]


def test_movement_mask_only_on_a_heterogeneous_network():
    assert phase_mask([8, 8, 8], 8) is None
    assert phase_mask([8, 4], 8)[1].tolist() == [True] * 4 + [False] * 4


def test_oracle_node_without_inbound_lanes():
    """Krakow: a signal without lanes in `in_lanes` gives 0, not a division by zero."""
    env = SimpleNamespace(features=features({"1_a", "1_b"}))
    env.features.structs[2] = SimpleNamespace(in_lanes=[])
    gate = c.OracleGate(env)
    assert gate.mask(None).tolist() == [False, True, False, False]
    assert gate.last_score[2] == 0.0


# ------------------------------------------- best checkpoint (Krakow, `t2_krakow_best`)


def test_checkpoint_best_points_to_another_file_and_changes_the_fingerprint(tmp_path):
    """`checkpoint: best` = `core_best.pt` of the same parent; it must change the
    fingerprint, or the best-checkpoint column would be skipped as done."""
    plan = d.load_plan("t2_krakow_best")
    parent = SimpleNamespace(path=tmp_path, id="r", meta={"params_hash": "abc"})
    net = "sumo_krakow3w_mid_s1"
    assert checkpoint(plan, d.Unit("gatppo", net, 0), parent).name == "core.pt"
    assert checkpoint(plan, d.Unit("gatppo_best", net, 0), parent).name == "core_best.pt"
    a = unit_params(plan, d.Unit("gatppo", net, 0), [parent])
    b = unit_params(plan, d.Unit("gatppo_best", net, 0), [parent])
    assert "checkpoint" not in a and b["checkpoint"] == "best"
    assert not shared_checkpoint(plan, "gatppo_best", "gatppo")


def test_a_checkpoint_typo_is_rejected(tmp_path):
    p = plan_uw(tmp_path)
    p.write_text(p.read_text().replace("uw: {tau: 0.25}", "checkpoint: bestest"))
    with pytest.raises(SystemExit, match="checkpoint"):
        d.load_plan(p)
