"""Training with the mechanism: suite consistency, `uw` validation, rule thresholds.

The failure mix of the control arm (`p1_fail`) and of the mechanism arms must be
IDENTICAL: a different weight raises nothing and gives a table that answers
another question.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from tsc_marl.domain.sensors import FailureMix
from tsc_marl.paths import CONFIGS
from tsc_marl.robustness import controllers as c
from tsc_marl.robustness import rule_ref, training
from tsc_marl.robustness.training import check_uw, gate_reference, uw_signal
from tsc_marl.uncertainty import obs_head as oh


def suite(name: str) -> dict:
    return yaml.safe_load((CONFIGS / "suites" / f"{name}.yml").read_text())


def suite_mix(name: str) -> dict:
    return suite(name)["overrides"]["model"]["failure_mix"]


# ------------------------------------------------------ the arms are consistent


def test_arms_share_the_failure_mix():
    """`p1_fail` (without UW) and `p1_uw` (with UW) differ ONLY in the mechanism."""
    assert suite_mix("p1_fail") == suite_mix("p1_uw")


def test_only_the_uw_arm_enables_the_mechanism():
    without = suite("p1_fail")["overrides"]["model"]
    with_uw = suite("p1_uw")["overrides"]["model"]
    assert "uw" not in without
    assert with_uw["uw"]["tau"] == 0.25
    assert set(with_uw) - set(without) == {"uw"}


def test_suite_mix_parses():
    """The YAML must give a mix, not an exception three hours into a run."""
    m = FailureMix.from_config(suite_mix("p1_uw"))
    assert m is not None
    assert "clean" in {s.slug() for s in m.specs}
    # observable, inferable and unobservable failures are all represented
    assert any(s.kind == "gauss" for s in m.specs)
    assert any(s.kind == "mask" and s.substitution == "hold" for s in m.specs)
    assert any(s.kind == "drift" for s in m.specs)


def test_rule_arm_shares_the_mix_with_the_control():
    """`p1_uw_rule` vs `p1_fail`: the only difference is the mechanism."""
    assert suite_mix("p1_fail") == suite_mix("p1_uw_rule")
    without = suite("p1_fail")["overrides"]["model"]
    with_uw = suite("p1_uw_rule")["overrides"]["model"]
    assert set(with_uw) - set(without) == {"uw"}
    assert with_uw["uw"] == {"tau": 0.25, "side": "source", "u": "rule"}


@pytest.mark.parametrize(
    "uw, fragment",
    [
        ({"u": "rule"}, "side: source"),  # the rule exists only on the source side
        ({"u": "rul", "side": "source"}, "u="),
        ({"side": "source", "u": "oracle", "channels": "loops"}, "channels="),
        ({"channels": "self"}, "channels only"),  # the head has no channels
        ({"tau": 0.25, "threshhold": 0.5}, "unknown keys"),
        ({"side": "source", "u": "oracle", "threshold": 0.5}, "threshold only"),
    ],
)
def test_typos_in_uw_are_errors_not_defaults(uw, fragment):
    """One validator for plans and suites: a typo in a training suite would
    silently fall back to the head on the target side."""
    error = check_uw(uw)
    assert error is not None and fragment in error


def test_valid_uw_specs_pass():
    for uw in (
        {"tau": 0.25},
        {"tau": 0.25, "side": "source"},
        {"tau": 0.25, "side": "source", "u": "rule"},
        {"tau": 0.25, "side": "source", "u": "oracle", "channels": "edges"},
    ):
        assert check_uw(uw) is None, uw


def test_uw_signal_builds_the_rule_from_gate_thresholds():
    """Training and evaluation build the signal with the same function; for
    `u: rule` window and threshold come from this network's `gate_ref`."""
    network = "cityflow_jinan3x4_4ph"
    if not oh.gate_path(network).is_file():
        pytest.skip("no runs/heads/gate_ref (computed by `tsc obshead`)")

    env = SimpleNamespace(
        graph=SimpleNamespace(edge_index=torch.tensor([[0, 1], [1, 0]])),
        features=SimpleNamespace(n_mov=8, movement_mask=lambda: None),
    )
    signal = uw_signal({"tau": 0.25, "side": "source", "u": "rule"}, network, env)
    thresholds = gate_reference(network)
    assert isinstance(signal, c.RuleUncertainty)
    assert signal._gate.window == thresholds.window
    assert signal._gate.threshold == pytest.approx(thresholds.rule)


# ---------------------------------------------- rule threshold alone (networks without a head)


def test_rule_threshold_is_the_false_alarm_quantile():
    scores = np.zeros((1000, 1))
    scores[-4:] = 1.0  # 0.4% fully frozen decisions: within the 0.5% budget
    assert rule_ref.threshold(scores, 0.005) == 0.0
    scores[-6:] = 1.0  # 0.6%: over budget, the threshold must rise
    assert rule_ref.threshold(scores, 0.005) > 0.0


def test_rule_threshold_round_trip_and_gate_ref_precedence(tmp_path, monkeypatch):
    """A network with `gate_ref` (CityFlow, with a head) reads IT, not `rule_ref`,
    so the JiNan/HangZhou numbers cannot change; one without reads `rule_ref`."""
    monkeypatch.setattr(rule_ref, "rule_path", lambda net: tmp_path / f"rule_ref__{net}.json")
    blob = {
        "format": rule_ref.RULE_FORMAT,
        "network": "sumo_x",
        "window": 12,
        "rule": 0.4,
        "false_alarm": 0.005,
        "expert": "advancedmp",
        "episodes": 3,
        "decisions": 700,
    }
    (tmp_path / "rule_ref__sumo_x.json").write_text(json.dumps(blob))
    ref = training.gate_reference("sumo_x")
    assert (ref.window, ref.rule, ref.uq) == (12, 0.4, None)
    with pytest.raises(SystemExit, match="rulecal"):
        training.gate_reference("sumo_missing")

    if oh.gate_path("cityflow_jinan3x4_4ph").is_file():
        assert training.gate_reference("cityflow_jinan3x4_4ph").uq is not None


def test_krakow_dr_uses_the_p1_fail_mix():
    """The Krakow replication measures the same lever as JiNan/HangZhou."""
    assert suite_mix("k1_fail") == suite_mix("p1_fail")
    core = suite("k1_core")
    assert "overrides" not in core or "failure_mix" not in core["overrides"].get("model", {})


def test_k2_is_k1_plus_best_checkpoint():
    """The repeat differs from `k1_*` ONLY in `save_best`."""
    for k1, k2 in (("k1_core", "k2_core"), ("k1_fail", "k2_fail")):
        a, b = suite(k1), suite(k2)
        model_a = (a.get("overrides") or {}).get("model", {})
        model_b = dict(b["overrides"]["model"])
        assert model_b.pop("save_best") is True
        assert model_a == model_b
        assert {k: v for k, v in a.items() if k != "overrides"} == {
            k: v for k, v in b.items() if k != "overrides"
        }
