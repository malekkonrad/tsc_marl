"""`tsc core` entry point.

Short but fragile: the Registry interface sequence is order-sensitive and has
already caused a silent bug once (an ignored `root_dir`). So the end-to-end test
runs the WHOLE sequence on a real CityFlow, with an episode cut to 10 decisions.
The config tests check above all that every PPO hyperparameter is in YAML: a
value living only in Python is outside the params fingerprint, so skip-if-done
would treat a run with another `clip_range` as already done.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from tsc_marl import libsignal
from tsc_marl.core.ppo import PPOConfig
from tsc_marl.libsignal import experiment
from tsc_marl.runner.train_core import parse_args

NETWORK = "cityflow_jinan3x4_4ph"


def test_default_agent_is_the_core():
    """Without `-a` the agent is `gatppo`; an explicit choice still wins."""
    assert parse_args([]).agent == "gatppo"
    assert parse_args(["-a", "colight"]).agent == "colight"
    assert parse_args(["--agent", "advancedcolight"]).agent == "advancedcolight"


@pytest.fixture(scope="module")
def core_config():
    libsignal.bootstrap()
    return experiment.resolve_config(parse_args(["-n", NETWORK]))


def test_core_sees_the_baseline_protocol_and_state(core_config):
    """The core is compared with Advanced-CoLight, so the ONLY difference is the
    learning algorithm: protocol (interval, yellow, episode length), state (ATS,
    167 m window, no always-green links) and GAT architecture come from the same
    files."""
    model, trainer = core_config["model"], core_config["trainer"]
    assert (trainer["action_interval"], trainer["yellow_length"], trainer["steps"]) == (15, 5, 3600)
    assert (model["state"], model["reward"]) == ("ats", "queue")
    assert model["obs_length"] == 167 and model["drop_always_green"] is True
    assert model["NUM_HEADS"] == [5, 5] and model["NODE_EMB_DIM"] == [128, 128]


def test_every_ppo_hyperparameter_is_in_the_config(core_config):
    """The run fingerprint covers `model` and `trainer`; a hyperparameter only in
    `PPOConfig` would not change it."""
    missing = [f.name for f in fields(PPOConfig) if f.name not in core_config["model"]]
    assert missing == [], f"PPO hyperparameters outside the config: {missing}"


def test_steps_overrides_both_episodes():
    """`--steps` shortens training AND test episodes together."""
    libsignal.bootstrap()
    config = experiment.resolve_config(parse_args(["-n", NETWORK, "--steps", "150"]))
    assert config["trainer"]["steps"] == config["trainer"]["test_steps"] == 150


@pytest.mark.slow
def test_end_to_end_run_stores_results(tmp_path, monkeypatch):
    """Registry interfaces -> world -> `CoreTrainer` -> `runlog`, one episode of
    10 decisions; `runs/` is redirected to `tmp_path`."""
    from runlog import Experiment
    from tsc_marl.runner import train_core

    monkeypatch.setattr(
        experiment, "experiment", lambda *a, **kw: Experiment(experiment.EXPERIMENT, root=tmp_path)
    )
    train_core.main(["-n", NETWORK, "--episodes", "1", "--steps", "150", "--run-id", "smoke"])

    run_dir = tmp_path / experiment.EXPERIMENT / "smoke"
    from runlog import RunView

    view = RunView(run_dir)
    assert view.status == "ok"
    assert view.result["att"] > 0 and view.result["episodes_logged"] == 1
    assert (run_dir / "engine" / "core.pt").is_file(), "core checkpoint was not written"

    metrics = (run_dir / "metrics.csv").read_text()
    assert "train" in metrics and "test" in metrics
    assert "approx_kl" in metrics, "PPO diagnostics must reach the run metrics"
