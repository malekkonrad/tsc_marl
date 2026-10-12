"""`runlog` tests: no engine, no `tsc_marl`, no networks.

A test of the tool, not the project: if `runlog` ever moves to its own
repository, this file goes with it and must pass without anything from the thesis.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import yaml

from runlog import Experiment, RunExists, RunIdCollision, params_hash
from runlog import store as rstore

PARAMS = {"agent": "colight", "network": "jinan", "seed": 0}


@pytest.fixture
def exp(tmp_path):
    return Experiment("t1", root=tmp_path / "runs")


# --------------------------------------------------------------- life cycle


def test_successful_run_ends_with_status_ok(exp):
    with exp.start("r1", PARAMS) as run:
        run.result(att=299.2)
    view = exp.runs()[0]
    assert view.status == "ok"
    assert view.result["att"] == 299.2
    assert view.meta["duration_s"] >= 0


def test_exception_records_failed_and_is_not_swallowed(exp):
    with pytest.raises(ValueError, match="simulator crashed"), exp.start("r1", PARAMS) as run:
        run.log(step=0, att=1.0)
        raise ValueError("simulator crashed")

    view = exp.runs()[0]
    assert view.status == "failed"
    assert "simulator crashed" in view.meta["error"]
    # metrics from before the failure survive, for diagnosis
    assert (view.path / rstore.METRICS).read_text().count("\n") == 2


def test_running_status_is_visible_meanwhile(exp):
    run = exp.start("r1", PARAMS)
    assert exp.runs()[0].status == "running"
    run.finish()
    assert exp.runs()[0].status == "ok"


# ------------------------------------------------------------------ collisions


def test_same_id_other_params_is_an_error_not_an_overwrite(exp):
    with exp.start("r1", PARAMS) as run:
        run.result(att=1.0)

    with pytest.raises(RunIdCollision, match="DIFFERENT parameters"):
        exp.start("r1", {**PARAMS, "seed": 1})

    # the old run is untouched
    assert exp.runs()[0].result["att"] == 1.0


def test_a_finished_run_is_not_computed_twice(exp):
    with exp.start("r1", PARAMS):
        pass
    with pytest.raises(RunExists):
        exp.start("r1", PARAMS)


def test_a_live_run_blocks_too(exp):
    """`running` may be a process computing right now; it must not be trampled."""
    exp.start("r1", PARAMS)
    with pytest.raises(RunExists):
        exp.start("r1", PARAMS)


def _dead_pid() -> int:
    """The pid of a process that is certainly dead."""
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def _set_meta(exp, run_id, **fields):
    path = exp.path / run_id / rstore.META
    meta = json.loads(path.read_text())
    meta.update(fields)
    path.write_text(json.dumps(meta))


def test_an_orphaned_run_may_be_recomputed(exp):
    """`running` written on this machine by a process that is gone never closes;
    without this rule it would block resuming a suite."""
    exp.start("r1", PARAMS)
    _set_meta(exp, "r1", pid=_dead_pid())
    with exp.start("r1", PARAMS) as run:
        run.result(att=1.0)
    assert exp.runs()[0].status == "ok"


def test_running_from_another_machine_or_without_pid_still_blocks(exp):
    """Conservative: another machine cannot be checked, old metadata has no pid."""
    exp.start("r1", PARAMS)
    _set_meta(exp, "r1", pid=_dead_pid(), host="other-machine")
    with pytest.raises(RunExists):
        exp.start("r1", PARAMS)
    _set_meta(exp, "r1", pid=None, host=None)
    with pytest.raises(RunExists):
        exp.start("r1", PARAMS)


def test_a_failed_run_may_be_repeated(exp):
    with pytest.raises(RuntimeError), exp.start("r1", PARAMS):
        raise RuntimeError("boom")
    with exp.start("r1", PARAMS) as run:  # without replace: resuming after a failure is legal
        run.result(att=2.0)
    assert exp.runs()[0].status == "ok"


def test_replace_overwrites_deliberately(exp):
    with exp.start("r1", PARAMS) as run:
        run.result(att=1.0)
    with exp.start("r1", {**PARAMS, "seed": 9}, replace=True) as run:
        run.result(att=2.0)
    assert exp.runs()[0].result["att"] == 2.0


# ------------------------------------------------------------- lookup


def test_find_sees_only_finished_runs_by_default(exp):
    with exp.start("ok1", PARAMS):
        pass
    exp.start("hanging", {**PARAMS, "seed": 1})  # stays running
    with pytest.raises(RuntimeError), exp.start("bad", {**PARAMS, "seed": 2}):
        raise RuntimeError

    assert [r.id for r in exp.find()] == ["ok1"]
    assert len(exp.find(status=None)) == 3
    assert [r.id for r in exp.find(seed=1, status=None)] == ["hanging"]


def test_is_done_reacts_to_changed_parameters(exp):
    with exp.start("r1", PARAMS):
        pass
    assert exp.is_done("r1", PARAMS)
    assert not exp.is_done("r1", {**PARAMS, "lr": 0.001})  # another config: recompute
    assert not exp.is_done("does-not-exist", PARAMS)


# ----------------------------------------------------------------- metrics


def test_metrics_in_long_format_concatenated_across_runs(exp):
    for seed in (0, 1):
        with exp.start(f"s{seed}", {**PARAMS, "seed": seed}) as run:
            for e in range(3):
                run.log(step=e, split="train", att=100.0 + e, loss=0.5)
                run.log(step=e, split="test", att=90.0 + e)

    df = exp.metrics()
    assert list(df.columns) == ["run_id", "step", "split", "metric", "value"]
    assert len(df) == 2 * 3 * 3
    assert set(df["metric"]) == {"att", "loss"}
    assert set(df["run_id"]) == {"s0", "s1"}


def test_none_does_not_reach_the_metrics(exp):
    """Methods without training have no `loss`: no value means no row."""
    with exp.start("r1", PARAMS) as run:
        run.log(step=0, att=1.0, loss=None)
    assert list(exp.metrics()["metric"]) == ["att"]


def test_metrics_are_visible_before_the_run_ends(exp):
    run = exp.start("r1", PARAMS)
    run.log(step=0, att=1.0)
    assert "att" in (run.path / rstore.METRICS).read_text()  # flushed, not buffered
    run.finish()


# --------------------------------------------------------------- aggregation


def test_aggregate_computes_mean_std_and_unique_seeds(exp):
    for seed, att in [(0, 100.0), (1, 110.0), (2, 120.0)]:
        with exp.start(f"c-s{seed}", {"agent": "colight", "seed": seed}) as run:
            run.result(att=att)
    with exp.start("mp-s0", {"agent": "maxpressure", "seed": 0}) as run:
        run.result(att=300.0)

    out = exp.aggregate(by=["agent"], over="seed", metrics=["att"]).set_index("agent")
    assert out.loc["colight", "att_mean"] == pytest.approx(110.0)
    assert out.loc["colight", "att_std"] == pytest.approx(10.0)
    assert out.loc["colight", "n"] == 3
    assert out.loc["maxpressure", "n"] == 1


def test_aggregate_does_not_count_a_repeated_seed_twice(exp):
    for rid in ("a", "b"):
        with exp.start(rid, {"agent": "colight", "seed": 0, "tag": rid}) as run:
            run.result(att=100.0)
    out = exp.aggregate(by=["agent"], over="seed", metrics=["att"])
    assert out.loc[0, "n"] == 1  # two runs, one seed


def test_aggregate_refuses_over_in_by(exp):
    with exp.start("r1", PARAMS):
        pass
    with pytest.raises(ValueError, match="both"):
        exp.aggregate(by=["agent", "seed"], over="seed", metrics=["att"])


def test_empty_experiments_do_not_break_reading(tmp_path):
    empty = Experiment("empty", root=tmp_path / "runs")
    assert empty.runs() == []
    assert empty.results().empty
    assert list(empty.metrics().columns) == ["run_id", "step", "split", "metric", "value"]


# ------------------------------------------------------- write durability


def test_writes_are_atomic_and_leave_no_garbage(tmp_path, monkeypatch):
    """A failed write must not damage the previous file version or leave a
    temporary file behind."""
    path = tmp_path / "meta.json"
    rstore.write_json(path, {"version": 1})

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        rstore.write_json(path, {"version": 2})

    assert json.loads(path.read_text()) == {"version": 1}
    assert list(tmp_path.glob(".*tmp")) == []


def test_a_broken_json_does_not_break_the_scan(exp):
    """A run directory damaged from outside is skipped, not fatal to the query."""
    with exp.start("good", PARAMS) as run:
        run.result(att=1.0)
    bad = exp.path / "bad"
    bad.mkdir()
    (bad / rstore.META).write_text("{ this is not json")

    ids = {r.id for r in exp.runs()}
    assert ids == {"good", "bad"}
    assert [r.id for r in exp.find()] == ["good"]  # the broken one has status "unknown"


def test_two_runs_side_by_side_do_not_collide(exp):
    """No central index = no write race between processes."""
    a = exp.start("a", {**PARAMS, "seed": 0})
    b = exp.start("b", {**PARAMS, "seed": 1})
    a.log(step=0, att=1.0)
    b.log(step=0, att=2.0)
    a.result(att=1.0)
    b.result(att=2.0)
    a.finish()
    b.finish()
    assert {r.id: r.result["att"] for r in exp.find()} == {"a": 1.0, "b": 2.0}


def test_params_hash_ignores_key_order():
    assert params_hash({"a": 1, "b": 2}) == params_hash({"b": 2, "a": 1})
    assert params_hash({"a": 1}) != params_hash({"a": 2})


# ------------------------------------------------------------ provenance


def test_provenance_without_git_does_not_break_the_run(exp, tmp_path):
    with exp.start("r1", PARAMS, repos={"none": tmp_path / "no-repo"}) as run:
        pass
    git = exp.runs()[0].meta["provenance"]["git"]["none"]
    assert git["sha"] is None and git["dirty"] is None
    assert run.meta["provenance"]["python"]


def test_config_snapshot_is_saved_next_to_the_run(exp):
    with exp.start("r1", PARAMS, config={"trainer": {"yellow_length": 0}}) as run:
        pass
    saved = yaml.safe_load((run.path / rstore.CONFIG).read_text())
    assert saved["trainer"]["yellow_length"] == 0
