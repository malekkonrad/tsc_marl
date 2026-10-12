"""LibSignal -> runlog adapter: parsing engine logs and the reporting protocol.

No simulator needed: the tests feed log content LibSignal really writes
(`tsc_trainer.writeLog` and the `_BRF` handler). This is where "report = mean of
the last 10 test episodes" becomes a number, so an error here would be invisible
in the results.
"""

from __future__ import annotations

import pytest

from runlog import Experiment
from tsc_marl.libsignal import experiment as tx

BRF = (
    "Final Travel Time is 316.1114, mean rewards: -447.5272, queue: 10.4976, "
    "delay: 2.7385, throughput: 6151\nTotal time taken: 14.38\n"
)


def dtl(rows):
    """Rows in `tsc_trainer.writeLog` format: name TAB mode TAB episode TAB ATT ..."""
    return "".join(
        f"advancedcolight\t{mode}\t{e}\t{att:.1f}\t0.0\t-1.0\t2.0\t0.3\t100\n"
        for mode, e, att in rows
    )


@pytest.fixture
def logdir(tmp_path):
    d = tmp_path / "logger"
    d.mkdir()
    return d


def test_run_id_is_readable_and_stable():
    assert tx.run_id("colight", "cityflow_jinan3x4_4ph", 0, 5) == (
        "colight__cityflow_jinan3x4_4ph__s0_y5"
    )
    assert tx.run_id("colight", "jinan", 1, 0, suffix="drop25") == "colight__jinan__s1_y0__drop25"


def test_parse_final_reads_the_last_result_from_the_file(logdir):
    (logdir / "2026_08_23-10_00_00_BRF.log").write_text(BRF)
    out = tx.parse_final(logdir)
    assert out == {
        "att": 316.1114,
        "reward": -447.5272,
        "queue": 10.4976,
        "delay": 2.7385,
        "throughput": 6151,
    }


def test_parse_episodes_separates_train_and_test(logdir):
    (logdir / "x_DTL.log").write_text(dtl([("TRAIN", 0, 600.0), ("TEST", 0, 1200.0)]))
    rows = tx.parse_episodes(logdir)
    assert [(r["split"], r["episode"], r["att"]) for r in rows] == [
        ("train", 0, 600.0),
        ("test", 0, 1200.0),
    ]


def test_missing_logs_do_not_break_parsing(logdir):
    assert tx.parse_episodes(logdir) == []
    assert tx.parse_final(logdir) == {}


def test_att_last10_is_the_mean_of_the_last_ten_tests(tmp_path, logdir):
    """The paper's protocol: with unstable learning the last episode can be hundreds
    of seconds off the mean, so a window is reported, not a point."""
    tests = [("TEST", e, 100.0 + e) for e in range(15)]  # 100..114, last 10 = 105..114
    (logdir / "x_DTL.log").write_text(dtl(tests))
    (logdir / "x_BRF.log").write_text(BRF)

    exp = Experiment("t", root=tmp_path / "runs")
    with exp.start("r", {"a": 1}) as run:
        out = tx.ingest(run, logdir)

    assert out["att_last10"] == pytest.approx(109.5)
    assert out["att_best"] == pytest.approx(100.0)
    assert out["att"] == 316.1114  # from _BRF: the final test, another number than the window
    assert out["episodes_logged"] == 15


def test_a_method_without_training_has_last10_equal_to_att(tmp_path, logdir):
    """No `_DTL.log` (fixedtime, max-pressure methods): one measurement, window == point."""
    (logdir / "x_BRF.log").write_text(BRF)

    exp = Experiment("t", root=tmp_path / "runs")
    with exp.start("r", {"a": 1}) as run:
        out = tx.ingest(run, logdir)

    assert out["att_last10"] == out["att_best"] == out["att"] == 316.1114
    assert out["episodes_logged"] == 0


def test_ingest_writes_metrics_into_the_run(tmp_path, logdir):
    (logdir / "x_DTL.log").write_text(dtl([("TRAIN", 0, 600.0), ("TEST", 0, 1200.0)]))
    (logdir / "x_BRF.log").write_text(BRF)

    exp = Experiment("t", root=tmp_path / "runs")
    with exp.start("r", {"a": 1}) as run:
        tx.ingest(run, logdir)

    df = exp.metrics()
    assert set(df["split"]) == {"train", "test"}
    assert set(df["metric"]) >= {"att", "queue", "throughput"}
    assert exp.runs()[0].result["att_last10"] == 1200.0


# ------------------------------------------------------------------- plots


def test_gather_reads_metrics_csv(tmp_path, logdir):
    """A finished run: the curve comes from the `metrics.csv` written by ingest."""
    from tsc_marl.analysis import plot

    (logdir / "x_DTL.log").write_text(dtl([("TEST", 0, 100.0), ("TEST", 1, 90.0)]))
    (logdir / "x_BRF.log").write_text(BRF)
    exp = Experiment("t", root=tmp_path / "runs")
    with exp.start("r", {"a": 1}) as run:
        tx.ingest(run, logdir)

    assert plot.gather(exp, None, "att", "test") == {"r": [(0, 100.0), (1, 90.0)]}


def test_gather_falls_back_to_dtl_for_a_run_in_progress(tmp_path):
    """A RUNNING run has no `metrics.csv` yet (ingest runs at the end), so the curve
    comes straight from the engine log."""
    from tsc_marl.analysis import plot

    exp = Experiment("t", root=tmp_path / "runs")
    run = exp.start(
        "running",
        {"agent": "advancedcolight", "world": "cityflow", "network": "jinan"},
    )
    logdir = run.engine_dir / "tsc" / "cityflow_advancedcolight" / "jinan" / "test" / "logger"
    logdir.mkdir(parents=True)
    (logdir / "x_DTL.log").write_text(dtl([("TEST", 0, 500.0), ("TEST", 1, 480.0)]))

    assert not (run.path / "metrics.csv").exists()
    assert plot.gather(exp, None, "att", "test") == {"running": [(0, 500.0), (1, 480.0)]}


def test_the_band_uses_only_shared_episodes(tmp_path):
    """Runs of different length: mean +- std only where ALL seeds have a point,
    otherwise the longest run's tail poses as the group mean."""
    from tsc_marl.analysis import plot

    groups = {"g": [[(0, 10.0), (1, 20.0), (2, 30.0)], [(0, 20.0), (1, 40.0)]]}
    fig = plot.draw_band(groups, "att", "test", "t")
    line = fig.axes[0].lines[0]
    assert list(line.get_xdata()) == [0, 1]  # episode 2 has only one run, dropped
    assert list(line.get_ydata()) == [15.0, 30.0]
