"""Reports: a run variant (run_id suffix) must not mix with the method.

The `p1_abl_*` ablations share agent, network, seed and yellow with the core; without
the suffix `tsc table` would silently mix them into the core row of T1 and the
`gatppo` side of the GATE verdict.
"""

from __future__ import annotations

import pytest

from runlog import Experiment
from tsc_marl.analysis import report


@pytest.mark.parametrize(
    ("run_id", "variant"),
    [
        ("gatppo__cityflow_jinan3x4_4ph__s0_y5", ""),
        ("gatppo__cityflow_jinan3x4_4ph__s2_y5__p1_abl_meanpool", "p1_abl_meanpool"),
        ("advancedcolight__cityflow_jinan3x4__s0_y0__sanity60", "sanity60"),
        ("odd_id", ""),
    ],
)
def test_variant_from_the_suffix(run_id, variant):
    assert report.variant_of(run_id) == variant


def _exp(tmp_path, agent="gatppo"):
    exp = Experiment("t1", root=tmp_path / "runs")
    net = "cityflow_jinan3x4_4ph"
    for seed, att in enumerate((300.0, 302.0, 301.0)):
        base = {"agent": agent, "network": net, "seed": seed, "yellow_length": 5}
        with exp.start(f"{agent}__{net}__s{seed}_y5", base) as run:
            run.result(att=att, att_last10=att)
        with exp.start(f"{agent}__{net}__s{seed}_y5__p1_abl_nognn", {**base, "x": 1}) as run:
            run.result(att=att + 30, att_last10=att + 30)
    return exp


def test_training_results_has_a_variant_column(tmp_path):
    df = report.training_results(_exp(tmp_path))
    assert sorted(df["variant"].unique()) == ["", "p1_abl_nognn"]
    assert (df["variant"] == "").sum() == 3


def test_baseline_averages_seeds_and_skips_variants(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    # the baseline lists only agents on its own list; CoLight is on it
    monkeypatch.setitem(report.LITERATURE, "cityflow_jinan3x4_4ph", {"colight": 300.0})
    rows = report._records(_exp(tmp_path, agent="colight"))
    captured = {}
    monkeypatch.setattr("builtins.print", lambda s: captured.setdefault("out", s))
    report.build_baseline_report(rows)
    text = (tmp_path / "t1_baseline.md").read_text()
    # core: mean 301.0 over 3 seeds; the ablation (331) must not raise it
    assert "301.0 (n=3)" in text
    assert "331" not in text


def test_ablation_report_computes_delta_against_the_core(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    report.build_ablation_report(_exp(tmp_path))
    text = (tmp_path / "p1_ablations.md").read_text()
    assert "GAT (core)" in text and "no GNN (MLP)" in text
    assert "+9.97%" in text  # 331 / 301 - 1
