"""Testy raportów — wariant biegu (sufiks run_id) nie może mieszać się z metodą.

Ablacje `p1_abl_*` mają tego samego agenta, sieć, seed i yellow co rdzeń. Bez
rozróżnienia po sufiksie `tsc table` wmieszałby je do wiersza rdzenia w T1
i do strony `gatppo` w werdykcie GATE — cicho, bo liczby nadal wyglądają sensownie.
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
        ("dziwny_id", ""),
    ],
)
def test_wariant_z_sufiksu(run_id, variant):
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


def test_training_results_ma_kolumne_wariantu(tmp_path):
    df = report.training_results(_exp(tmp_path))
    assert sorted(df["variant"].unique()) == ["", "p1_abl_nognn"]
    assert (df["variant"] == "").sum() == 3


def test_baseline_usrednia_seedy_i_pomija_warianty(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    # baseline wypisuje tylko agentów ze swojej listy — CoLight na niej jest
    monkeypatch.setitem(report.LITERATURE, "cityflow_jinan3x4_4ph", {"colight": 300.0})
    rows = report._records(_exp(tmp_path, agent="colight"))
    captured = {}
    monkeypatch.setattr("builtins.print", lambda s: captured.setdefault("out", s))
    report.build_baseline_report(rows)
    text = (tmp_path / "t1_baseline.md").read_text()
    # rdzeń: średnia 301,0 z 3 seedów; ablacja (331) nie może podbić tej liczby
    assert "301.0 (n=3)" in text
    assert "331" not in text


def test_raport_ablacji_liczy_delte_wzgledem_rdzenia(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    report.build_ablation_report(_exp(tmp_path))
    text = (tmp_path / "p1_ablations.md").read_text()
    assert "GAT (rdzeń)" in text and "bez GNN (MLP)" in text
    assert "+9.97%" in text  # 331 / 301 − 1
