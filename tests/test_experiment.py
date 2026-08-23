"""Testy adaptera LibSignal → runlog: parsowanie logów silnika i protokół raportowania.

Nie potrzebują symulatora — karmimy je treścią logów, którą LibSignal naprawdę pisze
(`tsc_trainer.writeLog` i handler `_BRF`). To jest miejsce, gdzie protokół „raport =
średnia z ostatnich 10 epizodów testowych" zamienia się w liczbę, więc błąd tutaj jest
niewidoczny w wynikach i widoczny dopiero w recenzji.
"""

from __future__ import annotations

import pytest

from runlog import Experiment
from tsc_marl import experiment as tx

BRF = (
    "Final Travel Time is 316.1114, mean rewards: -447.5272, queue: 10.4976, "
    "delay: 2.7385, throughput: 6151\nTotal time taken: 14.38\n"
)


def dtl(rows):
    """Wiersze w formacie `tsc_trainer.writeLog`: nazwa TAB tryb TAB epizod TAB ATT ..."""
    return "".join(
        f"advancedcolight\t{mode}\t{e}\t{att:.1f}\t0.0\t-1.0\t2.0\t0.3\t100\n"
        for mode, e, att in rows
    )


@pytest.fixture
def logdir(tmp_path):
    d = tmp_path / "logger"
    d.mkdir()
    return d


def test_run_id_jest_czytelny_i_stabilny():
    assert tx.run_id("colight", "cityflow_jinan3x4_4ph", 0, 5) == (
        "colight__cityflow_jinan3x4_4ph__s0_y5"
    )
    assert tx.run_id("colight", "jinan", 1, 0, suffix="drop25") == "colight__jinan__s1_y0__drop25"


def test_parse_final_czyta_ostatni_wynik_z_pliku(logdir):
    (logdir / "2026_08_23-10_00_00_BRF.log").write_text(BRF)
    out = tx.parse_final(logdir)
    assert out == {
        "att": 316.1114,
        "reward": -447.5272,
        "queue": 10.4976,
        "delay": 2.7385,
        "throughput": 6151,
    }


def test_parse_episodes_rozdziela_train_i_test(logdir):
    (logdir / "x_DTL.log").write_text(dtl([("TRAIN", 0, 600.0), ("TEST", 0, 1200.0)]))
    rows = tx.parse_episodes(logdir)
    assert [(r["split"], r["episode"], r["att"]) for r in rows] == [
        ("train", 0, 600.0),
        ("test", 0, 1200.0),
    ]


def test_brak_logow_nie_wywraca_parsowania(logdir):
    assert tx.parse_episodes(logdir) == []
    assert tx.parse_final(logdir) == {}


def test_att_last10_to_srednia_z_dziesieciu_ostatnich_testow(tmp_path, logdir):
    """Sedno protokołu z papieru. Przy niestabilnym uczeniu ostatni epizod potrafi
    być gorszy od średniej o setki sekund — dlatego raportujemy okno, nie punkt."""
    tests = [("TEST", e, 100.0 + e) for e in range(15)]  # 100..114, ostatnie 10 to 105..114
    (logdir / "x_DTL.log").write_text(dtl(tests))
    (logdir / "x_BRF.log").write_text(BRF)

    exp = Experiment("t", root=tmp_path / "runs")
    with exp.start("r", {"a": 1}) as run:
        out = tx.ingest(run, logdir)

    assert out["att_last10"] == pytest.approx(109.5)
    assert out["att_best"] == pytest.approx(100.0)
    assert out["att"] == 316.1114  # z _BRF: końcowy test po treningu, inna liczba niż okno
    assert out["episodes_logged"] == 15


def test_metoda_bez_treningu_ma_last10_rowne_att(tmp_path, logdir):
    """Brak `_DTL.log` (fixedtime, maxpressure, MP-y) — jeden pomiar, więc okno == punkt."""
    (logdir / "x_BRF.log").write_text(BRF)

    exp = Experiment("t", root=tmp_path / "runs")
    with exp.start("r", {"a": 1}) as run:
        out = tx.ingest(run, logdir)

    assert out["att_last10"] == out["att_best"] == out["att"] == 316.1114
    assert out["episodes_logged"] == 0


def test_ingest_zapisuje_metryki_do_biegu(tmp_path, logdir):
    (logdir / "x_DTL.log").write_text(dtl([("TRAIN", 0, 600.0), ("TEST", 0, 1200.0)]))
    (logdir / "x_BRF.log").write_text(BRF)

    exp = Experiment("t", root=tmp_path / "runs")
    with exp.start("r", {"a": 1}) as run:
        tx.ingest(run, logdir)

    df = exp.metrics()
    assert set(df["split"]) == {"train", "test"}
    assert set(df["metric"]) >= {"att", "queue", "throughput"}
    assert exp.runs()[0].result["att_last10"] == 1200.0


# ------------------------------------------------------------------- wykresy


def test_gather_czyta_metrics_csv(tmp_path, logdir):
    """Bieg zakończony — krzywa idzie z `metrics.csv` zapisanego przez ingest."""
    from tsc_marl import plot

    (logdir / "x_DTL.log").write_text(dtl([("TEST", 0, 100.0), ("TEST", 1, 90.0)]))
    (logdir / "x_BRF.log").write_text(BRF)
    exp = Experiment("t", root=tmp_path / "runs")
    with exp.start("r", {"a": 1}) as run:
        tx.ingest(run, logdir)

    assert plot.gather(exp, None, "att", "test") == {"r": [(0, 100.0), (1, 90.0)]}


def test_gather_spada_na_dtl_dla_biegu_w_toku(tmp_path):
    """Bieg TRWA: `metrics.csv` jeszcze nie istnieje (ingest leci na końcu), więc
    krzywa musi przyjść wprost z logu silnika — inaczej 7-godzinnego treningu nie
    da się oglądać w trakcie."""
    from tsc_marl import plot

    exp = Experiment("t", root=tmp_path / "runs")
    run = exp.start(
        "wtoku",
        {"agent": "advancedcolight", "world": "cityflow", "network": "jinan"},
    )
    logdir = run.engine_dir / "tsc" / "cityflow_advancedcolight" / "jinan" / "test" / "logger"
    logdir.mkdir(parents=True)
    (logdir / "x_DTL.log").write_text(dtl([("TEST", 0, 500.0), ("TEST", 1, 480.0)]))

    assert not (run.path / "metrics.csv").exists()
    assert plot.gather(exp, None, "att", "test") == {"wtoku": [(0, 500.0), (1, 480.0)]}


def test_wstega_liczy_sie_tylko_na_wspolnych_epizodach(tmp_path):
    """Biegi o różnej długości: mean±std wolno liczyć wyłącznie tam, gdzie WSZYSTKIE
    seedy mają punkt — inaczej ogon najdłuższego udaje średnią z całej grupy."""
    from tsc_marl import plot

    groups = {"g": [[(0, 10.0), (1, 20.0), (2, 30.0)], [(0, 20.0), (1, 40.0)]]}
    fig = plot.draw_band(groups, "att", "test", "t")
    line = fig.axes[0].lines[0]
    assert list(line.get_xdata()) == [0, 1]  # epizod 2 ma tylko jeden bieg → odpada
    assert list(line.get_ydata()) == [15.0, 30.0]
