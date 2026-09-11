"""Testy `runlog` — bez silnika, bez `tsc_marl`, bez sieci.

To jest test narzędzia, nie projektu: gdyby `runlog` wylądował kiedyś w osobnym
repo, ten plik jedzie razem z nim i musi przechodzić bez niczego z tej pracy.
"""

from __future__ import annotations

import json
import os

import pytest

from runlog import Experiment, RunExists, RunIdCollision, params_hash
from runlog import store as rstore

PARAMS = {"agent": "colight", "network": "jinan", "seed": 0}


@pytest.fixture
def exp(tmp_path):
    return Experiment("t1", root=tmp_path / "runs")


# --------------------------------------------------------------- cykl życia


def test_udany_bieg_konczy_sie_statusem_ok(exp):
    with exp.start("r1", PARAMS) as run:
        run.result(att=299.2)
    view = exp.runs()[0]
    assert view.status == "ok"
    assert view.result["att"] == 299.2
    assert view.meta["duration_s"] >= 0


def test_wyjatek_w_bloku_zapisuje_failed_i_nie_polyka_bledu(exp):
    with pytest.raises(ValueError, match="symulator padł"), exp.start("r1", PARAMS) as run:
        run.log(step=0, att=1.0)
        raise ValueError("symulator padł")

    view = exp.runs()[0]
    assert view.status == "failed"
    assert "symulator padł" in view.meta["error"]
    # metryki sprzed awarii muszą przetrwać — inaczej nie da się zdiagnozować, gdzie się wywalił
    assert (view.path / rstore.METRICS).read_text().count("\n") == 2


def test_status_running_widac_w_trakcie(exp):
    run = exp.start("r1", PARAMS)
    assert exp.runs()[0].status == "running"
    run.finish()
    assert exp.runs()[0].status == "ok"


# ------------------------------------------------------------------ kolizje


def test_ten_sam_id_inne_parametry_to_blad_a_nie_nadpisanie(exp):
    with exp.start("r1", PARAMS) as run:
        run.result(att=1.0)

    with pytest.raises(RunIdCollision, match="INNYMI parametrami"):
        exp.start("r1", {**PARAMS, "seed": 1})

    # stary bieg nietknięty
    assert exp.runs()[0].result["att"] == 1.0


def test_policzony_bieg_nie_liczy_sie_drugi_raz(exp):
    with exp.start("r1", PARAMS):
        pass
    with pytest.raises(RunExists):
        exp.start("r1", PARAMS)


def test_zywy_bieg_tez_blokuje(exp):
    """Status `running` może być procesem, który właśnie liczy — nie wolno go zdeptać."""
    exp.start("r1", PARAMS)
    with pytest.raises(RunExists):
        exp.start("r1", PARAMS)


def _dead_pid() -> int:
    """Pid procesu, który na pewno już nie żyje."""
    import subprocess
    import sys

    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def _set_meta(exp, run_id, **fields):
    path = exp.path / run_id / rstore.META
    meta = json.loads(path.read_text())
    meta.update(fields)
    path.write_text(json.dumps(meta))


def test_osierocony_bieg_wolno_policzyc_od_nowa(exp):
    """`running` zapisany na tej maszynie przez proces, którego już nie ma (kill -TERM,
    restart) nigdy się nie domknie — bez tej reguły blokowałby wznowienie siatki."""
    exp.start("r1", PARAMS)
    _set_meta(exp, "r1", pid=_dead_pid())
    with exp.start("r1", PARAMS) as run:
        run.result(att=1.0)
    assert exp.runs()[0].status == "ok"


def test_running_z_innej_maszyny_albo_bez_pid_nadal_blokuje(exp):
    """Ostrożnie: cudzej maszyny nie sprawdzimy, a stare metadane nie mają pid."""
    exp.start("r1", PARAMS)
    _set_meta(exp, "r1", pid=_dead_pid(), host="inna-maszyna")
    with pytest.raises(RunExists):
        exp.start("r1", PARAMS)
    _set_meta(exp, "r1", pid=None, host=None)
    with pytest.raises(RunExists):
        exp.start("r1", PARAMS)


def test_nieudany_bieg_wolno_powtorzyc(exp):
    with pytest.raises(RuntimeError), exp.start("r1", PARAMS):
        raise RuntimeError("boom")
    with exp.start("r1", PARAMS) as run:  # bez replace — wznowienie po awarii jest legalne
        run.result(att=2.0)
    assert exp.runs()[0].status == "ok"


def test_replace_nadpisuje_swiadomie(exp):
    with exp.start("r1", PARAMS) as run:
        run.result(att=1.0)
    with exp.start("r1", {**PARAMS, "seed": 9}, replace=True) as run:
        run.result(att=2.0)
    assert exp.runs()[0].result["att"] == 2.0


# ------------------------------------------------------------- wyszukiwanie


def test_find_domyslnie_widzi_tylko_skonczone(exp):
    with exp.start("ok1", PARAMS):
        pass
    exp.start("wisi", {**PARAMS, "seed": 1})  # zostaje w running
    with pytest.raises(RuntimeError), exp.start("zly", {**PARAMS, "seed": 2}):
        raise RuntimeError

    assert [r.id for r in exp.find()] == ["ok1"]
    assert len(exp.find(status=None)) == 3
    assert [r.id for r in exp.find(seed=1, status=None)] == ["wisi"]


def test_is_done_reaguje_na_zmiane_parametrow(exp):
    with exp.start("r1", PARAMS):
        pass
    assert exp.is_done("r1", PARAMS)
    assert not exp.is_done("r1", {**PARAMS, "lr": 0.001})  # inny config → policz ponownie
    assert not exp.is_done("innego-nie-ma", PARAMS)


# ----------------------------------------------------------------- metryki


def test_metryki_w_formacie_dlugim_i_sklejane_miedzy_biegami(exp):
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


def test_none_nie_trafia_do_metryk(exp):
    """Metody bez treningu nie mają `loss` — brak wartości to brak wiersza,
    a nie wiersz z pustką psujący agregację."""
    with exp.start("r1", PARAMS) as run:
        run.log(step=0, att=1.0, loss=None)
    assert list(exp.metrics()["metric"]) == ["att"]


def test_metryki_widac_zanim_bieg_sie_skonczy(exp):
    run = exp.start("r1", PARAMS)
    run.log(step=0, att=1.0)
    assert "att" in (run.path / rstore.METRICS).read_text()  # flush, nie bufor
    run.finish()


# --------------------------------------------------------------- agregacja


def test_aggregate_liczy_mean_std_i_unikalne_seedy(exp):
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


def test_aggregate_nie_liczy_powtorzonego_seeda_dwa_razy(exp):
    for rid in ("a", "b"):
        with exp.start(rid, {"agent": "colight", "seed": 0, "tag": rid}) as run:
            run.result(att=100.0)
    out = exp.aggregate(by=["agent"], over="seed", metrics=["att"])
    assert out.loc[0, "n"] == 1  # dwa biegi, ale jeden seed


def test_aggregate_odmawia_gdy_over_jest_w_by(exp):
    with exp.start("r1", PARAMS):
        pass
    with pytest.raises(ValueError, match="jednocześnie"):
        exp.aggregate(by=["agent", "seed"], over="seed", metrics=["att"])


def test_puste_eksperymenty_nie_wywracaja_odczytu(tmp_path):
    empty = Experiment("pusty", root=tmp_path / "runs")
    assert empty.runs() == []
    assert empty.results().empty
    assert list(empty.metrics().columns) == ["run_id", "step", "split", "metric", "value"]


# ------------------------------------------------------- trwałość zapisu


def test_zapis_jest_atomowy_i_nie_zostawia_smieci(tmp_path, monkeypatch):
    """Nieudany zapis nie może uszkodzić poprzedniej wersji pliku ani zostawić
    pliku tymczasowego — inaczej zabity proces psuje cały katalog biegu."""
    path = tmp_path / "meta.json"
    rstore.write_json(path, {"wersja": 1})

    def boom(*args, **kwargs):
        raise OSError("dysk pełny")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="dysk pełny"):
        rstore.write_json(path, {"wersja": 2})

    assert json.loads(path.read_text()) == {"wersja": 1}
    assert list(tmp_path.glob(".*tmp")) == []


def test_uszkodzony_json_nie_wywraca_skanu(exp):
    """Katalog biegu może zostać uszkodzony z zewnątrz; skan ma go pominąć,
    a nie wysypać całe odpytanie."""
    with exp.start("dobry", PARAMS) as run:
        run.result(att=1.0)
    zly = exp.path / "zly"
    zly.mkdir()
    (zly / rstore.META).write_text("{ to nie jest json")

    ids = {r.id for r in exp.runs()}
    assert ids == {"dobry", "zly"}
    assert [r.id for r in exp.find()] == ["dobry"]  # uszkodzony ma status "unknown"


def test_dwa_biegi_obok_siebie_nie_koliduja(exp):
    """Brak centralnego indeksu = brak wyścigu o zapis między procesami."""
    a = exp.start("a", {**PARAMS, "seed": 0})
    b = exp.start("b", {**PARAMS, "seed": 1})
    a.log(step=0, att=1.0)
    b.log(step=0, att=2.0)
    a.result(att=1.0)
    b.result(att=2.0)
    a.finish()
    b.finish()
    assert {r.id: r.result["att"] for r in exp.find()} == {"a": 1.0, "b": 2.0}


def test_params_hash_jest_niezalezny_od_kolejnosci_kluczy():
    assert params_hash({"a": 1, "b": 2}) == params_hash({"b": 2, "a": 1})
    assert params_hash({"a": 1}) != params_hash({"a": 2})


# ------------------------------------------------------------ proweniencja


def test_proweniencja_bez_gita_nie_wywraca_biegu(exp, tmp_path):
    with exp.start("r1", PARAMS, repos={"nic": tmp_path / "nie-ma-repo"}) as run:
        pass
    git = exp.runs()[0].meta["provenance"]["git"]["nic"]
    assert git["sha"] is None and git["dirty"] is None
    assert run.meta["provenance"]["python"]


def test_snapshot_configu_zapisuje_sie_obok_biegu(exp):
    with exp.start("r1", PARAMS, config={"trainer": {"yellow_length": 0}}) as run:
        pass
    import yaml

    saved = yaml.safe_load((run.path / rstore.CONFIG).read_text())
    assert saved["trainer"]["yellow_length"] == 0
