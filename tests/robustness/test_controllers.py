"""Testy sterowników pod awarią i planu T2.

Sterowniki MP liczą decyzję z WEKTORA STANU zamiast z silnika — i to jest jedyny
powód, dla którego ich krzywe degradacji w ogóle coś znaczą. Testy pilnują więc
wzorów (Def. 5, Alg. 1, zatrzask t_min) na ręcznie ułożonych stanach, bez CityFlow.
"""

from __future__ import annotations

import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from tsc_marl.domain.sensors import FailureSpec
from tsc_marl.robustness import controllers as c
from tsc_marl.robustness import degrade as d

# 2 skrzyżowania, 2 fazy, 3 ruchy. Faza 0 obsługuje ruchy {0,1}, faza 1 — {2}.
INC = np.zeros((2, 2, 3), dtype=bool)
INC[:, 0, [0, 1]] = True
INC[:, 1, [2]] = True


def inter(phase=0, time=100, n_phases=2):
    return SimpleNamespace(
        current_phase=phase, current_phase_time=time, phases=list(range(n_phases))
    )


def obs(ep, demand=None, n_phases=2):
    """Wiersz stanu: [EP (3) | demand (3) | one-hot fazy]."""
    ep = np.asarray(ep, dtype=np.float32)
    demand = np.zeros_like(ep) if demand is None else np.asarray(demand, dtype=np.float32)
    return np.concatenate([ep, demand, np.zeros((ep.shape[0], n_phases), np.float32)], axis=1)


def test_efficient_mp_wybiera_faze_o_najwiekszym_pressure():
    ctrl = c.PressureController(INC, [inter(), inter()], 3, t_min=15)
    # węzeł 0: p = [1+1, 5] -> faza 1; węzeł 1: p = [3+3, 1] -> faza 0
    x = obs([[1, 1, 5], [3, 3, 1]])
    assert list(ctrl.act(x)) == [1, 0]


def test_zatrzask_minimalnego_zielonego():
    """Faza młodsza niż t_min zostaje, niezależnie od stanu — tak jak w agencie."""
    ctrl = c.PressureController(INC, [inter(phase=0, time=5), inter(phase=0, time=15)], 3, t_min=15)
    x = obs([[0, 0, 9], [0, 0, 9]])
    assert list(ctrl.act(x)) == [0, 1]


def test_advanced_mp_broni_biezacej_fazy_demandem():
    """Alg. 1: d(a_cur)·W1 > max p(s) -> utrzymaj. Warunek OSTRY."""
    ctrl = c.PressureController(INC, [inter(phase=0), inter(phase=0)], 3, t_min=15, w1=1.0)
    x = obs([[0, 0, 4], [0, 0, 4]], demand=[[3, 2, 0], [2, 2, 0]])
    # węzeł 0: d(0)=5 > max p=4 -> trzyma 0; węzeł 1: d(0)=4, NIE > 4 -> argmax p = 1
    assert list(ctrl.act(x)) == [0, 1]


def test_advanced_mp_wymaga_bloku_demand():
    ctrl = c.PressureController(INC, [inter(), inter()], 3, t_min=15, w1=1.0)
    with pytest.raises(ValueError, match="demand"):
        ctrl.act(np.zeros((2, 4), np.float32))


def test_popsuty_odczyt_zmienia_decyzje_mp():
    """Sedno całego modułu: MP pod awarią czyta TEN SAM popsuty stan co polityka.
    Zamaskowany (wyzerowany) ruch przestaje ciągnąć zielone do siebie."""
    ctrl = c.PressureController(INC, [inter(), inter()], 3, t_min=15)
    prawda = obs([[0, 0, 5], [0, 0, 5]])
    maska = prawda.copy()
    maska[0, 2] = 0.0  # martwy detektor na ruchu 2 skrzyżowania 0
    assert list(ctrl.act(prawda)) == [1, 1]
    assert list(ctrl.act(maska)) == [0, 1]


def test_fixedtime_cykluje_po_t_fixed():
    ctrl = c.FixedTimeController([inter(phase=1, time=29), inter(phase=1, time=30)], t_fixed=30)
    assert list(ctrl.act(np.zeros((2, 8)))) == [1, 0]
    assert ctrl.state_independent


def test_colight_odmawia_przy_innej_kolejnosci():
    with pytest.raises(ValueError, match="kolejność"):
        c.ColightController(object(), identity_order=False)


# ------------------------------------------------------------------ plan


def plan_file(tmp_path, extra=""):
    p = tmp_path / "probny.yml"
    p.write_text(
        textwrap.dedent(
            """
            networks: [siec_a, siec_b]
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


def test_plan_dokleja_czysty_scenariusz_na_poczatek(tmp_path):
    plan = d.load_plan(plan_file(tmp_path))
    assert [d.slug(s) for s in plan.scenarios] == ["clean", "gauss1", "mask0p5_hold_t10"]


def test_plan_seedy_tylko_dla_uczonych(tmp_path):
    """Reguły są deterministyczne — N seedów dałoby N identycznych liczb."""
    plan = d.load_plan(plan_file(tmp_path))
    units = plan.units()
    assert sum(u.controller == "gatppo" for u in units) == 4  # 2 sieci × 2 seedy
    assert sum(u.controller == "efficientmp" for u in units) == 2
    assert {u.seed for u in units if u.controller != "gatppo"} == {0}


def test_plan_rodzic_i_run_id(tmp_path):
    plan = d.load_plan(plan_file(tmp_path))
    unit = d.Unit("gatppo", "siec_a", 1)
    assert plan.parent_id(unit) == "gatppo__siec_a__s1_y5"
    assert plan.parent_id(d.Unit("efficientmp", "siec_a", 0)) is None
    assert plan.run_id(unit).endswith("__probny")


def test_fixedtime_liczony_tylko_na_czystym(tmp_path):
    plan = d.load_plan(plan_file(tmp_path))
    specs = plan.scenarios_for(d.Unit("fixedtime", "siec_a", 0))
    assert [d.slug(s) for s in specs] == ["clean"]


def plan_uw(tmp_path, agent: str = "gatppo"):
    """Plan z dodatkową kolumną UW. Wstrzyknięte PODMIANĄ w blok `controllers`,
    bo `extra` w `plan_file` dokleja się na końcu pliku, czyli pod `scenarios`."""
    p = plan_file(tmp_path)
    p.write_text(
        p.read_text().replace(
            "fixedtime: {}",
            "fixedtime: {}\n"
            f"  {agent}_uw:\n"
            f"    agent: {agent}\n"
            "    seeds: [0, 1]\n"
            "    uw: {tau: 0.25}",
        )
    )
    return p


def test_plan_uw_to_osobny_rodzaj_ale_ten_sam_rodzic(tmp_path):
    """Kolumna UW jedzie TYM SAMYM checkpointem co rdzeń — modulacja jest
    bezparametrowa, więc różnica między kolumnami to wyłącznie mechanizm."""
    plan = d.load_plan(plan_uw(tmp_path))
    assert plan.kind("gatppo_uw") == "policy_uw" and plan.kind("gatppo") == "policy"
    assert plan.parent_id(d.Unit("gatppo_uw", "siec_a", 1)) == "gatppo__siec_a__s1_y5"
    assert len([u for u in plan.units() if u.controller == "gatppo_uw"]) == 4  # 2 sieci × 2 seedy


def test_uw_szuka_wag_tam_gdzie_rdzen(tmp_path):
    """Regresja. `checkpoint()` rozgałęzia się po `kind`, a nie po agencie — gdy
    `policy_uw` wypadnie z tej gałęzi, leci do układu katalogów LibSignala, którego
    rdzeń nigdy nie zapisuje. Punkt jest wtedy POMIJANY z komunikatem „brak wag",
    czyli plan liczy się dalej BEZ kolumny, dla której powstał."""
    from tsc_marl.runner.degrade import checkpoint  # noqa: PLC0415

    plan = d.load_plan(plan_uw(tmp_path))
    parent = SimpleNamespace(path=tmp_path / "rodzic", params={"network": "siec_a"})
    sciezki = {
        ctrl: checkpoint(plan, d.Unit(ctrl, "siec_a", 0), parent)
        for ctrl in ("gatppo", "gatppo_uw")
    }
    assert sciezki["gatppo_uw"] == sciezki["gatppo"] == tmp_path / "rodzic" / "engine" / "core.pt"


def test_plan_uw_tylko_z_actorem_rdzenia(tmp_path):
    with pytest.raises(SystemExit, match="UW-attention"):
        d.load_plan(plan_uw(tmp_path, agent="efficientmp"))


# --------------------------------------- jawna kolumna odniesienia (`baseline:`)


def plan_trzy_ramiona(tmp_path, baseline: str = "gatppo_fail"):
    """Plan w układzie `t2_train_uw`: DWIE kolumny bez mechanizmu (trening na
    czystym i trening z awariami) plus jedna z nim."""
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


def test_baseline_wskazuje_ramie_kontrolne(tmp_path):
    """Bez `baseline:` raport sparowałby kolumnę UW z PIERWSZĄ kolumną `policy`,
    czyli z treningiem na czystym — i mierzyłby trening z awariami RAZEM
    z mechanizmem zamiast samego mechanizmu. Tabela wychodzi, tylko odpowiada
    na inne pytanie; nic tego nie zgłasza."""
    from tsc_marl.analysis.degradation import mechanism_pairs  # noqa: PLC0415

    plan = d.load_plan(plan_trzy_ramiona(tmp_path))
    assert plan.baseline("gatppo_uw") == "gatppo_fail"
    assert mechanism_pairs(plan) == [("gatppo_uw", "gatppo_fail")]

    bez = d.load_plan(plan_uw(tmp_path))
    assert bez.baseline("gatppo_uw") is None
    assert mechanism_pairs(bez) == [("gatppo_uw", "gatppo")]  # domyśl zostaje


def test_baseline_musi_istniec(tmp_path):
    with pytest.raises(SystemExit, match="baseline"):
        d.load_plan(plan_trzy_ramiona(tmp_path, baseline="nie_ma_takiego"))
    with pytest.raises(SystemExit, match="sam siebie"):
        d.load_plan(plan_trzy_ramiona(tmp_path, baseline="gatppo_uw"))


def test_plan_t2_train_uw_sie_laduje():
    """Plan z repo, nie z tmp_path: trzy ramiona, jawne odniesienie, rodzice
    wskazujący na suite'y `p1_fail` / `p1_uw`."""
    from tsc_marl.analysis.degradation import mechanism_pairs  # noqa: PLC0415

    plan = d.load_plan("t2_train_uw")
    assert plan.kind("gatppo_fail_uw") == "policy_uw"
    assert plan.kind("gatppo_fail") == "policy"
    assert mechanism_pairs(plan) == [("gatppo_fail_uw", "gatppo_fail")]
    unit = d.Unit("gatppo_fail_uw", "cityflow_jinan3x4_4ph", 1)
    assert plan.parent_id(unit) == "gatppo__cityflow_jinan3x4_4ph__s1_y5__p1_uw"
    inne = d.Unit("gatppo_fail", "cityflow_jinan3x4_4ph", 1)
    assert plan.parent_id(inne) == "gatppo__cityflow_jinan3x4_4ph__s1_y5__p1_fail"
    # ramię 1 zostaje przy domyślnym rodzicu, czyli biegu z `p1_gate`
    assert plan.parent_id(d.Unit("gatppo", "cityflow_jinan3x4_4ph", 1)) == (
        "gatppo__cityflow_jinan3x4_4ph__s1_y5"
    )


def test_plan_odrzuca_nieznany_sterownik_i_pole(tmp_path):
    p = plan_file(tmp_path)
    p.write_text(p.read_text().replace("fixedtime: {}", "fixedtime: {}\n  rakieta: {}"))
    with pytest.raises(SystemExit, match="sterownik"):
        d.load_plan(p)
    p.write_text(p.read_text() + "literowka: 1\n")
    with pytest.raises(SystemExit, match="nieznane"):
        d.load_plan(p)


# ------------------------------------------------------------------ pętla


class FakeEnv:
    """Liczy wywołania; ATT = liczba kroków + numer losowania (rozróżnialne wiersze)."""

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


def test_evaluate_czysty_raz_awaria_n_razy():
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
