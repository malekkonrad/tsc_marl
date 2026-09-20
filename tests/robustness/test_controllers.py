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


def plan_uw(tmp_path, agent: str = "gatppo", tau: float = 0.25):
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
            f"    uw: {{tau: {tau}}}",
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


def test_sciezka_wag_nie_zalezy_od_rodzaju_sterownika(tmp_path, monkeypatch):
    """Przyczyna źródłowa pułapki wyżej: format pliku z wagami zależy od AGENTA
    (kto trenował), a nie od rodzaju sterownika (jak zbudować sterownik z tych wag).

    Dopóki warunek w `checkpoint()` był wyliczanką rodzajów, każda nowa nakładka na
    rdzeń — ensemble, UW-attention, bramki Fazy 3 — musiała pamiętać, żeby się do niej
    dopisać, a zapomnienie kończyło się CICHYM pominięciem całej kolumny. Test udaje
    rodzaj, którego jeszcze nie ma: wagi i tak mają się rozwiązać na `core.pt`."""
    from tsc_marl.runner.degrade import checkpoint  # noqa: PLC0415

    plan = d.load_plan(plan_uw(tmp_path))
    monkeypatch.setattr(type(plan), "kind", lambda self, ctrl: "policy_jeszcze_nie_istnieje")
    parent = SimpleNamespace(path=tmp_path / "rodzic", params={"network": "siec_a"})
    unit = d.Unit("gatppo_uw", "siec_a", 0)
    assert checkpoint(plan, unit, parent) == tmp_path / "rodzic" / "engine" / "core.pt"


def test_odcisk_biegu_widzi_parametry_mechanizmu(tmp_path):
    """Zmiana τ (albo progu bramki) MUSI dać inny odcisk biegu.

    `is_done` porównuje `params_hash`, a nie treść planu. Dopóki parametry
    mechanizmu były poza odciskiem, sweep po τ liczył pierwszą wartość, a każdą
    następną pomijał z komunikatem „pomijam (jest wynik)" — i zwracał trzy razy tę
    samą liczbę pod trzema etykietami. To ten sam błąd, przed którym ostrzega
    docstring `FailureSpec`: spec żyjący poza odciskiem nie unieważnia wyniku.

    Kolumna BEZ mechanizmu ma zostać nietknięta — inaczej dopisanie pola
    unieważniłoby policzone biegi rdzenia."""
    from runlog import params_hash  # noqa: PLC0415
    from tsc_marl.runner.degrade import unit_params  # noqa: PLC0415

    rodzic = [SimpleNamespace(id="rodzic", meta={"params_hash": "abc"})]

    def odcisk(plan, ctrl):
        return params_hash(unit_params(plan, d.Unit(ctrl, "siec_a", 0), rodzic))

    katalogi = [tmp_path / "a", tmp_path / "b"]
    for k in katalogi:
        k.mkdir()
    maly = d.load_plan(plan_uw(katalogi[0], tau=0.25))
    duzy = d.load_plan(plan_uw(katalogi[1], tau=0.50))
    assert odcisk(maly, "gatppo_uw") != odcisk(duzy, "gatppo_uw")
    assert odcisk(maly, "gatppo") == odcisk(duzy, "gatppo")


def test_plan_uw_tylko_z_actorem_rdzenia(tmp_path):
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


def test_plan_bramka_to_osobny_rodzaj_ale_ten_sam_rodzic(tmp_path):
    """Jak przy UW: kolumna z bramką jedzie TYM SAMYM checkpointem co rdzeń, bo
    bramka nie zmienia wag — przełącza tylko, czy akcja węzła idzie z polityki."""
    plan = d.load_plan(plan_gate(tmp_path))
    assert plan.kind("gatppo_gate") == "policy_gate"
    assert plan.parent_id(d.Unit("gatppo_gate", "siec_a", 1)) == "gatppo__siec_a__s1_y5"
    assert d.KINDS[plan.agent("gatppo_gate")] == "policy"  # stąd wagi w `core.pt`


def test_plan_odrzuca_bramke_z_literowka_w_kind(tmp_path):
    """Nieznany `kind` nie ma prawa po cichu zbudować wyroczni: to ona jest górnym
    ograniczeniem, więc pomylona z bramką realną zawyżyłaby wynik o różnicę między
    idealną a rzeczywistą detekcją — czyli o to, co ten plan ma zmierzyć."""
    with pytest.raises(SystemExit, match="kind"):
        d.load_plan(plan_gate(tmp_path, kind="wyrocznia"))


def test_plan_bramka_tylko_z_actorem_rdzenia(tmp_path):
    p = plan_gate(tmp_path)
    p.write_text(p.read_text().replace("agent: gatppo", "agent: efficientmp"))
    with pytest.raises(SystemExit, match="bramka"):
        d.load_plan(p)


def test_sweep_progu_daje_rozne_odciski(tmp_path):
    """Sweep stoi i upada na odcisku: cztery kolumny różnią się WYŁĄCZNIE progiem,
    więc bez `MECHANISM_KEYS` w `params_hash` miałyby ten sam odcisk co kolumna
    policzona wcześniej i `is_done` pominąłby je z komunikatem „jest wynik" —
    dostałbyś krzywą zapasu złożoną z czterech kopii tej samej liczby.

    Druga asercja pilnuje odwrotnej strony: kolumna BEZ mechanizmu ma odcisk
    niezależny od progów, czyli policzone punkty rdzenia zostają policzone."""
    from runlog import params_hash  # noqa: PLC0415
    from tsc_marl.runner.degrade import unit_params  # noqa: PLC0415

    rodzic = [SimpleNamespace(id="rodzic", meta={"params_hash": "abc"})]
    plany = []
    for i, prog in enumerate((0.25, 0.5, 0.75, 1.0)):
        katalog = tmp_path / f"p{i}"
        katalog.mkdir()
        plany.append(d.load_plan(plan_gate(katalog, threshold=prog)))

    odciski = {
        params_hash(unit_params(plan, d.Unit("gatppo_gate", "siec_a", 0), rodzic)) for plan in plany
    }
    assert len(odciski) == 4

    rdzen = {
        params_hash(unit_params(plan, d.Unit("gatppo", "siec_a", 0), rodzic)) for plan in plany
    }
    assert len(rdzen) == 1


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
    # OBA odczyty z nagłówka planu mają wiersz w raporcie: ramię 2 vs 1 (ile daje sam
    # trening z awariami) i ramię 3 vs 2 (ile daje mechanizm ponad to).
    assert mechanism_pairs(plan) == [
        ("gatppo_fail", "gatppo"),
        ("gatppo_fail_uw", "gatppo_fail"),
    ]
    unit = d.Unit("gatppo_fail_uw", "cityflow_jinan3x4_4ph", 1)
    assert plan.parent_id(unit) == "gatppo__cityflow_jinan3x4_4ph__s1_y5__p1_uw"
    inne = d.Unit("gatppo_fail", "cityflow_jinan3x4_4ph", 1)
    assert plan.parent_id(inne) == "gatppo__cityflow_jinan3x4_4ph__s1_y5__p1_fail"
    # ramię 1 zostaje przy domyślnym rodzicu, czyli biegu z `p1_gate`
    assert plan.parent_id(d.Unit("gatppo", "cityflow_jinan3x4_4ph", 1)) == (
        "gatppo__cityflow_jinan3x4_4ph__s1_y5"
    )


def test_gola_kolumna_z_baseline_dostaje_wiersz(tmp_path):
    """Regresja: kolumna BEZ mechanizmu inferencyjnego, ale z jawnym `baseline:`,
    musi trafić do tabeli Δ.

    Tak wypadł z raportu `t2_train_uw` najmocniejszy wynik planu — ramię trenowane
    z awariami (goły `policy`) bije rdzeń o 140 s przy `mask0p5`, a `mechanism_pairs`
    przepuszczał wyłącznie `policy_uw` / `policy_gate`. Plan się liczył, `baseline:`
    walidowało się poprawnie, tabela po prostu nie powstawała — cisza nie do odróżnienia
    od braku efektu."""
    from tsc_marl.analysis.degradation import mechanism_pairs  # noqa: PLC0415

    plan = d.load_plan(plan_trzy_ramiona(tmp_path))
    assert plan.kind("gatppo_fail") == "policy"
    assert plan.baseline("gatppo_fail") is None
    assert ("gatppo_fail", "gatppo") not in mechanism_pairs(plan)

    z_baza = d.load_plan(
        _write(
            tmp_path / "z_baza.yml",
            plan_trzy_ramiona(tmp_path)
            .read_text()
            .replace(
                "    parent: '{agent}__{network}__s{seed}_y{yellow}__p1_fail'",
                "    parent: '{agent}__{network}__s{seed}_y{yellow}__p1_fail'\n"
                "    baseline: gatppo",
            ),
        )
    )
    assert ("gatppo_fail", "gatppo") in mechanism_pairs(z_baza)


def test_sweep_tylko_przy_jednym_odniesieniu(tmp_path):
    """`_sweep_section` składa JEDNĄ tabelę wokół JEDNEJ kolumny bazowej (`pary[0][1]`).
    Plan o kilku odniesieniach (`t2_gate_dr`) wpuszczony w tę gałąź liczyłby bramkę
    na rdzeniu DR przeciw rdzeniowi CZYSTEMU — czyli bramkę razem z treningiem
    z awariami, dokładnie ten błąd, przed którym powstało `Plan.baseline`."""
    from tsc_marl.analysis.degradation import _sweep_section, mechanism_pairs  # noqa: PLC0415

    plan = d.load_plan("t2_gate_dr")
    pary = mechanism_pairs(plan)
    assert len({baza for _, baza in pary}) == 2, pary
    with pytest.raises(ValueError, match="jedno odniesienie"):
        _sweep_section(plan, None, pary)

    # sweepy jednego parametru zostają sweepami — warunek nie może ich zepsuć
    for nazwa in ("t2_gate", "t2_gate_real", "t2_uw_tau"):
        bazy = {baza for _, baza in mechanism_pairs(d.load_plan(nazwa))}
        assert bazy == {"gatppo"}, (nazwa, bazy)


def test_plan_t2_gate_dr_sie_laduje():
    """Kwadrat 2×2 {rdzeń czysty, rdzeń DR} × {bez bramki, z bramką}. Trzy odczyty:
    sama bramka, sam trening z awariami, bramka PONAD treningiem z awariami."""
    from tsc_marl.analysis.degradation import mechanism_pairs  # noqa: PLC0415

    plan = d.load_plan("t2_gate_dr")
    assert plan.kind("gatppo_fail_gate_rule") == "policy_gate"
    assert mechanism_pairs(plan) == [
        ("gatppo_gate_rule", "gatppo"),
        ("gatppo_fail", "gatppo"),
        ("gatppo_fail_gate_rule", "gatppo_fail"),
    ]
    # obie kolumny bramkowane MUSZĄ czytać ten sam detektor — inaczej różnica między
    # nimi miesza bramkę z jej rekalibracją pod nową politykę
    assert (
        plan.controllers["gatppo_gate_rule"]["gate"]
        == (plan.controllers["gatppo_fail_gate_rule"]["gate"])
    )
    # złożenie i sam DR biorą wagi z tego samego suite'u `p1_fail`
    for kol in ("gatppo_fail", "gatppo_fail_gate_rule"):
        unit = d.Unit(kol, "cityflow_jinan3x4_4ph", 2)
        assert plan.parent_id(unit) == "gatppo__cityflow_jinan3x4_4ph__s2_y5__p1_fail"
    # rdzeń czysty zostaje przy domyślnym rodzicu (bieg bez sufiksu suite'u)
    assert plan.parent_id(d.Unit("gatppo", "cityflow_jinan3x4_4ph", 2)) == (
        "gatppo__cityflow_jinan3x4_4ph__s2_y5"
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


# -------------------------------------------------- bramka fallbacku (wyrocznia)

POLITYKA, REGULA = 3, 7  # akcje muszą się RÓŻNIĆ, inaczej testy przechodzą trywialnie


def features(broken=(), *, live=True, sensors=True, n=4, prefix=""):
    """Atrapa `StateBuilder`: tylko to, co czyta bramka. `sensors=False` = czysty
    scenariusz (warstwa awarii powstaje wyłącznie dla aktywnego `FailureSpec`).
    `prefix` zmienia NAZWY pasów przy tej samej liczbie węzłów — tak testuje się
    unieważnienie cache pas→węzeł."""
    return SimpleNamespace(
        n_nodes=n,
        structs=[SimpleNamespace(in_lanes=[f"{prefix}{i}_{c}" for c in "ab"]) for i in range(n)],
        sensors=SimpleNamespace(broken=frozenset(broken), live=live) if sensors else None,
    )


def gated(env, *, trace=False, **kwargs):
    """`kwargs` idą do WYROCZNI (próg), `trace` do powłoki — to różne obiekty."""
    polityka = SimpleNamespace(act=lambda obs: np.full(env.features.n_nodes, POLITYKA))
    regula = SimpleNamespace(act=lambda obs: np.full(env.features.n_nodes, REGULA))
    return c.GatedController(polityka, regula, c.OracleGate(env, **kwargs), trace=trace)


# ------------------------------------------------- action masking (Faza 4)


def masked(n_nodes=2, n_phases=4, legalne=None, k_max=3, akcja=0):
    """Polityka-atrapa wybierająca ZAWSZE `akcja` — czyli głodząca wszystko inne.
    `legalne[i]` = ile faz ma naprawdę węzeł i."""
    legalne = legalne or [n_phases] * n_nodes
    legal = np.zeros((n_nodes, n_phases), dtype=bool)
    for i, k in enumerate(legalne):
        legal[i, :k] = True
    polityka = SimpleNamespace(act=lambda obs: np.full(n_nodes, akcja))
    return c.MaskedController(polityka, legal=legal, k_max=k_max)


def test_maska_wymusza_zaglodzona_faze():
    """Polityka trzyma fazę 0 w nieskończoność. Przy k_max=3 maska musi wejść
    najpóźniej w 4. decyzji i oddać zielone fazie, która czeka najdłużej."""
    m = masked(n_nodes=1, n_phases=3, k_max=3)
    wybrane = [int(m.act(None)[0]) for _ in range(8)]
    assert wybrane[:3] == [0, 0, 0], "przed progiem maska milczy"
    assert set(wybrane[3:]) != {0}, "po progu musi wymusić coś innego"
    assert m.episode_stats()["mask_forced"] > 0


def test_maska_trzyma_GWARANCJE_wieku_obslugi():
    """H1 planu `t2_mask` i jedyny test POPRAWNOŚCI, nie wyniku. Przy kilku fazach
    przeterminowanych naraz obsłużyć można jedną, więc gwarancja nie jest twardym
    `k_max`, a `k_max + liczba_legalnych_faz − 1`. Ten test liczy faktyczny wiek
    krok po kroku i sprawdza tę granicę — gdyby wybór szedł po INDEKSIE zamiast po
    wieku, ostatnia faza zostałaby zagłodzona na zawsze i granica pękłaby."""
    n_faz, k_max = 4, 3
    m = masked(n_nodes=1, n_phases=n_faz, k_max=k_max)
    wiek = np.zeros(n_faz, dtype=int)
    najstarszy = 0
    for _ in range(200):
        a = int(m.act(None)[0])
        wiek += 1
        wiek[a] = 0
        najstarszy = max(najstarszy, int(wiek.max()))
    assert najstarszy <= k_max + n_faz - 1, f"wiek {najstarszy} > granicy {k_max + n_faz - 1}"


def test_maska_NIE_wymusza_fazy_nieistniejacej():
    """Ta sama pułapka co w metrykach ogona: `n_phases` to maksimum po sieci, więc
    węzeł o 2 fazach ma resztę kolumn martwych. Wymuszenie takiej kolumny oznaczałoby
    akcję poza przestrzenią decyzji tego skrzyżowania — na sieci niejednorodnej
    (Kraków: 2–7 faz) sterowanie w pustkę."""
    m = masked(n_nodes=1, n_phases=5, legalne=[2], k_max=2)
    wybrane = {int(m.act(None)[0]) for _ in range(40)}
    assert wybrane <= {0, 1}, f"maska wyszła poza legalne fazy: {wybrane}"


def test_maska_liczy_tylko_NADPISANIA_nie_zgodnosci():
    """`mask_share` musi mierzyć nadpisania, nie aktywność — mechanizm, który się
    zgodził z polityką, jest wobec wyniku no-opem. Ta sama zasada co `gate_share`."""
    # polityka sama obraca fazami, więc nic nie ma prawa się przeterminować
    licznik = {"i": -1}

    def obrot(_obs):
        licznik["i"] += 1
        return np.array([licznik["i"] % 2])

    m = c.MaskedController(SimpleNamespace(act=obrot), legal=np.array([[True, True]]), k_max=3)
    for _ in range(20):
        m.act(None)
    assert m.episode_stats() == {"mask_share": 0.0, "mask_forced": 0.0}


def test_maska_zeruje_wiek_na_starcie_epizodu():
    """`reset_stats` woła `robustness/degrade.py` na starcie KAŻDEGO epizodu. Bez
    zerowania pierwsze decyzje nowego epizodu wymuszałyby fazy z zaległości poprzedniego
    — ten sam błąd, przed którym broni się okno bramki UQ."""
    m = masked(n_nodes=1, n_phases=3, k_max=3)
    for _ in range(10):
        m.act(None)
    m.reset_stats()
    assert [int(m.act(None)[0]) for _ in range(3)] == [0, 0, 0]


def test_maska_odrzuca_bezsensowne_k_max():
    with pytest.raises(ValueError, match="k_max"):
        masked(k_max=0)


def test_plan_t2_mask_sie_laduje():
    from tsc_marl.analysis.degradation import mechanism_pairs  # noqa: PLC0415

    plan = d.load_plan("t2_mask")
    assert plan.kind("gatppo_mask_k16") == "policy_mask"
    # cztery ograniczenia, wszystkie przeciw rdzeniowi -> jedna zbiorcza tabela
    assert {b for _, b in mechanism_pairs(plan)} == {"gatppo"}
    assert len(mechanism_pairs(plan)) == 4


def test_plan_odrzuca_maske_bez_k_max(tmp_path):
    """`k_max` JEST całym mechanizmem tej maski. Literówka w nazwie klucza dałaby
    po cichu wartość domyślną, a sweep po ograniczeniu policzyłby jedną kolumnę
    i pominął trzy jako „jest wynik" — ta sama pułapka co przy τ i progach bramki."""
    p = plan_file(tmp_path)
    p.write_text(
        p.read_text().replace(
            "  fixedtime: {}",
            "  gatppo_m:\n    agent: gatppo\n    mask: {kmax: 8}\n  fixedtime: {}",
        )
    )
    with pytest.raises(SystemExit, match="k_max"):
        d.load_plan(p)


def test_slad_bramki_wyjasnia_POJEDYNCZE_przelaczenie():
    """Instrumentacja pod explainability (Faza 4). `gate_share` mówi ILE par
    (decyzja, węzeł) przejęła reguła — nie mówi KTÓRE ani KIEDY, a wyjaśnienie
    wdrożeniowe brzmi „dlaczego TO skrzyżowanie oddało sterowanie". Ślad zapisuje
    wynik SUROWY obok obu masek, bo bez niego nie widać, czy węzeł był tuż pod
    progiem, a bez rozdzielenia surowa/po histerezie nie da się odróżnić
    „detektor nie zadziałał" od „histereza jeszcze nie przepuściła"."""
    env = SimpleNamespace(features=features({"1_a", "1_b"}))
    sterownik = gated(env, trace=True)
    sterownik.act(None)
    slad = sterownik.episode_trace()

    assert len(slad) == 4, "jeden wiersz na węzeł na decyzję"
    assert [w[0] for w in slad] == [0] * 4  # numer decyzji
    assert [w[1] for w in slad] == [0, 1, 2, 3]  # numer węzła
    # węzeł 1 ma oba wloty martwe -> wynik surowy 1,0 i tryb reguły; reszta 0,0
    assert [w[2] for w in slad] == [0.0, 1.0, 0.0, 0.0]
    assert [w[4] for w in slad] == [False, True, False, False]


def test_slad_domyslnie_WYLACZONY():
    """Domyślnie pusty, bo 240 decyzji × N węzłów × scenariusze × losowania to
    dane, których nikt nie zamawiał. Włącza go `tsc degrade run --trace`."""
    env = SimpleNamespace(features=features({"1_a", "1_b"}))
    sterownik = gated(env)
    sterownik.act(None)
    assert sterownik.episode_trace() == []


def test_trace_NIE_wchodzi_do_odcisku_biegu():
    """PUŁAPKA, dla której `--trace` jest flagą CLI, a nie kluczem w planie.
    `gate` należy do `MECHANISM_KEYS`, więc dopisanie tam flagi LOGOWANIA
    unieważniłoby wszystkie policzone punkty — mimo że nie zmienia ani jednej
    decyzji. Ten test pilnuje, żeby nikt nie „uprościł" tego z powrotem do planu."""
    from tsc_marl.robustness.degrade import MECHANISM_KEYS  # noqa: PLC0415

    assert "trace" not in MECHANISM_KEYS
    plan = d.load_plan("t2_gate_real")
    assert "trace" not in plan.controllers["gatppo_gate_rule"]["gate"]


def test_bramka_na_czystym_nie_rusza_polityki():
    """Na czystym scenariuszu `sensors is None` i sterownik ma być NIEODRÓŻNIALNY
    od samej polityki — ten sam warunek co `test_no_op` dla UW-attention, i z tego
    samego powodu: bez niego wynik na czystym nie jest porównywalny z T1, a każda
    różnica pod awarią przestaje być przypisywalna do bramki."""
    env = SimpleNamespace(features=features(sensors=False))
    sterownik = gated(env)
    assert sterownik.act(None).tolist() == [POLITYKA] * 4
    assert sterownik.episode_stats() == {"gate_share": 0.0, "gate_switches": 0.0}


def test_bramka_trafia_w_osleplony_wezel_a_nie_w_sasiada():
    """Sedno kroku. Węzeł 1 ma OBA pasy wjazdowe martwe (2/2), węzeł 2 jeden (1/2),
    reszta zero — przy progu 0,5 przełączyć mają się 1 i 2.

    Test nie przechodzi trywialnie, bo akcje polityki i reguły są różne (3 vs 7):
    gdyby maska była pusta albo odwrócona, wektor wyjściowy byłby inny. Gdyby oba
    sterowniki zwracały to samo, asercja byłaby prawdziwa dla KAŻDEJ maski."""
    env = SimpleNamespace(features=features({"1_a", "1_b", "2_a"}))
    assert gated(env).act(None).tolist() == [POLITYKA, REGULA, REGULA, POLITYKA]


def test_prog_jest_nieostry_i_dziala_w_obie_strony():
    """Rozstrzygnięcie remisu z docstringu `OracleGate`: `>=`, czyli „co najmniej
    połowa detektorów martwa". Przy `unit: lane` i `mask0p5` co czwarty węzeł siada
    dokładnie na tej granicy, więc konwencja ma tu realne skutki — i ma być
    sprawdzona, a nie tylko opisana. Próg jest parametrem (sweep), więc druga
    asercja pilnuje, że faktycznie działa, a nie jest ozdobą konstruktora."""
    env = SimpleNamespace(features=features({"2_a"}))  # węzeł 2: dokładnie 1/2
    assert gated(env).act(None).tolist() == [POLITYKA, POLITYKA, REGULA, POLITYKA]
    assert gated(env, threshold=0.6).act(None).tolist() == [POLITYKA] * 4


def test_bramka_milczy_przed_onsetem():
    """`SensorModel.reset()` losuje `broken` na starcie epizodu, ale przed `onset`
    `_measure` zwraca CZYSTE wartości — detektory jeszcze żyją. Bramka pytająca
    tylko o `broken` przełączyłaby się od kroku 0 i scenariusze `mask*_hold_t60`
    mierzyłyby coś innego, niż jest w planie (a są tam celowo: `hold` to awaria
    bez śladu, na której rule-gate nie wygra trywialnie)."""
    env = SimpleNamespace(features=features({"1_a", "1_b"}, live=False))
    assert gated(env).act(None).tolist() == [POLITYKA] * 4


def test_bramka_widzi_podmiane_scenariusza():
    """`robustness/degrade.py` podmienia `env.features` przy KAŻDYM scenariuszu,
    a sterownik powstaje raz. Gdyby bramka zapamiętała `features` albo `sensors`
    w konstruktorze, czytałaby do końca biegu warstwę z pierwszego scenariusza —
    a pierwszy jest zawsze `clean` (`load_plan` dokleja go na początek), czyli
    bramka nie strzeliłaby ANI RAZU i wyrocznia wyszłaby „bez efektu".

    Drugi przebieg ma INNE NAZWY PASÓW przy tej samej liczbie węzłów: gdyby cache
    mapowania pas→węzeł przeżył podmianę, żaden pas z nowego `broken` by się do
    niego nie dopasował i maska wyszłaby pusta. Liczba węzłów zostaje, bo scenariusz
    podmienia warstwę awarii, a nie sieć."""
    env = SimpleNamespace(features=features({"1_a", "1_b"}))
    sterownik = gated(env)
    assert sterownik.act(None).tolist() == [POLITYKA, REGULA, POLITYKA, POLITYKA]

    env.features = features({"X0_a", "X0_b"}, prefix="X")
    assert sterownik.act(None).tolist() == [REGULA, POLITYKA, POLITYKA, POLITYKA]


def test_reset_stats_czysci_takze_bramke():
    """Delegacja, która przy wyroczni jest no-opem, a przy bramce UQ będzie
    warunkiem poprawności: tamta trzyma okno ostatnich odczytów i bez czyszczenia
    weszłaby w nowy epizod z danymi sprzed `world.reset()`. `degrade.evaluate`
    woła `reset_stats` na starcie epizodu — to jedyny moment, w którym bramka
    może się dowiedzieć, że epizod się zmienił."""
    wolania = []
    bramka = SimpleNamespace(
        mask=lambda obs: np.zeros(2, dtype=bool), reset=lambda: wolania.append(1)
    )
    polityka = SimpleNamespace(act=lambda obs: np.full(2, POLITYKA))
    sterownik = c.GatedController(polityka, polityka, bramka)
    assert wolania == [1]  # konstruktor woła reset_stats

    sterownik.act(None)
    sterownik.reset_stats()
    assert wolania == [1, 1] and sterownik.episode_stats() == {}


def test_przelaczenia_sa_liczone_na_wezlach():
    """`gate_switches` liczy ZMIANY trybu, bo to one psują ruch (przełączenie
    resetuje fazę), a nie sam odsetek węzłów w regule. Bez tej metryki raport nie
    odróżni bramki stabilnej od migoczącej przy tym samym `gate_share`."""
    env = SimpleNamespace(features=features({"1_a", "1_b"}))
    sterownik = gated(env)
    sterownik.act(None)
    env.features = features({"2_a", "2_b"})  # jeden węzeł wchodzi, drugi wychodzi
    sterownik.act(None)
    assert sterownik.episode_stats() == {"gate_share": 0.25, "gate_switches": 2.0}


# ------------------------------------------- bramki realne: reguła, UQ, histereza

N_MOV, OKNO = 2, 3


def obserwacje(kroki, *, zamrozony=()):
    """[T, N=2, 5]: 2 ruchy × 2 cechy + one-hot. Węzeł 0 zdrowy (odczyty rosną),
    węzeł 1 ma zamrożone ruchy podane w `zamrozony`."""
    x = np.zeros((kroki, 2, 5), np.float32)
    for t in range(kroki):
        x[t, :, :4] = t + 1.0
        for m in zamrozony:
            x[t, 1, m] = 7.0  # EP ruchu m stoi
            x[t, 1, N_MOV + m] = 0.0  # demand ruchu m stoi (zero)
    return x


def test_regula_widzi_zamrozony_detektor_a_nie_zdrowy():
    """Kryterium reguły: WSZYSTKIE cechy ruchu stoją przez całe okno. Węzeł 1 ma oba
    ruchy zamrożone (ułamek 1,0), węzeł 0 żaden (0,0) — przy progu 0,5 przełącza się
    tylko ten pierwszy. Akcje obu sterowników są różne, więc test nie przechodzi
    trywialnie."""
    bramka = c.RuleGate(window=OKNO, threshold=0.5, n_mov=N_MOV)
    x = obserwacje(OKNO, zamrozony=(0, 1))
    for t in range(OKNO):
        maska = bramka.mask(x[t])
    assert maska.tolist() == [False, True]


def test_regula_milczy_na_rozgrzewce():
    """Przy niepełnym buforze okno jest przycięte i KAŻDY ruch wygląda na zamrożony —
    bez strażnika bramka przełączyłaby całą sieć na starcie każdego epizodu. Ta sama
    pułapka zatruła kalibrację progu (kwantyl wychodził 1,0), stąd `full_window`."""
    bramka = c.RuleGate(window=OKNO, threshold=0.5, n_mov=N_MOV)
    x = obserwacje(OKNO, zamrozony=(0, 1))
    assert bramka.mask(x[0]).tolist() == [False, False]  # 1 z 3 odczytów
    assert bramka.mask(x[1]).tolist() == [False, False]  # 2 z 3
    assert bramka.mask(x[2]).tolist() == [False, True]  # pełne okno


def test_regula_czysci_bufor_na_resecie():
    bramka = c.RuleGate(window=OKNO, threshold=0.5, n_mov=N_MOV)
    x = obserwacje(OKNO, zamrozony=(0, 1))
    for t in range(OKNO):
        bramka.mask(x[t])
    bramka.reset()
    assert bramka.mask(x[0]).tolist() == [False, False]


def test_uq_porownuje_ruchy_z_wlasna_norma():
    """Bramka UQ ma tę samą POSTAĆ sygnału co reguła — ułamek podejrzanych ruchów —
    więc ablacja porównuje kryterium, a nie dwie różne statystyki."""
    u = np.array([[0.1, 0.1], [0.9, 0.9]])  # węzeł 1 powyżej normy w obu ruchach
    online = SimpleNamespace(movements=lambda obs: u, ready=True, reset=lambda: None)
    bramka = c.UQGate(online, u_mov_ref=np.full((2, 2), 0.5), threshold=0.5)
    assert bramka.mask(None).tolist() == [False, True]

    online.ready = False  # rozgrzewka — jak w regule
    assert bramka.mask(None).tolist() == [False, False]


def test_histereza_opoznia_wejscie_i_wyjscie():
    """`k_on` kolejnych decyzji na wejście, `k_off` na wyjście. Bez tego bramka
    o szumiącym sygnale migocze co decyzję, a każde przełączenie resetuje fazę.
    Pierwsza decyzja przechodzi bez wygładzania — nie ma czego wygładzać."""
    sygnal = iter(
        [
            [False],  # start: polityka
            [True],  # 1. powyżej progu — za mało (k_on=2)
            [True],  # 2. — wchodzi
            [False],  # 1. poniżej — za mało (k_off=3)
            [False],  # 2. — wciąż reguła
            [False],  # 3. — wychodzi
        ]
    )
    bramka = SimpleNamespace(mask=lambda obs: np.array(next(sygnal)), reset=lambda: None)
    polityka = SimpleNamespace(act=lambda obs: np.array([POLITYKA]))
    regula = SimpleNamespace(act=lambda obs: np.array([REGULA]))
    sterownik = c.GatedController(polityka, regula, bramka, k_on=2, k_off=3)

    akcje = [int(sterownik.act(None)[0]) for _ in range(6)]
    assert akcje == [POLITYKA, POLITYKA, REGULA, REGULA, REGULA, POLITYKA]
    assert sterownik.episode_stats()["gate_switches"] == 2.0


def test_diagnostyka_detekcji_liczy_sie_wzgledem_wyroczni():
    """Precyzja i czułość rozdzielają „detektor jest słaby" od „przełączanie mało
    daje". Wyrocznia jest tu WYŁĄCZNIE diagnostyką — jej maska nie może wpłynąć na
    akcje, więc test sprawdza też, że decyzja idzie z bramki właściwej."""
    bramka = SimpleNamespace(mask=lambda obs: np.array([True, True, False]), reset=lambda: None)
    wyrocznia = SimpleNamespace(mask=lambda obs: np.array([True, False, True]), reset=lambda: None)
    polityka = SimpleNamespace(act=lambda obs: np.full(3, POLITYKA))
    regula = SimpleNamespace(act=lambda obs: np.full(3, REGULA))
    sterownik = c.GatedController(polityka, regula, bramka, oracle=wyrocznia)

    assert sterownik.act(None).tolist() == [REGULA, REGULA, POLITYKA]  # maska BRAMKI
    stats = sterownik.episode_stats()
    assert stats["gate_precision"] == pytest.approx(0.5)  # 1 trafiony z 2 wskazanych
    assert stats["gate_recall"] == pytest.approx(0.5)  # 1 z 2 faktycznie zepsutych


def test_summarize_usrednia_metryki_obecne_tylko_w_czesci_losowan():
    """`gate_precision`/`gate_recall` są emitowane WARUNKOWO — przy losowaniu, w którym
    nie było czego wykrywać, mianownik nie istnieje i klucza nie ma. Sprawdzanie
    obecności w pierwszym wierszu grupy zakładało jednolity komplet kluczy i kończyło
    się `KeyError` po policzeniu CAŁEGO punktu: 35 epizodów przeliczonych, wynik
    wyrzucony w ostatniej linii (zmierzone na gliderze, 6 punktów)."""
    rows = [
        {"scenario": "mask0p5", "draw": 0, "att": 700.0, "gate_recall": 0.4},
        {"scenario": "mask0p5", "draw": 1, "att": 800.0},  # brak mianownika
    ]
    out = d.summarize(rows)
    assert out["att__mask0p5"] == pytest.approx(750.0)
    assert out["gate_recall__mask0p5"] == pytest.approx(0.4)  # średnia z OKREŚLONYCH
