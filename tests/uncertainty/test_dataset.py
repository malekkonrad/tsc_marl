"""Testy zrzutu (stan → akcja eksperta) dla prototypu 2.0.

Najważniejszy test tego modułu nie jest testem — jest nim `expert_check`, wołany
przy KAŻDYM zrzucie: jedna liczba sprawdza naraz kolejność cech, incydencję faz,
przenumerowanie akcji `rank -> node_idx` i to, że etykieta pochodzi z czystego
stanu. Testy poniżej pilnują, żeby ta liczba znaczyła to, co ma znaczyć, oraz
żeby warstwa awarii nie przeciekła do trajektorii eksperta.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsc_marl.domain import ats
from tsc_marl.domain.sensors import FailureSpec
from tsc_marl.uncertainty import dataset

W1 = 1.0


def make_x(ep, demand, phase, n_phases=2):
    """Wiersz stanu w układzie `StateBuilder`: blok EP, blok demand, one-hot fazy."""
    row = np.zeros(len(ep) + len(demand) + n_phases, dtype=np.float32)
    row[: len(ep)] = ep
    row[len(ep) : len(ep) + len(demand)] = demand
    row[len(ep) + len(demand) + phase] = 1.0
    return row


IDENTITY = np.eye(2, dtype=bool)[None]  # jedno skrzyżowanie, faza k obsługuje ruch k


def check(rows, y):
    x = np.stack(rows)[:, None, :]  # [T, 1, F]
    return dataset.expert_check(x, np.array(y, dtype=np.int64)[:, None], IDENTITY, W1, 2)


# ------------------------------------------------------------- expert_check


def test_odtwarza_argmax_pressure():
    """Bez zatrzasku demandu Alg. 1 to zwykły argmax p(s)."""
    r = check([make_x([1.0, 5.0], [0.0, 0.0], 0), make_x([7.0, 2.0], [0.0, 0.0], 1)], [1, 0])
    assert r["agreement"] == 1.0
    assert r["tie_rate"] == 0.0


def test_odtwarza_zatrzask_demandu():
    """`d(a_cur)·W1 > max p(s)` utrzymuje BIEŻĄCĄ fazę, mimo że nie ma
    największego pressure — to jest cała różnica Advanced-MP vs Efficient-MP."""
    row = make_x([1.0, 5.0], [9.0, 0.0], phase=0)
    assert check([row], [0])["agreement"] == 1.0  # trzyma fazę 0
    assert check([row], [1])["agreement"] == 0.0  # argmax dałby 1


def test_remis_nie_liczy_sie_do_zgodnosci():
    """Przy równym p(s) o wyniku decyduje kolejność sumowania, a nie stan —
    to nie jest informacja, której naszemu wektorowi brakuje, tylko taka,
    której nie ma nikt. Wpada do `tie_rate`, wypada ze zgodności."""
    r = check([make_x([3.0, 3.0], [0.0, 0.0], 0)], [1])
    assert r["tie_rate"] == 1.0
    assert np.isnan(r["agreement_decidable"])  # brak próbek rozstrzygalnych
    assert r["margin"][0, 0] == pytest.approx(0.0)


def test_granica_zatrzasku_tez_wypada():
    """`d(a_cur)·W1 == max p(s)`: warunek jest OSTRY, więc rozstrzyga ostatni bit."""
    r = check([make_x([4.0, 1.0], [4.0, 0.0], 0)], [0])
    assert r["boundary_rate"] == 1.0


def test_margines_mierzy_ambiwalencje():
    """`tie_margin` to zmienna, nie zastrzeżenie w tekście: plan ostrzega, że
    niepewność DECYZJI myli się z genuiczną ambiwalencją faz (2.2), więc chcemy
    ją mieć zmierzoną obok każdej próbki."""
    r = check([make_x([5.0, 1.0], [0.0, 0.0], 0), make_x([5.0, 4.9], [0.0, 0.0], 0)], [0, 0])
    assert r["margin"][0, 0] == pytest.approx(4.0)
    assert r["margin"][1, 0] == pytest.approx(0.1, abs=1e-5)


# ------------------------------------------------------------------ incydencja


def test_incydencja_pokrywa_dokladnie_ruchy_fazy(roadnet):
    """Most Def. 11 → Def. 5/10. Indeksy muszą wskazywać na `controlled_movements`,
    bo to ta lista wyznacza kolejność kolumn w wektorze stanu."""
    for inter, _ in roadnet:
        flat = ats.controlled_movements(inter)
        per_phase = ats.phase_movements(inter)
        incidence = ats.phase_incidence(inter)
        assert len(incidence) == len(per_phase)
        for phase, idx in enumerate(incidence):
            assert len(idx) == len(per_phase[phase])
            assert len(set(idx)) == len(idx), "ruch policzony w fazie dwa razy"
            assert all(0 <= i < len(flat) for i in idx)
            # ten sam zbiór ruchów, tylko wyrażony indeksami w płaskiej liście
            assert {tuple(map(tuple, flat[i])) for i in idx} == {
                tuple(map(tuple, m)) for m in per_phase[phase]
            }


def test_incydencja_zgadza_sie_z_pressure_fazy(roadnet):
    """p(s) policzone przez incydencję z bloku EP musi dać to samo, co
    `ats.phase_pressure` liczone wprost z kolejek — inaczej rekonstrukcja
    eksperta mierzyłaby własny błąd."""
    rng = np.random.default_rng(0)
    for inter, _ in roadnet:
        flat = ats.controlled_movements(inter)
        lanes = sorted(ats.all_lanes(flat))
        q = dict(zip(lanes, rng.integers(0, 8, len(lanes)).tolist(), strict=True))
        ep = np.array(ats.movement_pressure(flat, q))
        direct = ats.phase_pressure(ats.phase_movements(inter), q)
        via_incidence = [ep[idx].sum() for idx in ats.phase_incidence(inter)]
        assert np.allclose(direct, via_incidence)


# --------------------------------------------------------------- integracja


@pytest.fixture(scope="module")
def dumped():
    """Krótki zrzut na prawdziwym CityFlow: 2 epizody × 30 decyzji (~20 s).
    Dwa epizody, bo to na ich RÓŻNICY stoi najważniejszy test integracyjny."""
    return dataset.collect(episodes=2, decisions=30)


@pytest.mark.slow
def test_zadanie_bc_jest_realizowalne(dumped):
    """Decyzja Advanced-MP jest funkcją DOKŁADNIE tej informacji, którą niesie
    stan ATS. Gdyby zgodność spadła poniżej 1.0 na próbkach rozstrzygalnych,
    znaczyłoby to przesuniętą kolumnę albo złe przenumerowanie akcji — i każdy
    późniejszy wynik prototypu 2.0 mierzyłby ten błąd, nie niepewność."""
    assert dumped.meta["agreement_decidable"] == 1.0


@pytest.mark.slow
def test_awaria_nie_dotyka_eksperta_ani_trajektorii(dumped):
    """Ekspert czyta silnik, a warstwa awarii siedzi w `StateBuilder` — więc
    scenariusz NIE MOŻE zmienić przebiegu symulacji. Epizody różnią się wyłącznie
    losowaniem awarii, więc identyczne `y` i `x_clean` to dowód rozdziału."""
    d = dumped.decisions if hasattr(dumped, "decisions") else dumped.meta["decisions_per_episode"]
    assert np.array_equal(dumped.y[:d], dumped.y[d : 2 * d])
    assert np.array_equal(dumped.x_clean[:d], dumped.x_clean[d : 2 * d])
    assert dumped.meta["att_per_episode"][0] == dumped.meta["att_per_episode"][1]


@pytest.mark.slow
def test_scenariusze_roznia_sie_miedzy_epizodami(dumped):
    """Druga strona tego samego: gdyby awaria była losowana raz na zawsze,
    replikacja w Fazie 3 (N LOSOWAŃ AWARII) nie miałaby czego uśredniać."""
    d = dumped.meta["decisions_per_episode"]
    dead = dumped.dead["mask0p25"]
    assert not np.array_equal(dead[0], dead[d]), "ten sam zbiór martwych detektorów"


@pytest.mark.slow
def test_czysty_scenariusz_jest_czysty(dumped):
    """Wiersz odniesienia: `FailureSpec()` nie ma prawa niczego ruszyć."""
    assert np.array_equal(dumped.x[dataset.CLEAN], dumped.x_clean)
    assert dumped.meta["changed_cells"][dataset.CLEAN] == 0.0
    assert dumped.meta["dead_movements"][dataset.CLEAN] == 0.0


@pytest.mark.slow
def test_martwe_ruchy_to_nie_to_samo_co_zmienione_cechy(dumped):
    """Sedno „cichej awarii": zamaskowany detektor na pustym pasie raportuje 0,
    czyli wartość PRAWDZIWĄ — jest martwy, a cecha się nie zmienia. Dlatego
    ground truth lokalizacji musi iść z warstwy, a nie z różnicy cech."""
    assert dumped.meta["dead_movements"]["mask0p25"] > 3 * dumped.meta["changed_cells"]["mask0p25"]
    assert dumped.meta["dead_movements"]["gauss3p5"] == 0.0  # szum niczego nie uśmierca
    assert dumped.meta["changed_cells"]["gauss3p5"] > 0.5


@pytest.mark.slow
def test_ksztalty_i_zapis(dumped, tmp_path):
    n, f = dumped.meta["n_nodes"], dumped.x_clean.shape[-1]
    t = dumped.meta["decisions_per_episode"] * dumped.meta["episodes"]
    assert dumped.x_clean.shape == (t, n, f)
    assert dumped.y.shape == dumped.phase_time.shape == dumped.tie_margin.shape == (t, n)
    assert dumped.incidence.shape[:2] == (n, dumped.incidence.shape[1])
    assert dumped.edge_index.shape[0] == 2

    loaded = np.load(dumped.save(tmp_path / "d.npz"))
    assert set(loaded.files) >= {"x_clean", "y", "incidence", "tie_margin", "edge_index"}
    assert np.array_equal(loaded["x__mask0p25"], dumped.x["mask0p25"])
    assert np.array_equal(loaded["dead__mask0p25"], dumped.dead["mask0p25"])


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (FailureSpec(), "clean"),
        (FailureSpec(kind="gauss", level=3.5), "gauss3p5"),
        (FailureSpec(kind="mask", level=0.25, unit="intersection"), "mask0p25_intersection"),
    ],
)
def test_slug_nadaje_sie_na_klucz_npz(spec, expected):
    """Klucze npz stają się nazwami plików w archiwum, więc bez kropek."""
    assert dataset.slug(spec) == expected
    assert "." not in dataset.slug(spec)
