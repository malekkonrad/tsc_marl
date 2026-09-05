"""Testy agenta Advanced-CoLight — wymagają postawionego CityFlow, więc `slow`.

Pilnują KONTRAKTU między `get_ob()` a `_build_model()`: układu wiersza
obserwacji i jego szerokości. Rozjazd tych dwóch rzeczy wywala się głośno
(`nn.Linear` rzuci błędem kształtu), ale przestawienie bloków WEWNĄTRZ wiersza
— już nie: sieć nauczy się czegokolwiek, tylko gorzej, i nie da się tego
odróżnić od "no i tyle wychodzi".
"""

from __future__ import annotations

import numpy as np
import pytest

from tsc_marl.domain import ats

pytestmark = pytest.mark.slow

ACTION_INTERVAL = 15


def test_ob_length_zgadza_sie_z_siecia(advanced_colight_world):
    """Szerokość wiersza == wejście pierwszej warstwy modelu. Jedyne miejsce,
    gdzie wymiar obserwacji wchodzi do architektury."""
    _, model = advanced_colight_world
    structs = model._ats_structs()
    n_mov = max(len(s.movements) for s in structs)

    assert model.ob_length == 2 * n_mov + model.action_space.n
    assert model.ob_length == 20  # JiNan: 8 pressure + 8 running + 4 fazy
    assert model.model.embedding_MLP.embedding_node[0].in_features == model.ob_length
    assert model.target_model.embedding_MLP.embedding_node[0].in_features == model.ob_length


def test_ksztalt_i_typ_obserwacji(advanced_colight_world):
    world, model = advanced_colight_world
    ob = model.get_ob()
    assert ob.shape == (len(world.intersections), model.ob_length)
    assert ob.dtype == np.float32


def test_one_hot_wskazuje_biezaca_faze_wlasciwego_skrzyzowania(advanced_colight_world):
    """Dwa kontrakty naraz: (1) w bloku fazy jest dokładnie jedna jedynka,
    (2) wiersz i opisuje `world.intersections[i]` — czyli kolejność wierszy
    zgadza się z kolejnością, w której indeksowany jest graf sąsiedztwa."""
    world, model = advanced_colight_world
    n_mov = max(len(s.movements) for s in model._ats_structs())

    # różne fazy na różnych skrzyżowaniach, żeby test nie przeszedł przypadkiem
    actions = [i % len(inter.phases) for i, inter in enumerate(world.intersections)]
    for _ in range(ACTION_INTERVAL):
        world.step(actions)

    ob = model.get_ob()
    phase_block = ob[:, 2 * n_mov :]
    assert np.array_equal(phase_block.sum(axis=1), np.ones(len(world.intersections)))
    assert len(set(np.argmax(phase_block, axis=1).tolist())) > 1, "wszystkie fazy identyczne"
    for idx, inter in enumerate(world.intersections):
        assert np.argmax(phase_block[idx]) == inter.current_phase, inter.id


def test_bloki_wiersza_zgadzaja_sie_z_prymitywami_ats(advanced_colight_world):
    """Niezależne przeliczenie e(l,m) i r(l,m) prosto z `ats` — łapie przestawienie
    bloków, przesunięcie o jeden i pomylenie pressure z demand."""
    world, model = advanced_colight_world
    n_mov = max(len(s.movements) for s in model._ats_structs())
    obs_length = 167  # jak w configu

    for _ in range(ACTION_INTERVAL):
        world.step([0] * len(world.intersections))
    ob = model.get_ob()

    eng = world.eng
    lane_vehicles, distances, speeds = (
        eng.get_lane_vehicles(),
        eng.get_vehicle_distance(),
        eng.get_vehicle_speed(),
    )
    for idx, s in enumerate(model._ats_structs()):
        q = ats.windowed_queue(
            s.all_lanes, lane_vehicles, distances, speeds, world.lane_length, obs_length
        )
        running = ats.effective_running(
            s.in_lanes, lane_vehicles, distances, speeds, world.lane_length, s.effective_range
        )
        expected_p = ats.movement_pressure(s.movements, q)
        expected_r = ats.movement_running(s.movements, running)

        assert ob[idx, : len(expected_p)] == pytest.approx(expected_p), idx
        assert ob[idx, n_mov : n_mov + len(expected_r)] == pytest.approx(expected_r), idx


def test_demand_nieujemny_a_pressure_bywa_ujemne(advanced_colight_world):
    """Sanity check semantyki: liczba pojazdów nie może być ujemna, a pressure
    to różnica — jeśli NIGDY nie schodzi poniżej zera, znaczy że zgubiliśmy
    człon downstream i cały spillback jest niewidoczny."""
    world, model = advanced_colight_world
    n_mov = max(len(s.movements) for s in model._ats_structs())

    seen_negative = False
    for _ in range(20):
        for _ in range(ACTION_INTERVAL):
            world.step([1] * len(world.intersections))
        ob = model.get_ob()
        assert (ob[:, n_mov : 2 * n_mov] >= 0).all()
        seen_negative |= bool((ob[:, :n_mov] < 0).any())
    assert seen_negative, "pressure nigdy nie było ujemne — sprawdź człon downstream"
