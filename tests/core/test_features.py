"""Testy stanu i nagrody rdzenia (`features.StateBuilder`).

Dlaczego akurat te: żaden błąd w tym module się nie wywala. Przestawiony blok
w wierszu, pomylony wariant pressure z EP, nagroda liczona po innym zbiorze
pasów — wszystko to daje sieć, która się uczy, tylko gorzej. Stąd nacisk na
niezależne przeliczenia i na porównanie z upstreamem tam, gdzie się da.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsc_marl.core.features import REWARD_VARIANTS, STATE_VARIANTS, StateBuilder
from tsc_marl.domain import ats

ACTION_INTERVAL = 15


def test_odrzuca_nieznany_wariant():
    """Walidacja przed dotknięciem świata — literówka w YAML-u ma się wywalić
    przy budowie, nie po 200 epizodach dziwnych wyników."""
    with pytest.raises(ValueError, match="state="):
        StateBuilder(None, [], state="ATS")  # wielkość liter ma znaczenie
    with pytest.raises(ValueError, match="reward="):
        StateBuilder(None, [], reward="delay")


@pytest.mark.slow
def test_szerokosc_wiersza_per_wariant(advanced_colight_world):
    """Tylko ATS ma dwa bloki cech (pressure + demand); reszta po jednym."""
    world, _ = advanced_colight_world
    for variant in STATE_VARIANTS:
        b = StateBuilder.from_registry(world, world.intersections, state=variant)
        expected = (2 if variant == "ats" else 1) * b.n_mov + b.n_phases
        assert b.ob_length == expected, variant
        assert b.observe().shape == (len(world.intersections), expected), variant
        assert b.observe().dtype == np.float32, variant


@pytest.mark.slow
def test_one_hot_zawsze_na_koncu_wiersza(advanced_colight_world):
    """Blok fazy to OSTATNIE `n_phases` kolumn w KAŻDYM wariancie — bloki są
    dopychane zerami osobno, więc padding nigdy nie przesuwa one-hota."""
    world, _ = advanced_colight_world
    actions = [i % len(inter.phases) for i, inter in enumerate(world.intersections)]
    for _ in range(ACTION_INTERVAL):
        world.step(actions)

    for variant in STATE_VARIANTS:
        b = StateBuilder.from_registry(world, world.intersections, state=variant)
        phase_block = b.observe()[:, -b.n_phases :]
        assert np.array_equal(phase_block.sum(axis=1), np.ones(b.n_nodes)), variant
        assert len(set(np.argmax(phase_block, axis=1).tolist())) > 1, "wszystkie fazy identyczne"
        for idx, inter in enumerate(world.intersections):
            assert np.argmax(phase_block[idx]) == inter.current_phase, (variant, inter.id)


@pytest.mark.slow
def test_ep_to_pierwszy_blok_ats(advanced_colight_world):
    """ATS = EP + demand, więc pierwszy blok musi być identyczny. Gdyby się
    rozjechał, jeden z wariantów liczy pressure inaczej niż drugi."""
    world, _ = advanced_colight_world
    for _ in range(ACTION_INTERVAL):
        world.step([0] * len(world.intersections))

    ep = StateBuilder.from_registry(world, world.intersections, state="ep")
    full = StateBuilder.from_registry(world, world.intersections, state="ats")
    assert ep.observe()[:, : ep.n_mov] == pytest.approx(full.observe()[:, : full.n_mov])


@pytest.mark.slow
def test_pressure_to_nieznormalizowany_ep(advanced_colight_world):
    """Jedyna różnica pressure vs EP to normalizacja liczbą pasów (Wu 2021).
    Przeliczone niezależnie z prymitywów `ats`, nie z drugiego wariantu."""
    world, _ = advanced_colight_world
    for _ in range(ACTION_INTERVAL):
        world.step([0] * len(world.intersections))

    b = StateBuilder.from_registry(world, world.intersections, state="pressure")
    rows = b.observe()
    q = world.get_info("lane_waiting_count")
    for idx, s in enumerate(b.structs):
        # `obs_length` jest ustawione w configu, więc kolejka liczy się w oknie —
        # bierzemy ją z buildera, żeby test sprawdzał wariant, nie okno.
        expected = ats.movement_pressure(
            s.movements, b._queue(s, b._snapshot(), q), normalize=False
        )
        assert rows[idx, : len(expected)] == pytest.approx(expected), idx


@pytest.mark.slow
def test_nagroda_kolejkowa_zgadza_sie_z_colightem(advanced_colight_world):
    """Najmocniejszy test modułu: nagroda liczona przez nas MUSI się zgadzać
    z `LaneVehicleGenerator(average="all", negative=True) × 12` z upstreamu —
    inaczej rdzeń optymalizuje coś innego niż baseline, do którego się porównuje.
    Stała 12 to liczba pasów wjazdowych skrzyżowania (4 drogi × 3 pasy), więc
    `średnia × 12` to suma."""
    world, model = advanced_colight_world
    for _ in range(ACTION_INTERVAL):
        world.step([2] * len(world.intersections))

    b = StateBuilder.from_registry(world, world.intersections, reward="queue")
    assert all(len(s.reward_lanes) == 12 for s in b.structs), "stała 12 z upstreamu nie pasuje"
    assert b.rewards() == pytest.approx(model.get_reward(), abs=1e-4)


@pytest.mark.slow
def test_nagroda_pressure_jest_niedodatnia(advanced_colight_world):
    """Wariant PressLight karze NIEZRÓWNOWAŻENIE (moduł), więc nigdy nie nagradza."""
    world, _ = advanced_colight_world
    b = StateBuilder.from_registry(world, world.intersections, reward="pressure")
    for _ in range(3):
        for _ in range(ACTION_INTERVAL):
            world.step([1] * len(world.intersections))
        assert (b.rewards() <= 0).all()
    assert set(REWARD_VARIANTS) == {"queue", "pressure"}


@pytest.mark.slow
def test_kolejnosc_wierszy_idzie_za_podana_lista(advanced_colight_world):
    """Kolejność jest WEJŚCIEM buildera, nie założeniem — rdzeń poda
    `graph.order_intersections()`, agent `world.intersections`."""
    world, _ = advanced_colight_world
    for _ in range(ACTION_INTERVAL):
        world.step([0] * len(world.intersections))

    forward = StateBuilder.from_registry(world, world.intersections)
    backward = StateBuilder.from_registry(world, list(reversed(world.intersections)))
    assert backward.observe() == pytest.approx(forward.observe()[::-1])
    assert backward.rewards() == pytest.approx(forward.rewards()[::-1])


@pytest.mark.slow
def test_demand_nieujemny_a_pressure_bywa_ujemne(advanced_colight_world):
    """Sanity check semantyki na poziomie buildera (bliźniaczy do testu agenta):
    liczba pojazdów nie może być ujemna, a pressure to różnica — jeśli NIGDY nie
    schodzi poniżej zera, zgubiliśmy człon downstream i spillback jest niewidoczny."""
    world, _ = advanced_colight_world
    b = StateBuilder.from_registry(world, world.intersections, state="ats")

    seen_negative = False
    for _ in range(20):
        for _ in range(ACTION_INTERVAL):
            world.step([1] * len(world.intersections))
        rows = b.observe()
        assert (rows[:, b.n_mov : 2 * b.n_mov] >= 0).all()
        seen_negative |= bool((rows[:, : b.n_mov] < 0).any())
    assert seen_negative, "pressure nigdy nie było ujemne — sprawdź człon downstream"
