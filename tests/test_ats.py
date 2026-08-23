"""Testy strukturalne rozbicia faz na ruchy (l,m) — Def. 4/5/9/10 z Zhang 2022.

Dlaczego akurat to: `phase_movements` jest jedynym miejscem, gdzie z lanelinków
CityFlow (para PASÓW) odtwarzamy ruch z papieru (para DRÓG). Pomyłka tutaj nie
wywala się — daje cicho zaniżony pressure i wynik gorszy o kilka procent, czyli
dokładnie taki, jaki da się wziąć za "no i tyle wychodzi".
"""

from __future__ import annotations

import pytest

from tsc_marl.agents import ats


def right_turn_lanelinks(raw: dict) -> set[tuple[str, str]]:
    """Skręty w prawo prosto z roadnetu (`roadLinks[i]["type"]`) — niezależne
    źródło prawdy dla heurystyki "obecne we wszystkich fazach"."""
    out = set()
    for rl in raw["roadLinks"]:
        if rl["type"] != "turn_right":
            continue
        for ll in rl["laneLinks"]:
            out.add(
                (
                    f"{rl['startRoad']}_{ll['startLaneIndex']}",
                    f"{rl['endRoad']}_{ll['endLaneIndex']}",
                )
            )
    return out


def test_road_of_odcina_tylko_indeks_pasa():
    # roadId sam zawiera '_' — split() bez limitu urwałby drogę do "road"
    assert ats.road_of("road_2_1_1_0") == "road_2_1_1"
    assert ats.road_of("road_1_2_0_2") == "road_1_2_0"


def test_always_green_to_dokladnie_skrety_w_prawo(roadnet):
    for inter, raw in roadnet:
        assert ats.always_green_lanelinks(inter) == right_turn_lanelinks(raw), inter.id


def test_liczba_faz_i_ruchow(roadnet):
    """Protokół pracy: 4 fazy (bez żółtej), każda = para przeciwstawnych ruchów."""
    for inter, _ in roadnet:
        movements = ats.phase_movements(inter)
        assert len(movements) == len(inter.phases) == 4, inter.id
        assert [len(m) for m in movements] == [2, 2, 2, 2], inter.id


def test_faza_zolta_nie_wchodzi_do_indeksowania(roadnet):
    """`inter.phases` ma wyciętą żółtą (id 0 w CityFlow), a `phase_movements`
    indeksuje po niej — więc indeks ruchu == indeks akcji modelu."""
    for inter, raw in roadnet:
        assert len(raw["trafficLight"]["lightphases"]) == len(inter.phases) + 1, inter.id
        assert inter.yellow_phase_id == [0], inter.id


def test_ruch_nie_miesza_drog(roadnet):
    """Każdy ruch (l,m) to jedna para dróg: wszystkie pasy wjazdowe z tej samej
    drogi, wszystkie wyjazdowe z tej samej — inaczej M/N z Def. 4 nie mają sensu."""
    for inter, _ in roadnet:
        for moves in ats.phase_movements(inter):
            for ins, outs in moves:
                assert len({ats.road_of(lane) for lane in ins}) == 1, inter.id
                assert len({ats.road_of(lane) for lane in outs}) == 1, inter.id


def test_pasy_wjazdowe_sa_startlanes_i_nie_sa_wyjazdowymi(roadnet):
    for inter, _ in roadnet:
        movements = ats.controlled_movements(inter)
        ins = ats.in_lanes(movements)
        outs = ats.all_lanes(movements) - ins
        assert ins <= set(inter.startlanes), inter.id
        assert not (ins & outs), inter.id


def test_drop_always_green_zmienia_tylko_skrety_w_prawo(roadnet):
    """Sanity check flagi: z włączonymi skrętami w prawo dochodzą ruchy, ale
    zbiór pasów wjazdowych powiększa się wyłącznie o pasy skrętu w prawo."""
    for inter, raw in roadnet:
        kept = ats.in_lanes(ats.controlled_movements(inter, drop_always_green=False))
        dropped = ats.in_lanes(ats.controlled_movements(inter, drop_always_green=True))
        assert dropped < kept, inter.id
        assert kept - dropped == {start for start, _ in right_turn_lanelinks(raw)}, inter.id


def test_pressure_i_demand_na_zadanej_kolejce(roadnet):
    """p(s) i d(s) na ręcznie ustawionym stanie: 10 aut na każdym pasie
    wjazdowym, pusto na wyjazdowych → e(l,m) = 10/M − 0 dla każdego ruchu."""
    for inter, _ in roadnet:
        movements = ats.phase_movements(inter)
        ins = ats.in_lanes(ats.controlled_movements(inter))
        q = dict.fromkeys(ats.all_lanes(ats.controlled_movements(inter)), 0.0) | dict.fromkeys(
            ins, 10.0
        )

        expected = [sum(10.0 for _ in moves) for moves in movements]  # M=1 na tych roadnetach
        assert ats.phase_pressure(movements, q) == pytest.approx(expected), inter.id
        assert ats.phase_demand(movements, dict.fromkeys(ins, 3)) == pytest.approx(
            [3 * sum(len(i) for i, _ in moves) for moves in movements]
        ), inter.id

        # symetria: te same liczby na wyjazdowych zerują pressure
        all_ten = dict.fromkeys(ats.all_lanes(ats.controlled_movements(inter)), 10.0)
        assert ats.phase_pressure(movements, all_ten) == pytest.approx([0.0] * len(movements)), (
            inter.id
        )


def test_ruchy_plaskie_sa_odduplikowane(roadnet):
    """`controlled_movements` = te same ruchy co `phase_movements`, ale każdy raz."""
    for inter, _ in roadnet:
        flat = ats.controlled_movements(inter)
        keys = [(ats.road_of(ins[0]), ats.road_of(outs[0])) for ins, outs in flat]
        assert len(keys) == len(set(keys)), inter.id
        assert set(keys) == {
            (ats.road_of(ins[0]), ats.road_of(outs[0]))
            for moves in ats.phase_movements(inter)
            for ins, outs in moves
        }, inter.id
        # 4 wloty × {prosto, w lewo}; skręty w prawo odpadły
        assert len(flat) == 8, inter.id


def test_faza_to_suma_ruchow(roadnet):
    """Def. 5 musi być DOSŁOWNIE sumą Def. 4 — inaczej dwie ścieżki liczenia
    pressure (stan ATS vs decyzja Advanced-MP) mogą się rozjechać."""
    for inter, _ in roadnet:
        by_phase = ats.phase_movements(inter)
        q = {
            lane: float(hash(lane) % 17) for lane in ats.all_lanes(ats.controlled_movements(inter))
        }
        run = {
            lane: float(hash(lane) % 5) for lane in ats.in_lanes(ats.controlled_movements(inter))
        }
        assert ats.phase_pressure(by_phase, q) == pytest.approx(
            [sum(ats.movement_pressure(moves, q)) for moves in by_phase]
        ), inter.id
        assert ats.phase_demand(by_phase, run) == pytest.approx(
            [sum(ats.movement_running(moves, run)) for moves in by_phase]
        ), inter.id


def test_movement_pressure_nie_gubi_downstream():
    """e(l,m) = up − down, a nie samo up: zapchany pas odbiorczy MUSI zbijać
    pressure, bo to jedyne miejsce, gdzie ATS widzi spillback."""
    movements = [(["in_0"], ["out_0", "out_1"])]
    assert ats.movement_pressure(movements, {"in_0": 10.0, "out_0": 0.0, "out_1": 0.0}) == [10.0]
    assert ats.movement_pressure(movements, {"in_0": 10.0, "out_0": 8.0, "out_1": 4.0}) == [4.0]
    assert ats.movement_pressure(movements, {"in_0": 0.0, "out_0": 6.0, "out_1": 6.0}) == [-6.0]


def test_movement_pressure_zwraca_ruch_a_nie_faze(roadnet):
    """Regresja na pomyłkę kosztującą połowę stanu: 8 liczb (ruchy), nie 4 (fazy),
    i każda odpowiada SWOJEMU ruchowi, a nie ostatniemu w fazie."""
    for inter, _ in roadnet:
        flat = ats.controlled_movements(inter)
        # unikalna kolejka na każdym wjeździe → wszystkie e(l,m) muszą być różne
        q = dict.fromkeys(ats.all_lanes(flat), 0.0)
        q |= {lane: float(i + 1) for i, lane in enumerate(sorted(ats.in_lanes(flat)))}

        pressures = ats.movement_pressure(flat, q)
        assert len(pressures) == len(flat) == 8, inter.id
        assert len(set(pressures)) == 8, inter.id
        assert sum(pressures) == pytest.approx(
            sum(ats.phase_pressure(ats.phase_movements(inter), q))
        )


def test_kolejnosc_ruchow_jest_ta_sama_na_kazdym_skrzyzowaniu(roadnet):
    """CoLight dzieli JEDEN model między wszystkie skrzyżowania (index-free
    parameter sharing), więc cecha nr k musi znaczyć to samo wszędzie. Sygnatura
    korzysta z konwencji nazw CityFlow 'road_<x>_<y>_<kierunek>' — dla roadnetu
    bez tej konwencji test trzeba przepisać na geometrię, nie usunąć."""
    signatures = set()
    for inter, _ in roadnet:
        flat = ats.controlled_movements(inter)
        signatures.add(
            tuple(
                (ats.road_of(ins[0]).split("_")[-1], ats.road_of(outs[0]).split("_")[-1])
                for ins, outs in flat
            )
        )
    assert len(signatures) == 1, (
        f"kolejność ruchów rozjeżdża się między skrzyżowaniami: {signatures}"
    )


def test_okno_obserwacji_odcina_dalekie_pojazdy():
    """windowed_queue/effective_running — podział na stojące i jadące oraz
    zasięg L liczony od STOP-LINII (koniec pasa), nie od jego początku."""
    lane = "road_1_1_0_0"
    lane_length = {lane: 300.0}
    lane_vehicles = {lane: ["stoi_blisko", "stoi_daleko", "jedzie_blisko", "jedzie_daleko"]}
    distances = {  # dystans od POCZĄTKU pasa; do stop-linii zostaje 300 − distance
        "stoi_blisko": 250.0,
        "stoi_daleko": 100.0,
        "jedzie_blisko": 200.0,
        "jedzie_daleko": 50.0,
    }
    speeds = {"stoi_blisko": 0.0, "stoi_daleko": 0.0, "jedzie_blisko": 11.0, "jedzie_daleko": 11.0}

    q = ats.windowed_queue([lane], lane_vehicles, distances, speeds, lane_length, obs_length=167.0)
    assert q == {lane: 1}  # tylko stoi_blisko (50 m od stop-linii); stoi_daleko jest 200 m

    running = ats.effective_running(
        [lane], lane_vehicles, distances, speeds, lane_length, effective_range=167.0
    )
    assert running == {lane: 1}  # jedzie_blisko (100 m); jedzie_daleko jest 250 m

    unlimited = ats.effective_running(
        [lane], lane_vehicles, distances, speeds, lane_length, effective_range=0.0, unlimited=True
    )
    assert unlimited == {lane: 2}  # ablacja Config3: cały pas, nadal tylko jadące


def test_nieznany_pojazd_nie_wpada_do_zadnej_polowy_ATS():
    """Pojazd bez wpisu w `speeds` (rozjazd bufora silnika) musi wypaść i z
    kolejki, i z demand — inaczej ATS liczyłby go dwa razy."""
    lane = "road_1_1_0_0"
    args = ([lane], {lane: ["duch"]}, {"duch": 100.0}, {}, {lane: 200.0})
    assert ats.windowed_queue(*args, obs_length=167.0) == {lane: 0}
    assert ats.effective_running(*args, effective_range=167.0) == {lane: 0}
