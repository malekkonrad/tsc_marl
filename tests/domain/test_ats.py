"""Splitting phases into movements (l, m): Def 4, 5, 9 and 10 of Zhang et al. (2022).

`phase_movements` is the only place where engine lane links (pairs of lanes)
become the paper's movements (pairs of roads). A mistake here raises nothing;
it silently lowers pressure and costs a few percent of the result.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tsc_marl.domain import ats


def right_turn_lanelinks(raw: dict) -> set[tuple[str, str]]:
    """Right turns read from the roadnet (`roadLinks[i]["type"]`): an independent
    source of truth for the "present in every phase" heuristic."""
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


def test_road_of_strips_only_the_lane_index():
    # roadId itself contains '_'; split() without a limit would cut it to "road"
    assert ats.road_of("road_2_1_1_0") == "road_2_1_1"
    assert ats.road_of("road_1_2_0_2") == "road_1_2_0"


def test_always_green_links_are_exactly_right_turns(roadnet):
    for inter, raw in roadnet:
        assert ats.always_green_lanelinks(inter) == right_turn_lanelinks(raw), inter.id


def test_phase_and_movement_counts(roadnet):
    """Protocol: 4 phases (yellow excluded), each a pair of opposing movements."""
    for inter, _ in roadnet:
        movements = ats.phase_movements(inter)
        assert len(movements) == len(inter.phases) == 4, inter.id
        assert [len(m) for m in movements] == [2, 2, 2, 2], inter.id


def test_yellow_phase_is_not_indexed(roadnet):
    """`inter.phases` has yellow (id 0 in CityFlow) removed and `phase_movements`
    indexes over it, so movement phase index == model action index."""
    for inter, raw in roadnet:
        assert len(raw["trafficLight"]["lightphases"]) == len(inter.phases) + 1, inter.id
        assert inter.yellow_phase_id == [0], inter.id


def test_movement_does_not_mix_roads(roadnet):
    """Each movement is one pair of roads: all in lanes from one road, all out
    lanes from one road; otherwise the lane counts in Def 4 make no sense."""
    for inter, _ in roadnet:
        for moves in ats.phase_movements(inter):
            for ins, outs in moves:
                assert len({ats.road_of(lane) for lane in ins}) == 1, inter.id
                assert len({ats.road_of(lane) for lane in outs}) == 1, inter.id


def test_in_lanes_are_start_lanes_and_not_out_lanes(roadnet):
    for inter, _ in roadnet:
        movements = ats.controlled_movements(inter)
        ins = ats.in_lanes(movements)
        outs = ats.all_lanes(movements) - ins
        assert ins <= set(inter.startlanes), inter.id
        assert not (ins & outs), inter.id


def test_drop_always_green_removes_only_right_turns(roadnet):
    """With right turns kept, the in-lane set grows by exactly the right-turn lanes."""
    for inter, raw in roadnet:
        kept = ats.in_lanes(ats.controlled_movements(inter, drop_always_green=False))
        dropped = ats.in_lanes(ats.controlled_movements(inter, drop_always_green=True))
        assert dropped < kept, inter.id
        assert kept - dropped == {start for start, _ in right_turn_lanelinks(raw)}, inter.id


def test_pressure_and_demand_on_a_given_queue(roadnet):
    """p(s) and d(s) on a hand-made state: 10 vehicles on every in lane, empty out
    lanes, so e(l,m) = 10 / M - 0 for every movement."""
    for inter, _ in roadnet:
        movements = ats.phase_movements(inter)
        ins = ats.in_lanes(ats.controlled_movements(inter))
        q = dict.fromkeys(ats.all_lanes(ats.controlled_movements(inter)), 0.0) | dict.fromkeys(
            ins, 10.0
        )

        expected = [sum(10.0 for _ in moves) for moves in movements]  # M = 1 on these roadnets
        assert ats.phase_pressure(movements, q) == pytest.approx(expected), inter.id
        assert ats.phase_demand(movements, dict.fromkeys(ins, 3)) == pytest.approx(
            [3 * sum(len(i) for i, _ in moves) for moves in movements]
        ), inter.id

        # symmetry: the same counts on the out lanes cancel the pressure
        all_ten = dict.fromkeys(ats.all_lanes(ats.controlled_movements(inter)), 10.0)
        assert ats.phase_pressure(movements, all_ten) == pytest.approx([0.0] * len(movements)), (
            inter.id
        )


def test_flat_movements_are_deduplicated(roadnet):
    """`controlled_movements` has the movements of `phase_movements`, each once."""
    for inter, _ in roadnet:
        flat = ats.controlled_movements(inter)
        keys = [(ats.road_of(ins[0]), ats.road_of(outs[0])) for ins, outs in flat]
        assert len(keys) == len(set(keys)), inter.id
        assert set(keys) == {
            (ats.road_of(ins[0]), ats.road_of(outs[0]))
            for moves in ats.phase_movements(inter)
            for ins, outs in moves
        }, inter.id
        # 4 approaches x {through, left}; right turns dropped
        assert len(flat) == 8, inter.id


def test_phase_value_is_the_sum_over_movements(roadnet):
    """Def 5 must be literally the sum of Def 4, otherwise the two pressure paths
    (ATS state vs Advanced-MP decision) can drift apart."""
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


def test_movement_pressure_keeps_downstream():
    """e(l,m) = up - down, not just up: a full receiving lane must lower pressure,
    it is the only place where ATS sees spillback."""
    movements = [(["in_0"], ["out_0", "out_1"])]
    assert ats.movement_pressure(movements, {"in_0": 10.0, "out_0": 0.0, "out_1": 0.0}) == [10.0]
    assert ats.movement_pressure(movements, {"in_0": 10.0, "out_0": 8.0, "out_1": 4.0}) == [4.0]
    assert ats.movement_pressure(movements, {"in_0": 0.0, "out_0": 6.0, "out_1": 6.0}) == [-6.0]


def test_movement_pressure_returns_movements_not_phases(roadnet):
    """8 numbers (movements), not 4 (phases), each for its own movement."""
    for inter, _ in roadnet:
        flat = ats.controlled_movements(inter)
        # a distinct queue on every in lane, so all e(l,m) must differ
        q = dict.fromkeys(ats.all_lanes(flat), 0.0)
        q |= {lane: float(i + 1) for i, lane in enumerate(sorted(ats.in_lanes(flat)))}

        pressures = ats.movement_pressure(flat, q)
        assert len(pressures) == len(flat) == 8, inter.id
        assert len(set(pressures)) == 8, inter.id
        assert sum(pressures) == pytest.approx(
            sum(ats.phase_pressure(ats.phase_movements(inter), q))
        )


def test_movement_order_is_the_same_on_every_intersection(roadnet):
    """A shared policy needs feature k to mean the same everywhere. The signature
    uses CityFlow's naming 'road_<x>_<y>_<direction>'; for a roadnet without it
    the test must be rewritten on geometry, not deleted."""
    signatures = set()
    for inter, _ in roadnet:
        flat = ats.controlled_movements(inter)
        signatures.add(
            tuple(
                (ats.road_of(ins[0]).split("_")[-1], ats.road_of(outs[0]).split("_")[-1])
                for ins, outs in flat
            )
        )
    assert len(signatures) == 1, f"movement order differs between intersections: {signatures}"


def test_observation_window_cuts_distant_vehicles():
    """Split into queued and moving vehicles, and the range L measured from the
    stop line (end of the lane), not from the lane start."""
    lane = "road_1_1_0_0"
    lane_length = {lane: 300.0}
    lane_vehicles = {lane: ["queued_near", "queued_far", "moving_near", "moving_far"]}
    distances = {  # from the lane START; distance to the stop line is 300 - distance
        "queued_near": 250.0,
        "queued_far": 100.0,
        "moving_near": 200.0,
        "moving_far": 50.0,
    }
    speeds = {"queued_near": 0.0, "queued_far": 0.0, "moving_near": 11.0, "moving_far": 11.0}

    q = ats.windowed_queue([lane], lane_vehicles, distances, speeds, lane_length, obs_length=167.0)
    assert q == {lane: 1}  # queued_near only (50 m from the stop line); queued_far is at 200 m

    running = ats.effective_running(
        [lane], lane_vehicles, distances, speeds, lane_length, effective_range=167.0
    )
    assert running == {lane: 1}  # moving_near (100 m); moving_far is at 250 m

    unlimited = ats.effective_running(
        [lane], lane_vehicles, distances, speeds, lane_length, effective_range=0.0, unlimited=True
    )
    assert unlimited == {lane: 2}  # Config3 ablation: whole lane, still only moving vehicles


def test_unknown_vehicle_is_in_neither_half_of_ats():
    """A vehicle without a speed entry must drop out of both queue and demand,
    otherwise ATS would count it twice."""
    lane = "road_1_1_0_0"
    args = ([lane], {lane: ["ghost"]}, {"ghost": 100.0}, {}, {lane: 200.0})
    assert ats.windowed_queue(*args, obs_length=167.0) == {lane: 0}
    assert ats.effective_running(*args, effective_range=167.0) == {lane: 0}


def test_in_road_lanes_handles_both_road_formats():
    """CityFlow keeps a road as a roadnet dict, SUMO as an edge id with a lane
    map. Both must give '<road>_<index>' lane ids, which the reward and the
    detector owners in `sensors.failure_units` rely on."""
    cityflow = SimpleNamespace(in_roads=[{"id": "road_1_1_0", "lanes": [{}, {}, {}]}])
    assert ats.in_road_lanes(cityflow) == ["road_1_1_0_0", "road_1_1_0_1", "road_1_1_0_2"]

    sumo = SimpleNamespace(
        in_roads=["-123#4"],
        # lane order from getControlledLinks is arbitrary; the exit is not part of the reward
        road_lane_mapping={"-123#4": ["-123#4_1", "-123#4_0"], "exit": ["exit_0"]},
    )
    assert ats.in_road_lanes(sumo) == ["-123#4_0", "-123#4_1"]
