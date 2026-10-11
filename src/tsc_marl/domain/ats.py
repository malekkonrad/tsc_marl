"""Advanced Traffic State (Zhang et al., 2022) and pressure as stateless functions.

Everything takes a LibSignal `Intersection` or plain engine dicts and keeps no
state, because two class hierarchies need the same maths: Advanced-MP is a
per-intersection agent, while Advanced-CoLight and the core build the state of
the whole network (`core/features.py`).

Definitions from the paper, per movement (l, m) unless stated otherwise:

    Def 4   efficient pressure  e(l,m) = mean q(in lanes) - mean q(out lanes)
    Def 5   phase pressure      p(s)   = sum of e(l,m) over the movements of s
    Def 8   effective range     L      = V_max * action interval
    Def 9   effective running   r(l,m) = moving vehicles within L on the in lanes
    Def 10  phase demand        d(s)   = sum of r(l,m) over the movements of s
    Def 11  ATS                 {e(l,m), r(l,m)} for every movement
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # type-only: the submodule is importable only after `libsignal.bootstrap()`
    from world.world_cityflow import Intersection

WAITING_SPEED = 0.1  # m/s, below this a vehicle is queued (same threshold as world_cityflow)

# A movement (l, m) after grouping lane links: (used in lanes, used out lanes).
Movement = tuple[list[str], list[str]]


def road_of(lane: str) -> str:
    """Lane id is '<roadId>_<laneIndex>' and roadId itself contains '_'
    ('road_2_1_1_0' is lane 0 of road 'road_2_1_1'), hence rsplit with limit 1."""
    return lane.rsplit("_", 1)[0]


def in_road_lanes(inter: Intersection) -> list[str]:
    """All lanes of the incoming roads: the reward domain (as `LaneVehicleGenerator`
    with `in_only=True`) and the detector owners in `sensors.failure_units`.

    CityFlow keeps roads as roadnet dicts (`id`, `lanes`), SUMO as edge ids with
    `road_lane_mapping`. Lane ids are '<road>_<index>' in both, so the rest of
    the code does not need to know which engine runs.
    """
    lanes: list[str] = []
    for road in inter.in_roads:
        if isinstance(road, str):
            lanes += sorted(inter.road_lane_mapping[road])
        else:
            lanes += [f"{road['id']}_{i}" for i in range(len(road["lanes"]))]
    return lanes


def always_green_lanelinks(inter: Intersection) -> set[tuple[str, str]]:
    """Lane links present in every phase; on the CityFlow roadnets these are the
    right turns. They do not compete for green, so they belong to neither
    pressure nor demand; the authors' reference code sums only through and left
    movements."""
    per_phase = [set(map(tuple, ll)) for ll in inter.phase_available_lanelinks]
    return set.intersection(*per_phase) if per_phase else set()


def phase_movements(inter: Intersection, drop_always_green: bool = True) -> list[list[Movement]]:
    """[phase][movement] -> (used in lanes, used out lanes).

    A movement (l, m) in Def 4 is a pair of roads, while the engine gives lane
    links, so links are grouped by `(road_of(start), road_of(end))`. The lane
    counts in Def 4 are the lanes this movement uses, not all lanes of the road.

    Phase index == action index: `phase_available_lanelinks` is built over
    `inter.phases`, from which LibSignal has already removed the yellow phase.
    """
    always = always_green_lanelinks(inter) if drop_always_green else set()
    per_phase: list[list[Movement]] = []
    for phase_id in range(len(inter.phases)):
        moves: dict[tuple[str, str], tuple[set[str], set[str]]] = {}
        for start, end in inter.phase_available_lanelinks[phase_id]:
            if (start, end) in always:
                continue
            ins, outs = moves.setdefault((road_of(start), road_of(end)), (set(), set()))
            ins.add(start)
            outs.add(end)
        per_phase.append([(sorted(i), sorted(o)) for i, o in moves.values()])
    return per_phase


def controlled_movements(inter: Intersection, drop_always_green: bool = True) -> list[Movement]:
    """Flat, deduplicated list of movements: the ATS domain (Def 11).

    A movement green in several phases enters the state once, so the
    observation size depends on the geometry, not on the phase plan. Order is
    first occurrence over phases; with a shared policy feature k must mean the
    same on every intersection (checked by a test).
    """
    seen: dict[tuple[str, str], Movement] = {}
    for moves in phase_movements(inter, drop_always_green):
        for ins, outs in moves:
            seen.setdefault((road_of(ins[0]), road_of(outs[0])), (ins, outs))
    return list(seen.values())


def in_lanes(movements: Sequence[Movement]) -> set[str]:
    """In lanes of the controlled movements: the demand domain (Def 9)."""
    return {lane for ins, _ in movements for lane in ins}


def all_lanes(movements: Sequence[Movement]) -> set[str]:
    """In and out lanes: the pressure domain (Def 4 looks both ways)."""
    return {lane for ins, outs in movements for lane in (*ins, *outs)}


def movement_pressure(
    movements: Sequence[Movement], q: Mapping[str, float], normalize: bool = True
) -> list[float]:
    """e(l,m) per movement (Def 4), not summed over a phase (that is Def 5).

    `normalize=False` gives the classic pressure (Varaiya, 2013): difference of
    sums instead of difference of means, which is the only change Wu et al.
    (2021) made. Both live here so the state variants share one implementation."""
    if not normalize:
        return [
            sum(q[lane] for lane in ins) - sum(q[lane] for lane in outs) for ins, outs in movements
        ]
    return [
        sum(q[lane] for lane in ins) / len(ins) - sum(q[lane] for lane in outs) / len(outs)
        for ins, outs in movements
    ]


def movement_running(movements: Sequence[Movement], running: Mapping[str, float]) -> list[float]:
    """r(l,m) per movement (Def 9), not summed over a phase (that is Def 10)."""
    return [sum(running.get(lane, 0) for lane in ins) for ins, _ in movements]


def phase_pressure(movements: Sequence[Sequence[Movement]], q: Mapping[str, float]) -> list[float]:
    """p(s) per phase (Def 5): the sum of Def 4 over the phase's movements.
    Takes the nested shape of `phase_movements`, not the flat list."""
    return [sum(movement_pressure(moves, q)) for moves in movements]


def phase_demand(
    movements: Sequence[Sequence[Movement]], running: Mapping[str, float]
) -> list[float]:
    """d(s) per phase (Def 10): the sum of Def 9 over the phase's movements."""
    return [sum(movement_running(moves, running)) for moves in movements]


def windowed_queue(
    lanes: Iterable[str],
    lane_vehicles: Mapping[str, Iterable[str]],
    distances: Mapping[str, float],
    speeds: Mapping[str, float],
    lane_length: Mapping[str, float],
    obs_length: float,
) -> dict[str, int]:
    """Queue counted only within `obs_length` of the stop line, as in the authors'
    reference code (OBS_LENGTH = 167). The paper suggests the whole lane."""
    q = {}
    for lane in lanes:
        length = lane_length.get(lane, 0.0)
        n = 0
        for vid in lane_vehicles.get(lane, []):
            if speeds.get(vid, 1.0) > WAITING_SPEED:  # no speed entry: neither queued nor running
                continue
            if length - distances.get(vid, 0.0) <= obs_length:
                n += 1
        q[lane] = n
    return q


def effective_running(
    lanes: Iterable[str],
    lane_vehicles: Mapping[str, Iterable[str]],
    distances: Mapping[str, float],
    speeds: Mapping[str, float],
    lane_length: Mapping[str, float],
    effective_range: float | Mapping[str, float],
    unlimited: bool = False,
) -> dict[str, int]:
    """Moving vehicles within the effective range L of the stop line, per in lane
    (Def 8 and 9). `effective_range` is either a fixed window (the authors'
    OBS_LENGTH) or a map lane -> V_max(lane) * action interval.
    `unlimited=True` is the paper's "Config3" ablation (whole lane)."""
    counts = {}
    for lane in lanes:
        limit = (
            effective_range
            if isinstance(effective_range, (int, float))
            else effective_range.get(lane, 0.0)
        )
        length = lane_length.get(lane, 0.0)
        n = 0
        for vid in lane_vehicles.get(lane, []):
            if speeds.get(vid, 0.0) < WAITING_SPEED:  # no speed entry: neither queued nor running
                continue
            if unlimited or length - distances.get(vid, 0.0) <= limit:
                n += 1
        counts[lane] = n
    return counts


def phase_incidence(inter: Intersection, drop_always_green: bool = True) -> list[list[int]]:
    """[phase] -> indices into `controlled_movements` served by that phase.

    Bridges the state, defined per movement (Def 11), and the decision, made per
    phase (Def 5 and 10). With it a max-pressure decision can be recomputed from
    the observation vector alone. A movement green in several phases appears in
    several rows.
    """
    flat = controlled_movements(inter, drop_always_green)
    index = {(road_of(ins[0]), road_of(outs[0])): i for i, (ins, outs) in enumerate(flat)}
    return [
        [index[(road_of(ins[0]), road_of(outs[0]))] for ins, outs in moves]
        for moves in phase_movements(inter, drop_always_green)
    ]
