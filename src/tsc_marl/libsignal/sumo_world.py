"""SUMO with the contract the core knows from CityFlow (backend parity, Krakow case study).

Registered as `sumo`, so it REPLACES upstream `world_sumo.World` in every `-w sumo` run,
the core's and LibSignal's baselines alike: otherwise one network would have two
definitions of ATT, reward and phase transitions. Nine differences from upstream:

1. `lane_length`, `all_lanes_speed`: lane -> m, m/s dicts as in `world_cityflow`, from
   `sumolib` (upstream closes the SUMO connection at the end of `__init__`).
2. `cityflow_api`: the three CityFlow engine methods the core calls directly.
3. Intersection IDENTITY ACROSS RESET: upstream builds NEW `Intersection` objects in
   `reset()`; here old objects take the fresh state, so references stay valid.
4. Permissive green `g` in `phase_available_lanelinks` (upstream takes only `G`, `s`).
5. `lane_waiting_count` = vehicles halted NOW (< 0.1 m/s, whole lane), as in CityFlow.
6. ATT as in CityFlow: finished plus still running vehicles, counted from entry.
7. Yellow from the protocol (`trainer.yellow_length`) instead of the program's shortest phase.
8. Phase TRANSITIONS as `world_cityflow.Intersection.step` (`cityflow_transition`).
9. The neighbour GRAPH (`build_graph`), hooked in `session.build_interfaces`.

Points 4-9 change the numbers of LibSignal's baselines on SUMO too.
"""

from __future__ import annotations

import heapq
from types import MethodType
from typing import Any

import numpy as np
import sumolib
from common.registry import Registry
from world import world_sumo

from tsc_marl.core.graph import RoadNetGraph

#: Signal states that let traffic go: G = green with priority, g = green yielding,
#: s = green arrow (turn after stopping).
SERVING = frozenset("Ggs")

#: Neighbourhood reach of the SUMO graph [m]; 600 m gives Krakow an in-degree of 0-4,
#: like |N_i| - 1 on the JiNan/HangZhou grids.
GRAPH_REACH_M = 600.0


class CityFlowAPI:
    """The three CityFlow engine methods the core and Advanced-MP call directly.

    CityFlow semantics: a vehicle's position is the distance from the lane START,
    ordinary lanes only (no SUMO internal lanes), like the `lane_length` keys. The
    engine is read on every call, since `world.eng` is replaced in `reset()`.
    """

    def __init__(self, world: World) -> None:
        self._world = world

    def get_lane_vehicles(self) -> dict[str, list[str]]:
        lane = self._world.eng.lane
        return {lid: list(lane.getLastStepVehicleIDs(lid)) for lid in self._world.lane_length}

    def get_vehicle_distance(self) -> dict[str, float]:
        vehicle = self._world.eng.vehicle
        return {v: vehicle.getLanePosition(v) for v in vehicle.getIDList()}

    def get_vehicle_speed(self) -> dict[str, float]:
        vehicle = self._world.eng.vehicle
        return {v: vehicle.getSpeed(v) for v in vehicle.getIDList()}


def serving_lanelinks(inter: Any) -> tuple[list[list[list[str]]], list[list[str]]]:
    """(lanelinks, start lanes) of every green phase, as upstream but with `g`. Reads
    `inter.lanelinks` rather than the engine, so it works before the first `reset()`."""
    lanelinks, startlanes = [], []
    for phase in inter.green_phases:
        links, starts = [], []
        # strict=False on purpose: a phase state can be longer than the link list, and
        # upstream breaks there too
        for signal, controlled in zip(phase.state, inter.lanelinks, strict=False):
            if signal not in SERVING or not controlled:
                continue
            start, end = controlled[0][0], controlled[0][1]
            links.append([start, end])
            if start not in starts:
                starts.append(start)
        lanelinks.append(links)
        startlanes.append(starts)
    return lanelinks, startlanes


def cityflow_transition(inter: Any, action: int) -> None:
    """`Intersection.pseudo_step` with the semantics of `world_cityflow.Intersection.step`.

    - A phase change first runs `yellow_phase_time` s of a transition phase; actions
      meanwhile are ignored and `current_phase` is still the OLD phase, then the target
      phase follows and the counter keeps counting (CityFlow's 'add' mode).
    - Holding the phase = counter + 1.
    - No transition when nothing turns red (no yellow for this phase pair) or when
      `yellow_phase_time == 0` (benchmark mode).

    Every step re-sets the phase in SUMO: RL program phases last 1 s, and SUMO would
    switch them on its own otherwise.
    """
    tl = inter.eng.trafficlight
    action = int(action)
    inter.virtual_phase = action  # upstream field: the last requested action
    if inter.transition is not None:  # in transition: (program index, target phase)
        yellow, target = inter.transition
        if inter.current_phase_time == inter.yellow_phase_time:
            inter.transition = None
            tl.setPhase(inter.id, target)
            inter.current_phase = target
        else:
            tl.setPhase(inter.id, yellow)
        inter.current_phase_time += 1
        return
    if action == inter.current_phase:
        tl.setPhase(inter.id, action)
        inter.current_phase_time += 1
        return
    yellow = inter.yellow_dict.get(f"{inter.current_phase}_{action}")
    if inter.yellow_phase_time > 0 and yellow is not None:
        inter.transition = (yellow, action)
        tl.setPhase(inter.id, yellow)
    else:
        tl.setPhase(inter.id, action)
        inter.current_phase = action
    inter.current_phase_time = 1


def travel_time(finished: dict[str, float], entered: dict[str, float], now: float) -> float:
    """ATT as in CityFlow: (sum of finished times + sum of (now - entry) running) / count."""
    running = [now - t for v, t in entered.items() if v not in finished]
    n = len(finished) + len(running)
    return (sum(finished.values()) + sum(running)) / n if n else 0.0


def protocol_yellow() -> int | None:
    """`trainer.yellow_length` from the Registry; None outside a run (tests, manual use)."""
    try:
        return int(Registry.mapping["trainer_mapping"]["setting"].param["yellow_length"])
    except (KeyError, AttributeError):
        return None


def _downstream(light: Any, owner: dict[str, str], reach_m: float) -> set[str]:
    """Signals that traffic from `light`'s exits reaches through unsignalised nodes
    within `reach_m` (Dijkstra over edge lengths)."""
    me = light.getID()
    exits = {out.getEdge() for _, out, _ in light.getConnections()}
    heap = [(edge.getLength(), edge.getID(), edge) for edge in exits]
    heapq.heapify(heap)
    seen: set[str] = set()
    found: set[str] = set()
    while heap:
        dist, edge_id, edge = heapq.heappop(heap)
        if edge_id in seen or dist > reach_m:
            continue
        seen.add(edge_id)
        other = owner.get(edge.getToNode().getID())
        if other is not None:  # a path ends at the FIRST signal, its own included
            if other != me:
                found.add(other)
            continue
        for nxt in edge.getOutgoing():
            heapq.heappush(heap, (dist + nxt.getLength(), nxt.getID(), nxt))
    return found


def build_graph(net_file: str, reach_m: float = GRAPH_REACH_M) -> dict[str, Any]:
    """Signal graph of a SUMO network in the `common/utils.build_index_intersection_map_*`
    format, replacing `build_index_intersection_map_sumo`.

    Edge t->u (traffic from t reaches u, so u aggregates from t) exists when some exit
    of t reaches a node controlled by u through unsignalised nodes within `reach_m`.
    Node ids and order as in LibSignal (the same `readNet(net_file)`, without
    `withPrograms`, which reorders signals). No duplicate edges or self loops (the
    `RoadNetGraph` contract); degree lists over incoming edges, as upstream.
    """
    net = sumolib.net.readNet(net_file)
    lights = net.getTrafficLights()
    # node -> signal: the node its controlled lanes lead into
    owner = {
        in_lane.getEdge().getToNode().getID(): light.getID()
        for light in lights
        for in_lane, _, _ in light.getConnections()
    }
    ids = [RoadNetGraph.bare_id(light.getID()) for light in lights]
    index = {light.getID(): i for i, light in enumerate(lights)}
    pairs = sorted(
        {
            (index[light.getID()], index[other])
            for light in lights
            for other in _downstream(light, owner, reach_m)
        }
    )
    incoming = [[k for k, (_, dst) in enumerate(pairs) if dst == i] for i in range(len(lights))]
    return {
        "node_idx2id": dict(enumerate(ids)),
        "node_id2idx": {nid: i for i, nid in enumerate(ids)},
        "edge_idx2id": {k: f"{ids[s]}->{ids[d]}" for k, (s, d) in enumerate(pairs)},
        "edge_id2idx": {f"{ids[s]}->{ids[d]}": k for k, (s, d) in enumerate(pairs)},
        "node_degrees": np.array([len(ks) for ks in incoming]),
        "sparse_adj": np.array(pairs, dtype=np.int64).reshape(-1, 2),
        "node_list": [sorted(pairs[k][0] for k in ks) for ks in incoming],
        "edge_list": incoming,
    }


@Registry.register_world("sumo")
class World(world_sumo.World):
    def __init__(self, sumo_config: str, placeholder: int = 0, **kwargs: Any) -> None:
        super().__init__(sumo_config, placeholder, **kwargs)
        net = sumolib.net.readNet(self.net)
        lanes = [lane for edge in net.getEdges() for lane in edge.getLanes()]
        self.lane_length = {lane.getID(): lane.getLength() for lane in lanes}
        self.all_lanes_speed = {lane.getID(): lane.getSpeed() for lane in lanes}
        self.cityflow_api = CityFlowAPI(self)
        self.yellow = protocol_yellow()
        for inter in self.intersections:
            self._patch(inter, install=False)  # connection closed; reset() installs the program

    def _patch(self, inter: Any, *, install: bool) -> None:
        """Points 4, 7 and 8 of the module docstring on one intersection."""
        lanelinks, startlanes = serving_lanelinks(inter)
        inter.phase_available_lanelinks = lanelinks
        inter.phase_available_startlanes = startlanes
        inter.startlanes = list(dict.fromkeys(lane for starts in startlanes for lane in starts))
        inter.transition = None
        inter.pseudo_step = MethodType(cityflow_transition, inter)
        if self.yellow is None or self.yellow == inter.yellow_phase_time:
            return
        inter.yellow_phase_time = self.yellow
        inter.full_phases, inter.yellow_dict = inter.create_yellows(
            inter.green_phases, self.yellow, inter.interface_flag
        )
        if install:  # the same three calls as at the end of `Intersection.__init__`
            program = inter.id + "_rl"
            logic = self.eng.trafficlight.Logic(program, 0, 0, inter.full_phases)
            self.eng.trafficlight.setProgramLogic(inter.id, logic)
            self.eng.trafficlight.setProgram(inter.id, program)

    def reset(self) -> None:
        kept = self.id2intersection
        super().reset()  # restarts SUMO; upstream builds NEW Intersection objects here
        for tl_id, fresh in self.id2intersection.items():
            kept[tl_id].__dict__.update(fresh.__dict__)
        self.id2intersection = {tl_id: kept[tl_id] for tl_id in self.id2intersection}
        self.intersections = list(self.id2intersection.values())
        for inter in self.intersections:
            self._patch(inter, install=True)

    def get_lane_waiting_vehicle_count(self) -> dict[str, int]:
        """Point 5: the same keys as upstream (`inter.lanes`), another definition."""
        lane = self.eng.lane
        return {
            lid: lane.getLastStepHaltingNumber(lid)
            for inter in self.intersections
            for lid in inter.lanes
        }

    def get_average_travel_time(self) -> float:
        """Point 6."""
        return travel_time(self.vehicles, self.inside_vehicles, self.get_current_time())
