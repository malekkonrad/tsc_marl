"""SUMO z kontraktem, który rdzeń zna z CityFlow — parytet backendu (Faza 5, case study Krakowa).

Rejestrowany w Registry pod nazwą `sumo`, czyli ZASTĘPUJE upstreamowy
`world_sumo.World` w każdym biegu `-w sumo` — rdzenia i baseline'ów LibSignala.
Gdyby adapter dostawał tylko rdzeń, ta sama sieć miałaby dwie definicje ATT,
nagrody i przejść między fazami, a tabela porównywałaby metody liczone różnie.

DZIEWIĘĆ RÓŻNIC WOBEC UPSTREAMU — każda naprawia coś, co rdzeń albo metryka zakłada:

1. `lane_length`, `all_lanes_speed` — słowniki pas → m, m/s, jak w `world_cityflow`
   (okno L = V_max·Δt i ruch efektywny ATS). Z `sumolib`, bo upstream zamyka
   połączenie z SUMO na końcu `__init__` i do pierwszego `reset()` nie ma kogo pytać.
2. `cityflow_api` — widok silnika z trzema metodami CityFlow, których rdzeń używa
   wprost (`get_lane_vehicles`, `get_vehicle_distance`, `get_vehicle_speed`);
   wybiera go `core.features.engine_api`.
3. TOŻSAMOŚĆ SKRZYŻOWAŃ PRZEZ RESET. Upstream buduje w `reset()` NOWE obiekty
   `Intersection`, więc każdy, kto trzymał referencję (StateBuilder, `inter_obj`
   agentów LibSignala), czytał potem fazę sprzed resetu. Obserwacja z zamrożonym
   one-hotem nie wywala się — sieć uczy się szumu. Tutaj stare obiekty dostają stan
   świeżych (`__dict__`), więc referencje zostają ważne, tak jak w CityFlow.
4. ZIELONE WARUNKOWE `g` w `phase_available_lanelinks`. Upstream bierze tylko `G`
   i `s`, więc skręt w lewo z ustąpieniem nie należy do żadnej fazy — ATS
   i MaxPressure go nie widzą (13 z 25 sygnalizacji Krakowa, 32 lanelinki).
5. `lane_waiting_count` = pojazdy stojące TERAZ (< 0,1 m/s, cały pas), jak CityFlow.
   Upstream liczy pojazd jako czekający od pierwszego zatrzymania, dopóki skrzyżowanie
   go widzi (`Intersection.waiting_times` nie są czyszczone), i tylko 200 m przed
   sygnalizacją. Na Krakowie efekt jest mały (mid, cykl stałoczasowy: 17 z 1770 próbek
   pas × chwila zawyżonych), ale to NAGRODA rdzenia — definicja ma być ta z CityFlow.
6. ATT jak CityFlow (`Engine::getAverageTravelTime`): dojechane + wciąż jadące,
   liczone od wjazdu. Upstream liczy tylko dojechane, więc pojazdy zablokowane
   w sieci nie wchodzą do średniej — metoda, która blokuje ruch, nie płaci za tych,
   których zablokowała. Różnica idzie w obie strony (Kraków mid, cykl stałoczasowy
   900 s: 294 s upstream vs 255 s wg CityFlow), więc liczby nie są porównywalne.
7. Żółte z protokołu (`trainer.yellow_length`, 5 s = 3 + 2 all-red) zamiast najkrótszej
   fazy programu z pliku sieci — inaczej `_y5` w id biegu byłoby fikcją.
8. PRZEJŚCIA FAZ jak `world_cityflow.Intersection.step` (`cityflow_transition`).
   Upstreamowy `pseudo_step` zeruje licznik fazy tylko w gałęzi, do której po
   pierwszych sekundach epizodu nie da się już wejść: każda późniejsza zmiana fazy
   jest natychmiastowa, bez przejścia, a `t_min` baseline'ów przestaje działać
   (Kraków mid, 900 s: 0 kroków w fazie przejściowej po 30. sekundzie).
9. GRAF SĄSIEDZTWA (`build_graph`, podpinany w `session.build_interfaces`). Graf
   LibSignala łączy sygnalizacje tylko przez jedną krawędź węzeł–węzeł o id równym
   id sygnalizacji; na sieci z OSM daje 1 krawędź na 25 węzłów — GAT bez sąsiadów.

Punkty 4–9 zmieniają liczby także baseline'om LibSignala na SUMO. Wyniki `-w sumo`
sprzed tego modułu nie są porównywalne z późniejszymi.
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

#: Stany sygnału, przy których ruch może jechać: G — zielone z pierwszeństwem,
#: g — zielone z ustąpieniem, s — zielona strzałka (skręt po zatrzymaniu).
SERVING = frozenset("Ggs")

#: Zasięg sąsiedztwa w grafie SUMO [m]. 600 m daje na Krakowie stopień wejściowy 0–4
#: (mediana 3) — tyle, ile |N_i| − 1 na siatkach JiNan/HangZhou (2–4). Bez limitu
#: ścieżki bocznymi ulicami dochodzą do 18–20 sygnalizacji. Slot ablacyjny.
GRAPH_REACH_M = 600.0


class CityFlowAPI:
    """Trzy metody silnika CityFlow, które rdzeń i Advanced-MP wołają wprost.

    Semantyka jak w CityFlow: pozycja pojazdu to odległość od POCZĄTKU pasa, pasy
    tylko zwykłe (bez wewnętrznych `:węzeł_…` SUMO) — tak jak klucze `lane_length`.
    Silnik czytamy przy każdym wywołaniu, bo `world.eng` podmienia się w `reset()`.
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
    """(lanelinki, pasy startowe) każdej fazy zielonej — jak upstream, ale z `g`.

    Czyta `inter.lanelinks` (wynik `getControlledLinks` z konstrukcji), a nie silnik,
    żeby działało także przed pierwszym `reset()`, gdy połączenie jest zamknięte.
    """
    lanelinks, startlanes = [], []
    for phase in inter.green_phases:
        links, starts = [], []
        # strict=False celowo: stan fazy bywa dłuższy niż lista linków, a upstream
        # w tym miejscu robi `break` — ucinamy tak samo
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
    """`Intersection.pseudo_step` o semantyce `world_cityflow.Intersection.step` (Δ = 1 s).

    - Zmiana fazy = najpierw `yellow_phase_time` s fazy przejściowej (LibSignal nie
      zapala w niej `y`, tylko gasi kolidujące sygnały na `r` — jak faza 0 w CityFlow).
      Akcje w tym czasie są ignorowane, a `current_phase` to wciąż STARA faza; potem
      faza docelowa, a licznik liczy dalej (tryb 'add' w CityFlow).
    - Utrzymanie fazy = licznik +1.
    - Bez przejścia, gdy nic nie gaśnie (`create_yellows` nie zna tej pary faz, np.
      faza docelowa tylko dokłada zielone) albo `yellow_phase_time == 0` (tryb benchmark).

    Każdy krok podtrzymuje fazę w SUMO: fazy programu RL trwają 1 s i bez tego
    SUMO przełączyłoby je samo.
    """
    tl = inter.eng.trafficlight
    action = int(action)
    inter.virtual_phase = action  # pole upstreamu: ostatnia żądana akcja
    if inter.transition is not None:  # trwa przejście: (indeks w programie, faza docelowa)
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
    """ATT jak CityFlow: (Σ czasów dojechanych + Σ (teraz − wjazd) jadących) / ich liczba."""
    running = [now - t for v, t in entered.items() if v not in finished]
    n = len(finished) + len(running)
    return (sum(finished.values()) + sum(running)) / n if n else 0.0


def protocol_yellow() -> int | None:
    """`trainer.yellow_length` z Registry; None poza biegiem (testy, ręczne użycie)."""
    try:
        return int(Registry.mapping["trainer_mapping"]["setting"].param["yellow_length"])
    except (KeyError, AttributeError):
        return None


def _downstream(light: Any, owner: dict[str, str], reach_m: float) -> set[str]:
    """Sygnalizacje, do których ruch z wylotów `light` dojeżdża przez węzły bez
    sygnalizacji, na odcinku ≤ `reach_m` (Dijkstra po długościach krawędzi)."""
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
        if other is not None:  # ścieżka kończy się na PIERWSZEJ sygnalizacji — także własnej
            if other != me:
                found.add(other)
            continue
        for nxt in edge.getOutgoing():
            heapq.heappush(heap, (dist + nxt.getLength(), nxt.getID(), nxt))
    return found


def build_graph(net_file: str, reach_m: float = GRAPH_REACH_M) -> dict[str, Any]:
    """Graf sygnalizacji sieci SUMO w formacie `common/utils.build_index_intersection_map_*`.

    Zastępuje `build_index_intersection_map_sumo`, który łączy dwie sygnalizacje tylko
    wtedy, gdy JEDNA krawędź prowadzi wprost z węzła do węzła, a id węzła równa się id
    sygnalizacji. Na sieci z OSM oba warunki zwykle zawodzą: między sygnalizacjami są
    węzły pośrednie (zjazdy, przejścia), a sygnalizacja złączona (`joinedS_…`) steruje
    kilkoma węzłami o innych id (Kraków: 8 z 25). Wynik: 1 krawędź na 25 węzłów.

    Tu krawędź t→u (ruch z t dojeżdża do u, więc u agreguje od t) istnieje, gdy od
    któregoś wylotu t da się dojechać do węzła sterowanego przez u przez węzły bez
    sygnalizacji, na odcinku ≤ `reach_m`. Id i kolejność węzłów jak w LibSignalu —
    sieć czytana tym samym `readNet(net_file)` (z `withPrograms` sumolib tworzy
    sygnalizacje w innej kolejności) i `RoadNetGraph.bare_id`. Krawędzie bez duplikatów
    i pętli własnych (kontrakt `RoadNetGraph`); `node_degrees`/`node_list`/`edge_list`
    liczone po krawędziach wjazdowych, jak upstream.
    """
    net = sumolib.net.readNet(net_file)
    lights = net.getTrafficLights()
    # węzeł → sygnalizacja: węzeł, do którego wjeżdżają pasy przez nią sterowane
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
            self._patch(inter, install=False)  # połączenie zamknięte — program wgra reset()

    def _patch(self, inter: Any, *, install: bool) -> None:
        """Punkty 4, 7 i 8 z docstringu modułu na jednym skrzyżowaniu."""
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
        if install:  # te same trzy wywołania co na końcu `Intersection.__init__`
            program = inter.id + "_rl"
            logic = self.eng.trafficlight.Logic(program, 0, 0, inter.full_phases)
            self.eng.trafficlight.setProgramLogic(inter.id, logic)
            self.eng.trafficlight.setProgram(inter.id, program)

    def reset(self) -> None:
        kept = self.id2intersection
        super().reset()  # restart SUMO; upstream buduje tu NOWE obiekty Intersection
        for tl_id, fresh in self.id2intersection.items():
            kept[tl_id].__dict__.update(fresh.__dict__)
        self.id2intersection = {tl_id: kept[tl_id] for tl_id in self.id2intersection}
        self.intersections = list(self.id2intersection.values())
        for inter in self.intersections:
            self._patch(inter, install=True)

    def get_lane_waiting_vehicle_count(self) -> dict[str, int]:
        """Punkt 5. Te same klucze co upstream (`inter.lanes`), inna definicja."""
        lane = self.eng.lane
        return {
            lid: lane.getLastStepHaltingNumber(lid)
            for inter in self.intersections
            for lid in inter.lanes
        }

    def get_average_travel_time(self) -> float:
        """Punkt 6."""
        return travel_time(self.vehicles, self.inside_vehicles, self.get_current_time())
