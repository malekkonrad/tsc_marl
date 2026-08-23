"""ATS — Advanced Traffic State (Zhang i in. 2022, "Expression might be enough").

Bezstanowe prymitywy: wszystko bierze `inter` (obiekt `Intersection` LibSignala)
albo gołe słowniki z silnika, nic nie trzyma w `self`. Powód: Advanced-MP jest
agentem JEDNEGO skrzyżowania (`BaseAgent`, `self.inter_obj`), a Advanced-CoLight
trzyma wszystkie skrzyżowania w jednym modelu i musi liczyć ATS w pętli po
`world.intersections`. Ta sama matematyka, dwie hierarchie klas — więc mieszka
poza obiema.

Def 4  efficient pressure:  e(l,m) = (1/M)·Σ q(l') − (1/N)·Σ q(m')
Def 5  phase pressure:      p(s)   = Σ e(l,m), (l,m) ∈ s
Def 8  effective range:     L      = V_max · t_duration      (165 m przy Δt=15)
Def 9  effective running:   r(l,m) = Σ r_e(l'),  l' ∈ l
Def 10 phase demand:        d(s)   = Σ r(l,m), (l,m) ∈ s
Def 11 ATS:                 {e(l,m); r(l,m)} dla każdego RUCHU (nie fazy — sumy
                            po fazie, Def. 5 i 10, to już decyzja Advanced-MP)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # tylko dla typów — inaczej moduł nie zaimportuje się bez
    from world.world_cityflow import Intersection  # bootstrapu ścieżki do submodułu

WAITING_SPEED = 0.1  # próg "stoi" — ten sam, którego używa world_cityflow

# Ruch (l,m) po sparowaniu lanelinków: (używane pasy wjazdowe, używane wyjazdowe).
Movement = tuple[list[str], list[str]]


def road_of(lane: str) -> str:
    """Id pasa w CityFlow to '<roadId>_<laneIndex>', a sam roadId zawiera '_'
    ('road_2_1_1_0' → droga 'road_2_1_1', pas '0'). Stąd rsplit z limitem 1."""
    return lane.rsplit("_", 1)[0]


def always_green_lanelinks(inter: Intersection) -> set[tuple[str, str]]:
    """Lanelinki obecne we WSZYSTKICH fazach — w roadnetach CityFlow (JiNan,
    HangZhou) są to skręty w prawo. Nie konkurują o zielone, więc nie należą ani
    do pressure, ani do demand; referencyjna implementacja autorów sumuje
    wyłącznie ruchy przelotowe i w lewo (`phase_1 = tr_mo_pr[1] + tr_mo_pr[4]`)."""
    per_phase = [set(map(tuple, ll)) for ll in inter.phase_available_lanelinks]
    return set.intersection(*per_phase) if per_phase else set()


def phase_movements(inter: Intersection, drop_always_green: bool = True) -> list[list[Movement]]:
    """[faza][ruch] → (używane pasy wjazdowe, używane pasy wyjazdowe).

    Ruch (l,m) z Def. 4 to para DRÓG, a CityFlow daje lanelinki na poziomie
    PASÓW — stąd grupowanie po `(road_of(start), road_of(end))`. M i N z Def. 4
    to liczba UŻYWANYCH pasów danego ruchu, nie wszystkich pasów drogi, więc
    zbieramy zbiory `ins`/`outs` zamiast liczyć pasy z roadnetu.

    Indeks fazy == indeks akcji: `phase_available_lanelinks` budowane jest pętlą
    po `inter.phases`, a tam żółta jest już wycięta (`world_cityflow.py`).
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
    """Płaska, ODDUPLIKOWANA lista ruchów (l,m) — dziedzina ATS z Def. 11.

    `phase_movements` grupuje po fazach, więc ruch mający zielone w kilku fazach
    występuje tam wielokrotnie. Do wektora stanu wchodzi raz: wymiar obserwacji
    ma zależeć od geometrii skrzyżowania, nie od schematu faz. Kolejność =
    pierwsze wystąpienie idąc po fazach; przy parameter sharing CoLighta cecha
    nr k musi znaczyć to samo na każdym skrzyżowaniu (patrz test kolejności).
    """
    seen: dict[tuple[str, str], Movement] = {}
    for moves in phase_movements(inter, drop_always_green):
        for ins, outs in moves:
            seen.setdefault((road_of(ins[0]), road_of(outs[0])), (ins, outs))
    return list(seen.values())


def in_lanes(movements: Sequence[Movement]) -> set[str]:
    """Pasy wjazdowe ruchów sterowanych — dziedzina demand (Def. 9)."""
    return {lane for ins, _ in movements for lane in ins}


def all_lanes(movements: Sequence[Movement]) -> set[str]:
    """Pasy wjazdowe i wyjazdowe — dziedzina pressure (Def. 4 patrzy w obie strony)."""
    return {lane for ins, outs in movements for lane in (*ins, *outs)}


def movement_pressure(movements: Sequence[Movement], q: Mapping[str, float]) -> list[float]:
    """e(l,m) dla każdego ruchu — Def. 4, BEZ sumowania po fazie (to robi Def. 5)."""
    return [
        sum(q[lane] for lane in ins) / len(ins) - sum(q[lane] for lane in outs) / len(outs)
        for ins, outs in movements
    ]


def movement_running(movements: Sequence[Movement], running: Mapping[str, float]) -> list[float]:
    """r(l,m) dla każdego ruchu — Def. 9, BEZ sumowania po fazie (to robi Def. 10)."""
    return [sum(running.get(lane, 0) for lane in ins) for ins, _ in movements]


def phase_pressure(movements: Sequence[Sequence[Movement]], q: Mapping[str, float]) -> list[float]:
    """p(s) dla każdej fazy — Def. 5, czyli DOSŁOWNIE suma Def. 4 po ruchach fazy.
    Bierze kształt z `phase_movements`, nie płaską listę."""
    return [sum(movement_pressure(moves, q)) for moves in movements]


def phase_demand(
    movements: Sequence[Sequence[Movement]], running: Mapping[str, float]
) -> list[float]:
    """d(s) dla każdej fazy — Def. 10, suma Def. 9 po ruchach fazy."""
    return [sum(movement_running(moves, running)) for moves in movements]


def windowed_queue(
    lanes: Iterable[str],
    lane_vehicles: Mapping[str, Iterable[str]],
    distances: Mapping[str, float],
    speeds: Mapping[str, float],
    lane_length: Mapping[str, float],
    obs_length: float,
) -> dict[str, int]:
    """Kolejka liczona TYLKO w oknie `obs_length` od stop-linii — tak jak liczy ją
    referencyjna implementacja autorów (`_get_part_observations`, OBS_LENGTH=167).
    Papier tego nie mówi: Def. 4 sugeruje kolejkę całego pasa."""
    q = {}
    for lane in lanes:
        length = lane_length.get(lane, 0.0)
        n = 0
        for vid in lane_vehicles.get(lane, []):
            if speeds.get(vid, 1.0) > WAITING_SPEED:  # domyślnie "jedzie" → nie kolejka
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
    """r_e(l') dla pasów wjazdowych — Def. 8 + Def. 9. Tylko pojazdy JADĄCE i
    będące w zasięgu L od skrzyżowania. `effective_range` to albo stałe okno
    (OBS_LENGTH autorów), albo mapa pas → L = V_max(pas)·t_duration.
    `unlimited=True` = ablacja "Config3" z papieru (pełna obserwacja pasa)."""
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
            if speeds.get(vid, 0.0) < WAITING_SPEED:  # domyślnie "stoi" → to kolejka, nie demand
                continue
            if unlimited or length - distances.get(vid, 0.0) <= limit:
                n += 1
        counts[lane] = n
    return counts
