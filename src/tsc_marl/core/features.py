"""Stan i nagroda dla rdzenia Fazy 1 — jedno źródło prawdy dla wszystkich wariantów.

Ekstrakcję stanu wyjęto z `libsignal/agents/advanced_colight.py`, bo rdzeń (GAT actor +
centralny krytyk) nie jest agentem LibSignala i nie może dziedziczyć po
`CoLightAgent`. Dwie kopie tej samej matematyki to dokładnie ta klasa błędu,
której najbardziej się tu boimy: ATS policzony odrobinę inaczej nie wywala się,
tylko daje wynik gorszy o kilka procent — nie do odróżnienia od "no i tyle
wychodzi". Advanced-CoLight woła więc ten sam kod (patrz `advanced_colight.py`).

WARIANTY STANU (ablacja 1.6) — to jest progresja z literatury, od lewej do prawej:

    queue     q(l,m)                     średnia kolejka pasów wjazdowych ruchu
    pressure  Σ_in q − Σ_out q           klasyczny MaxPressure (Varaiya)
    ep        (1/M)Σ_in q − (1/N)Σ_out q efficient pressure, Def. 4 ([[wu-2021-efficient-pressure]])
    ats       {e(l,m); r(l,m)}           + effective running, Def. 11 ([[zhang-2022-advanced-mp]])

Każdy wariant kończy się one-hotem bieżącej fazy. Bloki są dopychane zerami
OSOBNO, nie na końcu wiersza: przy sieci z różnymi skrzyżowaniami (NY) padding
na końcu przesunąłby one-hot na inne kolumny na różnych skrzyżowaniach, a CoLight
dzieli JEDEN model między wszystkie — cecha nr k musi znaczyć wszędzie to samo.

WARIANTY NAGRODY (ablacja 1.6 + odporność w Fazie 3):

    queue     −Σ q po pasach wjazdowych skrzyżowania   (Advanced-CoLight, domyślny)
    pressure  −|Σ_(l,m) (Σ_in q − Σ_out q)|            (PressLight)

Nagroda kolejkowa jest domyślna, bo to ona stoi za wynikami Advanced-CoLighta:
efficient pressure siedzi w STANIE, nie w nagrodzie. Uzasadnienie teoretyczne —
minimalizacja kolejki ⟺ minimalizacja ATT ([[cai-2025-xlight]]).

Zgodność z CoLightem: upstream liczy nagrodę jako `LaneVehicleGenerator(...,
average="all", negative=True)` razy stała 12 w `agent/colight.py:243`. Ta stała
to liczba pasów wjazdowych na skrzyżowaniach JiNan/HangZhou (4 drogi × 3 pasy),
więc `średnia × 12` to po prostu SUMA — i tak to tutaj liczymy, wprost.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from tsc_marl.domain import ats
from tsc_marl.domain.sensors import FailureSpec, SensorModel, failure_units

if TYPE_CHECKING:  # tylko dla typów — inaczej moduł nie zaimportuje się bez
    from world.world_cityflow import Intersection, World  # bootstrapu ścieżki do submodułu

DEFAULT_MAX_SPEED = 11.11  # fallback jak w world_cityflow, gdy pas nie ma maxSpeed

STATE_VARIANTS = ("queue", "pressure", "ep", "ats")
REWARD_VARIANTS = ("queue", "pressure")


def engine_api(world: World) -> Any:
    """Silnik z API CityFlow: `world.eng` w CityFlow, widok `cityflow_api` w SUMO
    (`libsignal/sumo_world.py`). Jedno miejsce tej decyzji — rdzeń i Advanced-MP
    czytają kolejki i pozycje pojazdów tą samą drogą na obu silnikach."""
    return getattr(world, "cityflow_api", None) or world.eng


def phase_mask(phase_counts: Sequence[int], n_phases: int) -> np.ndarray | None:
    """[N, n_phases] bool — które akcje istnieją na danym skrzyżowaniu.

    None, gdy wszystkie mają po `n_phases` faz (JiNan/HangZhou): maska z samych
    jedynek niczego nie zmienia, a ścieżka bez niej zostaje bit w bit tą, na której
    policzono GATE Fazy 1. Na sieci niejednorodnej (Kraków: 2–7 faz) bez maski
    polityka wybrałaby fazę, której skrzyżowanie nie ma.
    """
    counts = np.asarray(phase_counts)
    if (counts >= n_phases).all():
        return None
    return np.arange(n_phases)[None, :] < counts[:, None]


class Structs(NamedTuple):
    """Statyka jednego skrzyżowania — nie zmienia się w trakcie symulacji, więc
    liczona raz przy budowie."""

    movements: list[ats.Movement]
    in_lanes: list[str]  # pasy wjazdowe ruchów STEROWANYCH — dziedzina demand
    all_lanes: list[str]  # wjazdowe + wyjazdowe ruchów sterowanych — dziedzina pressure
    reward_lanes: list[str]  # WSZYSTKIE pasy dróg wjazdowych — dziedzina nagrody (jak CoLight)
    effective_range: dict[str, float]  # pas → L = V_max·Δt (Def. 8)


class StateBuilder:
    """Buduje obserwacje [N, F] i nagrody [N] dla ustalonej kolejności skrzyżowań.

    Kolejność jest WEJŚCIEM, nie założeniem: rdzeń Fazy 1 poda
    `graph.order_intersections(world)`, a Advanced-CoLight `world.intersections`
    (tę, w której rodzic zbudował generatory). Patrz KONTRAKT KOLEJNOŚCI
    w `graph.py` — na roadnetach CityFlow te dwie kolejności są dziś takie same,
    ale nie wolno na tym polegać.
    """

    def __init__(
        self,
        world: World,
        intersections: Sequence[Intersection],
        *,
        state: str = "ats",
        reward: str = "queue",
        drop_always_green: bool = True,
        obs_length: float | None = None,
        action_interval: int = 15,
        n_phases: int | None = None,
        failure: FailureSpec | None = None,
    ) -> None:
        if state not in STATE_VARIANTS:
            raise ValueError(f"state={state!r}, dozwolone: {STATE_VARIANTS}")
        if reward not in REWARD_VARIANTS:
            raise ValueError(f"reward={reward!r}, dozwolone: {REWARD_VARIANTS}")

        self.world = world
        # `get_info` czyta cache odświeżany w `World._update_infos`, a ten liczy
        # TYLKO zasubskrybowane funkcje. LibSignalowe agenty subskrybują to przy
        # okazji, budując `LaneVehicleGenerator` — my generatorów nie mamy, więc
        # bez tej linii `rewards()` leci `KeyError: 'lane_waiting_count'` przy
        # pierwszym kroku rdzenia. Subskrypcja jest idempotentna.
        world.subscribe("lane_waiting_count")
        self.intersections = list(intersections)
        self.state = state
        self.reward = reward
        self.obs_length = obs_length
        self.drop_always_green = drop_always_green
        self.structs = [
            self._build_structs(inter, drop_always_green, action_interval, obs_length)
            for inter in self.intersections
        ]
        self.n_mov = max(len(s.movements) for s in self.structs)
        self.n_phases = n_phases or max(len(inter.phases) for inter in self.intersections)

        # Warstwa awarii sensora (Faza 2/3). JEDNA instancja na wszystkie
        # skrzyżowania — ten sam fizyczny detektor bywa czytany przez dwa
        # sąsiadujące skrzyżowania i musi im pokazać tę samą wartość.
        self.failure = failure or FailureSpec()
        self.sensors: SensorModel | None = None
        if self.failure.active:
            lanes = sorted({lane for st in self.structs for lane in st.all_lanes})
            self.sensors = SensorModel(
                self.failure, failure_units(self.intersections, lanes, self.failure.unit)
            )

    @classmethod
    def from_registry(
        cls, world: World, intersections: Sequence[Intersection], **overrides: Any
    ) -> StateBuilder:
        """Wariant konfigurowany tak jak reszta repo — przez YAML-e LibSignala."""
        from common.registry import Registry  # noqa: PLC0415 - dopiero po bootstrapie

        param = Registry.mapping["model_mapping"]["setting"].param
        kwargs: dict[str, Any] = {
            "state": param.get("state", "ats"),
            "reward": param.get("reward", "queue"),
            "drop_always_green": param.get("drop_always_green", True),
            "obs_length": param.get("obs_length", None),
            "failure": FailureSpec.from_registry(),
            "action_interval": Registry.mapping["trainer_mapping"]["setting"].param[
                "action_interval"
            ],
        }
        return cls(world, intersections, **{**kwargs, **overrides})

    def _build_structs(
        self,
        inter: Intersection,
        drop_always_green: bool,
        action_interval: int,
        obs_length: float | None,
    ) -> Structs:
        movements = ats.controlled_movements(inter, drop_always_green)
        lanes_in = sorted(ats.in_lanes(movements))
        return Structs(
            movements=movements,
            in_lanes=lanes_in,
            all_lanes=sorted(ats.all_lanes(movements)),
            # Nagroda liczy WSZYSTKIE pasy dróg wjazdowych, także skręty w prawo
            # wypadające z ruchów sterowanych — tak robi `LaneVehicleGenerator`
            # z `in_only=True`, więc nagroda zostaje porównywalna z CoLightem.
            reward_lanes=ats.in_road_lanes(inter),
            effective_range={
                lane: (
                    obs_length
                    if obs_length is not None
                    else self.world.all_lanes_speed.get(lane, DEFAULT_MAX_SPEED) * action_interval
                )
                for lane in lanes_in
            },
        )

    # ------------------------------------------------------------------ wymiary

    @property
    def n_nodes(self) -> int:
        return len(self.intersections)

    @property
    def ob_length(self) -> int:
        """Szerokość wiersza — JEDYNE miejsce, w którym wymiar obserwacji wchodzi
        do architektury (pierwszy `nn.Linear` encodera)."""
        blocks = 2 if self.state == "ats" else 1
        return blocks * self.n_mov + self.n_phases

    def phase_mask(self) -> np.ndarray | None:
        """[N, n_phases] bool w kolejności skrzyżowań tego buildera — patrz `phase_mask`
        na poziomie modułu. None na sieciach jednorodnych."""
        return phase_mask([len(inter.phases) for inter in self.intersections], self.n_phases)

    def movement_mask(self) -> np.ndarray | None:
        """[N, n_mov] bool — które kolumny bloku cech to PRAWDZIWE ruchy skrzyżowania,
        a które dopchane zerami do `n_mov` (maksimum po sieci). None na sieciach
        jednorodnych — ta sama konwencja co `phase_mask`.

        Potrzebne licznikowi zamrożonych odczytów (`RuleGate`, `RuleUncertainty`):
        dopchane zero NIGDY się nie zmienia, więc bez maski węzeł z jednym wlotem
        (Kraków: 1–7 wlotów) wyglądałby na w większości martwy także na czystym ruchu,
        a próg kalibrowany kwantylem na czystym wyszedłby ~1,0 — bramka nigdy by nie
        strzeliła."""
        return phase_mask([len(st.movements) for st in self.structs], self.n_mov)

    # ------------------------------------------------------------------ odczyt

    def _snapshot(self) -> tuple[dict, dict, dict]:
        """JEDNO pobranie na wywołanie: te trzy zwracają stan CAŁEJ sieci, nie
        skrzyżowania, więc wołanie ich w pętli kopiuje ten sam słownik N razy.
        Subskrypcja przez `world.get_info` nie pomoże — cache odświeża się co KROK
        symulacji (1 s), a decyzja zapada raz na Δt."""
        eng = engine_api(self.world)
        return eng.get_lane_vehicles(), eng.get_vehicle_distance(), eng.get_vehicle_speed()

    def _queue(self, struct: Structs, snapshot, full_queue) -> Mapping[str, float]:
        """Kolejka w oknie `obs_length` od stop-linii (tak liczy ją referencyjna
        implementacja autorów) albo pełnopasmowa, gdy okna nie ustawiono."""
        if self.obs_length is None:
            return full_queue
        lane_vehicles, distances, speeds = snapshot
        return ats.windowed_queue(
            struct.all_lanes,
            lane_vehicles,
            distances,
            speeds,
            self.world.lane_length,
            self.obs_length,
        )

    def observe(self, *, corrupt: bool = True) -> np.ndarray:
        """[skrzyżowania, ob_length], float32, w kolejności podanej przy budowie.

        `corrupt=False` omija warstwę awarii NIE RUSZAJĄC jej stanu (żadnego
        losowania, żadnej aktualizacji `hold`). Dzięki temu dwa wywołania w tym
        samym momencie symulacji dają parę (czysta, popsuta) z DOKŁADNIE tego
        samego stanu silnika — to jest zbiór uczący prototypu 2.0 i miara,
        względem której liczy się dotkliwość awarii.
        """
        sensors = self.sensors if corrupt else None
        if sensors is not None:
            sensors.begin_decision()
        snapshot = self._snapshot()
        lane_vehicles, distances, speeds = snapshot
        full_queue = (
            None if self.obs_length is not None else self.world.get_info("lane_waiting_count")
        )

        rows = np.zeros((self.n_nodes, self.ob_length), dtype=np.float32)
        for idx, (inter, s) in enumerate(zip(self.intersections, self.structs, strict=True)):
            q = self._queue(s, snapshot, full_queue)
            if sensors is not None:
                q = sensors.read(q, s.all_lanes, channel="queue")

            if self.state == "queue":
                head = [sum(q[lane] for lane in ins) / len(ins) for ins, _ in s.movements]
            elif self.state == "pressure":
                head = ats.movement_pressure(s.movements, q, normalize=False)
            else:  # "ep" oraz pierwszy blok "ats"
                head = ats.movement_pressure(s.movements, q)
            rows[idx, : len(head)] = head

            if self.state == "ats":
                running = ats.effective_running(
                    s.in_lanes,
                    lane_vehicles,
                    distances,
                    speeds,
                    self.world.lane_length,
                    s.effective_range,
                )
                if sensors is not None:
                    running = sensors.read(running, s.in_lanes, channel="running")
                demand = ats.movement_running(s.movements, running)
                rows[idx, self.n_mov : self.n_mov + len(demand)] = demand

            # Bieżąca faza: indeks w `inter.phases`, czyli w przestrzeni AKCJI (bez
            # żółtej). W trakcie przejścia trzyma fazę DOCELOWĄ — i o to chodzi,
            # bo obserwacja opisuje decyzję, nie stan lampy.
            rows[idx, self.ob_length - self.n_phases + inter.current_phase] = 1.0
        return rows

    def phase_incidence(self) -> np.ndarray:
        """[N, n_faz, n_mov] bool — które ruchy (kolumny bloku cech) obsługuje która faza.

        Most między stanem opisanym per RUCH a decyzją podejmowaną per FAZĘ: p(s) =
        Σ e(l,m) po ruchach fazy (Def. 5). Z tą macierzą decyzję MaxPressure da się
        policzyć z SAMEGO wektora stanu — czyli także z popsutego, co jest całą
        potrzebą ewaluacji pod awarią (`robustness/controllers.py`).
        """
        out = np.zeros((self.n_nodes, self.n_phases, self.n_mov), dtype=bool)
        for i, inter in enumerate(self.intersections):
            for phase, movements in enumerate(ats.phase_incidence(inter, self.drop_always_green)):
                out[i, phase, movements] = True
        return out

    def reset_sensors(self, episode: int) -> None:
        """Przelosuj TRWAŁY stan detektorów na nowy epizod. Bez tego ten sam
        zbiór pasów byłby zepsuty przez cały trening, a `hold` przeciekałby
        wartości sprzed `world.reset()`."""
        if self.sensors is not None:
            self.sensors.reset(episode)

    def rewards(self) -> np.ndarray:
        """[skrzyżowania], float32. Zawsze z PEŁNEJ kolejki pasa, nigdy z okna
        `obs_length` — okno jest własnością obserwacji (tak liczy je referencyjna
        implementacja), a nagroda ma zostać porównywalna z CoLightem.

        Warstwa awarii jej NIE DOTYKA i nie jest to kwestia flagi: `get_info`
        czyta cache świata, nie ścieżkę obserwacji. To jest scope CORE Fazy 3 —
        trening i nagroda na czystym, awaria tylko na wejściu polityki."""
        queue = self.world.get_info("lane_waiting_count")
        out = np.zeros(self.n_nodes, dtype=np.float32)
        for idx, s in enumerate(self.structs):
            if self.reward == "queue":
                out[idx] = -sum(queue[lane] for lane in s.reward_lanes)
            else:  # "pressure" — PressLight: karzemy NIEZRÓWNOWAŻENIE, stąd moduł
                out[idx] = -abs(sum(ats.movement_pressure(s.movements, queue, normalize=False)))
        return out
