"""Pętla środowiskowa rdzenia — spina `ppo.py` z symulatorem CityFlow.

`ppo.py` nie zna symulatora, `features.py` nie zna uczenia. Tutaj obie strony się
spotykają, i to jest jedyne miejsce w rdzeniu, które musi ODTWORZYĆ PROTOKÓŁ
`trainer/tsc_trainer.py:train()` co do joty — inaczej nasze liczby przestaną być
porównywalne z CoLightem i Advanced-CoLightem, a cała tabela T1 traci sens.

CZTERY RZECZY PRZEPISANE Z `tsc_trainer` DOSŁOWNIE (i po co):

1. DECYZJA CO Δt, NIE CO KROK. Akcja zapada raz na `action_interval` (15 s), po
   czym silnik robi 15 kroków po 1 s z tą samą akcją. Epizod 3600 s = 240 decyzji.

2. NAGRODA TO ŚREDNIA Z TYCH 15 KROKÓW, nie suma i nie wartość na końcu. Suma
   byłaby 15× większa, co samo w sobie nic by nie psuło, ale przewagi i strata
   wartości wyszłyby w innej skali niż u baseline'u — a porównujemy właśnie
   nagrody.

3. KOLEJKĘ I OPÓŹNIENIE ZBIERAMY RAZ NA DECYZJĘ, po wykonaniu interwału (tak
   robi `Metrics.update`), i uśredniamy tak samo: `queue` i `delay` dzielone
   przez (liczba decyzji × liczba skrzyżowań), `reward` tylko przez liczbę decyzji.
   Te dzielniki są różne w LibSignalu i to nie jest pomyłka — powtarzamy je,
   żeby kolumny znaczyły to samo.

4. KOLEJNOŚĆ AKCJI. `World.step` robi `self.intersections[i].step(actions[i])`,
   czyli indeksuje po SWOJEJ liście, a nasz wektor akcji jest indeksowany
   `node_idx` z grafu. Bez przenumerowania akcja jednego skrzyżowania trafiłaby
   do drugiego — nic by się nie wywaliło, sieć uczyłaby się szumu. Permutację
   liczymy raz w konstruktorze i testujemy osobno.

CZEGO TU NIE MA: epsilon-greedy, `learning_start`, sieci target. PPO uczy się
z każdego zebranego segmentu, a eksploracja bierze się z entropii polityki —
dlatego pętla jest krótsza niż DQN-owa i nie ma w niej `total_decision_num`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from tsc_marl.core.features import StateBuilder
from tsc_marl.core.graph import RoadNetGraph
from tsc_marl.core.nets import ActorCritic
from tsc_marl.core.ppo import PPO, PPOConfig, RolloutBuffer
from tsc_marl.domain.sensors import FailureSpec

if TYPE_CHECKING:
    from collections.abc import Callable

    from world.world_cityflow import World

    from runlog import Run
    from tsc_marl.robustness.training import FailureMix, UWStream

# Protokół raportowania z papieru (Advanced-CoLight 5.1): wynik biegu to ŚREDNIA
# Z 10 OSTATNICH epizodów testowych, nie pojedynczy test po treningu. Przy
# niestabilnym uczeniu różnica sięgała w tym repo setek sekund.
REPORT_WINDOW = 10


@dataclass(frozen=True)
class LoopConfig:
    """Parametry przebiegu. Wartości domyślne = zamrożony protokół pracy."""

    steps: int = 3600
    test_steps: int = 3600
    action_interval: int = 15
    episodes: int = 200
    test_when_train: bool = True

    @property
    def decisions(self) -> int:
        return self.steps // self.action_interval

    @property
    def test_decisions(self) -> int:
        return self.test_steps // self.action_interval

    @classmethod
    def from_registry(cls, **overrides: Any) -> LoopConfig:
        from common.registry import Registry  # noqa: PLC0415 - dopiero po bootstrapie

        param = Registry.mapping["trainer_mapping"]["setting"].param
        keys = ("steps", "test_steps", "action_interval", "episodes", "test_when_train")
        base = {k: param[k] for k in keys if k in param}
        return cls(**{**base, **overrides})


class TrafficEnv:
    """`World` LibSignala widziany na poziomie DECYZJI, nie kroku symulacji.

    Nie dziedziczy po `TSCEnv` z submodułu, bo tamten woła `agent.get_ob()` na
    agentach LibSignala — a nasz rdzeń nie jest agentem LibSignala. Robimy to
    samo, co `TSCEnv.step` robi z `world`, tylko obserwacje bierzemy ze
    `StateBuilder`.
    """

    def __init__(
        self,
        world: World,
        graph: RoadNetGraph,
        features: StateBuilder,
        action_interval: int,
        *,
        make_features: Callable[[FailureSpec], StateBuilder] | None = None,
        failures: FailureMix | None = None,
    ) -> None:
        self.world = world
        self.graph = graph
        self.features = features
        self.action_interval = action_interval

        # AWARIE W TRENINGU (Faza 3). `StateBuilder` dostaje scenariusz w
        # konstruktorze, a nie w `reset` — poziom szumu i sposób podstawienia są
        # jego stanem, nie stanem epizodu. Zmiana scenariusza między epizodami
        # znaczy więc NOWY builder; `make_features` jest tą samą fabryką, której
        # `robustness/degrade.py` używa do przechodzenia po scenariuszach T2.
        # Buforujemy po specyfikacji, bo mieszanka ma kilka pozycji, a epizodów
        # są setki — i bo `_build_structs` czyta topologię, nie stan ruchu.
        self._make_features = make_features
        self.failures = failures
        self._builders: dict[FailureSpec, StateBuilder] = {}
        if failures is not None and make_features is None:
            raise ValueError("mieszanka awarii bez `make_features` — nie ma czym przebudować stanu")

        self.episode = -1
        self._to_world = world_order(graph, world)
        self._delay = _delay_generators(world, graph.order_intersections(world))
        self._reset_stats()

    # ---------------------------------------------------------------- protokół

    def _reset_stats(self) -> None:
        n = self.graph.n_nodes
        self._decisions = 0
        self._reward_sum = np.zeros(n, dtype=np.float64)
        self._queue_sum = np.zeros(n, dtype=np.float64)
        self._delay_sum = np.zeros(n, dtype=np.float64)
        # WIEK OBSŁUGI (Faza 4). Ile decyzji minęło, odkąd dana faza dostała zielone.
        # Metryka GŁODZENIA, której `att`/`queue` nie widzą: uśrednianie po sieci i po
        # czasie zamazuje jedno skrzyżowanie trzymane w martwej fazie przez pół epizodu.
        # Indeks [węzeł, faza] w KOLEJNOŚCI GRAFU — tej samej co `_queue_sum`, `_delay`
        # i wektor akcji (permutacja do świata jest dopiero w `step`). Maska legalnych
        # faz jest konieczna, bo `n_phases` to MAKSIMUM po sieci i węzły o 2 fazach mają
        # resztę kolumn martwych; bez niej `wait_max` mierzyłby fazy nieistniejące
        # i rósł liniowo z długością epizodu na każdej sieci niejednorodnej.
        self._since_green = np.zeros((n, self.features.n_phases), dtype=np.float64)
        self._legal_phase = np.zeros((n, self.features.n_phases), dtype=bool)
        for idx, inter in enumerate(self.features.intersections):
            self._legal_phase[idx, : len(inter.phases)] = True
        self._age_max = 0.0
        # WIEK OBSŁUGI Z POPYTEM. Sam `_age_max` nie wie, czy na zagłodzonej fazie
        # ktokolwiek stoi — a faza bez ruchu może legalnie czekać dowolnie długo
        # (w praktyce miejskiej bywa to celowe: wloty bez detekcji dostają zielone
        # bezwarunkowo, patrz `tsc_marl_research/krakow-its.md`). Bez tego rozróżnienia
        # metryka karze politykę za to, że nie obsługuje pustego wlotu.
        self._phase_mov = self.features.phase_incidence()  # [N, n_faz, n_mov]
        self._age_demand_max = 0.0

    def _builder(self, spec: FailureSpec) -> StateBuilder:
        if spec not in self._builders:
            self._builders[spec] = self._make_features(spec)  # type: ignore[misc]
        return self._builders[spec]

    def reset(
        self, episode: int | None = None, *, failure: FailureSpec | None = None
    ) -> np.ndarray:
        """Nowy epizod. -> obserwacja [N, ob_length].

        `episode` steruje LOSOWANIEM AWARII (`features.reset_sensors`), więc
        podany jawnie czyni bieg eval-only odtwarzalnym niezależnie od tego, ile
        epizodów treningowych go poprzedziło. Domyślnie licznik idzie sam.

        `failure` wymusza scenariusz zamiast losowania z mieszanki — tak
        `CoreTrainer.evaluate` trzyma epizod TESTOWY na czystym: krzywa `att` ma
        pozostać porównywalna z rdzeniem z Fazy 1 i z tabelą T1, a odporność
        mierzy się osobno (`tsc degrade`), na zamrożonym checkpoincie.
        """
        self.episode = self.episode + 1 if episode is None else episode
        if self._make_features is not None:
            spec = failure
            if spec is None:
                spec = self.failures.sample(self.episode) if self.failures else FailureSpec()
            self.features = self._builder(spec)
        elif failure is not None and failure.active:
            raise ValueError("env bez `make_features` nie umie włączyć awarii w locie")
        self.world.reset()
        self.features.reset_sensors(self.episode)
        self._reset_stats()
        return self.features.observe()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        """Jedna decyzja: `action_interval` kroków silnika z tą samą akcją.

        -> (obserwacja, nagroda uśredniona po interwale, done).

        `done` jest ZAWSZE False: epizod urywa się po 3600 s, ale stan nie jest
        terminalny — ruch płynie dalej. Bootstrap `last_value` załatwia to
        poprawnie (patrz `ppo.py`, pułapka 1).
        """
        if actions.shape != (self.graph.n_nodes,):
            raise ValueError(
                f"akcje mają być [{self.graph.n_nodes}], są {tuple(np.shape(actions))}"
            )
        in_world_order = np.asarray(actions)[self._to_world]

        reward_acc = np.zeros(self.graph.n_nodes, dtype=np.float64)
        for _ in range(self.action_interval):
            self.world.step(in_world_order)
            reward_acc += self.features.rewards()
        rewards = reward_acc / self.action_interval

        kolejka, popyt = self._queue_and_demand()
        self._reward_sum += rewards
        self._queue_sum += kolejka
        self._delay_sum += np.array([g.generate() for g in self._delay], dtype=np.float64).ravel()
        self._decisions += 1
        # Faza wybrana w tej decyzji DOSTAŁA zielone, reszta czeka o jedną decyzję dłużej.
        # `actions` jest w kolejności grafu (permutacja `_to_world` dotyczy tylko silnika).
        self._since_green += 1.0
        self._since_green[np.arange(self.graph.n_nodes), np.asarray(actions, dtype=int)] = 0.0
        self._age_max = max(self._age_max, float(self._since_green[self._legal_phase].max()))
        # Faza „z popytem" = obsługuje choć jeden ruch, na którym ktoś TERAZ stoi.
        z_popytem = self._legal_phase & (self._phase_mov & popyt[:, None, :]).any(-1)
        if z_popytem.any():
            self._age_demand_max = max(
                self._age_demand_max, float(self._since_green[z_popytem].max())
            )
        return self.features.observe(), rewards.astype(np.float32), False

    def _queue(self) -> np.ndarray:
        """Suma oczekujących na pasach wjazdowych, per skrzyżowanie — to samo, co
        liczy `LaneVehicleGenerator(['lane_waiting_count'], in_only=True)`
        w `Metrics`. Liczymy z `reward_lanes`, a nie z nagrody, żeby metryka nie
        zależała od tego, który wariant nagrody jest akurat włączony."""
        return self._queue_and_demand()[0]

    def _queue_and_demand(self) -> tuple[np.ndarray, np.ndarray]:
        """(kolejka per węzeł [N], czy na ruchu ktoś stoi [N, n_mov]).

        Jedno odpytanie silnika na obie rzeczy. Czytamy PRAWDĘ z silnika, nie wektor
        obserwacji — bo pod awarią detektora obserwacja kłamie, a metryka głodzenia
        ma mierzyć, co NAPRAWDĘ stało na wlocie, nie co zaraportował zepsuty czujnik.
        """
        waiting = self.world.get_info("lane_waiting_count")
        kolejka = np.zeros(self.graph.n_nodes, dtype=np.float64)
        popyt = np.zeros((self.graph.n_nodes, self.features.n_mov), dtype=bool)
        for idx, s in enumerate(self.features.structs):
            kolejka[idx] = sum(waiting[lane] for lane in s.reward_lanes)
            for m, (wjazdy, _) in enumerate(s.movements):
                popyt[idx, m] = sum(waiting[lane] for lane in wjazdy) > 0
        return kolejka, popyt

    # ---------------------------------------------------------------- metryki

    def metrics(self) -> dict[str, float]:
        """Kolumny zgodne z `Metrics` LibSignala — patrz punkt 3 w docstringu."""
        if self._decisions == 0:
            raise RuntimeError("epizod nie wystartował — najpierw reset() i step()")
        n = self.graph.n_nodes
        # Kolejka i opóźnienie PER WĘZEŁ (średnia po decyzjach) — z nich liczą się
        # zarówno kolumny zgodne z LibSignalem (średnia po sieci), jak i OGON.
        kolejka = self._queue_sum / self._decisions  # [N]
        return {
            "att": float(self.world.get_average_travel_time()),
            "throughput": float(self.world.get_cur_throughput()),
            "reward": float(self._reward_sum.sum() / self._decisions),
            "queue": float(kolejka.sum() / n),
            "delay": float(self._delay_sum.sum() / (self._decisions * n)),
            # --- OGON (Faza 4). Średnia sieciowa ukrywa katastrofę na jednej czwartej
            # sieci — to samo zastrzeżenie, przez które bramka jest per skrzyżowanie,
            # a nie globalna. Mechanizmy celujące w GŁODZENIE i SPRAWIEDLIWOŚĆ obsługi
            # (action masking) nie ruszą `att`; ruszą te trzy kolumny. Mierzenie ich
            # przez `att` daje zero i wygląda jak porażka mechanizmu, a jest
            # niezgodnością metryki z celem — patrz `faza3-po-t2.md` §7.
            "queue_max": float(kolejka.max()),
            "queue_p90": float(np.quantile(kolejka, 0.9)),
            # Sekundy, nie decyzje — żeby czytało się w tej samej jednostce co `att`.
            # To MAKSIMUM PO EPIZODZIE z maksimum po (węzeł, faza legalna), czyli
            # najdłuższe zagłodzenie, jakie w epizodzie wystąpiło gdziekolwiek.
            # UWAGA przy interpretacji: mierzy odstęp między DECYZJAMI wyboru fazy,
            # więc nie odejmuje żółtego ani międzyzielonego.
            "wait_max": float(self._age_max * self.action_interval),
            # Ten sam wiek, ale liczony TYLKO dla faz, na których ktoś stoi. To jest
            # metryka, którą należy raportować jako głodzenie; `wait_max` zostaje jako
            # diagnostyka surowa (odstęp między zielonymi, niezależnie od popytu).
            # UWAGA przy interpretacji: to GÓRNE ograniczenie czasu oczekiwania
            # konkretnego pojazdu — wiek fazy liczy się od ostatniego zielonego,
            # a pojazd mógł podjechać później.
            "wait_max_demand": float(self._age_demand_max * self.action_interval),
        }


def _bare_id(intersection_id: str) -> str:
    """`world_sumo` prefiksuje id `GS_`, a graf trzyma je bez prefiksu."""
    return intersection_id[3:] if intersection_id.startswith("GS_") else intersection_id


def world_order(graph: RoadNetGraph, world: World) -> np.ndarray:
    """Permutacja `pozycja w world.intersections -> node_idx` (punkt 4 w docstringu).

    Użycie: `actions_dla_swiata = actions[world_order(...)]`. Osobna funkcja, bo to
    najgroźniejszy cichy błąd w całej pętli — akcja jednego skrzyżowania wykonana
    na drugim nie wywala niczego, po prostu uczy sieć szumu — i chcemy go testować
    bez stawiania CityFlow.
    """
    return np.array(
        [graph.node_id2idx[_bare_id(inter.id)] for inter in world.intersections],
        dtype=np.int64,
    )


def _delay_generators(world: World, intersections) -> list[Any]:
    from generator.lane_vehicle import LaneVehicleGenerator  # noqa: PLC0415

    return [
        LaneVehicleGenerator(world, inter, ["lane_delay"], in_only=True, average="all")
        for inter in intersections
    ]


class CoreTrainer:
    """Trening rdzenia: epizod zbierania → jeden update PPO → (opcjonalnie) test.

    Segment rolloutu = CAŁY epizod (240 decyzji). To wybór, nie konieczność: PPO
    zwykle tnie epizod na krótsze segmenty, ale przy 12–16 agentach jeden epizod
    to już 2880–3840 próbek, a segment równy epizodowi znaczy, że każdy update
    widzi pełny cykl ruchu i nie trzeba bootstrapować w środku.
    """

    def __init__(
        self,
        env: TrafficEnv,
        ppo: PPO,
        config: LoopConfig | None = None,
        run: Run | None = None,
        logger: logging.Logger | None = None,
        uncertainty: UWStream | None = None,
    ) -> None:
        self.env = env
        self.ppo = ppo
        self.config = config or LoopConfig()
        self.run = run
        # `uncertainty is None` = rdzeń Fazy 1 bit w bit (warstwa dostaje log_rho=None,
        # czyli ścieżkę oryginału z `agent/colight.py`), a nie „wektor zer".
        self.uncertainty = uncertainty
        # Bieg trwa godzinami — bez linii na epizod nie widać, czy uczenie żyje.
        # `run.log` pisze do CSV, ten logger na konsolę i do logu biegu.
        self.logger = logger
        # Urządzenie CZYTAMY z sieci — `from_world` przeniósł ją tam przed
        # zbudowaniem PPO. `edge_index` jest stały przez cały bieg, więc jedzie
        # na urządzenie raz, a nie co decyzję.
        self.device = next(ppo.ac.parameters()).device
        self.edge_index = env.graph.edge_index.to(self.device)
        self.history: list[dict[str, float]] = []
        self._episode_failure = "clean"

    # --------------------------------------------------------------- budowanie

    @classmethod
    def from_world(
        cls,
        world: World,
        *,
        ppo_config: PPOConfig | None = None,
        loop_config: LoopConfig | None = None,
        run: Run | None = None,
        logger: logging.Logger | None = None,
        device: torch.device | str = "cpu",
        failures: FailureMix | None = None,
        uncertainty: UWStream | None = None,
        **feature_overrides: Any,
    ) -> CoreTrainer:
        """Składa graf, cechy, sieci i PPO z żywego `world` + Registry.

        Kolejność skrzyżowań idzie WYŁĄCZNIE przez `graph.order_intersections`,
        czyli przez kontrakt z `graph.py`. To jedyne wejście do świata przy
        budowie obserwacji, nagród i akcji.

        `device` domyślnie CPU, bo rdzeń ma się dać zbudować i przetestować bez
        karty. Rozwiązuje je `runner/train_core.py` z `--ngpu` (patrz
        `libsignal/session.py:torch_device`) — rdzeń go NIE zgaduje ani nie czyta
        z Registry, żeby dało się go uruchomić poza procesem LibSignala.
        """
        graph = RoadNetGraph.from_registry()
        order = graph.order_intersections(world)

        def make_features(spec: FailureSpec) -> StateBuilder:
            return StateBuilder.from_registry(world, order, failure=spec, **feature_overrides)

        # Bez mieszanki budujemy stan dokładnie tak jak dotąd — ze scenariuszem
        # z configu (`model.failure`, domyślnie brak). Z mieszanką scenariusz
        # wybiera `TrafficEnv.reset` na epizod, więc startowy builder ma być
        # czysty i i tak zostanie podmieniony przed pierwszym krokiem.
        features = (
            make_features(FailureSpec())
            if failures is not None
            else StateBuilder.from_registry(world, order, **feature_overrides)
        )
        cfg = loop_config or LoopConfig.from_registry()
        env = TrafficEnv(
            world,
            graph,
            features,
            cfg.action_interval,
            make_features=make_features if failures is not None else None,
            failures=failures,
        )
        # KOLEJNOŚĆ JEST ISTOTNA: sieć na urządzenie, dopiero potem `PPO`.
        # `PPO.__post_init__` tworzy optymalizatory Adama z parametrów `ac` —
        # zbudowane przed przeniesieniem trzymałyby stan (`exp_avg`) na CPU,
        # a parametry na GPU, i pierwszy `step()` by się wywalił.
        ac = ActorCritic.from_registry(features.ob_length, features.n_phases).to(device)
        return cls(env, PPO(ac, ppo_config or PPOConfig()), cfg, run, logger, uncertainty)

    # ---------------------------------------------------------------- epizody

    def _log_rho(self, obs: np.ndarray) -> torch.Tensor | None:
        """log ρ [E+N] dla TEGO odczytu albo None (bieg bez mechanizmu).

        Woła się PRZED `act`, bo głowa ma zobaczyć ten sam wektor stanu, który
        widzi polityka — razem z awarią, jeśli akurat trwa. To jest cała idea:
        głowa nie ma dostępu do warstwy awarii ani do prawdy, tak jak nie miałaby
        go na skrzyżowaniu.
        """
        return None if self.uncertainty is None else self.uncertainty.log_rho(obs)

    def collect(self, episode: int | None = None) -> tuple[RolloutBuffer, dict[str, float]]:
        """Jeden epizod treningowy: polityka PRÓBKUJE (eksploracja PPO)."""
        cfg = self.config
        buffer = RolloutBuffer(
            cfg.decisions,
            self.env.graph.n_nodes,
            self.env.features.ob_length,
            self.edge_index,
            device=self.device,
            with_log_rho=self.uncertainty is not None,
        )
        raw = self.env.reset(episode)
        if self.uncertainty is not None:
            self.uncertainty.reset()  # bufor okna — patrz `UWStream.reset`
        obs, log_rho = self._obs(raw), self._log_rho(raw)
        for _ in range(cfg.decisions):
            actions, log_probs, values = self.ppo.act(obs, self.edge_index, log_rho=log_rho)
            next_raw, rewards, _ = self.env.step(self._actions(actions))
            buffer.add(obs, actions, log_probs, values, self._obs(rewards), log_rho=log_rho)
            obs, log_rho = self._obs(next_raw), self._log_rho(next_raw)
        # Krytyk nie dostaje log ρ (decyzja z docstringu `Critic`), więc bootstrap
        # wartości za końcem segmentu jest tu niezmieniony.
        buffer.compute_returns(self.ppo.value_of(obs, self.edge_index), self.ppo.config)
        return buffer, self.env.metrics()

    @torch.no_grad()
    def evaluate(self, episode: int | None = None) -> dict[str, float]:
        """Epizod testowy: argmax, TYLKO ACTOR, ruch CZYSTY.

        Krytyk nie jest wołany — to jest ta „inferencja = tylko actor" z planu,
        i tak samo będzie wyglądać wdrożenie na Pi w Fazie 5.

        CZYSTY, mimo że trening może iść z awariami: `att_last10` z tej krzywej
        trafia do tabeli obok rdzenia z Fazy 1, a kolumna mieszająca scenariusze
        nie znaczyłaby nic. Odporność mierzy `tsc degrade` na zamrożonym
        checkpoincie, tym samym planem co wszystkie inne sterowniki.

        Mechanizm ZOSTAJE włączony, bo jest częścią polityki, a nie warunków —
        i dzięki temu czysty wynik ewaluatora T2 (`UWPolicyController`) ma prawo
        zgadzać się z ostatnim testem treningu.
        """
        raw = self.env.reset(episode, failure=FailureSpec())
        if self.uncertainty is not None:
            self.uncertainty.reset()
        for _ in range(self.config.test_decisions):
            logits = self.ppo.ac.actor(self._obs(raw), self.edge_index, log_rho=self._log_rho(raw))
            raw, _, _ = self.env.step(self._actions(logits.argmax(-1)))
        return self.env.metrics()

    # ------------------------------------------------------ granica numpy ↔ torch

    def _obs(self, array: np.ndarray) -> torch.Tensor:
        """Symulator mówi wyłącznie numpy, czyli CPU. To jedno z dwóch miejsc,
        w których dane przekraczają granicę urządzenia — trzymamy je razem, żeby
        nie trzeba było ich szukać po pętli."""
        return torch.from_numpy(array).to(self.device)

    def _actions(self, actions: torch.Tensor) -> np.ndarray:
        """Druga strona granicy. `.cpu()` MUSI tu być: `Tensor.numpy()` na
        tensorze CUDA rzuca wyjątkiem, a na CPU jest no-opem — więc ta wersja
        działa identycznie w obu trybach."""
        return actions.cpu().numpy()

    # ----------------------------------------------------------------- trening

    def train(self, episodes: int | None = None) -> dict[str, float]:
        """Pełny bieg. -> skalary raportowe (te same nazwy co `experiment.ingest`)."""
        n_episodes = episodes if episodes is not None else self.config.episodes
        test_att: list[float] = []

        for e in range(n_episodes):
            buffer, train_metrics = self.collect(e)
            stats = self.ppo.update(buffer)
            # ZBIERAMY PRZED `evaluate`: epizod testowy przestawia `env.features`
            # na czysty builder i zeruje bufor `UWStream`, więc odczytane później
            # `uw_spread` opisywałoby test, nie trening. Cicho i myląco.
            mech = self._mechanism_metrics()
            self._log(e, "train", {**train_metrics, **stats, **mech})

            test_metrics = None
            if self.config.test_when_train:
                test_metrics = self.evaluate(e)
                # log ρ na CZYSTYM w tej samej kolumnie co ATT testu — to jest A3a
                # mierzone w treningu: mechanizm ma na czystym prawie milczeć.
                self._log(e, "test", {**test_metrics, **self._mechanism_metrics(test=True)})
                test_att.append(test_metrics["att"])
                self.history.append(test_metrics)
            self._say(e, train_metrics, stats, test_metrics, mech)

        return self._report(test_att)

    def _mechanism_metrics(self, *, test: bool = False) -> dict[str, float]:
        """Diagnostyka epizodu: co mechanizm w ogóle zobaczył i jaka była awaria.

        Bez niej pusty wynik ramienia 3 jest nieodróżnialny od biegu, w którym
        modulacja przez cały trening milczała — a to jest właśnie ta przyczyna
        zera, przed którą ostrzega §3 notatki (`uw_active` ≈ 3% na czystym).
        Wołać ZAWSZE zanim `evaluate` podmieni `env.features`.
        """
        out: dict[str, float] = {}
        if self.env.failures is not None and not test:
            sensors = self.env.features.sensors
            self._episode_failure = self._failure_slug()
            out["failure_active"] = float(self.env.features.failure.active)
            out["n_broken"] = float(len(sensors.broken)) if sensors is not None else 0.0
        if self.uncertainty is not None:
            out.update(self.uncertainty.episode_stats())
        return out

    def _failure_slug(self) -> str:
        from tsc_marl.robustness.degrade import slug  # noqa: PLC0415 - cykl importu

        return slug(self.env.features.failure)

    def _log(self, episode: int, split: str, values: dict[str, float]) -> None:
        if self.run is not None:
            self.run.log(step=episode, split=split, **values)

    def _say(self, episode, train: dict, stats: dict, test: dict | None, mech: dict) -> None:
        """Jedna linia na epizod. `kl` i `clip` są tu obok ATT nieprzypadkowo —
        to one pokazują, że lr jest za duży, zanim krzywa ATT zdąży to pokazać."""
        if self.logger is None:
            return
        line = (
            f"ep {episode:3d} | train att {train['att']:7.1f} reward {train['reward']:8.1f} "
            f"| kl {stats['approx_kl']:.4f} clip {stats['clip_frac']:.3f} "
            f"entropy {stats['entropy']:.3f}"
        )
        if "failure_active" in mech:
            line += f" | awaria {self._episode_failure} ({int(mech['n_broken'])} pasów)"
        if "uw_spread" in mech:
            line += f" spread {mech['uw_spread']:.2f} active {mech['uw_active']:.1%}"
        if test is not None:
            line += f" | TEST att {test['att']:7.1f} throughput {test['throughput']:.0f}"
        self.logger.info(line)

    def _report(self, test_att: list[float]) -> dict[str, float]:
        result: dict[str, float] = {}
        if test_att:
            window = test_att[-REPORT_WINDOW:]
            result = {
                "att": test_att[-1],
                "att_last10": sum(window) / len(window),
                "att_best": min(test_att),
                "episodes_logged": float(len(test_att)),
            }
            if self.history:
                result.update({k: self.history[-1][k] for k in ("throughput", "queue", "delay")})
        if self.run is not None and result:
            self.run.result(**result)
        return result

    # ------------------------------------------------------------- checkpointy

    def save(self, path: Path | str) -> None:
        """Zapisujemy TĘ SIEĆ, KTÓRA PODEJMUJE DECYZJE — bez `ValueNorm` wznowiony
        trening liczyłby wartości w innej skali niż zapisany krytyk (ten sam błąd,
        co checkpoint sieci target w `colight.py`, znaleziony w Fazie 0)."""
        payload = {
            "actor": self.ppo.ac.actor.state_dict(),
            "critic": self.ppo.ac.critic.state_dict(),
            "actor_opt": self.ppo.actor_opt.state_dict(),
            "critic_opt": self.ppo.critic_opt.state_dict(),
        }
        if self.ppo.value_norm is not None:
            payload["value_norm"] = self.ppo.value_norm.state_dict()
        torch.save(payload, Path(path))

    def load(self, path: Path | str) -> None:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        self.ppo.ac.actor.load_state_dict(payload["actor"])
        self.ppo.ac.critic.load_state_dict(payload["critic"])
        self.ppo.actor_opt.load_state_dict(payload["actor_opt"])
        self.ppo.critic_opt.load_state_dict(payload["critic_opt"])
        if self.ppo.value_norm is not None and "value_norm" in payload:
            self.ppo.value_norm.load_state_dict(payload["value_norm"])
