"""Sterowniki do ewaluacji pod awarią — WSZYSTKIE czytają tę samą warstwę detektorów.

DLACZEGO NIE AGENCI LIBSIGNALA. `efficientmp` i `advancedmp` z `libsignal/agents/`
czytają silnik WPROST (`eng.get_lane_vehicles()`, `world.get_info(...)`), a warstwa
awarii siedzi wyłącznie w `StateBuilder`. Pod awarią sensora te baseline'y widziałyby
PRAWDĘ, a polityka śmieci — krzywe degradacji porównywałyby dwie różne rzeczy. Tutaj
MaxPressure liczy decyzję z WEKTORA STANU zbudowanego przez `StateBuilder` (więc
z tych samych, popsutych odczytów) przez macierz incydencji faz. Na czystym wejściu
ta rekonstrukcja zgadza się z agentem co do decyzji na próbkach rozstrzygalnych
(`uncertainty/dataset.expert_check`: 1,0000); różnice zostają tylko w remisach, gdzie
o wyniku decyduje ostatni bit sumowania.

Kontrakt: `act(obs)` dostaje obserwację [N, ob_length] w kolejności WĘZŁÓW GRAFU
i zwraca akcje [N] w tej samej kolejności — przenumerowanie na kolejność świata robi
`TrafficEnv.step`, jedno miejsce dla wszystkich sterowników.

`state_independent` = sterownik nie czyta obserwacji (FixedTime). Ewaluator liczy go
wtedy tylko na czystym scenariuszu — pod awarią ma z definicji ten sam wynik.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import torch


class Controller(Protocol):
    name: str
    state_independent: bool

    def act(self, obs: np.ndarray) -> np.ndarray: ...


class PressureController:
    """Efficient-MP (`w1=None`) albo Advanced-MP (Alg. 1) liczony z wektora stanu.

    p(s) = Σ_{(l,m)∈s} e(l,m) — blok EP (pierwsze `n_mov` kolumn);
    d(s) = Σ_{(l,m)∈s} r(l,m) — blok demand (kolejne `n_mov`, tylko stan `ats`).
    Zatrzask minimalnego zielonego (`current_phase_time < t_min`) jak w agencie:
    tej informacji nie ma w stanie, ale nie jest odczytem detektora, tylko stanem
    sterownika, więc awaria sensora jej nie dotyczy.
    """

    state_independent = False

    def __init__(
        self,
        incidence: np.ndarray,
        intersections: Sequence[Any],
        n_mov: int,
        *,
        t_min: int,
        w1: float | None = None,
        name: str | None = None,
    ) -> None:
        if incidence.shape[0] != len(intersections) or incidence.shape[2] != n_mov:
            raise ValueError(f"incydencja {incidence.shape} nie pasuje do N={len(intersections)}")
        self.incidence = incidence.astype(np.float64)
        self.intersections = list(intersections)
        self.n_mov = n_mov
        self.t_min = t_min
        self.w1 = w1
        self.name = name or ("efficientmp" if w1 is None else "advancedmp")

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float64)
        if self.w1 is not None and x.shape[1] < 2 * self.n_mov:
            raise ValueError("Advanced-MP potrzebuje bloku demand — stan musi być `ats`")
        pressure = np.einsum("npm,nm->np", self.incidence, x[:, : self.n_mov])
        choice = pressure.argmax(-1)
        cur = np.array([inter.current_phase for inter in self.intersections], dtype=np.int64)
        if self.w1 is not None:
            demand = np.einsum("npm,nm->np", self.incidence, x[:, self.n_mov : 2 * self.n_mov])
            keep = demand[np.arange(len(cur)), cur] * self.w1 > pressure.max(-1)
            choice = np.where(keep, cur, choice)
        young = np.array(
            [inter.current_phase_time < self.t_min for inter in self.intersections], dtype=bool
        )
        return np.where(young, cur, choice).astype(np.int64)


class FixedTimeController:
    """Jak `agent/fixedtime.py`: trzymaj fazę `t_fixed` sekund, potem następna.
    Jedyny sterownik niezależny od odczytów — dolna warstwa fallbacku z planu."""

    state_independent = True

    def __init__(self, intersections: Sequence[Any], *, t_fixed: int, name: str = "fixedtime"):
        self.intersections = list(intersections)
        self.t_fixed = t_fixed
        self.name = name

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.array(
            [
                inter.current_phase
                if inter.current_phase_time < self.t_fixed
                else (inter.current_phase + 1) % len(inter.phases)
                for inter in self.intersections
            ],
            dtype=np.int64,
        )


class PolicyController:
    """Actor rdzenia (GAT+PPO+CTDE): argmax po logitach, TYLKO actor — jak w
    `CoreTrainer.evaluate` i jak na docelowym urządzeniu brzegowym."""

    state_independent = False

    def __init__(self, actor: torch.nn.Module, edge_index: torch.Tensor, name: str = "gatppo"):
        self.actor = actor.eval()
        self.edge_index = edge_index
        self.name = name

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        logits = self.actor(torch.from_numpy(np.asarray(obs, dtype=np.float32)), self.edge_index)
        return logits.argmax(-1).numpy().astype(np.int64)


class ColightController:
    """Advanced-CoLight z checkpointu: `get_action(test=True)` agenta LibSignala.

    Agent przyjmuje wiersze w kolejności `world.intersections`, a my podajemy je
    w kolejności węzłów grafu. Na roadnetach CityFlow te kolejności są dziś
    identyczne (kontrakt w `core/graph.py`) — i to jest WARUNEK tego sterownika,
    sprawdzany przy budowie (`identity_order`), a nie zakładany."""

    state_independent = False

    def __init__(self, agent: Any, *, identity_order: bool, name: str = "advancedcolight"):
        if not identity_order:
            raise ValueError(
                "kolejność world.intersections != kolejność grafu — ColightController "
                "podawałby obserwacje niewłaściwym skrzyżowaniom"
            )
        self.agent = agent
        self.name = name

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self.agent.get_action(obs, None, test=True), dtype=np.int64)


class EnsembleController:
    """Kilka actorów rdzenia (różne seedy): decyduje PIERWSZY, pozostali tylko głosują.

    Rozbieżność ich rozkładów nad fazami to epistemic w samej polityce RL — bez
    dodatkowego treningu i bez EDL. Ta sama dekompozycja co BALD i ensemble z 2.0:
    MI = H(średniego rozkładu) − średnia H(rozkładów). Obok odsetek członków, którzy
    wybraliby inną fazę niż decydujący — prostsza, łatwa do opisania miara.

    Decyduje jeden seed, a nie większość: pytamy, czy rozbieżność WYKRYWA kłopoty
    wdrożonej polityki, więc trajektoria ma być trajektorią tej polityki."""

    state_independent = False

    def __init__(self, actors, edge_index: torch.Tensor, name: str = "gatppo_ens") -> None:
        if len(actors) < 2:
            raise ValueError("ensemble potrzebuje co najmniej dwóch actorów")
        self.actors = [a.eval() for a in actors]
        self.edge_index = edge_index
        self.name = name
        self.reset_stats()

    def reset_stats(self) -> None:
        self._mi: list[np.ndarray] = []
        self._disagree: list[np.ndarray] = []
        self._aleatoric: list[np.ndarray] = []

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        probs = torch.softmax(torch.stack([a(x, self.edge_index) for a in self.actors]), -1)
        mean = probs.mean(0)  # [N, P]
        total = -(mean * torch.log(mean + 1e-10)).sum(-1)
        aleatoric = -(probs * torch.log(probs + 1e-10)).sum(-1).mean(0)
        votes = probs.argmax(-1)  # [K, N]
        driver = votes[0]
        self._mi.append((total - aleatoric).clamp_min(0).numpy())
        self._aleatoric.append(aleatoric.numpy())
        self._disagree.append((votes[1:] != driver).float().mean(0).numpy())
        return driver.numpy().astype(np.int64)

    def episode_stats(self) -> dict[str, float]:
        if not self._mi:
            return {}
        mi = np.concatenate(self._mi)
        return {
            "ens_mi": float(mi.mean()),
            "ens_mi_p90": float(np.quantile(mi, 0.9)),
            "ens_aleatoric": float(np.concatenate(self._aleatoric).mean()),
            "ens_disagree": float(np.concatenate(self._disagree).mean()),
        }
