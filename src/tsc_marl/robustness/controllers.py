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


class Gate(Protocol):
    """Sygnał bramki fallbacku: [N] bool, True = ten węzeł oddaje sterowanie regule.

    Trzy implementacje różnią się WYŁĄCZNIE sposobem liczenia maski — wyrocznia
    (prawda z warstwy awarii, górne ograniczenie), reguła (utknięte/zerowe odczyty
    z okna) i UQ (głowa obserwacyjna). Powłoka `GatedController` jest jedna, więc
    histereza, diagnostyka i test no-op piszą się raz dla wszystkich trzech.

    `obs` w sygnaturze `mask` bywa nieużywane (wyrocznia czyta warstwę awarii, nie
    obserwację) i tak ma zostać: powłoka woła każdą bramkę tak samo.

    `last_score` [N] float — SUROWY wynik z ostatniego `mask()`, przed progiem
    i przed histerezą. Wszystkie trzy bramki liczą go i tak (ułamek martwych wlotów /
    zamrożonych ruchów / ruchów ponad `u_ref`), więc wystawienie go nic nie kosztuje,
    a jest jedyną wielkością, z której da się odpowiedzieć na pytanie „DLACZEGO ten
    węzeł oddał sterowanie" — sama maska mówi tylko, że oddał. Przed pierwszym
    `mask()` i w oknie rozgrzewki jest wektorem zer.
    """

    last_score: np.ndarray

    def mask(self, obs: np.ndarray) -> np.ndarray: ...

    def reset(self) -> None: ...


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


class UWPolicyController:
    """Actor rdzenia + modulacja wiarygodnością krawędzi (Faza 3, UW-attention).

    TE SAME WAGI co `PolicyController` i to jest sedno kolumny w T2: modulacja jest
    BEZPARAMETROWA (log ρ przychodzi z zewnątrz, `state_dict` się nie zmienia), więc
    „rdzeń" i „rdzeń + UW-attention" liczą się z jednego checkpointu z `runs/t1`.
    Różnica między kolumnami to wyłącznie mechanizm, nie inny trening, nie inny seed.

    Co się dzieje w `act`: odczyt ląduje w oknie głowy obserwacyjnej -> σ na cechę
    -> u na ruch -> u na krawędź (średnia — kryterium (f) bramki 2.2) -> log ρ =
    −relu(u − u_ref)/τ -> attention waży sąsiadów tą liczbą. Wszystko z TEGO SAMEGO
    popsutego wektora stanu, który dostaje polityka: głowa nie ma dostępu do prawdy
    ani do warstwy awarii, tak jak nie miałaby go na skrzyżowaniu.

    `reset_stats` czyści bufor okna — `robustness/degrade.py` woła go na starcie
    każdego epizodu. Bez tego pierwsze kroki epizodu liczyłyby u z odczytów sprzed
    `world.reset()`, czyli z zupełnie innego stanu ruchu.
    """

    state_independent = False

    def __init__(
        self,
        actor: torch.nn.Module,
        edge_index: torch.Tensor,
        uncertainty: Any,
        name: str = "gatppo_uw",
    ) -> None:
        self.actor = actor.eval()
        self.edge_index = edge_index
        self.uncertainty = uncertainty
        self.name = name
        self.reset_stats()

    def reset_stats(self) -> None:
        self.uncertainty.reset()
        self._spread: list[float] = []
        self._active: list[float] = []

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        from tsc_marl.uncertainty.obs_head import group_spread  # noqa: PLC0415

        x = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        log_rho = self.uncertainty.log_rho(obs)
        self._spread.append(group_spread(log_rho, self.edge_index, x.shape[0]))
        self._active.append(float((log_rho < 0).float().mean()))
        return self.actor(x, self.edge_index, log_rho=log_rho).argmax(-1).numpy().astype(np.int64)

    def episode_stats(self) -> dict[str, float]:
        """Diagnostyka „czy mechanizm w ogóle zadziałał" — bez niej pusty wynik T2
        jest nieodróżnialny od mechanizmu, który przez cały epizod milczał.

        `uw_spread` to rozstęp log ρ w grupie softmaxu, czyli JEDYNA część sygnału,
        którą attention widzi (na stałą wspólną dla grupy jest niezmienniczy).
        """
        if not self._spread:
            return {}
        return {
            "uw_spread": float(np.mean(self._spread)),
            "uw_spread_p90": float(np.quantile(self._spread, 0.9)),
            "uw_active": float(np.mean(self._active)),
        }


class GatedController:
    """Polityka + reguła pod jedną bramką: akcja KAŻDEGO węzła z jednego albo drugiego.

    DLACZEGO OWIJA ZWYKŁĄ POLITYKĘ, a nie `UWPolicyController`: kolumna w T2 ma się
    różnić od kolumny rdzenia DOKŁADNIE jedną rzeczą. Owinięcie sterownika z
    modulacją attention mierzyłoby bramkę I modulację naraz, a wyniku nie dałoby się
    przypisać — tym bardziej że modulacja wyszła wobec ATT neutralna (`results/t2_uw.md`).

    DLACZEGO PER WĘZEŁ, a nie globalnie: `mask0p25_intersection` to 25% skrzyżowań
    martwych i 75% zdrowych, a globalnie rdzeń (510,0 s) i FixedTime (507,0 s) idą
    łeb w łeb — średnia ukrywa katastrofę na jednej czwartej sieci. Bramka globalna
    musiałaby wybrać między zepsuciem zdrowych a zostawieniem martwych. Akcje i tak
    są per węzeł, więc to nic nie kosztuje.

    OBA STEROWNIKI WOŁANE W KAŻDEJ DECYZJI, dla wszystkich węzłów; maska tylko
    wybiera. Przy `FixedTimeController` jest to obojętne (czyta stan świata, nie
    obserwację), ale przy bramkowaniu sterownika z PAMIĘCIĄ — `UWPolicyController`
    trzyma okno ostatnich odczytów — pominięcie wywołania zrobiłoby dziurę w oknie.
    Kontrakt „wołamy oba zawsze" zostaje więc częścią powłoki, nie szczegółem.
    """

    state_independent = False

    def __init__(
        self,
        policy: Controller,
        rule: Controller,
        gate: Gate,
        *,
        k_on: int = 1,
        k_off: int = 1,
        oracle: Gate | None = None,
        name: str = "gated",
        trace: bool = False,
    ) -> None:
        self.policy = policy
        self.rule = rule
        self.gate = gate
        # ŚLAD PER (DECYZJA, WĘZEŁ) — wyłączony domyślnie. Wyjaśnia pojedyncze
        # przełączenie („dlaczego TO skrzyżowanie oddało sterowanie w 47. decyzji"),
        # czego żadna metryka epizodu nie umie: `gate_share` mówi ILE, nie KTÓRE
        # i KIEDY. Włącza go `tsc degrade run --trace`, a NIE klucz w planie —
        # celowo, bo `gate` wchodzi do odcisku biegu (`MECHANISM_KEYS`) i dopisanie
        # tam flagi logowania unieważniłoby wszystkie policzone punkty, choć nie
        # zmienia ani jednej decyzji.
        self.trace = trace
        # HISTEREZA: wejście po `k_on` kolejnych decyzjach powyżej progu, wyjście po
        # `k_off` poniżej. Asymetria (k_off > k_on) celowa — wchodź ostrożnie, wychodź
        # jeszcze ostrożniej. Bez niej bramka o szumiącym sygnale migocze co decyzję,
        # a każde przełączenie resetuje fazę i psuje ruch bardziej niż jeden zły wybór.
        #
        # DOMYŚLNIE WYŁĄCZONA (1, 1) z dwóch powodów. Wyrocznia z `t2_gate` była
        # zmierzona bez niej i włączenie jej domyślnie po cichu zmieniłoby te liczby.
        # A poza tym to parametr MECHANIZMU: należy do planu, a stamtąd do odcisku
        # biegu (`MECHANISM_KEYS`), żeby jego zmiana unieważniała wynik zamiast go
        # dziedziczyć. Sygnał wyroczni jest stały w epizodzie, więc jej histereza i
        # tak nic nie zmienia — potrzebują jej bramki realne.
        self.k_on, self.k_off = k_on, k_off
        # WYŁĄCZNIE DIAGNOSTYKA. Wyrocznia czyta prawdę z warstwy awarii, więc nie
        # ma prawa dotknąć decyzji — służy do policzenia precyzji i czułości bramki
        # realnej, czyli do rozdzielenia „detektor jest słaby" od „przełączanie mało
        # daje". Te dwie rzeczy samo ATT skleja.
        #
        # Jej próg definiuje, co UZNAJEMY za węzeł oślepiony, i jest NIEZALEŻNY od
        # progu, na którym pracuje kolumna-wyrocznia. Zbyt ostry (0,75) daje przy
        # maskach pasowych zbiór pusty i obie metryki stają się nieokreślone —
        # zmierzone: przy `mask0p5` wyrocznia@0,75 nie wskazała ani jednego węzła.
        self.oracle = oracle
        self.name = name
        self.reset_stats()

    def reset_stats(self) -> None:
        """Woła też `gate.reset()` — i to jest tu najważniejsze. Wyrocznia jest
        bezstanowa, ale bramka UQ trzyma okno odczytów, a `robustness/degrade.py`
        odpala `reset_stats` na starcie każdego epizodu. Bez tej delegacji okno
        wchodziłoby w nowy epizod z odczytami sprzed `world.reset()`, czyli z
        zupełnie innego stanu ruchu."""
        self.gate.reset()
        if self.oracle is not None:
            self.oracle.reset()
        self._share: list[float] = []
        self._switches = 0
        self._previous: np.ndarray | None = None
        self._licznik: np.ndarray | None = None  # kolejne decyzje po tej samej stronie progu
        self._trafienia = np.zeros(3)  # [wspólne, tylko bramka, tylko wyrocznia]
        self._trace: list[tuple[int, int, float, bool, bool]] = []

    def act(self, obs: np.ndarray) -> np.ndarray:
        surowa = self.gate.mask(obs)
        maska = self._hysteresis(surowa)
        if self.oracle is not None:
            prawda = self.oracle.mask(obs)
            self._trafienia += [
                float((maska & prawda).sum()),
                float((maska & ~prawda).sum()),
                float((~maska & prawda).sum()),
            ]
        self._share.append(float(np.mean(maska)))
        if self._previous is not None:
            self._switches += int((maska != self._previous).sum())
        if self.trace:
            # Zapisujemy SUROWY wynik i obie maski: bez surowego nie widać, czy węzeł
            # był tuż pod progiem, a bez rozdzielenia surowa/po histerezie nie da się
            # odróżnić „detektor nie zadziałał" od „histereza jeszcze nie przepuściła".
            wynik = getattr(self.gate, "last_score", None)
            decyzja = len(self._share) - 1
            for node in range(len(maska)):
                self._trace.append(
                    (
                        decyzja,
                        node,
                        float(wynik[node]) if wynik is not None else float("nan"),
                        bool(surowa[node]),
                        bool(maska[node]),
                    )
                )
        self._previous = maska
        return np.where(maska, self.rule.act(obs), self.policy.act(obs))

    def _hysteresis(self, surowa: np.ndarray) -> np.ndarray:
        """Surowa maska -> maska po histerezie, per węzeł."""
        if self._previous is None:  # pierwsza decyzja: nie ma czego wygładzać
            self._licznik = np.zeros(len(surowa), dtype=int)
            return surowa
        zgodne = surowa != self._previous
        self._licznik = np.where(zgodne, self._licznik + 1, 0)
        prog = np.where(self._previous, self.k_off, self.k_on)
        zmiana = zgodne & (self._licznik >= prog)
        self._licznik = np.where(zmiana, 0, self._licznik)
        return np.where(zmiana, surowa, self._previous)

    def episode_trace(self) -> list[tuple[int, int, float, bool, bool]]:
        """Ślad epizodu: [(decyzja, węzeł, wynik surowy, maska surowa, tryb reguły)].

        Pusty, gdy `trace=False`. Czyści go `reset_stats`, czyli start epizodu —
        ta sama granica, na której zerują się `gate_share` i histereza."""
        return self._trace

    def episode_stats(self) -> dict[str, float]:
        """Diagnostyka „czy bramka w ogóle strzeliła" — bez niej pusty wynik jest
        nieodróżnialny od bramki, która przez cały epizod milczała (ta sama lekcja
        co `uw_spread` przy UW-attention). `gate_switches` liczy zmiany trybu na
        węzłach, bo to one, a nie sam odsetek, psują ruch przez reset fazy."""
        if not self._share:
            return {}
        out = {
            "gate_share": float(np.mean(self._share)),
            "gate_switches": float(self._switches),
        }
        # Precyzja i czułość TYLKO gdy mianownik istnieje. Przy scenariuszu bez awarii
        # (albo gdy odniesienie nie wskazało ani jednego węzła) „0,00" czytałoby się
        # jak zła detekcja, a znaczy „nie było czego wykrywać".
        wspolne, falszywe, przeoczone = self._trafienia
        if wspolne + falszywe > 0:
            out["gate_precision"] = float(wspolne / (wspolne + falszywe))
        if wspolne + przeoczone > 0:
            out["gate_recall"] = float(wspolne / (wspolne + przeoczone))
        return out


class MaskedController:
    """ACTION MASKING: twarde ograniczenie WIEKU OBSŁUGI. Żadna legalna faza nie może
    czekać dłużej niż `k_max` decyzji — gdy czeka, maska nadpisuje wybór polityki
    najstarszą przeterminowaną fazą.

    DLACZEGO BEZ SYGNAŁU O POPYCIE, choć „wymuś tylko tam, gdzie ktoś stoi" brzmi
    mądrzej. Dwa powody, oba rozstrzygające:

    1. **Deployowalność.** Popyt per RUCH dałoby się czytać z wektora obserwacji, ale
       pierwszy blok cech to przy stanie `ats` znormalizowane ciśnienie ruchu, które
       bywa UJEMNE — „popyt > 0" nie jest z niego odczytywalny bez arbitralnego progu.
       Czytanie prawdy z silnika byłoby wyrocznią, nie sterownikiem.
    2. **Odporność z KONSTRUKCJI.** Ta maska nie czyta ŻADNEGO czujnika — liczy
       wyłącznie własne akcje. Jest więc niewrażliwa na awarię detektorów, a to jest
       dokładnie reżim, w którym rdzeń porzuca kierunki: pod maskami `wait_max` rdzenia
       to 2386–3160 s przy epizodzie 3600 s (`faza3-po-t2.md` §8.7). Mechanizm, który
       tego nie widzi przez czujnik, bo go nie potrzebuje, jest w tym reżimie mocniejszy
       niż jakikolwiek oparty na odczytach.

    Spełnia trzy warunki opłacalności z §7.7 Z DEFINICJI: (1) nie jest estymatorem
    niepewności, tylko licznikiem własnych decyzji; (2) jest tani, gdy się myli —
    wymuszenie pustej fazy kosztuje jeden interwał; (3) jest DYSKRETNYM ograniczeniem
    zbioru akcji, nie ciągłym przeważeniem, na które polityka nie była trenowana.

    OGRANICZENIE GWARANCJI, do nazwania w tekście. Przy jednej przeterminowanej fazie
    na węzeł wiek nie przekroczy `k_max + 1` decyzji. Gdy przeterminowanych jest kilka
    naraz, obsłużyć można jedną, więc w najgorszym razie wiek dobija do
    `k_max + liczba_legalnych_faz − 1`. To NIE jest twarde `k_max` i tak trzeba to podać.

    CZEGO TA MASKA NIE ROBI: nie dotyka strefy dylematu (wymaga prędkości i odległości
    pojazdu, czego stan nie niesie — u [[bodagala-2026-ucatsc]] to osobny filtr na SUMO)
    ani minimalnego zielonego (`t_min` pilnuje zatrzask w silniku).
    """

    def __init__(
        self,
        policy: Controller,
        *,
        legal: np.ndarray,
        k_max: int,
        name: str = "masked",
    ) -> None:
        if k_max < 1:
            raise ValueError(f"k_max musi być ≥ 1, jest {k_max}")
        self.policy = policy
        self.legal = np.asarray(legal, dtype=bool)  # [N, n_faz]
        self.k_max = int(k_max)
        self.name = name
        self.reset_stats()

    def reset_stats(self) -> None:
        """Zeruje wiek na starcie epizodu — tak jak `GatedController` zeruje okno
        i histerezę. Bez tego pierwsze decyzje nowego epizodu wymuszałyby fazy na
        podstawie zaległości z poprzedniego."""
        self._since = np.zeros(self.legal.shape, dtype=np.int64)
        self._nadpisane = 0
        self._decyzje = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        akcje = np.asarray(self.policy.act(obs), dtype=np.int64).copy()
        # Przeterminowane = legalne ORAZ czekające co najmniej `k_max` decyzji.
        spozniona = self.legal & (self._since >= self.k_max)
        wymus = spozniona.any(-1)
        if wymus.any():
            # NAJSTARSZA przeterminowana, nie pierwsza z indeksu: przy kilku naraz
            # kolejność po wieku daje round-robin, a po indeksie zagłodziłaby ostatnią.
            wiek = np.where(spozniona, self._since, -1)
            wybor = wiek.argmax(-1)
            self._nadpisane += int((wymus & (akcje != wybor)).sum())
            akcje = np.where(wymus, wybor, akcje)
        self._since += 1
        self._since[np.arange(len(akcje)), akcje] = 0
        self._decyzje += 1
        return akcje

    def episode_stats(self) -> dict[str, float]:
        """`mask_share` = odsetek par (decyzja, węzeł), w których maska ZMIENIŁA akcję
        polityki — nie tych, w których była aktywna i zgodziła się z polityką. Ta sama
        zasada co przy `gate_share` i z tego samego powodu: mechanizm, który nic nie
        nadpisał, jest wobec wyniku no-opem i musi to być widoczne."""
        if not self._decyzje:
            return {}
        return {
            "mask_share": float(self._nadpisane / (self._decyzje * len(self.legal))),
            "mask_forced": float(self._nadpisane),
        }


class OracleGate:
    """Bramka-WYROCZNIA: czyta prawdę z warstwy awarii, nie z obserwacji.

    Nie jest sterownikiem do wdrożenia — jest GÓRNYM OGRANICZENIEM dla bramek
    realnych (reguła, UQ). Jeśli przy idealnej detekcji przełączanie nic nie daje,
    żaden detektor nie da więcej i nie ma po co go budować. Dlatego powstaje
    PIERWSZA, przed kalibracją jakiegokolwiek progu.

    CZYM JEST „OŚLEPIONY WĘZEŁ": ułamek jego PASÓW WJAZDOWYCH w `sensors.broken`,
    porównany z `threshold`. `in_lanes`, nie `all_lanes`, bo `failure_units`
    przypisuje pas do skrzyżowania, DO KTÓREGO pas dojeżdża — „bo tam stoi kamera".
    Pasy wyjazdowe węzła należą więc do jego sąsiadów w dół strumienia. Zmierzone na
    JiNan: każde sterowane skrzyżowanie ma 8 pasów wjazdowych i 20 wszystkich, czyli
    przy `all_lanes` całkowicie martwa kamera dawałaby ułamek 8/20 = 0,40 — i przy
    progu 0,5 bramka NIE strzeliłaby w scenariuszach `mask*_intersection`, czyli
    dokładnie tam, gdzie powstała.

    ZNANY LIMIT TEJ DEFINICJI: węzeł ze zdrowymi własnymi kamerami może dostawać
    skażone `pressure`, bo ono patrzy też na wyjazdy, czyli na detektory sąsiada.
    Wyrocznia tego nie łapie — sąsiad przełączy się sam, ale ciśnienie tego węzła
    zostaje częściowo zatrute. Świadomy wybór, nie przeoczenie.

    PRÓG: porównanie nieostre (`>=`), czyli „co najmniej połowa detektorów martwa”.
    Przy `unit: intersection` ułamek jest 0 albo 1 i rozstrzygnięcie remisu nie ma
    znaczenia; przy `unit: lane` ma, bo `mask` bierze DOKŁADNY ułamek jednostek, więc
    przy `mask0p5` i ośmiu pasach wjazdowych co czwarty węzeł siada na dokładnie 4/8.
    Wybór w stronę przełączenia, bo tam polityka wyraźnie przegrywa (743 s wobec
    507 s FixedTime). Rozstrzyga to jednak sweep `threshold`, nie ta konwencja —
    stąd parametr konstruktora, a nie stała.
    """

    def __init__(self, env: Any, *, threshold: float = 0.5) -> None:
        # `env`, NIE `env.features`: `robustness/degrade.py:212` podmienia
        # `env.features` przy każdym scenariuszu, a sterownik powstaje raz.
        # Zapamiętany snapshot to na zawsze `clean` (pierwszy scenariusz planu),
        # czyli bramka, która nigdy nie strzela — i fałszywy wniosek „nie ma o co grać".
        self.env = env
        self.threshold = threshold
        self._in_lanes: list[list[str]] | None = None
        self._built_for: Any = None

    def reset(self) -> None:
        """Wyrocznia jest bezstanowa; metoda istnieje dla jednolitego interfejsu
        z bramkami, które stan mają (UQ trzyma okno odczytów)."""

    def mask(self, obs: np.ndarray) -> np.ndarray:
        features = self.env.features
        sensors = features.sensors
        # `sensors is None` na czystym scenariuszu (`StateBuilder` tworzy warstwę
        # tylko dla aktywnej awarii), a `live` odsiewa scenariusze z `onset`:
        # `broken` jest niepuste od startu epizodu, ale przed onsetem `_measure`
        # zwraca czyste wartości, więc bramka nie ma czego gasić.
        if sensors is None or not sensors.live:
            self.last_score = np.zeros(features.n_nodes, dtype=np.float64)
            return np.zeros(features.n_nodes, dtype=bool)
        if features is not self._built_for:  # cache przeżywa epizod,
            self._in_lanes = [list(s.in_lanes) for s in features.structs]
            self._built_for = features  # ale nie podmianę scenariusza

        broken = sensors.broken
        part = [sum(lane in broken for lane in lanes) / len(lanes) for lanes in self._in_lanes]
        self.last_score = np.asarray(part, dtype=np.float64)
        return self.last_score >= self.threshold


class OracleUncertainty:
    """WYROCZNIA MODULACJI: log ρ dla UW-attention z PRAWDY warstwy awarii, nie z głowy.

    Ta sama rola co `OracleGate` dla bramek: GÓRNE OGRANICZENIE, nie sterownik do
    wdrożenia. Odpowiada na jedno pytanie, którego `t2_uw`/`t2_uw_src` nie rozstrzygają:
    czy ważenie sąsiadów wiarygodnością może pomóc PRZY IDEALNYM SYGNALE. Jeśli nie —
    mechanizm jest martwy niezależnie od głowy i C5 zamyka się bez zastrzeżeń. Jeśli
    tak — wąskim gardłem była głowa (trafienie 0,54 na zrzucie BC), nie idea.

    u węzła = ułamek jego pasów WJAZDOWYCH w `sensors.broken`, czyli DOKŁADNIE
    `OracleGate.last_score` — liczone przez nią, żeby „oślepiony węzeł" miał w pracy
    jedną definicję (i jej znany limit: `pressure` patrzy też na wyjazdy, czyli na
    detektory sąsiadów w dół strumienia). Tylko STRONA ŹRÓDŁA: krawędź j→i dostaje u(j),
    bo niesie `hidden_embedding(x_j)`, a pętla własna i→i — u(i). Wariant po stronie
    celu nie ma tu sensu jako sufit: mierzyłby idealnie coś, co nie jest wiarygodnością
    wiadomości (`obs_head.EDGE_SIDES`).

        log ρ = −u / τ,  u ∈ [0, 1]

    BEZ `u_ref`: prawda na zdrowej sieci to dokładnie 0 (`sensors is None` albo przed
    `onset`), więc no-op na czystym (A3a) jest tu tożsamością, nie kalibracją. Przy
    τ = 0,25 całkowicie martwy węzeł dostaje −4 naty, czyli ~55× mniejszą wagę.

    Interfejs jak `OnlineUncertainty` (`reset`, `log_rho`), więc wpina się w
    `UWPolicyController` bez zmian i dostaje tę samą diagnostykę `uw_spread`.
    """

    def __init__(self, env: Any, edge_index: torch.Tensor, *, tau: float) -> None:
        if tau <= 0:
            raise ValueError(f"τ musi być dodatnie, jest {tau}")
        # `env`, nie `env.features` — powód w `OracleGate.__init__`
        self._gate = OracleGate(env)
        self.src = edge_index[0].numpy()
        self.tau = float(tau)

    def reset(self) -> None:
        self._gate.reset()

    def log_rho(self, obs: np.ndarray) -> torch.Tensor:
        """[E + N] w kolejności `edge_index`, potem pętle własne — kontrakt
        `UncertaintyWeightedAttention` (tak dokleja je `add_self_loops`)."""
        self._gate.mask(obs)
        u = self._gate.last_score  # [N], zera na czystym i przed onsetem
        out = -np.concatenate([u[self.src], u]) / self.tau
        return torch.from_numpy(out.astype(np.float32))


class RuleGate:
    """Bramka-REGUŁA: ułamek ruchów węzła, których odczyty STOJĄ W MIEJSCU przez całe
    okno. Zero uczenia, zero artefaktów — licznik zamrożonych detektorów.

    Jedno kryterium łapie obie postacie martwego detektora: podstawienie `zero`
    (zero też się nie zmienia) i `hold` (zamrożony ostatni odczyt, awaria bez śladu
    w samej wartości). Szczegóły i wbudowana słabość — pusty wlot o małym ruchu
    raportuje zero i jest ZDROWY — w `obs_head.suspicious_movements`.

    To jest DOLNY punkt odniesienia dla bramki UQ: jeśli licznik zamrożeń bramkuje
    równie dobrze jak głowa niepewności, głowa nie zarabia w tej pracy na siebie.
    """

    def __init__(self, *, window: int, threshold: float, n_mov: int) -> None:
        self.window, self.threshold, self.n_mov = window, threshold, n_mov
        self._buf: list[np.ndarray] = []

    def reset(self) -> None:
        self._buf.clear()

    def mask(self, obs: np.ndarray) -> np.ndarray:
        from tsc_marl.uncertainty.obs_head import suspicious_movements  # noqa: PLC0415

        obs = np.asarray(obs, dtype=np.float32)
        self._buf.insert(0, obs)  # najnowszy z przodu — jak `window_index`
        del self._buf[self.window :]
        # STRAŻNIK ROZGRZEWKI. Przy niepełnym buforze okno jest przycięte, więc
        # KAŻDY ruch wygląda na zamrożony i bramka przełączyłaby całą sieć na
        # starcie epizodu. Ta sama reguła co w kalibracji progu (`full_window`).
        if len(self._buf) < self.window:
            self.last_score = np.zeros(obs.shape[0], dtype=np.float64)
            return np.zeros(obs.shape[0], dtype=bool)

        hist = np.stack(self._buf)  # [k, N, F]
        win = np.arange(self.window)[None, :]  # okno NAJNOWSZEGO kroku
        podejrzane = suspicious_movements(hist, win, self.n_mov)[0]  # [N, n_mov]
        self.last_score = podejrzane.mean(-1)
        return self.last_score >= self.threshold


class UQGate:
    """Bramka-UQ: ułamek ruchów węzła, których u przekracza własną normę z czystego
    okna (`GateReference.u_mov_ref`).

    TA SAMA POSTAĆ SYGNAŁU co w `RuleGate` — ułamek podejrzanych ruchów — żeby
    ablacja porównywała kryterium podejrzliwości, a nie dwie różne statystyki.
    Różnica: reguła pyta „czy ten detektor stoi", UQ pyta „czy ten odczyt jest
    nietypowo niepewny".

    PRZEWIDYWANA SŁABOŚĆ, zapisana przed pomiarem: σ rośnie także przy SZUMIE, gdzie
    detektory żyją i polityka wciąż bije FixedTime (gauss2: 319 s vs 507 s). Na
    zrzucie BC ten sygnał przekracza próg w 88% decyzji przy `gauss2` wobec 0% dla
    reguły — czyli bramka UQ ma tam masowo przełączać i tracić. Miara niepewności
    nie jest miarą BRAKU odczytu i ta kolumna ma to pokazać, a nie ukryć.
    """

    def __init__(self, uncertainty: Any, *, u_mov_ref: np.ndarray, threshold: float) -> None:
        self.uncertainty = uncertainty
        self.u_mov_ref = u_mov_ref
        self.threshold = threshold

    def reset(self) -> None:
        self.uncertainty.reset()

    def mask(self, obs: np.ndarray) -> np.ndarray:
        u_mov = self.uncertainty.movements(obs)  # [N, n_mov]
        if not self.uncertainty.ready:  # strażnik rozgrzewki — jak w regule
            self.last_score = np.zeros(u_mov.shape[0], dtype=np.float64)
            return np.zeros(u_mov.shape[0], dtype=bool)
        self.last_score = (u_mov >= self.u_mov_ref).mean(-1)
        return self.last_score >= self.threshold


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
