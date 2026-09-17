"""Model awarii sensorów — przyrząd pomiarowy Fazy 2 (bramka 2.0) i protokołu 3.2.

GDZIE WSTRZYKUJEMY: na poziomie DETEKTORA, czyli w per-pasowe zliczenia
`q[lane]` i `running[lane]`, pomiędzy `windowed_queue`/`effective_running`
a `movement_pressure`/`movement_running` (patrz `features.observe`).

Nie na poziomie POJAZDU, bo `get_vehicle_distance`/`get_vehicle_speed` są
kluczowane id pojazdu GLOBALNIE dla całej sieci — psucie per pas wymagałoby ich
przebudowy, a realny detektor (pętla, kamera) i tak raportuje LICZBĘ, nie listę
pojazdów. Nie na poziomie RUCHU (gotowego wiersza cech), bo `movement_pressure`
zmieszało już kilka pasów w jedną liczbę: zepsuty detektor na jednym pasie to nie
to samo co popsuty cały ruch, a na tej granularności stoi claim per-edge z 2.2.

CO WYCHODZI Z TEGO ZA DARMO: `features.rewards()` czyta
`world.get_info('lane_waiting_count')` niezależnie od tej ścieżki, więc nagroda
zostaje CZYSTA z konstrukcji, a nie z flagi — czyli dokładnie scope CORE Fazy 3
(trening na czystym, awaria w runtime na ZAMROŻONEJ polityce).

DWIE KLASY USZKODZEŃ, o różnym czasie życia:

    TRWAŁE     losowane RAZ na epizod, kluczowane JEDNOSTKĄ awarii — które
               detektory są martwe (`mask`) i jak są rozkalibrowane (`drift`).
               Zepsuty czujnik nie migocze: losowanie maski co decyzję
               symulowałoby awarię fizycznie nieistniejącą, a głowa niepewności
               nauczyłaby się tego migotania jako sygnału.
    CHWILOWE   losowane co decyzję — szum pomiarowy (`gauss`, `urand`).

Ten podział to operacjonalizacja rozdziału aleatoric/epistemic (bramka 2.0 b),
a nie porządkowanie kodu.

TRZY POZIOMY OBSERWOWALNOŚCI — oś, wokół której układa się Faza 3:

    obserwowalne     `valid=False` czytane wprost przez bramkę → rule-gate
                     wystarcza i UQ jest zbędne. To BASELINE ablacji (d), nie CORE.
    wnioskowalne     `mask` + podstawienie: dziury nie widać wprost, ale zostaje
                     ślad — przy `zero` wyraźny (dokładne 0 przez wiele decyzji),
                     przy `hold` praktycznie żaden.
    nieobserwowalne  `drift`: wartości sensowne, żadnych dziur ani skoków
                     wariancji. Rule-gate jest tu ślepy z definicji i tylko tu
                     UQ zarabia na siebie.

PODSTAWIENIE NIE JEST NEUTRALNE. Martwy pas raportujący 0 nie znaczy „nie wiem",
tylko „pusty wlot" — dla EP/MaxPressure to konkretna instrukcja „nie dawaj tu
zielonego". `zero` wstrzykuje więc systematyczne kłamstwo o ZNANYM KIERUNKU;
zostaje domyślne, bo taki protokół maskowania przejmujemy z [[li-2025-robustlight]]
i tylko tak krzywe degradacji są porównywalne z ich liczbami. `hold` jest
wariantem TRUDNIEJSZYM (i bliższym realnemu sterownikowi): nie zostawia śladu,
na którym rule-gate mógłby wygrać trywialnie.

FLAGA BRAKU NIE WCHODZI DO WEKTORA STANU. Rule-gate to BRAMKA, nie wejście
polityki — czyta `broken` / `valid` obok obserwacji. Dzięki temu `ob_length`
zostaje identyczne we wszystkich scenariuszach awarii i polityka z Fazy 1 pasuje
do nich bez zmiany architektury.

KALIBRACJA POZIOMÓW — PUŁAPKA PORÓWNYWALNOŚCI. RobustLight ma stan EP + running
(czyli nasz ATS) i podaje skalę 3,5–4,0 dla `gauss`/`urand` oraz maski 25/50%,
ale psuje na poziomie STANU, a my na poziomie DETEKTORA. To nie jest ta sama
liczba: EP to różnica ŚREDNICH po M pasach, więc sigma wstrzyknięta per pas
wychodzi na cesze osłabiona o ~1/sqrt(M), a różnica wjazd−wyjazd wariancje
dodaje. Przelicznik trzeba ZMIERZYĆ i podać w pracy (`SensorModel` nie zgaduje).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from tsc_marl.domain.ats import in_road_lanes, road_of

if TYPE_CHECKING:
    from world.world_cityflow import Intersection

FAILURE_KINDS = ("none", "gauss", "urand", "mask", "drift")
FAILURE_UNITS = ("lane", "road", "intersection")
SUBSTITUTIONS = ("zero", "hold", "mean")

#: Kanały odczytu. Jeden FIZYCZNY detektor obsługuje oba — pas z martwym
#: czujnikiem nie raportuje ani kolejki, ani pojazdów jadących.
CHANNELS = ("queue", "running")


@dataclass(frozen=True)
class FailureSpec:
    """Scenariusz awarii. Zamrożony w YAML-u (sekcja `model`), NIE w Pythonie:
    `experiment.build_params` liczy odcisk sekcji `model`+`trainer`, więc spec
    żyjący tylko w kodzie nie zmieniłby odcisku i `skip-if-done` uznałby bieg
    z innym poziomem szumu za już policzony. Ten sam błąd już raz kosztował
    (patrz `run_core.py` i hiperparametry PPO)."""

    kind: str = "none"
    level: float = 0.0
    #: Co jest JEDNĄ awarią: pojedynczy detektor, cała kamera na wlocie, czy całe
    #: skrzyżowanie bez sensorów (scenariusz kriging z 3.2). Dotyczy uszkodzeń
    #: TRWAŁYCH (`mask`, `drift`) — szum pomiarowy jest zawsze per odczyt.
    unit: str = "lane"
    substitution: str = "zero"
    #: Od której DECYZJI w epizodzie awaria jest aktywna. 0 = od początku (tak
    #: maskuje [[li-2025-robustlight]], więc to zostaje domyślne i porównywalne).
    #: Wartość > 0 daje awarię W RUNTIME — narrację deploymentową z planu Fazy 3
    #: — i jest WARUNKIEM sensowności `substitution='hold'`: detektor martwy od
    #: pierwszej decyzji nie ma czego trzymać, więc `hold` degeneruje się wtedy
    #: do `zero`. Powrót sensora (metryka time-to-recover) to osobne rozszerzenie.
    onset: int = 0
    #: Strumień NIEZALEŻNY od `--seed`. Powód: `--seed` steruje tylko
    #: random/numpy/torch, a ruch jest ustalony w flow.json — gdyby wzorzec
    #: awarii szedł z tego samego ziarna, seedy różniłyby się jednocześnie
    #: inicjalizacją sieci I tym, które detektory padły, i w tabeli T2 nie dałoby
    #: się rozdzielić źródeł rozrzutu.
    seed: int = 0

    def __post_init__(self) -> None:
        if self.kind not in FAILURE_KINDS:
            raise ValueError(f"kind={self.kind!r}, dozwolone: {FAILURE_KINDS}")
        if self.unit not in FAILURE_UNITS:
            raise ValueError(f"unit={self.unit!r}, dozwolone: {FAILURE_UNITS}")
        if self.substitution not in SUBSTITUTIONS:
            raise ValueError(f"substitution={self.substitution!r}, dozwolone: {SUBSTITUTIONS}")
        if self.level < 0:
            raise ValueError(f"level={self.level}, musi być >= 0")
        if self.onset < 0:
            raise ValueError(f"onset={self.onset}, musi być >= 0")
        if self.kind == "mask" and self.level > 1:
            raise ValueError(f"level={self.level} dla 'mask' to UŁAMEK jednostek, więc <= 1")
        # Fail loud zamiast cichego ignorowania: `unit` nie ma sensu dla szumu
        # pomiarowego, a literówka w YAML-u ma się wywalić przy budowie, nie po
        # 200 epizodach dziwnych wyników.
        if self.kind in ("gauss", "urand") and self.unit != "lane":
            raise ValueError(f"kind={self.kind!r} jest per ODCZYT — unit musi zostać 'lane'")

    @property
    def active(self) -> bool:
        return self.kind != "none" and self.level > 0

    @classmethod
    def from_registry(cls, **overrides: Any) -> FailureSpec:
        """Wariant konfigurowany jak reszta repo — przez YAML-e LibSignala
        (`model.failure`, żeby wpadł do odcisku configu)."""
        from common.registry import Registry  # noqa: PLC0415 - dopiero po bootstrapie

        param = Registry.mapping["model_mapping"]["setting"].param
        raw = dict(param.get("failure") or {})
        raw.update(overrides)
        return cls(**raw)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "level": self.level,
            "unit": self.unit,
            "substitution": self.substitution,
            "onset": self.onset,
            "seed": self.seed,
        }

    def suffix(self) -> str:
        """Człon `run_id` — bez tego dwa scenariusze awarii nadpisałyby się
        nawzajem w `runs/` (ten sam agent, sieć, seed i yellow)."""
        if not self.active:
            return ""
        # kropka wypada: ten sufiks trafia do `run_id` i do nazw kluczy w npz,
        # a tam kropka rozdziela rozszerzenie
        base = f"{self.kind}{self.level:g}".replace(".", "p")
        parts = [base] if self.unit == "lane" else [base, self.unit]
        if self.substitution != "zero":
            parts.append(self.substitution)
        if self.onset:
            parts.append(f"t{self.onset}")
        return "_".join(parts)


def failure_units(
    intersections: Sequence[Intersection], lanes: Iterable[str], unit: str
) -> dict[str, str]:
    """pas -> klucz jednostki awarii.

    Dla `intersection` właścicielem detektora jest skrzyżowanie, DO KTÓREGO pas
    dojeżdża (pas wjazdowy) — bo tam stoi kamera. Pas wyjazdowy jednego
    skrzyżowania jest pasem wjazdowym następnego, więc przypisanie jest spójne
    globalnie; pasy wyjeżdżające poza sieć (sąsiad wirtualny, poza listą) nie
    mają właściciela i stanowią własną, jednoelementową jednostkę.
    """
    if unit not in FAILURE_UNITS:
        raise ValueError(f"unit={unit!r}, dozwolone: {FAILURE_UNITS}")
    lanes = list(lanes)
    if unit == "lane":
        return {lane: lane for lane in lanes}
    if unit == "road":
        return {lane: road_of(lane) for lane in lanes}

    owner: dict[str, str] = {}
    for inter in intersections:
        for lane in in_road_lanes(inter):
            owner[lane] = inter.id
    return {lane: owner.get(lane, lane) for lane in lanes}


class SensorModel:
    """Stan detektorów CAŁEJ sieci, kluczowany id pasa.

    Jedna instancja na `StateBuilder`, współdzielona przez wszystkie
    skrzyżowania. To jest istotne, a nie wygodne: pas wyjazdowy skrzyżowania A
    jest pasem wjazdowym skrzyżowania B, więc ten sam fizyczny detektor jest
    czytany dwa razy w jednej decyzji. Gdyby stan (albo szum) był losowany
    osobno dla każdego czytelnika, jeden zepsuty czujnik pokazywałby dwóm
    sąsiadom dwie różne wartości — sieć uśredniłaby szum między sąsiadami
    i awaria wyszłaby ŁAGODNIEJSZA, niż jest naprawdę.
    """

    def __init__(self, spec: FailureSpec, groups: Mapping[str, str]) -> None:
        self.spec = spec
        self.groups = dict(groups)
        self._lanes = sorted(self.groups)
        self._broken: frozenset[str] = frozenset()
        self._gain: dict[str, float] = {}
        self._last: dict[tuple[str, str], float] = {}
        self._cache: dict[tuple[str, str], tuple[float, bool]] = {}
        self._noise = np.random.default_rng([spec.seed, 0, 1])
        self.episode = -1
        self._decision = -1
        self.reset(0)

    # --------------------------------------------------------------- epizod

    def reset(self, episode: int = 0) -> None:
        """Nowy epizod: przelosuj stan TRWAŁY, wyczyść pamięć podręczną.

        `hold` też się czyści — po `world.reset()` ostatnia znana wartość
        z poprzedniego epizodu opisuje inny ruch i byłaby przeciekiem.
        """
        self.episode = episode
        self._decision = -1
        self._cache.clear()
        self._last.clear()
        persistent = np.random.default_rng([self.spec.seed, episode, 0])
        self._noise = np.random.default_rng([self.spec.seed, episode, 1])
        self._broken = frozenset()
        self._gain = {}

        if not self.spec.active:
            return
        keys = sorted(set(self.groups.values()))
        if self.spec.kind == "mask":
            # UŁAMEK DOKŁADNY, nie prawdopodobieństwo per jednostka: protokół
            # mówi "maski 25/50%", więc 25% ma znaczyć 25%, a nie "średnio 25%".
            n = int(round(self.spec.level * len(keys)))
            chosen = {keys[i] for i in persistent.permutation(len(keys))[:n]}
            self._broken = frozenset(l for l in self._lanes if self.groups[l] in chosen)
        elif self.spec.kind == "drift":
            # Błąd kalibracji jest MULTIPLIKATYWNY (detektor liczy o x% za mało),
            # nie addytywny: tylko wtedy wartości zostają sensowne w całym
            # zakresie obciążenia i scenariusz jest naprawdę nieobserwowalny.
            gains = 1.0 + persistent.normal(0.0, self.spec.level, size=len(keys))
            per_key = dict(zip(keys, np.clip(gains, 0.0, None), strict=True))
            self._gain = {l: float(per_key[self.groups[l]]) for l in self._lanes}

    def begin_decision(self) -> None:
        """Nowy odczyt wszystkich detektorów. Woła to `StateBuilder.observe()`
        RAZ na decyzję, przed pętlą po skrzyżowaniach — pamięć podręczna
        gwarantuje, że jeden detektor daje w tej decyzji jedną wartość.

        Liczy też decyzje, bo od nich zależy `onset`. Odczyt `corrupt=False`
        licznika NIE rusza: obserwacja czysta jest pomiarem referencyjnym,
        nie decyzją sterownika."""
        self._decision += 1
        self._cache.clear()

    @property
    def live(self) -> bool:
        """Czy awaria już trwa. Przed pierwszym `begin_decision()` — nie: żadna
        decyzja się jeszcze nie zaczęła, więc detektory są zdrowe."""
        return self._decision >= self.spec.onset  # onset >= 0, więc -1 nigdy nie przechodzi

    # --------------------------------------------------------------- odczyt

    def read(
        self, values: Mapping[str, float], lanes: Sequence[str], *, channel: str
    ) -> dict[str, float]:
        """Odczyt detektorów `lanes` w kanale `channel`. Zwraca NOWY słownik —
        `values` bywa cachem świata (`lane_waiting_count`), który czyta też
        `rewards()`, więc modyfikacja w miejscu skaziłaby nagrodę."""
        if channel not in CHANNELS:
            raise ValueError(f"channel={channel!r}, dozwolone: {CHANNELS}")
        clean = {lane: float(values.get(lane, 0.0)) for lane in lanes}
        if not self.spec.active:
            return clean

        reported: dict[str, float] = {}
        valid: dict[str, bool] = {}
        for lane in lanes:
            hit = self._cache.get((channel, lane))
            if hit is None:
                hit = self._measure(lane, clean[lane], channel)
                self._cache[(channel, lane)] = hit
            reported[lane], valid[lane] = hit

        return self._substitute(reported, valid, channel)

    def _measure(self, lane: str, clean: float, channel: str) -> tuple[float, bool]:
        if not self.live:
            # Przed onsetem detektor jest zdrowy — i to jest jedyny moment,
            # w którym `hold` ma co zapamiętać.
            self._last[(channel, lane)] = clean
            return clean, True
        if lane in self._broken:
            return 0.0, False  # wartość podstawi `_substitute`
        value = clean * self._gain.get(lane, 1.0)
        if self.spec.kind == "gauss":
            value += float(self._noise.normal(0.0, self.spec.level))
        elif self.spec.kind == "urand":
            value += float(self._noise.uniform(-self.spec.level, self.spec.level))
        # Zliczenie nie bywa ujemne. NIE zaokrąglamy: EP to i tak różnica
        # średnich (ułamkowa), a kwantyzacja dołożyłaby drugi, niemodelowany
        # mechanizm i zjadałaby małe sigma.
        value = max(0.0, value)
        self._last[(channel, lane)] = value
        return value, True

    def _substitute(
        self, reported: dict[str, float], valid: dict[str, bool], channel: str
    ) -> dict[str, float]:
        broken = [lane for lane, ok in valid.items() if not ok]
        if not broken:
            return reported
        if self.spec.substitution == "mean":
            live = [reported[lane] for lane, ok in valid.items() if ok]
            fill = float(np.mean(live)) if live else 0.0
            for lane in broken:
                reported[lane] = fill
        elif self.spec.substitution == "hold":
            # Detektor martwy od początku epizodu nie ma czego trzymać -> 0.
            for lane in broken:
                reported[lane] = self._last.get((channel, lane), 0.0)
        return reported

    # --------------------------------------------------------- ground truth

    @property
    def broken(self) -> frozenset[str]:
        """Pasy z martwym detektorem w TYM epizodzie. To jest ground truth, na
        którym bramka 2.0 mierzy, czy głowa niepewności lokalizuje awarię —
        bez tego sweep degradacji nie ma osi Y."""
        return self._broken

    @property
    def gain(self) -> Mapping[str, float]:
        """Rozkalibrowanie per pas (1.0 = detektor zdrowy). Ground truth dla
        scenariusza NIEOBSERWOWALNEGO."""
        return dict(self._gain)

    def valid(self, lanes: Iterable[str]) -> dict[str, bool]:
        """Czy detektor raportuje cokolwiek. Czyta to BRAMKA (rule-gate
        z ablacji (d) i poziom (b) fallbacku), nigdy wektor stanu."""
        return {lane: lane not in self._broken for lane in lanes}

    def report(self) -> dict[str, Any]:
        """Do `runlog`: scenariusz + co faktycznie padło. Sam spec nie wystarcza,
        bo ta sama specyfikacja na innej sieci psuje inne pasy."""
        return {
            **self.spec.as_dict(),
            "episode": self.episode,
            "n_units": len(set(self.groups.values())),
            "n_broken": len(self._broken),
            "broken": sorted(self._broken),
        }

    def __repr__(self) -> str:
        s = self.spec
        return (
            f"SensorModel(kind={s.kind!r}, level={s.level:g}, unit={s.unit!r}, "
            f"substitution={s.substitution!r}, onset={s.onset}, "
            f"broken={len(self._broken)}/{len(self._lanes)})"
        )
