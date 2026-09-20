"""Ewaluacja ZAMROŻONEGO sterownika na liście scenariuszy awarii (T2 bez mechanizmu).

To jest oś X głównej figury pracy i test wykonalności fallbacku: zanim zbudujemy
mechanizm niepewności, trzeba wiedzieć, JAK BARDZO polityka z Fazy 1 degraduje się
sama z siebie. Fallback do FixedTime opłaca się tylko tam, gdzie polityka pod
awarią spada poniżej FixedTime — a tego nikt w tym repo jeszcze nie zmierzył.

PLAN (`configs/robustness/<nazwa>.yml`) opisuje: sieci, sterowniki (z seedami dla
uczonych), scenariusze awarii i liczbę LOSOWAŃ na scenariusz. Replikacją pod awarią
jest N LOSOWAŃ AWARII (`episode` → `SensorModel.reset`), nie N seedów symulatora —
ruch w flow.json jest ustalony (patrz `domain/sensors.py`).

Jeden bieg `runlog` = jeden sterownik × sieć × seed, WSZYSTKIE scenariusze w środku
(`metrics.csv`: split = scenariusz, step = losowanie). Jeden świat na proces,
nowy `StateBuilder` na scenariusz — trajektoria różni się między scenariuszami
wyłącznie tym, co zrobił sterownik.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tsc_marl.domain.sensors import FailureSpec
from tsc_marl.paths import CONFIGS

PLANS = CONFIGS / "robustness"
EXPERIMENT = "t2"
YELLOW_BY_MODE = {"realistic": 5, "benchmark": 0}

#: agent -> rodzaj sterownika. Uczone mają rodzica w `runs/t1` (checkpoint).
KINDS = {
    "gatppo": "policy",
    "advancedcolight": "colight",
    "efficientmp": "pressure",
    "advancedmp": "pressure",
    "fixedtime": "fixed",
}
#: `ensemble` = kilka actorów rdzenia naraz: decyduje pierwszy, reszta głosuje — rozbieżność
#: ich rozkładów to epistemic W SAMEJ POLITYCE RL (test gałęzi epistemic, plan t2_ood).
#: `policy_uw` = TEN SAM checkpoint co `policy`, tylko attention dostaje log ρ z głowy
#: obserwacyjnej (Faza 3). Modulacja jest bezparametrowa, więc kolumny „rdzeń" i
#: „rdzeń + UW-attention" różnią się wyłącznie mechanizmem — ten sam plik z wagami.
#: `policy_gate` = TEN SAM checkpoint co `policy`, ale akcja każdego węzła idzie
#: z polityki ALBO z reguły, zależnie od bramki (Faza 3, dolny poziom fallbacku).
LEARNED = ("policy", "colight", "ensemble", "policy_uw", "policy_gate")
#: Rodzaje bramek. `oracle` czyta prawdę z warstwy awarii i jest GÓRNYM
#: OGRANICZENIEM, nie sterownikiem do wdrożenia. `rule` (licznik zamrożonych
#: detektorów) i `uq` (głowa obserwacyjna) są wdrażalne i mierzy się je WZGLĘDEM
#: wyroczni — ta ablacja rozstrzyga, czy głowa niepewności zarabia na siebie.
GATE_KINDS = ("oracle", "rule", "uq")
#: Klucze spec sterownika, które zmieniają MECHANIZM, a nie tylko etykietę kolumny:
#: `uw` niesie τ modulacji, `gate` — rodzaj bramki, próg i długość cyklu reguły.
#: MUSZĄ wejść do odcisku biegu (`unit_params`), inaczej sweep po τ albo po progu
#: liczy pierwszą wartość, a każdą następną pomija jako „jest wynik" — ten sam błąd,
#: przed którym ostrzega docstring `FailureSpec`.
MECHANISM_KEYS = ("uw", "gate")
#: Sufiks wariantu ruchu w nazwie sieci (`_d0p25`, `_d1p5`, `_r2000`, `_r5816`). `{base}`
#: w szablonie rodzica = sieć bez sufiksu, czyli sieć, NA KTÓREJ polityka się uczyła.
VARIANT_SUFFIX = re.compile(r"_(?:d\d+(?:p\d+)?|r\d+)$")


def base_network(network: str) -> str:
    return VARIANT_SUFFIX.sub("", network)


#: Kolumny zapisywane do `metrics.csv` każdego biegu T2. `queue_max`/`queue_p90`/
#: `wait_max` to OGON (Faza 4): średnia sieciowa nie widzi jednego skrzyżowania
#: trzymanego w martwej fazie ani ćwiartki sieci zepsutej przy zdrowej reszcie.
#: Dopisane PRZED biegami maskingu i Krakowa celowo — metryki liczą się w epizodzie,
#: więc dodanie ich później oznacza przeliczenie wszystkiego od nowa.
METRICS = (
    "att",
    "throughput",
    "queue",
    "delay",
    "reward",
    "queue_max",
    "queue_p90",
    "wait_max",
    "wait_max_demand",
)


def slug(spec: FailureSpec) -> str:
    return spec.suffix() or "clean"


@dataclass(frozen=True)
class Unit:
    """Jeden punkt planu = jeden podproces = jeden bieg `runlog`."""

    controller: str  # etykieta z planu (klucz w `controllers`)
    network: str
    seed: int

    def label(self) -> str:
        return f"{self.network:26s} {self.controller:18s} seed={self.seed}"


@dataclass
class Plan:
    name: str
    path: Path
    networks: list[str]
    controllers: dict[str, dict[str, Any]]
    scenarios: tuple[FailureSpec, ...]
    mode: str = "realistic"
    draws: int = 5
    timeout: int = 7200
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def yellow(self) -> int:
        return YELLOW_BY_MODE[self.mode]

    def agent(self, controller: str) -> str:
        return self.controllers[controller].get("agent", controller)

    def kind(self, controller: str) -> str:
        spec = self.controllers[controller]
        if "ensemble" in spec:
            return "ensemble"
        if spec.get("uw"):
            return "policy_uw"
        if spec.get("gate"):
            return "policy_gate"
        return KINDS[self.agent(controller)]

    def baseline(self, controller: str) -> str | None:
        """Kolumna, z którą raport ma SPAROWAĆ ten sterownik, gdy domyślny wybór
        („pierwsza kolumna `policy` tego samego agenta") byłby zły.

        Potrzebne od planu `t2_train_uw`: są tam DWA ramiona bez mechanizmu
        (trening na czystym i trening z awariami), oba z agentem `gatppo`, więc
        domyśl trafiłby w pierwsze i mierzył trening razem z mechanizmem zamiast
        samego mechanizmu. Różnica jest cicha — tabela wychodzi, tylko odpowiada
        na inne pytanie."""
        return self.controllers[controller].get("baseline")

    def units(self) -> list[Unit]:
        """Uczone sterowniki × ich seedy; reguły MP i FixedTime są deterministyczne,
        więc liczone raz (seed 0) — N seedów dałoby N identycznych liczb."""
        out = []
        for net in self.networks:
            for label, spec in self.controllers.items():
                kind = self.kind(label)
                if kind == "ensemble":
                    seeds = [spec["ensemble"][0]]  # jeden punkt: seed, który podejmuje decyzje
                elif kind in LEARNED:
                    seeds = spec.get("seeds", [0])
                else:
                    seeds = [0]
                out += [Unit(label, net, int(s)) for s in seeds]
        return out

    def parent_ids(self, unit: Unit) -> list[str]:
        """Biegi treningowe w `runs/t1`, z których bierzemy checkpointy — jeden dla
        polityki, kilka dla ensemble (pierwszy = ten, który decyduje)."""
        kind = self.kind(unit.controller)
        if kind not in LEARNED:
            return []
        spec = self.controllers[unit.controller]
        template = spec.get("parent", "{agent}__{network}__s{seed}_y{yellow}")
        seeds = spec["ensemble"] if kind == "ensemble" else [unit.seed]
        return [
            template.format(
                agent=self.agent(unit.controller),
                network=unit.network,
                base=base_network(unit.network),
                seed=s,
                yellow=self.yellow,
            )
            for s in seeds
        ]

    def parent_id(self, unit: Unit) -> str | None:
        ids = self.parent_ids(unit)
        return ids[0] if ids else None

    def run_id(self, unit: Unit) -> str:
        return f"{unit.controller}__{unit.network}__s{unit.seed}_y{self.yellow}__{self.name}"

    def scenarios_for(self, unit: Unit) -> tuple[FailureSpec, ...]:
        """FixedTime nie czyta odczytów, więc pod awarią ma z definicji wynik czysty."""
        if self.kind(unit.controller) == "fixed":
            return tuple(s for s in self.scenarios if not s.active)
        return self.scenarios


def load_plan(path: str | Path) -> Plan:
    """Nazwa (`t2_nomech`) albo ścieżka. Czysty scenariusz jest dokładany zawsze,
    na początek — bez niego nie ma względem czego liczyć degradacji."""
    import yaml  # noqa: PLC0415

    p = Path(path)
    if not p.suffix:
        p = PLANS / f"{p.name}.yml"
    if not p.is_file():
        dostepne = ", ".join(sorted(x.stem for x in PLANS.glob("*.yml"))) or "brak"
        raise SystemExit(f"nie ma planu '{path}' ({p}). Dostępne: {dostepne}")
    raw = yaml.safe_load(p.read_text()) or {}
    known = {"name", "networks", "controllers", "scenarios", "mode", "draws", "timeout"}
    unknown = set(raw) - known
    if unknown:
        raise SystemExit(f"plan '{p.name}': nieznane pola {sorted(unknown)}")
    specs = [FailureSpec(**s) for s in raw.get("scenarios", [])]
    if not any(not s.active for s in specs):
        specs.insert(0, FailureSpec())
    plan = Plan(
        name=raw.get("name", p.stem),
        path=p.resolve(),
        networks=list(raw["networks"]),
        controllers=dict(raw["controllers"]),
        scenarios=tuple(specs),
        mode=raw.get("mode", "realistic"),
        draws=int(raw.get("draws", 5)),
        timeout=int(raw.get("timeout", 7200)),
    )
    for label, spec in plan.controllers.items():
        if plan.agent(label) not in KINDS:
            raise SystemExit(f"plan '{p.name}': nieznany sterownik '{plan.agent(label)}'")
        if "ensemble" in spec and KINDS[plan.agent(label)] != "policy":
            raise SystemExit(f"plan '{p.name}': ensemble tylko z actorów rdzenia ('{label}')")
        if "ensemble" in spec and len(spec["ensemble"]) < 2:
            raise SystemExit(f"plan '{p.name}': ensemble '{label}' potrzebuje ≥ 2 seedów")
        if spec.get("uw") and KINDS[plan.agent(label)] != "policy":
            raise SystemExit(f"plan '{p.name}': UW-attention tylko z actorem rdzenia ('{label}')")
        if spec.get("gate"):
            if KINDS[plan.agent(label)] != "policy":
                raise SystemExit(f"plan '{p.name}': bramka tylko z actorem rdzenia ('{label}')")
            # Literówka w `kind` nie ma prawa po cichu zbudować wyroczni: to ona jest
            # górnym ograniczeniem, więc pomylona z realną bramką zawyżyłaby wynik.
            if spec["gate"].get("kind") not in GATE_KINDS:
                raise SystemExit(
                    f"plan '{p.name}': bramka '{label}' ma kind="
                    f"{spec['gate'].get('kind')!r}, dozwolone: {GATE_KINDS}"
                )
        baza = spec.get("baseline")
        if baza is not None and baza not in plan.controllers:
            raise SystemExit(
                f"plan '{p.name}': '{label}' wskazuje baseline '{baza}', którego nie ma "
                f"w `controllers` ({sorted(plan.controllers)})"
            )
        if baza is not None and baza == label:
            raise SystemExit(f"plan '{p.name}': '{label}' wskazuje sam siebie jako baseline")
    slugs = [slug(s) for s in plan.scenarios]
    if len(set(slugs)) != len(slugs):
        raise SystemExit(f"plan '{p.name}': zdublowane scenariusze {slugs}")
    if plan.mode not in YELLOW_BY_MODE:
        raise SystemExit(f"plan '{p.name}': mode musi być jednym z {sorted(YELLOW_BY_MODE)}")
    return plan


# ------------------------------------------------------------------- pętla


def evaluate(
    env: Any,
    controller: Any,
    specs: tuple[FailureSpec, ...],
    draws: int,
    make_builder: Callable[[FailureSpec], Any],
    decisions: int,
    on_row: Callable[[dict[str, Any]], None] | None = None,
    on_trace: Callable[[str, int, list[Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Każdy scenariusz × losowanie = jeden pełny epizod testowy. -> wiersze metryk.

    Czysty scenariusz jest deterministyczny (argmax + deterministyczny CityFlow),
    więc liczony RAZ; scenariusz z awarią `draws` razy, każde losowanie z innym
    `episode` — to on wybiera, które detektory padły i jaki szum poleciał.
    """
    rows: list[dict[str, Any]] = []
    for spec in specs:
        env.features = make_builder(spec)
        n = draws if spec.active else 1
        for draw in range(n):
            obs = env.reset(draw)
            if hasattr(controller, "reset_stats"):
                controller.reset_stats()
            for _ in range(decisions):
                obs, _, _ = env.step(controller.act(obs))
            row: dict[str, Any] = {"scenario": slug(spec), "draw": draw, **env.metrics()}
            if hasattr(controller, "episode_stats"):
                row.update(controller.episode_stats())
            sensors = getattr(env.features, "sensors", None)
            row["n_broken"] = len(sensors.broken) if sensors is not None else 0
            rows.append(row)
            if on_row is not None:
                on_row(row)
            # Ślad bramki (opcjonalny) wychodzi PO metrykach i przed `reset_stats`
            # następnego epizodu — czyli w jedynym momencie, w którym jest kompletny.
            if on_trace is not None and hasattr(controller, "episode_trace"):
                slad = controller.episode_trace()
                if slad:
                    on_trace(slug(spec), draw, slad)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Skalary do `result.json`: średnie po losowaniach, `att__<scenariusz>`."""
    out: dict[str, float] = {}
    by: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["scenario"], []).append(r)
    for name, group in by.items():
        for metric in (
            "att", "throughput", "ens_mi", "ens_disagree", "uw_spread", "uw_active",
            "gate_share", "gate_switches", "gate_precision", "gate_recall",
        ):  # fmt: skip
            # Średnia po losowaniach, w których metryka BYŁA OKREŚLONA — sprawdzanie
            # samego `group[0]` zakładało, że wszystkie wiersze mają ten sam komplet
            # kluczy. `gate_precision`/`gate_recall` tego nie spełniają: przy części
            # losowań mianownik nie istnieje (nie było czego wykrywać), więc klucza
            # nie ma. Skutkiem był `KeyError` po POLICZENIU całego punktu — wszystkie
            # epizody przeliczone, wynik wyrzucony w ostatniej linii.
            wartosci = [r[metric] for r in group if metric in r]
            if wartosci:
                out[f"{metric}__{name}"] = sum(wartosci) / len(wartosci)
    if "att__clean" in out:
        out["att"] = out["att__clean"]
    return out
