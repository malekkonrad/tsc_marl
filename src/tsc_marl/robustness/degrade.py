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
LEARNED = ("policy", "colight")
METRICS = ("att", "throughput", "queue", "delay", "reward")


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
        return KINDS[self.agent(controller)]

    def units(self) -> list[Unit]:
        """Uczone sterowniki × ich seedy; reguły MP i FixedTime są deterministyczne,
        więc liczone raz (seed 0) — N seedów dałoby N identycznych liczb."""
        out = []
        for net in self.networks:
            for label, spec in self.controllers.items():
                seeds = spec.get("seeds", [0]) if self.kind(label) in LEARNED else [0]
                out += [Unit(label, net, int(s)) for s in seeds]
        return out

    def parent_id(self, unit: Unit) -> str | None:
        """Bieg treningowy w `runs/t1`, z którego bierzemy checkpoint."""
        if self.kind(unit.controller) not in LEARNED:
            return None
        template = self.controllers[unit.controller].get(
            "parent", "{agent}__{network}__s{seed}_y{yellow}"
        )
        return template.format(
            agent=self.agent(unit.controller),
            network=unit.network,
            seed=unit.seed,
            yellow=self.yellow,
        )

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
    for label in plan.controllers:
        if plan.agent(label) not in KINDS:
            raise SystemExit(f"plan '{p.name}': nieznany sterownik '{plan.agent(label)}'")
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
            for _ in range(decisions):
                obs, _, _ = env.step(controller.act(obs))
            row: dict[str, Any] = {"scenario": slug(spec), "draw": draw, **env.metrics()}
            sensors = getattr(env.features, "sensors", None)
            row["n_broken"] = len(sensors.broken) if sensors is not None else 0
            rows.append(row)
            if on_row is not None:
                on_row(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Skalary do `result.json`: średnie po losowaniach, `att__<scenariusz>`."""
    out: dict[str, float] = {}
    by: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["scenario"], []).append(r)
    for name, group in by.items():
        for metric in ("att", "throughput"):
            out[f"{metric}__{name}"] = sum(r[metric] for r in group) / len(group)
    if "att__clean" in out:
        out["att"] = out["att__clean"]
    return out
