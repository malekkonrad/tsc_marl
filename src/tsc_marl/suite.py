"""Deklaratywne siatki biegów — „co policzyć" w pliku YAML, nie w historii powłoki.

Suite opisuje INTENCJĘ eksperymentu: które punkty {agent × sieć × seed}, jakim
wejściem i z jakimi nadpisaniami configu. Uruchomienie jest wtedy powtarzalne
i widoczne w gicie, a nie odtwarzalne z pamięci.

Każdy punkt leci w OSOBNYM podprocesie, bo LibSignal trzyma konfigurację
w globalnym `Registry` i nie da się go czysto zresetować między punktami.
Stąd jedyny nieoczywisty element całego pliku — sposób przekazania nadpisań
przez granicę procesu (patrz `argv`).

    tsc suite run p1_gate
    tsc suite status p1_gate
"""

from __future__ import annotations

import itertools
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tsc_marl import experiment, libsignal
from tsc_marl.libsignal import CONFIGS, REPO_ROOT

SUITES = CONFIGS / "suites"

# Dwa tryby raportowania to jedna wartość `yellow_length`, a nie dwa drzewa configów.
YELLOW_BY_MODE = {"realistic": 5, "benchmark": 0}

# `entry` wybiera moduł wejściowy. Bez tego runner umiał odpalać wyłącznie agentów
# LibSignala — rdzeń Fazy 1 (własna pętla PPO) był z niego nieosiągalny.
ENTRIES = {"tsc": "tsc_marl.run", "core": "tsc_marl.run_core"}

# Pola, po których wolno rozpinać siatkę. `world` też, bo Faza 5 (SUMO) będzie
# chciała ten sam suite na dwóch symulatorach.
GRID_FIELDS = ("agent", "world", "network", "seed")


@dataclass(frozen=True)
class Point:
    """Jeden punkt siatki = jeden bieg = jeden podproces."""

    agent: str
    world: str
    network: str
    seed: int

    def label(self) -> str:
        return f"{self.network:26s} {self.agent:14s} seed={self.seed}"


@dataclass
class Suite:
    name: str
    path: Path
    entry: str = "tsc"
    mode: str = "realistic"
    timeout: int = 3600
    grid: dict[str, list[Any]] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def yellow(self) -> int:
        return YELLOW_BY_MODE[self.mode]

    @property
    def module(self) -> str:
        return ENTRIES[self.entry]

    @property
    def suffix(self) -> str:
        """Sufiks `run_id` dla suite'ów, które nadpisują config.

        Bez niego dwa suite'y z różnymi `overrides` na tym samym punkcie dałyby ten
        sam `run_id` przy innym `config_hash` — czyli `RunIdCollision` przy każdym
        uruchomieniu. Suite BEZ nadpisań sufiksu nie dostaje: liczy dokładnie to,
        co policzyłbyś ręcznie, więc ma trafiać w te same identyfikatory i widzieć
        wcześniejsze wyniki jako gotowe.
        """
        return self.name if self.overrides else ""

    def points(self) -> list[Point]:
        """Iloczyn kartezjański `grid`; pola spoza siatki biorą wartość z `defaults`."""
        axes = {}
        for f in GRID_FIELDS:
            values = self.grid.get(f, self.defaults.get(f))
            if values is None:
                raise SystemExit(f"suite '{self.name}': brak '{f}' w `grid` ani w `defaults`")
            axes[f] = values if isinstance(values, list) else [values]
        return [
            Point(**dict(zip(GRID_FIELDS, combo, strict=True)))
            for combo in itertools.product(*(axes[f] for f in GRID_FIELDS))
        ]

    def argv(self, point: Point, replace: bool = False) -> list[str]:
        """Argumenty dla podprocesu liczącego ten punkt.

        Flagami jedzie wyłącznie TOŻSAMOŚĆ punktu. Nadpisania jadą jako `--suite
        <ścieżka>` — dziecko czyta ten sam plik samo. Alternatywa (serializacja
        słownika do `--set model.NUM_HEADS='[5,5]'`) wymagałaby własnego parsera
        typów YAML-owych w obie strony i dawałaby komendę, której nie da się
        skopiować z logu i powtórzyć ręcznie.
        """
        argv = [
            "-a", point.agent,
            "-w", point.world,
            "-n", point.network,
            "--seed", str(point.seed),
            "--yellow", str(self.yellow),
            "--suite", str(self.path),
        ]  # fmt: skip
        if replace:
            argv.append("--replace")
        return argv


def load(path: str | Path) -> Suite:
    """Wczytaj suite. Przyjmuje ścieżkę albo samą nazwę (`p1_gate`)."""
    import yaml

    p = Path(path)
    if not p.suffix:
        p = SUITES / f"{p.name}.yml"
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.is_file():
        dostepne = ", ".join(sorted(s.stem for s in SUITES.glob("*.yml"))) or "brak"
        raise SystemExit(f"nie ma suite'a '{path}' ({p}). Dostępne: {dostepne}")

    raw = yaml.safe_load(p.read_text()) or {}
    # `path` ustawiamy sami — w YAML-u byłoby duplikatem argumentu konstruktora
    dozwolone = set(Suite.__dataclass_fields__) - {"path"}
    nieznane = set(raw) - dozwolone
    if nieznane:
        raise SystemExit(f"suite '{p.name}': nieznane pola {sorted(nieznane)}")
    raw.setdefault("name", p.stem)
    raw.setdefault("defaults", {}).setdefault("world", "cityflow")
    if raw.get("entry") == "core":
        raw["defaults"].setdefault("agent", "gatppo")
    if raw.get("mode", "realistic") not in YELLOW_BY_MODE:
        raise SystemExit(f"suite '{p.name}': `mode` musi być jednym z {sorted(YELLOW_BY_MODE)}")
    if raw.get("entry", "tsc") not in ENTRIES:
        raise SystemExit(f"suite '{p.name}': `entry` musi być jednym z {sorted(ENTRIES)}")
    return Suite(path=p, **raw)


# ------------------------------------------------------------------ planowanie


def plan(suite: Suite, point: Point):
    """(args, params, run_id) dla punktu — bez uruchamiania czegokolwiek.

    Idzie DOKŁADNIE tą drogą co proces liczący (`session.resolve`), żeby
    „czy już policzone?" znaczyło to samo po obu stronach granicy procesu.
    """
    from tsc_marl import session
    from tsc_marl.run import parse_args

    args = parse_args(suite.argv(point))
    config, suffix = session.resolve(args)
    params = experiment.build_params(args, config)
    rid = experiment.run_id(
        point.agent, point.network, point.seed, config["trainer"]["yellow_length"], suffix=suffix
    )
    return args, params, rid


# ---------------------------------------------------------------- uruchamianie


def run_one(suite: Suite, point: Point, replace: bool = False) -> tuple[str, float]:
    """Odpala jeden punkt w podprocesie. Zwraca (status, sekundy)."""
    cmd = [sys.executable, "-m", suite.module, *suite.argv(point, replace)]
    t0 = time.time()
    try:
        p = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=libsignal.child_env(),
            timeout=suite.timeout,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return "timeout", time.time() - t0
    if p.returncode != 0:
        tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-3:])
        print(f"    ! bieg nieudany:\n      {tail}", file=sys.stderr)
        return "failed", time.time() - t0
    return "ok", time.time() - t0


def run(suite: Suite, replace: bool = False) -> dict[str, int]:
    """Policz wszystkie punkty suite'a. Zwraca licznik statusów."""
    libsignal.bootstrap()  # potrzebne do czytania configów przy planowaniu siatki
    exp = experiment.experiment()
    licznik = {"ok": 0, "pominięte": 0, "failed": 0, "timeout": 0}

    print(f"suite '{suite.name}' → {suite.module}, yellow={suite.yellow}, {suite.path}")
    for point in suite.points():
        _, params, rid = plan(suite, point)
        # skip-if-done po ODCISKU PARAMETRÓW, nie po nazwie: zmiana czegokolwiek
        # w configu metody (albo w `overrides` suite'a) unieważnia poprzedni wynik
        if not replace and exp.is_done(rid, params):
            print(f"  {point.label()}  pomijam (jest wynik)")
            licznik["pominięte"] += 1
            continue
        print(f"  {point.label()} ...", end=" ", flush=True)
        status, dt = run_one(suite, point, replace)
        print(f"{status} ({dt:.0f}s)")
        licznik[status] = licznik.get(status, 0) + 1

    print(f"\n{licznik}. Biegi: {exp.path}")
    return licznik


def status(suite: Suite) -> list[dict[str, Any]]:
    """Stan każdego punktu siatki: status biegu i jego wynik raportowy.

    Punkt, którego nigdy nie policzono, ma status `brak` — a nie znika z tabeli.
    Suite jest listą TODO, więc pusta pozycja to informacja, nie brak danych.
    """
    libsignal.bootstrap()
    exp = experiment.experiment()
    wiersze = []
    for point in suite.points():
        _, params, rid = plan(suite, point)
        view = next((r for r in exp.runs() if r.id == rid), None)
        wiersze.append(
            {
                "run_id": rid,
                "agent": point.agent,
                "network": point.network,
                "seed": point.seed,
                "status": "brak" if view is None else view.status,
                "aktualny": bool(view and exp.is_done(rid, params)),
                "att_last10": (view.result.get("att_last10") if view else None),
            }
        )
    return wiersze
