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

from tsc_marl import libsignal
from tsc_marl.libsignal import experiment
from tsc_marl.paths import CONFIGS, REPO_ROOT

SUITES = CONFIGS / "suites"

# Dwa tryby raportowania to jedna wartość `yellow_length`, a nie dwa drzewa configów.
YELLOW_BY_MODE = {"realistic": 5, "benchmark": 0}

# `entry` wybiera moduł wejściowy. Bez tego runner umiał odpalać wyłącznie agentów
# LibSignala — rdzeń Fazy 1 (własna pętla PPO) był z niego nieosiągalny.
ENTRIES = {"tsc": "tsc_marl.runner.train", "core": "tsc_marl.runner.train_core"}

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
    # Karta, na której ma liczyć KAŻDY punkt siatki. "-1" = CPU (domyślnie, bo
    # tak wygląda maszyna deweloperska). To pole opisuje MASZYNĘ, nie eksperyment,
    # więc leci flagą do dziecka i ląduje w sekcji `command` configu — jedynej,
    # której `experiment.build_params` NIE bierze do odcisku. Dzięki temu ten sam
    # bieg policzony na CPU i na GPU ma ten sam `params_hash`, czyli skip-if-done
    # widzi go jako jeden eksperyment, a nie dwa.
    ngpu: str = "-1"
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
            "--ngpu", str(self.ngpu),
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
    from tsc_marl.libsignal import session
    from tsc_marl.runner.train import parse_args

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


def run(
    suite: Suite, replace: bool = False, jobs: int = 1, stagger: float | None = None
) -> dict[str, int]:
    """Policz wszystkie punkty suite'a. Zwraca licznik statusów.

    `jobs > 1` liczy punkty RÓWNOLEGLE (osobne podprocesy — i tak każdy punkt ma
    własny, bo Registry jest globalne). Biegi i tak żyją w osobnych katalogach,
    więc `runlog` nie ma tu nic do synchronizacji; jedyny współdzielony zasób to
    `configs/sim/*.cfg` — patrz `run_parallel`.
    """
    libsignal.bootstrap()  # potrzebne do czytania configów przy planowaniu siatki
    exp = experiment.experiment()
    licznik = {"ok": 0, "pominięte": 0, "failed": 0, "timeout": 0}

    print(f"suite '{suite.name}' → {suite.module}, yellow={suite.yellow}, {suite.path}")
    todo: list[Point] = []
    for point in suite.points():
        _, params, rid = plan(suite, point)
        # skip-if-done po ODCISKU PARAMETRÓW, nie po nazwie: zmiana czegokolwiek
        # w configu metody (albo w `overrides` suite'a) unieważnia poprzedni wynik
        if not replace and exp.is_done(rid, params):
            print(f"  {point.label()}  pomijam (jest wynik)")
            licznik["pominięte"] += 1
            continue
        todo.append(point)

    if jobs <= 1:
        for point in todo:
            print(f"  {point.label()} ...", end=" ", flush=True)
            status, dt = run_one(suite, point, replace)
            print(f"{status} ({dt:.0f}s)")
            licznik[status] = licznik.get(status, 0) + 1
    else:
        gap = STAGGER_S if stagger is None else stagger
        for status in run_parallel(suite, todo, replace, jobs, gap).values():
            licznik[status] = licznik.get(status, 0) + 1

    print(f"\n{licznik}. Biegi: {exp.path}")
    return licznik


# ------------------------------------------------------------ równolegle

#: Odstęp między STARTAMI podprocesów. Nie kosmetyka: `World_param_Interface`
#: LibSignala przepisuje `configs/sim/<sieć>.cfg` przez `open(path, 'w')`
#: (obcina plik, potem pisze), a CityFlow czyta go przy budowie świata. Dwa
#: procesy startujące w tej samej sekundzie potrafią więc przeczytać plik
#: ucięty w połowie — `JSONDecodeError` albo pad silnika. Świat powstaje w
#: pierwszych ~10 s życia procesu, później pliku nikt już nie czyta.
STAGGER_S = 15.0
#: Pad szybszy niż to jest traktowany jako pad STARTU (patrz wyżej) i ponawiany
#: raz. Pad po kilku minutach to prawdziwy błąd biegu — jego nie powtarzamy.
EARLY_FAIL_S = 90.0


def _child_threads(jobs: int) -> str:
    """Wątki BLAS/torch na dziecko. Domyślnie torch bierze WSZYSTKIE rdzenie, więc
    N równoległych biegów to N×rdzenie wątków walczących o te same rdzenie —
    przełączanie kontekstów zjada więcej, niż daje równoległość."""
    import os  # noqa: PLC0415

    return str(max(1, (os.cpu_count() or 1) // jobs))


def logs_dir(suite: Suite) -> Path:
    """stdout/stderr podprocesów. Do PLIKU, nie do potoku: przy kilku dzieciach
    naraz nikt nie czyta potoków na bieżąco, a pełny bufor potoku (64 KB)
    zatrzymuje proces piszący — trening wisiałby bez żadnego komunikatu."""
    from tsc_marl.paths import RUNS  # noqa: PLC0415

    d = RUNS / "_suite_logs" / suite.name
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_parallel(
    suite: Suite,
    points: list[Point],
    replace: bool,
    jobs: int,
    stagger: float = STAGGER_S,
) -> dict[Point, str]:
    """Licz `points` w `jobs` podprocesach naraz. -> {punkt: status}."""
    env = libsignal.child_env()
    env.setdefault("OMP_NUM_THREADS", _child_threads(jobs))
    env.setdefault("MKL_NUM_THREADS", env["OMP_NUM_THREADS"])
    logs = logs_dir(suite)

    def launch(point: Point):
        cmd = [sys.executable, "-m", suite.module, *suite.argv(point, replace)]
        name = f"{point.agent}__{point.network}__s{point.seed}.log"
        log = open(logs / name, "a", encoding="utf-8")  # noqa: SIM115 - zamyka schedule
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        return proc, log

    print(f"  {len(points)} punktów, {jobs} naraz, logi: {logs}")
    # Migawka configów symulatora dla CAŁEJ siatki: dziecko przywraca to, co
    # zastało na starcie, a przy równoległych biegach mogło zastać plik już
    # zabrudzony przez sąsiada. Ostatnie słowo należy więc do rodzica.
    with libsignal.sim_configs_untouched():
        return schedule(points, launch, jobs, suite.timeout, stagger)


def _label(point) -> str:
    return point.label() if hasattr(point, "label") else str(point)


def schedule(points, launch, jobs: int, timeout: float, stagger: float) -> dict:
    """Kolejka z `jobs` slotami. Rozdzielona od `run_parallel`, żeby dało się ją
    przetestować na zwykłych procesach `sleep`, bez stawiania CityFlow.

    `launch(point) -> (Popen, plik_logu)`. Zwraca {punkt: 'ok'|'failed'|'timeout'}.
    """
    pending = list(points)
    retried: set = set()
    active: dict = {}  # punkt -> (proc, log, t0)
    done: dict = {}
    last_start = -float("inf")

    while pending or active:
        # start nowych — najwyżej jeden na `stagger` sekund
        while pending and len(active) < jobs and time.time() - last_start >= stagger:
            point = pending.pop(0)
            proc, log = launch(point)
            active[point] = (proc, log, time.time())
            last_start = time.time()
            print(f"  → start  {_label(point)}", flush=True)

        for point, (proc, log, t0) in list(active.items()):
            dt = time.time() - t0
            code = proc.poll()
            if code is None and dt > timeout:
                proc.kill()
                proc.wait()
                code = "timeout"
            if code is None:
                continue
            log.close()
            del active[point]
            if code == 0:
                done[point] = "ok"
            elif code != "timeout" and dt < EARLY_FAIL_S and point not in retried:
                retried.add(point)
                pending.insert(0, point)  # najpierw ponów to, co padło przy starcie
                print(f"  ! pad po {dt:.0f}s (start?) — ponawiam {_label(point)}", flush=True)
                continue
            else:
                done[point] = "timeout" if code == "timeout" else "failed"
            print(f"  ✓ {done[point]:7s} {_label(point)} ({dt:.0f}s)", flush=True)
        time.sleep(0.2)
    return done


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
