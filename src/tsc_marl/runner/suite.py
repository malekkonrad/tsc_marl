"""Declarative run grids (suites): what to compute lives in a YAML file, not in
shell history.

A suite states the intent of an experiment: which points {agent x network x
seed}, which entry point and which config overrides, so a run is repeatable and
visible in git. Every point runs in its OWN subprocess, because LibSignal keeps
configuration in a global Registry; overrides cross the process boundary as a
path to the suite file (see `argv`).

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
from tsc_marl.paths import REPO_ROOT

SUITES = experiment.SUITES

# Two reporting modes are one `yellow_length` value, not two config trees.
YELLOW_BY_MODE = {"realistic": 5, "benchmark": 0}

# `entry` picks the entry module: LibSignal agents or the core (its own PPO loop).
ENTRIES = {"tsc": "tsc_marl.runner.train", "core": "tsc_marl.runner.train_core"}

# Fields a grid may span (`world` too: one suite on both simulators).
GRID_FIELDS = ("agent", "world", "network", "seed")


@dataclass(frozen=True)
class Point:
    """One grid point = one run = one subprocess."""

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
    # GPU for EVERY point, "-1" = CPU (default). It describes the MACHINE, not the
    # experiment, so it travels as a flag into the config's `command` section, the
    # one `experiment.build_params` leaves out of the fingerprint: a point computed
    # on CPU and on GPU is one experiment, not two.
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
        """Run id suffix, see `experiment.suite_suffix`."""
        return experiment.suite_suffix(self.name, self.overrides)

    def points(self) -> list[Point]:
        """Cartesian product of `grid`; fields outside the grid come from `defaults`."""
        axes = {}
        for f in GRID_FIELDS:
            values = self.grid.get(f, self.defaults.get(f))
            if values is None:
                raise SystemExit(f"suite '{self.name}': no '{f}' in `grid` or `defaults`")
            axes[f] = values if isinstance(values, list) else [values]
        return [
            Point(**dict(zip(GRID_FIELDS, combo, strict=True)))
            for combo in itertools.product(*(axes[f] for f in GRID_FIELDS))
        ]

    def argv(self, point: Point, replace: bool = False) -> list[str]:
        """Arguments of the subprocess computing this point.

        Flags carry only the point's IDENTITY; overrides travel as `--suite <path>`
        and the child reads the same file. Serialising a dict into argv would need a
        YAML type parser both ways and give a command one cannot copy from a log.
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
    """Load and validate a suite, given as a path or just a name (`p1_gate`)."""
    import yaml

    p = experiment.suite_path(path)
    raw = yaml.safe_load(p.read_text()) or {}
    # `path` is set here; in YAML it would duplicate a constructor argument
    allowed = set(Suite.__dataclass_fields__) - {"path"}
    unknown = set(raw) - allowed
    if unknown:
        raise SystemExit(f"suite '{p.name}': unknown fields {sorted(unknown)}")
    raw.setdefault("name", p.stem)
    raw.setdefault("defaults", {}).setdefault("world", "cityflow")
    if raw.get("entry") == "core":
        raw["defaults"].setdefault("agent", "gatppo")
    if raw.get("mode", "realistic") not in YELLOW_BY_MODE:
        raise SystemExit(f"suite '{p.name}': `mode` must be one of {sorted(YELLOW_BY_MODE)}")
    if raw.get("entry", "tsc") not in ENTRIES:
        raise SystemExit(f"suite '{p.name}': `entry` must be one of {sorted(ENTRIES)}")
    return Suite(path=p, **raw)


# ------------------------------------------------------------------ planning


def plan(suite: Suite, point: Point):
    """(args, params, run_id) for a point without running anything, along EXACTLY
    the computing process's path (`session.resolve`), so "already computed" means
    the same on both sides of the process boundary."""
    from tsc_marl.libsignal import session
    from tsc_marl.libsignal.experiment import parse_args

    args = parse_args(suite.argv(point))
    config, suffix = session.resolve(args)
    params = experiment.build_params(args, config)
    rid = experiment.run_id(
        point.agent, point.network, point.seed, config["trainer"]["yellow_length"], suffix=suffix
    )
    return args, params, rid


# ---------------------------------------------------------------- running


def run_one(suite: Suite, point: Point, replace: bool = False) -> tuple[str, float]:
    """Run one point in a subprocess. -> (status, seconds)."""
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
        print(f"    ! run failed:\n      {tail}", file=sys.stderr)
        return "failed", time.time() - t0
    return "ok", time.time() - t0


def run(
    suite: Suite, replace: bool = False, jobs: int = 1, stagger: float | None = None
) -> dict[str, int]:
    """Compute every point of a suite. -> status counts.

    `jobs > 1` runs points IN PARALLEL; runs live in separate directories, so
    `runlog` needs no synchronisation; the only shared resource is
    `configs/sim/*.cfg` (see `run_parallel`).
    """
    libsignal.bootstrap()  # needed to read configs while planning
    exp = experiment.experiment()
    counts = {"ok": 0, "skipped": 0, "failed": 0, "timeout": 0}

    print(f"suite '{suite.name}' -> {suite.module}, yellow={suite.yellow}, {suite.path}")
    todo: list[Point] = []
    for point in suite.points():
        _, params, rid = plan(suite, point)
        # skip-if-done by PARAMS FINGERPRINT, not by name: any change to the method
        # config (or the suite's `overrides`) invalidates the previous result
        if not replace and exp.is_done(rid, params):
            print(f"  {point.label()}  skipped (result exists)")
            counts["skipped"] += 1
            continue
        todo.append(point)

    if jobs <= 1:
        for point in todo:
            print(f"  {point.label()} ...", end=" ", flush=True)
            status, dt = run_one(suite, point, replace)
            print(f"{status} ({dt:.0f}s)")
            counts[status] = counts.get(status, 0) + 1
    else:
        gap = STAGGER_S if stagger is None else stagger
        for status in run_parallel(suite, todo, replace, jobs, gap).values():
            counts[status] = counts.get(status, 0) + 1

    print(f"\n{counts}. Runs: {exp.path}")
    return counts


# ------------------------------------------------------------ parallel

#: Gap between subprocess STARTS. LibSignal's `World_param_Interface` rewrites
#: `configs/sim/<network>.cfg` with `open(path, 'w')` (truncate, then write) and
#: CityFlow reads it while building the world, so two processes starting together
#: can read a half-written file (`JSONDecodeError` or an engine crash).
STAGGER_S = 15.0
#: A failure faster than this counts as a START failure (see above) and is retried
#: once; a failure after minutes is a real error and is not repeated.
EARLY_FAIL_S = 90.0


def _child_threads(jobs: int) -> str:
    """BLAS/torch threads per child: by default torch takes ALL cores, and N
    parallel runs would fight over them."""
    import os

    return str(max(1, (os.cpu_count() or 1) // jobs))


def logs_dir(suite: Suite) -> Path:
    """Children's stdout/stderr go to a FILE, not a pipe: nobody reads pipes of
    several children live, and a full pipe buffer (64 KB) blocks the writer."""
    from tsc_marl.paths import RUNS

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
    """Compute `points` in `jobs` subprocesses at once. -> {point: status}."""
    env = libsignal.child_env()
    env.setdefault("OMP_NUM_THREADS", _child_threads(jobs))
    env.setdefault("MKL_NUM_THREADS", env["OMP_NUM_THREADS"])
    logs = logs_dir(suite)

    def launch(point: Point):
        cmd = [sys.executable, "-m", suite.module, *suite.argv(point, replace)]
        name = f"{point.agent}__{point.network}__s{point.seed}.log"
        log = open(logs / name, "a", encoding="utf-8")  # noqa: SIM115 - closed by schedule
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        return proc, log

    print(f"  {len(points)} points, {jobs} at once, logs: {logs}")
    # A snapshot of the simulator configs for the WHOLE grid: a child restores what
    # it found at start, which may already be dirtied by a sibling, so the parent
    # has the last word.
    with libsignal.sim_configs_untouched():
        return schedule(points, launch, jobs, suite.timeout, stagger)


def _label(point) -> str:
    return point.label() if hasattr(point, "label") else str(point)


def schedule(points, launch, jobs: int, timeout: float, stagger: float) -> dict:
    """A queue with `jobs` slots, separate from `run_parallel` so it can be tested
    with plain `sleep` processes. `launch(point) -> (Popen, log_file)`.
    -> {point: 'ok'|'failed'|'timeout'}.
    """
    pending = list(points)
    retried: set = set()
    active: dict = {}  # point -> (proc, log, t0)
    done: dict = {}
    last_start = -float("inf")

    while pending or active:
        # start new ones, at most one per `stagger` seconds
        while pending and len(active) < jobs and time.time() - last_start >= stagger:
            point = pending.pop(0)
            proc, log = launch(point)
            active[point] = (proc, log, time.time())
            last_start = time.time()
            print(f"  -> start  {_label(point)}", flush=True)

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
                pending.insert(0, point)  # retry start failures first
                print(f"  ! failed after {dt:.0f}s (start?), retrying {_label(point)}", flush=True)
                continue
            else:
                done[point] = "timeout" if code == "timeout" else "failed"
            print(f"  - {done[point]:7s} {_label(point)} ({dt:.0f}s)", flush=True)
        time.sleep(0.2)
    return done


def status(suite: Suite) -> list[dict[str, Any]]:
    """State of every grid point: run status and report result. A point never
    computed has status `missing` instead of vanishing from the table."""
    libsignal.bootstrap()
    exp = experiment.experiment()
    rows = []
    for point in suite.points():
        _, params, rid = plan(suite, point)
        view = next((r for r in exp.runs() if r.id == rid), None)
        rows.append(
            {
                "run_id": rid,
                "agent": point.agent,
                "network": point.network,
                "seed": point.seed,
                "status": "missing" if view is None else view.status,
                "current": bool(view and exp.is_done(rid, params)),
                "att_last10": (view.result.get("att_last10") if view else None),
            }
        )
    return rows
