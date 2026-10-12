"""Adapter between LibSignal and `runlog`, the only place that knows both.

`runlog` knows nothing about traffic and LibSignal nothing about `runlog`; this
module translates one into the other: command-line arguments, run ids,
parameters, suite overrides, and parsing engine logs into metrics.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from runlog import Experiment
from tsc_marl.paths import CONFIGS, LIBSIGNAL, REPO_ROOT, RUNS

EXPERIMENT = "t1"
SUITES = CONFIGS / "suites"

# Config sections describing the SCIENTIFIC content of a run. Their fingerprint is
# part of the parameters, so changing `NUM_HEADS` or a learning rate invalidates
# skip-if-done. `command`/`logger` are left out on purpose: paths, prefixes and the
# device differ between runs without changing the experiment.
CONFIG_SECTIONS = ("model", "trainer")

# "Final Travel Time is 316.1114, mean rewards: -447.5272, queue: 10.4976,
#  delay: 2.7385, throughput: 6151"
FINAL_RE = re.compile(
    r"Final Travel Time is ([\d.]+), mean rewards: (-?[\d.]+), queue: ([\d.]+), "
    r"delay: ([\d.]+), throughput: (\d+)"
)
# Columns of `tsc_trainer.writeLog`: name, mode, episode, then the six below
DTL_COLUMNS = ("att", "loss", "reward", "queue", "delay", "throughput")
REPORT_WINDOW = 10  # Advanced-CoLight protocol: mean of the last 10 test episodes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Arguments of a training or evaluation process, shared by every entry point
    so flags mean the same everywhere (all of them write to the same `runlog`)."""
    p = argparse.ArgumentParser(description="Train or evaluate a TSC agent")
    # --- arguments LibSignal's Registry expects (names must match) ---
    p.add_argument("-t", "--task", default="tsc")
    p.add_argument("-a", "--agent", default="efficientmp", help="name in the Registry")
    p.add_argument("-w", "--world", default="cityflow", choices=["cityflow", "sumo"])
    p.add_argument("-n", "--network", default="cityflow_jinan3x4_4ph")
    p.add_argument("-d", "--dataset", default="onfly")
    p.add_argument("--thread_num", type=int, default=4)
    p.add_argument("--ngpu", default="-1", help="-1 = CPU")
    p.add_argument("--prefix", default="test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--debug", type=bool, default=True)
    p.add_argument("--interface", default="libsumo", choices=["libsumo", "traci"])
    p.add_argument("--delay_type", default="apx", choices=["apx", "real"])
    # --- ours ---
    p.add_argument(
        "--yellow",
        type=int,
        default=None,
        metavar="S",
        help="override the protocol's yellow_length: 5 = realistic mode (default, "
        "configs/tsc/thesis_protocol.yml), 0 = benchmark mode as in the papers",
    )
    p.add_argument("--episodes", type=int, default=None, help="override the number of episodes")
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        metavar="S",
        help="override the episode length (training AND test) in simulated seconds; "
        "smoke tests and debugging only, the protocol is 3600",
    )
    p.add_argument(
        "--suite",
        default=None,
        metavar="FILE",
        help="suite file (configs/suites/*.yml or just its name); its `overrides` block "
        "enters the config BEFORE the run fingerprint. This is how the suite runner passes "
        "overrides to a child process: as a path, not as a dict serialised into argv",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="override the run id (default {agent}__{network}__s{seed}_y{yellow})",
    )
    p.add_argument(
        "--replace",
        action="store_true",
        help="recompute a run that already succeeded (an error by default)",
    )
    return p.parse_args(argv)


def experiment(root: Path | str = RUNS) -> Experiment:
    return Experiment(EXPERIMENT, root=root)


def repos() -> dict[str, Path]:
    """Provenance: a result depends on both trees, ours and the engine submodule."""
    return {"tsc_marl": REPO_ROOT, "libsignal": LIBSIGNAL}


def run_id(agent: str, network: str, seed: int, yellow: int, suffix: str = "") -> str:
    """Human-readable, so a checkpoint can be found without opening metadata."""
    base = f"{agent}__{network}__s{seed}_y{yellow}"
    return f"{base}__{suffix}" if suffix else base


# ------------------------------------------------------------------ suites


def suite_path(path: str | Path) -> Path:
    """A suite given as a path or just a name (`p1_gate`) -> its file."""
    p = Path(path)
    if not p.suffix:
        p = SUITES / f"{p.name}.yml"
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.is_file():
        available = ", ".join(sorted(s.stem for s in SUITES.glob("*.yml"))) or "none"
        raise SystemExit(f"no suite '{path}' ({p}). Available: {available}")
    return p


def suite_suffix(name: str, overrides: dict[str, Any]) -> str:
    """Run id suffix of a suite that overrides the config.

    Without it two suites with different `overrides` on the same point would get
    the same run id with another fingerprint, i.e. `RunIdCollision`. A suite WITHOUT
    overrides gets no suffix: it computes exactly what a manual run would, so it
    hits the same run ids and sees earlier results as done.
    """
    return name if overrides else ""


def suite_overrides(path: str | Path) -> tuple[dict[str, Any], str]:
    """(config overrides, run id suffix) of a suite file, as the child process
    reads them; full validation of the suite happens in `runner.suite.load`."""
    import yaml

    p = suite_path(path)
    raw = yaml.safe_load(p.read_text()) or {}
    overrides = raw.get("overrides") or {}
    return overrides, suite_suffix(raw.get("name", p.stem), overrides)


# ------------------------------------------------------------------ configs and params


def resolve_config(args) -> dict[str, Any]:
    """Config after merging `includes` and CLI overrides, i.e. what REALLY reaches
    the simulator. One function for the computing process and the suite planner:
    if they computed parameters separately, skip-if-done would drift silently."""
    from utils.logger import build_config  # after bootstrap

    config, _ = build_config(args)
    if args.yellow is not None:
        config["trainer"]["yellow_length"] = args.yellow
    if args.episodes is not None:
        config["trainer"]["episodes"] = args.episodes
    if getattr(args, "steps", None) is not None:
        # both at once: a short training with a full test (or vice versa) would give
        # numbers comparable to nothing
        config["trainer"]["steps"] = config["trainer"]["test_steps"] = args.steps
    return config


def plan(agent: str, world: str, network: str, seed: int, yellow: int):
    """(args, params, run_id) for a grid point without running anything, so the
    runner checks "already computed" the same way the process will."""
    args = parse_args(
        ["-a", agent, "-w", world, "-n", network, "--seed", str(seed), "--yellow", str(yellow)]
    )
    config = resolve_config(args)
    params = build_params(args, config)
    return args, params, run_id(agent, network, seed, config["trainer"]["yellow_length"])


def build_params(args, config: dict[str, Any]) -> dict[str, Any]:
    """Run parameters = what identifies the experiment. The config fingerprint is a
    single field: cheap to compare and sensitive to EVERY hyperparameter."""
    from runlog import params_hash

    trainer = config.get("trainer", {})
    model = config.get("model", {})
    return {
        "agent": args.agent,
        "world": args.world,
        "network": args.network,
        "seed": args.seed,
        "yellow_length": trainer.get("yellow_length"),
        "episodes": trainer.get("episodes"),
        "action_interval": trainer.get("action_interval"),
        "steps": trainer.get("steps"),
        "test_steps": trainer.get("test_steps"),
        # max-pressure parameters: without them the row is NOT reproducible
        **{k: model.get(k) for k in ("t_min", "w1", "obs_length", "drop_always_green")},
        "config_hash": params_hash({k: config.get(k) for k in CONFIG_SECTIONS}),
    }


# ------------------------------------------------------------------ engine logs


def engine_log_dir(run, args) -> Path:
    """Where LibSignal stores the logs of THIS run (`get_output_file_path` after our
    patch: `root_dir` points to `engine/` inside the run directory)."""
    return (
        run.engine_dir
        / "tsc"
        / f"{args.world}_{args.agent}"
        / args.network
        / args.prefix
        / "logger"
    )


def _newest(logdir: Path, pattern: str) -> Path | None:
    logs = sorted(logdir.glob(pattern), key=lambda p: p.stat().st_mtime) if logdir.is_dir() else []
    return logs[-1] if logs else None


def parse_episodes(logdir: Path) -> list[dict[str, Any]]:
    """`_DTL.log` rows -> per-episode records; methods without training have no
    such file and give an empty list."""
    path = _newest(logdir, "*_DTL.log")
    if path is None:
        return []
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        try:
            values = [float(v) for v in parts[3 : 3 + len(DTL_COLUMNS)]]
        except ValueError:
            continue
        rows.append(
            {
                "split": parts[1].lower(),
                "episode": int(parts[2]),
                **dict(zip(DTL_COLUMNS, values, strict=False)),
            }
        )
    return rows


def parse_final(logdir: Path) -> dict[str, Any]:
    """Last line of `_BRF.log` -> final scalars, read from the FILE (an artefact of
    the run), not from the subprocess's stdout, which may be a truncated buffer."""
    path = _newest(logdir, "*_BRF.log")
    if path is None:
        return {}
    hits = FINAL_RE.findall(path.read_text())
    if not hits:
        return {}
    att, reward, queue, delay, throughput = hits[-1]
    return {
        "att": float(att),
        "reward": float(reward),
        "queue": float(queue),
        "delay": float(delay),
        "throughput": int(throughput),
    }


def ingest(run, logdir: Path) -> dict[str, Any]:
    """Move engine logs into the `runlog` run and return the report scalars.

    `att_last10` is the paper's protocol (mean of the last 10 TEST episodes), not
    a single test after training; for unstable learning they differ by hundreds
    of seconds. Methods without training have one measurement, so last10 == att.
    """
    episodes = parse_episodes(logdir)
    for row in episodes:
        run.log(
            step=row["episode"],
            split=row["split"],
            **{k: row.get(k) for k in DTL_COLUMNS if k in row},
        )

    result = parse_final(logdir)
    test_att = [r["att"] for r in episodes if r["split"] == "test" and "att" in r]
    if test_att:
        window = test_att[-REPORT_WINDOW:]
        result["att_last10"] = sum(window) / len(window)
        result["att_best"] = min(test_att)
    elif "att" in result:
        result["att_last10"] = result["att_best"] = result["att"]

    result["episodes_logged"] = len(episodes)
    run.result(**result)
    return result
