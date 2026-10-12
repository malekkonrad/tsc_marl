"""`tsc eval`: evaluate a frozen LibSignal policy from a checkpoint, without training.

A separate entry point because `TSCTask.run()` calls `trainer.test()` with the
default `drop_load=True`, so through the task path weights are NEVER loaded.
Weights come from ANOTHER run directory and the result goes to its OWN, so an
evaluation never overwrites the training logs and several evaluations of one
policy can live side by side.

    tsc eval --load-run runs/t1/colight__jinan__s0_y5 --checkpoint best
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

from tsc_marl import libsignal
from tsc_marl.libsignal import session

CKPT_RE = re.compile(r"^(\d+)_(\d+)\.pt$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a frozen policy")
    p.add_argument("--load-run", required=True, help="training run directory (runs/t1/...)")
    p.add_argument(
        "--checkpoint",
        default="best",
        help="'best' (best test episode AMONG SAVED ones), 'last' or an episode number",
    )
    p.add_argument("--tag", default="eval", help="run_id suffix, the evaluation scenario name")
    p.add_argument("--seed", type=int, default=None, help="default: the source run's seed")
    p.add_argument("--yellow", type=int, default=None, help="default: as in the source run")
    p.add_argument("--ngpu", default="-1", help="-1 = CPU, as in `train.py`")
    return p.parse_args(argv)


def engine_run_dir(run_path: Path, world: str, agent: str, network: str, prefix: str) -> Path:
    """Where LibSignal kept the artefacts of a run."""
    return run_path / "engine" / "tsc" / f"{world}_{agent}" / network / prefix


def available_checkpoints(model_dir: Path) -> dict[int, list[Path]]:
    """{episode: [agent weight files]}; `save_model` names them `{episode}_{rank}.pt`."""
    out: dict[int, list[Path]] = {}
    for f in sorted(model_dir.glob("*.pt")):
        m = CKPT_RE.match(f.name)
        if m:
            out.setdefault(int(m.group(1)), []).append(f)
    return out


def pick_episode(spec: str, saved: dict[int, list[Path]], run_path: Path) -> int:
    """Which checkpoint to take. `best` is chosen ONLY among episodes with saved
    weights (saved every `save_rate`), so the best logged episode that has no file
    is never picked silently."""
    if not saved:
        raise SystemExit(f"No checkpoints in {run_path}; is this a training run?")
    if spec == "last":
        return max(saved)
    if spec.isdigit():
        e = int(spec)
        if e not in saved:
            raise SystemExit(f"No checkpoint for episode {e}. Saved: {sorted(saved)}")
        return e
    if spec != "best":
        raise SystemExit(f"Unknown --checkpoint value: {spec}")

    import pandas as pd

    metrics = run_path / "metrics.csv"
    if not metrics.is_file():
        raise SystemExit(f"No metrics.csv in {run_path}; 'best' has nothing to choose from.")
    df = pd.read_csv(metrics)
    df = df[(df["split"] == "test") & (df["metric"] == "att") & (df["step"].isin(saved))]
    if df.empty:
        raise SystemExit("No saved episode has a test metric; use 'last'.")
    return int(df.loc[df["value"].idxmin(), "step"])


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    libsignal.bootstrap()

    from tsc_marl.libsignal import experiment
    from tsc_marl.libsignal.experiment import parse_args as run_parse_args

    src = Path(args.load_run).resolve()
    exp = experiment.experiment()
    parent = next((r for r in exp.runs() if r.path == src), None)
    if parent is None:
        raise SystemExit(f"{src} does not look like a runlog run directory (no meta.json)")

    p = parent.params
    seed = args.seed if args.seed is not None else p["seed"]
    yellow = args.yellow if args.yellow is not None else p["yellow_length"]

    run_args = run_parse_args(
        [
            "-a",
            p["agent"],
            "-w",
            p["world"],
            "-n",
            p["network"],
            "--seed",
            str(seed),
            "--yellow",
            str(yellow),
            "--ngpu",
            args.ngpu,
            # the SOURCE run's episode count: nothing is trained, but a default 200
            # in the result row would suggest a training that never happened
            "--episodes",
            str(p["episodes"]),
        ]
    )

    # Pick the checkpoint BEFORE registering the run: a typo in `--checkpoint` ends
    # with a message, not a `failed` run directory in `runs/`.
    model_dir = engine_run_dir(src, p["world"], p["agent"], p["network"], run_args.prefix) / "model"
    saved = available_checkpoints(model_dir)
    episode = pick_episode(args.checkpoint, saved, src)

    with session.session(
        run_args,
        # BEFORE build_params: `train_model` is in the `model` section, part of the
        # fingerprint; otherwise LibSignal would retrain from scratch under a
        # fingerprint claiming the training config.
        config_patch={"model": {"train_model": False, "test_model": True}},
        params_extra={
            "parent_run": parent.id,
            "checkpoint": args.checkpoint,
            "scenario": args.tag,
        },
        run_id=experiment.run_id(p["agent"], p["network"], seed, yellow, suffix=args.tag),
        log_level=logging.INFO,
    ) as s:
        import torch
        from common.registry import Registry

        task_name = Registry.mapping["command_mapping"]["setting"].param["task"]
        runner = Registry.mapping["trainer_mapping"][task_name](s.logger)
        # world, agents and metrics are built in BaseTrainer.__init__ (self.create());
        # calling create_* again would build a second CityFlow engine
        for ag in runner.agents:
            weights = model_dir / f"{episode}_{ag.rank}.pt"
            state = torch.load(weights, map_location="cpu")
            ag.model.load_state_dict(state)
            ag.target_model.load_state_dict(state)
        s.logger.info(f"Loaded episode {episode} from {model_dir}")

        runner.test(drop_load=True)  # weights loaded by hand; do not overwrite them
        result = experiment.ingest(s.run, experiment.engine_log_dir(s.run, run_args))
        s.run.result(loaded_episode=episode, loaded_from=str(src))
    print(f"runlog: {s.run.path}  ATT={result.get('att')}")


if __name__ == "__main__":
    main()
