"""Ewaluacja zamrożonej polityki z checkpointu — bez treningu.

Po co osobne wejście: `TSCTask.run()` woła `trainer.test()` z domyślnym
`drop_load=True`, więc przez ścieżkę taskową wagi NIGDY się nie ładują. Faza 3
(ta sama polityka pod różnymi awariami sensorów) jest bez tego nieosiągalna —
każdy scenariusz wymagałby powtórzenia treningu.

Wagi ładujemy z CUDZEGO katalogu biegu, a wynik zapisujemy do WŁASNEGO: inaczej
ewaluacja nadpisałaby logi treningu, a Faza 3 nie mogłaby trzymać N ewaluacji
jednej polityki obok siebie.

    scripts/run.sh --eval --load-run runs/t1/colight__jinan__s0_y5 --checkpoint best
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path

from tsc_marl import libsignal

CKPT_RE = re.compile(r"^(\d+)_(\d+)\.pt$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ewaluacja zamrożonej polityki")
    p.add_argument("--load-run", required=True, help="katalog biegu treningowego (runs/t1/...)")
    p.add_argument(
        "--checkpoint",
        default="best",
        help="'best' (najlepszy epizod testowy SPOŚRÓD ZAPISANYCH), 'last' albo numer epizodu",
    )
    p.add_argument("--tag", default="eval", help="sufiks run_id — nazwa scenariusza ewaluacji")
    p.add_argument("--seed", type=int, default=None, help="domyślnie seed biegu źródłowego")
    p.add_argument("--yellow", type=int, default=None, help="domyślnie jak w biegu źródłowym")
    return p.parse_args(argv)


def engine_run_dir(run_path: Path, world: str, agent: str, network: str, prefix: str) -> Path:
    """Katalog, w którym LibSignal trzymał artefakty danego biegu."""
    return run_path / "engine" / "tsc" / f"{world}_{agent}" / network / prefix


def available_checkpoints(model_dir: Path) -> dict[int, list[Path]]:
    """{epizod: [pliki wag agentów]} — `save_model` nazywa je `{epizod}_{rank}.pt`."""
    out: dict[int, list[Path]] = {}
    for f in sorted(model_dir.glob("*.pt")):
        m = CKPT_RE.match(f.name)
        if m:
            out.setdefault(int(m.group(1)), []).append(f)
    return out


def pick_episode(spec: str, saved: dict[int, list[Path]], run_path: Path) -> int:
    """Który checkpoint wziąć.

    `best` liczymy WYŁĄCZNIE spośród epizodów, które mają zapisane wagi — wagi
    lecą co `save_rate`, więc najlepszy epizod w logu często nie ma pliku. Wybór
    „najlepszego, którego nie da się załadować" byłby cichym kłamstwem."""
    if not saved:
        raise SystemExit(f"Brak checkpointów w {run_path} — czy ten bieg to trening?")
    if spec == "last":
        return max(saved)
    if spec.isdigit():
        e = int(spec)
        if e not in saved:
            raise SystemExit(f"Brak checkpointu dla epizodu {e}. Zapisane: {sorted(saved)}")
        return e
    if spec != "best":
        raise SystemExit(f"Nieznana wartość --checkpoint: {spec}")

    import pandas as pd  # noqa: PLC0415

    metrics = run_path / "metrics.csv"
    if not metrics.is_file():
        raise SystemExit(f"Brak metrics.csv w {run_path} — 'best' nie ma z czego wybrać.")
    df = pd.read_csv(metrics)
    df = df[(df["split"] == "test") & (df["metric"] == "att") & (df["step"].isin(saved))]
    if df.empty:
        raise SystemExit("Żaden zapisany epizod nie ma metryki testowej — użyj 'last'.")
    return int(df.loc[df["value"].idxmin(), "step"])


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    libsignal.bootstrap()

    import agent  # noqa: F401, PLC0415
    import dataset  # noqa: F401, PLC0415
    import task  # noqa: F401, PLC0415
    import torch  # noqa: PLC0415
    import trainer  # noqa: F401, PLC0415
    from common import interface  # noqa: PLC0415
    from common.registry import Registry  # noqa: PLC0415
    from utils.logger import setup_logging  # noqa: PLC0415

    import tsc_marl.agents  # noqa: F401, PLC0415
    from tsc_marl import experiment  # noqa: PLC0415
    from tsc_marl.run import parse_args as run_parse_args  # noqa: PLC0415
    from tsc_marl.seeding import set_global_seed  # noqa: PLC0415

    src = Path(args.load_run).resolve()
    exp = experiment.experiment()
    parent = next((r for r in exp.runs() if r.path == src), None)
    if parent is None:
        raise SystemExit(f"{src} nie wygląda na katalog biegu runlog (brak meta.json)")

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
            # liczba epizodów z biegu ŹRÓDŁOWEGO: ewaluacja niczego nie trenuje, ale
            # gdyby wiersz wyniku pokazywał domyślne 200, sugerowałby trening, którego nie było
            "--episodes",
            str(p["episodes"]),
        ]
    )
    set_global_seed(seed)
    config = experiment.resolve_config(run_args)
    # tylko ewaluacja: bez tego LibSignal przetrenowałby model od zera
    config["model"]["train_model"] = False
    config["model"]["test_model"] = True

    params = experiment.build_params(run_args, config) | {
        "parent_run": parent.id,
        "checkpoint": args.checkpoint,
        "scenario": args.tag,
    }
    rid = experiment.run_id(p["agent"], p["network"], seed, yellow, suffix=args.tag)
    run = exp.start(rid, params, config=config, repos=experiment.repos())
    config["logger"]["root_dir"] = str(run.engine_dir)

    interface.Command_Setting_Interface(config)
    interface.Logger_param_Interface(config)
    interface.World_param_Interface(config)
    if config["model"].get("graphic", False):
        wp = Registry.mapping["world_mapping"]["setting"].param
        interface.Graph_World_Interface(wp["dir"] + wp["roadnetFile"])
    interface.Logger_path_Interface(config)
    os.makedirs(Registry.mapping["logger_mapping"]["path"].path, exist_ok=True)
    interface.Trainer_param_Interface(config)
    interface.ModelAgent_param_Interface(config)

    logger = setup_logging(logging.INFO)
    task_name = Registry.mapping["command_mapping"]["setting"].param["task"]

    # budowa runnera TEŻ pod blokiem — pad przy stawianiu świata ma dać status `failed`,
    # a nie zostawić bieg na wieki w `running` (patrz komentarz w run.py)
    with run, libsignal.sim_configs_untouched():
        runner = Registry.mapping["trainer_mapping"][task_name](logger)
        # świat, agenci i metryki powstają już w BaseTrainer.__init__ (woła self.create()),
        # więc runner jest gotowy — ponowne create_* zbudowałoby drugi silnik CityFlow
        model_dir = (
            engine_run_dir(src, p["world"], p["agent"], p["network"], run_args.prefix) / "model"
        )
        saved = available_checkpoints(model_dir)
        episode = pick_episode(args.checkpoint, saved, src)
        for ag in runner.agents:
            weights = model_dir / f"{episode}_{ag.rank}.pt"
            state = torch.load(weights, map_location="cpu")
            ag.model.load_state_dict(state)
            ag.target_model.load_state_dict(state)
        logger.info(f"Wczytano epizod {episode} z {model_dir}")

        runner.test(drop_load=True)  # wagi już wczytane ręcznie — niech ich nie nadpisze
        result = experiment.ingest(run, experiment.engine_log_dir(run, run_args))
        run.result(loaded_episode=episode, loaded_from=str(src))
    print(f"runlog: {run.path}  ATT={result.get('att')}")


if __name__ == "__main__":
    main()
