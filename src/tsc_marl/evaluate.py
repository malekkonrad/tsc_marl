"""Ewaluacja zamrożonej polityki z checkpointu — bez treningu.

Po co osobne wejście: `TSCTask.run()` woła `trainer.test()` z domyślnym
`drop_load=True`, więc przez ścieżkę taskową wagi NIGDY się nie ładują. Faza 3
(ta sama polityka pod różnymi awariami sensorów) jest bez tego nieosiągalna —
każdy scenariusz wymagałby powtórzenia treningu.

Wagi ładujemy z CUDZEGO katalogu biegu, a wynik zapisujemy do WŁASNEGO: inaczej
ewaluacja nadpisałaby logi treningu, a Faza 3 nie mogłaby trzymać N ewaluacji
jednej polityki obok siebie.

    tsc eval --load-run runs/t1/colight__jinan__s0_y5 --checkpoint best
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

from tsc_marl import libsignal, session

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
    p.add_argument("--ngpu", default="-1", help="-1 = CPU; jak w `run.py`")
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

    from tsc_marl import experiment
    from tsc_marl.run import parse_args as run_parse_args

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
            "--ngpu",
            args.ngpu,
            # liczba epizodów z biegu ŹRÓDŁOWEGO: ewaluacja niczego nie trenuje, ale
            # gdyby wiersz wyniku pokazywał domyślne 200, sugerowałby trening, którego nie było
            "--episodes",
            str(p["episodes"]),
        ]
    )

    # Wybór checkpointu PRZED rejestracją biegu: literówka w `--checkpoint` ma się
    # skończyć komunikatem, a nie katalogiem biegu ze statusem `failed` w `runs/`.
    # Nic z tego nie potrzebuje Registry — wystarczy katalog biegu źródłowego.
    model_dir = engine_run_dir(src, p["world"], p["agent"], p["network"], run_args.prefix) / "model"
    saved = available_checkpoints(model_dir)
    episode = pick_episode(args.checkpoint, saved, src)

    with session.session(
        run_args,
        # PRZED build_params: `train_model` siedzi w sekcji `model`, która wchodzi do
        # `config_hash`. Bez tego LibSignal przetrenowałby model od zera, a odcisk
        # biegu i tak twierdziłby, że to ten sam config co trening.
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
        # świat, agenci i metryki powstają już w BaseTrainer.__init__ (woła self.create()),
        # więc runner jest gotowy — ponowne create_* zbudowałoby drugi silnik CityFlow
        for ag in runner.agents:
            weights = model_dir / f"{episode}_{ag.rank}.pt"
            state = torch.load(weights, map_location="cpu")
            ag.model.load_state_dict(state)
            ag.target_model.load_state_dict(state)
        s.logger.info(f"Wczytano epizod {episode} z {model_dir}")

        runner.test(drop_load=True)  # wagi już wczytane ręcznie — niech ich nie nadpisze
        result = experiment.ingest(s.run, experiment.engine_log_dir(s.run, run_args))
        s.run.result(loaded_episode=episode, loaded_from=str(src))
    print(f"runlog: {s.run.path}  ATT={result.get('att')}")


if __name__ == "__main__":
    main()
