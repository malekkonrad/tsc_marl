"""Adapter między LibSignalem a `runlog` — jedyne miejsce, które zna oba światy.

`runlog` nie wie nic o ruchu drogowym, LibSignal nie wie nic o `runlog`. Tutaj
tłumaczymy jedno na drugie: identyfikator biegu, parametry, i parsowanie logów
silnika na metryki.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from runlog import Experiment
from tsc_marl.libsignal import LIBSIGNAL, REPO_ROOT, RUNS

EXPERIMENT = "t1"

# Sekcje configu, które opisują TREŚĆ NAUKOWĄ biegu. Ich odcisk wchodzi do
# parametrów, więc zmiana `NUM_HEADS` czy `learning_rate` unieważnia skip-if-done.
# `command`/`logger` celowo pominięte — tam siedzą ścieżki i prefiksy, które
# różnią się między biegami, nie zmieniając eksperymentu.
CONFIG_SECTIONS = ("model", "trainer")

# "Final Travel Time is 316.1114, mean rewards: -447.5272, queue: 10.4976,
#  delay: 2.7385, throughput: 6151"
FINAL_RE = re.compile(
    r"Final Travel Time is ([\d.]+), mean rewards: (-?[\d.]+), queue: ([\d.]+), "
    r"delay: ([\d.]+), throughput: (\d+)"
)
# Kolumny `tsc_trainer.writeLog`: nazwa, tryb, epizod, ATT, loss, reward, queue, delay, throughput
DTL_COLUMNS = ("att", "loss", "reward", "queue", "delay", "throughput")
REPORT_WINDOW = 10  # protokół Advanced-CoLight 5.1: średnia z ostatnich 10 epizodów testowych


def experiment(root: Path | str = RUNS) -> Experiment:
    return Experiment(EXPERIMENT, root=root)


def repos() -> dict[str, Path]:
    """Proweniencja: wynik zależy od obu drzew — naszego i submodułu silnika."""
    return {"tsc_marl": REPO_ROOT, "libsignal": LIBSIGNAL}


def run_id(agent: str, network: str, seed: int, yellow: int, suffix: str = "") -> str:
    """Czytelny gołym okiem, żeby dało się znaleźć checkpoint bez zaglądania do metadanych."""
    base = f"{agent}__{network}__s{seed}_y{yellow}"
    return f"{base}__{suffix}" if suffix else base


def resolve_config(args) -> dict[str, Any]:
    """Config po scaleniu `includes` i po nadpisaniach z CLI — czyli to, co REALNIE
    trafia do symulatora. Jedna funkcja dla procesu liczącego i dla runnera siatki:
    gdyby liczyły parametry osobno, skip-if-done rozjechałby się po cichu."""
    from utils.logger import build_config  # noqa: PLC0415 - dopiero po bootstrapie

    config, _ = build_config(args)
    if args.yellow is not None:
        config["trainer"]["yellow_length"] = args.yellow
    if args.episodes is not None:
        config["trainer"]["episodes"] = args.episodes
    return config


def plan(agent: str, world: str, network: str, seed: int, yellow: int):
    """(args, params, run_id) dla zadanego punktu siatki — bez uruchamiania czegokolwiek.
    Pozwala runnerowi sprawdzić „czy już policzone" tą samą drogą, którą pójdzie proces."""
    from tsc_marl.run import parse_args  # noqa: PLC0415

    args = parse_args(
        ["-a", agent, "-w", world, "-n", network, "--seed", str(seed), "--yellow", str(yellow)]
    )
    config = resolve_config(args)
    params = build_params(args, config)
    return args, params, run_id(agent, network, seed, config["trainer"]["yellow_length"])


def build_params(args, config: dict[str, Any]) -> dict[str, Any]:
    """Parametry biegu = to, co identyfikuje eksperyment. Odcisk configu wchodzi
    jako pojedyncze pole, żeby porównanie było tanie, a jednocześnie czułe na
    KAŻDĄ zmianę hiperparametrów."""
    from runlog import params_hash  # noqa: PLC0415

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
        # parametry metod MP-owych: bez nich wiersz NIE jest odtwarzalny
        **{k: model.get(k) for k in ("t_min", "w1", "obs_length", "drop_always_green")},
        "config_hash": params_hash({k: config.get(k) for k in CONFIG_SECTIONS}),
    }


def engine_log_dir(run, args) -> Path:
    """Gdzie LibSignal składuje logi TEGO biegu. Odwzorowuje `get_output_file_path`
    po naszym patchu: root_dir wskazuje na `engine/` wewnątrz katalogu biegu."""
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
    """Wiersze `_DTL.log` → rekordy per epizod. Metody bez treningu nie mają tego
    pliku i zwracają pustą listę."""
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
    """Ostatnia linia `_BRF.log` → skalary końcowe. Czytamy z PLIKU, nie ze stdout
    podprocesu: log jest artefaktem biegu, stdout bywa uciętym buforem."""
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
    """Przenosi logi silnika do biegu `runlog` i zwraca skalary raportowe.

    `att_last10` to protokół z papieru (średnia z ostatnich 10 epizodów TESTOWYCH),
    a nie pojedynczy test po treningu — dla niestabilnego uczenia różnica sięga
    setek sekund. Metody bez treningu mają jeden pomiar, więc last10 == att.
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
