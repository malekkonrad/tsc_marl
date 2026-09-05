"""Entry point treningu/ewaluacji — nasz odpowiednik `run.py` z LibSignala.

Różnice względem oryginału, i po co one są:

1. **Rejestruje naszych agentów.** `import agent` (submoduł) + `import tsc_marl.agents`
   zapełniają wspólne Registry, więc `-a efficientmp` działa tak samo jak `-a colight`.

2. **`--yellow` nadpisuje protokół po zbudowaniu configu.** W sandboxie tryb
   "benchmark" (yellow_length: 0, tak liczą swoje wyniki implementacje referencyjne
   z papierów) wymagał TRZECIEJ kopii całego drzewa LibSignala. `yellow_length` jedzie
   przez `Registry.mapping['trainer_mapping']['setting']`, więc wystarczy podmienić
   `config['trainer']` przed `Trainer_param_Interface()` — jedno drzewo, dwa tryby.

3. **Wyniki lądują w `runs/`**, nie w `data/output_data` submodułu.

Przygotowanie procesu (bootstrap, seed, sekwencja interfejsów Registry, rejestracja
biegu w `runlog`) siedzi w `tsc_marl/session.py` — wspólne dla tego wejścia,
`run_core.py` i `evaluate.py`. Tutaj zostaje wyłącznie to, co robi LibSignal:
`TSCTask(runner).run()`.

Uruchamiaj przez `tsc train` — CLI ustawia PYTHONHASHSEED przed startem interpretera
(restartuje się, patrz `cli.ensure_hashseed`). Bez tego `world_sumo` jest
niedeterministyczny.
"""

from __future__ import annotations

import argparse
import time

from tsc_marl import experiment, session


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trening/ewaluacja agenta TSC")
    # --- argumenty, których oczekuje Registry LibSignala (nazwy muszą się zgadzać) ---
    p.add_argument("-t", "--task", default="tsc")
    p.add_argument("-a", "--agent", default="efficientmp", help="nazwa z Registry")
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
    # --- nasze ---
    p.add_argument(
        "--yellow",
        type=int,
        default=None,
        metavar="S",
        help="nadpisz yellow_length z protokołu: 5 = tryb realistyczny (domyślny "
        "z configs/tsc/thesis_protocol.yml), 0 = tryb benchmark jak w papierach",
    )
    p.add_argument("--episodes", type=int, default=None, help="nadpisz liczbę epizodów")
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        metavar="S",
        help="nadpisz długość epizodu (treningowego I testowego) w sekundach symulacji. "
        "TYLKO do smoke-testów i debugowania — protokół pracy to 3600",
    )
    p.add_argument(
        "--suite",
        default=None,
        metavar="PLIK",
        help="plik suite'a (configs/suites/*.yml albo sama nazwa) — jego blok `overrides` "
        "wchodzi do configu PRZED odciskiem biegu. Tak runner siatki przekazuje "
        "nadpisania do podprocesu: ścieżką, a nie serializacją słownika w argv",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="nadpisz identyfikator biegu (domyślnie {agent}__{sieć}__s{seed}_y{yellow})",
    )
    p.add_argument(
        "--replace",
        action="store_true",
        help="policz ponownie bieg, który już się udał (domyślnie to błąd)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    with session.session(args) as s:
        from common.registry import Registry

        task_name = Registry.mapping["command_mapping"]["setting"].param["task"]
        started = time.time()
        runner = Registry.mapping["trainer_mapping"][task_name](s.logger)
        Registry.mapping["task_mapping"][task_name](runner).run()
        s.logger.info(f"Total time taken: {time.time() - started}")
        # logi silnika → metryki per epizod + skalary raportowe (w tym att_last10)
        experiment.ingest(s.run, experiment.engine_log_dir(s.run, args))
    print(f"runlog: {s.run.path}")


if __name__ == "__main__":
    main()
