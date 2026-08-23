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

Uruchamiaj przez `scripts/run.sh`, żeby PYTHONHASHSEED był ustawiony przed startem
interpretera — bez tego `world_sumo` jest niedeterministyczny.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

from tsc_marl import libsignal
from tsc_marl.seeding import set_global_seed


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
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    libsignal.bootstrap()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.ngpu
    set_global_seed(args.seed)

    # importy PO bootstrapie — dopiero wtedy submoduł jest na sys.path
    # Wszystkie cztery są importowane WYŁĄCZNIE dla efektu ubocznego: dekoratory
    # zapełniają Registry. Nie usuwaj ich jako "nieużywanych" — bez `task` i `trainer`
    # poniższe Registry.mapping[...] leci KeyError.
    import agent  # noqa: F401
    import dataset  # noqa: F401
    import task  # noqa: F401
    import trainer  # noqa: F401
    from common import interface
    from common.registry import Registry
    from utils.logger import build_config, setup_logging

    import tsc_marl.agents  # noqa: F401 - rejestruje NASZYCH agentów

    config, _ = build_config(args)

    if args.yellow is not None:
        config["trainer"]["yellow_length"] = args.yellow
    if args.episodes is not None:
        config["trainer"]["episodes"] = args.episodes

    # kolejność wywołań jak w run.py LibSignala — zależności między interfejsami
    interface.Command_Setting_Interface(config)
    interface.Logger_param_Interface(config)
    interface.World_param_Interface(config)
    if config["model"].get("graphic", False):
        param = Registry.mapping["world_mapping"]["setting"].param
        roadnet = (
            param["dir"] + param["roadnetFile"]
            if config["command"]["world"] in ("cityflow", "sumo")
            else param["road_file_addr"]
        )
        interface.Graph_World_Interface(roadnet)
    interface.Logger_path_Interface(config)
    os.makedirs(Registry.mapping["logger_mapping"]["path"].path, exist_ok=True)
    interface.Trainer_param_Interface(config)
    interface.ModelAgent_param_Interface(config)

    logger = setup_logging(logging.DEBUG if args.debug else logging.INFO)
    task_name = Registry.mapping["command_mapping"]["setting"].param["task"]
    runner = Registry.mapping["trainer_mapping"][task_name](logger)
    started = time.time()
    try:
        Registry.mapping["task_mapping"][task_name](runner).run()
    finally:
        libsignal.restore_sim_configs()
    logger.info(f"Total time taken: {time.time() - started}")


if __name__ == "__main__":
    main()
