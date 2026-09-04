"""Wejście CLI rdzenia Fazy 1 (GAT + PPO + CTDE) — odpowiednik `run.py` dla rdzenia.

`run.py` oddaje sterowanie LibSignalowi (`Registry.mapping['task_mapping']`), bo
tam agentem jest klasa z Registry. Rdzeń nie jest agentem LibSignala: pętlę ma
własną (`loop.py`), a od symulatora potrzebuje tylko `World`. Dlatego ten plik
powtarza z `run.py` DOKŁADNIE tę część, która przygotowuje środowisko —
bootstrap, seed, sekwencję interfejsów Registry, rejestrację biegu w `runlog` —
a w miejscu, gdzie tamten woła `TSCTask(runner).run()`, buduje `CoreTrainer`.

Dlaczego nie jeden wspólny `run.py` z gałęzią: sekwencja interfejsów jest
kolejnościowo wrażliwa i już raz kosztowała nas cichy błąd (`root_dir`
ignorowany → `att_last10` przyjmowało wartość `att`, 27% różnicy). Dwa krótkie,
czytelne wejścia są bezpieczniejsze niż jedno z `if`-em w środku tej sekwencji.

Uruchamiaj przez `scripts/run.sh`, żeby PYTHONHASHSEED był ustawiony PRZED
startem interpretera.
"""

from __future__ import annotations

import logging
import os
import sys
import time

from tsc_marl import libsignal
from tsc_marl.seeding import set_global_seed

DEFAULT_AGENT = "gatppo"


def parse_args(argv: list[str] | None = None):
    """Ten sam parser co `run.py` — tylko z innym domyślnym agentem.

    Świadomie nie kopiujemy definicji argumentów: `--yellow`, `--episodes`,
    `--run-id` i `--replace` muszą znaczyć to samo w obu wejściach, bo obie
    ścieżki zapisują biegi do jednego `runlog`.
    """
    from tsc_marl.run import parse_args as base_parse_args  # noqa: PLC0415

    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(a in ("-a", "--agent") for a in argv):
        argv += ["-a", DEFAULT_AGENT]
    return base_parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    libsignal.bootstrap()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.ngpu
    set_global_seed(args.seed)

    # importy PO bootstrapie i WYŁĄCZNIE dla efektu ubocznego: dekoratory zapełniają
    # Registry, z którego korzystają interfejsy poniżej. Nie usuwaj jako "nieużywanych".
    import agent  # noqa: F401
    import dataset  # noqa: F401
    import task  # noqa: F401
    import trainer  # noqa: F401
    from common import interface
    from common.registry import Registry
    from utils.logger import setup_logging

    import tsc_marl.agents  # noqa: F401 - rejestruje NASZYCH agentów (baseline'y)
    from tsc_marl import experiment
    from tsc_marl.loop import CoreTrainer
    from tsc_marl.ppo import PPOConfig

    config = experiment.resolve_config(args)

    exp = experiment.experiment()
    params = experiment.build_params(args, config)
    rid = args.run_id or experiment.run_id(
        args.agent, args.network, args.seed, config["trainer"]["yellow_length"]
    )
    run = exp.start(rid, params, config=config, repos=experiment.repos(), replace=args.replace)
    # checkpointy i logi silnika WEWNĄTRZ katalogu biegu — „jeden bieg = jeden katalog"
    config["logger"]["root_dir"] = str(run.engine_dir)

    with run, libsignal.sim_configs_untouched():
        interface.Command_Setting_Interface(config)
        interface.Logger_param_Interface(config)
        interface.World_param_Interface(config)
        # Bezwarunkowo, w odróżnieniu od `run.py`: rdzeń bez grafu nie istnieje.
        world_param = Registry.mapping["world_mapping"]["setting"].param
        interface.Graph_World_Interface(world_param["dir"] + world_param["roadnetFile"])
        interface.Logger_path_Interface(config)
        os.makedirs(Registry.mapping["logger_mapping"]["path"].path, exist_ok=True)
        interface.Trainer_param_Interface(config)
        interface.ModelAgent_param_Interface(config)

        logger = setup_logging(logging.DEBUG if args.debug else logging.INFO)
        cmd = Registry.mapping["command_mapping"]["setting"].param
        started = time.time()

        # Świat budujemy przez Registry, a nie `from world.world_cityflow import World`,
        # żeby `-w sumo` (Faza 5) działało tą samą ścieżką. To ten sam sposób,
        # w jaki robi to `BaseTrainer`.
        world = Registry.mapping["world_mapping"][cmd["world"]](
            os.path.join("configs/sim", f"{args.network}.cfg"),
            cmd["thread_num"],
            interface=cmd["interface"],
        )

        core = CoreTrainer.from_world(
            world, ppo_config=PPOConfig.from_registry(), run=run, logger=logger
        )
        logger.info(f"rdzeń: {core.env.graph} | ob_length {core.env.features.ob_length}")
        result = core.train()
        core.save(run.engine_dir / "core.pt")
        logger.info(f"Total time taken: {time.time() - started:.1f}s | {result}")
    print(f"runlog: {run.path}")


if __name__ == "__main__":
    main()
