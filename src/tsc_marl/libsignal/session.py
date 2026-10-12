"""Setting up a LibSignal process under a registered `runlog` run.

The only place that knows the ORDER of `common.interface.*` calls. The order is
sensitive and has already caused a silent bug (a `root_dir` set after
`Logger_path_Interface` was ignored, so `att_last10` took the value of `att`).
One canonical sequence serves every entry point.

The seam is where entry points really differ: a session ends with "Registry
filled, world NOT built yet". `runner/train.py` leaves the world to LibSignal's
trainer, `runner/train_core.py` builds it itself, `evaluate.py` gets it from the
runner.

    with session(args) as s:
        ...  # s.run, s.config, s.logger, Registry ready

The `with` body runs UNDER the open run: a failure while building the world gives
status `failed` instead of leaving the directory `running` forever.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from tsc_marl import libsignal
from tsc_marl.seeding import set_global_seed

# The order in which LibSignal's `run.py` fills the Registry, kept as data so a
# test can check it.
INTERFACE_ORDER = (
    "Command_Setting_Interface",
    "Logger_param_Interface",
    "World_param_Interface",
    "Graph_World_Interface",  # only with model.graphic, see build_interfaces
    "Logger_path_Interface",
    "Trainer_param_Interface",
    "ModelAgent_param_Interface",
)


@dataclass
class Session:
    """What an entry point gets from the session; `args` is kept because
    `experiment.engine_log_dir` rebuilds engine log paths from it."""

    run: Any  # runlog.Run, already open
    config: dict[str, Any]
    logger: logging.Logger
    args: Any
    device: Any = "cpu"  # torch.device resolved from `--ngpu`, see `torch_device`


def torch_device(ngpu: str):
    """`--ngpu` -> `torch.device`, the ONLY translation between the two.

    `session()` sets `CUDA_VISIBLE_DEVICES=<ngpu>` before torch touches CUDA, after
    which the chosen card is ALWAYS index 0 for torch: `--ngpu 2` gives `cuda:0`.
    A missing card with a requested GPU is not a hard error (the run is correct on
    the CPU, only slower), but it must be loud; `session()` logs the warning.
    """
    import torch  # heavy, imported only when needed

    if str(ngpu) == "-1" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda:0")


def register_engine_components() -> None:
    """Imports for their side effect only: decorators fill the Registry the
    interfaces use. Must run AFTER `libsignal.bootstrap()`."""
    import agent  # noqa: F401
    import dataset  # noqa: F401
    import task  # noqa: F401
    import trainer  # noqa: F401

    import tsc_marl.libsignal.agents  # noqa: F401 - registers OUR agents
    import tsc_marl.libsignal.sumo_world  # noqa: F401 - replaces 'sumo' in world_mapping


def build_interfaces(config: dict[str, Any]) -> list[str]:
    """Fill the Registry in `INTERFACE_ORDER`; returns the interfaces REALLY called,
    so a test can pin the order. `Graph_World_Interface` is conditional, as in
    `run.py`; the core declares `graphic: True` and `session()` checks it."""
    from common import interface
    from common.registry import Registry

    called = []

    interface.Command_Setting_Interface(config)
    called.append("Command_Setting_Interface")
    interface.Logger_param_Interface(config)
    called.append("Logger_param_Interface")
    interface.World_param_Interface(config)
    called.append("World_param_Interface")

    if config["model"].get("graphic", False):
        param = Registry.mapping["world_mapping"]["setting"].param
        roadnet = (
            param["dir"] + param["roadnetFile"]
            if config["command"]["world"] in ("cityflow", "sumo")
            else param["road_file_addr"]
        )
        interface.Graph_World_Interface(roadnet)
        if config["command"]["world"] == "sumo":
            # LibSignal's graph of an OSM network has 1 edge for 25 nodes (Krakow);
            # replaced for ALL graph agents, see `sumo_world.build_graph`
            from tsc_marl.libsignal.sumo_world import build_graph

            interface.Graph_World_Interface.graph = build_graph(roadnet)
        called.append("Graph_World_Interface")

    interface.Logger_path_Interface(config)
    called.append("Logger_path_Interface")
    # the directory must exist before anything writes to it
    os.makedirs(Registry.mapping["logger_mapping"]["path"].path, exist_ok=True)
    interface.Trainer_param_Interface(config)
    called.append("Trainer_param_Interface")
    interface.ModelAgent_param_Interface(config)
    called.append("ModelAgent_param_Interface")

    return called


def merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Recursive in-place merge of overrides; a shallow `update` would replace the
    whole `model` section when overriding one field."""
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def resolve(args, config_patch: dict[str, Any] | None = None) -> tuple[dict[str, Any], str]:
    """Config after ALL overrides, plus the `run_id` suffix.

    Shared by the computing process (`session`) and suite planning (`suite.plan`);
    separate computations would let skip-if-done drift silently. Override order:
    agent config -> suite `overrides` -> the caller's `config_patch`; the last wins,
    because `evaluate.py` must force `train_model: False` whatever the suite says.
    """
    from tsc_marl.libsignal import experiment

    config = experiment.resolve_config(args)
    suffix = ""
    if getattr(args, "suite", None):
        overrides, suffix = experiment.suite_overrides(args.suite)
        merge(config, overrides)
    if config_patch:
        merge(config, config_patch)
    return config, suffix


@contextmanager
def session(
    args,
    *,
    config_patch: dict[str, Any] | None = None,
    params_extra: dict[str, Any] | None = None,
    run_id: str | None = None,
    replace: bool | None = None,
    require_graph: bool = False,
    log_level: int | None = None,
):
    """Bootstrap -> seed -> config -> run registration -> interfaces -> logger.

    `config_patch` is applied BEFORE `build_params`: `train_model`/`test_model` and
    suite overrides live in the `model` section, part of the fingerprint; applied
    later they would change behaviour without changing the fingerprint.
    """
    libsignal.bootstrap()
    # one path to the GPU for training AND evaluation
    os.environ["CUDA_VISIBLE_DEVICES"] = args.ngpu
    set_global_seed(args.seed)

    register_engine_components()
    from utils.logger import setup_logging

    from tsc_marl.libsignal import experiment

    config, suffix = resolve(args, config_patch)

    if require_graph and not config["model"].get("graphic", False):
        raise SystemExit(
            f"agent config '{args.agent}' lacks `graphic: True`, and this entry needs the "
            "graph (Graph_World_Interface). Add `graphic: True` in configs/tsc/."
        )

    # register BEFORE the interfaces: `root_dir` must be set before LibSignal
    # computes its output paths in Logger_path_Interface
    exp = experiment.experiment()
    params = experiment.build_params(args, config) | dict(params_extra or {})
    rid = (
        run_id
        or getattr(args, "run_id", None)
        or experiment.run_id(
            args.agent, args.network, args.seed, config["trainer"]["yellow_length"], suffix=suffix
        )
    )
    if replace is None:
        replace = getattr(args, "replace", False)
    run = exp.start(rid, params, config=config, repos=experiment.repos(), replace=replace)
    # engine artefacts (checkpoints, logs, replays) go INSIDE the run directory, so
    # "one run = one directory" holds and data/ stays input-only (thanks to the
    # get_output_file_path patch in the submodule)
    config["logger"]["root_dir"] = str(run.engine_dir)

    if log_level is None:
        log_level = logging.DEBUG if getattr(args, "debug", False) else logging.INFO

    with run, libsignal.sim_configs_untouched():
        build_interfaces(config)
        logger = setup_logging(log_level)
        device = torch_device(args.ngpu)
        if args.ngpu != "-1" and device.type == "cpu":
            logger.warning(
                f"--ngpu {args.ngpu} requested, but torch sees no CUDA; running on the CPU. "
                "On a GPU machine check `make setup-gpu` (torch from the cu121 index)."
            )
        yield Session(run=run, config=config, logger=logger, args=args, device=device)


def world_kind(network: str) -> str:
    """`sumo` for SUMO networks (`configs/sim/sumo_*.cfg`), `cityflow` otherwise; by
    name, because `standalone_world` gets no `-w` flag."""
    return "sumo" if network.startswith("sumo_") else "cityflow"


@contextmanager
def standalone_world(
    agent: str,
    network: str,
    *,
    config_patch: dict[str, Any] | None = None,
    n_agents: int | None = None,
):
    """A LibSignal world and agents OUTSIDE the training loop (evaluation,
    diagnostics, datasets). No `runlog` run is opened here; a caller that wants
    provenance opens one. Everything runs under `sim_configs_untouched`, since
    `build_config` writes replay paths into `configs/sim/*.cfg`.

    -> (world, agents, config). `n_agents` defaults to one per intersection (the
    max-pressure methods); shared models such as Advanced-CoLight take `n_agents=1`.
    """
    libsignal.bootstrap()
    register_engine_components()

    from common.registry import Registry

    from tsc_marl import paths
    from tsc_marl.libsignal.experiment import parse_args

    kind = world_kind(network)
    args = parse_args(["-a", agent, "-w", kind, "-n", network])
    with libsignal.sim_configs_untouched():
        config, _ = resolve(args, config_patch)
        build_interfaces(config)

        cfg = paths.CONFIGS / "sim" / f"{network}.cfg"
        thread = config["trainer"].get("thread", 4)
        if kind == "sumo":
            # the `sumo_world` adapter (registered above), not upstream `world_sumo`:
            # the same path as `tsc core -w sumo`
            world = Registry.mapping["world_mapping"]["sumo"](str(cfg), thread, interface="libsumo")
        else:
            from world.world_cityflow import World

            world = World(str(cfg), thread_num=thread, interface="libsumo")
        n = len(world.intersections) if n_agents is None else n_agents
        agents = [Registry.mapping["model_mapping"][agent](world, rank) for rank in range(n)]
        yield world, agents, config
