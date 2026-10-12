"""`tsc core`: training entry point of the core (GAT + PPO + CTDE).

`train.py` hands control to LibSignal's task loop, because there the agent is a
Registry class. The core is not a LibSignal agent: it has its own loop
(`core/loop.py`) and needs only the `World`. Both entries share the process
setup (`session.session()`, one canonical Registry sequence) and differ in a
single step: where `train.py` calls `TSCTask(runner).run()`, this one builds a
`CoreTrainer`.

Run it through `tsc core`, which sets PYTHONHASHSEED before the interpreter starts.
"""

from __future__ import annotations

import os
import sys
import time

from tsc_marl.libsignal import session

DEFAULT_AGENT = "gatppo"


def parse_args(argv: list[str] | None = None):
    """`train.py`'s parser with another default agent, so flags mean the same in
    both entries (both write runs to the same `runlog`)."""
    from tsc_marl.libsignal.experiment import parse_args as base_parse_args

    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(a in ("-a", "--agent") for a in argv):
        argv += ["-a", DEFAULT_AGENT]
    return base_parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # The core needs the graph, which LibSignal builds only with `model.graphic`;
    # the config must declare it (`gatppo.yml` does) rather than this entry
    # changing the shared Registry sequence.
    with session.session(args, require_graph=True) as s:
        from common.registry import Registry

        from tsc_marl.core.loop import CoreTrainer
        from tsc_marl.core.ppo import PPOConfig
        from tsc_marl.domain.sensors import FailureMix
        from tsc_marl.robustness import training
        from tsc_marl.robustness.degrade import base_network

        cmd = Registry.mapping["command_mapping"]["setting"].param
        started = time.time()

        # Training with failures and/or UW-attention. Both switches live in the
        # `model` section (part of the fingerprint), not CLI flags, so a different
        # setting invalidates skip-if-done. Without them this is the plain core.
        failures = FailureMix.from_registry()
        uw = training.uw_config_from_registry()

        # The world comes from Registry, as in `BaseTrainer`, so `-w sumo` takes
        # the same path.
        world = Registry.mapping["world_mapping"][cmd["world"]](
            os.path.join("configs/sim", f"{args.network}.cfg"),
            cmd["thread_num"],
            interface=cmd["interface"],
        )

        core = CoreTrainer.from_world(
            world,
            ppo_config=PPOConfig.from_registry(),
            run=s.run,
            logger=s.logger,
            device=s.device,
            failures=failures,
        )
        if uw is not None:
            # After building the core: the signal needs `n_mov` and `edge_index`.
            # Its parts are frozen. `uw_signal` is the function `tsc degrade` uses,
            # so training sees exactly the signal it is later evaluated with.
            core.uncertainty = training.UWStream(
                training.uw_signal(uw, base_network(args.network), core.env),
                core.env.graph.edge_index,
                core.device,
                n_nodes=core.env.graph.n_nodes,
            )
        # `model.save_best`: also keep the checkpoint of the best test episode
        # (part of the fingerprint, so such runs get their own id).
        if Registry.mapping["model_mapping"]["setting"].param.get("save_best"):
            core.best_path = s.run.engine_dir / "core_best.pt"
        s.logger.info(
            f"core: {core.env.graph} | ob_length {core.env.features.ob_length} "
            f"| device {core.device}"
        )
        if failures is not None:
            s.logger.info(f"training failures: {failures.describe()} (seed {failures.seed})")
        if uw is not None:
            s.logger.info(
                f"UW-attention in training: u={uw.get('u', 'head')} "
                f"side={uw.get('side', 'target')} tau={uw.get('tau', 'default')}"
            )
        result = core.train()
        core.save(s.run.engine_dir / "core.pt")
        s.logger.info(f"Total time taken: {time.time() - started:.1f}s | {result}")
    print(f"runlog: {s.run.path}")


if __name__ == "__main__":
    main()
