"""`tsc train`: training or evaluating a LibSignal agent (our counterpart of its `run.py`).

Differences from the original:

1. Our agents are registered: importing `tsc_marl.libsignal.agents` fills the
   shared Registry, so `-a efficientmp` works like `-a colight`.
2. `--yellow` overrides the protocol after the config is built, so one tree serves
   both the realistic (5 s) and the benchmark (0 s) mode.
3. Results go to `runs/`, not to the submodule's `data/output_data`.

Process setup (bootstrap, seed, Registry sequence, run registration) lives in
`libsignal/session.py`, shared with `train_core.py` and `evaluate.py`; this entry
only runs LibSignal's `TSCTask(runner).run()`. Run it through `tsc train`, which
sets PYTHONHASHSEED before the interpreter starts (SUMO is nondeterministic without).
"""

from __future__ import annotations

import time

from tsc_marl.libsignal import experiment, session
from tsc_marl.libsignal.experiment import parse_args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    with session.session(args) as s:
        from common.registry import Registry

        task_name = Registry.mapping["command_mapping"]["setting"].param["task"]
        started = time.time()
        runner = Registry.mapping["trainer_mapping"][task_name](s.logger)
        Registry.mapping["task_mapping"][task_name](runner).run()
        s.logger.info(f"Total time taken: {time.time() - started}")
        # engine logs -> per-episode metrics and report scalars (incl. att_last10)
        experiment.ingest(s.run, experiment.engine_log_dir(s.run, args))
    print(f"runlog: {s.run.path}")


if __name__ == "__main__":
    main()
