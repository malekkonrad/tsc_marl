"""runlog: storing and querying experiment results.

Domain-independent: knows nothing about simulators, learning or the project it
lives in. Depends only on the standard library (writing) plus pandas and pyyaml
(reading and aggregation, imported lazily).

    from runlog import Experiment

    exp = Experiment("t1", root="runs")
    if not exp.is_done(run_id, params):
        with exp.start(run_id, params, config=cfg, repos={"repo": Path(".")}) as run:
            for e in range(episodes):
                run.log(step=e, split="train", att=..., loss=...)
            run.result(att=..., att_last10=...)

    exp.results()                                        # DataFrame for tables
    exp.aggregate(by=["agent"], over="seed", metrics=["att"])

Contract and design rationale: `src/runlog/README.md`.
"""

from runlog.experiment import Experiment
from runlog.provenance import collect, git_provenance
from runlog.run import FAILED, OK, RUNNING, TIMEOUT, Run, RunView
from runlog.store import RunExists, RunIdCollision, params_hash

__all__ = [
    "Experiment",
    "Run",
    "RunView",
    "RunExists",
    "RunIdCollision",
    "collect",
    "git_provenance",
    "params_hash",
    "OK",
    "FAILED",
    "RUNNING",
    "TIMEOUT",
]
