"""runlog — przechowywanie i odpytywanie wyników eksperymentów.

Niezależne od dziedziny: nie wie nic o symulatorach, uczeniu ani o projekcie,
w którym akurat stoi. Zależy tylko od stdlib (zapis) oraz pandas i pyyaml
(odczyt/agregacja, importowane leniwie).

    from runlog import Experiment

    exp = Experiment("t1", root="runs")
    if not exp.is_done(run_id, params):
        with exp.start(run_id, params, config=cfg, repos={"repo": Path(".")}) as run:
            for e in range(episodes):
                run.log(step=e, split="train", att=..., loss=...)
            run.result(att=..., att_last10=...)

    exp.results()                                        # DataFrame do tabel
    exp.aggregate(by=["agent"], over="seed", metrics=["att"])

Kontrakt i uzasadnienie decyzji projektowych: `src/runlog/README.md`.
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
