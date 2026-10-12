# runlog

Storing and querying experiment results. Domain-independent: it knows nothing
about simulators, machine learning or the project it lives in. Dependencies: the
standard library (writing) plus `pandas` and `pyyaml` (reading, imported lazily).

## Why

A typical research pipeline writes results to one CSV that points to config
files, which later change. The consequences are always the same: a row cannot be
reproduced, two experiments share a directory, and an interrupted run looks
finished. `runlog` turns these three things into invariants.

## Data model

One run = one directory. **The file system is the database**; the index is
computed by scanning, there is no central file, so parallel runs in separate
processes do not fight over writes and need no locks.

```
runs/<experiment>/<run_id>/
    meta.json            # params, params_hash, provenance, status, times
    config.resolved.yml  # snapshot of the INPUT after merging and overrides
    metrics.csv          # step,split,metric,value  (long format)
    result.json          # final scalars
    engine/              # artefacts of external tools (checkpoints, logs)
```

## Usage

```python
from runlog import Experiment

exp = Experiment("t1", root="runs")

if not exp.is_done(run_id, params):
    with exp.start(
        run_id, params, config=cfg, repos={"repo": Path("."), "lib": Path("external/lib")}
    ) as run:
        for e in range(episodes):
            run.log(step=e, split="train", att=..., loss=...)
        run.result(att=..., att_last10=...)

exp.results()  # DataFrame for tables
exp.metrics()  # long DataFrame for plots
exp.aggregate(by=["agent"], over="seed", metrics=["att"])  # mean/std/n
```

`start()` returns a context manager: leaving it normally gives status `ok`, an
exception gives `failed` with the traceback in `meta.json`, `KeyboardInterrupt`
gives `timeout`. The exception is not swallowed; the status records a fact, it
does not handle the error.

## Design decisions

**Collisions are errors, not overwrites.** The same `run_id` with another
`params_hash` raises `RunIdCollision`. Status `ok` or `running` raises
`RunExists` (`running` too, since it may be a live process), unless the run is an
orphan: written on this machine by a process that no longer exists. Status
`failed`/`timeout` allows resuming. `replace=True` bypasses this deliberately.

**Long metric format**, not wide. Methods log different metrics (DQN has
`q_loss`, PPO has `entropy` and `kl`); a wide CSV would need the columns up front
and break concatenating runs. For a quick look: `df.pivot_table(...)`.

**`n` in aggregation counts unique values of `over`**, not rows, so a repeated
run of the same seed does not pose as a larger sample.

**Atomic writes** (`tempfile` + `os.replace`) and metrics flushed after every
step: a killed process leaves no truncated JSON, and the history before the
crash survives for diagnosis.

**Tolerant reading.** A damaged `meta.json` gives a run with status `unknown`,
not an exception in the middle of a scan over a hundred directories.

**Provenance is mandatory**: SHA, branch and `dirty` flag of several repositories
at once (main repository, submodules), plus `argv`, Python version, host and an
allow-list of environment variables. An allow-list, not the whole `os.environ`,
which would leak secrets and add noise.

## What is not here

Launching runs, queueing, plotting, a server. `runlog` stores and queries;
launching a grid of experiments is the caller's job.
