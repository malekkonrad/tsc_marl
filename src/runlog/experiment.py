"""Experiment = a directory of runs: registration, lookup, aggregation."""

from __future__ import annotations

import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runlog import store
from runlog.provenance import collect
from runlog.run import OK, RUNNING, Run, RunView


def is_orphan(meta: dict[str, Any]) -> bool:
    """Whether a `running` run is dead: written on THIS machine by a process that
    no longer exists. Conservative both ways: a run from another machine or
    without a recorded `pid` counts as alive (trampling a live process is worse
    than a manual `replace=True`); a reused pid also reads as alive."""
    pid, host = meta.get("pid"), meta.get("host")
    if not isinstance(pid, int) or host != platform.node():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False  # the process exists, it is just not ours
    return False


class Experiment:
    def __init__(self, name: str, root: str | Path = "runs"):
        self.name = name
        self.root = Path(root)

    @property
    def path(self) -> Path:
        return self.root / self.name

    # ---------------------------------------------------------------- writing

    def start(
        self,
        run_id: str,
        params: dict[str, Any],
        config: Any = None,
        repos: dict[str, Path] | None = None,
        replace: bool = False,
    ) -> Run:
        """Register a run and return its handle (a context manager).

        Collision rules, the core of the tool:
        * same `run_id`, DIFFERENT `params` -> `RunIdCollision`. Never overwrite
          silently: otherwise two experiments share a directory undetectably.
        * status `ok` or `running` -> `RunExists` (unless `replace=True`);
          `running` blocks too, since it may be a LIVE run in another process,
          except an orphan (`is_orphan`: same machine, dead process), which would
          otherwise block resuming a suite forever.
        * status `failed`/`timeout` -> resuming allowed, the directory is cleared.
        """
        path = self.path / run_id
        digest = store.params_hash(params)
        existing = store.read_json(path / store.META)

        if existing and not replace:
            if existing.get("params_hash") != digest:
                raise store.RunIdCollision(
                    f"run_id '{run_id}' exists with DIFFERENT parameters "
                    f"({existing.get('params_hash')} vs {digest}). "
                    "Change the run_id or pass replace=True to really overwrite."
                )
            if existing.get("status") == OK or (
                existing.get("status") == RUNNING and not is_orphan(existing)
            ):
                raise store.RunExists(
                    f"run_id '{run_id}' has status '{existing.get('status')}'; "
                    "use find() to skip it or replace=True to recompute."
                )

        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)

        meta = {
            "run_id": run_id,
            "experiment": self.name,
            "status": RUNNING,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "params": params,
            "params_hash": digest,
            "provenance": collect(repos),
            # who computes it: without this an orphaned `running` looks alive
            "pid": os.getpid(),
            "host": platform.node(),
        }
        store.write_json(path / store.META, meta)
        run = Run(path, meta)
        if config is not None:
            run.attach_config(config)
        return run

    # --------------------------------------------------------------- reading

    def runs(self) -> list[RunView]:
        return [RunView(d) for d in store.run_dirs(self.root, self.name)]

    def find(self, status: str | None = OK, **params: Any) -> list[RunView]:
        """Runs matching the given parameters; only `ok` by default, since "already
        computed" means computed TO THE END. `status=None` disables the filter."""
        out = []
        for run in self.runs():
            if status is not None and run.status != status:
                continue
            if all(run.params.get(k) == v for k, v in params.items()):
                out.append(run)
        return out

    def is_done(self, run_id: str, params: dict[str, Any]) -> bool:
        """Skip-if-done: the run exists, finished `ok` and has THE SAME parameters;
        any config change gives another fingerprint and a recomputation."""
        meta = store.read_json(self.path / run_id / store.META)
        return bool(
            meta
            and meta.get("status") == OK
            and meta.get("params_hash") == store.params_hash(params)
        )

    # ------------------------------------------------------------- data frames

    def results(self, status: str | None = None):
        """DataFrame: one row per run, columns = params + result + metadata."""
        import pandas as pd  # lazy, so the write path does not pay for it

        rows = []
        for run in self.runs():
            if status is not None and run.status != status:
                continue
            row = {"run_id": run.id, "status": run.status}
            row.update(run.params)
            row.update(run.result)
            row["duration_s"] = run.meta.get("duration_s")
            rows.append(row)
        return pd.DataFrame(rows)

    def metrics(self, run_ids: list[str] | None = None):
        """Long DataFrame: run_id, step, split, metric, value, over all runs."""
        import pandas as pd

        frames = []
        for run in self.runs():
            if run_ids is not None and run.id not in run_ids:
                continue
            path = run.path / store.METRICS
            if not path.is_file():
                continue
            df = pd.read_csv(path)
            df.insert(0, "run_id", run.id)
            frames.append(df)
        return (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["run_id", *store.METRIC_FIELDS])
        )

    def aggregate(
        self,
        by: list[str],
        over: str,
        metrics: list[str],
        status: str | None = OK,
        df=None,
    ):
        """mean/std/n over `over` (usually 'seed'), grouped by `by`.

        `n` counts UNIQUE values of `over`, not rows: a repeated run of one seed
        must not pose as a larger sample. `df` lets the caller pass its own frame
        (e.g. with derived columns) instead of reading runs from disk."""
        if df is None:
            df = self.results(status=status)
        if df.empty:
            return df
        if over in by:
            raise ValueError(f"'{over}' cannot be in both `by` and `over`")
        present = [m for m in metrics if m in df.columns]
        agg = {m: ["mean", "std"] for m in present}
        agg[over] = ["nunique"]
        out = df.groupby(by, dropna=False).agg(agg)
        out.columns = ["n" if c == (over, "nunique") else f"{c[0]}_{c[1]}" for c in out.columns]
        return out.reset_index()
