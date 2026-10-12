"""A single run: metrics, result and status."""

from __future__ import annotations

import csv
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runlog import store

# Run life cycle. ok vs failed matters: without it "what is already computed" would
# treat an interrupted run as done and silently skip the missing result.
RUNNING = "running"
OK = "ok"
FAILED = "failed"
TIMEOUT = "timeout"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Run:
    """Handle to a run directory. Use it as a context manager, so the status is
    closed even when training raises."""

    def __init__(self, path: Path, meta: dict[str, Any]):
        self.path = Path(path)
        self.meta = meta
        self._metrics_file = None
        self._metrics_writer = None

    # ------------------------------------------------------------------ writing

    @property
    def id(self) -> str:
        return self.meta["run_id"]

    @property
    def engine_dir(self) -> Path:
        """Directory for artefacts of external tools (checkpoints, engine logs),
        INSIDE the run, so "one run = one directory" holds."""
        d = self.path / store.ENGINE
        d.mkdir(parents=True, exist_ok=True)
        return d

    def log(self, step: int, split: str = "train", **metrics: float) -> None:
        """Append metrics for a step, in long format (`step,split,metric,value`):
        methods log different metrics, and a wide CSV would need the columns up
        front and break concatenating runs."""
        if self._metrics_writer is None:
            path = self.path / store.METRICS
            new = not path.exists()
            self._metrics_file = open(path, "a", newline="", encoding="utf-8")  # noqa: SIM115
            self._metrics_writer = csv.writer(self._metrics_file)
            if new:
                self._metrics_writer.writerow(store.METRIC_FIELDS)
        for name, value in metrics.items():
            if value is None:
                continue
            self._metrics_writer.writerow([step, split, name, value])
        # flush every step: a long run is visible while it runs, and a crash
        # cannot eat history that already happened
        self._metrics_file.flush()

    def result(self, **values: Any) -> None:
        """Final scalars of the run, the ones that go into tables."""
        payload = store.read_json(self.path / store.RESULT) or {}
        payload.update(values)
        store.write_json(self.path / store.RESULT, payload)

    def attach_config(self, config: Any) -> None:
        """Snapshot of the RESOLVED input, after merging files and CLI overrides, so
        a result row never points at a file someone changed later."""
        store.write_yaml(self.path / store.CONFIG, config)

    def finish(self, status: str = OK, error: str | None = None) -> None:
        if self._metrics_file is not None:
            self._metrics_file.close()
            self._metrics_file = self._metrics_writer = None
        started = self.meta.get("started_at")
        self.meta.update(status=status, finished_at=_now())
        if error:
            self.meta["error"] = error
        if started:
            delta = datetime.fromisoformat(self.meta["finished_at"]) - datetime.fromisoformat(
                started
            )
            self.meta["duration_s"] = round(delta.total_seconds(), 1)
        store.write_json(self.path / store.META, self.meta)

    # -------------------------------------------------------- context manager

    def __enter__(self) -> Run:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.finish(OK)
        elif isinstance(exc, KeyboardInterrupt):
            self.finish(TIMEOUT, error="KeyboardInterrupt")
        else:
            self.finish(
                FAILED, error="".join(traceback.format_exception(exc_type, exc, tb))[-4000:]
            )
        return False  # never swallow: the status records a fact, it does not handle errors


class RunView:
    """A run read from disk, read-only."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.meta = store.read_json(self.path / store.META) or {}
        self.result = store.read_json(self.path / store.RESULT) or {}

    @property
    def id(self) -> str:
        return self.meta.get("run_id", self.path.name)

    @property
    def status(self) -> str:
        return self.meta.get("status", "unknown")

    @property
    def params(self) -> dict[str, Any]:
        return self.meta.get("params", {})

    def __repr__(self) -> str:
        return f"<RunView {self.id} [{self.status}]>"
