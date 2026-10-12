"""On-disk layout and safe writes.

The file system IS the database: one run = one directory, no central index.
Not a simplification but a requirement: runs execute in parallel processes, and
a shared index file would mean write races or locks.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

META = "meta.json"
RESULT = "result.json"
METRICS = "metrics.csv"
CONFIG = "config.resolved.yml"
ENGINE = "engine"  # artefacts of external tools (checkpoints, engine logs)

METRIC_FIELDS = ("step", "split", "metric", "value")


class RunIdCollision(Exception):
    """Same run_id, different parameters: never overwritten silently."""


class RunExists(Exception):
    """A run with this run_id is already done (or still running)."""


def params_hash(params: dict[str, Any]) -> str:
    """Stable parameter fingerprint; `default=str` so Path, enums and numpy floats
    serialise (the hash must be repeatable, not pretty)."""
    blob = json.dumps(params, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def atomic_write(path: Path, text: str) -> None:
    """Write via a temporary file and `os.replace` (atomic on POSIX and Windows), so
    a process killed mid-write never leaves a truncated JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_json(path: Path, payload: Any) -> None:
    atomic_write(path, json.dumps(payload, indent=2, default=str, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any | None:
    """None instead of raising: a run directory may be mid-write or damaged, and a
    scan must not fail because of one broken entry."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_yaml(path: Path, payload: Any) -> None:
    import yaml  # lazy: the metrics write path should not pay for the import

    atomic_write(path, yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def run_dirs(root: Path, experiment: str) -> list[Path]:
    """Run directories of an experiment: those with a `meta.json`."""
    base = root / experiment
    if not base.is_dir():
        return []
    return sorted(d for d in base.iterdir() if d.is_dir() and (d / META).is_file())
