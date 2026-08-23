"""Układ katalogów na dysku i bezpieczne zapisy.

System plików JEST bazą danych: jeden bieg = jeden katalog, brak centralnego
indeksu. To nie jest uproszczenie, tylko wymóg — biegi bywają odpalane
równolegle w osobnych procesach, a wspólny plik indeksu oznaczałby wyścig
o zapis albo blokady.
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
ENGINE = "engine"  # artefakty obcych narzędzi (checkpointy, logi silnika)

METRIC_FIELDS = ("step", "split", "metric", "value")


class RunIdCollision(Exception):
    """Ten sam run_id, inne parametry — nigdy nie nadpisujemy po cichu."""


class RunExists(Exception):
    """Bieg o tym run_id już się policzył (albo właśnie trwa)."""


def params_hash(params: dict[str, Any]) -> str:
    """Stabilny odcisk parametrów. `default=str` żeby Path/enum/np.float nie
    wywracały serializacji — odcisk ma być powtarzalny, nie ładny."""
    blob = json.dumps(params, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def atomic_write(path: Path, text: str) -> None:
    """Zapis przez plik tymczasowy + `os.replace` (atomowy na POSIX i Windows).
    Bez tego proces zabity w trakcie zapisu zostawia obcięty JSON, a cały bieg
    staje się nieczytelny — łącznie z metrykami, które przeżyły."""
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
    """Zwraca None zamiast rzucać: katalog biegu może być w trakcie zapisu albo
    uszkodzony przez coś spoza narzędzia, a skan po katalogach nie może się
    wywalić przez jeden zepsuty wpis."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_yaml(path: Path, payload: Any) -> None:
    import yaml  # noqa: PLC0415 - leniwie: ścieżka zapisu metryk nie ma płacić za import

    atomic_write(path, yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def run_dirs(root: Path, experiment: str) -> list[Path]:
    """Katalogi biegów danego eksperymentu — te, które mają `meta.json`."""
    base = root / experiment
    if not base.is_dir():
        return []
    return sorted(d for d in base.iterdir() if d.is_dir() and (d / META).is_file())
