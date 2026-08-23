"""Pojedynczy bieg: zapis metryk, wyniku i statusu."""

from __future__ import annotations

import csv
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runlog import store

# Cykl życia biegu. Rozróżnienie ok/failed to nie kosmetyka: bez niego wyszukiwanie
# "co już policzone" uzna przerwany bieg za zrobiony i cicho pominie brakujący wynik.
RUNNING = "running"
OK = "ok"
FAILED = "failed"
TIMEOUT = "timeout"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Run:
    """Uchwyt do katalogu biegu. Używaj jako context managera — wtedy status
    domknie się sam, także gdy trening poleci wyjątkiem."""

    def __init__(self, path: Path, meta: dict[str, Any]):
        self.path = Path(path)
        self.meta = meta
        self._metrics_file = None
        self._metrics_writer = None

    # ------------------------------------------------------------------ zapis

    @property
    def id(self) -> str:
        return self.meta["run_id"]

    @property
    def engine_dir(self) -> Path:
        """Katalog na artefakty obcych narzędzi (checkpointy, logi silnika).
        Trzymane WEWNĄTRZ biegu, żeby „jeden bieg = jeden katalog" było prawdą."""
        d = self.path / store.ENGINE
        d.mkdir(parents=True, exist_ok=True)
        return d

    def log(self, step: int, split: str = "train", **metrics: float) -> None:
        """Dopisuje metryki dla danego kroku. Format długi (`step,split,metric,value`),
        bo różne metody logują różne metryki — DQN `q_loss`, PPO `entropy`/`kl` —
        a szeroki CSV wymagałby znajomości kolumn z góry i psuł sklejanie biegów."""
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
        # flush po każdym kroku: 7-godzinny trening ma być widoczny w trakcie,
        # a crash nie może zjeść historii, która już się wydarzyła
        self._metrics_file.flush()

    def result(self, **values: Any) -> None:
        """Skalary końcowe biegu — to one trafiają do tabel."""
        payload = store.read_json(self.path / store.RESULT) or {}
        payload.update(values)
        store.write_json(self.path / store.RESULT, payload)

    def attach_config(self, config: Any) -> None:
        """Snapshot ROZWIĄZANEGO wejścia — po scaleniu plików i nadpisaniach z CLI.
        Zamyka dziurę „wiersz wyniku odsyła do pliku, który ktoś potem zmienił"."""
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
        return False  # nie połykamy wyjątku — status to zapis faktu, nie obsługa błędu


class RunView:
    """Bieg wczytany z dysku, tylko do odczytu."""

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
