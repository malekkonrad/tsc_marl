"""Eksperyment = katalog biegów. Rejestracja, wyszukiwanie, agregacja."""

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
    """Czy bieg w statusie `running` jest martwy: zapisany na TEJ maszynie przez
    proces, który już nie istnieje.

    Ostrożnie w obie strony: bieg z innej maszyny albo bez zapisanego `pid`
    (metadane sprzed tej zmiany) uznajemy za żywy — zdeptanie liczącego procesu
    jest gorsze niż ręczne `replace=True`. Ponowne użycie numeru pid przez inny
    proces też daje „żywy", czyli znowu błąd po bezpiecznej stronie."""
    pid, host = meta.get("pid"), meta.get("host")
    if not isinstance(pid, int) or host != platform.node():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False  # proces istnieje, tylko nie nasz
    return False


class Experiment:
    def __init__(self, name: str, root: str | Path = "runs"):
        self.name = name
        self.root = Path(root)

    @property
    def path(self) -> Path:
        return self.root / self.name

    # ---------------------------------------------------------------- zapis

    def start(
        self,
        run_id: str,
        params: dict[str, Any],
        config: Any = None,
        repos: dict[str, Path] | None = None,
        replace: bool = False,
    ) -> Run:
        """Rejestruje bieg i zwraca uchwyt (context manager).

        Reguły kolizji — sedno narzędzia:
        * te same `run_id`, INNE `params` → `RunIdCollision`. Nigdy, przenigdy
          nie nadpisujemy po cichu: to jest ta pomyłka, po której dwa różne
          eksperymenty dzielą jeden katalog i nie da się tego wykryć po fakcie.
        * status `ok` albo `running` → `RunExists` (chyba że `replace=True`).
          `running` też blokuje, bo może to być ŻYWY bieg w innym procesie —
          WYJĄTEK: bieg osierocony (`is_orphan`: ta sama maszyna, proces nie żyje),
          np. po `kill -TERM` albo restarcie maszyny. Taki bieg nigdy się nie
          domknie, a bez tego wyjątku blokowałby wznowienie siatki na zawsze.
        * status `failed`/`timeout` → wznowienie dozwolone, katalog czyszczony.
        """
        path = self.path / run_id
        digest = store.params_hash(params)
        existing = store.read_json(path / store.META)

        if existing and not replace:
            if existing.get("params_hash") != digest:
                raise store.RunIdCollision(
                    f"run_id '{run_id}' istnieje z INNYMI parametrami "
                    f"({existing.get('params_hash')} vs {digest}). "
                    "Zmień run_id albo podaj replace=True, jeśli naprawdę chcesz nadpisać."
                )
            if existing.get("status") == OK or (
                existing.get("status") == RUNNING and not is_orphan(existing)
            ):
                raise store.RunExists(
                    f"run_id '{run_id}' ma status '{existing.get('status')}' — "
                    "użyj find() żeby go pominąć, albo replace=True żeby policzyć ponownie."
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
            # kto liczy — bez tego osieroconego `running` nie da się odróżnić od żywego
            "pid": os.getpid(),
            "host": platform.node(),
        }
        store.write_json(path / store.META, meta)
        run = Run(path, meta)
        if config is not None:
            run.attach_config(config)
        return run

    # --------------------------------------------------------------- odczyt

    def runs(self) -> list[RunView]:
        return [RunView(d) for d in store.run_dirs(self.root, self.name)]

    def find(self, status: str | None = OK, **params: Any) -> list[RunView]:
        """Biegi pasujące do podanych parametrów. Domyślnie tylko `ok` — bo
        „czy to już policzone?" ma znaczyć policzone DO KOŃCA. `status=None`
        wyłącza filtr."""
        out = []
        for run in self.runs():
            if status is not None and run.status != status:
                continue
            if all(run.params.get(k) == v for k, v in params.items()):
                out.append(run)
        return out

    def is_done(self, run_id: str, params: dict[str, Any]) -> bool:
        """Skip-if-done: bieg istnieje, skończył się dobrze i ma TE SAME parametry.
        Zmiana czegokolwiek w configu daje inny odcisk → policzy się ponownie."""
        meta = store.read_json(self.path / run_id / store.META)
        return bool(
            meta
            and meta.get("status") == OK
            and meta.get("params_hash") == store.params_hash(params)
        )

    # ------------------------------------------------------------- ramki danych

    def results(self, status: str | None = None):
        """DataFrame: jeden wiersz na bieg, kolumny = params + result + metadane."""
        import pandas as pd  # noqa: PLC0415 - leniwie, żeby ścieżka zapisu nie płaciła

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
        """DataFrame długi: run_id, step, split, metric, value — sklejony z biegów."""
        import pandas as pd  # noqa: PLC0415

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
        """mean/std/n po `over` (zwykle 'seed'), pogrupowane po `by`.

        `n` liczy UNIKALNE wartości `over`, nie wiersze — powtórzony bieg tego
        samego seeda nie ma udawać większej próby.

        `df` pozwala podać własną ramkę (np. z kolumnami pochodnymi, po które
        `results()` nie sięga) zamiast czytać biegi z dysku."""
        if df is None:
            df = self.results(status=status)
        if df.empty:
            return df
        if over in by:
            raise ValueError(f"'{over}' nie może być jednocześnie w `by` i `over`")
        present = [m for m in metrics if m in df.columns]
        agg = {m: ["mean", "std"] for m in present}
        agg[over] = ["nunique"]
        out = df.groupby(by, dropna=False).agg(agg)
        out.columns = ["n" if c == (over, "nunique") else f"{c[0]}_{c[1]}" for c in out.columns]
        return out.reset_index()
