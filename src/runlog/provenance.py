"""Skąd wziął się ten wynik — zbierane automatycznie przy starcie biegu.

Proweniencja jest obywatelem pierwszej kategorii, a nie polem opcjonalnym:
wiersz wyniku, który odsyła do zmiennych plików, nie jest odtwarzalny. Stąd
`dirty` przy każdym repozytorium — bieg policzony na niezacommitowanym drzewie
jest nadal wynikiem, ale musi się do tego przyznać.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

# Zmienne, które REALNIE zmieniają wynik. Whitelista, nie całe środowisko —
# zrzucanie os.environ do metadanych to wyciek sekretów i szum.
DEFAULT_ENV_KEYS = ("PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")


def _git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def git_provenance(repos: dict[str, Path] | None = None) -> dict[str, Any]:
    """SHA + gałąź + flaga `dirty` dla wielu repozytoriów naraz (repo główne,
    submoduły, katalog danych). Brak gita albo brak repo → wpis z samym `None`,
    bez wyjątku: proweniencja nie może wywrócić biegu."""
    out = {}
    for name, path in (repos or {}).items():
        sha = _git(Path(path), "rev-parse", "--short", "HEAD")
        status = _git(Path(path), "status", "--porcelain")
        out[name] = {
            "sha": sha,
            "branch": _git(Path(path), "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status) if status is not None else None,
        }
    return out


def env_provenance(env_keys: tuple[str, ...] = DEFAULT_ENV_KEYS) -> dict[str, Any]:
    return {
        "argv": list(sys.argv),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cwd": os.getcwd(),
        "env": {k: os.environ.get(k) for k in env_keys},
    }


def collect(repos: dict[str, Path] | None = None, env_keys=DEFAULT_ENV_KEYS) -> dict[str, Any]:
    return {"git": git_provenance(repos), **env_provenance(env_keys)}
