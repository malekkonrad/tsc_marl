"""Where a result came from, collected automatically when a run starts.

Provenance is mandatory, not optional: a result row pointing at mutable files is
not reproducible. Hence `dirty` for every repository: a run computed on an
uncommitted tree is still a result, but it must say so.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

# Variables that REALLY change results. An allow-list, not the whole environment:
# dumping os.environ into metadata would leak secrets and add noise.
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
    """SHA, branch and `dirty` flag for several repositories (main repo, submodules,
    data). No git or no repo gives `None` entries, never an exception: provenance
    must not break a run."""
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
