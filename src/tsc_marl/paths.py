"""Repository layout — the only place that knows where things live.

Separate from the LibSignal bridge because `analysis/` and `runner/suite.py` need
only the paths, not the bridge. Imports nothing from the project and has no side
effects, so it is importable before LibSignal is on `sys.path`.

Anchored at this file's location, not at the CWD: LibSignal resolves paths against
the CWD, so it is LibSignal that gets adjusted to us (`libsignal/bridge.py`).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBSIGNAL = REPO_ROOT / "external" / "LibSignal"
CONFIGS = REPO_ROOT / "configs"
DATA = REPO_ROOT / "data"
RESULTS = REPO_ROOT / "results"
RUNS = REPO_ROOT / "runs"
