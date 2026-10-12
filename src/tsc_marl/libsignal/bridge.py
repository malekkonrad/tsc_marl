"""Bridge between this repository and LibSignal (the `external/LibSignal` submodule).

LibSignal resolves config paths relative to the CWD, not to its package
(`utils/logger.py:build_config()` reads `./configs/{task}/{agent}.yml`, the trainer
reads `configs/sim/*.cfg`). So it runs from the root of THIS repository: it reads
our `configs/` and `data/`, and the submodule stays upstream plus engine patches
(`git diff master..thesis` in the submodule is our whole contribution to it).

Easy to forget: LibSignal REWRITES `configs/sim/*.cfg` on every run (fresh replay
log paths), see `sim_configs_untouched()`.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

from tsc_marl import paths


def sumo_home() -> Path:
    """SUMO is installed with pip (`eclipse-sumo`); LibSignal needs SUMO_HOME even
    for `-w cityflow` (module-level import in `world/__init__.py`)."""
    if env := os.environ.get("SUMO_HOME"):
        return Path(env)
    import sumo  # optional dependency, imported lazily

    return Path(sumo.__file__).parent


def check_ready() -> None:
    """A quick diagnosis instead of an ImportError three levels deeper."""
    if not (paths.LIBSIGNAL / "run.py").is_file():
        raise SystemExit(
            f"LibSignal submodule missing in {paths.LIBSIGNAL}.\n"
            "Fix:  git submodule update --init --recursive"
        )


def bootstrap(pythonhashseed: str = "0") -> None:
    """Set the process up so that `import agent` / `import trainer` reach the
    submodule and relative paths the repository root.

    PYTHONHASHSEED only works when set BEFORE the interpreter starts; here it is
    set for child processes only, while `tsc` restarts the interpreter properly
    (`runner/cli.py:ensure_hashseed`).
    """
    check_ready()
    os.environ.setdefault("SUMO_HOME", str(sumo_home()))
    os.environ.setdefault("PYTHONHASHSEED", pythonhashseed)
    if str(paths.LIBSIGNAL) not in sys.path:
        sys.path.insert(0, str(paths.LIBSIGNAL))
    os.chdir(paths.REPO_ROOT)


def child_env(**extra: str) -> dict[str, str]:
    """Environment for child processes (every run is a separate process, because
    LibSignal keeps state in a global Registry that cannot be reset cleanly)."""
    env = dict(os.environ)
    env["SUMO_HOME"] = str(sumo_home())
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(paths.LIBSIGNAL), str(paths.REPO_ROOT / "src"), env.get("PYTHONPATH")])
    )
    env.update(extra)
    return env


@contextmanager
def sim_configs_untouched():
    """Undo what LibSignal writes into `configs/sim/*.cfg` during a run (timestamped
    replay log paths), so `git status` shows only real changes.

    Works on a BYTE SNAPSHOT taken before the run, not `git checkout -- configs/sim`,
    which would also delete REAL uncommitted edits to these files.
    """
    sim = paths.CONFIGS / "sim"
    snapshot = {p: p.read_bytes() for p in sim.glob("*.cfg")}
    try:
        yield
    finally:
        for path, data in snapshot.items():
            if path.exists() and path.read_bytes() != data:
                path.write_bytes(data)
