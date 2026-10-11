"""Import direction between the layers of `tsc_marl` (see the package docstring).

A layer may import itself and the layers above it in `LAYERS`, never the ones
below. Every import counts (module level, inside functions and under
`TYPE_CHECKING`), because a lazy import is still a dependency, just one that
fails later. `runlog` is a separate tool and must not import `tsc_marl` at all.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tsc_marl import paths

SRC = paths.REPO_ROOT / "src"

LAYERS = (
    ("paths", "seeding"),
    ("domain",),
    ("core",),
    ("libsignal",),
    ("uncertainty",),
    ("robustness",),
    ("analysis",),
    ("runner",),
)
RANK = {name: rank for rank, names in enumerate(LAYERS) for name in names}

# Known upward imports, each scheduled for removal. The test fails both on a new
# violation and on a listed one that no longer exists, so the list cannot go stale.
KNOWN = {
    ("core/loop.py", "robustness"),
    ("libsignal/experiment.py", "runner"),
    ("libsignal/session.py", "runner"),
}


def imported_modules(path: Path) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
            out.update(f"{node.module}.{alias.name}" for alias in node.names)
    return out


def layer_of(module: str) -> str | None:
    parts = module.split(".")
    if parts[0] != "tsc_marl" or len(parts) < 2:
        return None
    return parts[1] if parts[1] in RANK else None


def upward_imports() -> set[tuple[str, str]]:
    root = SRC / "tsc_marl"
    found = set()
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        own = rel.parts[0].removesuffix(".py")
        if own not in RANK:
            continue
        for module in imported_modules(path):
            target = layer_of(module)
            if target is not None and RANK[target] > RANK[own]:
                found.add((rel.as_posix(), target))
    return found


def test_layers_cover_the_package():
    """A new subpackage must be placed in `LAYERS`, otherwise it is not checked."""
    names = {
        p.name.removesuffix(".py")
        for p in (SRC / "tsc_marl").iterdir()
        if not p.name.startswith("_") and p.name != "__pycache__"
    }
    assert names == set(RANK)


def test_no_upward_imports():
    found = upward_imports()
    assert found - KNOWN == set(), "new upward import"
    assert KNOWN - found == set(), "fixed, remove it from KNOWN"


def test_runlog_is_independent():
    offenders = [
        path.relative_to(SRC).as_posix()
        for path in sorted((SRC / "runlog").rglob("*.py"))
        if any(m.split(".")[0] == "tsc_marl" for m in imported_modules(path))
    ]
    assert offenders == []
