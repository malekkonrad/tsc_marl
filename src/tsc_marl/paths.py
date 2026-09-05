"""Layout tego repo — jedyne miejsce, które wie, gdzie co leży.

Wydzielone z mostu do LibSignala, bo `analysis/` i `runner/suite.py` potrzebują
samych ścieżek (`RUNS`, `RESULTS`, `CONFIGS`), a nie całego mostu. Ten moduł nie
importuje niczego z projektu i nie ma efektów ubocznych — można go zaimportować
zanim LibSignal w ogóle znajdzie się na `sys.path`.

Kotwicą jest położenie TEGO pliku (`src/tsc_marl/paths.py` → dwa poziomy w górę),
a nie CWD — LibSignal rozwiązuje ścieżki względem CWD i to on musi się dostroić
do nas (patrz `libsignal/bridge.py`), nie odwrotnie.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBSIGNAL = REPO_ROOT / "external" / "LibSignal"
CONFIGS = REPO_ROOT / "configs"
DATA = REPO_ROOT / "data"
RESULTS = REPO_ROOT / "results"
RUNS = REPO_ROOT / "runs"
