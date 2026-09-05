"""Most między tym repo a LibSignalem (submoduł `external/LibSignal`).

Dlaczego to w ogóle istnieje: LibSignal rozwiązuje ścieżki configów **względem
CWD**, a nie względem swojego pakietu — `utils/logger.py:build_config()` sięga po
`./configs/{task}/{agent}.yml`, `trainer/base_trainer.py` po `configs/sim/*.cfg`,
a `common/interface.py` po `os.getcwd()`. Uruchamiamy go więc z korzenia TEGO
repo: wtedy czyta nasze `configs/` i `data/`, a submoduł zostaje czystym
upstreamem plus patche silnika (`git diff master..thesis` w submodule = cały nasz
wkład w cudzy kod).

Konsekwencja, o której łatwo zapomnieć: LibSignal **nadpisuje** `configs/sim/*.cfg`
przy każdym uruchomieniu (wstawia świeże ścieżki `roadnetLogFile`/`replayLogFile`),
więc te pliki po każdym biegu wychodzą zabrudzone w `git status`. To nie jest twoja
zmiana — patrz `sim_configs_untouched()`.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBSIGNAL = REPO_ROOT / "external" / "LibSignal"
CONFIGS = REPO_ROOT / "configs"
DATA = REPO_ROOT / "data"
RESULTS = REPO_ROOT / "results"
RUNS = REPO_ROOT / "runs"


def sumo_home() -> Path:
    """SUMO instalowane jest pipem (`eclipse-sumo`), a LibSignal wymaga SUMO_HOME
    nawet dla `-w cityflow` (import na poziomie modułu w `world/__init__.py`)."""
    if env := os.environ.get("SUMO_HOME"):
        return Path(env)
    import sumo  # noqa: PLC0415 - opcjonalna zależność, importujemy leniwie

    return Path(sumo.__file__).parent


def check_ready() -> None:
    """Szybka diagnostyka zamiast ImportError trzy poziomy głębiej."""
    if not (LIBSIGNAL / "run.py").is_file():
        raise SystemExit(
            f"Brak submodułu LibSignal w {LIBSIGNAL}.\n"
            "Napraw:  git submodule update --init --recursive"
        )


def bootstrap(pythonhashseed: str = "0") -> None:
    """Ustaw proces tak, żeby `import agent` / `import trainer` trafiały do
    submodułu, a ścieżki względne — do korzenia repo.

    PYTHONHASHSEED musi być ustawiony PRZED startem interpretera, żeby zadziałał
    (`world_sumo` jest bez niego niedeterministyczny) — tutaj ustawiamy go tylko
    dla procesów potomnych — wejście przez `tsc` restartuje interpreter i robi to
    poprawnie (patrz `cli.ensure_hashseed`).
    """
    check_ready()
    os.environ.setdefault("SUMO_HOME", str(sumo_home()))
    os.environ.setdefault("PYTHONHASHSEED", pythonhashseed)
    if str(LIBSIGNAL) not in sys.path:
        sys.path.insert(0, str(LIBSIGNAL))
    os.chdir(REPO_ROOT)


def child_env(**extra: str) -> dict[str, str]:
    """Środowisko dla uruchomień w podprocesie (bench uruchamia każdy run osobno,
    bo LibSignal trzyma stan w globalnym Registry i nie da się go czysto zresetować)."""
    env = dict(os.environ)
    env["SUMO_HOME"] = str(sumo_home())
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(LIBSIGNAL), str(REPO_ROOT / "src"), env.get("PYTHONPATH")])
    )
    env.update(extra)
    return env


@contextmanager
def sim_configs_untouched():
    """Cofnij zabrudzenia, które LibSignal wpisuje do `configs/sim/*.cfg` w trakcie
    biegu (ścieżki logów replaya ze znacznikiem czasu), żeby `git status` pokazywał
    tylko prawdziwe zmiany.

    Działa na MIGAWCE BAJTÓW zdjętej przed biegiem, nie na `git checkout -- configs/sim`.
    Powód jest bolesny: tamta wersja kasowała również PRAWDZIWE, niezacommitowane zmiany
    w tych plikach — poprawiasz ścieżkę do roadnetu, odpalasz bieg, poprawka znika, i nie
    ma po niej śladu ani w gicie, ani w logu. Migawka przywraca dokładnie to, co było,
    niezależnie od tego, czy jest zacommitowane.
    """
    sim = CONFIGS / "sim"
    snapshot = {p: p.read_bytes() for p in sim.glob("*.cfg")}
    try:
        yield
    finally:
        for path, data in snapshot.items():
            if path.exists() and path.read_bytes() != data:
                path.write_bytes(data)
