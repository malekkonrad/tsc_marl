"""Most do LibSignala (submoduł `external/LibSignal`) — cała styczność z cudzym kodem.

Trzy warstwy, od najniższej:

* `bridge`     — `sys.path`, `SUMO_HOME`, CWD, środowisko podprocesów,
* `session`    — postawiony proces LibSignala pod zarejestrowanym biegiem `runlog`,
* `experiment` — adapter LibSignal ↔ `runlog` (id biegu, parsowanie ich logów),
* `agents/`    — nasze wtyczki rejestrowane w ICH `Registry`.

Funkcje `bridge` są re-eksportowane tutaj, bo wołający chce po prostu
`libsignal.bootstrap()`, a nie pamiętać, w którym pliku most trzyma implementację.
Ścieżek repo tu NIE MA — te mieszkają w `tsc_marl.paths` i nie wymagają mostu.
"""

from tsc_marl.libsignal.bridge import (  # noqa: F401
    bootstrap,
    check_ready,
    child_env,
    sim_configs_untouched,
    sumo_home,
)
