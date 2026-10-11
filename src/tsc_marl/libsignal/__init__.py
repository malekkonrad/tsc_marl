"""Bridge to LibSignal (`external/LibSignal` submodule) - all contact with third-party code.

Layers, lowest first:

* `bridge`     — `sys.path`, `SUMO_HOME`, CWD, subprocess environment,
* `session`    — a LibSignal process set up under a registered `runlog` run,
* `experiment` — LibSignal ↔ `runlog` adapter (run ids, log parsing),
* `sumo_world` — SUMO world with the CityFlow contract (replaces upstream `world_sumo`),
* `agents/`    — our agents registered in LibSignal's `Registry`.

`bridge` functions are re-exported so callers can write `libsignal.bootstrap()`.
Repository paths live in `tsc_marl.paths` and do not need the bridge.
"""

from tsc_marl.libsignal.bridge import (  # noqa: F401
    bootstrap,
    check_ready,
    child_env,
    sim_configs_untouched,
    sumo_home,
)
