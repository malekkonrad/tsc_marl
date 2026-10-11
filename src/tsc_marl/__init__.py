"""Multi-agent traffic signal control over the road-network graph.

The package is layered; dependencies point only DOWN this list:

* `paths`, `seeding` — foundation, no internal dependencies,
* `domain/`          — traffic formulas from the literature (ATS), framework-free,
* `core/`            — the controller: GAT + PPO + CTDE,
* `libsignal/`       — bridge to the `external/LibSignal` submodule and its Registry plugins,
* `robustness/`      — controllers and wrappers evaluated under sensor failure,
* `uncertainty/`     — observation-uncertainty head and its calibration,
* `runner/`          — process entry points (`tsc ...`) and run orchestration,
* `analysis/`        — tables and figures from stored runs.

Key boundary: `core/` does not import `libsignal/`. The simulator is injected by
`runner/train_core.py`, so the core is testable without an engine.
"""
