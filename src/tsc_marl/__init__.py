"""Multi-agent traffic signal control over the road-network graph.

The package is layered; dependencies point only DOWN this list:

* `paths`, `seeding`: foundation, no internal dependencies,
* `domain/`         : traffic formulas from the literature (ATS), framework-free,
* `core/`           : the GAT + PPO + CTDE controller,
* `libsignal/`      : bridge to the `external/LibSignal` submodule and its Registry plugins,
* `uncertainty/`    : observation-uncertainty head and its calibration,
* `robustness/`     : controllers and wrappers evaluated under sensor failure,
* `analysis/`       : tables and figures from stored runs,
* `runner/`         : process entry points (`tsc ...`) and run orchestration.

Key boundary: `core/` never builds a simulator. The `World` is injected by
`runner/train_core.py`, so the core's contracts are testable without an engine.
"""
