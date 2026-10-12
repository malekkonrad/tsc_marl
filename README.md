# tsc-marl

Code of the engineering thesis **"Wykorzystanie podejścia wieloagentowego w problemie
zarządzania sygnalizacją świetlną"** (Using a multi-agent approach for traffic signal
control): multi-agent traffic signal control with a graph neural network, evaluated
under detector (sensor) failure.

The road network is a graph; each intersection is an agent that weighs information
from its neighbours with graph attention (GAT). The policy is trained with PPO under
centralised training and decentralised execution (CTDE). On top of the trained policy
the repository adds robustness mechanisms as wrappers (fallback gates, action masking,
uncertainty-weighted attention) and a sensor failure layer that every controller reads
through, so all methods are compared on the same corrupted detectors.

Day-to-day commands: **[USAGE.md](USAGE.md)**.

---

## Contents
1. [Layout](#layout) - 2. [Installation](#installation) - 3. [Running](#running) -
4. [Experimental protocol](#experimental-protocol) - 5. [Runs and results](#runs-and-results) -
6. [Relation to LibSignal](#relation-to-libsignal) - 7. [Pitfalls](#pitfalls)

---

## Layout

```
tsc_marl/
  src/tsc_marl/            OWN CODE, layers from the lowest
    paths.py, seeding.py     repository layout, global seeding
    domain/                  formulas from the literature, framework-free
      ats.py                   Advanced Traffic State and pressure
      sensors.py               sensor failure layer (FailureSpec, SensorModel, FailureMix)
      demand.py                demand variants (out-of-distribution traffic)
    core/                    the controller: GAT + PPO + CTDE
      graph.py                 road-network graph and node-order contract
      features.py              observation and reward of the whole network
      nets.py                  GAT actor, centralised critic, ablation encoders
      ppo.py                   MAPPO update, GAE, minibatches
      loop.py                  environment loop (the simulator is injected)
      uw_attention.py          attention modulated by edge reliability
    libsignal/               bridge to the submodule, all contact with third-party code
      bridge.py                sys.path, SUMO_HOME, CWD, child-process environment
      session.py               a LibSignal process under a registered run
      experiment.py            LibSignal <-> runlog adapter, argument parsing, suites
      sumo_world.py            SUMO world with CityFlow's contract
      agents/                  Registry plugins: Efficient-MP, Advanced-MP, Advanced-CoLight
    uncertainty/             behaviour-cloning dataset and uncertainty heads
    robustness/              controllers and wrappers under sensor failure, plans
    analysis/                tables, reports and figures from stored runs
    runner/                  process entry points (`tsc ...`) and orchestration
  src/runlog/              domain-independent experiment store (own README)
  configs/
    tsc/                     method configs + thesis_protocol.yml (frozen protocol)
    sim/                     network definitions (CityFlow JiNan/HangZhou, SUMO Krakow)
    suites/                  training grids (what to compute)
    robustness/              evaluation plans under sensor failure
  data/raw_data/           networks and traffic used in the thesis
  tests/                   mirrors src/
  results/                 numbers for the thesis: CSV, generated reports, figures
  runs/                    (gitignored) runs: meta, config, metrics, checkpoints
  external/LibSignal/      submodule: LibSignal fork with engine patches
```

The split is deliberate on two levels.

**`src/` is our code, `external/` is third-party.** Everything added to LibSignal is
visible with one command, see [Relation to LibSignal](#relation-to-libsignal).

**Inside `src/tsc_marl/` dependencies point one way**, down the list above
(`tests/test_layers.py` enforces it). In particular `core/` never builds a simulator:
the `World` is injected by `runner/train_core.py`, so the core is tested without an
engine.

## Installation

Requirements: Python 3.10, `uv`, `cmake` + `g++` (CityFlow is built from source).

```bash
git clone --recursive git@github.com:malekkonrad/tsc_marl.git
cd tsc_marl
make setup     # submodule + venv + CPU dependencies
make check     # ~15 s; should print "Final Travel Time is 316.1114"
make test      # fast unit tests, no simulator
make test-all  # + tests that start a simulator (marker `slow`)
```

On a GPU machine use `make setup-gpu` (torch and torch-scatter for CUDA 12.1).
If you cloned without `--recursive`: `git submodule update --init --recursive`.

## Running

Everything goes through one entry point, `tsc` (`python -m tsc_marl` works before
the console script is installed):

```bash
tsc doctor                                                    # environment OK?
tsc train -a efficientmp -n cityflow_jinan3x4_4ph --seed 0    # a LibSignal agent
tsc core -n cityflow_jinan3x4_4ph --seed 0                    # the GAT+PPO+CTDE core
tsc suite run p1_gate                                         # a training grid
tsc degrade run t2_nomech                                     # evaluation under failure
tsc table / tsc degrade report t2_nomech                      # reports into results/
```

[USAGE.md](USAGE.md) lists all commands, grids and plans.

`uv run tsc ...` works too. The project is marked `[tool.uv] managed = false`,
because the environment is built by `make setup` (torch from the CPU or CUDA index,
`torch-scatter` from an extra), not by `uv sync`. Without the flag `uv run` would sync
`.venv` from `pyproject.toml` alone, swap the torch build and remove `torch-scatter`.

### Experiment grids and plans

What to compute is a file, not a line in the shell history:

- **training grids** in `configs/suites/*.yml` (`tsc suite run <name>`): an entry
  point, a protocol mode, a grid of networks and seeds and optional config overrides;
- **evaluation plans** in `configs/robustness/*.yml` (`tsc degrade run <name>`):
  frozen controllers x networks x seeds x failure scenarios. Each plan starts with
  its hypothesis and pass condition, written before the measurement.

Both skip points that already have a current result: the comparison goes by the
parameter fingerprint, so changing anything in a method config (or in `overrides`)
invalidates the earlier run. `--replace` forces a recomputation.

The `ngpu` field describes the **machine, not the experiment**. It lives in the
`command` section of the config, the only section the fingerprint does NOT cover, so
the same point computed on CPU and on GPU has the same `params_hash` and is one
experiment. It applies to `entry: core`; `entry: tsc` stays on CPU, because
LibSignal's `agent/colight.py` has no device handling.

**Do not call `python -m tsc_marl.runner.train` directly.** `PYTHONHASHSEED=0` must be
set before the interpreter starts; setting it from Python has no effect, and without it
`world_sumo` gives different results for the same arguments. The CLI restarts the
interpreter for that (`runner/cli.py:ensure_hashseed`), and grid children get the
variable from `libsignal.child_env()`.

## Experimental protocol

The protocol is **frozen** in [`configs/tsc/thesis_protocol.yml`](configs/tsc/thesis_protocol.yml)
and shared by all methods: with an inconsistent protocol method rankings can flip, so
the comparison stops meaning anything. Decision interval 15 s, episode 3600 s,
`PYTHONHASHSEED=0`.

The only parameter changed on purpose is the transition length, because **results
are reported in two modes**:

| mode | `yellow_length` | meaning |
|---|---|---|
| `realistic` | 5 s (yellow 3 + all-red 2) | physically correct, the thesis protocol |
| `benchmark` | 0 s | how the reference implementations compute their results |

The difference is not cosmetic: it changes ATT by up to ~19% (`results/t1_baseline.md`).
The papers declare a transition phase, but their code does not simulate it; with the
transition off our numbers match the literature (to ~1% on JiNan for FixedTime and
Efficient-MP). Both modes stay, and every result row carries its `yellow_length`.
The mode is chosen with `--yellow 0` or `mode: benchmark` in a grid, not with a copy
of the config.

## Runs and results

One run = one directory. There is no central results file, because runs execute in
parallel in separate processes; the index is computed by scanning.

```
runs/<experiment>/<run_id>/
    meta.json            # params, fingerprint, provenance (SHA of the repo AND the submodule), status
    config.resolved.yml  # config after merging includes and CLI overrides
    metrics.csv          # step,split,metric,value, per episode
    result.json          # ATT, att_last10, throughput, queue, delay
    engine/              # LibSignal artefacts: checkpoints, logs, replays
```

This is handled by [`src/runlog/`](src/runlog/README.md), a **separate tool,
independent of this thesis**: it imports nothing from `tsc_marl` and knows nothing
about traffic. Translation between it and LibSignal lives in
[`src/tsc_marl/libsignal/experiment.py`](src/tsc_marl/libsignal/experiment.py).
`runlog` enforces three invariants, each answering a bug that already happened in this
project: **run status** (an interrupted training does not count as computed),
**refusing silent overwrites** (the same id with other parameters is an error) and
a **config snapshot** (a result row cannot point to a file someone changed later).

`results/` holds the processed numbers that go into the thesis. Every report is
regenerated from `runs/` (`tsc table`, `tsc degrade report <plan>`, ...), never edited
by hand; [USAGE.md](USAGE.md) maps reports to commands. `t1_raw.csv` keeps one row per
run with the full protocol metadata and agent parameters: the columns `t_min`, `w1`,
`obs_length`, `drop_always_green` are not decoration, since without them an
Advanced-MP row is not reproducible.

Checkpoints and logs go to `runs/` and are **not versioned**.

## Relation to LibSignal

[LibSignal](https://github.com/DaRL-LibSignal/LibSignal) provides the environment and
the baseline agents. It is a submodule pointing to a fork with the `thesis` branch, so
**all changes to third-party code are visible with one command**:

```bash
git -C external/LibSignal diff --stat origin/master HEAD
```

The `thesis` branch holds only engine patches that cannot be made from outside:

- `world/world_cityflow.py`: `yellow_length` read from the config instead of a
  hard-coded 5 (otherwise `benchmark` mode was unreachable), plus optional ATT
  logging computed as in Advanced-XLight's `summary.py`;
- `world/world_sumo.py`: `sumo-gui` starts automatically, with an adjustable delay;
- `utils/logger.py`: `get_output_file_path` honours `logger.root_dir` (otherwise
  training artefacts landed in `data/output_data/`, among the versioned inputs);
- `agent/colight.py`, `agent/dqn.py`: `save_model` saves the **online** network
  (`self.model`), not `target_model`. The online network makes the decisions, so it
  produced the learning curve; the target is a copy up to two episodes old, and the
  frozen-policy evaluation would otherwise measure a different network. The rest of
  LibSignal already saves `self.model`.

The fork also carries a ruff reformat of the touched files. Everything else
(agents, configs, protocol) lives in this repository. This works because LibSignal's
Registry is filled by decorators at import time and config paths are resolved against
the CWD, so we run it from the repository root and it reads our `configs/` and `data/`
(see [`bridge.py`](src/tsc_marl/libsignal/bridge.py)).

The SUMO world is replaced too, but from outside: `libsignal/sumo_world.py` overrides
upstream `world_sumo.World` in the Registry, with phase transitions, ATT and a
neighbour graph that match CityFlow (see `data/raw_data/krakow_3w/README.md`).

The baselines `fixedtime`, `sotl`, `maxpressure`, `presslight`, `colight`, `mplight`,
`frap` come from LibSignal. `efficientmp`, `advancedmp` and `advancedcolight` are
**our implementations** following the papers: LibSignal has none of them, and its
`maxpressure` computes pressure from `lane_count` rather than from queues.

## Pitfalls

Things that cost time and are not obvious from the code:

- **LibSignal rewrites `configs/sim/*.cfg` on every run** (fresh replay log paths).
  `libsignal.sim_configs_untouched()` undoes it from a BYTE SNAPSHOT taken before the
  run, not with `git checkout -- configs/sim`: the checkout version also erased real,
  uncommitted fixes in those files. Two processes starting in the same second can read
  a half-written file, so grid starts are staggered by 15 s.
- **`update_target_rate` counts decisions, not simulation steps.** The default 10 from
  `base.yml` meant a target update every 10 decisions instead of the usual 500-10000,
  and CoLight first converged and then fell apart. It affects every DQN-class method,
  so the fix is in each of their configs.
- **ATT is reported as the mean of the last 10 test episodes**, not as a single test
  after training (the Advanced-CoLight protocol). With unstable learning the last
  episode, the mean of 10 and the best episode can differ several-fold.
- **SUMO only from pip.** `apt` on Ubuntu noble ships 1.18, pip's `libsumo` is 1.27,
  and the versions must match.
- **`SUMO_HOME` is required even for `-w cityflow`**: LibSignal imports SUMO at module
  level.
- **Every run is a separate subprocess**, because the Registry is global and cannot be
  reset cleanly between configurations in one process.
