# Usage: everything through `uv run`

One entry point, `tsc`. Every command below is in its `uv run` form, without
activating the venv. `.venv/bin/tsc ...`, or `tsc ...` after
`source .venv/bin/activate`, works the same.

> **Not `uv sync`**: the project has `[tool.uv] managed = false` and `uv sync` will
> refuse. The environment is built by `make setup`. Details: [Why not `uv sync`](#why-not-uv-sync).

---

## First check the environment

```bash
uv run tsc doctor
```

Prints the paths, `SUMO_HOME`, `PYTHONHASHSEED` and the `torch` / `cityflow` /
`torch_geometric` versions. `torch` **must** end in `+cpu` (CPU machine) or `+cu121`
(GPU machine); a bare `2.4.1` means something reinstalled the environment.

```bash
uv run tsc --help              # list of commands
uv run tsc train --help        # all flags of a run
```

---

## A single run

```bash
# a LibSignal agent (a baseline, CoLight or Advanced-CoLight)
uv run tsc train -a advancedcolight -n cityflow_jinan3x4_4ph --seed 0

# the core: GAT + PPO + CTDE, own loop
uv run tsc core -n cityflow_jinan3x4_4ph --seed 0

# the core on the Krakow network (SUMO)
uv run tsc core -w sumo -n sumo_krakow3w_mid --seed 0

# evaluate a frozen LibSignal policy from a checkpoint (no training)
uv run tsc eval --load-run runs/t1/colight__cityflow_jinan3x4_4ph__s0_y5 \
                --checkpoint best --tag drop25
```

Common flags (`train` and `core` share them):

| flag | meaning |
|---|---|
| `-a`, `--agent` | name in the Registry (`core` defaults to `gatppo`) |
| `-w`, `--world` | `cityflow` (default) or `sumo` |
| `-n`, `--network` | a file in `configs/sim/` without `.cfg` |
| `--seed` | seed |
| `--yellow` | `5` = thesis protocol, `0` = paper mode |
| `--episodes`, `--steps` | shorten the run, **smoke tests only** |
| `--run-id` | own name of the run directory |
| `--replace` | recompute a run that already succeeded |
| `--ngpu` | `-1` = CPU, otherwise the GPU index |

Core smoke test, a few seconds:

```bash
uv run tsc core -n cityflow_jinan3x4_4ph --episodes 1 --steps 150 --run-id smoke --replace
```

Methods: `fixedtime`, `sotl`, `maxpressure`, `efficientmp`, `advancedmp`,
`advancedcolight`, `presslight`, `colight`, `mplight`, `frap`, `dqn`, `gatppo` (core).
Networks: `configs/sim/*.cfg` (JiNan 3x4, HangZhou 4x4 with traffic variants, Krakow
`sumo_krakow3w_{low,mid,high}`).

---

## Training grids

This is the normal way to compute; do not start ten runs by hand.

```bash
uv run tsc suite list                    # grids in configs/suites/
uv run tsc suite status p1_gate          # what is computed and what is missing; runs nothing
uv run tsc suite run p1_gate             # compute the missing points
uv run tsc suite run p1_gate --replace   # recompute everything
```

`suite run` skips points with a **current** result, so an interrupted computation
resumes with the same command. It is safe to run it repeatedly.

| name | what it computes |
|---|---|
| `t1_baselines` | methods without training x 2 networks, `yellow=5`: the reference point |
| `t1_paper` | the same at `yellow=0`, the mode the papers report in |
| `t1_learned` | CoLight, PressLight x 2 networks x 3 seeds, `yellow=5` (CPU) |
| `t1_learned_paper` | Advanced-CoLight, CoLight in paper mode (`yellow=0`, CPU) |
| `p1_reference` | Advanced-CoLight x 2 networks x 3 seeds: reference for the core |
| `p1_gate` | the core (GAT+PPO+CTDE) x 2 networks x 3 seeds |
| `p1_abl_meanpool` | ablation: mean-pool (GraphSAGE) actor instead of GAT |
| `p1_abl_nognn` | ablation: actor without neighbours (MLP) |
| `p1_abl_state_{queue,pressure,ep}` | ablation: state representation instead of ATS |
| `p1_fail` | the core trained with failures (DR), without UW-attention |
| `p1_uw`, `p1_uw_rule` | the core trained with failures and UW-attention (head / rule signal) |
| `k1_core`, `k1_fail` | Krakow (SUMO): the core on clean traffic / with failures |
| `k2_core`, `k2_fail` | the same, keeping the best checkpoint |

Points run **in parallel** with `-j`:

```bash
uv run tsc suite run t1_learned -j 4     # 4 subprocesses at once
```

Starts are staggered by 15 s (`--stagger`), because LibSignal rewrites
`configs/sim/*.cfg` while building the world, and two processes starting in the same
second can read a half-written file. A failure in the first 90 s is retried once.
Each child gets `OMP_NUM_THREADS = cores / j`, and its output goes to
`runs/_suite_logs/<suite>/`.

A new grid is a file in `configs/suites/`:

```yaml
entry: core                 # core -> runner.train_core, tsc -> runner.train
mode: realistic             # realistic = yellow 5, benchmark = yellow 0
timeout: 86400              # seconds per point
grid:
  network: [cityflow_jinan3x4_4ph, cityflow_hangzhou4x4_4ph]
  seed: [0, 1, 2]
overrides:                  # optional overrides of the agent config
  model: {NUM_HEADS: [8, 8]}
```

A grid **with** `overrides` appends its name to the `run_id` (so two different configs
do not fight over one directory); a grid **without** overrides hits the same ids as
manual runs and sees earlier results as done.

---

## Sensor failure: frozen controllers

A plan in `configs/robustness/<name>.yml`: networks, controllers (learned ones with
seeds and a parent training grid), failure scenarios and the number of draws. Each plan
starts with its hypothesis and pass condition, written before the measurement. One
point = controller x network x seed, all scenarios in one run `runs/t2/...`.

```bash
uv run tsc degrade status t2_nomech      # what is computed, which parents lack weights
uv run tsc degrade run t2_nomech -j 6    # compute the missing points
uv run tsc degrade report t2_nomech      # -> results/t2_nomech.md + results/figures/
```

Learned controllers take the **final weights** (or the best ones, if the plan says
so) of their training run in `runs/t1/`. A point without a weight file is skipped
with a message rather than failing the plan. The report checks that the evaluator's
clean result equals the last test of the training run; otherwise the evaluator would
measure a different policy than the training table.

| plan | question |
|---|---|
| `t2_nomech` | degradation of frozen controllers before any mechanism |
| `t2_gate` | fallback gate with an oracle: the ceiling of perfect detection |
| `t2_gate_real` | the rule gate: how much of the oracle ceiling a computable signal collects |
| `t2_gate_thr` | the rule threshold lowered |
| `t2_gate_dr` | gate x failure training (DR): do the two levers compose |
| `t2_mask` | action masking: a limit on service age |
| `t2_ablations` | does the core's advantage under failure come from attention |
| `t2_uw`, `t2_uw_tau`, `t2_uw_src`, `t2_uw_rule`, `t2_uw_channels` | UW-attention on a frozen policy |
| `t2_train_uw`, `t2_train_uw_rule` | policies trained with UW-attention |
| `t2_ood` | the epistemic branch on new traffic (`tsc degrade ood t2_ood`) |
| `t2_trace` | per-decision gate trace for the explanation figure |
| `t2_krakow`, `t2_krakow_best` | the final architecture on Krakow (SUMO) |

The gate trace figure:

```bash
uv run tsc degrade run t2_trace --trace -j 4
uv run python -m tsc_marl.analysis.gate_trace t2_trace   # -> results/gate_trace.md
```

---

## Uncertainty heads and gate thresholds (CPU, minutes)

```bash
uv run tsc dataset -n cityflow_jinan3x4_4ph --episodes 3   # BC dataset (state -> Advanced-MP action)
uv run tsc prototype                                        # decision head -> results/gate_2_0.md
uv run tsc obshead                                          # observation head -> results/gate_2_2.md
uv run tsc obshead --save --window 12 --persistence         # the head alone -> runs/heads/
uv run tsc severity                                         # failure severity -> results/sensor_severity.md
uv run tsc rulecal -n sumo_krakow3w_mid                     # rule threshold without a head
uv run tsc demand --factors 0.25,0.5,0.75                  # demand variants (OOD traffic)
```

OOD datasets for the epistemic test: `cityflow_jinan3x4_4ph_d0p{25,5,75}` (lower
demand), `cityflow_jinan3x4_4ph_r{2000,2500}` (another traffic pattern),
`cityflow_hangzhou4x4_4ph` (another topology); each must be dumped first with
`tsc dataset -n <network>`.

---

## Computing on another machine

Training the core needs a GPU (`entry: core` with `ngpu: "0"`); LibSignal agents,
evaluation plans, heads and reports run on CPU. Evaluation of learned controllers
needs the weights, so it runs where the weights are, or after copying them.

**The remote machine computes RUNS (`runs/`), reports are made locally (`results/`).**
`results/` is versioned, so generating it on the remote ends in a blocked `git pull`
("untracked working tree files would be overwritten"). A report takes seconds to
rebuild from `runs/`.

Do not compute the same `run_id` on both machines: when synchronising, one directory
overwrites the other. After computing remotely, copy the runs together with the
weights:

```bash
rsync -av remote:tsc_marl/runs/t1/ runs/t1/      # with weights (needed by degrade)
rsync -av remote:tsc_marl/runs/t2/ runs/t2/
rsync -av runs/heads/ remote:tsc_marl/runs/heads/   # heads + u_ref, the other way
```

An uncertainty head and its `u_ref` travel **together**: tau and the reference tables
are calibrated for THAT artefact, and a `u_ref` from another head is in other units
(`load_reference` checks the file fingerprint and refuses).

A run interrupted on a machine (a dead process left in status `running`) is recomputed
by the next `suite run` on that machine: `runlog` recognises it by the stored `pid`.

---

## Results

```bash
uv run tsc runs      # all runs: status, parameters, ATT
uv run tsc table     # T1 and core tables -> results/ (reads runs/, computes nothing)
uv run tsc plot      # learning curves -> runs/plots/ (also for a run in progress)
uv run tsc plot --split train
```

| report | command |
|---|---|
| `t1_baseline.md`, `t1_table.md`, `t1_raw.csv`, `t1_meta.json`, `p1_gate.md`, `p1_ablations.md` | `tsc table` |
| `t2_<plan>.md`, `t2_<plan>.csv` | `tsc degrade report <plan>` |
| `t2_ood_epistemic.md` | `tsc degrade ood t2_ood` |
| `gate_trace.md` | `python -m tsc_marl.analysis.gate_trace t2_trace` |
| `gate_2_0.md`, `gate_2_2.md`, `gate_2_2_masks.md`, `sensor_severity.md` | `tsc prototype`, `tsc obshead [--mask-variants]`, `tsc severity` |
| `t1_w1_sweep.md` | none (hand-written notes on a parameter sweep) |

---

## Shortcuts (Makefile)

Make targets call `.venv/bin/tsc` directly; `uv run` before `make` is not needed.

```bash
make check        # environment health: doctor + one Efficient-MP run (~15 s)
make gate         # p1_reference -> p1_gate -> status
make bench        # T1 grid, realistic mode + tables
make bench-paper  # the same in paper mode (yellow=0)
make test         # fast tests
make test-slow    # tests with a real simulator
make lint         # ruff check + format --check
```

Installing / rebuilding the environment:

```bash
make setup        # venv + CPU dependencies + LibSignal submodule
make setup-gpu    # the same with torch/torch-scatter for CUDA 12.1
```

---

## Why not `uv sync`

`uv sync` (and `uv run` without `managed = false`) syncs `.venv` from `pyproject.toml`
alone, without the indexes and extras that `make setup` passes. Concretely:

```
- torch==2.4.1+cpu             -> + torch==2.4.1     (CUDA build, ~2.5 GB of nvidia-* wheels)
- torch-scatter==2.1.2+pt24cpu -> removed            (lives in the `cpu`/`gpu` extra)
- ruff, pytest                 -> removed            (live in the `dev` extra)
```

One `uv run` would destroy the environment the results were produced in. Hence
`pyproject.toml` has `[tool.uv] managed = false`: `uv run` uses `.venv` without
syncing, and `uv sync` refuses with a readable message. `uv venv` and `uv pip` work
normally, since they do not go through project mode.

If the environment breaks anyway: `make setup` (or `make setup-gpu`) rebuilds it.

---

## One pitfall

**Do not call `python -m tsc_marl.runner.train` or `runner.train_core` directly.**

`PYTHONHASHSEED=0` must be set **before** the interpreter starts; setting it from
Python has no effect, and without it `world_sumo` gives different results for the same
arguments. `tsc` restarts itself for that (`cli.ensure_hashseed`), and subprocesses
of `tsc suite run` get the variable from `libsignal.child_env()`. Calling the module
directly bypasses both.
