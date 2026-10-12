# Phase 1 GATE: the core (GAT + PPO + CTDE) vs Advanced-CoLight

Generated: 2026-10-11 16:51. Suites `p1_reference` + `p1_gate`, realistic mode (yellow=5), protocol `configs/tsc/thesis_protocol.yml`.
ATT = mean of a run's last 10 test episodes; below, mean +- std over seeds.
**WARNING: runs from 2 different code versions**, an uncertain comparison; machine: `glider-B760-Pro-RS`.

| network | Advanced-CoLight | core (gatppo) | delta | p (Welch) | n | within tolerance |
|---|---:|---:|---:|---:|:-:|:-:|
| cityflow_jinan3x4_4ph | 293.5 +- 0.4 | 295.7 +- 1.0 | +0.74% | 0.004 | 5 / 5 | YES |
| cityflow_hangzhou4x4_4ph | 317.1 +- 0.2 | 317.2 +- 0.2 | +0.04% | 0.331 | 5 / 5 | YES |

## Per seed

| network | agent | ATT (last 10 tests) |
|---|---|---|
| cityflow_jinan3x4_4ph | Advanced-CoLight | s0 292.8, s1 293.8, s2 293.8, s3 293.5, s4 293.5 |
| cityflow_jinan3x4_4ph | core (gatppo) | s0 294.7, s1 296.1, s2 296.0, s3 294.7, s4 296.9 |
| cityflow_hangzhou4x4_4ph | Advanced-CoLight | s0 317.0, s1 317.0, s2 316.8, s3 317.2, s4 317.3 |
| cityflow_hangzhou4x4_4ph | core (gatppo) | s0 317.4, s1 317.4, s2 316.9, s3 317.1, s4 317.2 |

## Stability and cost

| network | agent | worst test (last 50) | convergence [episode] | throughput | queue | run time [s] | device |
|---|---|---:|---:|---:|---:|---:|---|
| cityflow_jinan3x4_4ph | Advanced-CoLight | 300.7 | 34-84 | 6173 | 7.86 | 3194 | cpu |
| cityflow_jinan3x4_4ph | core (gatppo) | 299.3 | 36-56 | 6174 | 8.03 | 1424 | cuda:0 |
| cityflow_hangzhou4x4_4ph | Advanced-CoLight | 349.7 | 27-39 | 2931 | 1.14 | 2867 | cpu |
| cityflow_hangzhou4x4_4ph | core (gatppo) | 318.4 | 27-32 | 2929 | 1.14 | 1121 | cuda:0 |

Convergence = the first episode from which a rolling 10 tests stays within 2% of the final result (range over seeds). The worst tail test catches one-off policy collapses the mean of 10 does not show.

## Learning curves

Figures in `results/figures/` (PNG here, PDF for the thesis), rebuilt together with this report.

![Test ATT per episode. Line = mean over seeds, band = min-max (n = 5). Top row: full range; bottom: zoom on the plateau with the gate threshold (ATT of Advanced-CoLight + 5%).](figures/p1_att_test.png)

*Test ATT per episode. Line = mean over seeds, band = min-max (n = 5). Top row: full range; bottom: zoom on the plateau with the gate threshold (ATT of Advanced-CoLight + 5%).*

![PPO update signals, core (gatppo), one update per episode. Rows = networks; line = mean over seeds, band = min-max (n = 5). Value loss on a log scale.](figures/p1_ppo_diagnostics.png)

*PPO update signals, core (gatppo), one update per episode. Rows = networks; line = mean over seeds, band = min-max (n = 5). Value loss on a log scale.*

![TD loss of Advanced-CoLight per episode (log scale; episodes before `learning_start` left out). Line = mean over seeds, band = min-max (n = 5).](figures/p1_colight_loss.png)

*TD loss of Advanced-CoLight per episode (log scale; episodes before `learning_start` left out). Line = mean over seeds, band = min-max (n = 5).*

## Exit criterion

The core within **5%** of Advanced-CoLight's ATT on every network (delta > 0 = the core is worse). The tolerance is the upper bound of the phrase in `p1_gate.yml` ("within a few percent"), stored as `GATE_TOLERANCE` in `analysis/report.py` AFTER seeing the results, hence the smallest tolerance at which the verdict holds is given below as well.

* **cityflow_jinan3x4_4ph**: a statistically distinguishable difference (Welch, p = 0.004) in favour of Advanced-CoLight; in practice 0.74% ATT.
* **cityflow_hangzhou4x4_4ph**: the methods are indistinguishable (Welch, p = 0.331); delta +0.04% is within the spread over seeds.

## Verdict

**Phase 1 GATE: PASSED.** Largest delta = +0.74%; the verdict holds at every tolerance >= 0.74%.

## Where the run time difference comes from

**The times CANNOT be compared directly: Advanced-CoLight ran on `cpu`, the core on `cuda:0`.** The simulation is the same on both paths (a training + a test episode of 3600 s), but the time difference mixes two causes these runs cannot separate:

* **Advanced-CoLight**: in `external/LibSignal/trainer/tsc_trainer.py` the training block (l. 172) sits at the `while` level, OUTSIDE `if i % action_interval == 0` (l. 139). With `update_model_rate: 1` (`configs/tsc/base.yml`) the condition holds at every simulator tick: after `learning_start` that is 3600 optimiser steps per episode, each with three GAT forward passes (target on b_tp, model twice on b_t).
* **core**: one PPO update per episode, `epochs x n_minibatches` = 10 x 4 = 40 optimiser steps (`configs/tsc/gatppo.yml`).
* **hardware**: Advanced-CoLight cannot move to a GPU (`p1_reference.yml`: the submodule has no `.cuda()`), the core ran on a GPU as `ngpu` in `p1_gate.yml` says. How much of the time difference is the card and how much the step count is unknown.

Conclusion for the thesis: the hardware-independent claim is ~90x fewer optimiser steps per episode at the same ATT. Do NOT cite run time as an advantage of the method without a core run on the CPU (`--ngpu -1`). Part of the difference is also an artefact of LibSignal's loop, deliberately left alone, since changing the cadence would change the baseline matched to the literature (`t1_baseline.md`).

## What this table does NOT say

* Run time is `duration_s` from `meta.json`, the WHOLE run. Neither path logs a split into simulation and update; the split above follows from the code, not a measurement.
* HangZhou separates methods poorly in realistic mode: the classic baselines land close together (`t1_baseline.md`, column "realistic"), and reproducing the literature on this network is weak. JiNan carries the burden of proof of the gate.
