# T2 for the policy trained with the mechanism: plan `t2_train_uw`

Mode realistic (yellow=5), 5 failure draws per scenario, evaluation of FROZEN controllers (`tsc degrade`). All read the same detector layer: MaxPressure computes its decision from the broken state vector, not the engine.

Cell = ATT [s], the mean over draws, then over policy seeds (+- std over seeds); in brackets delta relative to the same controller's clean result. **(!) = worse than FixedTime**, the only place the fallback's lower layer can gain.

## cityflow_jinan3x4_4ph

FixedTime (independent of readings): **507.0 s**.

| scenario | core GAT+PPO+CTDE (n=3) | core trained with failures (n=3) | core with failures + UW-attention (n=3) |
|---|---:|---:|---:|
| `clean` | 295.9 +- 0.6 | 302.0 +- 1.7 | 303.7 +- 1.1 |
| `gauss1` | 304.5 +- 1.4 (+2.9%) | 310.4 +- 1.2 (+2.8%) | 309.3 +- 1.4 (+1.8%) |
| `gauss2` | 319.9 +- 2.2 (+8.1%) | 321.8 +- 3.2 (+6.6%) | 321.6 +- 1.7 (+5.9%) |
| `gauss4` | 350.0 +- 4.5 (+18.3%) | 345.9 +- 8.4 (+14.5%) | 344.8 +- 1.9 (+13.5%) |
| `mask0p1` | 354.2 +- 20.0 (+19.7%) | 353.4 +- 9.8 (+17.0%) | 353.6 +- 4.1 (+16.4%) |
| `mask0p25` | 424.9 +- 31.1 (+43.6%) | 407.0 +- 12.5 (+34.8%) | 419.1 +- 13.3 (+38.0%) |
| `mask0p5` | 732.6 +- 28.5 (+147.5%) (!) | 592.5 +- 19.4 (+96.2%) (!) | 633.5 +- 41.4 (+108.6%) (!) |
| `mask0p25_intersection` | 471.1 +- 68.8 (+59.2%) | 389.2 +- 42.7 (+28.9%) | 395.2 +- 19.1 (+30.1%) |
| `mask0p5_intersection` | 690.0 +- 115.3 (+133.2%) (!) | 483.8 +- 47.4 (+60.2%) | 509.8 +- 39.1 (+67.8%) (!) |
| `mask0p25_hold_t60` | 354.7 +- 24.8 (+19.9%) | 334.4 +- 8.7 (+10.7%) | 337.6 +- 0.5 (+11.2%) |
| `mask0p5_hold_t60` | 475.0 +- 38.9 (+60.5%) | 473.6 +- 29.4 (+56.8%) | 473.2 +- 20.9 (+55.8%) |
| `drift0p3` | 301.5 +- 0.5 (+1.9%) | 309.3 +- 0.7 (+2.4%) | 309.1 +- 2.3 (+1.8%) |
| `drift0p5` | 328.6 +- 4.3 (+11.0%) | 339.9 +- 8.2 (+12.5%) | 336.6 +- 1.1 (+10.8%) |

* **core GAT+PPO+CTDE**: worst scenario `mask0p5`: 732.6 s (+147.5%). Worse than FixedTime in 2 scenarios: `mask0p5`, `mask0p5_intersection`.
* **core trained with failures**: worst scenario `mask0p5`: 592.5 s (+96.2%). Worse than FixedTime in 1 scenarios: `mask0p5`.
* **core with failures + UW-attention**: worst scenario `mask0p5`: 633.5 s (+108.6%). Worse than FixedTime in 2 scenarios: `mask0p5`, `mask0p5_intersection`.


![degradation curves cityflow_jinan3x4_4ph](figures/t2_train_uw_cityflow_jinan3x4_4ph.png)

## Arm: core trained with failures vs core GAT+PPO+CTDE

The columns have DIFFERENT checkpoints (separate trainings), so pairing is by (network, scenario, seed, draw): traffic and failure pattern stay shared, the spread between seeds enters the difference. The reference column is set in the plan (`baseline: gatppo`), not assumed. `delta` < 0 = the arm is better. `|t| >= 2.8` is the threshold corrected for as many comparisons as this table has.

The columns differ in HOW THEY WERE TRAINED, not in a mechanism switched on at inference, so there is no "did the mechanism start" diagnostic. The spread between seeds enters the difference in full, and the number of INDEPENDENT replications is the number of seeds, not of pairs: failure draws share a checkpoint, so |t| is optimistic with respect to them.

### cityflow_jinan3x4_4ph

| scenario | delta ATT [s] | sd | t | better in |
|---|---:|---:|---:|---:|
| `clean` | **+6.06** | 2.2 | +4.76 | 0/3 |
| `gauss1` | **+5.86** | 2.6 | +8.58 | 0/15 |
| `gauss2` | +1.94 | 4.7 | +1.59 | 6/15 |
| `gauss4` | -4.19 | 11.2 | -1.45 | 9/15 |
| `mask0p1` | -0.83 | 13.4 | -0.24 | 6/15 |
| `mask0p25` | -17.92 | 26.4 | -2.63 | 12/15 |
| `mask0p5` | **-140.03** | 50.3 | -10.77 | 15/15 |
| `mask0p25_intersection` | -81.83 | 115.0 | -2.76 | 12/15 |
| `mask0p5_intersection` | **-206.15** | 170.4 | -4.69 | 14/15 |
| `mask0p25_hold_t60` | -20.35 | 47.7 | -1.65 | 8/15 |
| `mask0p5_hold_t60` | -1.42 | 85.2 | -0.06 | 7/15 |
| `drift0p3` | **+7.77** | 2.0 | +14.93 | 0/15 |
| `drift0p5` | +11.27 | 19.3 | +2.27 | 1/15 |

* Above the threshold: `clean` (+6.06 s, t = +4.76, better in 0/3 pairs), `drift0p3` (+7.77 s, t = +14.93, better in 0/15 pairs), `gauss1` (+5.86 s, t = +8.58, better in 0/15 pairs), `mask0p5` (-140.03 s, t = -10.77, better in 15/15 pairs), `mask0p5_intersection` (-206.15 s, t = -4.69, better in 14/15 pairs)

## Mechanism: core with failures + UW-attention vs core trained with failures

The columns have DIFFERENT checkpoints (separate trainings), so pairing is by (network, scenario, seed, draw): traffic and failure pattern stay shared, the spread between seeds enters the difference. The reference column is set in the plan (`baseline: gatppo_fail`), not assumed. `delta` < 0 = the mechanism is better. `|t| >= 2.8` is the threshold corrected for as many comparisons as this table has.

`uw_spread` = mean log rho spread within a node's SOFTMAX GROUP, the only part of the signal attention sees (it is invariant to a constant shared by the group). 0 = the layer sees no difference between neighbours, ln 10 ~ 2.3 = they differ 10x. `uw_active` = share of edges with a nonzero log rho.

### cityflow_jinan3x4_4ph

| scenario | uw_spread | uw_active | delta ATT [s] | sd | t | better in |
|---|---:|---:|---:|---:|---:|---:|
| `clean` | 0.46 | 3.4% | +1.72 | 1.7 | +1.79 | 1/3 |
| `gauss1` | 0.79 | 9.0% | -1.04 | 2.0 | -2.03 | 10/15 |
| `gauss2` | 2.19 | 22.9% | -0.25 | 3.6 | -0.26 | 8/15 |
| `gauss4` | 6.00 | 43.8% | -1.08 | 7.6 | -0.55 | 5/15 |
| `mask0p1` | 0.65 | 4.7% | +0.25 | 25.8 | +0.04 | 6/15 |
| `mask0p25` | 0.84 | 6.1% | +12.09 | 29.5 | +1.59 | 7/15 |
| `mask0p5` | 0.88 | 7.4% | +40.95 | 73.4 | +2.16 | 5/15 |
| `mask0p25_intersection` | 0.76 | 6.6% | +5.90 | 82.3 | +0.28 | 7/15 |
| `mask0p5_intersection` | 0.85 | 8.6% | +25.91 | 113.9 | +0.88 | 6/15 |
| `mask0p25_hold_t60` | 0.51 | 4.2% | +3.26 | 14.1 | +0.89 | 6/15 |
| `mask0p5_hold_t60` | 1.04 | 7.8% | -0.33 | 80.6 | -0.02 | 8/15 |
| `drift0p3` | 0.55 | 4.0% | -0.14 | 3.0 | -0.18 | 9/15 |
| `drift0p5` | 0.76 | 5.1% | -3.22 | 21.6 | -0.58 | 9/15 |

* No scenario crosses the threshold: the mechanism is **neutral** for ATT.

## Check: the evaluator measures the same policy as T1

The evaluator's clean result vs the last training test, 6 learned runs: largest difference **0.000 s**, consistent to the log's precision.

## What this table does NOT say

* Levels are DETECTOR levels. For `gauss` they transfer to the feature with a factor of ~0.8-0.9, `urand` is ~1.6x weaker at the same `level` (`results/sensor_severity.md`).
* A mask with `zero` substitution changes far fewer feature cells than the fraction of dead lanes (zero is the modal value); the same number does not mean the same thing on networks with different loads.
* `hold` scenarios start at decision 60 (minute 15), so the first quarter of the episode is clean; their delta is not directly comparable with a mask from the start.
