# T2: plan `t2_krakow`

Mode realistic (yellow=5), 3 failure draws per scenario, evaluation of FROZEN controllers (`tsc degrade`). All read the same detector layer: MaxPressure computes its decision from the broken state vector, not the engine.

Cell = ATT [s], the mean over draws, then over policy seeds (+- std over seeds); in brackets delta relative to the same controller's clean result. **(!) = worse than FixedTime**, the only place the fallback's lower layer can gain.

## sumo_krakow3w_mid

FixedTime (independent of readings): **347.6 s**.

| scenario | core GAT+PPO+CTDE (n=3) | core with failures (DR) (n=3) | core + rule gate (n=3) | DR + rule gate (n=3) | Advanced-MP (n=1) | Efficient-MP (n=1) |
|---|---:|---:|---:|---:|---:|---:|
| `clean` | 376.3 +- 46.1 (!) | 358.7 +- 17.2 (!) | 361.1 +- 6.8 (!) | 359.3 +- 11.9 (!) | 396.9 (!) | 373.5 (!) |
| `gauss2` | 512.5 +- 8.0 (+36.2%) (!) | 516.3 +- 57.8 (+44.0%) (!) | 512.5 +- 8.0 (+41.9%) (!) | 516.3 +- 57.8 (+43.7%) (!) | 445.3 (+12.2%) (!) | 375.9 (+0.7%) (!) |
| `mask0p25` | 424.3 +- 56.8 (+12.7%) (!) | 373.7 +- 17.4 (+4.2%) (!) | 391.3 +- 11.1 (+8.4%) (!) | 375.4 +- 8.5 (+4.5%) (!) | 480.4 (+21.0%) (!) | 441.7 (+18.3%) (!) |
| `mask0p5` | 529.0 +- 85.4 (+40.6%) (!) | 410.8 +- 66.0 (+14.5%) (!) | 468.1 +- 16.2 (+29.6%) (!) | 409.7 +- 62.2 (+14.0%) (!) | 736.5 (+85.6%) (!) | 718.1 (+92.3%) (!) |
| `mask0p25_intersection` | 812.2 +- 218.6 (+115.8%) (!) | 414.7 +- 22.6 (+15.6%) (!) | 382.4 +- 32.3 (+5.9%) (!) | 357.2 +- 4.2 (-0.6%) (!) | 1020.1 (+157.0%) (!) | 1008.0 (+169.9%) (!) |
| `mask0p5_intersection` | 911.4 +- 250.9 (+142.2%) (!) | 416.7 +- 16.0 (+16.2%) (!) | 391.3 +- 38.7 (+8.3%) (!) | 357.7 +- 10.0 (-0.5%) (!) | 1164.6 (+193.4%) (!) | 1158.6 (+210.2%) (!) |
| `mask0p5_hold_t60` | 486.4 +- 19.6 (+29.2%) (!) | 422.0 +- 18.9 (+17.7%) (!) | 405.8 +- 14.8 (+12.4%) (!) | 395.7 +- 25.1 (+10.1%) (!) | 602.2 (+51.7%) (!) | 538.0 (+44.1%) (!) |
| `drift0p3` | 367.0 +- 9.6 (-2.5%) (!) | 357.8 +- 6.3 (-0.3%) (!) | 368.1 +- 7.3 (+1.9%) (!) | 358.7 +- 5.9 (-0.2%) (!) | 402.9 (+1.5%) (!) | 379.4 (+1.6%) (!) |

* **core GAT+PPO+CTDE**: worst scenario `mask0p5_intersection`: 911.4 s (+142.2%). Worse than FixedTime in 7 scenarios: `drift0p3`, `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **core with failures (DR)**: worst scenario `gauss2`: 516.3 s (+44.0%). Worse than FixedTime in 7 scenarios: `drift0p3`, `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **core + rule gate**: worst scenario `gauss2`: 512.5 s (+41.9%). Worse than FixedTime in 7 scenarios: `drift0p3`, `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **DR + rule gate**: worst scenario `gauss2`: 516.3 s (+43.7%). Worse than FixedTime in 7 scenarios: `drift0p3`, `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **Advanced-MP**: worst scenario `mask0p5_intersection`: 1164.6 s (+193.4%). Worse than FixedTime in 7 scenarios: `drift0p3`, `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **Efficient-MP**: worst scenario `mask0p5_intersection`: 1158.6 s (+210.2%). Worse than FixedTime in 7 scenarios: `drift0p3`, `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.


![degradation curves sumo_krakow3w_mid](figures/t2_krakow_sumo_krakow3w_mid.png)

## sumo_krakow3w_low

FixedTime (independent of readings): **297.9 s**.

| scenario | core GAT+PPO+CTDE (n=3) | core with failures (DR) (n=3) | core + rule gate (n=3) | DR + rule gate (n=3) | Advanced-MP (n=1) | Efficient-MP (n=1) |
|---|---:|---:|---:|---:|---:|---:|
| `clean` | 277.4 +- 0.7 | 281.4 +- 0.7 | 277.7 +- 0.6 | 281.4 +- 1.1 | 279.8 | 282.6 |
| `gauss2` | 364.0 +- 8.2 (+31.2%) (!) | 327.3 +- 20.2 (+16.3%) (!) | 364.0 +- 8.2 (+31.1%) (!) | 327.3 +- 20.2 (+16.3%) (!) | 338.8 (+21.1%) (!) | 320.4 (+13.4%) (!) |
| `mask0p25` | 292.0 +- 2.9 (+5.3%) | 286.3 +- 1.8 (+1.8%) | 289.9 +- 5.4 (+4.4%) | 286.8 +- 1.8 (+1.9%) | 313.3 (+12.0%) (!) | 312.1 (+10.4%) (!) |
| `mask0p5` | 351.1 +- 38.0 (+26.6%) (!) | 294.1 +- 5.4 (+4.5%) | 323.9 +- 16.7 (+16.6%) (!) | 291.8 +- 2.0 (+3.7%) | 561.7 (+100.7%) (!) | 556.5 (+96.9%) (!) |
| `mask0p25_intersection` | 658.7 +- 171.2 (+137.4%) (!) | 314.6 +- 13.0 (+11.8%) (!) | 299.3 +- 6.3 (+7.8%) (!) | 290.5 +- 2.4 (+3.2%) | 789.5 (+182.2%) (!) | 784.8 (+177.7%) (!) |
| `mask0p5_intersection` | 742.3 +- 203.1 (+167.6%) (!) | 329.4 +- 11.8 (+17.1%) (!) | 298.8 +- 4.6 (+7.6%) (!) | 293.3 +- 1.9 (+4.2%) | 936.4 (+234.7%) (!) | 935.8 (+231.2%) (!) |
| `mask0p5_hold_t60` | 319.4 +- 30.3 (+15.1%) (!) | 298.0 +- 16.6 (+5.9%) (!) | 304.5 +- 7.1 (+9.7%) (!) | 295.1 +- 8.4 (+4.9%) | 392.0 (+40.1%) (!) | 446.4 (+58.0%) (!) |
| `drift0p3` | 280.7 +- 3.9 (+1.2%) | 282.2 +- 0.8 (+0.3%) | 278.9 +- 0.8 (+0.4%) | 282.1 +- 0.7 (+0.2%) | 286.6 (+2.4%) | 284.3 (+0.6%) |

* **core GAT+PPO+CTDE**: worst scenario `mask0p5_intersection`: 742.3 s (+167.6%). Worse than FixedTime in 5 scenarios: `gauss2`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **core with failures (DR)**: worst scenario `mask0p5_intersection`: 329.4 s (+17.1%). Worse than FixedTime in 4 scenarios: `gauss2`, `mask0p25_intersection`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **core + rule gate**: worst scenario `gauss2`: 364.0 s (+31.1%). Worse than FixedTime in 5 scenarios: `gauss2`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **DR + rule gate**: worst scenario `gauss2`: 327.3 s (+16.3%). Worse than FixedTime in 1 scenarios: `gauss2`.
* **Advanced-MP**: worst scenario `mask0p5_intersection`: 936.4 s (+234.7%). Worse than FixedTime in 6 scenarios: `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **Efficient-MP**: worst scenario `mask0p5_intersection`: 935.8 s (+231.2%). Worse than FixedTime in 6 scenarios: `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.


![degradation curves sumo_krakow3w_low](figures/t2_krakow_sumo_krakow3w_low.png)

## sumo_krakow3w_high

FixedTime (independent of readings): **390.5 s**.

| scenario | core GAT+PPO+CTDE (n=3) | core with failures (DR) (n=3) | core + rule gate (n=3) | DR + rule gate (n=3) | Advanced-MP (n=1) | Efficient-MP (n=1) |
|---|---:|---:|---:|---:|---:|---:|
| `clean` | 459.1 +- 42.7 (!) | 442.5 +- 7.3 (!) | 432.0 +- 10.5 (!) | 445.0 +- 5.6 (!) | 481.5 (!) | 477.2 (!) |
| `gauss2` | 584.4 +- 17.3 (+27.3%) (!) | 591.1 +- 48.3 (+33.6%) (!) | 584.4 +- 17.3 (+35.3%) (!) | 591.1 +- 48.3 (+32.8%) (!) | 541.8 (+12.5%) (!) | 469.4 (-1.6%) (!) |
| `mask0p25` | 550.4 +- 72.4 (+19.9%) (!) | 457.2 +- 8.9 (+3.3%) (!) | 508.2 +- 45.9 (+17.6%) (!) | 473.7 +- 36.6 (+6.5%) (!) | 550.5 (+14.3%) (!) | 563.0 (+18.0%) (!) |
| `mask0p5` | 663.6 +- 70.1 (+44.5%) (!) | 514.7 +- 64.2 (+16.3%) (!) | 601.5 +- 98.0 (+39.2%) (!) | 506.1 +- 81.9 (+13.7%) (!) | 814.4 (+69.1%) (!) | 779.8 (+63.4%) (!) |
| `mask0p25_intersection` | 868.3 +- 165.7 (+89.1%) (!) | 483.8 +- 27.0 (+9.3%) (!) | 472.5 +- 20.7 (+9.4%) (!) | 446.1 +- 15.9 (+0.3%) (!) | 1082.4 (+124.8%) (!) | 1088.2 (+128.0%) (!) |
| `mask0p5_intersection` | 975.5 +- 231.3 (+112.5%) (!) | 527.6 +- 35.3 (+19.2%) (!) | 490.5 +- 39.2 (+13.6%) (!) | 450.1 +- 21.1 (+1.1%) (!) | 1230.8 (+155.6%) (!) | 1207.7 (+153.1%) (!) |
| `mask0p5_hold_t60` | 559.2 +- 55.8 (+21.8%) (!) | 501.9 +- 14.2 (+13.4%) (!) | 491.3 +- 4.3 (+13.7%) (!) | 498.8 +- 25.3 (+12.1%) (!) | 649.4 (+34.9%) (!) | 623.0 (+30.6%) (!) |
| `drift0p3` | 447.6 +- 13.4 (-2.5%) (!) | 474.4 +- 27.2 (+7.2%) (!) | 440.9 +- 5.2 (+2.1%) (!) | 478.5 +- 48.1 (+7.5%) (!) | 481.2 (-0.1%) (!) | 460.9 (-3.4%) (!) |

* **core GAT+PPO+CTDE**: worst scenario `mask0p5_intersection`: 975.5 s (+112.5%). Worse than FixedTime in 7 scenarios: `drift0p3`, `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **core with failures (DR)**: worst scenario `gauss2`: 591.1 s (+33.6%). Worse than FixedTime in 7 scenarios: `drift0p3`, `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **core + rule gate**: worst scenario `mask0p5`: 601.5 s (+39.2%). Worse than FixedTime in 7 scenarios: `drift0p3`, `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **DR + rule gate**: worst scenario `gauss2`: 591.1 s (+32.8%). Worse than FixedTime in 7 scenarios: `drift0p3`, `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **Advanced-MP**: worst scenario `mask0p5_intersection`: 1230.8 s (+155.6%). Worse than FixedTime in 7 scenarios: `drift0p3`, `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **Efficient-MP**: worst scenario `mask0p5_intersection`: 1207.7 s (+153.1%). Worse than FixedTime in 7 scenarios: `drift0p3`, `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.


![degradation curves sumo_krakow3w_high](figures/t2_krakow_sumo_krakow3w_high.png)

## Arm: core with failures (DR) vs core GAT+PPO+CTDE

The columns have DIFFERENT checkpoints (separate trainings), so pairing is by (network, scenario, seed, draw): traffic and failure pattern stay shared, the spread between seeds enters the difference. The reference column is set in the plan (`baseline: gatppo`), not assumed. `delta` < 0 = the arm is better. `|t| >= 2.8` is the threshold corrected for as many comparisons as this table has.

The columns differ in HOW THEY WERE TRAINED, not in a mechanism switched on at inference, so there is no "did the mechanism start" diagnostic. The spread between seeds enters the difference in full, and the number of INDEPENDENT replications is the number of seeds, not of pairs: failure draws share a checkpoint, so |t| is optimistic with respect to them.

### sumo_krakow3w_mid

| scenario | delta ATT [s] | sd | t | better in |
|---|---:|---:|---:|---:|
| `clean` | -17.67 | 44.7 | -0.69 | 2/3 |
| `gauss2` | +3.86 | 95.7 | +0.12 | 7/9 |
| `mask0p25` | -50.56 | 96.2 | -1.58 | 7/9 |
| `mask0p5` | -118.22 | 230.5 | -1.54 | 7/9 |
| `mask0p25_intersection` | **-397.54** | 323.1 | -3.69 | 7/9 |
| `mask0p5_intersection` | **-494.74** | 322.6 | -4.60 | 8/9 |
| `mask0p5_hold_t60` | **-64.41** | 51.6 | -3.75 | 8/9 |
| `drift0p3` | -9.28 | 22.7 | -1.22 | 6/9 |

* Above the threshold: `mask0p25_intersection` (-397.54 s, t = -3.69, better in 7/9 pairs), `mask0p5_hold_t60` (-64.41 s, t = -3.75, better in 8/9 pairs), `mask0p5_intersection` (-494.74 s, t = -4.60, better in 8/9 pairs)

### sumo_krakow3w_low

| scenario | delta ATT [s] | sd | t | better in |
|---|---:|---:|---:|---:|
| `clean` | **+3.94** | 1.4 | +4.72 | 0/3 |
| `gauss2` | **-36.69** | 17.7 | -6.20 | 9/9 |
| `mask0p25` | **-5.71** | 5.0 | -3.44 | 8/9 |
| `mask0p5` | -57.00 | 81.2 | -2.11 | 8/9 |
| `mask0p25_intersection` | **-344.11** | 259.2 | -3.98 | 9/9 |
| `mask0p5_intersection` | **-412.96** | 269.0 | -4.61 | 9/9 |
| `mask0p5_hold_t60` | -21.44 | 23.3 | -2.75 | 9/9 |
| `drift0p3` | +1.47 | 6.3 | +0.70 | 1/9 |

* Above the threshold: `clean` (+3.94 s, t = +4.72, better in 0/3 pairs), `gauss2` (-36.69 s, t = -6.20, better in 9/9 pairs), `mask0p25` (-5.71 s, t = -3.44, better in 8/9 pairs), `mask0p25_intersection` (-344.11 s, t = -3.98, better in 9/9 pairs), `mask0p5_intersection` (-412.96 s, t = -4.61, better in 9/9 pairs)

### sumo_krakow3w_high

| scenario | delta ATT [s] | sd | t | better in |
|---|---:|---:|---:|---:|
| `clean` | -16.67 | 48.4 | -0.60 | 1/3 |
| `gauss2` | +6.70 | 103.4 | +0.19 | 5/9 |
| `mask0p25` | -93.14 | 157.0 | -1.78 | 7/9 |
| `mask0p5` | -148.95 | 234.5 | -1.91 | 7/9 |
| `mask0p25_intersection` | **-384.48** | 296.0 | -3.90 | 9/9 |
| `mask0p5_intersection` | **-447.85** | 293.3 | -4.58 | 8/9 |
| `mask0p5_hold_t60` | -57.25 | 73.2 | -2.35 | 6/9 |
| `drift0p3` | +26.87 | 44.8 | +1.80 | 5/9 |

* Above the threshold: `mask0p25_intersection` (-384.48 s, t = -3.90, better in 9/9 pairs), `mask0p5_intersection` (-447.85 s, t = -4.58, better in 8/9 pairs)

## Mechanism: core + rule gate vs core GAT+PPO+CTDE

Both columns are computed from THE SAME checkpoint and the same failure draws (`draw` chooses which detectors failed), so they are compared **in pairs**, not through the +-std of the table above: the shared spread between seeds and draws cancels in the difference. `delta` < 0 = the mechanism is better. `|t| >= 2.8` is the threshold corrected for as many comparisons as this table has.

`gate_share` = share of (decision, node) pairs where the rule took control. `gate_switches` = number of mode CHANGES at nodes per episode; these hurt traffic (a switch resets the phase), not the share itself. `gate_precision` and `gate_recall` are relative to the ORACLE (the truth from the failure layer, diagnostic only) and separate "the detector is weak" from "switching gains little", two things ATT alone merges.

### sumo_krakow3w_mid

| scenario | gate_share | gate_switches | gate_precision | gate_recall | delta ATT [s] | sd | t | better in |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 0.1% | 2.00 | 0.00 | - | -15.21 | 48.5 | -0.54 | 2/3 |
| `gauss2` | 0.0% | 0.00 | - | - | +0.00 | 0.0 | - | 0/0 |
| `mask0p25` | 3.5% | 24.89 | 0.41 | 0.19 | -32.96 | 60.7 | -1.63 | 5/9 |
| `mask0p5` | 16.7% | 56.33 | 0.92 | 0.30 | -60.95 | 88.3 | -2.07 | 7/9 |
| `mask0p25_intersection` | 25.8% | 39.56 | 0.99 | 0.78 | **-429.83** | 333.8 | -3.86 | 9/9 |
| `mask0p5_intersection` | 38.1% | 43.78 | 1.00 | 0.81 | **-520.12** | 303.9 | -5.13 | 9/9 |
| `mask0p5_hold_t60` | 12.0% | 40.00 | 0.93 | 0.29 | **-80.57** | 59.5 | -4.07 | 8/9 |
| `drift0p3` | 0.2% | 2.89 | 0.00 | - | +1.03 | 15.6 | +0.20 | 3/9 |

* Above the threshold: `mask0p25_intersection` (-429.83 s, t = -3.86, better in 9/9 pairs), `mask0p5_hold_t60` (-80.57 s, t = -4.07, better in 8/9 pairs), `mask0p5_intersection` (-520.12 s, t = -5.13, better in 9/9 pairs)

### sumo_krakow3w_low

| scenario | gate_share | gate_switches | gate_precision | gate_recall | delta ATT [s] | sd | t | better in |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 2.3% | 17.33 | 0.00 | - | **+0.29** | 0.2 | +3.28 | 0/3 |
| `gauss2` | 0.0% | 0.00 | - | - | +0.00 | 0.0 | - | 0/0 |
| `mask0p25` | 5.5% | 46.00 | 0.18 | 0.11 | -2.14 | 4.7 | -1.38 | 7/9 |
| `mask0p5` | 12.7% | 69.56 | 0.88 | 0.22 | -27.22 | 47.6 | -1.72 | 7/9 |
| `mask0p25_intersection` | 28.0% | 58.78 | 0.92 | 0.79 | **-359.37** | 264.2 | -4.08 | 9/9 |
| `mask0p5_intersection` | 40.1% | 55.22 | 0.96 | 0.83 | **-443.53** | 268.8 | -4.95 | 9/9 |
| `mask0p5_hold_t60` | 10.1% | 54.89 | 0.80 | 0.21 | -14.87 | 45.2 | -0.99 | 6/9 |
| `drift0p3` | 2.3% | 21.00 | 0.00 | - | -1.82 | 6.3 | -0.87 | 4/9 |

* Above the threshold: `clean` (+0.29 s, t = +3.28, better in 0/3 pairs), `mask0p25_intersection` (-359.37 s, t = -4.08, better in 9/9 pairs), `mask0p5_intersection` (-443.53 s, t = -4.95, better in 9/9 pairs)

### sumo_krakow3w_high

| scenario | gate_share | gate_switches | gate_precision | gate_recall | delta ATT [s] | sd | t | better in |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 0.2% | 5.00 | 0.00 | - | -27.15 | 39.4 | -1.19 | 2/3 |
| `gauss2` | 0.0% | 0.00 | - | - | +0.00 | 0.0 | - | 0/0 |
| `mask0p25` | 3.5% | 25.89 | 0.52 | 0.22 | -42.12 | 112.4 | -1.12 | 5/9 |
| `mask0p5` | 15.9% | 59.89 | 0.92 | 0.28 | -62.15 | 248.4 | -0.75 | 7/9 |
| `mask0p25_intersection` | 23.7% | 51.44 | 0.99 | 0.70 | **-395.80** | 304.7 | -3.90 | 8/9 |
| `mask0p5_intersection` | 37.0% | 49.33 | 0.99 | 0.79 | **-484.93** | 258.6 | -5.63 | 9/9 |
| `mask0p5_hold_t60` | 11.2% | 43.56 | 0.90 | 0.26 | **-67.85** | 68.0 | -2.99 | 8/9 |
| `drift0p3` | 0.3% | 4.78 | 0.00 | - | -6.65 | 23.5 | -0.85 | 6/9 |

* Above the threshold: `mask0p25_intersection` (-395.80 s, t = -3.90, better in 8/9 pairs), `mask0p5_hold_t60` (-67.85 s, t = -2.99, better in 8/9 pairs), `mask0p5_intersection` (-484.93 s, t = -5.63, better in 9/9 pairs)

## Mechanism: DR + rule gate vs core with failures (DR)

Both columns are computed from THE SAME checkpoint and the same failure draws (`draw` chooses which detectors failed), so they are compared **in pairs**, not through the +-std of the table above: the shared spread between seeds and draws cancels in the difference. `delta` < 0 = the mechanism is better. `|t| >= 2.8` is the threshold corrected for as many comparisons as this table has.

`gate_share` = share of (decision, node) pairs where the rule took control. `gate_switches` = number of mode CHANGES at nodes per episode; these hurt traffic (a switch resets the phase), not the share itself. `gate_precision` and `gate_recall` are relative to the ORACLE (the truth from the failure layer, diagnostic only) and separate "the detector is weak" from "switching gains little", two things ATT alone merges.

### sumo_krakow3w_mid

| scenario | gate_share | gate_switches | gate_precision | gate_recall | delta ATT [s] | sd | t | better in |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 0.2% | 3.33 | 0.00 | - | +0.67 | 7.1 | +0.16 | 1/3 |
| `gauss2` | 0.0% | 0.00 | - | - | +0.00 | 0.0 | - | 0/0 |
| `mask0p25` | 3.4% | 23.00 | 0.46 | 0.20 | +1.74 | 15.5 | +0.34 | 3/9 |
| `mask0p5` | 16.1% | 46.00 | 0.94 | 0.29 | -1.05 | 24.9 | -0.13 | 5/9 |
| `mask0p25_intersection` | 24.4% | 40.56 | 0.99 | 0.73 | -57.48 | 79.2 | -2.18 | 7/9 |
| `mask0p5_intersection` | 37.6% | 41.89 | 1.00 | 0.80 | **-58.94** | 34.7 | -5.09 | 9/9 |
| `mask0p5_hold_t60` | 11.9% | 34.44 | 0.94 | 0.29 | -26.31 | 40.4 | -1.95 | 7/9 |
| `drift0p3` | 0.2% | 3.33 | 0.00 | - | +0.95 | 16.7 | +0.17 | 4/8 |

* Above the threshold: `mask0p5_intersection` (-58.94 s, t = -5.09, better in 9/9 pairs)

### sumo_krakow3w_low

| scenario | gate_share | gate_switches | gate_precision | gate_recall | delta ATT [s] | sd | t | better in |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 2.6% | 22.33 | 0.00 | - | +0.05 | 1.6 | +0.06 | 1/3 |
| `gauss2` | 0.0% | 0.00 | - | - | +0.00 | 0.0 | - | 0/0 |
| `mask0p25` | 5.3% | 44.00 | 0.19 | 0.11 | +0.46 | 1.3 | +1.02 | 4/9 |
| `mask0p5` | 12.9% | 65.11 | 0.88 | 0.22 | -2.31 | 6.9 | -1.00 | 3/9 |
| `mask0p25_intersection` | 26.7% | 51.44 | 0.92 | 0.76 | **-24.10** | 23.1 | -3.13 | 6/9 |
| `mask0p5_intersection` | 39.3% | 53.78 | 0.96 | 0.81 | **-36.09** | 12.5 | -8.65 | 9/9 |
| `mask0p5_hold_t60` | 10.0% | 57.33 | 0.80 | 0.21 | -2.82 | 26.2 | -0.32 | 5/9 |
| `drift0p3` | 2.5% | 22.89 | 0.00 | - | -0.11 | 0.9 | -0.38 | 3/9 |

* Above the threshold: `mask0p25_intersection` (-24.10 s, t = -3.13, better in 6/9 pairs), `mask0p5_intersection` (-36.09 s, t = -8.65, better in 9/9 pairs)

### sumo_krakow3w_high

| scenario | gate_share | gate_switches | gate_precision | gate_recall | delta ATT [s] | sd | t | better in |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 0.2% | 3.33 | 0.00 | - | +2.55 | 6.4 | +0.69 | 1/2 |
| `gauss2` | 0.0% | 0.00 | - | - | +0.00 | 0.0 | - | 0/0 |
| `mask0p25` | 3.3% | 23.56 | 0.46 | 0.19 | +16.52 | 28.6 | +1.74 | 4/9 |
| `mask0p5` | 15.4% | 46.56 | 0.95 | 0.28 | -8.58 | 54.1 | -0.48 | 6/9 |
| `mask0p25_intersection` | 22.5% | 38.67 | 0.99 | 0.68 | -37.65 | 50.1 | -2.25 | 7/9 |
| `mask0p5_intersection` | 36.0% | 41.11 | 1.00 | 0.77 | **-77.55** | 75.3 | -3.09 | 8/9 |
| `mask0p5_hold_t60` | 11.1% | 38.22 | 0.93 | 0.27 | -3.14 | 45.0 | -0.21 | 5/9 |
| `drift0p3` | 0.2% | 3.33 | 0.00 | - | +4.09 | 38.3 | +0.32 | 4/6 |

* Above the threshold: `mask0p5_intersection` (-77.55 s, t = -3.09, better in 8/9 pairs)

## Check: the evaluator measures the same policy as T1

The evaluator's clean result vs the last training test, 6 learned runs: largest difference **0.000 s**, consistent to the log's precision.

## What this table does NOT say

* Levels are DETECTOR levels. For `gauss` they transfer to the feature with a factor of ~0.8-0.9, `urand` is ~1.6x weaker at the same `level` (`results/sensor_severity.md`).
* A mask with `zero` substitution changes far fewer feature cells than the fraction of dead lanes (zero is the modal value); the same number does not mean the same thing on networks with different loads.
* `hold` scenarios start at decision 60 (minute 15), so the first quarter of the episode is clean; their delta is not directly comparable with a mask from the start.
