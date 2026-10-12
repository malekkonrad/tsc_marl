# T2: plan `t2_train_uw_rule`

Mode realistic (yellow=5), 5 failure draws per scenario, evaluation of FROZEN controllers (`tsc degrade`). All read the same detector layer: MaxPressure computes its decision from the broken state vector, not the engine.

Cell = ATT [s], the mean over draws, then over policy seeds (+- std over seeds); in brackets delta relative to the same controller's clean result. **(!) = worse than FixedTime**, the only place the fallback's lower layer can gain.

## cityflow_jinan3x4_4ph

FixedTime (independent of readings): **507.0 s**.

| scenario | core GAT+PPO+CTDE (n=3) | core trained with failures (n=3) | gatppo_fail_uwinf (n=3) | gatppo_fail_uw_rule (n=3) | gatppo_fail_gate (n=3) | gatppo_fail_uw_rule_gate (n=3) |
|---|---:|---:|---:|---:|---:|---:|
| `clean` | 295.9 +- 0.6 | 302.0 +- 1.7 | 302.6 +- 1.3 | 299.9 +- 1.1 | 305.2 +- 1.0 | 302.7 +- 2.1 |
| `gauss1` | 304.5 +- 1.4 (+2.9%) | 310.4 +- 1.2 (+2.8%) | 310.4 +- 1.2 (+2.6%) | 308.6 +- 0.9 (+2.9%) | 310.4 +- 1.2 (+1.7%) | 308.6 +- 0.9 (+2.0%) |
| `gauss2` | 319.9 +- 2.2 (+8.1%) | 321.8 +- 3.2 (+6.6%) | 321.8 +- 3.2 (+6.4%) | 319.5 +- 1.0 (+6.5%) | 321.8 +- 3.2 (+5.4%) | 319.5 +- 1.0 (+5.6%) |
| `gauss4` | 350.0 +- 4.5 (+18.3%) | 345.9 +- 8.4 (+14.5%) | 345.9 +- 8.4 (+14.3%) | 342.1 +- 0.6 (+14.1%) | 345.9 +- 8.4 (+13.3%) | 342.1 +- 0.6 (+13.0%) |
| `mask0p1` | 354.2 +- 20.0 (+19.7%) | 353.4 +- 9.8 (+17.0%) | 353.4 +- 6.9 (+16.8%) | 359.9 +- 9.5 (+20.0%) | 345.8 +- 2.6 (+13.3%) | 348.7 +- 1.2 (+15.2%) |
| `mask0p25` | 424.9 +- 31.1 (+43.6%) | 407.0 +- 12.5 (+34.8%) | 405.8 +- 9.4 (+34.1%) | 412.5 +- 21.1 (+37.6%) | 388.4 +- 4.9 (+27.3%) | 393.5 +- 5.6 (+30.0%) |
| `mask0p5` | 732.6 +- 28.5 (+147.5%) (!) | 592.5 +- 19.4 (+96.2%) (!) | 600.4 +- 46.8 (+98.4%) (!) | 625.9 +- 17.0 (+108.7%) (!) | 481.1 +- 11.3 (+57.6%) | 480.8 +- 11.8 (+58.9%) |
| `mask0p25_intersection` | 471.1 +- 68.8 (+59.2%) | 389.2 +- 42.7 (+28.9%) | 391.1 +- 47.5 (+29.3%) | 354.1 +- 6.2 (+18.1%) | 357.0 +- 8.4 (+16.9%) | 354.1 +- 4.4 (+17.0%) |
| `mask0p5_intersection` | 690.0 +- 115.3 (+133.2%) (!) | 483.8 +- 47.4 (+60.2%) | 466.5 +- 47.5 (+54.2%) | 447.0 +- 26.7 (+49.0%) | 399.1 +- 4.7 (+30.7%) | 395.2 +- 1.6 (+30.6%) |
| `mask0p25_hold_t60` | 354.7 +- 24.8 (+19.9%) | 334.4 +- 8.7 (+10.7%) | 342.8 +- 10.7 (+13.3%) | 330.3 +- 8.0 (+10.1%) | 337.2 +- 6.1 (+10.5%) | 330.2 +- 5.0 (+9.1%) |
| `mask0p5_hold_t60` | 475.0 +- 38.9 (+60.5%) | 473.6 +- 29.4 (+56.8%) | 458.6 +- 8.7 (+51.5%) | 428.1 +- 29.5 (+42.7%) | 406.0 +- 10.6 (+33.0%) | 394.5 +- 9.3 (+30.3%) |
| `drift0p3` | 301.5 +- 0.5 (+1.9%) | 309.3 +- 0.7 (+2.4%) | 308.9 +- 1.2 (+2.1%) | 307.7 +- 1.0 (+2.6%) | 309.7 +- 1.6 (+1.4%) | 309.1 +- 1.8 (+2.1%) |
| `drift0p5` | 328.6 +- 4.3 (+11.0%) | 339.9 +- 8.2 (+12.5%) | 341.1 +- 8.2 (+12.7%) | 334.7 +- 1.3 (+11.6%) | 342.8 +- 3.9 (+12.3%) | 346.4 +- 1.9 (+14.4%) |

* **core GAT+PPO+CTDE**: worst scenario `mask0p5`: 732.6 s (+147.5%). Worse than FixedTime in 2 scenarios: `mask0p5`, `mask0p5_intersection`.
* **core trained with failures**: worst scenario `mask0p5`: 592.5 s (+96.2%). Worse than FixedTime in 1 scenarios: `mask0p5`.
* **gatppo_fail_uwinf**: worst scenario `mask0p5`: 600.4 s (+98.4%). Worse than FixedTime in 1 scenarios: `mask0p5`.
* **gatppo_fail_uw_rule**: worst scenario `mask0p5`: 625.9 s (+108.7%). Worse than FixedTime in 1 scenarios: `mask0p5`.
* **gatppo_fail_gate**: worst scenario `mask0p5`: 481.1 s (+57.6%). Never falls below FixedTime: at these levels a fallback to FixedTime has nothing to gain.
* **gatppo_fail_uw_rule_gate**: worst scenario `mask0p5`: 480.8 s (+58.9%). Never falls below FixedTime: at these levels a fallback to FixedTime has nothing to gain.


![degradation curves cityflow_jinan3x4_4ph](figures/t2_train_uw_rule_cityflow_jinan3x4_4ph.png)

## cityflow_hangzhou4x4_4ph

FixedTime (independent of readings): **568.9 s**.

| scenario | core GAT+PPO+CTDE (n=3) | core trained with failures (n=3) | gatppo_fail_uwinf (n=3) | gatppo_fail_uw_rule (n=3) | gatppo_fail_gate (n=3) | gatppo_fail_uw_rule_gate (n=3) |
|---|---:|---:|---:|---:|---:|---:|
| `clean` | 317.5 +- 1.0 | 321.2 +- 0.3 | 322.0 +- 1.0 | 321.6 +- 2.4 | 322.4 +- 0.2 | 321.2 +- 2.1 |
| `gauss1` | 344.9 +- 1.1 (+8.6%) | 341.9 +- 3.8 (+6.5%) | 341.9 +- 3.8 (+6.2%) | 341.4 +- 2.8 (+6.2%) | 341.9 +- 3.8 (+6.1%) | 341.4 +- 2.8 (+6.3%) |
| `gauss2` | 372.1 +- 4.2 (+17.2%) | 364.3 +- 10.4 (+13.4%) | 364.3 +- 10.4 (+13.1%) | 365.2 +- 9.0 (+13.6%) | 364.3 +- 10.4 (+13.0%) | 365.2 +- 9.0 (+13.7%) |
| `gauss4` | 419.3 +- 13.8 (+32.1%) | 401.6 +- 26.6 (+25.0%) | 401.6 +- 26.6 (+24.7%) | 407.3 +- 23.2 (+26.6%) | 401.6 +- 26.6 (+24.6%) | 407.3 +- 23.2 (+26.8%) |
| `mask0p1` | 363.4 +- 4.2 (+14.4%) | 345.1 +- 2.4 (+7.4%) | 346.6 +- 1.3 (+7.6%) | 366.1 +- 35.5 (+13.8%) | 345.4 +- 2.7 (+7.1%) | 348.7 +- 7.2 (+8.6%) |
| `mask0p25` | 460.1 +- 21.2 (+44.9%) | 394.1 +- 5.9 (+22.7%) | 392.7 +- 10.4 (+21.9%) | 410.0 +- 22.1 (+27.5%) | 395.7 +- 7.6 (+22.8%) | 402.2 +- 10.6 (+25.2%) |
| `mask0p5` | 739.2 +- 98.5 (+132.8%) (!) | 471.1 +- 36.5 (+46.7%) | 472.2 +- 40.1 (+46.6%) | 469.2 +- 21.5 (+45.9%) | 472.9 +- 29.7 (+46.7%) | 472.5 +- 22.8 (+47.1%) |
| `mask0p25_intersection` | 472.2 +- 58.9 (+48.7%) | 361.1 +- 9.4 (+12.4%) | 356.2 +- 10.4 (+10.6%) | 362.7 +- 23.4 (+12.8%) | 352.9 +- 3.1 (+9.5%) | 348.7 +- 7.1 (+8.6%) |
| `mask0p5_intersection` | 765.9 +- 188.2 (+141.2%) (!) | 414.7 +- 1.0 (+29.1%) | 405.5 +- 6.2 (+25.9%) | 430.8 +- 34.7 (+34.0%) | 392.3 +- 7.0 (+21.7%) | 394.2 +- 24.7 (+22.7%) |
| `mask0p25_hold_t60` | 411.1 +- 16.8 (+29.5%) | 365.3 +- 7.3 (+13.7%) | 391.0 +- 6.1 (+21.4%) | 384.9 +- 32.4 (+19.7%) | 372.7 +- 11.7 (+15.6%) | 386.7 +- 20.1 (+20.4%) |
| `mask0p5_hold_t60` | 544.4 +- 11.8 (+71.5%) | 442.6 +- 30.1 (+37.8%) | 525.4 +- 44.0 (+63.1%) | 505.7 +- 26.1 (+57.2%) | 452.4 +- 38.2 (+40.3%) | 458.8 +- 26.2 (+42.9%) |
| `drift0p3` | 319.5 +- 0.3 (+0.6%) | 323.5 +- 0.4 (+0.7%) | 323.8 +- 0.7 (+0.5%) | 323.4 +- 2.0 (+0.5%) | 324.4 +- 0.6 (+0.6%) | 323.8 +- 1.8 (+0.8%) |
| `drift0p5` | 339.9 +- 1.1 (+7.0%) | 334.3 +- 0.8 (+4.1%) | 334.5 +- 2.1 (+3.9%) | 337.7 +- 1.4 (+5.0%) | 335.7 +- 1.0 (+4.1%) | 339.1 +- 3.3 (+5.6%) |

* **core GAT+PPO+CTDE**: worst scenario `mask0p5_intersection`: 765.9 s (+141.2%). Worse than FixedTime in 2 scenarios: `mask0p5`, `mask0p5_intersection`.
* **core trained with failures**: worst scenario `mask0p5`: 471.1 s (+46.7%). Never falls below FixedTime: at these levels a fallback to FixedTime has nothing to gain.
* **gatppo_fail_uwinf**: worst scenario `mask0p5_hold_t60`: 525.4 s (+63.1%). Never falls below FixedTime: at these levels a fallback to FixedTime has nothing to gain.
* **gatppo_fail_uw_rule**: worst scenario `mask0p5_hold_t60`: 505.7 s (+57.2%). Never falls below FixedTime: at these levels a fallback to FixedTime has nothing to gain.
* **gatppo_fail_gate**: worst scenario `mask0p5`: 472.9 s (+46.7%). Never falls below FixedTime: at these levels a fallback to FixedTime has nothing to gain.
* **gatppo_fail_uw_rule_gate**: worst scenario `mask0p5`: 472.5 s (+47.1%). Never falls below FixedTime: at these levels a fallback to FixedTime has nothing to gain.


![degradation curves cityflow_hangzhou4x4_4ph](figures/t2_train_uw_rule_cityflow_hangzhou4x4_4ph.png)

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

### cityflow_hangzhou4x4_4ph

| scenario | delta ATT [s] | sd | t | better in |
|---|---:|---:|---:|---:|
| `clean` | **+3.66** | 0.7 | +8.57 | 0/3 |
| `gauss1` | **-2.95** | 2.8 | -4.02 | 13/15 |
| `gauss2` | **-7.76** | 6.5 | -4.61 | 12/15 |
| `gauss4` | **-17.78** | 13.1 | -5.26 | 14/15 |
| `mask0p1` | **-18.30** | 18.1 | -3.92 | 10/15 |
| `mask0p25` | **-66.06** | 68.9 | -3.71 | 14/15 |
| `mask0p5` | **-268.04** | 157.6 | -6.59 | 15/15 |
| `mask0p25_intersection` | **-111.06** | 152.9 | -2.81 | 12/15 |
| `mask0p5_intersection` | **-351.20** | 230.9 | -5.89 | 15/15 |
| `mask0p25_hold_t60` | **-45.85** | 51.5 | -3.45 | 14/15 |
| `mask0p5_hold_t60` | **-101.78** | 78.1 | -5.05 | 13/15 |
| `drift0p3` | **+3.98** | 1.0 | +15.70 | 0/15 |
| `drift0p5` | -5.56 | 12.2 | -1.77 | 7/15 |

* Above the threshold: `clean` (+3.66 s, t = +8.57, better in 0/3 pairs), `drift0p3` (+3.98 s, t = +15.70, better in 0/15 pairs), `gauss1` (-2.95 s, t = -4.02, better in 13/15 pairs), `gauss2` (-7.76 s, t = -4.61, better in 12/15 pairs), `gauss4` (-17.78 s, t = -5.26, better in 14/15 pairs), `mask0p1` (-18.30 s, t = -3.92, better in 10/15 pairs), `mask0p25` (-66.06 s, t = -3.71, better in 14/15 pairs), `mask0p25_hold_t60` (-45.85 s, t = -3.45, better in 14/15 pairs), `mask0p25_intersection` (-111.06 s, t = -2.81, better in 12/15 pairs), `mask0p5` (-268.04 s, t = -6.59, better in 15/15 pairs), `mask0p5_hold_t60` (-101.78 s, t = -5.05, better in 13/15 pairs), `mask0p5_intersection` (-351.20 s, t = -5.89, better in 15/15 pairs)

## Mechanism: gatppo_fail_uwinf vs core trained with failures

Both columns are computed from THE SAME checkpoint and the same failure draws (`draw` chooses which detectors failed), so they are compared **in pairs**, not through the +-std of the table above: the shared spread between seeds and draws cancels in the difference. `delta` < 0 = the mechanism is better. `|t| >= 2.8` is the threshold corrected for as many comparisons as this table has.

`uw_spread` = mean log rho spread within a node's SOFTMAX GROUP, the only part of the signal attention sees (it is invariant to a constant shared by the group). 0 = the layer sees no difference between neighbours, ln 10 ~ 2.3 = they differ 10x. `uw_active` = share of edges with a nonzero log rho.

### cityflow_jinan3x4_4ph

| scenario | uw_spread | uw_active | delta ATT [s] | sd | t | better in |
|---|---:|---:|---:|---:|---:|---:|
| `clean` | 0.02 | 0.8% | +0.60 | 1.1 | +0.94 | 1/3 |
| `gauss1` | 0.00 | 0.0% | +0.00 | 0.0 | - | 0/0 |
| `gauss2` | 0.00 | 0.0% | +0.00 | 0.0 | - | 0/0 |
| `gauss4` | 0.00 | 0.0% | +0.00 | 0.0 | - | 0/0 |
| `mask0p1` | 0.15 | 4.8% | -0.01 | 13.6 | -0.00 | 6/15 |
| `mask0p25` | 0.43 | 13.1% | -1.23 | 13.1 | -0.36 | 9/15 |
| `mask0p5` | 1.42 | 57.9% | +7.83 | 31.9 | +0.95 | 6/15 |
| `mask0p25_intersection` | 1.43 | 25.2% | +1.87 | 17.6 | +0.41 | 9/15 |
| `mask0p5_intersection` | 2.26 | 49.4% | -17.31 | 40.3 | -1.66 | 9/15 |
| `mask0p25_hold_t60` | 0.32 | 9.0% | +8.44 | 24.1 | +1.36 | 8/15 |
| `mask0p5_hold_t60` | 1.03 | 41.4% | -15.01 | 70.5 | -0.82 | 8/15 |
| `drift0p3` | 0.02 | 1.1% | -0.35 | 2.5 | -0.55 | 9/15 |
| `drift0p5` | 0.15 | 4.5% | +1.24 | 15.3 | +0.31 | 5/15 |

* No scenario crosses the threshold: the mechanism is **neutral** for ATT.

### cityflow_hangzhou4x4_4ph

| scenario | uw_spread | uw_active | delta ATT [s] | sd | t | better in |
|---|---:|---:|---:|---:|---:|---:|
| `clean` | 0.10 | 1.2% | +0.86 | 0.6 | +2.36 | 0/3 |
| `gauss1` | 0.00 | 0.0% | +0.00 | 0.0 | - | 0/0 |
| `gauss2` | 0.00 | 0.0% | +0.00 | 0.0 | - | 0/0 |
| `gauss4` | 0.00 | 0.0% | +0.00 | 0.0 | - | 0/0 |
| `mask0p1` | 0.11 | 1.4% | +1.50 | 6.4 | +0.91 | 5/15 |
| `mask0p25` | 0.18 | 2.2% | -1.42 | 8.4 | -0.65 | 9/15 |
| `mask0p5` | 1.05 | 9.6% | +1.08 | 12.0 | +0.35 | 8/15 |
| `mask0p25_intersection` | 0.54 | 4.6% | -4.91 | 18.7 | -1.02 | 11/15 |
| `mask0p5_intersection` | 1.45 | 14.0% | -9.24 | 23.9 | -1.50 | 9/15 |
| `mask0p25_hold_t60` | 0.14 | 1.5% | **+25.74** | 33.9 | +2.94 | 3/15 |
| `mask0p5_hold_t60` | 0.78 | 6.5% | **+82.77** | 94.4 | +3.40 | 3/15 |
| `drift0p3` | 0.10 | 1.2% | +0.28 | 0.7 | +1.51 | 5/15 |
| `drift0p5` | 0.10 | 1.2% | +0.13 | 2.6 | +0.20 | 6/15 |

* Above the threshold: `mask0p25_hold_t60` (+25.74 s, t = +2.94, better in 3/15 pairs), `mask0p5_hold_t60` (+82.77 s, t = +3.40, better in 3/15 pairs)

## Mechanism: gatppo_fail_uw_rule vs gatppo_fail_uwinf

The columns have DIFFERENT checkpoints (separate trainings), so pairing is by (network, scenario, seed, draw): traffic and failure pattern stay shared, the spread between seeds enters the difference. The reference column is set in the plan (`baseline: gatppo_fail_uwinf`), not assumed. `delta` < 0 = the mechanism is better. `|t| >= 2.8` is the threshold corrected for as many comparisons as this table has.

`uw_spread` = mean log rho spread within a node's SOFTMAX GROUP, the only part of the signal attention sees (it is invariant to a constant shared by the group). 0 = the layer sees no difference between neighbours, ln 10 ~ 2.3 = they differ 10x. `uw_active` = share of edges with a nonzero log rho.

### cityflow_jinan3x4_4ph

| scenario | uw_spread | uw_active | delta ATT [s] | sd | t | better in |
|---|---:|---:|---:|---:|---:|---:|
| `clean` | 0.02 | 1.0% | -2.68 | 1.7 | -2.74 | 3/3 |
| `gauss1` | 0.00 | 0.0% | -1.79 | 2.5 | -2.77 | 12/15 |
| `gauss2` | 0.00 | 0.0% | **-2.32** | 2.9 | -3.11 | 11/15 |
| `gauss4` | 0.00 | 0.0% | -3.75 | 8.1 | -1.78 | 8/15 |
| `mask0p1` | 0.19 | 5.5% | +6.55 | 25.8 | +0.98 | 10/15 |
| `mask0p25` | 0.46 | 14.2% | +6.74 | 41.0 | +0.64 | 8/15 |
| `mask0p5` | 1.46 | 60.9% | +25.54 | 73.3 | +1.35 | 5/15 |
| `mask0p25_intersection` | 1.41 | 24.1% | -37.00 | 81.5 | -1.76 | 12/15 |
| `mask0p5_intersection` | 2.27 | 50.5% | -19.57 | 88.5 | -0.86 | 8/15 |
| `mask0p25_hold_t60` | 0.31 | 8.9% | -12.51 | 33.3 | -1.46 | 8/15 |
| `mask0p5_hold_t60` | 0.99 | 38.7% | -30.45 | 67.8 | -1.74 | 10/15 |
| `drift0p3` | 0.02 | 1.1% | -1.23 | 2.2 | -2.18 | 10/15 |
| `drift0p5` | 0.14 | 4.3% | -6.36 | 17.9 | -1.38 | 10/15 |

* Above the threshold: `gauss2` (-2.32 s, t = -3.11, better in 11/15 pairs)

### cityflow_hangzhou4x4_4ph

| scenario | uw_spread | uw_active | delta ATT [s] | sd | t | better in |
|---|---:|---:|---:|---:|---:|---:|
| `clean` | 0.09 | 1.1% | -0.44 | 2.7 | -0.29 | 2/3 |
| `gauss1` | 0.00 | 0.0% | -0.55 | 2.3 | -0.94 | 10/15 |
| `gauss2` | 0.00 | 0.0% | +0.93 | 3.6 | +1.00 | 8/15 |
| `gauss4` | 0.00 | 0.0% | +5.69 | 8.1 | +2.72 | 4/15 |
| `mask0p1` | 0.11 | 1.4% | +19.54 | 63.7 | +1.19 | 7/15 |
| `mask0p25` | 0.18 | 2.3% | +17.34 | 43.2 | +1.55 | 7/15 |
| `mask0p5` | 1.05 | 9.6% | -3.00 | 80.4 | -0.14 | 6/15 |
| `mask0p25_intersection` | 0.53 | 4.6% | +6.51 | 46.1 | +0.55 | 6/15 |
| `mask0p5_intersection` | 1.47 | 14.4% | +25.34 | 46.3 | +2.12 | 6/15 |
| `mask0p25_hold_t60` | 0.13 | 1.4% | -6.13 | 52.7 | -0.45 | 9/15 |
| `mask0p5_hold_t60` | 0.71 | 5.8% | -19.74 | 60.3 | -1.27 | 9/15 |
| `drift0p3` | 0.09 | 1.1% | -0.45 | 2.3 | -0.75 | 10/15 |
| `drift0p5` | 0.10 | 1.2% | +3.21 | 4.5 | +2.77 | 2/15 |

* No scenario crosses the threshold: the mechanism is **neutral** for ATT.

## Mechanism: gatppo_fail_gate vs core trained with failures

Both columns are computed from THE SAME checkpoint and the same failure draws (`draw` chooses which detectors failed), so they are compared **in pairs**, not through the +-std of the table above: the shared spread between seeds and draws cancels in the difference. `delta` < 0 = the mechanism is better. `|t| >= 2.8` is the threshold corrected for as many comparisons as this table has.

`gate_share` = share of (decision, node) pairs where the rule took control. `gate_switches` = number of mode CHANGES at nodes per episode; these hurt traffic (a switch resets the phase), not the share itself. `gate_precision` and `gate_recall` are relative to the ORACLE (the truth from the failure layer, diagnostic only) and separate "the detector is weak" from "switching gains little", two things ATT alone merges.

### cityflow_jinan3x4_4ph

| scenario | gate_share | gate_switches | gate_precision | gate_recall | delta ATT [s] | sd | t | better in |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 1.2% | 9.33 | 0.00 | - | **+3.25** | 0.9 | +6.08 | 0/3 |
| `gauss1` | 0.0% | 0.00 | - | - | +0.00 | 0.0 | - | 0/0 |
| `gauss2` | 0.0% | 0.00 | - | - | +0.00 | 0.0 | - | 0/0 |
| `gauss4` | 0.0% | 0.00 | - | - | +0.00 | 0.0 | - | 0/0 |
| `mask0p1` | 6.1% | 19.67 | 0.02 | 0.03 | -7.62 | 22.3 | -1.32 | 9/15 |
| `mask0p25` | 15.3% | 31.60 | 0.16 | 0.30 | -18.58 | 27.7 | -2.60 | 12/15 |
| `mask0p5` | 55.8% | 35.20 | 0.74 | 0.69 | **-111.40** | 51.1 | -8.45 | 15/15 |
| `mask0p25_intersection` | 25.9% | 18.20 | 0.90 | 0.90 | -32.29 | 51.8 | -2.41 | 9/15 |
| `mask0p5_intersection` | 48.9% | 23.07 | 0.93 | 0.95 | **-84.78** | 61.0 | -5.38 | 15/15 |
| `mask0p25_hold_t60` | 10.2% | 21.33 | 0.15 | 0.25 | +2.82 | 22.8 | +0.48 | 7/15 |
| `mask0p5_hold_t60` | 40.5% | 27.13 | 0.73 | 0.66 | **-67.52** | 67.0 | -3.90 | 13/15 |
| `drift0p3` | 1.6% | 12.27 | 0.00 | - | +0.41 | 2.0 | +0.79 | 7/15 |
| `drift0p5` | 5.3% | 15.47 | 0.00 | - | +2.97 | 16.5 | +0.70 | 5/15 |

* Above the threshold: `clean` (+3.25 s, t = +6.08, better in 0/3 pairs), `mask0p5` (-111.40 s, t = -8.45, better in 15/15 pairs), `mask0p5_hold_t60` (-67.52 s, t = -3.90, better in 13/15 pairs), `mask0p5_intersection` (-84.78 s, t = -5.38, better in 15/15 pairs)

### cityflow_hangzhou4x4_4ph

| scenario | gate_share | gate_switches | gate_precision | gate_recall | delta ATT [s] | sd | t | better in |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 1.4% | 14.67 | 0.00 | - | **+1.19** | 0.2 | +13.52 | 0/3 |
| `gauss1` | 0.0% | 0.00 | - | - | +0.00 | 0.0 | - | 0/0 |
| `gauss2` | 0.0% | 0.00 | - | - | +0.00 | 0.0 | - | 0/0 |
| `gauss4` | 0.0% | 0.00 | - | - | +0.00 | 0.0 | - | 0/0 |
| `mask0p1` | 1.9% | 16.27 | 0.00 | - | +0.29 | 6.3 | +0.18 | 7/15 |
| `mask0p25` | 2.8% | 21.80 | 0.23 | 0.06 | +1.65 | 14.2 | +0.45 | 4/15 |
| `mask0p5` | 10.3% | 45.33 | 0.88 | 0.14 | +1.78 | 13.9 | +0.50 | 7/15 |
| `mask0p25_intersection` | 5.2% | 26.07 | 0.72 | 0.19 | -8.16 | 12.1 | -2.62 | 10/15 |
| `mask0p5_intersection` | 15.0% | 42.27 | 0.94 | 0.29 | **-22.38** | 27.9 | -3.10 | 13/15 |
| `mask0p25_hold_t60` | 1.6% | 16.13 | 0.02 | 0.00 | +7.40 | 37.6 | +0.76 | 6/15 |
| `mask0p5_hold_t60` | 6.0% | 33.87 | 0.65 | 0.09 | +9.78 | 46.3 | +0.82 | 6/15 |
| `drift0p3` | 1.4% | 14.27 | 0.00 | - | **+0.85** | 1.1 | +3.07 | 2/15 |
| `drift0p5` | 1.5% | 14.27 | 0.00 | - | +1.32 | 2.7 | +1.89 | 3/15 |

* Above the threshold: `clean` (+1.19 s, t = +13.52, better in 0/3 pairs), `drift0p3` (+0.85 s, t = +3.07, better in 2/15 pairs), `mask0p5_intersection` (-22.38 s, t = -3.10, better in 13/15 pairs)

## Mechanism: gatppo_fail_uw_rule_gate vs gatppo_fail_gate

The columns have DIFFERENT checkpoints (separate trainings), so pairing is by (network, scenario, seed, draw): traffic and failure pattern stay shared, the spread between seeds enters the difference. The reference column is set in the plan (`baseline: gatppo_fail_gate`), not assumed. `delta` < 0 = the mechanism is better. `|t| >= 2.8` is the threshold corrected for as many comparisons as this table has.

`gate_share` = share of (decision, node) pairs where the rule took control. `gate_switches` = number of mode CHANGES at nodes per episode; these hurt traffic (a switch resets the phase), not the share itself. `gate_precision` and `gate_recall` are relative to the ORACLE (the truth from the failure layer, diagnostic only) and separate "the detector is weak" from "switching gains little", two things ATT alone merges. `uw_spread` = mean log rho spread within a node's SOFTMAX GROUP, the only part of the signal attention sees (it is invariant to a constant shared by the group). 0 = the layer sees no difference between neighbours, ln 10 ~ 2.3 = they differ 10x. `uw_active` = share of edges with a nonzero log rho.

### cityflow_jinan3x4_4ph

| scenario | gate_share | gate_switches | gate_precision | gate_recall | uw_spread | uw_active | delta ATT [s] | sd | t | better in |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 1.5% | 12.00 | 0.00 | - | 0.02 | 1.0% | **-2.59** | 1.1 | -4.00 | 3/3 |
| `gauss1` | 0.0% | 0.00 | - | - | 0.00 | 0.0% | -1.79 | 2.5 | -2.77 | 12/15 |
| `gauss2` | 0.0% | 0.00 | - | - | 0.00 | 0.0% | **-2.32** | 2.9 | -3.11 | 11/15 |
| `gauss4` | 0.0% | 0.00 | - | - | 0.00 | 0.0% | -3.75 | 8.1 | -1.78 | 8/15 |
| `mask0p1` | 6.1% | 19.60 | 0.03 | 0.04 | 0.15 | 4.5% | +2.94 | 11.8 | +0.96 | 8/15 |
| `mask0p25` | 15.9% | 35.07 | 0.16 | 0.31 | 0.43 | 13.1% | +5.07 | 19.3 | +1.02 | 7/15 |
| `mask0p5` | 56.0% | 37.53 | 0.74 | 0.69 | 1.34 | 52.4% | -0.31 | 47.4 | -0.03 | 10/15 |
| `mask0p25_intersection` | 26.0% | 18.20 | 0.90 | 0.90 | 1.41 | 24.2% | -2.82 | 19.6 | -0.56 | 10/15 |
| `mask0p5_intersection` | 49.0% | 23.20 | 0.93 | 0.95 | 2.26 | 49.2% | -3.89 | 6.1 | -2.47 | 11/15 |
| `mask0p25_hold_t60` | 10.3% | 24.20 | 0.15 | 0.25 | 0.30 | 8.5% | -6.97 | 17.2 | -1.57 | 10/15 |
| `mask0p5_hold_t60` | 40.6% | 29.80 | 0.73 | 0.66 | 1.00 | 38.3% | -11.55 | 25.6 | -1.75 | 10/15 |
| `drift0p3` | 1.7% | 13.33 | 0.00 | - | 0.02 | 1.1% | -0.58 | 3.4 | -0.66 | 8/15 |
| `drift0p5` | 5.2% | 15.20 | 0.00 | - | 0.14 | 4.3% | +3.53 | 15.6 | +0.88 | 7/15 |

* Above the threshold: `clean` (-2.59 s, t = -4.00, better in 3/3 pairs), `gauss2` (-2.32 s, t = -3.11, better in 11/15 pairs)

### cityflow_hangzhou4x4_4ph

| scenario | gate_share | gate_switches | gate_precision | gate_recall | uw_spread | uw_active | delta ATT [s] | sd | t | better in |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 1.3% | 14.00 | 0.00 | - | 0.07 | 0.9% | -1.21 | 2.0 | -1.04 | 2/3 |
| `gauss1` | 0.0% | 0.00 | - | - | 0.00 | 0.0% | -0.55 | 2.3 | -0.94 | 10/15 |
| `gauss2` | 0.0% | 0.00 | - | - | 0.00 | 0.0% | +0.93 | 3.6 | +1.00 | 8/15 |
| `gauss4` | 0.0% | 0.00 | - | - | 0.00 | 0.0% | +5.69 | 8.1 | +2.72 | 4/15 |
| `mask0p1` | 1.8% | 16.53 | 0.00 | - | 0.11 | 1.4% | +3.37 | 18.4 | +0.71 | 9/15 |
| `mask0p25` | 2.6% | 20.93 | 0.23 | 0.05 | 0.17 | 2.1% | +6.49 | 35.0 | +0.72 | 7/15 |
| `mask0p5` | 10.0% | 45.27 | 0.90 | 0.13 | 0.95 | 8.5% | -0.41 | 73.7 | -0.02 | 6/15 |
| `mask0p25_intersection` | 5.3% | 25.87 | 0.71 | 0.19 | 0.52 | 4.4% | -4.28 | 15.0 | -1.10 | 10/15 |
| `mask0p5_intersection` | 15.1% | 43.27 | 0.95 | 0.29 | 1.48 | 14.1% | +1.88 | 36.7 | +0.20 | 8/15 |
| `mask0p25_hold_t60` | 1.7% | 16.87 | 0.05 | 0.01 | 0.11 | 1.2% | +14.02 | 46.7 | +1.16 | 3/15 |
| `mask0p5_hold_t60` | 6.5% | 34.60 | 0.66 | 0.10 | 0.66 | 5.4% | +6.41 | 76.2 | +0.33 | 6/15 |
| `drift0p3` | 1.3% | 14.27 | 0.00 | - | 0.08 | 0.9% | -0.52 | 1.7 | -1.22 | 11/15 |
| `drift0p5` | 1.4% | 14.40 | 0.00 | - | 0.09 | 1.0% | +3.40 | 12.0 | +1.10 | 6/15 |

* No scenario crosses the threshold: the mechanism is **neutral** for ATT.

## Check: the evaluator measures the same policy as T1

The evaluator's clean result vs the last training test, 12 learned runs: largest difference **0.000 s**, consistent to the log's precision.

## What this table does NOT say

* Levels are DETECTOR levels. For `gauss` they transfer to the feature with a factor of ~0.8-0.9, `urand` is ~1.6x weaker at the same `level` (`results/sensor_severity.md`).
* A mask with `zero` substitution changes far fewer feature cells than the fraction of dead lanes (zero is the modal value); the same number does not mean the same thing on networks with different loads.
* `hold` scenarios start at decision 60 (minute 15), so the first quarter of the episode is clean; their delta is not directly comparable with a mask from the start.
