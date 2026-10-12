# Prototype 2.2 gate: OBSERVATION head (uncertainty of input features)

Window of 12 readings + upstream neighbour mean, 60 epochs, Gaussian NLL against `x_clean`, trained on: clean, gauss1, gauss2, mask0p25. Evaluated on the episode tail. u = sigma in units of the clean feature deviation; movement u = max over its features (EP, demand).

| scenario | train | corrupted movements | wrong movements | mean u | AUROC movement (corrupted) | AUROC movement (wrong) | AUROC edge (max) | AUROC edge (mean) | 90% coverage | input RMSE | mu RMSE |
|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | x | 0.0% | 0.0% | 0.492 | - | - | - | - | 66.7% | 0.000 | 0.520 |
| drift0p2 |  | 23.3% | 23.3% | 0.517 | 0.529 | 0.529 | 0.513 | 0.515 | 62.5% | 0.446 | 0.661 |
| drift0p3 |  | 34.5% | 34.5% | 0.542 | 0.536 | 0.536 | 0.521 | 0.519 | 59.9% | 0.668 | 0.798 |
| gauss1 | x | 74.7% | 74.7% | 0.974 | 0.856 | 0.856 | 0.832 | 0.860 | 82.3% | 0.865 | 0.818 |
| gauss2 | x | 89.0% | 89.0% | 1.374 | 0.903 | 0.903 | 0.886 | 0.909 | 82.9% | 1.627 | 1.107 |
| gauss3p5 |  | 94.1% | 94.1% | 1.848 | 0.920 | 0.920 | 0.912 | 0.930 | 80.5% | 2.698 | 1.407 |
| gauss4 |  | 94.9% | 94.9% | 2.000 | 0.922 | 0.922 | 0.917 | 0.934 | 79.6% | 3.047 | 1.493 |
| mask0p25 | x | 68.4% | 21.2% | 0.696 | 0.656 | 0.788 | 0.652 | 0.652 | 72.5% | 0.759 | 0.774 |
| mask0p25_intersection |  | 52.8% | 29.1% | 0.606 | 0.600 | 0.651 | 0.580 | 0.583 | 61.8% | 1.031 | 1.025 |
| mask0p5 |  | 91.7% | 44.3% | 0.749 | 0.671 | 0.739 | 0.698 | 0.701 | 68.4% | 1.207 | 1.123 |
| urand3p5 |  | 91.6% | 91.6% | 1.427 | 0.913 | 0.913 | 0.897 | 0.918 | 84.8% | 1.658 | 1.113 |

### The same metrics on OOD sets (traffic outside training)

| set | scenario | mean u | AUROC movement | 90% coverage |
|---|---|---:|---:|---:|
| r2000 | clean | 0.380 | - | 78.4% |
| r2000 | gauss2 | 1.083 | 0.918 | 88.4% |
| r2000 | mask0p5 | 0.561 | 0.651 | 79.3% |
| r2000 | drift0p3 | 0.401 | 0.528 | 73.5% |
| hz | clean | 0.479 | - | 86.8% |
| hz | gauss2 | 0.935 | 0.857 | 91.2% |
| hz | mask0p5 | 0.639 | 0.534 | 82.8% |
| hz | drift0p3 | 0.493 | 0.481 | 84.4% |
| d0p25 | clean | 0.475 | - | 86.8% |
| d0p25 | gauss2 | 0.946 | 0.873 | 91.2% |
| d0p25 | mask0p5 | 0.616 | 0.572 | 85.3% |
| d0p25 | drift0p3 | 0.486 | 0.524 | 83.9% |

## 2x2 separation: do the two channels measure two different things

`u_obs` = mean u of the observation head, `u_ep` = deep-ensemble epistemic (MI between members, prototype 2.0). Row = state of readings, column = traffic.

| readings \ traffic | training | d0p25 | hz | r2000 |
|---|---|---|---|---|
| clean | u_obs 0.492, u_ep 0.3527 | u_obs 0.475, u_ep 0.2862 | u_obs 0.479, u_ep 0.2695 | u_obs 0.380, u_ep 0.3332 |
| gauss2 | u_obs 1.374, u_ep 0.4506 | u_obs 0.946, u_ep 0.5041 | u_obs 0.935, u_ep 0.4801 | u_obs 1.083, u_ep 0.4880 |
| mask0p5 | u_obs 0.749, u_ep 0.3083 | u_obs 0.616, u_ep 0.2064 | u_obs 0.639, u_ep 0.1944 | u_obs 0.561, u_ep 0.2777 |

* `u_obs`: gain from failure (row) **+0.628**, from OOD (column) **-0.048**.
* `u_ep`: gain from failure **+0.1703**, from OOD **-0.0564**.
* `u_obs` DROPS on traffic outside training: the head's sigma scales with load (fewer vehicles, smaller possible reading error) and the OOD sets are lighter. Harmless for a fallback (no false alarms), but a threshold on `u_obs` must be relative to the sigma predicted on clean traffic, not an absolute constant.
* Separation verdict: `u_obs` reacts to failure, not to OOD; `u_ep` does NOT react more to OOD than to failure, so a two-level fallback has no empirical basis here.

## Exit criteria

* **(a) monotonicity**: u over noise sigma (0, 1, 2, 3.5, 4): 0.492, 0.974, 1.374, 1.848, 2.000 -> **YES** (3.5 and 4.0 UNSEEN in training).
* **(c) calibration**: 90% interval coverage on clean 66.7%, on seen noise 82.6%.
* **(d) detection on seen**: min movement AUROC (gauss1, gauss2, mask0p25): **0.788** (threshold 0.8).
* **(e) detection on UNSEEN**: min AUROC over gauss3p5, gauss4, urand3p5, mask0p5, mask0p25_intersection: **0.651** (threshold 0.7); `drift` (another family): max AUROC **0.536**.
* **(f) movement -> edge aggregation**: AUROC(max) - AUROC(mean), mean: -0.011 -> in attention **mean**.

**Gate 2.2 (observation head): NOT PASSED.**

## What this table does NOT say

* The prototype is SUPERVISED (the simulator knows `x_clean`). In the control loop the head works the same way: trained offline, inference without the truth.
* AUROC compares corrupted cells with the CLEAN scenario; it does not measure localisation within one episode (where all detectors are noisy there is nothing to localise).
* "Corrupted movement" = a dead detector or a feature error > 0.5 vehicles. Drift of small counts rarely crosses it, so its AUROC uses few cells.
