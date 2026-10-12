# Prototype 2.0 gate: DECISION uncertainty vs sensor failure and OOD

Mode `edl`, 0-hop (one intersection), 150 epochs, trained on: clean, gauss1, gauss2, mask0p25.
Evaluated on the episode TAIL (last 33% of decisions). Epistemic reference: **ensemble** (K=5).

| set | scenario | train | acc | ECE | EDL total | EDL epist | epist/total | vacuity | ens epist | MC epist | ties | EDL total (decidable) | dead movements |
|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | clean | x | 0.750 | 0.111 | 0.570 | 0.0851 | 0.149 | 0.274 | 0.3527 | 0.0047 | 10.0% | 0.558 | 0.0% |
| base | drift0p2 |  | 0.718 | 0.139 | 0.564 | 0.0850 | 0.151 | 0.274 | 0.3423 | 0.0044 | 10.0% | 0.550 | 0.0% |
| base | drift0p3 |  | 0.687 | 0.179 | 0.557 | 0.0840 | 0.151 | 0.271 | 0.3427 | 0.0040 | 10.0% | 0.544 | 0.0% |
| base | gauss1 | x | 0.605 | 0.204 | 0.622 | 0.0956 | 0.154 | 0.309 | 0.3835 | 0.0046 | 10.0% | 0.607 | 0.0% |
| base | gauss2 | x | 0.486 | 0.305 | 0.621 | 0.0960 | 0.155 | 0.310 | 0.4506 | 0.0051 | 10.0% | 0.612 | 0.0% |
| base | gauss3p5 |  | 0.413 | 0.364 | 0.595 | 0.0927 | 0.156 | 0.300 | 0.5191 | 0.0066 | 10.0% | 0.589 | 0.0% |
| base | gauss4 |  | 0.399 | 0.388 | 0.580 | 0.0904 | 0.156 | 0.293 | 0.5312 | 0.0067 | 10.0% | 0.572 | 0.0% |
| base | mask0p25 | x | 0.664 | 0.170 | 0.588 | 0.0899 | 0.153 | 0.290 | 0.3394 | 0.0043 | 10.0% | 0.575 | 68.4% |
| base | mask0p25_intersection |  | 0.622 | 0.176 | 0.641 | 0.0994 | 0.155 | 0.321 | 0.3367 | 0.0041 | 10.0% | 0.630 | 52.8% |
| base | mask0p5 |  | 0.551 | 0.256 | 0.612 | 0.0953 | 0.156 | 0.308 | 0.3083 | 0.0039 | 10.0% | 0.603 | 91.7% |
| base | urand3p5 |  | 0.486 | 0.304 | 0.631 | 0.0980 | 0.155 | 0.317 | 0.4576 | 0.0054 | 10.0% | 0.625 | 0.0% |
| d0p75 | clean |  | 0.735 | 0.089 | 0.615 | 0.0926 | 0.151 | 0.299 | 0.3162 | 0.0036 | 14.0% | 0.585 | 0.0% |
| d0p5 | clean |  | 0.700 | 0.114 | 0.718 | 0.1120 | 0.156 | 0.361 | 0.3421 | 0.0037 | 23.6% | 0.668 | 0.0% |
| d0p25 | clean |  | 0.690 | 0.133 | 0.777 | 0.1207 | 0.155 | 0.389 | 0.2862 | 0.0034 | 46.4% | 0.718 | 0.0% |
| r2000 | clean |  | 0.718 | 0.112 | 0.701 | 0.1111 | 0.158 | 0.359 | 0.3332 | 0.0033 | 21.5% | 0.666 | 0.0% |
| r2500 | clean |  | 0.730 | 0.096 | 0.616 | 0.0946 | 0.154 | 0.306 | 0.3421 | 0.0034 | 13.0% | 0.597 | 0.0% |
| hz | clean |  | 0.732 | 0.122 | 0.729 | 0.1115 | 0.153 | 0.359 | 0.2695 | 0.0030 | 44.7% | 0.704 | 0.0% |

## Uncertainty gain on OOD (relative to clean base), maximum per group

| group | EDL total | EDL total (decidable) | EDL epist | EDL epist (decidable) | ens epist | ens epist (decidable) |
|---|---:|---:|---:|---:|---:|---:|
| demand x | +0.207 | +0.160 | +0.0356 | +0.0247 | -0.0106 | -0.0193 |
| pattern | +0.131 | +0.109 | +0.0261 | +0.0211 | -0.0106 | -0.0125 |
| topology | +0.158 | +0.146 | +0.0264 | +0.0239 | -0.0832 | -0.0522 |
| *noise gauss2 (for comparison)* | - | - | +0.0109 | - | +0.0979 | - |

## Exit criteria

* **(a) monotonicity**: EDL total over `gauss` (sigma = 0, 1, 2, 3.5, 4): 0.570, 0.622, 0.621, 0.595, 0.580 -> **NO**.
* **(b) separation**: EDL aleatoric on seen noise +0.039, unseen +0.004; EDL epistemic on OOD +0.0356 vs on noise +0.0109.
* **(c) calibration**: ECE clean 0.111, worst scenario 0.388.
* **(d) EDL vs reference**: epistemic correlation over rows with the ensemble **-0.358**, with last-layer MC-dropout -0.464.
* **(e) unseen**: noise 3.5/4.0 and `urand` above clean: **YES**; 50% mask and intersection mask: **YES**; `drift`: **NO** (delta -0.013 with an accuracy drop of 0.063).
* **(g) is EDL epistemic a separate channel**: epistemic/total in [0.149, 0.158] over all 17 rows (spread 6.0% of the mean) -> **ONE SIGNAL**.

## Verdict

**Gate 2.0 for the DECISION head: NOT PASSED.**

1. **Outside the training range decision uncertainty stops growing** (peak at index 1 of 0.570, 0.622, 0.621, 0.595, 0.580), while accuracy keeps falling (by 0.351) and ECE rises to 0.388: confident and wrong exactly where the fallback should act.
2. **`drift` is invisible** to decision uncertainty (delta -0.013 with an accuracy drop of 0.063).
3. **EDL epistemic is not a separate channel**: epistemic/total stays in [0.149, 0.158] in every row, so epistemic, vacuity and total are one signal on different scales. A gain in "epistemic on OOD" is a gain in total, not separate information about an unknown state.
4. **Even the ensemble does not separate OOD from noise** (best group "pattern": -0.0125 vs noise +0.0979); an epistemic fallback channel has no empirical basis on these sets.
5. **Conclusion for 2.2:** the per-edge signal comes from a heteroscedastic head on the INPUT FEATURES (`results/gate_2_2.md`), not from the Dirichlet spread.

## What this table does NOT say

Base dataset `tie_rate` = **14.1%**: in that many decisions two phases have equal p(s) and high DECISION uncertainty is correct, unrelated to failure. The *ties* column shows their share per set; it grows at low demand, which is why OOD gains are also given on decidable samples.

MC-dropout drops the LAST LAYER only; the reference for (d) is a deep ensemble (K `softmax` members, different seeds), because MC epistemic is of order 0.004.
