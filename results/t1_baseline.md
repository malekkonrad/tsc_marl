# T1 baseline: the reference point of the thesis

Generated: 2026-10-11 16:51. Protocol: `configs/tsc/thesis_protocol.yml` (dt=15 s, episode 3600 s, `PYTHONHASHSEED=0`), networks in the 4-phase variant, as in the papers' protocol.

ATT = mean of the last 10 test episodes (methods without training: one run).
delta is computed against the literature in **benchmark** mode, the mode the papers report in.

## cityflow_jinan3x4_4ph

| agent | benchmark (yellow=0) | realistic (yellow=5) | literature | delta benchmark |
|---|---|---|---|---|
| fixedtime | 424.7 | 507.0 | 428.11 | -0.8% |
| maxpressure | 316.8 | 361.0 | 273.96 | +15.6% |
| sotl | 318.7 | 352.9 | - | - |
| efficientmp | 272.2 | 316.1 | 269.87 | +0.9% |
| advancedmp | 274.3 | 299.2 | 253.61 | +8.2% |
| presslight | - | 321.3 (n=3) | - | - |
| colight | 284.2 (n=3) | 323.6 (n=3) | 272.06 | +4.5% |
| advancedcolight | 268.8 (n=3) | 293.5 (n=5) | 245.73 | +9.4% |

## cityflow_hangzhou4x4_4ph

| agent | benchmark (yellow=0) | realistic (yellow=5) | literature | delta benchmark |
|---|---|---|---|---|
| fixedtime | 506.2 | 568.9 | 495.57 | +2.2% |
| maxpressure | 374.3 | 392.7 | 288.54 | +29.7% |
| sotl | 344.9 | 367.2 | - | - |
| efficientmp | 314.0 | 330.1 | 284.44 | +10.4% |
| advancedmp | 314.7 | 325.4 | 279.47 | +12.6% |
| presslight | - | 339.2 (n=3) | - | - |
| colight | 326.0 (n=3) | 342.5 (n=3) | 297.02 | +9.8% |
| advancedcolight | 306.6 (n=3) | 317.1 (n=5) | 270.45 | +13.4% |

## How to read delta

- **`maxpressure` is the LibSignal implementation**, computing pressure from
  `lane_count` (all vehicles, no per-lane normalisation) rather than the paper's
  definition; its deviation is NOT a gate criterion. `efficientmp` plays the role
  of a correct MP.
- **Reproducibility depends on the network**: JiNan matches the literature to ~1%
  (FixedTime -0.8%, Efficient-MP +0.9%), HangZhou does not (Efficient-MP +10.4%).
  Same protocol, same code, data verified identical to the authors' files
  (`hangzhou_4x4_advanced/` vs `hangzhou_4x4_gudang_18041610_1h/`: results agree
  to 0.01 s). To be described in the thesis as a limitation of the comparison.
- **`advancedmp` is reported at a global W1 = 1.0.** In benchmark mode the demand
  term does nothing (without a transition cost holding a phase gains nothing), so
  the method degenerates to Efficient-MP and +8.2% against the literature is
  EXPECTED. The paper's Advanced-MP number cannot be reproduced in benchmark mode
  at any W1, the strongest argument that this mode loses the mechanism the paper
  describes.
- The `realistic` column is our protocol (5 s phase transition). The difference
  between the modes is a result in itself, see `t1_w1_sweep.md`.
