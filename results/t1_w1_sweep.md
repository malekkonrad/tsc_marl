# Advanced-MP: tuning W1 (counterpart of Figure 2 in Zhang 2022)

dt = 15 s, `obs_length` = 167 m, `drop_always_green` = true, seed 0, JiNan 3x4 / HangZhou 4x4,
4-phase variant. Reference point = Efficient-MP in the same mode.

## JiNan 3x4 (4 phases)

| W1 | ATT, realistic (yellow=5) | ATT, benchmark (yellow=0) |
|---|---|---|
| 0.05 | 315.8 | **271.7** |
| 0.25 | 313.3 | 271.9 |
| 0.50 | 311.2 | 271.8 |
| 0.75 | 303.3 | 273.4 |
| **1.00** | **299.2** | 274.3 |
| 1.50 | 302.4 | 279.8 |
| 2.00 | 305.7 | 285.1 |
| 3.00 | - | 297.1 |
| *Efficient-MP* | *316.1* | *272.2* |

## HangZhou 4x4 (4 phases), realistic mode

| W1 | 0.25 | 0.50 | 0.75 | 1.00 | 1.50 | 2.00 |
|---|---|---|---|---|---|---|
| ATT | 326.2 | **323.9** | 324.0 | 325.4 | 327.9 | 331.4 |

## Conclusions

1. **W1 = 1.0**, the value of the authors' reference implementation
   (`run_advanced_maxpressure.py`: `"W": 1`), is the optimum on JiNan and lies 0.5% from
   the optimum on HangZhou. One global value for the whole thesis is justified.
2. **The demand term works ONLY in realistic mode.** At yellow=5 it gives -5.3%
   against Efficient-MP (299.2 vs 316.1; the paper claims -6.0%). At yellow=0 the
   optimum escapes to W1 -> 0, i.e. the method degenerates to Efficient-MP, because
   without a phase transition cost "holding the phase" saves nothing.
   This is the core of the Advanced-MP mechanism and an argument that benchmark mode
   loses phenomena that decide the method's advantage in a real deployment.
3. **W1 = 0 does NOT degenerate to Efficient-MP**: when `max p(s) <= 0` (24% of decisions
   on JiNan), the condition `d * W1 > max p` holds trivially and the phase is held.
   That is why the first sweep over the grid {0,1,2,3,4} could not find the optimum.
