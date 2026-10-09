# T2 — plan `t2_train_uw_rule`

Tryb realistic (yellow=5), 5 losowań awarii na scenariusz, ewaluacja ZAMROŻONYCH sterowników (`tsc degrade`). Wszystkie czytają tę samą warstwę detektorów: MaxPressure liczy decyzję z popsutego wektora stanu, nie z silnika.

Komórka = ATT [s], średnia po losowaniach, potem po seedach polityki (± std po seedach); w nawiasie Δ względem czystego wyniku tego samego sterownika. **▲ = gorzej niż FixedTime** — tylko tam dolna warstwa fallbacku ma pole do zysku.

## cityflow_jinan3x4_4ph

FixedTime (niezależny od odczytów): **507.0 s**.

| scenariusz | rdzeń GAT+PPO+CTDE (n=3) | rdzeń trenowany z awariami (n=3) | gatppo_fail_uwinf (n=3) | gatppo_fail_uw_rule (n=3) | gatppo_fail_gate (n=3) | gatppo_fail_uw_rule_gate (n=3) |
|---|---:|---:|---:|---:|---:|---:|
| `clean` | 295.9 ± 0.6 | 302.0 ± 1.7 | 302.6 ± 1.3 | 299.9 ± 1.1 | 305.2 ± 1.0 | 302.7 ± 2.1 |
| `gauss1` | 304.5 ± 1.4 (+2.9%) | 310.4 ± 1.2 (+2.8%) | 310.4 ± 1.2 (+2.6%) | 308.6 ± 0.9 (+2.9%) | 310.4 ± 1.2 (+1.7%) | 308.6 ± 0.9 (+2.0%) |
| `gauss2` | 319.9 ± 2.2 (+8.1%) | 321.8 ± 3.2 (+6.6%) | 321.8 ± 3.2 (+6.4%) | 319.5 ± 1.0 (+6.5%) | 321.8 ± 3.2 (+5.4%) | 319.5 ± 1.0 (+5.6%) |
| `gauss4` | 350.0 ± 4.5 (+18.3%) | 345.9 ± 8.4 (+14.5%) | 345.9 ± 8.4 (+14.3%) | 342.1 ± 0.6 (+14.1%) | 345.9 ± 8.4 (+13.3%) | 342.1 ± 0.6 (+13.0%) |
| `mask0p1` | 354.2 ± 20.0 (+19.7%) | 353.4 ± 9.8 (+17.0%) | 353.4 ± 6.9 (+16.8%) | 359.9 ± 9.5 (+20.0%) | 345.8 ± 2.6 (+13.3%) | 348.7 ± 1.2 (+15.2%) |
| `mask0p25` | 424.9 ± 31.1 (+43.6%) | 407.0 ± 12.5 (+34.8%) | 405.8 ± 9.4 (+34.1%) | 412.5 ± 21.1 (+37.6%) | 388.4 ± 4.9 (+27.3%) | 393.5 ± 5.6 (+30.0%) |
| `mask0p5` | 732.6 ± 28.5 (+147.5%) ▲ | 592.5 ± 19.4 (+96.2%) ▲ | 600.4 ± 46.8 (+98.4%) ▲ | 625.9 ± 17.0 (+108.7%) ▲ | 481.1 ± 11.3 (+57.6%) | 480.8 ± 11.8 (+58.9%) |
| `mask0p25_intersection` | 471.1 ± 68.8 (+59.2%) | 389.2 ± 42.7 (+28.9%) | 391.1 ± 47.5 (+29.3%) | 354.1 ± 6.2 (+18.1%) | 357.0 ± 8.4 (+16.9%) | 354.1 ± 4.4 (+17.0%) |
| `mask0p5_intersection` | 690.0 ± 115.3 (+133.2%) ▲ | 483.8 ± 47.4 (+60.2%) | 466.5 ± 47.5 (+54.2%) | 447.0 ± 26.7 (+49.0%) | 399.1 ± 4.7 (+30.7%) | 395.2 ± 1.6 (+30.6%) |
| `mask0p25_hold_t60` | 354.7 ± 24.8 (+19.9%) | 334.4 ± 8.7 (+10.7%) | 342.8 ± 10.7 (+13.3%) | 330.3 ± 8.0 (+10.1%) | 337.2 ± 6.1 (+10.5%) | 330.2 ± 5.0 (+9.1%) |
| `mask0p5_hold_t60` | 475.0 ± 38.9 (+60.5%) | 473.6 ± 29.4 (+56.8%) | 458.6 ± 8.7 (+51.5%) | 428.1 ± 29.5 (+42.7%) | 406.0 ± 10.6 (+33.0%) | 394.5 ± 9.3 (+30.3%) |
| `drift0p3` | 301.5 ± 0.5 (+1.9%) | 309.3 ± 0.7 (+2.4%) | 308.9 ± 1.2 (+2.1%) | 307.7 ± 1.0 (+2.6%) | 309.7 ± 1.6 (+1.4%) | 309.1 ± 1.8 (+2.1%) |
| `drift0p5` | 328.6 ± 4.3 (+11.0%) | 339.9 ± 8.2 (+12.5%) | 341.1 ± 8.2 (+12.7%) | 334.7 ± 1.3 (+11.6%) | 342.8 ± 3.9 (+12.3%) | 346.4 ± 1.9 (+14.4%) |

* **rdzeń GAT+PPO+CTDE** — najgorszy scenariusz `mask0p5`: 732.6 s (+147.5%). Gorzej niż FixedTime w 2 scenariuszach: `mask0p5`, `mask0p5_intersection`.
* **rdzeń trenowany z awariami** — najgorszy scenariusz `mask0p5`: 592.5 s (+96.2%). Gorzej niż FixedTime w 1 scenariuszach: `mask0p5`.
* **gatppo_fail_uwinf** — najgorszy scenariusz `mask0p5`: 600.4 s (+98.4%). Gorzej niż FixedTime w 1 scenariuszach: `mask0p5`.
* **gatppo_fail_uw_rule** — najgorszy scenariusz `mask0p5`: 625.9 s (+108.7%). Gorzej niż FixedTime w 1 scenariuszach: `mask0p5`.
* **gatppo_fail_gate** — najgorszy scenariusz `mask0p5`: 481.1 s (+57.6%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **gatppo_fail_uw_rule_gate** — najgorszy scenariusz `mask0p5`: 480.8 s (+58.9%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.


![krzywe degradacji cityflow_jinan3x4_4ph](figures/t2_train_uw_rule_cityflow_jinan3x4_4ph.png)

## cityflow_hangzhou4x4_4ph

FixedTime (niezależny od odczytów): **568.9 s**.

| scenariusz | rdzeń GAT+PPO+CTDE (n=3) | rdzeń trenowany z awariami (n=3) | gatppo_fail_uwinf (n=3) | gatppo_fail_uw_rule (n=3) | gatppo_fail_gate (n=3) | gatppo_fail_uw_rule_gate (n=3) |
|---|---:|---:|---:|---:|---:|---:|
| `clean` | 317.5 ± 1.0 | 321.2 ± 0.3 | 322.0 ± 1.0 | 321.6 ± 2.4 | 322.4 ± 0.2 | 321.2 ± 2.1 |
| `gauss1` | 344.9 ± 1.1 (+8.6%) | 341.9 ± 3.8 (+6.5%) | 341.9 ± 3.8 (+6.2%) | 341.4 ± 2.8 (+6.2%) | 341.9 ± 3.8 (+6.1%) | 341.4 ± 2.8 (+6.3%) |
| `gauss2` | 372.1 ± 4.2 (+17.2%) | 364.3 ± 10.4 (+13.4%) | 364.3 ± 10.4 (+13.1%) | 365.2 ± 9.0 (+13.6%) | 364.3 ± 10.4 (+13.0%) | 365.2 ± 9.0 (+13.7%) |
| `gauss4` | 419.3 ± 13.8 (+32.1%) | 401.6 ± 26.6 (+25.0%) | 401.6 ± 26.6 (+24.7%) | 407.3 ± 23.2 (+26.6%) | 401.6 ± 26.6 (+24.6%) | 407.3 ± 23.2 (+26.8%) |
| `mask0p1` | 363.4 ± 4.2 (+14.4%) | 345.1 ± 2.4 (+7.4%) | 346.6 ± 1.3 (+7.6%) | 366.1 ± 35.5 (+13.8%) | 345.4 ± 2.7 (+7.1%) | 348.7 ± 7.2 (+8.6%) |
| `mask0p25` | 460.1 ± 21.2 (+44.9%) | 394.1 ± 5.9 (+22.7%) | 392.7 ± 10.4 (+21.9%) | 410.0 ± 22.1 (+27.5%) | 395.7 ± 7.6 (+22.8%) | 402.2 ± 10.6 (+25.2%) |
| `mask0p5` | 739.2 ± 98.5 (+132.8%) ▲ | 471.1 ± 36.5 (+46.7%) | 472.2 ± 40.1 (+46.6%) | 469.2 ± 21.5 (+45.9%) | 472.9 ± 29.7 (+46.7%) | 472.5 ± 22.8 (+47.1%) |
| `mask0p25_intersection` | 472.2 ± 58.9 (+48.7%) | 361.1 ± 9.4 (+12.4%) | 356.2 ± 10.4 (+10.6%) | 362.7 ± 23.4 (+12.8%) | 352.9 ± 3.1 (+9.5%) | 348.7 ± 7.1 (+8.6%) |
| `mask0p5_intersection` | 765.9 ± 188.2 (+141.2%) ▲ | 414.7 ± 1.0 (+29.1%) | 405.5 ± 6.2 (+25.9%) | 430.8 ± 34.7 (+34.0%) | 392.3 ± 7.0 (+21.7%) | 394.2 ± 24.7 (+22.7%) |
| `mask0p25_hold_t60` | 411.1 ± 16.8 (+29.5%) | 365.3 ± 7.3 (+13.7%) | 391.0 ± 6.1 (+21.4%) | 384.9 ± 32.4 (+19.7%) | 372.7 ± 11.7 (+15.6%) | 386.7 ± 20.1 (+20.4%) |
| `mask0p5_hold_t60` | 544.4 ± 11.8 (+71.5%) | 442.6 ± 30.1 (+37.8%) | 525.4 ± 44.0 (+63.1%) | 505.7 ± 26.1 (+57.2%) | 452.4 ± 38.2 (+40.3%) | 458.8 ± 26.2 (+42.9%) |
| `drift0p3` | 319.5 ± 0.3 (+0.6%) | 323.5 ± 0.4 (+0.7%) | 323.8 ± 0.7 (+0.5%) | 323.4 ± 2.0 (+0.5%) | 324.4 ± 0.6 (+0.6%) | 323.8 ± 1.8 (+0.8%) |
| `drift0p5` | 339.9 ± 1.1 (+7.0%) | 334.3 ± 0.8 (+4.1%) | 334.5 ± 2.1 (+3.9%) | 337.7 ± 1.4 (+5.0%) | 335.7 ± 1.0 (+4.1%) | 339.1 ± 3.3 (+5.6%) |

* **rdzeń GAT+PPO+CTDE** — najgorszy scenariusz `mask0p5_intersection`: 765.9 s (+141.2%). Gorzej niż FixedTime w 2 scenariuszach: `mask0p5`, `mask0p5_intersection`.
* **rdzeń trenowany z awariami** — najgorszy scenariusz `mask0p5`: 471.1 s (+46.7%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **gatppo_fail_uwinf** — najgorszy scenariusz `mask0p5_hold_t60`: 525.4 s (+63.1%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **gatppo_fail_uw_rule** — najgorszy scenariusz `mask0p5_hold_t60`: 505.7 s (+57.2%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **gatppo_fail_gate** — najgorszy scenariusz `mask0p5`: 472.9 s (+46.7%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **gatppo_fail_uw_rule_gate** — najgorszy scenariusz `mask0p5`: 472.5 s (+47.1%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.


![krzywe degradacji cityflow_hangzhou4x4_4ph](figures/t2_train_uw_rule_cityflow_hangzhou4x4_4ph.png)

## Ramię: rdzeń trenowany z awariami vs rdzeń GAT+PPO+CTDE

Kolumny mają RÓŻNE checkpointy (osobne treningi), więc parujemy po (sieć, scenariusz, seed, losowanie): wspólne zostają ruch i wzorzec awarii, rozrzut między seedami wchodzi do różnicy. Kolumna odniesienia jest wskazana w planie (`baseline: gatppo`), a nie domyślona. `Δ` < 0 = ramię lepsze. `|t| ≥ 2.8` to próg po poprawce na tyle porównań, ile ma ta tabela.

Kolumny różni SPOSÓB TRENOWANIA, nie mechanizm włączany w inferencji, więc nie ma diagnostyki „czy mechanizm wystartował” — jest albo go nie ma już na poziomie wag. Rozrzut między seedami wchodzi tu do różnicy w całości (patrz zdanie o parowaniu wyżej), a liczba NIEZALEŻNYCH replikacji to liczba seedów, nie liczba par: losowania awarii dzielą checkpoint, więc |t| jest wobec nich optymistyczne.

### cityflow_jinan3x4_4ph

| scenariusz | Δ ATT [s] | sd | t | lepszy w |
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

* Powyżej progu: `clean` (+6.06 s, t = +4.76, lepszy w 0/3 parach), `drift0p3` (+7.77 s, t = +14.93, lepszy w 0/15 parach), `gauss1` (+5.86 s, t = +8.58, lepszy w 0/15 parach), `mask0p5` (-140.03 s, t = -10.77, lepszy w 15/15 parach), `mask0p5_intersection` (-206.15 s, t = -4.69, lepszy w 14/15 parach)

### cityflow_hangzhou4x4_4ph

| scenariusz | Δ ATT [s] | sd | t | lepszy w |
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

* Powyżej progu: `clean` (+3.66 s, t = +8.57, lepszy w 0/3 parach), `drift0p3` (+3.98 s, t = +15.70, lepszy w 0/15 parach), `gauss1` (-2.95 s, t = -4.02, lepszy w 13/15 parach), `gauss2` (-7.76 s, t = -4.61, lepszy w 12/15 parach), `gauss4` (-17.78 s, t = -5.26, lepszy w 14/15 parach), `mask0p1` (-18.30 s, t = -3.92, lepszy w 10/15 parach), `mask0p25` (-66.06 s, t = -3.71, lepszy w 14/15 parach), `mask0p25_hold_t60` (-45.85 s, t = -3.45, lepszy w 14/15 parach), `mask0p25_intersection` (-111.06 s, t = -2.81, lepszy w 12/15 parach), `mask0p5` (-268.04 s, t = -6.59, lepszy w 15/15 parach), `mask0p5_hold_t60` (-101.78 s, t = -5.05, lepszy w 13/15 parach), `mask0p5_intersection` (-351.20 s, t = -5.89, lepszy w 15/15 parach)

## Mechanizm: gatppo_fail_uwinf vs rdzeń trenowany z awariami

Obie kolumny liczą się z TEGO SAMEGO checkpointu i tych samych losowań awarii (`draw` wybiera, które detektory padły), więc porównujemy je **parami**, nie przez ±std z tabeli wyżej: wspólny rozrzut między seedami i losowaniami skraca się w różnicy. `Δ` < 0 = mechanizm lepszy. `|t| ≥ 2.8` to próg po poprawce na tyle porównań, ile ma ta tabela.

`uw_spread` = średni rozstęp log ρ w GRUPIE SOFTMAXU węzła, czyli jedyna część sygnału, którą attention widzi (na stałą wspólną dla grupy jest niezmienniczy). 0 = warstwa nie widzi różnicy między sąsiadami, ln 10 ≈ 2,3 = różnią się 10×. `uw_active` = odsetek krawędzi z niezerowym log ρ.

### cityflow_jinan3x4_4ph

| scenariusz | uw_spread | uw_active | Δ ATT [s] | sd | t | lepszy w |
|---|---:|---:|---:|---:|---:|---:|
| `clean` | 0.02 | 0.8% | +0.60 | 1.1 | +0.94 | 1/3 |
| `gauss1` | 0.00 | 0.0% | +0.00 | 0.0 | — | 0/0 |
| `gauss2` | 0.00 | 0.0% | +0.00 | 0.0 | — | 0/0 |
| `gauss4` | 0.00 | 0.0% | +0.00 | 0.0 | — | 0/0 |
| `mask0p1` | 0.15 | 4.8% | -0.01 | 13.6 | -0.00 | 6/15 |
| `mask0p25` | 0.43 | 13.1% | -1.23 | 13.1 | -0.36 | 9/15 |
| `mask0p5` | 1.42 | 57.9% | +7.83 | 31.9 | +0.95 | 6/15 |
| `mask0p25_intersection` | 1.43 | 25.2% | +1.87 | 17.6 | +0.41 | 9/15 |
| `mask0p5_intersection` | 2.26 | 49.4% | -17.31 | 40.3 | -1.66 | 9/15 |
| `mask0p25_hold_t60` | 0.32 | 9.0% | +8.44 | 24.1 | +1.36 | 8/15 |
| `mask0p5_hold_t60` | 1.03 | 41.4% | -15.01 | 70.5 | -0.82 | 8/15 |
| `drift0p3` | 0.02 | 1.1% | -0.35 | 2.5 | -0.55 | 9/15 |
| `drift0p5` | 0.15 | 4.5% | +1.24 | 15.3 | +0.31 | 5/15 |

* Żaden scenariusz nie przekracza progu — mechanizm jest wobec ATT **neutralny**.

### cityflow_hangzhou4x4_4ph

| scenariusz | uw_spread | uw_active | Δ ATT [s] | sd | t | lepszy w |
|---|---:|---:|---:|---:|---:|---:|
| `clean` | 0.10 | 1.2% | +0.86 | 0.6 | +2.36 | 0/3 |
| `gauss1` | 0.00 | 0.0% | +0.00 | 0.0 | — | 0/0 |
| `gauss2` | 0.00 | 0.0% | +0.00 | 0.0 | — | 0/0 |
| `gauss4` | 0.00 | 0.0% | +0.00 | 0.0 | — | 0/0 |
| `mask0p1` | 0.11 | 1.4% | +1.50 | 6.4 | +0.91 | 5/15 |
| `mask0p25` | 0.18 | 2.2% | -1.42 | 8.4 | -0.65 | 9/15 |
| `mask0p5` | 1.05 | 9.6% | +1.08 | 12.0 | +0.35 | 8/15 |
| `mask0p25_intersection` | 0.54 | 4.6% | -4.91 | 18.7 | -1.02 | 11/15 |
| `mask0p5_intersection` | 1.45 | 14.0% | -9.24 | 23.9 | -1.50 | 9/15 |
| `mask0p25_hold_t60` | 0.14 | 1.5% | **+25.74** | 33.9 | +2.94 | 3/15 |
| `mask0p5_hold_t60` | 0.78 | 6.5% | **+82.77** | 94.4 | +3.40 | 3/15 |
| `drift0p3` | 0.10 | 1.2% | +0.28 | 0.7 | +1.51 | 5/15 |
| `drift0p5` | 0.10 | 1.2% | +0.13 | 2.6 | +0.20 | 6/15 |

* Powyżej progu: `mask0p25_hold_t60` (+25.74 s, t = +2.94, lepszy w 3/15 parach), `mask0p5_hold_t60` (+82.77 s, t = +3.40, lepszy w 3/15 parach)

## Mechanizm: gatppo_fail_uw_rule vs gatppo_fail_uwinf

Kolumny mają RÓŻNE checkpointy (osobne treningi), więc parujemy po (sieć, scenariusz, seed, losowanie): wspólne zostają ruch i wzorzec awarii, rozrzut między seedami wchodzi do różnicy. Kolumna odniesienia jest wskazana w planie (`baseline: gatppo_fail_uwinf`), a nie domyślona. `Δ` < 0 = mechanizm lepszy. `|t| ≥ 2.8` to próg po poprawce na tyle porównań, ile ma ta tabela.

`uw_spread` = średni rozstęp log ρ w GRUPIE SOFTMAXU węzła, czyli jedyna część sygnału, którą attention widzi (na stałą wspólną dla grupy jest niezmienniczy). 0 = warstwa nie widzi różnicy między sąsiadami, ln 10 ≈ 2,3 = różnią się 10×. `uw_active` = odsetek krawędzi z niezerowym log ρ.

### cityflow_jinan3x4_4ph

| scenariusz | uw_spread | uw_active | Δ ATT [s] | sd | t | lepszy w |
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

* Powyżej progu: `gauss2` (-2.32 s, t = -3.11, lepszy w 11/15 parach)

### cityflow_hangzhou4x4_4ph

| scenariusz | uw_spread | uw_active | Δ ATT [s] | sd | t | lepszy w |
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

* Żaden scenariusz nie przekracza progu — mechanizm jest wobec ATT **neutralny**.

## Mechanizm: gatppo_fail_gate vs rdzeń trenowany z awariami

Obie kolumny liczą się z TEGO SAMEGO checkpointu i tych samych losowań awarii (`draw` wybiera, które detektory padły), więc porównujemy je **parami**, nie przez ±std z tabeli wyżej: wspólny rozrzut między seedami i losowaniami skraca się w różnicy. `Δ` < 0 = mechanizm lepszy. `|t| ≥ 2.8` to próg po poprawce na tyle porównań, ile ma ta tabela.

`gate_share` = odsetek par (decyzja, węzeł), w których sterowanie przejęła reguła. `gate_switches` = liczba ZMIAN trybu na węzłach w epizodzie — to one psują ruch (przełączenie resetuje fazę), a nie sam udział. `gate_precision` i `gate_recall` liczą się względem WYROCZNI (prawda z warstwy awarii, wyłącznie diagnostycznie) i rozdzielają „detektor jest słaby” od „przełączanie mało daje” — dwie rzeczy, które samo ATT skleja.

### cityflow_jinan3x4_4ph

| scenariusz | gate_share | gate_switches | gate_precision | gate_recall | Δ ATT [s] | sd | t | lepszy w |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 1.2% | 9.33 | 0.00 | — | **+3.25** | 0.9 | +6.08 | 0/3 |
| `gauss1` | 0.0% | 0.00 | — | — | +0.00 | 0.0 | — | 0/0 |
| `gauss2` | 0.0% | 0.00 | — | — | +0.00 | 0.0 | — | 0/0 |
| `gauss4` | 0.0% | 0.00 | — | — | +0.00 | 0.0 | — | 0/0 |
| `mask0p1` | 6.1% | 19.67 | 0.02 | 0.03 | -7.62 | 22.3 | -1.32 | 9/15 |
| `mask0p25` | 15.3% | 31.60 | 0.16 | 0.30 | -18.58 | 27.7 | -2.60 | 12/15 |
| `mask0p5` | 55.8% | 35.20 | 0.74 | 0.69 | **-111.40** | 51.1 | -8.45 | 15/15 |
| `mask0p25_intersection` | 25.9% | 18.20 | 0.90 | 0.90 | -32.29 | 51.8 | -2.41 | 9/15 |
| `mask0p5_intersection` | 48.9% | 23.07 | 0.93 | 0.95 | **-84.78** | 61.0 | -5.38 | 15/15 |
| `mask0p25_hold_t60` | 10.2% | 21.33 | 0.15 | 0.25 | +2.82 | 22.8 | +0.48 | 7/15 |
| `mask0p5_hold_t60` | 40.5% | 27.13 | 0.73 | 0.66 | **-67.52** | 67.0 | -3.90 | 13/15 |
| `drift0p3` | 1.6% | 12.27 | 0.00 | — | +0.41 | 2.0 | +0.79 | 7/15 |
| `drift0p5` | 5.3% | 15.47 | 0.00 | — | +2.97 | 16.5 | +0.70 | 5/15 |

* Powyżej progu: `clean` (+3.25 s, t = +6.08, lepszy w 0/3 parach), `mask0p5` (-111.40 s, t = -8.45, lepszy w 15/15 parach), `mask0p5_hold_t60` (-67.52 s, t = -3.90, lepszy w 13/15 parach), `mask0p5_intersection` (-84.78 s, t = -5.38, lepszy w 15/15 parach)

### cityflow_hangzhou4x4_4ph

| scenariusz | gate_share | gate_switches | gate_precision | gate_recall | Δ ATT [s] | sd | t | lepszy w |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 1.4% | 14.67 | 0.00 | — | **+1.19** | 0.2 | +13.52 | 0/3 |
| `gauss1` | 0.0% | 0.00 | — | — | +0.00 | 0.0 | — | 0/0 |
| `gauss2` | 0.0% | 0.00 | — | — | +0.00 | 0.0 | — | 0/0 |
| `gauss4` | 0.0% | 0.00 | — | — | +0.00 | 0.0 | — | 0/0 |
| `mask0p1` | 1.9% | 16.27 | 0.00 | — | +0.29 | 6.3 | +0.18 | 7/15 |
| `mask0p25` | 2.8% | 21.80 | 0.23 | 0.06 | +1.65 | 14.2 | +0.45 | 4/15 |
| `mask0p5` | 10.3% | 45.33 | 0.88 | 0.14 | +1.78 | 13.9 | +0.50 | 7/15 |
| `mask0p25_intersection` | 5.2% | 26.07 | 0.72 | 0.19 | -8.16 | 12.1 | -2.62 | 10/15 |
| `mask0p5_intersection` | 15.0% | 42.27 | 0.94 | 0.29 | **-22.38** | 27.9 | -3.10 | 13/15 |
| `mask0p25_hold_t60` | 1.6% | 16.13 | 0.02 | 0.00 | +7.40 | 37.6 | +0.76 | 6/15 |
| `mask0p5_hold_t60` | 6.0% | 33.87 | 0.65 | 0.09 | +9.78 | 46.3 | +0.82 | 6/15 |
| `drift0p3` | 1.4% | 14.27 | 0.00 | — | **+0.85** | 1.1 | +3.07 | 2/15 |
| `drift0p5` | 1.5% | 14.27 | 0.00 | — | +1.32 | 2.7 | +1.89 | 3/15 |

* Powyżej progu: `clean` (+1.19 s, t = +13.52, lepszy w 0/3 parach), `drift0p3` (+0.85 s, t = +3.07, lepszy w 2/15 parach), `mask0p5_intersection` (-22.38 s, t = -3.10, lepszy w 13/15 parach)

## Mechanizm: gatppo_fail_uw_rule_gate vs gatppo_fail_gate

Kolumny mają RÓŻNE checkpointy (osobne treningi), więc parujemy po (sieć, scenariusz, seed, losowanie): wspólne zostają ruch i wzorzec awarii, rozrzut między seedami wchodzi do różnicy. Kolumna odniesienia jest wskazana w planie (`baseline: gatppo_fail_gate`), a nie domyślona. `Δ` < 0 = mechanizm lepszy. `|t| ≥ 2.8` to próg po poprawce na tyle porównań, ile ma ta tabela.

`gate_share` = odsetek par (decyzja, węzeł), w których sterowanie przejęła reguła. `gate_switches` = liczba ZMIAN trybu na węzłach w epizodzie — to one psują ruch (przełączenie resetuje fazę), a nie sam udział. `gate_precision` i `gate_recall` liczą się względem WYROCZNI (prawda z warstwy awarii, wyłącznie diagnostycznie) i rozdzielają „detektor jest słaby” od „przełączanie mało daje” — dwie rzeczy, które samo ATT skleja. `uw_spread` = średni rozstęp log ρ w GRUPIE SOFTMAXU węzła, czyli jedyna część sygnału, którą attention widzi (na stałą wspólną dla grupy jest niezmienniczy). 0 = warstwa nie widzi różnicy między sąsiadami, ln 10 ≈ 2,3 = różnią się 10×. `uw_active` = odsetek krawędzi z niezerowym log ρ.

### cityflow_jinan3x4_4ph

| scenariusz | gate_share | gate_switches | gate_precision | gate_recall | uw_spread | uw_active | Δ ATT [s] | sd | t | lepszy w |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 1.5% | 12.00 | 0.00 | — | 0.02 | 1.0% | **-2.59** | 1.1 | -4.00 | 3/3 |
| `gauss1` | 0.0% | 0.00 | — | — | 0.00 | 0.0% | -1.79 | 2.5 | -2.77 | 12/15 |
| `gauss2` | 0.0% | 0.00 | — | — | 0.00 | 0.0% | **-2.32** | 2.9 | -3.11 | 11/15 |
| `gauss4` | 0.0% | 0.00 | — | — | 0.00 | 0.0% | -3.75 | 8.1 | -1.78 | 8/15 |
| `mask0p1` | 6.1% | 19.60 | 0.03 | 0.04 | 0.15 | 4.5% | +2.94 | 11.8 | +0.96 | 8/15 |
| `mask0p25` | 15.9% | 35.07 | 0.16 | 0.31 | 0.43 | 13.1% | +5.07 | 19.3 | +1.02 | 7/15 |
| `mask0p5` | 56.0% | 37.53 | 0.74 | 0.69 | 1.34 | 52.4% | -0.31 | 47.4 | -0.03 | 10/15 |
| `mask0p25_intersection` | 26.0% | 18.20 | 0.90 | 0.90 | 1.41 | 24.2% | -2.82 | 19.6 | -0.56 | 10/15 |
| `mask0p5_intersection` | 49.0% | 23.20 | 0.93 | 0.95 | 2.26 | 49.2% | -3.89 | 6.1 | -2.47 | 11/15 |
| `mask0p25_hold_t60` | 10.3% | 24.20 | 0.15 | 0.25 | 0.30 | 8.5% | -6.97 | 17.2 | -1.57 | 10/15 |
| `mask0p5_hold_t60` | 40.6% | 29.80 | 0.73 | 0.66 | 1.00 | 38.3% | -11.55 | 25.6 | -1.75 | 10/15 |
| `drift0p3` | 1.7% | 13.33 | 0.00 | — | 0.02 | 1.1% | -0.58 | 3.4 | -0.66 | 8/15 |
| `drift0p5` | 5.2% | 15.20 | 0.00 | — | 0.14 | 4.3% | +3.53 | 15.6 | +0.88 | 7/15 |

* Powyżej progu: `clean` (-2.59 s, t = -4.00, lepszy w 3/3 parach), `gauss2` (-2.32 s, t = -3.11, lepszy w 11/15 parach)

### cityflow_hangzhou4x4_4ph

| scenariusz | gate_share | gate_switches | gate_precision | gate_recall | uw_spread | uw_active | Δ ATT [s] | sd | t | lepszy w |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 1.3% | 14.00 | 0.00 | — | 0.07 | 0.9% | -1.21 | 2.0 | -1.04 | 2/3 |
| `gauss1` | 0.0% | 0.00 | — | — | 0.00 | 0.0% | -0.55 | 2.3 | -0.94 | 10/15 |
| `gauss2` | 0.0% | 0.00 | — | — | 0.00 | 0.0% | +0.93 | 3.6 | +1.00 | 8/15 |
| `gauss4` | 0.0% | 0.00 | — | — | 0.00 | 0.0% | +5.69 | 8.1 | +2.72 | 4/15 |
| `mask0p1` | 1.8% | 16.53 | 0.00 | — | 0.11 | 1.4% | +3.37 | 18.4 | +0.71 | 9/15 |
| `mask0p25` | 2.6% | 20.93 | 0.23 | 0.05 | 0.17 | 2.1% | +6.49 | 35.0 | +0.72 | 7/15 |
| `mask0p5` | 10.0% | 45.27 | 0.90 | 0.13 | 0.95 | 8.5% | -0.41 | 73.7 | -0.02 | 6/15 |
| `mask0p25_intersection` | 5.3% | 25.87 | 0.71 | 0.19 | 0.52 | 4.4% | -4.28 | 15.0 | -1.10 | 10/15 |
| `mask0p5_intersection` | 15.1% | 43.27 | 0.95 | 0.29 | 1.48 | 14.1% | +1.88 | 36.7 | +0.20 | 8/15 |
| `mask0p25_hold_t60` | 1.7% | 16.87 | 0.05 | 0.01 | 0.11 | 1.2% | +14.02 | 46.7 | +1.16 | 3/15 |
| `mask0p5_hold_t60` | 6.5% | 34.60 | 0.66 | 0.10 | 0.66 | 5.4% | +6.41 | 76.2 | +0.33 | 6/15 |
| `drift0p3` | 1.3% | 14.27 | 0.00 | — | 0.08 | 0.9% | -0.52 | 1.7 | -1.22 | 11/15 |
| `drift0p5` | 1.4% | 14.40 | 0.00 | — | 0.09 | 1.0% | +3.40 | 12.0 | +1.10 | 6/15 |

* Żaden scenariusz nie przekracza progu — mechanizm jest wobec ATT **neutralny**.

## Kontrola: ewaluator mierzy tę samą politykę co T1

Czysty wynik ewaluatora vs ostatni test treningu, 12 biegów uczonych: największa różnica **0.000 s** — zgodne co do precyzji logu.

## Czego ta tabela NIE mówi

* Poziomy są poziomami DETEKTORA. Na cechę `gauss` przenosi się ze współczynnikiem ~0,8–0,9, `urand` jest ~1,6× słabszy przy tym samym `level` (`results/sensor_severity.md`).
* Maska z podstawieniem `zero` zmienia znacznie mniej komórek cech, niż wynosi ułamek martwych pasów (zero jest wartością modalną) — ta sama liczba nie znaczy tego samego na sieciach o różnym obciążeniu.
* Scenariusze `hold` zaczynają się w decyzji 60 (15. minuta), więc pierwsza ćwiartka epizodu jest czysta — ich Δ nie jest porównywalne wprost z maską od początku.
