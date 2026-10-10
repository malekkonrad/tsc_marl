# T2 — plan `t2_krakow`

Tryb realistic (yellow=5), 3 losowań awarii na scenariusz, ewaluacja ZAMROŻONYCH sterowników (`tsc degrade`). Wszystkie czytają tę samą warstwę detektorów: MaxPressure liczy decyzję z popsutego wektora stanu, nie z silnika.

Komórka = ATT [s], średnia po losowaniach, potem po seedach polityki (± std po seedach); w nawiasie Δ względem czystego wyniku tego samego sterownika. **▲ = gorzej niż FixedTime** — tylko tam dolna warstwa fallbacku ma pole do zysku.

## sumo_krakow3w_mid

FixedTime (niezależny od odczytów): **347.6 s**.

| scenariusz | rdzeń GAT+PPO+CTDE (n=3) | rdzeń z awariami (DR) (n=3) | rdzeń + bramka-reguła (n=3) | DR + bramka-reguła (n=3) | Advanced-MP (n=1) | Efficient-MP (n=1) |
|---|---:|---:|---:|---:|---:|---:|
| `clean` | 376.3 ± 46.1 ▲ | 358.7 ± 17.2 ▲ | 361.1 ± 6.8 ▲ | 359.3 ± 11.9 ▲ | 396.9 ▲ | 373.5 ▲ |
| `gauss2` | 512.5 ± 8.0 (+36.2%) ▲ | 516.3 ± 57.8 (+44.0%) ▲ | 512.5 ± 8.0 (+41.9%) ▲ | 516.3 ± 57.8 (+43.7%) ▲ | 445.3 (+12.2%) ▲ | 375.9 (+0.7%) ▲ |
| `mask0p25` | 424.3 ± 56.8 (+12.7%) ▲ | 373.7 ± 17.4 (+4.2%) ▲ | 391.3 ± 11.1 (+8.4%) ▲ | 375.4 ± 8.5 (+4.5%) ▲ | 480.4 (+21.0%) ▲ | 441.7 (+18.3%) ▲ |
| `mask0p5` | 529.0 ± 85.4 (+40.6%) ▲ | 410.8 ± 66.0 (+14.5%) ▲ | 468.1 ± 16.2 (+29.6%) ▲ | 409.7 ± 62.2 (+14.0%) ▲ | 736.5 (+85.6%) ▲ | 718.1 (+92.3%) ▲ |
| `mask0p25_intersection` | 812.2 ± 218.6 (+115.8%) ▲ | 414.7 ± 22.6 (+15.6%) ▲ | 382.4 ± 32.3 (+5.9%) ▲ | 357.2 ± 4.2 (-0.6%) ▲ | 1020.1 (+157.0%) ▲ | 1008.0 (+169.9%) ▲ |
| `mask0p5_intersection` | 911.4 ± 250.9 (+142.2%) ▲ | 416.7 ± 16.0 (+16.2%) ▲ | 391.3 ± 38.7 (+8.3%) ▲ | 357.7 ± 10.0 (-0.5%) ▲ | 1164.6 (+193.4%) ▲ | 1158.6 (+210.2%) ▲ |
| `mask0p5_hold_t60` | 486.4 ± 19.6 (+29.2%) ▲ | 422.0 ± 18.9 (+17.7%) ▲ | 405.8 ± 14.8 (+12.4%) ▲ | 395.7 ± 25.1 (+10.1%) ▲ | 602.2 (+51.7%) ▲ | 538.0 (+44.1%) ▲ |
| `drift0p3` | 367.0 ± 9.6 (-2.5%) ▲ | 357.8 ± 6.3 (-0.3%) ▲ | 368.1 ± 7.3 (+1.9%) ▲ | 358.7 ± 5.9 (-0.2%) ▲ | 402.9 (+1.5%) ▲ | 379.4 (+1.6%) ▲ |

* **rdzeń GAT+PPO+CTDE** — najgorszy scenariusz `mask0p5_intersection`: 911.4 s (+142.2%). Gorzej niż FixedTime w 7 scenariuszach: `drift0p3`, `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **rdzeń z awariami (DR)** — najgorszy scenariusz `gauss2`: 516.3 s (+44.0%). Gorzej niż FixedTime w 7 scenariuszach: `drift0p3`, `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **rdzeń + bramka-reguła** — najgorszy scenariusz `gauss2`: 512.5 s (+41.9%). Gorzej niż FixedTime w 7 scenariuszach: `drift0p3`, `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **DR + bramka-reguła** — najgorszy scenariusz `gauss2`: 516.3 s (+43.7%). Gorzej niż FixedTime w 7 scenariuszach: `drift0p3`, `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **Advanced-MP** — najgorszy scenariusz `mask0p5_intersection`: 1164.6 s (+193.4%). Gorzej niż FixedTime w 7 scenariuszach: `drift0p3`, `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **Efficient-MP** — najgorszy scenariusz `mask0p5_intersection`: 1158.6 s (+210.2%). Gorzej niż FixedTime w 7 scenariuszach: `drift0p3`, `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.


![krzywe degradacji sumo_krakow3w_mid](figures/t2_krakow_sumo_krakow3w_mid.png)

## sumo_krakow3w_low

FixedTime (niezależny od odczytów): **297.9 s**.

| scenariusz | rdzeń GAT+PPO+CTDE (n=3) | rdzeń z awariami (DR) (n=3) | rdzeń + bramka-reguła (n=3) | DR + bramka-reguła (n=3) | Advanced-MP (n=1) | Efficient-MP (n=1) |
|---|---:|---:|---:|---:|---:|---:|
| `clean` | 277.4 ± 0.7 | 281.4 ± 0.7 | 277.7 ± 0.6 | 281.4 ± 1.1 | 279.8 | 282.6 |
| `gauss2` | 364.0 ± 8.2 (+31.2%) ▲ | 327.3 ± 20.2 (+16.3%) ▲ | 364.0 ± 8.2 (+31.1%) ▲ | 327.3 ± 20.2 (+16.3%) ▲ | 338.8 (+21.1%) ▲ | 320.4 (+13.4%) ▲ |
| `mask0p25` | 292.0 ± 2.9 (+5.3%) | 286.3 ± 1.8 (+1.8%) | 289.9 ± 5.4 (+4.4%) | 286.8 ± 1.8 (+1.9%) | 313.3 (+12.0%) ▲ | 312.1 (+10.4%) ▲ |
| `mask0p5` | 351.1 ± 38.0 (+26.6%) ▲ | 294.1 ± 5.4 (+4.5%) | 323.9 ± 16.7 (+16.6%) ▲ | 291.8 ± 2.0 (+3.7%) | 561.7 (+100.7%) ▲ | 556.5 (+96.9%) ▲ |
| `mask0p25_intersection` | 658.7 ± 171.2 (+137.4%) ▲ | 314.6 ± 13.0 (+11.8%) ▲ | 299.3 ± 6.3 (+7.8%) ▲ | 290.5 ± 2.4 (+3.2%) | 789.5 (+182.2%) ▲ | 784.8 (+177.7%) ▲ |
| `mask0p5_intersection` | 742.3 ± 203.1 (+167.6%) ▲ | 329.4 ± 11.8 (+17.1%) ▲ | 298.8 ± 4.6 (+7.6%) ▲ | 293.3 ± 1.9 (+4.2%) | 936.4 (+234.7%) ▲ | 935.8 (+231.2%) ▲ |
| `mask0p5_hold_t60` | 319.4 ± 30.3 (+15.1%) ▲ | 298.0 ± 16.6 (+5.9%) ▲ | 304.5 ± 7.1 (+9.7%) ▲ | 295.1 ± 8.4 (+4.9%) | 392.0 (+40.1%) ▲ | 446.4 (+58.0%) ▲ |
| `drift0p3` | 280.7 ± 3.9 (+1.2%) | 282.2 ± 0.8 (+0.3%) | 278.9 ± 0.8 (+0.4%) | 282.1 ± 0.7 (+0.2%) | 286.6 (+2.4%) | 284.3 (+0.6%) |

* **rdzeń GAT+PPO+CTDE** — najgorszy scenariusz `mask0p5_intersection`: 742.3 s (+167.6%). Gorzej niż FixedTime w 5 scenariuszach: `gauss2`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **rdzeń z awariami (DR)** — najgorszy scenariusz `mask0p5_intersection`: 329.4 s (+17.1%). Gorzej niż FixedTime w 4 scenariuszach: `gauss2`, `mask0p25_intersection`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **rdzeń + bramka-reguła** — najgorszy scenariusz `gauss2`: 364.0 s (+31.1%). Gorzej niż FixedTime w 5 scenariuszach: `gauss2`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **DR + bramka-reguła** — najgorszy scenariusz `gauss2`: 327.3 s (+16.3%). Gorzej niż FixedTime w 1 scenariuszach: `gauss2`.
* **Advanced-MP** — najgorszy scenariusz `mask0p5_intersection`: 936.4 s (+234.7%). Gorzej niż FixedTime w 6 scenariuszach: `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **Efficient-MP** — najgorszy scenariusz `mask0p5_intersection`: 935.8 s (+231.2%). Gorzej niż FixedTime w 6 scenariuszach: `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.


![krzywe degradacji sumo_krakow3w_low](figures/t2_krakow_sumo_krakow3w_low.png)

## sumo_krakow3w_high

FixedTime (niezależny od odczytów): **390.5 s**.

| scenariusz | rdzeń GAT+PPO+CTDE (n=3) | rdzeń z awariami (DR) (n=3) | rdzeń + bramka-reguła (n=3) | DR + bramka-reguła (n=3) | Advanced-MP (n=1) | Efficient-MP (n=1) |
|---|---:|---:|---:|---:|---:|---:|
| `clean` | 459.1 ± 42.7 ▲ | 442.5 ± 7.3 ▲ | 432.0 ± 10.5 ▲ | 445.0 ± 5.6 ▲ | 481.5 ▲ | 477.2 ▲ |
| `gauss2` | 584.4 ± 17.3 (+27.3%) ▲ | 591.1 ± 48.3 (+33.6%) ▲ | 584.4 ± 17.3 (+35.3%) ▲ | 591.1 ± 48.3 (+32.8%) ▲ | 541.8 (+12.5%) ▲ | 469.4 (-1.6%) ▲ |
| `mask0p25` | 550.4 ± 72.4 (+19.9%) ▲ | 457.2 ± 8.9 (+3.3%) ▲ | 508.2 ± 45.9 (+17.6%) ▲ | 473.7 ± 36.6 (+6.5%) ▲ | 550.5 (+14.3%) ▲ | 563.0 (+18.0%) ▲ |
| `mask0p5` | 663.6 ± 70.1 (+44.5%) ▲ | 514.7 ± 64.2 (+16.3%) ▲ | 601.5 ± 98.0 (+39.2%) ▲ | 506.1 ± 81.9 (+13.7%) ▲ | 814.4 (+69.1%) ▲ | 779.8 (+63.4%) ▲ |
| `mask0p25_intersection` | 868.3 ± 165.7 (+89.1%) ▲ | 483.8 ± 27.0 (+9.3%) ▲ | 472.5 ± 20.7 (+9.4%) ▲ | 446.1 ± 15.9 (+0.3%) ▲ | 1082.4 (+124.8%) ▲ | 1088.2 (+128.0%) ▲ |
| `mask0p5_intersection` | 975.5 ± 231.3 (+112.5%) ▲ | 527.6 ± 35.3 (+19.2%) ▲ | 490.5 ± 39.2 (+13.6%) ▲ | 450.1 ± 21.1 (+1.1%) ▲ | 1230.8 (+155.6%) ▲ | 1207.7 (+153.1%) ▲ |
| `mask0p5_hold_t60` | 559.2 ± 55.8 (+21.8%) ▲ | 501.9 ± 14.2 (+13.4%) ▲ | 491.3 ± 4.3 (+13.7%) ▲ | 498.8 ± 25.3 (+12.1%) ▲ | 649.4 (+34.9%) ▲ | 623.0 (+30.6%) ▲ |
| `drift0p3` | 447.6 ± 13.4 (-2.5%) ▲ | 474.4 ± 27.2 (+7.2%) ▲ | 440.9 ± 5.2 (+2.1%) ▲ | 478.5 ± 48.1 (+7.5%) ▲ | 481.2 (-0.1%) ▲ | 460.9 (-3.4%) ▲ |

* **rdzeń GAT+PPO+CTDE** — najgorszy scenariusz `mask0p5_intersection`: 975.5 s (+112.5%). Gorzej niż FixedTime w 7 scenariuszach: `drift0p3`, `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **rdzeń z awariami (DR)** — najgorszy scenariusz `gauss2`: 591.1 s (+33.6%). Gorzej niż FixedTime w 7 scenariuszach: `drift0p3`, `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **rdzeń + bramka-reguła** — najgorszy scenariusz `mask0p5`: 601.5 s (+39.2%). Gorzej niż FixedTime w 7 scenariuszach: `drift0p3`, `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **DR + bramka-reguła** — najgorszy scenariusz `gauss2`: 591.1 s (+32.8%). Gorzej niż FixedTime w 7 scenariuszach: `drift0p3`, `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **Advanced-MP** — najgorszy scenariusz `mask0p5_intersection`: 1230.8 s (+155.6%). Gorzej niż FixedTime w 7 scenariuszach: `drift0p3`, `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **Efficient-MP** — najgorszy scenariusz `mask0p5_intersection`: 1207.7 s (+153.1%). Gorzej niż FixedTime w 7 scenariuszach: `drift0p3`, `gauss2`, `mask0p25`, `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.


![krzywe degradacji sumo_krakow3w_high](figures/t2_krakow_sumo_krakow3w_high.png)

## Ramię: rdzeń z awariami (DR) vs rdzeń GAT+PPO+CTDE

Kolumny mają RÓŻNE checkpointy (osobne treningi), więc parujemy po (sieć, scenariusz, seed, losowanie): wspólne zostają ruch i wzorzec awarii, rozrzut między seedami wchodzi do różnicy. Kolumna odniesienia jest wskazana w planie (`baseline: gatppo`), a nie domyślona. `Δ` < 0 = ramię lepsze. `|t| ≥ 2.8` to próg po poprawce na tyle porównań, ile ma ta tabela.

Kolumny różni SPOSÓB TRENOWANIA, nie mechanizm włączany w inferencji, więc nie ma diagnostyki „czy mechanizm wystartował” — jest albo go nie ma już na poziomie wag. Rozrzut między seedami wchodzi tu do różnicy w całości (patrz zdanie o parowaniu wyżej), a liczba NIEZALEŻNYCH replikacji to liczba seedów, nie liczba par: losowania awarii dzielą checkpoint, więc |t| jest wobec nich optymistyczne.

### sumo_krakow3w_mid

| scenariusz | Δ ATT [s] | sd | t | lepszy w |
|---|---:|---:|---:|---:|
| `clean` | -17.67 | 44.7 | -0.69 | 2/3 |
| `gauss2` | +3.86 | 95.7 | +0.12 | 7/9 |
| `mask0p25` | -50.56 | 96.2 | -1.58 | 7/9 |
| `mask0p5` | -118.22 | 230.5 | -1.54 | 7/9 |
| `mask0p25_intersection` | **-397.54** | 323.1 | -3.69 | 7/9 |
| `mask0p5_intersection` | **-494.74** | 322.6 | -4.60 | 8/9 |
| `mask0p5_hold_t60` | **-64.41** | 51.6 | -3.75 | 8/9 |
| `drift0p3` | -9.28 | 22.7 | -1.22 | 6/9 |

* Powyżej progu: `mask0p25_intersection` (-397.54 s, t = -3.69, lepszy w 7/9 parach), `mask0p5_hold_t60` (-64.41 s, t = -3.75, lepszy w 8/9 parach), `mask0p5_intersection` (-494.74 s, t = -4.60, lepszy w 8/9 parach)

### sumo_krakow3w_low

| scenariusz | Δ ATT [s] | sd | t | lepszy w |
|---|---:|---:|---:|---:|
| `clean` | **+3.94** | 1.4 | +4.72 | 0/3 |
| `gauss2` | **-36.69** | 17.7 | -6.20 | 9/9 |
| `mask0p25` | **-5.71** | 5.0 | -3.44 | 8/9 |
| `mask0p5` | -57.00 | 81.2 | -2.11 | 8/9 |
| `mask0p25_intersection` | **-344.11** | 259.2 | -3.98 | 9/9 |
| `mask0p5_intersection` | **-412.96** | 269.0 | -4.61 | 9/9 |
| `mask0p5_hold_t60` | -21.44 | 23.3 | -2.75 | 9/9 |
| `drift0p3` | +1.47 | 6.3 | +0.70 | 1/9 |

* Powyżej progu: `clean` (+3.94 s, t = +4.72, lepszy w 0/3 parach), `gauss2` (-36.69 s, t = -6.20, lepszy w 9/9 parach), `mask0p25` (-5.71 s, t = -3.44, lepszy w 8/9 parach), `mask0p25_intersection` (-344.11 s, t = -3.98, lepszy w 9/9 parach), `mask0p5_intersection` (-412.96 s, t = -4.61, lepszy w 9/9 parach)

### sumo_krakow3w_high

| scenariusz | Δ ATT [s] | sd | t | lepszy w |
|---|---:|---:|---:|---:|
| `clean` | -16.67 | 48.4 | -0.60 | 1/3 |
| `gauss2` | +6.70 | 103.4 | +0.19 | 5/9 |
| `mask0p25` | -93.14 | 157.0 | -1.78 | 7/9 |
| `mask0p5` | -148.95 | 234.5 | -1.91 | 7/9 |
| `mask0p25_intersection` | **-384.48** | 296.0 | -3.90 | 9/9 |
| `mask0p5_intersection` | **-447.85** | 293.3 | -4.58 | 8/9 |
| `mask0p5_hold_t60` | -57.25 | 73.2 | -2.35 | 6/9 |
| `drift0p3` | +26.87 | 44.8 | +1.80 | 5/9 |

* Powyżej progu: `mask0p25_intersection` (-384.48 s, t = -3.90, lepszy w 9/9 parach), `mask0p5_intersection` (-447.85 s, t = -4.58, lepszy w 8/9 parach)

## Mechanizm: rdzeń + bramka-reguła vs rdzeń GAT+PPO+CTDE

Obie kolumny liczą się z TEGO SAMEGO checkpointu i tych samych losowań awarii (`draw` wybiera, które detektory padły), więc porównujemy je **parami**, nie przez ±std z tabeli wyżej: wspólny rozrzut między seedami i losowaniami skraca się w różnicy. `Δ` < 0 = mechanizm lepszy. `|t| ≥ 2.8` to próg po poprawce na tyle porównań, ile ma ta tabela.

`gate_share` = odsetek par (decyzja, węzeł), w których sterowanie przejęła reguła. `gate_switches` = liczba ZMIAN trybu na węzłach w epizodzie — to one psują ruch (przełączenie resetuje fazę), a nie sam udział. `gate_precision` i `gate_recall` liczą się względem WYROCZNI (prawda z warstwy awarii, wyłącznie diagnostycznie) i rozdzielają „detektor jest słaby” od „przełączanie mało daje” — dwie rzeczy, które samo ATT skleja.

### sumo_krakow3w_mid

| scenariusz | gate_share | gate_switches | gate_precision | gate_recall | Δ ATT [s] | sd | t | lepszy w |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 0.1% | 2.00 | 0.00 | — | -15.21 | 48.5 | -0.54 | 2/3 |
| `gauss2` | 0.0% | 0.00 | — | — | +0.00 | 0.0 | — | 0/0 |
| `mask0p25` | 3.5% | 24.89 | 0.41 | 0.19 | -32.96 | 60.7 | -1.63 | 5/9 |
| `mask0p5` | 16.7% | 56.33 | 0.92 | 0.30 | -60.95 | 88.3 | -2.07 | 7/9 |
| `mask0p25_intersection` | 25.8% | 39.56 | 0.99 | 0.78 | **-429.83** | 333.8 | -3.86 | 9/9 |
| `mask0p5_intersection` | 38.1% | 43.78 | 1.00 | 0.81 | **-520.12** | 303.9 | -5.13 | 9/9 |
| `mask0p5_hold_t60` | 12.0% | 40.00 | 0.93 | 0.29 | **-80.57** | 59.5 | -4.07 | 8/9 |
| `drift0p3` | 0.2% | 2.89 | 0.00 | — | +1.03 | 15.6 | +0.20 | 3/9 |

* Powyżej progu: `mask0p25_intersection` (-429.83 s, t = -3.86, lepszy w 9/9 parach), `mask0p5_hold_t60` (-80.57 s, t = -4.07, lepszy w 8/9 parach), `mask0p5_intersection` (-520.12 s, t = -5.13, lepszy w 9/9 parach)

### sumo_krakow3w_low

| scenariusz | gate_share | gate_switches | gate_precision | gate_recall | Δ ATT [s] | sd | t | lepszy w |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 2.3% | 17.33 | 0.00 | — | **+0.29** | 0.2 | +3.28 | 0/3 |
| `gauss2` | 0.0% | 0.00 | — | — | +0.00 | 0.0 | — | 0/0 |
| `mask0p25` | 5.5% | 46.00 | 0.18 | 0.11 | -2.14 | 4.7 | -1.38 | 7/9 |
| `mask0p5` | 12.7% | 69.56 | 0.88 | 0.22 | -27.22 | 47.6 | -1.72 | 7/9 |
| `mask0p25_intersection` | 28.0% | 58.78 | 0.92 | 0.79 | **-359.37** | 264.2 | -4.08 | 9/9 |
| `mask0p5_intersection` | 40.1% | 55.22 | 0.96 | 0.83 | **-443.53** | 268.8 | -4.95 | 9/9 |
| `mask0p5_hold_t60` | 10.1% | 54.89 | 0.80 | 0.21 | -14.87 | 45.2 | -0.99 | 6/9 |
| `drift0p3` | 2.3% | 21.00 | 0.00 | — | -1.82 | 6.3 | -0.87 | 4/9 |

* Powyżej progu: `clean` (+0.29 s, t = +3.28, lepszy w 0/3 parach), `mask0p25_intersection` (-359.37 s, t = -4.08, lepszy w 9/9 parach), `mask0p5_intersection` (-443.53 s, t = -4.95, lepszy w 9/9 parach)

### sumo_krakow3w_high

| scenariusz | gate_share | gate_switches | gate_precision | gate_recall | Δ ATT [s] | sd | t | lepszy w |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 0.2% | 5.00 | 0.00 | — | -27.15 | 39.4 | -1.19 | 2/3 |
| `gauss2` | 0.0% | 0.00 | — | — | +0.00 | 0.0 | — | 0/0 |
| `mask0p25` | 3.5% | 25.89 | 0.52 | 0.22 | -42.12 | 112.4 | -1.12 | 5/9 |
| `mask0p5` | 15.9% | 59.89 | 0.92 | 0.28 | -62.15 | 248.4 | -0.75 | 7/9 |
| `mask0p25_intersection` | 23.7% | 51.44 | 0.99 | 0.70 | **-395.80** | 304.7 | -3.90 | 8/9 |
| `mask0p5_intersection` | 37.0% | 49.33 | 0.99 | 0.79 | **-484.93** | 258.6 | -5.63 | 9/9 |
| `mask0p5_hold_t60` | 11.2% | 43.56 | 0.90 | 0.26 | **-67.85** | 68.0 | -2.99 | 8/9 |
| `drift0p3` | 0.3% | 4.78 | 0.00 | — | -6.65 | 23.5 | -0.85 | 6/9 |

* Powyżej progu: `mask0p25_intersection` (-395.80 s, t = -3.90, lepszy w 8/9 parach), `mask0p5_hold_t60` (-67.85 s, t = -2.99, lepszy w 8/9 parach), `mask0p5_intersection` (-484.93 s, t = -5.63, lepszy w 9/9 parach)

## Mechanizm: DR + bramka-reguła vs rdzeń z awariami (DR)

Obie kolumny liczą się z TEGO SAMEGO checkpointu i tych samych losowań awarii (`draw` wybiera, które detektory padły), więc porównujemy je **parami**, nie przez ±std z tabeli wyżej: wspólny rozrzut między seedami i losowaniami skraca się w różnicy. `Δ` < 0 = mechanizm lepszy. `|t| ≥ 2.8` to próg po poprawce na tyle porównań, ile ma ta tabela.

`gate_share` = odsetek par (decyzja, węzeł), w których sterowanie przejęła reguła. `gate_switches` = liczba ZMIAN trybu na węzłach w epizodzie — to one psują ruch (przełączenie resetuje fazę), a nie sam udział. `gate_precision` i `gate_recall` liczą się względem WYROCZNI (prawda z warstwy awarii, wyłącznie diagnostycznie) i rozdzielają „detektor jest słaby” od „przełączanie mało daje” — dwie rzeczy, które samo ATT skleja.

### sumo_krakow3w_mid

| scenariusz | gate_share | gate_switches | gate_precision | gate_recall | Δ ATT [s] | sd | t | lepszy w |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 0.2% | 3.33 | 0.00 | — | +0.67 | 7.1 | +0.16 | 1/3 |
| `gauss2` | 0.0% | 0.00 | — | — | +0.00 | 0.0 | — | 0/0 |
| `mask0p25` | 3.4% | 23.00 | 0.46 | 0.20 | +1.74 | 15.5 | +0.34 | 3/9 |
| `mask0p5` | 16.1% | 46.00 | 0.94 | 0.29 | -1.05 | 24.9 | -0.13 | 5/9 |
| `mask0p25_intersection` | 24.4% | 40.56 | 0.99 | 0.73 | -57.48 | 79.2 | -2.18 | 7/9 |
| `mask0p5_intersection` | 37.6% | 41.89 | 1.00 | 0.80 | **-58.94** | 34.7 | -5.09 | 9/9 |
| `mask0p5_hold_t60` | 11.9% | 34.44 | 0.94 | 0.29 | -26.31 | 40.4 | -1.95 | 7/9 |
| `drift0p3` | 0.2% | 3.33 | 0.00 | — | +0.95 | 16.7 | +0.17 | 4/8 |

* Powyżej progu: `mask0p5_intersection` (-58.94 s, t = -5.09, lepszy w 9/9 parach)

### sumo_krakow3w_low

| scenariusz | gate_share | gate_switches | gate_precision | gate_recall | Δ ATT [s] | sd | t | lepszy w |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 2.6% | 22.33 | 0.00 | — | +0.05 | 1.6 | +0.06 | 1/3 |
| `gauss2` | 0.0% | 0.00 | — | — | +0.00 | 0.0 | — | 0/0 |
| `mask0p25` | 5.3% | 44.00 | 0.19 | 0.11 | +0.46 | 1.3 | +1.02 | 4/9 |
| `mask0p5` | 12.9% | 65.11 | 0.88 | 0.22 | -2.31 | 6.9 | -1.00 | 3/9 |
| `mask0p25_intersection` | 26.7% | 51.44 | 0.92 | 0.76 | **-24.10** | 23.1 | -3.13 | 6/9 |
| `mask0p5_intersection` | 39.3% | 53.78 | 0.96 | 0.81 | **-36.09** | 12.5 | -8.65 | 9/9 |
| `mask0p5_hold_t60` | 10.0% | 57.33 | 0.80 | 0.21 | -2.82 | 26.2 | -0.32 | 5/9 |
| `drift0p3` | 2.5% | 22.89 | 0.00 | — | -0.11 | 0.9 | -0.38 | 3/9 |

* Powyżej progu: `mask0p25_intersection` (-24.10 s, t = -3.13, lepszy w 6/9 parach), `mask0p5_intersection` (-36.09 s, t = -8.65, lepszy w 9/9 parach)

### sumo_krakow3w_high

| scenariusz | gate_share | gate_switches | gate_precision | gate_recall | Δ ATT [s] | sd | t | lepszy w |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 0.2% | 3.33 | 0.00 | — | +2.55 | 6.4 | +0.69 | 1/2 |
| `gauss2` | 0.0% | 0.00 | — | — | +0.00 | 0.0 | — | 0/0 |
| `mask0p25` | 3.3% | 23.56 | 0.46 | 0.19 | +16.52 | 28.6 | +1.74 | 4/9 |
| `mask0p5` | 15.4% | 46.56 | 0.95 | 0.28 | -8.58 | 54.1 | -0.48 | 6/9 |
| `mask0p25_intersection` | 22.5% | 38.67 | 0.99 | 0.68 | -37.65 | 50.1 | -2.25 | 7/9 |
| `mask0p5_intersection` | 36.0% | 41.11 | 1.00 | 0.77 | **-77.55** | 75.3 | -3.09 | 8/9 |
| `mask0p5_hold_t60` | 11.1% | 38.22 | 0.93 | 0.27 | -3.14 | 45.0 | -0.21 | 5/9 |
| `drift0p3` | 0.2% | 3.33 | 0.00 | — | +4.09 | 38.3 | +0.32 | 4/6 |

* Powyżej progu: `mask0p5_intersection` (-77.55 s, t = -3.09, lepszy w 8/9 parach)

## Kontrola: ewaluator mierzy tę samą politykę co T1

Czysty wynik ewaluatora vs ostatni test treningu, 6 biegów uczonych: największa różnica **0.000 s** — zgodne co do precyzji logu.

## Czego ta tabela NIE mówi

* Poziomy są poziomami DETEKTORA. Na cechę `gauss` przenosi się ze współczynnikiem ~0,8–0,9, `urand` jest ~1,6× słabszy przy tym samym `level` (`results/sensor_severity.md`).
* Maska z podstawieniem `zero` zmienia znacznie mniej komórek cech, niż wynosi ułamek martwych pasów (zero jest wartością modalną) — ta sama liczba nie znaczy tego samego na sieciach o różnym obciążeniu.
* Scenariusze `hold` zaczynają się w decyzji 60 (15. minuta), więc pierwsza ćwiartka epizodu jest czysta — ich Δ nie jest porównywalne wprost z maską od początku.
