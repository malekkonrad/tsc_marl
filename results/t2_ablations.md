# T2 bez mechanizmu — plan `t2_ablations`

Tryb realistic (yellow=5), 5 losowań awarii na scenariusz, ewaluacja ZAMROŻONYCH sterowników (`tsc degrade`). Wszystkie czytają tę samą warstwę detektorów: MaxPressure liczy decyzję z popsutego wektora stanu, nie z silnika.

Komórka = ATT [s], średnia po losowaniach, potem po seedach polityki (± std po seedach); w nawiasie Δ względem czystego wyniku tego samego sterownika. **▲ = gorzej niż FixedTime** — tylko tam dolna warstwa fallbacku ma pole do zysku.

## cityflow_jinan3x4_4ph

FixedTime (niezależny od odczytów): **507.0 s**.

| scenariusz | rdzeń GAT+PPO+CTDE (n=3) | actor mean-pool (n=3) | actor bez GNN (n=3) |
|---|---:|---:|---:|
| `clean` | 295.9 ± 0.6 | 291.4 ± 0.5 | 291.2 ± 0.3 |
| `gauss1` | 304.5 ± 1.4 (+2.9%) | 301.2 ± 0.3 (+3.4%) | 301.8 ± 1.0 (+3.7%) |
| `gauss2` | 319.9 ± 2.2 (+8.1%) | 318.3 ± 1.1 (+9.2%) | 321.1 ± 1.3 (+10.3%) |
| `gauss4` | 350.0 ± 4.5 (+18.3%) | 351.4 ± 4.2 (+20.6%) | 360.7 ± 5.1 (+23.9%) |
| `mask0p1` | 354.2 ± 20.0 (+19.7%) | 367.0 ± 9.5 (+25.9%) | 364.7 ± 6.9 (+25.2%) |
| `mask0p25` | 424.9 ± 31.1 (+43.6%) | 460.9 ± 14.1 (+58.2%) | 466.3 ± 9.4 (+60.1%) |
| `mask0p5` | 732.6 ± 28.5 (+147.5%) ▲ | 854.1 ± 50.8 (+193.1%) ▲ | 773.7 ± 60.1 (+165.7%) ▲ |
| `mask0p25_intersection` | 471.1 ± 68.8 (+59.2%) | 681.7 ± 247.1 (+133.9%) ▲ | 542.4 ± 146.5 (+86.3%) ▲ |
| `mask0p5_intersection` | 687.0 ± 110.5 (+132.2%) ▲ | 926.9 ± 342.7 (+218.1%) ▲ | 753.2 ± 211.5 (+158.7%) ▲ |
| `mask0p25_hold_t60` | 354.7 ± 24.8 (+19.9%) | 322.5 ± 3.7 (+10.7%) | 336.4 ± 15.6 (+15.5%) |
| `mask0p5_hold_t60` | 475.0 ± 38.9 (+60.5%) | 491.1 ± 23.4 (+68.5%) | 488.9 ± 30.4 (+67.9%) |
| `drift0p3` | 301.5 ± 0.5 (+1.9%) | 297.9 ± 0.6 (+2.2%) | 298.5 ± 0.2 (+2.5%) |
| `drift0p5` | 328.6 ± 4.3 (+11.0%) | 330.2 ± 1.4 (+13.3%) | 329.2 ± 7.0 (+13.1%) |

* **rdzeń GAT+PPO+CTDE** — najgorszy scenariusz `mask0p5`: 732.6 s (+147.5%). Gorzej niż FixedTime w 2 scenariuszach: `mask0p5`, `mask0p5_intersection`.
* **actor mean-pool** — najgorszy scenariusz `mask0p5_intersection`: 926.9 s (+218.1%). Gorzej niż FixedTime w 3 scenariuszach: `mask0p25_intersection`, `mask0p5`, `mask0p5_intersection`.
* **actor bez GNN** — najgorszy scenariusz `mask0p5`: 773.7 s (+165.7%). Gorzej niż FixedTime w 3 scenariuszach: `mask0p25_intersection`, `mask0p5`, `mask0p5_intersection`.


![krzywe degradacji cityflow_jinan3x4_4ph](figures/t2_ablations_cityflow_jinan3x4_4ph.png)

## cityflow_hangzhou4x4_4ph

FixedTime (niezależny od odczytów): **568.9 s**.

| scenariusz | rdzeń GAT+PPO+CTDE (n=3) | actor mean-pool (n=3) | actor bez GNN (n=3) |
|---|---:|---:|---:|
| `clean` | 317.5 ± 1.0 | 316.6 ± 0.3 | 316.0 ± 0.2 |
| `gauss1` | 344.9 ± 1.1 (+8.6%) | 342.2 ± 0.3 (+8.1%) | 347.3 ± 1.3 (+9.9%) |
| `gauss2` | 372.1 ± 4.2 (+17.2%) | 366.8 ± 1.8 (+15.9%) | 381.8 ± 4.8 (+20.8%) |
| `gauss4` | 419.3 ± 13.8 (+32.1%) | 409.6 ± 8.4 (+29.4%) | 449.7 ± 21.6 (+42.3%) |
| `mask0p1` | 363.4 ± 4.2 (+14.4%) | 374.8 ± 18.4 (+18.4%) | 365.2 ± 13.4 (+15.6%) |
| `mask0p25` | 460.1 ± 21.2 (+44.9%) | 501.8 ± 64.6 (+58.5%) | 530.2 ± 53.0 (+67.8%) |
| `mask0p5` | 739.2 ± 98.5 (+132.8%) ▲ | 823.3 ± 204.6 (+160.1%) ▲ | 879.4 ± 139.6 (+178.3%) ▲ |
| `mask0p25_intersection` | 472.2 ± 58.9 (+48.7%) | 555.8 ± 172.9 (+75.6%) | 633.8 ± 51.4 (+100.6%) ▲ |
| `mask0p5_intersection` | 765.9 ± 188.2 (+141.2%) ▲ | 889.4 ± 364.1 (+180.9%) ▲ | 1054.2 ± 83.3 (+233.6%) ▲ |
| `mask0p25_hold_t60` | 411.1 ± 16.8 (+29.5%) | 393.6 ± 7.1 (+24.3%) | 444.5 ± 12.0 (+40.7%) |
| `mask0p5_hold_t60` | 544.4 ± 11.8 (+71.5%) | 605.8 ± 16.7 (+91.4%) ▲ | 613.9 ± 43.4 (+94.2%) ▲ |
| `drift0p3` | 319.5 ± 0.3 (+0.6%) | 318.4 ± 0.0 (+0.6%) | 318.5 ± 0.1 (+0.8%) |
| `drift0p5` | 339.9 ± 1.1 (+7.0%) | 338.7 ± 5.0 (+7.0%) | 335.7 ± 2.5 (+6.2%) |

* **rdzeń GAT+PPO+CTDE** — najgorszy scenariusz `mask0p5_intersection`: 765.9 s (+141.2%). Gorzej niż FixedTime w 2 scenariuszach: `mask0p5`, `mask0p5_intersection`.
* **actor mean-pool** — najgorszy scenariusz `mask0p5_intersection`: 889.4 s (+180.9%). Gorzej niż FixedTime w 3 scenariuszach: `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **actor bez GNN** — najgorszy scenariusz `mask0p5_intersection`: 1054.2 s (+233.6%). Gorzej niż FixedTime w 4 scenariuszach: `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.


![krzywe degradacji cityflow_hangzhou4x4_4ph](figures/t2_ablations_cityflow_hangzhou4x4_4ph.png)

## Kontrola: ewaluator mierzy tę samą politykę co T1

Czysty wynik ewaluatora vs ostatni test treningu, 18 biegów uczonych: największa różnica **0.000 s** — zgodne co do precyzji logu.

## Czego ta tabela NIE mówi

* Poziomy są poziomami DETEKTORA. Na cechę `gauss` przenosi się ze współczynnikiem ~0,8–0,9, `urand` jest ~1,6× słabszy przy tym samym `level` (`results/sensor_severity.md`).
* Maska z podstawieniem `zero` zmienia znacznie mniej komórek cech, niż wynosi ułamek martwych pasów (zero jest wartością modalną) — ta sama liczba nie znaczy tego samego na sieciach o różnym obciążeniu.
* Scenariusze `hold` zaczynają się w decyzji 60 (15. minuta), więc pierwsza ćwiartka epizodu jest czysta — ich Δ nie jest porównywalne wprost z maską od początku.
