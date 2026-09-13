# T2 bez mechanizmu — plan `t2_ood`

Tryb realistic (yellow=5), 3 losowań awarii na scenariusz, ewaluacja ZAMROŻONYCH sterowników (`tsc degrade`). Wszystkie czytają tę samą warstwę detektorów: MaxPressure liczy decyzję z popsutego wektora stanu, nie z silnika.

Komórka = ATT [s], średnia po losowaniach, potem po seedach polityki (± std po seedach); w nawiasie Δ względem czystego wyniku tego samego sterownika. **▲ = gorzej niż FixedTime** — tylko tam dolna warstwa fallbacku ma pole do zysku.

## cityflow_jinan3x4_4ph

FixedTime (niezależny od odczytów): **507.0 s**.

| scenariusz | rdzeń GAT+PPO+CTDE (n=5) | Advanced-CoLight (n=5) | ensemble 5 actorów (decyduje seed 0) (n=1) | Efficient-MP (n=1) | Advanced-MP (n=1) |
|---|---:|---:|---:|---:|---:|
| `clean` | 295.8 ± 1.2 | 294.4 ± 1.8 | 295.2 | 313.9 | 301.8 |
| `gauss2` | 319.0 ± 2.2 (+7.9%) | 325.8 ± 7.1 (+10.7%) | 319.4 (+8.2%) | 328.9 (+4.8%) | 318.4 (+5.5%) |

* **rdzeń GAT+PPO+CTDE** — najgorszy scenariusz `gauss2`: 319.0 s (+7.9%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **Advanced-CoLight** — najgorszy scenariusz `gauss2`: 325.8 s (+10.7%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **ensemble 5 actorów (decyduje seed 0)** — najgorszy scenariusz `gauss2`: 319.4 s (+8.2%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **Efficient-MP** — najgorszy scenariusz `gauss2`: 328.9 s (+4.8%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **Advanced-MP** — najgorszy scenariusz `gauss2`: 318.4 s (+5.5%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.


![krzywe degradacji cityflow_jinan3x4_4ph](figures/t2_ood_cityflow_jinan3x4_4ph.png)

## cityflow_jinan3x4_4ph_d0p25

FixedTime (niezależny od odczytów): **322.3 s**.

| scenariusz | rdzeń GAT+PPO+CTDE (n=5) | Advanced-CoLight (n=5) | ensemble 5 actorów (decyduje seed 0) (n=1) | Efficient-MP (n=1) | Advanced-MP (n=1) |
|---|---:|---:|---:|---:|---:|
| `clean` | 253.2 ± 1.7 | 252.9 ± 1.1 | 250.4 | 258.8 | 256.8 |
| `gauss2` | 294.3 ± 4.2 (+16.2%) | 296.7 ± 6.2 (+17.3%) | 295.3 (+17.9%) | 288.4 (+11.4%) | 303.5 (+18.2%) |

* **rdzeń GAT+PPO+CTDE** — najgorszy scenariusz `gauss2`: 294.3 s (+16.2%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **Advanced-CoLight** — najgorszy scenariusz `gauss2`: 296.7 s (+17.3%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **ensemble 5 actorów (decyduje seed 0)** — najgorszy scenariusz `gauss2`: 295.3 s (+17.9%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **Efficient-MP** — najgorszy scenariusz `gauss2`: 288.4 s (+11.4%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **Advanced-MP** — najgorszy scenariusz `gauss2`: 303.5 s (+18.2%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.


![krzywe degradacji cityflow_jinan3x4_4ph_d0p25](figures/t2_ood_cityflow_jinan3x4_4ph_d0p25.png)

## cityflow_jinan3x4_4ph_r2000

FixedTime (niezależny od odczytów): **412.1 s**.

| scenariusz | rdzeń GAT+PPO+CTDE (n=5) | Advanced-CoLight (n=5) | ensemble 5 actorów (decyduje seed 0) (n=1) | Efficient-MP (n=1) | Advanced-MP (n=1) |
|---|---:|---:|---:|---:|---:|
| `clean` | 283.6 ± 0.6 | 283.1 ± 0.7 | 283.0 | 288.4 | 288.1 |
| `gauss2` | 310.0 ± 2.3 (+9.3%) | 314.4 ± 5.4 (+11.0%) | 310.6 (+9.8%) | 308.4 (+6.9%) | 313.9 (+9.0%) |

* **rdzeń GAT+PPO+CTDE** — najgorszy scenariusz `gauss2`: 310.0 s (+9.3%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **Advanced-CoLight** — najgorszy scenariusz `gauss2`: 314.4 s (+11.0%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **ensemble 5 actorów (decyduje seed 0)** — najgorszy scenariusz `gauss2`: 310.6 s (+9.8%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **Efficient-MP** — najgorszy scenariusz `gauss2`: 308.4 s (+6.9%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **Advanced-MP** — najgorszy scenariusz `gauss2`: 313.9 s (+9.0%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.


![krzywe degradacji cityflow_jinan3x4_4ph_r2000](figures/t2_ood_cityflow_jinan3x4_4ph_r2000.png)

## cityflow_hangzhou4x4_4ph

FixedTime (niezależny od odczytów): **568.9 s**.

| scenariusz | rdzeń GAT+PPO+CTDE (n=5) | Advanced-CoLight (n=5) | ensemble 5 actorów (decyduje seed 0) (n=1) | Efficient-MP (n=1) | Advanced-MP (n=1) |
|---|---:|---:|---:|---:|---:|
| `clean` | 317.5 ± 0.7 | 317.4 ± 0.5 | 318.1 | 331.2 | 325.5 |
| `gauss2` | 373.9 ± 4.8 (+17.8%) | 382.8 ± 7.6 (+20.6%) | 366.8 (+15.3%) | 356.4 (+7.6%) | 359.6 (+10.5%) |

* **rdzeń GAT+PPO+CTDE** — najgorszy scenariusz `gauss2`: 373.9 s (+17.8%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **Advanced-CoLight** — najgorszy scenariusz `gauss2`: 382.8 s (+20.6%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **ensemble 5 actorów (decyduje seed 0)** — najgorszy scenariusz `gauss2`: 366.8 s (+15.3%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **Efficient-MP** — najgorszy scenariusz `gauss2`: 356.4 s (+7.6%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **Advanced-MP** — najgorszy scenariusz `gauss2`: 359.6 s (+10.5%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.


![krzywe degradacji cityflow_hangzhou4x4_4ph](figures/t2_ood_cityflow_hangzhou4x4_4ph.png)

## cityflow_hangzhou4x4_4ph_r5816

FixedTime (niezależny od odczytów): **559.5 s**.

| scenariusz | rdzeń GAT+PPO+CTDE (n=5) | Advanced-CoLight (n=5) | ensemble 5 actorów (decyduje seed 0) (n=1) | Efficient-MP (n=1) | Advanced-MP (n=1) |
|---|---:|---:|---:|---:|---:|
| `clean` | 420.1 ± 1.5 | 437.7 ± 15.1 | 421.9 | 428.0 | 419.0 |
| `gauss2` | 452.3 ± 4.3 (+7.7%) | 465.4 ± 9.0 (+6.3%) | 446.6 (+5.8%) | 441.0 (+3.0%) | 436.3 (+4.1%) |

* **rdzeń GAT+PPO+CTDE** — najgorszy scenariusz `gauss2`: 452.3 s (+7.7%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **Advanced-CoLight** — najgorszy scenariusz `gauss2`: 465.4 s (+6.3%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **ensemble 5 actorów (decyduje seed 0)** — najgorszy scenariusz `gauss2`: 446.6 s (+5.8%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **Efficient-MP** — najgorszy scenariusz `gauss2`: 441.0 s (+3.0%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **Advanced-MP** — najgorszy scenariusz `gauss2`: 436.3 s (+4.1%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.


![krzywe degradacji cityflow_hangzhou4x4_4ph_r5816](figures/t2_ood_cityflow_hangzhou4x4_4ph_r5816.png)

## Kontrola: ewaluator mierzy tę samą politykę co T1

Czysty wynik ewaluatora vs ostatni test treningu, 22 biegów uczonych: największa różnica **0.047 s** — zgodne co do precyzji logu.

## Czego ta tabela NIE mówi

* Poziomy są poziomami DETEKTORA. Na cechę `gauss` przenosi się ze współczynnikiem ~0,8–0,9, `urand` jest ~1,6× słabszy przy tym samym `level` (`results/sensor_severity.md`).
* Maska z podstawieniem `zero` zmienia znacznie mniej komórek cech, niż wynosi ułamek martwych pasów (zero jest wartością modalną) — ta sama liczba nie znaczy tego samego na sieciach o różnym obciążeniu.
* Scenariusze `hold` zaczynają się w decyzji 60 (15. minuta), więc pierwsza ćwiartka epizodu jest czysta — ich Δ nie jest porównywalne wprost z maską od początku.
