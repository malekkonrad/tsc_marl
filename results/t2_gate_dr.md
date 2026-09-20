# T2 — plan `t2_gate_dr`

Tryb realistic (yellow=5), 5 losowań awarii na scenariusz, ewaluacja ZAMROŻONYCH sterowników (`tsc degrade`). Wszystkie czytają tę samą warstwę detektorów: MaxPressure liczy decyzję z popsutego wektora stanu, nie z silnika.

Komórka = ATT [s], średnia po losowaniach, potem po seedach polityki (± std po seedach); w nawiasie Δ względem czystego wyniku tego samego sterownika. **▲ = gorzej niż FixedTime** — tylko tam dolna warstwa fallbacku ma pole do zysku.

## cityflow_jinan3x4_4ph

FixedTime (niezależny od odczytów): **507.0 s**.

| scenariusz | rdzeń GAT+PPO+CTDE (n=3) | bramka: reguła (trwałość) (n=3) | rdzeń trenowany z awariami (n=3) | rdzeń z awariami + bramka: reguła (n=3) |
|---|---:|---:|---:|---:|
| `clean` | 295.9 ± 0.6 | 295.7 ± 0.8 | 302.0 ± 1.7 | 305.2 ± 1.0 |
| `gauss2` | 319.9 ± 2.2 (+8.1%) | 319.9 ± 2.2 (+8.2%) | 321.8 ± 3.2 (+6.6%) | 321.8 ± 3.2 (+5.4%) |
| `mask0p25` | 424.9 ± 31.1 (+43.6%) | 393.7 ± 10.4 (+33.2%) | 407.0 ± 12.5 (+34.8%) | 388.4 ± 4.9 (+27.3%) |
| `mask0p5` | 732.6 ± 28.5 (+147.5%) ▲ | 525.7 ± 23.8 (+77.8%) ▲ | 592.5 ± 19.4 (+96.2%) ▲ | 481.1 ± 11.3 (+57.6%) |
| `mask0p25_intersection` | 471.1 ± 68.8 (+59.2%) | 353.1 ± 5.3 (+19.4%) | 389.2 ± 42.7 (+28.9%) | 357.0 ± 8.4 (+16.9%) |
| `mask0p5_intersection` | 690.0 ± 115.3 (+133.2%) ▲ | 399.5 ± 2.9 (+35.1%) | 483.8 ± 47.4 (+60.2%) | 399.1 ± 4.7 (+30.7%) |
| `mask0p25_hold_t60` | 354.7 ± 24.8 (+19.9%) | 329.6 ± 10.4 (+11.5%) | 334.4 ± 8.7 (+10.7%) | 337.2 ± 6.1 (+10.5%) |
| `mask0p5_hold_t60` | 475.0 ± 38.9 (+60.5%) | 411.2 ± 13.6 (+39.1%) | 473.6 ± 29.4 (+56.8%) | 406.0 ± 10.6 (+33.0%) |

* **rdzeń GAT+PPO+CTDE** — najgorszy scenariusz `mask0p5`: 732.6 s (+147.5%). Gorzej niż FixedTime w 2 scenariuszach: `mask0p5`, `mask0p5_intersection`.
* **bramka: reguła (trwałość)** — najgorszy scenariusz `mask0p5`: 525.7 s (+77.8%). Gorzej niż FixedTime w 1 scenariuszach: `mask0p5`.
* **rdzeń trenowany z awariami** — najgorszy scenariusz `mask0p5`: 592.5 s (+96.2%). Gorzej niż FixedTime w 1 scenariuszach: `mask0p5`.
* **rdzeń z awariami + bramka: reguła** — najgorszy scenariusz `mask0p5`: 481.1 s (+57.6%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.


![krzywe degradacji cityflow_jinan3x4_4ph](figures/t2_gate_dr_cityflow_jinan3x4_4ph.png)

## cityflow_hangzhou4x4_4ph

FixedTime (niezależny od odczytów): **568.9 s**.

| scenariusz | rdzeń GAT+PPO+CTDE (n=3) | bramka: reguła (trwałość) (n=3) | rdzeń trenowany z awariami (n=3) | rdzeń z awariami + bramka: reguła (n=3) |
|---|---:|---:|---:|---:|
| `clean` | 317.5 ± 1.0 | 318.1 ± 0.1 | 321.2 ± 0.3 | 322.4 ± 0.2 |
| `gauss2` | 372.1 ± 4.2 (+17.2%) | 372.1 ± 4.2 (+16.9%) | 364.3 ± 10.4 (+13.4%) | 364.3 ± 10.4 (+13.0%) |
| `mask0p25` | 460.1 ± 21.2 (+44.9%) | 451.2 ± 7.6 (+41.8%) | 394.1 ± 5.9 (+22.7%) | 395.7 ± 7.6 (+22.8%) |
| `mask0p5` | 739.2 ± 98.5 (+132.8%) ▲ | 625.4 ± 38.6 (+96.6%) ▲ | 471.1 ± 36.5 (+46.7%) | 472.9 ± 29.7 (+46.7%) |
| `mask0p25_intersection` | 472.2 ± 58.9 (+48.7%) | 403.2 ± 17.0 (+26.7%) | 361.1 ± 9.4 (+12.4%) | 352.9 ± 3.1 (+9.5%) |
| `mask0p5_intersection` | 765.9 ± 188.2 (+141.2%) ▲ | 538.4 ± 43.6 (+69.2%) | 414.7 ± 1.0 (+29.1%) | 392.3 ± 7.0 (+21.7%) |
| `mask0p25_hold_t60` | 411.1 ± 16.8 (+29.5%) | 381.5 ± 21.4 (+19.9%) | 365.3 ± 7.3 (+13.7%) | 372.7 ± 11.7 (+15.6%) |
| `mask0p5_hold_t60` | 544.4 ± 11.8 (+71.5%) | 497.4 ± 22.9 (+56.4%) | 442.6 ± 30.1 (+37.8%) | 452.4 ± 38.2 (+40.3%) |

* **rdzeń GAT+PPO+CTDE** — najgorszy scenariusz `mask0p5_intersection`: 765.9 s (+141.2%). Gorzej niż FixedTime w 2 scenariuszach: `mask0p5`, `mask0p5_intersection`.
* **bramka: reguła (trwałość)** — najgorszy scenariusz `mask0p5`: 625.4 s (+96.6%). Gorzej niż FixedTime w 1 scenariuszach: `mask0p5`.
* **rdzeń trenowany z awariami** — najgorszy scenariusz `mask0p5`: 471.1 s (+46.7%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **rdzeń z awariami + bramka: reguła** — najgorszy scenariusz `mask0p5`: 472.9 s (+46.7%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.


![krzywe degradacji cityflow_hangzhou4x4_4ph](figures/t2_gate_dr_cityflow_hangzhou4x4_4ph.png)

## Mechanizm: bramka: reguła (trwałość) vs rdzeń GAT+PPO+CTDE

Obie kolumny liczą się z TEGO SAMEGO checkpointu i tych samych losowań awarii (`draw` wybiera, które detektory padły), więc porównujemy je **parami**, nie przez ±std z tabeli wyżej: wspólny rozrzut między seedami i losowaniami skraca się w różnicy. `Δ` < 0 = mechanizm lepszy. `|t| ≥ 2.8` to próg po poprawce na tyle porównań, ile ma ta tabela.

`gate_share` = odsetek par (decyzja, węzeł), w których sterowanie przejęła reguła. `gate_switches` = liczba ZMIAN trybu na węzłach w epizodzie — to one psują ruch (przełączenie resetuje fazę), a nie sam udział. `gate_precision` i `gate_recall` liczą się względem WYROCZNI (prawda z warstwy awarii, wyłącznie diagnostycznie) i rozdzielają „detektor jest słaby” od „przełączanie mało daje” — dwie rzeczy, które samo ATT skleja.

### cityflow_jinan3x4_4ph

| scenariusz | gate_share | gate_switches | gate_precision | gate_recall | Δ ATT [s] | sd | t | lepszy w |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 1.6% | 11.33 | 0.00 | — | -0.26 | 0.6 | -0.74 | 2/3 |
| `gauss2` | 0.0% | 0.00 | — | — | +0.00 | 0.0 | — | 0/0 |
| `mask0p25` | 15.8% | 34.80 | 0.17 | 0.31 | **-31.21** | 42.5 | -2.85 | 14/15 |
| `mask0p5` | 57.4% | 40.20 | 0.73 | 0.70 | **-206.88** | 59.2 | -13.55 | 15/15 |
| `mask0p25_intersection` | 26.2% | 18.67 | 0.89 | 0.90 | **-118.02** | 88.8 | -5.15 | 15/15 |
| `mask0p5_intersection` | 49.2% | 23.07 | 0.92 | 0.95 | **-290.45** | 148.6 | -7.57 | 15/15 |
| `mask0p25_hold_t60` | 11.0% | 25.47 | 0.15 | 0.27 | -25.11 | 43.3 | -2.25 | 11/15 |
| `mask0p5_hold_t60` | 42.1% | 34.87 | 0.71 | 0.67 | **-63.76** | 52.2 | -4.73 | 13/15 |

* Powyżej progu: `mask0p25` (-31.21 s, t = -2.85, lepszy w 14/15 parach), `mask0p25_intersection` (-118.02 s, t = -5.15, lepszy w 15/15 parach), `mask0p5` (-206.88 s, t = -13.55, lepszy w 15/15 parach), `mask0p5_hold_t60` (-63.76 s, t = -4.73, lepszy w 13/15 parach), `mask0p5_intersection` (-290.45 s, t = -7.57, lepszy w 15/15 parach)

### cityflow_hangzhou4x4_4ph

| scenariusz | gate_share | gate_switches | gate_precision | gate_recall | Δ ATT [s] | sd | t | lepszy w |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 1.3% | 14.00 | 0.00 | — | +0.62 | 1.0 | +1.08 | 1/3 |
| `gauss2` | 0.0% | 0.00 | — | — | +0.00 | 0.0 | — | 0/0 |
| `mask0p25` | 3.4% | 25.20 | 0.26 | 0.08 | -8.93 | 32.3 | -1.07 | 7/15 |
| `mask0p5` | 14.4% | 58.67 | 0.89 | 0.19 | **-113.77** | 86.1 | -5.12 | 14/15 |
| `mask0p25_intersection` | 6.6% | 33.00 | 0.74 | 0.23 | **-68.98** | 83.8 | -3.19 | 12/15 |
| `mask0p5_intersection` | 18.7% | 59.80 | 0.95 | 0.36 | **-227.48** | 174.0 | -5.06 | 14/15 |
| `mask0p25_hold_t60` | 1.6% | 16.33 | 0.02 | 0.00 | -29.56 | 56.5 | -2.03 | 12/15 |
| `mask0p5_hold_t60` | 7.0% | 38.80 | 0.66 | 0.10 | -46.99 | 73.2 | -2.49 | 12/15 |

* Powyżej progu: `mask0p25_intersection` (-68.98 s, t = -3.19, lepszy w 12/15 parach), `mask0p5` (-113.77 s, t = -5.12, lepszy w 14/15 parach), `mask0p5_intersection` (-227.48 s, t = -5.06, lepszy w 14/15 parach)

## Ramię: rdzeń trenowany z awariami vs rdzeń GAT+PPO+CTDE

Kolumny mają RÓŻNE checkpointy (osobne treningi), więc parujemy po (sieć, scenariusz, seed, losowanie): wspólne zostają ruch i wzorzec awarii, rozrzut między seedami wchodzi do różnicy. Kolumna odniesienia jest wskazana w planie (`baseline: gatppo`), a nie domyślona. `Δ` < 0 = ramię lepsze. `|t| ≥ 2.8` to próg po poprawce na tyle porównań, ile ma ta tabela.

Kolumny różni SPOSÓB TRENOWANIA, nie mechanizm włączany w inferencji, więc nie ma diagnostyki „czy mechanizm wystartował” — jest albo go nie ma już na poziomie wag. Rozrzut między seedami wchodzi tu do różnicy w całości (patrz zdanie o parowaniu wyżej), a liczba NIEZALEŻNYCH replikacji to liczba seedów, nie liczba par: losowania awarii dzielą checkpoint, więc |t| jest wobec nich optymistyczne.

### cityflow_jinan3x4_4ph

| scenariusz | Δ ATT [s] | sd | t | lepszy w |
|---|---:|---:|---:|---:|
| `clean` | **+6.06** | 2.2 | +4.76 | 0/3 |
| `gauss2` | +1.94 | 4.7 | +1.59 | 6/15 |
| `mask0p25` | -17.92 | 26.4 | -2.63 | 12/15 |
| `mask0p5` | **-140.03** | 50.3 | -10.77 | 15/15 |
| `mask0p25_intersection` | -81.83 | 115.0 | -2.76 | 12/15 |
| `mask0p5_intersection` | **-206.15** | 170.4 | -4.69 | 14/15 |
| `mask0p25_hold_t60` | -20.35 | 47.7 | -1.65 | 8/15 |
| `mask0p5_hold_t60` | -1.42 | 85.2 | -0.06 | 7/15 |

* Powyżej progu: `clean` (+6.06 s, t = +4.76, lepszy w 0/3 parach), `mask0p5` (-140.03 s, t = -10.77, lepszy w 15/15 parach), `mask0p5_intersection` (-206.15 s, t = -4.69, lepszy w 14/15 parach)

### cityflow_hangzhou4x4_4ph

| scenariusz | Δ ATT [s] | sd | t | lepszy w |
|---|---:|---:|---:|---:|
| `clean` | **+3.66** | 0.7 | +8.57 | 0/3 |
| `gauss2` | **-7.76** | 6.5 | -4.61 | 12/15 |
| `mask0p25` | **-66.06** | 68.9 | -3.71 | 14/15 |
| `mask0p5` | **-268.04** | 157.6 | -6.59 | 15/15 |
| `mask0p25_intersection` | **-111.06** | 152.9 | -2.81 | 12/15 |
| `mask0p5_intersection` | **-351.20** | 230.9 | -5.89 | 15/15 |
| `mask0p25_hold_t60` | **-45.85** | 51.5 | -3.45 | 14/15 |
| `mask0p5_hold_t60` | **-101.78** | 78.1 | -5.05 | 13/15 |

* Powyżej progu: `clean` (+3.66 s, t = +8.57, lepszy w 0/3 parach), `gauss2` (-7.76 s, t = -4.61, lepszy w 12/15 parach), `mask0p25` (-66.06 s, t = -3.71, lepszy w 14/15 parach), `mask0p25_hold_t60` (-45.85 s, t = -3.45, lepszy w 14/15 parach), `mask0p25_intersection` (-111.06 s, t = -2.81, lepszy w 12/15 parach), `mask0p5` (-268.04 s, t = -6.59, lepszy w 15/15 parach), `mask0p5_hold_t60` (-101.78 s, t = -5.05, lepszy w 13/15 parach), `mask0p5_intersection` (-351.20 s, t = -5.89, lepszy w 15/15 parach)

## Mechanizm: rdzeń z awariami + bramka: reguła vs rdzeń trenowany z awariami

Kolumny mają RÓŻNE checkpointy (osobne treningi), więc parujemy po (sieć, scenariusz, seed, losowanie): wspólne zostają ruch i wzorzec awarii, rozrzut między seedami wchodzi do różnicy. Kolumna odniesienia jest wskazana w planie (`baseline: gatppo_fail`), a nie domyślona. `Δ` < 0 = mechanizm lepszy. `|t| ≥ 2.8` to próg po poprawce na tyle porównań, ile ma ta tabela.

`gate_share` = odsetek par (decyzja, węzeł), w których sterowanie przejęła reguła. `gate_switches` = liczba ZMIAN trybu na węzłach w epizodzie — to one psują ruch (przełączenie resetuje fazę), a nie sam udział. `gate_precision` i `gate_recall` liczą się względem WYROCZNI (prawda z warstwy awarii, wyłącznie diagnostycznie) i rozdzielają „detektor jest słaby” od „przełączanie mało daje” — dwie rzeczy, które samo ATT skleja.

### cityflow_jinan3x4_4ph

| scenariusz | gate_share | gate_switches | gate_precision | gate_recall | Δ ATT [s] | sd | t | lepszy w |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 1.2% | 9.33 | 0.00 | — | **+3.25** | 0.9 | +6.08 | 0/3 |
| `gauss2` | 0.0% | 0.00 | — | — | +0.00 | 0.0 | — | 0/0 |
| `mask0p25` | 15.3% | 31.60 | 0.16 | 0.30 | -18.58 | 27.7 | -2.60 | 12/15 |
| `mask0p5` | 55.8% | 35.20 | 0.74 | 0.69 | **-111.40** | 51.1 | -8.45 | 15/15 |
| `mask0p25_intersection` | 25.9% | 18.20 | 0.90 | 0.90 | -32.29 | 51.8 | -2.41 | 9/15 |
| `mask0p5_intersection` | 48.9% | 23.07 | 0.93 | 0.95 | **-84.78** | 61.0 | -5.38 | 15/15 |
| `mask0p25_hold_t60` | 10.2% | 21.33 | 0.15 | 0.25 | +2.82 | 22.8 | +0.48 | 7/15 |
| `mask0p5_hold_t60` | 40.5% | 27.13 | 0.73 | 0.66 | **-67.52** | 67.0 | -3.90 | 13/15 |

* Powyżej progu: `clean` (+3.25 s, t = +6.08, lepszy w 0/3 parach), `mask0p5` (-111.40 s, t = -8.45, lepszy w 15/15 parach), `mask0p5_hold_t60` (-67.52 s, t = -3.90, lepszy w 13/15 parach), `mask0p5_intersection` (-84.78 s, t = -5.38, lepszy w 15/15 parach)

### cityflow_hangzhou4x4_4ph

| scenariusz | gate_share | gate_switches | gate_precision | gate_recall | Δ ATT [s] | sd | t | lepszy w |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 1.4% | 14.67 | 0.00 | — | **+1.19** | 0.2 | +13.52 | 0/3 |
| `gauss2` | 0.0% | 0.00 | — | — | +0.00 | 0.0 | — | 0/0 |
| `mask0p25` | 2.8% | 21.80 | 0.23 | 0.06 | +1.65 | 14.2 | +0.45 | 4/15 |
| `mask0p5` | 10.3% | 45.33 | 0.88 | 0.14 | +1.78 | 13.9 | +0.50 | 7/15 |
| `mask0p25_intersection` | 5.2% | 26.07 | 0.72 | 0.19 | -8.16 | 12.1 | -2.62 | 10/15 |
| `mask0p5_intersection` | 15.0% | 42.27 | 0.94 | 0.29 | **-22.38** | 27.9 | -3.10 | 13/15 |
| `mask0p25_hold_t60` | 1.6% | 16.13 | 0.02 | 0.00 | +7.40 | 37.6 | +0.76 | 6/15 |
| `mask0p5_hold_t60` | 6.0% | 33.87 | 0.65 | 0.09 | +9.78 | 46.3 | +0.82 | 6/15 |

* Powyżej progu: `clean` (+1.19 s, t = +13.52, lepszy w 0/3 parach), `mask0p5_intersection` (-22.38 s, t = -3.10, lepszy w 13/15 parach)

## Kontrola: ewaluator mierzy tę samą politykę co T1

Czysty wynik ewaluatora vs ostatni test treningu, 12 biegów uczonych: największa różnica **0.000 s** — zgodne co do precyzji logu.

## Czego ta tabela NIE mówi

* Poziomy są poziomami DETEKTORA. Na cechę `gauss` przenosi się ze współczynnikiem ~0,8–0,9, `urand` jest ~1,6× słabszy przy tym samym `level` (`results/sensor_severity.md`).
* Maska z podstawieniem `zero` zmienia znacznie mniej komórek cech, niż wynosi ułamek martwych pasów (zero jest wartością modalną) — ta sama liczba nie znaczy tego samego na sieciach o różnym obciążeniu.
* Scenariusze `hold` zaczynają się w decyzji 60 (15. minuta), więc pierwsza ćwiartka epizodu jest czysta — ich Δ nie jest porównywalne wprost z maską od początku.
