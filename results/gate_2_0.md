# Bramka prototypu 2.0 — niepewność DECYZJI vs awaria sensora i OOD

Tryb `edl`, 0-hop (jedno skrzyżowanie), 150 epok, trening na: clean, gauss1, gauss2, mask0p25.
Ewaluacja na OGONIE epizodu (ostatnie 33% decyzji). Wzorzec epistemic: **ensemble** (K=5).

| zbiór | scenariusz | tren. | acc | ECE | EDL total | EDL epist | epist/total | vacuity | ens epist | MC epist | remisy | EDL total (rozstrz.) | martwe ruchy |
|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bazowy | clean | x | 0.750 | 0.111 | 0.570 | 0.0851 | 0.149 | 0.274 | 0.3527 | 0.0047 | 10.0% | 0.558 | 0.0% |
| bazowy | drift0p2 |  | 0.718 | 0.139 | 0.564 | 0.0850 | 0.151 | 0.274 | 0.3423 | 0.0044 | 10.0% | 0.550 | 0.0% |
| bazowy | drift0p3 |  | 0.687 | 0.179 | 0.557 | 0.0840 | 0.151 | 0.271 | 0.3427 | 0.0040 | 10.0% | 0.544 | 0.0% |
| bazowy | gauss1 | x | 0.605 | 0.204 | 0.622 | 0.0956 | 0.154 | 0.309 | 0.3835 | 0.0046 | 10.0% | 0.607 | 0.0% |
| bazowy | gauss2 | x | 0.486 | 0.305 | 0.621 | 0.0960 | 0.155 | 0.310 | 0.4506 | 0.0051 | 10.0% | 0.612 | 0.0% |
| bazowy | gauss3p5 |  | 0.413 | 0.364 | 0.595 | 0.0927 | 0.156 | 0.300 | 0.5191 | 0.0066 | 10.0% | 0.589 | 0.0% |
| bazowy | gauss4 |  | 0.399 | 0.388 | 0.580 | 0.0904 | 0.156 | 0.293 | 0.5312 | 0.0067 | 10.0% | 0.572 | 0.0% |
| bazowy | mask0p25 | x | 0.664 | 0.170 | 0.588 | 0.0899 | 0.153 | 0.290 | 0.3394 | 0.0043 | 10.0% | 0.575 | 68.4% |
| bazowy | mask0p25_intersection |  | 0.622 | 0.176 | 0.641 | 0.0994 | 0.155 | 0.321 | 0.3367 | 0.0041 | 10.0% | 0.630 | 52.8% |
| bazowy | mask0p5 |  | 0.551 | 0.256 | 0.612 | 0.0953 | 0.156 | 0.308 | 0.3083 | 0.0039 | 10.0% | 0.603 | 91.7% |
| bazowy | urand3p5 |  | 0.486 | 0.304 | 0.631 | 0.0980 | 0.155 | 0.317 | 0.4576 | 0.0054 | 10.0% | 0.625 | 0.0% |
| d0p75 | clean |  | 0.735 | 0.089 | 0.615 | 0.0926 | 0.151 | 0.299 | 0.3162 | 0.0036 | 14.0% | 0.585 | 0.0% |
| d0p5 | clean |  | 0.700 | 0.114 | 0.718 | 0.1120 | 0.156 | 0.361 | 0.3421 | 0.0037 | 23.6% | 0.668 | 0.0% |
| d0p25 | clean |  | 0.690 | 0.133 | 0.777 | 0.1207 | 0.155 | 0.389 | 0.2862 | 0.0034 | 46.4% | 0.718 | 0.0% |
| r2000 | clean |  | 0.718 | 0.112 | 0.701 | 0.1111 | 0.158 | 0.359 | 0.3332 | 0.0033 | 21.5% | 0.666 | 0.0% |
| r2500 | clean |  | 0.730 | 0.096 | 0.616 | 0.0946 | 0.154 | 0.306 | 0.3421 | 0.0034 | 13.0% | 0.597 | 0.0% |
| hz | clean |  | 0.732 | 0.122 | 0.729 | 0.1115 | 0.153 | 0.359 | 0.2695 | 0.0030 | 44.7% | 0.704 | 0.0% |

## Przyrost niepewności na OOD (względem czystego bazowego), maksimum w grupie

| grupa | EDL total | EDL total (rozstrz.) | EDL epist | EDL epist (rozstrz.) | ens epist | ens epist (rozstrz.) |
|---|---:|---:|---:|---:|---:|---:|
| popyt × | +0.207 | +0.160 | +0.0356 | +0.0247 | -0.0106 | -0.0193 |
| wzorzec | +0.131 | +0.109 | +0.0261 | +0.0211 | -0.0106 | -0.0125 |
| topologia | +0.158 | +0.146 | +0.0264 | +0.0239 | -0.0832 | -0.0522 |
| *szum gauss2 (dla porównania)* | — | — | +0.0109 | — | +0.0979 | — |

## Kryteria wyjścia

* **(a) monotoniczność** — EDL total po `gauss` (σ = 0, 1, 2, 3,5, 4): 0.570, 0.622, 0.621, 0.595, 0.580 → **NIE**.
* **(b) rozdzielność** — aleatoric EDL na szumie widzianym +0.039, niewidzianym +0.004; epistemic EDL na OOD +0.0356 vs na szumie +0.0109.
* **(c) kalibracja** — ECE na czystym 0.111, najgorszy scenariusz 0.388.
* **(d) EDL ≈ wzorzec** — korelacja epistemic po wierszach: z ensemble **-0.358**, z MC-dropout ostatniej warstwy -0.464.
* **(e) niewidziane** — szum 3,5/4,0 i `urand` powyżej czystego: **TAK**; maska 50% i maska skrzyżowania: **TAK**; `drift`: **NIE** (Δ -0.013 przy spadku trafności o 0.063).
* **(g) czy epistemic EDL to osobny kanał** — stosunek epistemic/total w [0.149, 0.158] we wszystkich 17 wierszach (rozrzut 6.0% średniej) → **JEDEN SYGNAŁ**.

## Werdykt

**Bramka 2.0 dla głowy DECYZYJNEJ: NIEZALICZONA.**

1. **Poza zakresem treningowym niepewność decyzji przestaje rosnąć** (szczyt przy indeksie 1 ciągu 0.570, 0.622, 0.621, 0.595, 0.580), choć trafność spada dalej (o 0.351), a ECE rośnie do 0.388: pewny i w błędzie dokładnie tam, gdzie fallback ma zadziałać.
2. **`drift` jest niewidoczny** dla niepewności decyzji (Δ -0.013 przy spadku trafności o 0.063).
3. **Epistemic EDL nie jest osobnym kanałem**: epistemic/total mieści się w [0.149, 0.158] w każdym wierszu, więc epistemic, vacuity i total to ten sam sygnał w różnej skali. Przyrost „epistemic na OOD” jest przyrostem total, nie osobną informacją o nieznanym stanie.
4. **Nawet ensemble nie oddziela OOD od szumu** (najlepsza grupa „wzorzec”: -0.0125 vs szum +0.0979). Kanał epistemic → MaxPressure nie ma podstawy empirycznej na tych zbiorach — fallback jednopoziomowy (tylko obserwacyjny) jest uczciwszym planem.
5. **Wniosek dla 2.2 (bez zmian):** sygnał per-edge idzie z heteroscedastycznej głowy na CECHACH WEJŚCIOWYCH (`results/gate_2_2.md`), nie ze spreadu Dirichleta.

## Czego ta tabela NIE mówi

`tie_rate` zrzutu bazowego = **14.1%** — w tylu decyzjach dwie fazy mają równe p(s) i wysoka niepewność DECYZJI jest poprawna, bez związku z awarią. Kolumna *remisy* pokazuje ich udział w każdym zbiorze: na niskim popycie rośnie, dlatego przyrosty OOD podajemy też na próbkach rozstrzygalnych.

MC-dropout jest dropoutem OSTATNIEJ WARSTWY; jako wzorzec dla (d) służy deep ensemble (K członków `softmax`, różne ziarna), bo epistemic MC jest rzędu 0,004.
