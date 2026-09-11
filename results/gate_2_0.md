# Bramka prototypu 2.0 — niepewność vs awaria sensora

Tryb `edl`, 0-hop (jedno skrzyżowanie), 150 epok, trening na: clean, gauss1, gauss2, mask0p25.
Ewaluacja na OGONIE epizodu (ostatnie 33% decyzji), MC-dropout K=10.

| zbiór | scenariusz | tren. | acc | ECE | EDL total | EDL aleat | EDL epist | vacuity | MC aleat | MC epist | martwe ruchy |
|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bazowy | clean | x | 0.731 | 0.115 | 0.581 | 0.492 | 0.0889 | 0.287 | 0.581 | 0.0044 | 0.0% |
| bazowy | drift0p2 |  | 0.699 | 0.135 | 0.572 | 0.485 | 0.0869 | 0.281 | 0.573 | 0.0043 | 0.0% |
| bazowy | drift0p3 |  | 0.668 | 0.167 | 0.573 | 0.486 | 0.0878 | 0.284 | 0.573 | 0.0043 | 0.0% |
| bazowy | gauss1 | x | 0.618 | 0.212 | 0.601 | 0.511 | 0.0906 | 0.292 | 0.601 | 0.0043 | 0.0% |
| bazowy | gauss2 | x | 0.489 | 0.297 | 0.615 | 0.521 | 0.0946 | 0.306 | 0.615 | 0.0048 | 0.0% |
| bazowy | gauss3p5 |  | 0.411 | 0.390 | 0.597 | 0.505 | 0.0924 | 0.299 | 0.598 | 0.0066 | 0.0% |
| bazowy | gauss4 |  | 0.393 | 0.410 | 0.587 | 0.496 | 0.0910 | 0.294 | 0.587 | 0.0070 | 0.0% |
| bazowy | mask0p25 | x | 0.650 | 0.167 | 0.593 | 0.501 | 0.0920 | 0.297 | 0.592 | 0.0043 | 68.4% |
| bazowy | mask0p25_intersection |  | 0.610 | 0.186 | 0.652 | 0.549 | 0.1029 | 0.333 | 0.652 | 0.0042 | 52.8% |
| bazowy | mask0p5 |  | 0.544 | 0.243 | 0.611 | 0.517 | 0.0939 | 0.303 | 0.611 | 0.0041 | 91.7% |
| bazowy | urand3p5 |  | 0.495 | 0.309 | 0.622 | 0.527 | 0.0952 | 0.307 | 0.623 | 0.0048 | 0.0% |
| d0p75 | clean |  | 0.753 | 0.085 | 0.618 | 0.525 | 0.0939 | 0.303 | 0.619 | 0.0041 | 0.0% |
| d0p5 | clean |  | 0.698 | 0.078 | 0.726 | 0.612 | 0.1146 | 0.370 | 0.727 | 0.0033 | 0.0% |
| d0p25 | clean |  | 0.690 | 0.126 | 0.785 | 0.659 | 0.1256 | 0.406 | 0.785 | 0.0028 | 0.0% |

## Kryteria wyjścia

* **(a) monotoniczność** — niepewność rośnie z sigma szumu: **NIE** (EDL total po `gauss`: 0.581, 0.601, 0.615, 0.597, 0.587).
* **(b) rozdzielność** — aleatoric na szumie WIDZIANYM w treningu: **+0.028** (na niewidzianym: **+0.004**), epistemic na OOD popytu: **+0.0367**, epistemic na szumie: **+0.0057** (chcemy: pierwsza i druga dodatnie, trzecia wyraźnie mniejsza od drugiej).
* **(c) kalibracja** — ECE na czystym **0.115**, najgorszy scenariusz **0.410**.
* **(d) EDL ≈ MC-BALD** — korelacja epistemic po scenariuszach: **-0.567**.
* **(e) niewidziane w treningu** — szum 3,5/4,0 i `urand` powyżej czystego: **TAK**; maska 50% i maska całego skrzyżowania (inna dotkliwość, inna jednostka) powyżej czystego: **TAK**; `drift` — INNA RODZINA awarii, scenariusz nieobserwowalny: **NIE** (Δ niepewności -0.008 przy spadku trafności o 0.063).

## Werdykt

**Bramka 2.0 dla głowy DECYZYJNEJ: NIEZALICZONA.**

Rozstrzygnięcie, po które szło się do prototypu: **niepewność decyzji (Dirichlet nad
fazami) NIE jest estymatorem niepewności obserwacji, tylko detektorem awarii
WIDZIANYCH W TRENINGU.** Trzy pomiary składają się na to jedno zdanie:

1. Poza zakresem treningowym niepewność **przestaje rosnąć i zawraca** (0.581, 0.601, 0.615, 0.597, 0.587), choć trafność spada dalej (o 0.338), a ECE rośnie do 0.410. Model staje się PEWNY I W BŁĘDZIE dokładnie tam, gdzie fallback ma zadziałać.
2. `drift` — inna RODZINA awarii, scenariusz NIEOBSERWOWALNY — jest niewidoczny: Δ niepewności -0.008 przy spadku trafności o 0.063. Rule-gate jest tu ślepy z definicji, ale głowa decyzyjna też.
3. Tania separacja EDL **nie zgadza się** z drogim wzorcem: korelacja epistemic z MC-BALD po scenariuszach = -0.567. MC podnosi epistemic na szumie niewidzianym, EDL go tam obniża.

Co z tego wynika dla Fazy 2: **plan miał rację, stawiając mechanizm 2.2 na heteroscedastycznej głowie liczonej z CECH WEJŚCIOWYCH, a nie na spreadzie Dirichleta** — i jest to teraz stwierdzenie oparte na pomiarze, nie na argumencie z confounda. Niepewność decyzyjna zostaje przy fallbacku (3.1), gdzie jej rolą jest twarda bariera, a nie lokalizacja zepsutego detektora.

Co działa i warto zachować: epistemic reaguje na OOD POPYTU wyraźniej niż na szum (+0.0367 vs +0.0057), czyli kanał epistemic → MaxPressure z dwupoziomowego fallbacku ma podstawę empiryczną. Reaguje też na maskę o dotkliwości i jednostce niewidzianej w treningu — generalizuje W RODZINIE awarii, nie MIĘDZY rodzinami.

## Czego ta tabela NIE mówi

`tie_rate` zrzutu = **14.1%**: w tylu decyzjach dwie fazy mają równe p(s), więc wysoka niepewność DECYZJI jest tam poprawna i nie ma nic wspólnego z awarią sensora. To jest zmierzony confound, przed którym ostrzega 2.2 — i powód, dla którego sygnał per-edge ma iść z CECH WEJŚCIOWYCH, a nie ze spreadu Dirichleta. Kolumna `martwe ruchy` pokazuje drugą stronę: przy masce większość ruchów ma martwy detektor, a mimo to cechy zmieniają się rzadko (awaria CICHA).

MC-dropout jest tu dropoutem OSTATNIEJ WARSTWY, nie całej sieci — standardowe przybliżenie, ale nie pełny Gal-Ghahramani. Do nazwania w pracy.
