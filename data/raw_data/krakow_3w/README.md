# krakow_3w — korytarz al. Trzech Wieszczów (Kraków), SUMO

Case study z gałęzi `cracow-case-study`. Rozpoznanie krakowskiego ITS, ocena
i plan: `tsc_marl_research/krakow-its.md`. Obszar = Obszar 1 krakowskiego UTCS
(al. Krasińskiego / Mickiewicza / Słowackiego + 29 Listopada; Siemens MOTION,
preferencja ruchu kołowego), bbox `(50.050, 19.912, 50.077, 19.948)`.

## Budowa

```bash
.venv/bin/python data/raw_data/krakow_3w/build.py            # osm → net → demand → webster → profile
.venv/bin/python data/raw_data/krakow_3w/build.py demand webster  # sam popyt + programy Webstera
.venv/bin/python data/raw_data/krakow_3w/build.py profile    # sama diagnostyka
.venv/bin/python data/raw_data/krakow_3w/build.py --refresh  # nowa migawka OSM
```

| plik | co | odtwarzalny z |
|---|---|---|
| `3w.osm` | migawka OSM (drogi + torowiska), 2026-09-13 | Overpass (`--refresh` zmienia wynik!) |
| `3w.net.xml` | sieć SUMO, sam ruch samochodowy | `3w.osm` + `build.py net` |
| `3w_{low,mid,high}.rou.xml` | popyt syntetyczny, 3600 s, seed 0 — poziomy skalibrowane (niżej) | `3w.net.xml` + `build.py demand` |
| `3w_{low,mid,high}.webster.json` | program stałoczasowy Webstera per poziom: {sygnalizacja: {stan fazy: zielone [s]}} | `.rou.xml` + `build.py webster` |

Dane OSM: © współtwórcy OpenStreetMap, licencja ODbL.

Configi: `configs/sim/sumo_krakow3w_{low,mid,high}.cfg`, np.

```bash
.venv/bin/tsc train -a maxpressure -w sumo -n sumo_krakow3w_mid --seed 0   # baseline
.venv/bin/tsc core -w sumo -n sumo_krakow3w_mid --seed 0                   # rdzeń GAT+PPO
```

Próg bramki-reguły (sieć bez głowy niepewności) — `runs/heads/rule_ref__<sieć>.json`:

```bash
.venv/bin/tsc rulecal -n sumo_krakow3w_mid      # kwantyl 99,5% udziału zamrożonych ruchów, czysty ruch, Advanced-MP
```

| poziom | próg | fałszywe alarmy na czystym (budżet 0,5%) |
|---|---:|---:|
| low | 1,00 | **2,8%** |
| mid | 0,79 | 0,52% |
| high | 0,79 | 0,63% |

Na `low` budżetu NIE da się dotrzymać: przy małym ruchu całe skrzyżowania bywają puste
(wszystkie prawdziwe ruchy stoją), a reguła nie odróżni pustego od martwego — kwantyl siada
na 1,0. Bramka będzie tam czasem oddawać puste węzły Websterowi; koszt powinien być mały
(pusty węzeł), ale ma wyjść w kolumnie `clean`, nie zostać założony. Na mid/high lekkie
przekroczenie to dyskretność udziałów (węzeł ma 1–7 wlotów, udział ma kilka wartości).

## Poziomy popytu (skalibrowane 2026-10-09)

Sweep okresu `randomTrips` na programach netconverta, 1 h ruchu (`DEMAND_PERIOD` w `build.py`):

| popyt | strata czasu | teleporty | kolejka na wjazd | |
|---:|---:|---:|---:|---|
| 1800 poj/h | 100 s | 0 | 0 | **low** — wyraźnie niedosycony |
| 3000 poj/h | 119 s | 21 | 0 | (dawne mid; teleporty z zakleszczenia na Prądnickiej) |
| 4500 poj/h | 141 s | 2 | 0 | **mid** — tuż przed kolanem nasycenia |
| 5143 poj/h | 230 s | 31 (0,6%) | 0 | **high** — za kolanem, bez masowych teleportów |
| 5538 poj/h | 302 s | 107 (1,9%) | 18 | teleporty zaczynają psuć ATT |
| 6000 poj/h | 322 s | 239 | 26 | zakleszczenie |
| 8000+ poj/h | 492+ s | 1482+ | 370+ | ATT bez znaczenia |

Teleporty przy dawnym mid: 10 z 21 na 7-metrowym odcinku ul. Prądnickiej (krawędź
`117386961`, skrzyżowanie bez sygnalizacji, 1 → 2 → 1 pas) — artefakt importu OSM poza
sterowanymi skrzyżowaniami; przy obecnych poziomach nie występuje.

## FixedTime = Webster per poziom popytu

`build.py webster` → SUMO `tlsCycleAdaptation` (Webster: C₀ = (1,5L + 5)/(1 − Y), zielone
proporcjonalnie do przepływów krytycznych), strata 5 s/fazę (= przejście z protokołu:
3 s żółtego + 2 s all-red), nasycenie 1800 poj/h/pas. Każdy poziom popytu to inna pora
dnia — odpowiednik krakowskiej praktyki „≥4 programy na pory dnia" — a FixedTime dostaje
program pod TEN popyt, który widzą wszystkie sterowniki.

Program wchodzi przez klucz `fixedTimeProgram` w `configs/sim/sumo_krakow3w_*.cfg`:
`tsc degrade` buduje z niego i kolumnę `fixedtime`, i cel przełączenia bramki
(`runner/degrade.fixed_time_program`). Dopasowanie po STANIE sygnałów fazy, nie po
indeksie — adapter odkrywa fazy symulacją. LibSignalowy agent `fixedtime` (`tsc train -a
fixedtime`) programu NIE czyta — wynik FixedTime do pracy brać z `tsc degrade`.

## Baseline'y po kalibracji (2026-10-09, ścieżka `tsc degrade`, czysty ruch, seed 0)

ATT [s] / przepustowość (pojazdy, które dojechały):

| | low (1800) | mid (4500) | high (5143) |
|---|---:|---:|---:|
| FixedTime `t_fixed: 30` | 331,4 / 1618 | 390,2 / 3941 | 471,8 / 4111 |
| **FixedTime Webster** | 297,9 / 1645 | **347,6 / 3972** | **390,5 / 4374** |
| Efficient-MP | 282,6 / 1651 | 373,5 / 3846 | 477,2 / 3928 |
| Advanced-MP | **279,8 / 1655** | 396,9 / 3676 | 481,5 / 3840 |

Przy niskim ruchu metody ciśnieniowe wygrywają (−6% ATT); przy `mid`/`high` PRZEGRYWAJĄ
z Websterem o 7–23% i tracą przepustowość. Stare porównanie z `t_fixed: 30` maskowało to
całkowicie. Hipoteza do sprawdzenia (README z 2026-09-13 widziało to samo przy dawnym
`high`): krótkie odcinki między klastrami skrzyżowań + `t_min` → spillback, którego
ciśnienie lokalne nie widzi; Webster koordynuje przez sam podział zielonego.

## Świat SUMO = adapter, nie upstream

Każdy bieg `-w sumo` idzie przez `src/tsc_marl/libsignal/sumo_world.py`, który
nadpisuje upstreamowy `world_sumo.World` w Registry — dla rdzenia i baseline'ów
LibSignala naraz. Dziewięć różnic opisuje docstring modułu. Najważniejsze dla liczb:

- **przejścia faz jak w CityFlow** — upstream po pierwszych sekundach epizodu
  przełączał fazy natychmiast, bez fazy przejściowej (zmierzone: 0 kroków
  przejścia po 30. sekundzie), więc `yellow` i `t_min` nic nie znaczyły,
- **ATT jak w CityFlow** (dojechane + wciąż jadące) — upstream liczył tylko dojechane,
- **`lane_waiting_count` = stojące teraz** — to nagroda rdzenia,
- **zielone warunkowe `g` w fazach** — 32 lanelinki na 13 sygnalizacjach,
- **graf sąsiedztwa po drogach** (`build_graph`, zasięg 600 m) — graf LibSignala
  miał tu 1 krawędź na 25 węzłów, nasz ma 51 (stopień wejściowy 0–4).

**Wyniki `-w sumo` sprzed adaptera są nieważne** (w tym tabela z pierwszej wersji
tego README).

## Profil sieci

25 sygnalizacji, silnie niejednorodnych — dokładnie to, czego nie ma w JiNan/HangZhou:

- wloty: 1–7 (1×1, 6×2, 6×3, 6×4, 3×5, 2×6, 1×7),
- fazy zielone: 2–7 (13×2, 7×3, 3×4, 1×5, 1×7) → rdzeń maskuje nieistniejące fazy,
- 13 z 25 ma zielone warunkowe `g` (skręt w lewo z ustąpieniem),
- kilka to same przejścia dla pieszych (1–2 wloty, 4 ruchy),
- 8 sygnalizacji steruje kilkoma węzłami (klastry `joinedS_…`).

## Wyniki baseline'ów — STARE (2026-09-13, sprzed kalibracji popytu i Webstera)

⚠ Poziomy popytu i FixedTime z tej tabeli są nieaktualne — nowa tabela wyżej
(sekcja „Baseline'y po kalibracji").

| popyt | pojazdy | FixedTime | MaxPressure | Advanced-MP | throughput FT / MP / AMP |
|---|---|---|---|---|---|
| low  | 1800 | 331,4 | 274,4 | 282,0 | 1618 / 1656 / 1660 |
| mid  | 3000 | 336,7 | 276,2 | 283,0 | 2716 / 2756 / 2755 |
| high | 4500 | 390,2 | 394,5 | 466,6 | 3941 / **3662** / **3465** |

ATT w sekundach. Odczyt: przy low/mid MaxPressure bije FixedTime o ~18%, Advanced-MP
o ~16%, przy tej samej przepustowości. Przy high metody ciśnieniowe przepuszczają
mniej pojazdów (MP −7%, Advanced-MP −12%), a Advanced-MP ma najgorsze ATT —
do zbadania (spillback na krótkich odcinkach? przesycenie?). ATT raportować zawsze
razem z throughput.

Rdzeń GAT+PPO: na razie tylko smoke (2 epizody × 600 s, ~24 s) — działa, bez wyniku.
Pełny bieg (200 epizodów × 3600 s) szacunkowo ~3 h na laptopie CPU.

## Znane ograniczenia (do rozwiązania przed wynikami do pracy)

1. ~~Popyt nieskalibrowany~~ — skalibrowany 2026-10-09 (sekcja „Poziomy popytu").
2. ~~FixedTime to `t_fixed: 30`~~ — Webster per poziom od 2026-10-09 (sekcja „FixedTime").
3. **Zbiór sygnalizacji = cały bbox**, łącznie z Kijowską, Dietla, Pawią i przejściami
   dla pieszych. Do decyzji: przyciąć do korytarza (pozostałe `--tls.unset`?) czy zostawić.
4. **Graf:** 4 z 25 węzłów bez sąsiadów przy zasięgu 600 m (brzeg bboxu, przejścia
   dla pieszych) — GAT widzi tam tylko siebie. Zasięg to slot ablacyjny.
5. **Brak tramwajów** (usuwane w `net`) — priorytet tramwajowy poza zakresem na teraz.
