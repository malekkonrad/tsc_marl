# krakow_3w — korytarz al. Trzech Wieszczów (Kraków), SUMO

Case study z gałęzi `cracow-case-study`. Rozpoznanie krakowskiego ITS, ocena
i plan: `tsc_marl_research/krakow-its.md`. Obszar = Obszar 1 krakowskiego UTCS
(al. Krasińskiego / Mickiewicza / Słowackiego + 29 Listopada; Siemens MOTION,
preferencja ruchu kołowego), bbox `(50.050, 19.912, 50.077, 19.948)`.

## Budowa

```bash
.venv/bin/python data/raw_data/krakow_3w/build.py            # osm → net → demand → profile
.venv/bin/python data/raw_data/krakow_3w/build.py profile    # sama diagnostyka
.venv/bin/python data/raw_data/krakow_3w/build.py --refresh  # nowa migawka OSM
```

| plik | co | odtwarzalny z |
|---|---|---|
| `3w.osm` | migawka OSM (drogi + torowiska), 2026-09-13 | Overpass (`--refresh` zmienia wynik!) |
| `3w.net.xml` | sieć SUMO, sam ruch samochodowy | `3w.osm` + `build.py net` |
| `3w_{low,mid,high}.rou.xml` | popyt syntetyczny, 3600 s, seed 0 | `3w.net.xml` + `build.py demand` |

Dane OSM: © współtwórcy OpenStreetMap, licencja ODbL.

Configi: `configs/sim/sumo_krakow3w_{low,mid,high}.cfg`, np.

```bash
.venv/bin/tsc train -a maxpressure -w sumo -n sumo_krakow3w_mid --seed 0   # baseline
.venv/bin/tsc core -w sumo -n sumo_krakow3w_mid --seed 0                   # rdzeń GAT+PPO
```

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

## Wyniki baseline'ów (2026-09-13, adapter, laptop, seed 0, yellow 5, ~40 s na bieg)

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

1. **Popyt nieskalibrowany.** Okresy w `DEMAND_PERIOD` są robocze. `mid` miał
   21 teleportów przy programach netconverta — znaleźć węzeł, który się blokuje.
2. **FixedTime to LibSignalowe `t_fixed: 30`** (równe zielone po kolei), NIE
   Webster z 4 programami na pory dnia, jak zakłada plan porównania.
3. **Zbiór sygnalizacji = cały bbox**, łącznie z Kijowską, Dietla, Pawią i przejściami
   dla pieszych. Do decyzji: przyciąć do korytarza (pozostałe `--tls.unset`?) czy zostawić.
4. **Graf:** 4 z 25 węzłów bez sąsiadów przy zasięgu 600 m (brzeg bboxu, przejścia
   dla pieszych) — GAT widzi tam tylko siebie. Zasięg to slot ablacyjny.
5. **Brak tramwajów** (usuwane w `net`) — priorytet tramwajowy poza zakresem na teraz.
