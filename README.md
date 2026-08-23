# tsc-marl

Kod do pracy inżynierskiej **„Wykorzystanie podejścia wieloagentowego w problemie
zarządzania sygnalizacją świetlną"** — wieloagentowe sterowanie sygnalizacją oparte
na grafowej sieci neuronowej z kwantyfikacją niepewności.

Sieć drogowa jest traktowana jako graf, a agenci ważą informacje od sąsiadów przez
mechanizm uwagi (GAT). Kontrybucja pracy leży w warstwie niepewności: polityka
szacuje, **kiedy nie wie** — per krawędź grafu — i degraduje się łagodnie przy awarii
sensorów, zamiast rozsypywać się jak metody SOTA.

> Status: Faza 0 (odtworzenie baseline'ów) zamknięta. Fazy 1–3 w toku.

---

## Spis treści
1. [Co tu jest](#co-tu-jest) · 2. [Instalacja](#instalacja) · 3. [Uruchamianie](#uruchamianie)
4. [Protokół eksperymentalny](#protokół-eksperymentalny) · 5. [Wyniki](#wyniki)
6. [Relacja do LibSignala](#relacja-do-libsignala) · 7. [Pułapki](#pułapki)

---

## Co tu jest

```
tsc_marl/
├── src/tsc_marl/           # KOD WŁASNY
│   ├── run.py              #   entry point treningu/ewaluacji
│   ├── bench.py            #   runner siatki {agent}×{sieć}×{seed} → tabele T1
│   ├── libsignal.py        #   most do submodułu (ścieżki, sys.path, SUMO_HOME)
│   ├── seeding.py          #   deterministyczne ziarnowanie
│   └── agents/             #   agenci: Efficient-MP, Advanced-MP (własne implementacje)
│       └── ats.py          #     Advanced Traffic State — prymitywy wspólne dla MP i CoLight
├── configs/
│   ├── tsc/                #   configi metod + thesis_protocol.yml (protokół zamrożony)
│   └── sim/                #   definicje sieci (JiNan 3×4, HangZhou 4×4)
├── data/raw_data/          # sieci i potoki ruchu użyte w pracy (9,4 MB)
├── tests/                  # `make test` (szybkie) / `make test-slow` (z CityFlow)
├── results/                # liczby do pracy: CSV + wygenerowane tabele
├── runs/                   # (gitignore) checkpointy, logi, replaye
├── external/LibSignal/     # submoduł: fork LibSignala + patche silnika
└── deploy/hpc/             # ścieżka zapasowa: Docker/Apptainer/SLURM na Cyfronet
```

Podział jest celowy: **`src/` to kod własny, `external/` to cudzy**. Wszystko, co
zostało dołożone do LibSignala, widać jednym poleceniem — patrz
[Relacja do LibSignala](#relacja-do-libsignala).

## Instalacja

Wymagania: Python 3.10, `uv`, `cmake` + `g++` (CityFlow kompiluje się ze źródeł).

```bash
git clone --recursive git@github.com:malekkonrad/tsc_marl.git
cd tsc_marl
make setup     # submoduł + venv + zależności
make check     # ~15 s; powinno wypisać "Final Travel Time is 316.1114"
make test      # ~2 s; testy jednostkowe (nie potrzebują CityFlow)
make test-all  # + testy stawiające symulator (marker `slow`)
```

Na maszynie z GPU podmień koło `torch-scatter` na wariant CUDA:

```bash
uv pip install -e ".[gpu]" --find-links https://data.pyg.org/whl/torch-2.4.1+cu121.html
```

Jeśli klonowałeś bez `--recursive`: `git submodule update --init --recursive`.

## Uruchamianie

```bash
scripts/run.sh -a efficientmp -n cityflow_jinan3x4_4ph --seed 0   # jeden run
make bench                                                        # cała tabela T1
python -m tsc_marl.bench -a colight -s 0 1 2 --timeout 86400      # trening
```

**Uruchamiaj przez `scripts/run.sh`, nie przez `python -m tsc_marl.run`.** Wrapper
ustawia `PYTHONHASHSEED=0` przed startem interpretera; ustawiony z poziomu Pythona
nie zadziała, a bez niego `world_sumo` daje różne wyniki dla tych samych argumentów.

Dostępne metody: `fixedtime`, `sotl`, `maxpressure`, `efficientmp`, `advancedmp`,
`advancedcolight`, `presslight`, `colight`, `mplight`, `frap`, `dqn`.
Sieci: `cityflow_jinan3x4{,_4ph}`, `cityflow_hangzhou4x4{,_4ph}`.

## Protokół eksperymentalny

Protokół jest **zamrożony** w [`configs/tsc/thesis_protocol.yml`](configs/tsc/thesis_protocol.yml)
i współdzielony przez wszystkie metody. Powód jest metodologiczny: przy niespójnym
protokole rankingi metod potrafią się odwrócić, więc porównanie przestaje cokolwiek
znaczyć. Δt = 15 s, epizod 3600 s, `PYTHONHASHSEED=0`.

Jedyny parametr celowo zmieniany to długość fazy przejściowej, bo **wyniki są
raportowane w dwóch trybach**:

| tryb | `yellow_length` | co znaczy |
|---|---|---|
| `realistic` | 5 s (żółte 3 + all-red 2) | fizycznie poprawny — nasz protokół |
| `benchmark` | 0 s | tak liczą swoje wyniki implementacje referencyjne z papierów |

Rozjazd nie jest kosmetyczny: to **ok. 17% ATT**. Wyszedł przy odtwarzaniu
baseline'ów — publikacje deklarują modelowanie fazy przejściowej, ale ich kod jej nie
symuluje. Po wyłączeniu przejścia trafiamy w literaturę z dokładnością ~1%. Sama ta
różnica jest wynikiem wspierającym narrację pracy o luce wdrożeniowej, więc oba tryby
zostają, a każdy wiersz wyników niesie swój `yellow_length`.

Tryb wybiera się flagą — `--yellow 0` lub `bench -m benchmark` — a nie osobną kopią
konfiguracji.

## Wyniki

`results/` zawiera przetworzone liczby, które idą do pracy:

| plik | co to |
|---|---|
| `t1_raw.csv` | jeden wiersz na run: metryki + pełne metadane protokołu i parametry agenta |
| `t1_baseline.md` | nasze liczby obok literatury — GATE Fazy 0 |
| `t1_table.md` | mean ± std po ziarnach |
| `t1_meta.json` | protokół, wersje kodu (SHA obu repo), data |

Checkpointy i logi trafiają do `runs/` i **nie są wersjonowane** — w sandboxie urosły
do 642 MB. Kolumny `t_min`, `w1`, `obs_length`, `drop_always_green` w CSV nie są
ozdobą: bez nich wiersz Advanced-MP nie jest odtwarzalny, bo te parametry zmieniają
wynik o ~10%.

## Relacja do LibSignala

[LibSignal](https://github.com/DaRL-LibSignal/LibSignal) dostarcza środowisko i pętlę
treningową. Jest podpięty jako submoduł wskazujący na fork z gałęzią `thesis`, więc
**cały wkład w cudzy kod widać jednym poleceniem**:

```bash
git -C external/LibSignal diff master..thesis
```

Na gałęzi `thesis` są wyłącznie patche silnika, których nie da się zrobić z zewnątrz:

- `world_cityflow.py` — `yellow_length` czytane z configu zamiast zahardkodowanego 5
  (bez tego CityFlow ignorował ten parametr, więc tryb `benchmark` był nieosiągalny),
  plus opcjonalne logowanie ATT liczonego jak w `summary.py` linii Advanced-XLight.
- `world_sumo.py` — `sumo-gui` startuje automatycznie i z regulowanym opóźnieniem.

Wszystko inne — agenci, configi, protokół — mieszka w tym repo. Jest to możliwe, bo
Registry LibSignala zapełnia się dekoratorami przy imporcie, a ścieżki configów są
rozwiązywane względem CWD; uruchamiamy go więc z korzenia repo, żeby czytał nasze
`configs/` i `data/`. Szczegóły: [`src/tsc_marl/libsignal.py`](src/tsc_marl/libsignal.py).

Baseline'y `fixedtime`, `sotl`, `presslight`, `colight`, `mplight`, `frap` pochodzą
z LibSignala. `efficientmp`, `advancedmp` i `advancedcolight` są **implementacjami
własnymi** wg definicji
z papierów — LibSignal nie ma żadnej z tych metod, a jego `maxpressure` liczy pressure
z `lane_count` zamiast z kolejek, więc nie jest tym, co opisuje literatura.

## Pułapki

Rzeczy, które kosztowały czas i nie są oczywiste z kodu:

- **`update_target_rate` liczy się w decyzjach, nie w krokach symulacji.** Domyślne 10
  z `base.yml` znaczyło aktualizację sieci target co 10 kroków zamiast standardowych
  500–10000 — CoLight dochodził do ATT ~315 i rozpadał się do 1204. Dotyczy wszystkich
  metod DQN-owych, dlatego poprawka siedzi w configach każdej z nich, a nie w jednej.
- **ATT raportuje się jako średnią z 10 ostatnich epizodów testowych**, nie jako
  pojedynczy test po treningu (protokół Advanced-CoLight 5.1). Przy niestabilnym
  uczeniu różnica jest ogromna: 1204,5 (ostatni) vs 766,1 (średnia z 10) vs 314,6 (najlepszy).
- **SUMO tylko z pipa.** `apt` na Ubuntu noble daje 1.18, a pipowe `libsumo` jest 1.27
  i wersje muszą się zgadzać.
- **`SUMO_HOME` jest wymagane nawet dla `-w cityflow`** — LibSignal importuje SUMO na
  poziomie modułu.
- **LibSignal nadpisuje `configs/sim/*.cfg` w trakcie biegu**, wstawiając świeże
  ścieżki logów replaya. To nie są twoje zmiany; `restore_sim_configs()` cofa je po
  każdym runie, żeby `git status` pokazywał tylko prawdziwe różnice.
- **Każdy run idzie w osobnym podprocesie**, bo Registry jest globalne i nie da się go
  czysto zresetować między konfiguracjami w jednym procesie.

## Infrastruktura HPC

`deploy/hpc/` zawiera obraz Dockera, definicję Apptainera i skrypty SLURM pod Cyfronet
(Ares/Athena). To **ścieżka zapasowa** — treningi liczone są na lokalnej maszynie z GPU.
Katalog jest utrzymywany na wypadek, gdyby zabrakło mocy obliczeniowej.
