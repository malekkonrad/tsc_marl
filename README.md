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

Ściąga z komendami na co dzień: **[URUCHAMIANIE.md](URUCHAMIANIE.md)**.

---

## Co tu jest

```
tsc_marl/
├── src/tsc_marl/           # KOD WŁASNY — warstwy od najniższej
│   ├── paths.py            #   layout repo (REPO_ROOT, CONFIGS, RUNS, RESULTS)
│   ├── seeding.py          #   deterministyczne ziarnowanie
│   ├── domain/             #   DZIEDZINA: wzory z papierów, bez frameworków
│   │   └── ats.py          #     Advanced Traffic State — wspólne dla MP i CoLight
│   ├── core/               #   RDZEŃ FAZY 1: GAT + PPO + CTDE, nasza implementacja
│   │   ├── graph.py        #     topologia sieci drogowej
│   │   ├── features.py     #     stan i nagroda (warianty)
│   │   ├── nets.py         #     enkoder GAT + actor-critic
│   │   ├── ppo.py          #     MAPPO
│   │   └── loop.py         #     spięcie z symulatorem
│   ├── libsignal/          #   MOST do submodułu — cała styczność z cudzym kodem
│   │   ├── bridge.py       #     sys.path, SUMO_HOME, CWD, środowisko podprocesów
│   │   ├── session.py      #     proces LibSignala pod zarejestrowanym biegiem
│   │   ├── experiment.py   #     adapter LibSignal ↔ runlog (id biegu, parsowanie logów)
│   │   └── agents/         #     wtyczki do ICH Registry: Efficient-MP, Advanced-MP,
│   │                       #     Advanced-CoLight
│   ├── runner/             #   URUCHAMIANIE: wejścia procesu i orkiestracja
│   │   ├── cli.py          #     `tsc <komenda>` (cienkie, rozdziela ruch)
│   │   ├── train.py        #     trening/pomiar agenta LibSignala
│   │   ├── train_core.py   #     trening rdzenia Fazy 1
│   │   ├── evaluate.py     #     ewaluacja zamrożonej polityki z checkpointu
│   │   └── suite.py        #     deklaratywne siatki biegów
│   └── analysis/           #   WYNIKI: czyta zapisane biegi, nic nie uruchamia
│       ├── plot.py         #     krzywe uczenia (działa też dla biegu w toku)
│       └── report.py       #     tabele T1 do pracy
├── src/runlog/             # NARZĘDZIE (niezależne od dziedziny): przechowywanie,
│                           # odpytywanie i agregacja wyników eksperymentów
├── configs/
│   ├── tsc/                #   configi metod + thesis_protocol.yml (protokół zamrożony)
│   ├── sim/                #   definicje sieci (JiNan 3×4, HangZhou 4×4)
│   └── suites/             #   siatki biegów (co policzyć)
├── data/raw_data/          # sieci i potoki ruchu użyte w pracy (9,4 MB)
├── tests/                  # lustro src/: core/, domain/, libsignal/, runner/, runlog/
├── results/                # liczby do pracy: CSV + wygenerowane tabele
├── runs/                   # (gitignore) biegi: meta, config, metryki, checkpointy
├── external/LibSignal/     # submoduł: fork LibSignala + patche silnika
└── deploy/hpc/             # ścieżka zapasowa: Docker/Apptainer/SLURM na Cyfronet
```

Podział jest celowy na dwóch poziomach.

**`src/` to kod własny, `external/` to cudzy.** Wszystko, co zostało dołożone do
LibSignala, widać jednym poleceniem — patrz
[Relacja do LibSignala](#relacja-do-libsignala).

**Wewnątrz `src/tsc_marl/` zależności idą w jedną stronę**, od `paths`/`domain` w dół
listy do `runner`. Konkretnie: `core/` nie importuje `libsignal/` (silnik dostaje z
zewnątrz, od `runner/train_core.py`), a `domain/` nie importuje niczego z projektu.
Dzięki temu rdzeń Fazy 1 testuje się bez symulatora — `make test` chodzi ~2 s.

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

> Ściąga z gotowymi komendami (wariant `uv run`): **[URUCHAMIANIE.md](URUCHAMIANIE.md)**.

Wszystko idzie przez jedno wejście `tsc` (`python -m tsc_marl`, gdyby console script
nie był jeszcze zainstalowany):

```bash
tsc doctor                                                    # środowisko OK?
tsc train -a efficientmp -n cityflow_jinan3x4_4ph --seed 0    # jeden bieg
tsc core -n cityflow_jinan3x4_4ph --seed 0                    # rdzeń GAT+PPO+CTDE
tsc suite list                                                # jakie są siatki
tsc suite run t1_baselines                                    # policz siatkę
tsc suite status p1_gate                                      # co już jest, czego brak
tsc runs / tsc table / tsc plot                               # wyniki, tabele, krzywe
tsc eval --load-run runs/t1/colight__cityflow_jinan3x4_4ph__s0_y5 \
         --checkpoint best --tag drop25                       # zamrożona polityka
```

`tsc train --help`, `tsc core --help` i `tsc eval --help` wypisują pełne listy flag
(argumenty lecą wprost do parserów tych modułów).

Działa też `uv run tsc ...`. Projekt jest oznaczony `[tool.uv] managed = false`,
bo środowisko budujemy `make setup` (torch z indeksu CPU albo CUDA + `torch-scatter`
z extra), a nie `uv sync` — bez tej flagi `uv run` zsynchronizowałby `.venv` z samego
`pyproject.toml` i podmienił build torcha oraz skasował `torch-scatter`.

### Siatki eksperymentów

„Co policzyć" jest plikiem w `configs/suites/*.yml`, a nie linią w historii powłoki:

```yaml
entry: core                 # core → runner.train_core, tsc → runner.train
mode: realistic             # realistic = yellow 5 (protokół pracy), benchmark = 0
grid:
  network: [cityflow_jinan3x4_4ph, cityflow_hangzhou4x4_4ph]
  seed: [0, 1, 2]
overrides:                  # opcjonalne nadpisania configu agenta
  model: {NUM_HEADS: [8, 8]}
```

Pole `ngpu` opisuje **maszynę, nie eksperyment**: `"0"` liczy na pierwszej karcie,
`"-1"` (domyślnie) na CPU, a brak CUDA schodzi do CPU z ostrzeżeniem w logu biegu
zamiast wywalać siatkę. Trafia do sekcji `command` configu — jedynej, której odcisk
parametrów NIE obejmuje — więc ten sam punkt policzony na CPU i na GPU ma ten sam
`params_hash` i jest jednym eksperymentem, nie dwoma. Działa dla `entry: core`
(rdzeń); `entry: tsc` zostaje na CPU, bo `agent/colight.py` w submodule nie ma
żadnej obsługi urządzenia.

`tsc suite run` **pomija punkty, które już mają wynik** — porównanie idzie po odcisku
parametrów, więc zmiana czegokolwiek w configu metody (albo w `overrides`) unieważnia
poprzedni bieg. `--replace` wymusza przeliczenie. Suite z `overrides` dokleja swoją
nazwę do `run_id`, żeby dwa różne configi nie biły się o jeden katalog; suite bez
nadpisań trafia w te same identyfikatory co uruchomienia ręczne.

**Nie wołaj `python -m tsc_marl.runner.train` bezpośrednio.** `PYTHONHASHSEED=0` musi być
ustawiony przed startem interpretera — z poziomu Pythona nie zadziała, a bez niego
`world_sumo` daje różne wyniki dla tych samych argumentów. CLI załatwia to,
restartując interpreter (`runner/cli.py:ensure_hashseed`).

Dostępne metody: `fixedtime`, `sotl`, `maxpressure`, `efficientmp`, `advancedmp`,
`advancedcolight`, `presslight`, `colight`, `mplight`, `frap`, `dqn`.
Sieci: `cityflow_jinan3x4{,_4ph}`, `cityflow_hangzhou4x4{,_4ph}`.

## Wyniki: `runs/` i `runlog`

Jeden bieg = jeden katalog. Nie ma centralnego pliku z wynikami, bo biegi bywają
odpalane równolegle w osobnych procesach — indeks jest wyliczany przez skan katalogów.

```
runs/t1/<agent>__<sieć>__s<seed>_y<yellow>/
    meta.json            # parametry, odcisk configu, proweniencja (SHA repo I submodułu), status
    config.resolved.yml  # config po scaleniu includes i nadpisaniach z CLI
    metrics.csv          # step,split,metric,value — per epizod
    result.json          # ATT, att_last10, throughput, queue, delay
    engine/              # artefakty LibSignala: checkpointy, logi, replaye
```

`make plot` rysuje krzywe do `runs/plots/`. Działa **także dla biegu w toku**:
`metrics.csv` powstaje dopiero na końcu (metryki wczytujemy z logu silnika po fakcie),
więc dla biegów o statusie `running` wykres czyta wprost `engine/.../_DTL.log`.
`--band` rysuje mean±std po seedach — liczony tylko na epizodach, które ma każdy bieg
w grupie, żeby ogon najdłuższego nie udawał średniej.

Obsługuje to [`src/runlog/`](src/runlog/README.md) — **osobne narzędzie, niezależne od
tej pracy**: nie importuje niczego z `tsc_marl` i nie wie nic o ruchu drogowym.
Tłumaczenie między nim a LibSignalem siedzi w
[`src/tsc_marl/libsignal/experiment.py`](src/tsc_marl/libsignal/experiment.py).

Trzy niezmienniki, które `runlog` egzekwuje, bo każdy z nich odpowiada błędowi, który
już się w tym projekcie zdarzył: **status biegu** (przerwany trening nie liczy się jako
policzony), **odmowa cichego nadpisania** (ten sam identyfikator z innymi parametrami to
błąd, nie nadpisanie) i **snapshot configu** (wiersz wyniku nie może odsyłać do pliku,
który ktoś potem zmienił).

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
- `utils/logger.py` — `get_output_file_path` honoruje `logger.root_dir`. Bez tego
  ten parametr był martwym zapisem, a artefakty treningu lądowały w `data/output_data/`,
  czyli w katalogu wejściowych, wersjonowanych sieci.
- `agent/colight.py`, `agent/dqn.py` — `save_model` zapisuje sieć **online**
  (`self.model`), a nie `target_model`. Decyzje podejmuje online, więc to ona
  wyprodukowała liczby w krzywej uczenia; target to jej kopia sprzed nawet dwóch
  epizodów. Bez tego checkpoint jest INNĄ siecią niż zmierzona, a Faza 3 ewaluuje
  właśnie checkpoint. Reszta LibSignala (`presslight`, `frap`, `dqn_torch_agent`,
  `colight_pytorch_agent`) zapisuje `self.model` — patch usuwa niespójność.

Wszystko inne — agenci, configi, protokół — mieszka w tym repo. Jest to możliwe, bo
Registry LibSignala zapełnia się dekoratorami przy imporcie, a ścieżki configów są
rozwiązywane względem CWD; uruchamiamy go więc z korzenia repo, żeby czytał nasze
`configs/` i `data/`. Szczegóły: [`src/tsc_marl/libsignal/bridge.py`](src/tsc_marl/libsignal/bridge.py).

Baseline'y `fixedtime`, `sotl`, `presslight`, `colight`, `mplight`, `frap` pochodzą
z LibSignala. `efficientmp`, `advancedmp` i `advancedcolight` są **implementacjami
własnymi** wg definicji
z papierów — LibSignal nie ma żadnej z tych metod, a jego `maxpressure` liczy pressure
z `lane_count` zamiast z kolejek, więc nie jest tym, co opisuje literatura.

## Pułapki

Rzeczy, które kosztowały czas i nie są oczywiste z kodu:

- **`configs/sim/*.cfg` są nadpisywane przez LibSignal przy każdym biegu**
  (świeże ścieżki logów replaya). Cofa to `libsignal.sim_configs_untouched()` —
  przez MIGAWKĘ BAJTÓW zdjętą przed biegiem, nie przez `git checkout -- configs/sim`.
  Ta pierwsza wersja kasowała również prawdziwe, niezacommitowane poprawki w tych
  plikach: poprawiasz ścieżkę do roadnetu, odpalasz bieg, poprawka znika bez śladu.

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
