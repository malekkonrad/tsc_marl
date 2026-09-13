# Ściąga: wszystko przez `uv run`

Jedno wejście — `tsc`. Poniżej każda komenda w formie `uv run`, bez aktywowania venva.
Równoważnie działa `.venv/bin/tsc ...` albo `tsc ...` po `source .venv/bin/activate`.

> **`uv sync` NIE** — projekt ma `[tool.uv] managed = false` i `uv sync` odmówi.
> Środowisko buduje `make setup`. Szczegóły: [Dlaczego nie `uv sync`](#dlaczego-nie-uv-sync).

---

## Najpierw sprawdź, czy jest zdrowo

```bash
uv run tsc doctor
```

Wypisuje ścieżki, `SUMO_HOME`, `PYTHONHASHSEED` i wersje `torch` / `cityflow` /
`torch_geometric`. `torch` **musi** kończyć się na `+cpu` (maszyna robocza) albo
`+cu121` (maszyna z GPU). Gołe `2.4.1` znaczy, że coś przeinstalowało środowisko.

```bash
uv run tsc --help              # lista komend
uv run tsc train --help        # pełna lista flag biegu
```

---

## Pojedynczy bieg

```bash
# agent LibSignala (baseline albo CoLight/Advanced-CoLight)
uv run tsc train -a advancedcolight -n cityflow_jinan3x4_4ph --seed 0

# rdzeń Fazy 1 — GAT + PPO + CTDE, własna pętla
uv run tsc core -n cityflow_jinan3x4_4ph --seed 0

# ewaluacja zamrożonej polityki z checkpointu (nie trenuje)
uv run tsc eval --load-run runs/t1/colight__cityflow_jinan3x4_4ph__s0_y5 \
                --checkpoint best --tag drop25
```

Najczęstsze flagi (`train` i `core` mają te same):

| flaga | znaczenie |
|---|---|
| `-a`, `--agent` | nazwa z Registry (`core` domyślnie `gatppo`) |
| `-n`, `--network` | `cityflow_jinan3x4{,_4ph}`, `cityflow_hangzhou4x4{,_4ph}` |
| `--seed` | ziarno |
| `--yellow` | `5` = protokół pracy, `0` = tryb papierów |
| `--episodes`, `--steps` | skrócenie biegu — **tylko smoke-testy** |
| `--run-id` | własna nazwa katalogu biegu |
| `--replace` | policz ponownie bieg, który już się udał |
| `--ngpu` | `-1` = CPU, inaczej numer karty |

Smoke-test rdzenia, ~2 s:

```bash
uv run tsc core -n cityflow_jinan3x4_4ph --episodes 1 --steps 150 --run-id smoke --replace
```

---

## Siatki eksperymentów

To jest normalny sposób liczenia — nie odpalaj dziesięciu biegów z palca.

```bash
uv run tsc suite list                    # jakie siatki są w configs/suites/
uv run tsc suite status p1_gate          # co policzone, czego brak — nic nie uruchamia
uv run tsc suite run p1_gate             # policz brakujące punkty
uv run tsc suite run p1_gate --replace   # policz wszystko od nowa
```

`suite run` pomija punkty, które mają **aktualny** wynik, więc przerwane liczenie
wznawiasz tą samą komendą. Można ją odpalać w kółko bez szkody.

Gotowe siatki:

| nazwa | co liczy |
|---|---|
| `t1_baselines` | 5 metod × 2 sieci, `yellow=5` — punkt odniesienia pracy |
| `t1_paper` | to samo przy `yellow=0` — tryb, w którym liczą papiery |
| `p1_reference` | Advanced-CoLight × 2 sieci × 3 seedy — odniesienie dla GATE Fazy 1 |
| `p1_gate` | rdzeń GAT+PPO+CTDE × 2 sieci × 3 seedy |
| `t1_learned` | CoLight, PressLight × 2 sieci × 3 seedy, `yellow=5` — CPU |
| `t1_learned_paper` | Advanced-CoLight, CoLight w trybie papierów (`yellow=0`) — CPU |
| `p1_abl_meanpool` | ablacja: actor z mean-pool (GraphSAGE) zamiast GAT — GPU |
| `p1_abl_nognn` | ablacja: actor bez sąsiadów (MLP) — GPU |
| `p1_abl_state_{queue,pressure,ep}` | ablacja stanu zamiast ATS — GPU |

Punkty liczą się **równolegle** flagą `-j`:

```bash
uv run tsc suite run t1_learned -j 4     # 4 podprocesy naraz
```

Starty są rozsunięte o 15 s (`--stagger`), bo LibSignal przepisuje `configs/sim/*.cfg`
przy budowie świata i dwa procesy startujące w tej samej sekundzie potrafią przeczytać
plik ucięty w połowie. Pad w pierwszych 90 s jest ponawiany raz. Każde dziecko dostaje
`OMP_NUM_THREADS = rdzenie / j`, a jego wyjście idzie do `runs/_suite_logs/<suite>/`.

Własna siatka to plik w `configs/suites/`:

```yaml
entry: core                 # core → runner.train_core, tsc → runner.train
mode: realistic             # realistic = yellow 5, benchmark = yellow 0
timeout: 86400              # sekundy na jeden punkt
grid:
  network: [cityflow_jinan3x4_4ph, cityflow_hangzhou4x4_4ph]
  seed: [0, 1, 2]
overrides:                  # opcjonalne nadpisania configu agenta
  model: {NUM_HEADS: [8, 8]}
```

Siatka **z** `overrides` dokleja swoją nazwę do `run_id` (żeby dwa różne configi nie
biły się o jeden katalog); siatka **bez** nadpisań trafia w te same identyfikatory co
uruchomienia ręczne, więc od razu widzi wcześniejsze wyniki jako gotowe.

---

## Awarie sensorów — zamrożone sterowniki (T2)

Plan w `configs/robustness/<nazwa>.yml`: sieci, sterowniki (uczone z seedami),
scenariusze awarii i liczba losowań. Jeden punkt = sterownik × sieć × seed,
wszystkie scenariusze w jednym biegu `runs/t2/...`.

```bash
uv run tsc degrade status t2_nomech      # co policzone, komu brakuje wag rodzica
uv run tsc degrade run t2_nomech -j 6    # policz brakujące
uv run tsc degrade report t2_nomech      # -> results/t2_nomech.md + results/figures/
```

Uczone sterowniki (gatppo, Advanced-CoLight) biorą **końcowe wagi** z biegu treningowego
`runs/t1/<agent>__<sieć>__s<seed>_y5`. Bez pliku wag punkt jest pomijany z komunikatem,
nie wywala planu. Raport sprawdza, że czysty wynik ewaluatora równa się ostatniemu
testowi z treningu — inaczej ewaluator mierzyłby inną politykę niż tabela T1.

---

## Prototypy Fazy 2 (CPU, minuty)

```bash
uv run tsc dataset -n cityflow_jinan3x4_4ph --episodes 3   # zrzut BC (stan → akcja Advanced-MP)
uv run tsc prototype                                        # bramka 2.0 -> results/gate_2_0.md
uv run tsc obshead                                          # bramka 2.2 -> results/gate_2_2.md
uv run tsc severity                                         # przelicznik dotkliwości awarii
```

Zbiory OOD dla epistemic: `cityflow_jinan3x4_4ph_d0p{25,5,75}` (popyt w dół),
`cityflow_jinan3x4_4ph_r{2000,2500}` (inny wzorzec ruchu), `cityflow_hangzhou4x4_4ph`
(inna topologia) — każdy trzeba najpierw zrzucić `tsc dataset -n <sieć>`.

---

## Gdzie co liczyć — laptop vs glider

Wagi treningów żyją tylko na gliderze (kopie `runs/t1` na laptopie są bez `*.pt`).

| co | gdzie | dlaczego |
|---|---|---|
| `entry: core` (gatppo, ablacje `p1_abl_*`) | glider | GPU |
| `tsc degrade run` dla uczonych | glider | potrzebuje wag |
| `t1_learned`, `t1_learned_paper` | dowolna, CPU | LibSignal nie ma obsługi GPU |
| `tsc prototype`, `obshead`, `dataset`, reguły w `degrade` | laptop | CPU, minuty |

**Nocna kolejka na gliderze — jedna komenda:**

```bash
git pull
bash deploy/glider/noc.sh        # trzy kolejki w tle: gpu, cpu, t2
```

Kolejki: `gpu` trenuje rdzeń i ablacje, `cpu` trenuje baseline'y LibSignala, a `t2`
czeka, aż obie skończą, i dopiero wtedy liczy ewaluacje pod awarią (potrzebuje ich wag),
raporty T2 i `tsc table`. Skrypt sprawdza `tsc doctor` i CUDA, rusza kolejki odporne na wylogowanie
i wypisuje, gdzie patrzeć (`runs/_suite_logs/noc_<data>/`). Wznowienie po przerwie
to ta sama komenda: gotowe punkty są pomijane, a bieg przerwany na tej maszynie
(martwy proces w statusie `running`) liczy się od nowa — `runlog` rozpoznaje go
po zapisanym `pid`. `DRY=1` tylko wypisuje komendy; `CPU_J`/`GPU_J` zmieniają
równoległość.

**Test gałęzi epistemic — po zakończeniu `noc.sh`:**

```bash
bash deploy/glider/ood.sh        # plan t2_ood w tle; odmówi startu, gdy noc.sh jeszcze liczy
```

Liczy zamrożone polityki, Advanced-CoLighta, reguły i ensemble 5 actorów rdzenia na ruchu
treningowym, lżejszym (JiNan ×0,25, inny wzorzec) i cięższym (realny HangZhou 5816, ×2,3
pojazdów). Werdykt w `results/t2_ood_epistemic.md`: czy polityka przegrywa z MaxPressure na nowym
ruchu i czy rozbieżność actorów rośnie tam mocniej niż pod szumem detektorów.

Nie licz tego samego `run_id` na obu maszynach — przy synchronizacji jeden katalog
nadpisze drugi. Po liczeniu na gliderze przenieś wyniki razem z wagami:

```bash
rsync -av glider:tsc_marl/runs/t1/ runs/t1/      # z wagami (potrzebne do degrade)
rsync -av glider:tsc_marl/runs/t2/ runs/t2/
```

---

## Wyniki

```bash
uv run tsc runs      # wszystkie biegi: status, parametry, ATT
uv run tsc table     # przelicz tabele T1 → results/ (czyta runs/, nic nie liczy)
uv run tsc plot      # krzywe uczenia → runs/plots/ (działa też dla biegu w toku)
uv run tsc plot --split train
```

---

## Skróty (Makefile)

Cele Make wołają `.venv/bin/tsc` bezpośrednio — `uv run` przed `make` nie jest potrzebny.

```bash
make check        # zdrowie środowiska: doctor + jeden bieg Efficient-MP (~15 s)
make gate         # GATE Fazy 1: p1_reference → p1_gate → status
make bench        # siatka T1 realistyczna + tabele
make bench-paper  # to samo w trybie papierów (yellow=0)
make test         # testy szybkie (~2 s)
make test-slow    # testy z prawdziwym CityFlow (~30 s)
make lint         # ruff check + format --check
```

Instalacja / odtworzenie środowiska:

```bash
make setup        # venv + zależności CPU + submoduł LibSignal
make setup-gpu    # to samo, ale torch/torch-scatter w wariancie CUDA 12.1
```

---

## Dlaczego nie `uv sync`

`uv sync` (i `uv run` bez `managed = false`) synchronizuje `.venv` z samego
`pyproject.toml` — bez indeksów i extras, które podaje `make setup`. Konkretnie
zrobiłby to:

```
- torch==2.4.1+cpu             → + torch==2.4.1     (build CUDA, ~2,5 GB wheeli nvidia-*)
- torch-scatter==2.1.2+pt24cpu → skasowany           (żyje w extra `cpu`/`gpu`)
- ruff, pytest                 → skasowane           (żyją w extra `dev`)
```

Czyli jedno `uv run` niszczyłoby środowisko, w którym powstały wyniki. Dlatego
`pyproject.toml` ma `[tool.uv] managed = false`: `uv run` używa `.venv` bez
synchronizacji, a `uv sync` odmawia z czytelnym komunikatem. `uv venv` i `uv pip`
działają normalnie — nie chodzą przez tryb projektowy.

Gdyby środowisko jednak padło: `make setup` (albo `make setup-gpu`) odbudowuje je od zera.

---

## Jedna pułapka

**Nie wołaj `python -m tsc_marl.runner.train` ani `runner.train_core` bezpośrednio.**

`PYTHONHASHSEED=0` musi być ustawiony **zanim** wystartuje interpreter — z poziomu
Pythona nie zadziała, a bez niego `world_sumo` daje różne wyniki dla tych samych
argumentów. `tsc` restartuje się w tym celu sam (`cli.ensure_hashseed`), a podprocesy
z `tsc suite run` dostają zmienną z `libsignal.child_env()`. Bezpośrednie wywołanie
modułu omija jedno i drugie.
