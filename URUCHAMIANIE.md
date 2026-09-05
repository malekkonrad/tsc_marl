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
