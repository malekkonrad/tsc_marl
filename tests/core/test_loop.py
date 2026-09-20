"""Testy pętli środowiskowej.

Pętla jest jedynym miejscem rdzenia, w którym trzeba ODTWORZYĆ protokół
`tsc_trainer.train()` — a każde odstępstwo daje liczby niedające się porównać
z baseline'ami, nie błąd. Stąd testy celują w trzy rzeczy: przenumerowanie akcji
(akcja jednego skrzyżowania wykonana na drugim nie wywala niczego), zgodność
metryk z `Metrics` LibSignala, i to, że ewaluacja naprawdę nie dotyka krytyka.

Testy z prawdziwym CityFlow są `slow` i chodzą na krótkim epizodzie — sprawdzają
mechanikę, nie jakość polityki.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tsc_marl import paths
from tsc_marl.core.graph import RoadNetGraph
from tsc_marl.core.loop import CoreTrainer, LoopConfig, world_order
from tsc_marl.core.nets import ActorCritic
from tsc_marl.core.ppo import PPO, PPOConfig

ROADNET = paths.DATA / "raw_data" / "jinan_3x4" / "roadnet_3_4.json"

# Krótki epizod: 10 decyzji zamiast 240. Testujemy mechanikę pętli, nie zbieżność.
SHORT = LoopConfig(steps=150, test_steps=150, action_interval=15, episodes=1)


@pytest.fixture(scope="module")
def graph():
    if not ROADNET.is_file():
        pytest.skip(f"brak roadnetu {ROADNET} (dane pobiera się osobno — patrz data/README.md)")
    return RoadNetGraph.from_roadnet(ROADNET)


# ------------------------------------------------------------------- szybkie


def test_loop_config_liczy_decyzje():
    """Δt = 15 s i epizod 3600 s dają 240 decyzji — liczba, od której zależy
    rozmiar bufora i długość segmentu GAE."""
    cfg = LoopConfig()
    assert cfg.decisions == 240 and cfg.test_decisions == 240


def test_world_order_przenumerowuje_akcje(graph):
    """NAJWAŻNIEJSZY szybki test w tym pliku. `World.step` indeksuje po SWOJEJ
    liście skrzyżowań, nasz wektor akcji po `node_idx`. Bez tej permutacji akcja
    trafia do niewłaściwego skrzyżowania i nic tego nie zgłasza — sieć po prostu
    uczy się szumu.

    Świat udajemy z ODWRÓCONĄ kolejnością: gdyby permutacja była no-opem, test
    przeszedłby przypadkiem na prawdziwym świecie, gdzie kolejności dziś się
    pokrywają.
    """
    fake = SimpleNamespace(
        intersections=[SimpleNamespace(id=nid) for nid in reversed(graph.node_idx2id)]
    )
    perm = world_order(graph, fake)
    akcje = np.arange(graph.n_nodes)  # akcja węzła i == i
    dla_swiata = akcje[perm]

    for pozycja, inter in enumerate(fake.intersections):
        assert dla_swiata[pozycja] == graph.node_id2idx[inter.id]
    assert list(perm) == list(reversed(range(graph.n_nodes)))


def test_world_order_zdejmuje_prefiks_sumo(graph):
    """`world_sumo` prefiksuje id `GS_`, graf trzyma je bez prefiksu (Faza 5)."""
    fake = SimpleNamespace(
        intersections=[SimpleNamespace(id=f"GS_{nid}") for nid in graph.node_idx2id]
    )
    assert list(world_order(graph, fake)) == list(range(graph.n_nodes))


def test_checkpoint_odtwarza_polityke_i_skale_wartosci(graph, tmp_path):
    """Checkpoint musi zawierać `ValueNorm`: bez statystyk wznowiony trening liczy
    wartości w innej skali niż zapisany krytyk. To ten sam rodzaj błędu, co
    zapisywanie sieci target zamiast online w `colight.py` (znalezione w Fazie 0)."""
    torch.manual_seed(0)
    ob_length = 24
    stub_env = SimpleNamespace(graph=graph, features=SimpleNamespace(ob_length=ob_length))
    trainer = CoreTrainer(stub_env, PPO(ActorCritic(ob_length, 8), PPOConfig()), SHORT)
    trainer.ppo.value_norm.update(torch.randn(100) * 7 + 3)

    x = torch.randn(graph.n_nodes, ob_length)
    with torch.no_grad():
        przed = trainer.ppo.ac.actor(x, graph.edge_index).clone()
    przed_skala = (float(trainer.ppo.value_norm.mean), float(trainer.ppo.value_norm.var))
    sciezka = tmp_path / "core.pt"
    trainer.save(sciezka)

    with torch.no_grad():  # psujemy wagi i statystyki, żeby load miał co odtworzyć
        for p in trainer.ppo.ac.actor.parameters():
            p.add_(1.0)
    trainer.ppo.value_norm.update(torch.full((1000,), 500.0))
    trainer.load(sciezka)

    with torch.no_grad():
        assert torch.allclose(trainer.ppo.ac.actor(x, graph.edge_index), przed)
    po_skala = (float(trainer.ppo.value_norm.mean), float(trainer.ppo.value_norm.var))
    assert po_skala == pytest.approx(przed_skala)


# ------------------------------------------------- ogon metryk (Faza 4), bez CityFlow


def _metrics_atrapa(n_phases, legalne, n_nodes=1, waiting=None):
    """`TrafficEnv` z podstawionymi współpracownikami — testujemy ARYTMETYKĘ ogona,
    nie silnik. `legalne[i]` = ile faz ma naprawdę węzeł i. `waiting` = mapa
    pas -> liczba stojących (domyślnie pusto, czyli zero popytu wszędzie)."""
    from collections import defaultdict  # noqa: PLC0415

    waiting = defaultdict(int) if waiting is None else defaultdict(int, waiting)
    from tsc_marl.core.loop import TrafficEnv  # noqa: PLC0415

    m = object.__new__(TrafficEnv)
    m.graph = SimpleNamespace(n_nodes=n_nodes)
    # Jeden ruch na fazę, ruch `m` obsługiwany przez fazę `m`; pasy nazwane `iN_m`,
    # żeby `popyt` dało się sterować zawartością `waiting`.
    incydencja = np.zeros((n_nodes, n_phases, n_phases), dtype=bool)
    for i in range(n_nodes):
        for f in range(n_phases):
            incydencja[i, f, f] = True
    m.features = SimpleNamespace(
        n_phases=n_phases,
        n_mov=n_phases,
        intersections=[SimpleNamespace(phases=list(range(k))) for k in legalne],
        structs=[
            SimpleNamespace(
                reward_lanes=[],
                movements=[([f"i{i}_{f}"], []) for f in range(n_phases)],
            )
            for i in range(n_nodes)
        ],
        phase_incidence=lambda: incydencja,
        rewards=lambda: np.zeros(n_nodes),
        observe=lambda: np.zeros((n_nodes, 1)),
    )
    m.world = SimpleNamespace(
        step=lambda _a: None,
        get_info=lambda _k: waiting,
        get_average_travel_time=lambda: 300.0,
        get_cur_throughput=lambda: 1000.0,
    )
    m.action_interval = 15
    m._to_world = np.arange(n_nodes)
    m._delay = [SimpleNamespace(generate=lambda: 0.0) for _ in range(n_nodes)]
    m._reset_stats()
    return m


def test_wait_max_mierzy_glodzenie_fazy():
    """Węzeł trzymany w fazie 0 głodzi pozostałe. Po k decyzjach najdłużej
    nieobsłużona czeka k decyzji = k × 15 s — i to jest liczba, której `att`
    ani `queue` nie pokazują."""
    m = _metrics_atrapa(n_phases=3, legalne=[3])
    for _ in range(10):
        m.step(np.array([0]))
    assert m.metrics()["wait_max"] == pytest.approx(10 * 15)


def test_wait_max_ignoruje_fazy_NIEISTNIEJACE():
    """PUŁAPKA, dla której ten test istnieje. `n_phases` to MAKSIMUM po sieci, więc
    węzeł o 2 fazach ma trzecią kolumnę martwą — nigdy nie dostanie zielonego.
    Bez maski legalnych faz `wait_max` liczyłby tę kolumnę i rósł liniowo z długością
    epizodu na KAŻDEJ sieci niejednorodnej (Kraków: 2–7 faz), udając głodzenie,
    którego nie ma. Tu węzeł na przemian obsługuje obie swoje fazy, więc prawdziwe
    zagłodzenie to jedna decyzja."""
    m = _metrics_atrapa(n_phases=3, legalne=[2])
    for i in range(10):
        m.step(np.array([i % 2]))
    assert m.metrics()["wait_max"] == pytest.approx(15), "faza 2 nie istnieje — nie głoduje"


def test_wait_max_demand_nie_karze_za_PUSTA_faze():
    """SEDNO rozdzielenia obu metryk. Węzeł trzymany w fazie 0 głodzi fazy 1 i 2,
    ale nikt na nich nie stoi — a w praktyce miejskiej wlot bez popytu może czekać
    dowolnie długo i bywa to celowe (Kraków: wloty bez detekcji dostają zielone
    bezwarunkowo). `wait_max` ma to POKAZAĆ jako odstęp, `wait_max_demand` ma
    MILCZEĆ, bo nikomu nic się nie stało."""
    m = _metrics_atrapa(n_phases=3, legalne=[3])
    for _ in range(10):
        m.step(np.array([0]))
    wynik = m.metrics()
    assert wynik["wait_max"] == pytest.approx(10 * 15), "surowy odstęp ma rosnąć"
    assert wynik["wait_max_demand"] == 0.0, "pusta faza to nie głodzenie"


def test_wait_max_demand_liczy_faze_NA_KTOREJ_KTOS_STOI():
    """Ta sama sytuacja, ale na ruchu fazy 1 stoi pojazd. Teraz to JEST głodzenie
    i obie metryki mają je pokazać."""
    m = _metrics_atrapa(n_phases=3, legalne=[3], waiting={"i0_1": 4})
    for _ in range(10):
        m.step(np.array([0]))
    wynik = m.metrics()
    assert wynik["wait_max"] == pytest.approx(10 * 15)
    assert wynik["wait_max_demand"] == pytest.approx(10 * 15)


def test_wait_max_demand_ignoruje_fazy_nieistniejace():
    """Ta sama pułapka co przy `wait_max`, ale groźniejsza: martwa kolumna NIGDY nie
    dostanie zielonego, więc jej wiek rośnie przez cały epizod, a gdyby jeszcze
    „miała popyt", metryka raportowałaby narastające głodzenie na fazie, której nie ma.
    Tu węzeł ma 2 fazy, obsługuje je na przemian (zero głodzenia), a 9 pojazdów stoi
    na ruchu fazy 2 — nieistniejącej. Bez maski legalnych faz wyszłoby 10 × 15 = 150 s."""
    m = _metrics_atrapa(n_phases=3, legalne=[2], waiting={"i0_2": 9})
    for i in range(10):
        m.step(np.array([i % 2]))
    assert m.metrics()["wait_max_demand"] == 0.0


def test_ogon_kolejki_to_maksimum_po_wezlach_a_nie_srednia():
    """`queue` uśrednia po sieci i po czasie — jedno zakorkowane skrzyżowanie przy
    trzech pustych znika. `queue_max` je pokazuje. To jest ta sama różnica, przez
    którą bramka jest per skrzyżowanie, a nie globalna."""
    m = _metrics_atrapa(n_phases=2, legalne=[2, 2, 2, 2], n_nodes=4)
    m._queue_sum = np.array([120.0, 0.0, 0.0, 0.0])
    m._delay_sum = np.zeros(4)
    m._decisions = 10
    wynik = m.metrics()
    assert wynik["queue"] == pytest.approx(3.0)  # 120 / (10 decyzji × 4 węzły)
    assert wynik["queue_max"] == pytest.approx(12.0)  # 120 / 10 na jednym węźle
    assert wynik["queue_p90"] > wynik["queue"]


# ----------------------------------------------------------- z prawdziwym CityFlow


@pytest.fixture(scope="module")
def trainer(advanced_colight_world):
    world, _ = advanced_colight_world
    torch.manual_seed(0)
    return CoreTrainer.from_world(world, loop_config=SHORT)


@pytest.mark.slow
def test_epizod_treningowy_wypelnia_bufor_i_daje_metryki(trainer):
    """End-to-end: 10 decyzji × 15 kroków silnika, bufor pełny, GAE policzone,
    metryki skończone. Sprawdza też, że nagroda kolejkowa jest ujemna (karzemy
    kolejkę) i że ATT wyszło dodatnie — czyli że silnik naprawdę chodził."""
    buffer, metryki = trainer.collect()

    assert len(buffer) == SHORT.decisions == 10
    assert buffer.obs.shape == (10, trainer.env.graph.n_nodes, trainer.env.features.ob_length)
    assert torch.isfinite(buffer.advantages).all() and torch.isfinite(buffer.returns).all()
    assert set(metryki) == {
        "att",
        "throughput",
        "reward",
        "queue",
        "delay",
        "queue_max",
        "queue_p90",
        "wait_max",
        "wait_max_demand",
    }
    assert metryki["queue_max"] >= metryki["queue_p90"] >= 0
    assert metryki["wait_max"] >= metryki["wait_max_demand"] >= 0, (
        "wiek z popytem jest podzbiorem wieku surowego"
    )
    assert all(np.isfinite(v) for v in metryki.values())
    assert metryki["att"] > 0 and metryki["queue"] >= 0
    assert metryki["reward"] <= 0, "nagroda kolejkowa nie może być dodatnia"


@pytest.mark.slow
def test_akcje_trafiaja_do_wlasciwych_skrzyzowan(trainer):
    """Behawioralny test permutacji: nadajemy KAŻDEMU węzłowi inną fazę i czytamy
    ze świata, co się faktycznie ustawiło. Interwał 15 s > yellow 5 s, więc po
    jednej decyzji faza docelowa jest już wykonana."""
    env = trainer.env
    env.reset()
    n_faz = env.features.n_phases
    akcje = np.array([(i % (n_faz - 1)) + 1 for i in range(env.graph.n_nodes)])  # nigdy 0

    env.step(akcje)

    for inter in env.world.intersections:
        node_idx = env.graph.node_id2idx[inter.id]
        assert inter.current_phase == akcje[node_idx], (
            f"{inter.id}: świat ma fazę {inter.current_phase}, "
            f"a węzeł {node_idx} dostał {akcje[node_idx]}"
        )


@pytest.mark.slow
def test_ewaluacja_nie_dotyka_krytyka_i_jest_powtarzalna(trainer):
    """„Inferencja = tylko actor" z planu 1.4 — tak samo będzie na Pi w Fazie 5.
    Krytyk podmieniamy na wybuchający, więc każde jego wywołanie wywali test.

    Powtarzalność: ta sama polityka deterministyczna na zresetowanym silniku musi
    dać identyczne ATT. Gdyby nie dawała, żadne porównanie „przed/po awarii"
    z Fazy 3 nie miałoby sensu.
    """

    def wybuch(*args, **kwargs):
        raise AssertionError("krytyk wołany w ewaluacji")

    oryginal = trainer.ppo.ac.critic.forward
    trainer.ppo.ac.critic.forward = wybuch
    try:
        pierwsze = trainer.evaluate()
        drugie = trainer.evaluate()
    finally:
        trainer.ppo.ac.critic.forward = oryginal

    assert pierwsze["att"] == pytest.approx(drugie["att"], rel=1e-9)
    assert pierwsze["throughput"] == drugie["throughput"]


# ----------------------------------------------------------------- urządzenie


@pytest.mark.slow
def test_rdzen_domyslnie_na_cpu(trainer):
    """Rdzeń bez jawnego `device` stoi na CPU — sieć, bufor i ValueNorm razem.

    Rozjazd między nimi nie jest hipotetyczny: `ValueNorm` trzyma statystyki
    w buforach i gdyby został na CPU przy sieci na GPU, `denormalize` dostałoby
    tensory z dwóch urządzeń dopiero przy PIERWSZYM update'cie — czyli po
    kilkudziesięciu sekundach zbierania epizodu, nie przy starcie.
    """
    core = trainer

    assert core.device.type == "cpu"
    assert next(core.ppo.ac.parameters()).device == core.device
    assert core.edge_index.device == core.device
    assert core.ppo.value_norm.mean.device == core.device


@pytest.mark.slow
def test_edge_index_jedzie_na_urzadzenie_raz(trainer):
    """`edge_index` jest stały przez cały bieg, więc `CoreTrainer` trzyma własną
    kopię już przeniesioną — pętla decyzyjna nie może go przenosić co krok."""
    core = trainer

    assert core.edge_index.device == core.device
    assert torch.equal(core.edge_index, core.env.graph.edge_index)


@pytest.mark.slow
def test_granica_numpy_dziala_w_obie_strony(trainer):
    """`_obs`/`_actions` to jedyne dwa miejsca, w których dane przekraczają
    granicę urządzenia. `.cpu()` w `_actions` jest no-opem na CPU i warunkiem
    działania na CUDA — test pilnuje, że nie zniknie jako „zbędny"."""
    core = trainer

    obs = core._obs(core.env.reset())
    assert obs.device == core.device

    akcje = core.ppo.ac.actor(obs, core.edge_index).argmax(-1)
    numpy_akcje = core._actions(akcje)
    assert isinstance(numpy_akcje, np.ndarray)
    assert numpy_akcje.shape == (core.env.graph.n_nodes,)
