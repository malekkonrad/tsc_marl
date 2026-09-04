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

from tsc_marl import libsignal
from tsc_marl.graph import RoadNetGraph
from tsc_marl.loop import CoreTrainer, LoopConfig, world_order
from tsc_marl.nets import ActorCritic
from tsc_marl.ppo import PPO, PPOConfig

ROADNET = libsignal.DATA / "raw_data" / "jinan_3x4" / "roadnet_3_4.json"

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
    assert set(metryki) == {"att", "throughput", "reward", "queue", "delay"}
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
