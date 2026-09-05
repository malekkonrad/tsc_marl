"""Testy pętli MAPPO.

Błędy w PPO są wyjątkowo drogie diagnostycznie: nic się nie wywala, trening po
prostu nie zbiega, a przyczyną bywa jeden znak we wzorze na GAE albo przestawiona
kolejność przy spłaszczaniu [T, N] → [T·N]. Dlatego GAE porównujemy z NIEZALEŻNĄ
implementacją (jawna suma po definicji, nie ta sama rekurencja), sklejanie grafów
z `torch_geometric.Batch`, a na koniec sprawdzamy, że całość faktycznie się uczy.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch_geometric.data import Batch, Data

from tsc_marl import paths
from tsc_marl.core.graph import RoadNetGraph
from tsc_marl.core.nets import ActorCritic
from tsc_marl.core.ppo import (
    PPO,
    PPOConfig,
    RolloutBuffer,
    ValueNorm,
    batch_vector,
    compute_gae,
    tiled_edge_index,
)

ROADNET = paths.DATA / "raw_data" / "jinan_3x4" / "roadnet_3_4.json"
OB_LENGTH = 24
N_PHASES = 8


@pytest.fixture(scope="module")
def graph():
    if not ROADNET.is_file():
        pytest.skip(f"brak roadnetu {ROADNET} (dane pobiera się osobno — patrz data/README.md)")
    return RoadNetGraph.from_roadnet(ROADNET)


# ----------------------------------------------------------------------- GAE


def gae_reference(rewards, values, dones, last_value, gamma, lam):
    """Niezależna implementacja Z DEFINICJI: A_t = Σ_l (γλ)^l δ_{t+l}.

    Celowo napisana jako jawna suma, a nie rekurencja wstecz — gdyby oba kody
    miały ten sam błąd, test nie byłby wart nic."""
    T, N = rewards.shape
    adv = torch.zeros(T, N)
    for t in range(T):
        for i in range(N):
            acc, coef = 0.0, 1.0
            for step in range(t, T):
                nv = last_value[i] if step == T - 1 else values[step + 1, i]
                nonterm = 1.0 - float(dones[step, i])
                delta = float(rewards[step, i]) + gamma * nonterm * float(nv)
                delta -= float(values[step, i])
                acc += coef * delta
                coef *= gamma * lam * nonterm
            adv[t, i] = acc
    return adv


@pytest.fixture
def segment():
    g = torch.Generator().manual_seed(0)
    T, N = 12, 4
    return (
        torch.randn(T, N, generator=g),  # rewards
        torch.randn(T, N, generator=g),  # values
        torch.zeros(T, N),  # dones
        torch.randn(N, generator=g),  # last_value
    )


@pytest.mark.parametrize("lam", [0.0, 0.5, 0.95, 1.0])
def test_gae_zgadza_sie_z_definicja(segment, lam):
    rewards, values, dones, last_value = segment
    adv, ret = compute_gae(rewards, values, dones, last_value, gamma=0.95, gae_lambda=lam)
    oczekiwane = gae_reference(rewards, values, dones, last_value, 0.95, lam)
    assert torch.allclose(adv, oczekiwane, atol=1e-5)
    assert torch.allclose(ret, adv + values)


def test_gae_nie_przecieka_przez_koniec_epizodu(segment):
    """`done` MUSI uciąć bootstrap. Bez tego przewaga ostatniego kroku epizodu
    zawiera wartość stanu z epizodu NASTĘPNEGO — pierwszego po resecie, czyli
    pustej sieci drogowej. Nagrody są wtedy systematycznie zawyżone."""
    rewards, values, dones, last_value = segment
    dones = dones.clone()
    dones[5] = 1.0
    adv, _ = compute_gae(rewards, values, dones, last_value, 0.95, 0.95)
    assert torch.allclose(
        adv, gae_reference(rewards, values, dones, last_value, 0.95, 0.95), atol=1e-5
    )
    assert torch.allclose(adv[5], rewards[5] - values[5], atol=1e-6)


def test_gae_bootstrapuje_ostatni_krok(segment):
    """Epizod TSC kończy się po 3600 s, ale stan nie jest terminalny (ruch płynie
    dalej), więc ostatni krok musi bootstrapować `last_value`. Gdyby zamiast tego
    wstawić zero, wartości pod koniec epizodu byłyby zaniżone o V(s_T)."""
    rewards, values, dones, last_value = segment
    adv, _ = compute_gae(rewards, values, dones, last_value, 0.95, 0.0)
    assert torch.allclose(adv[-1], rewards[-1] + 0.95 * last_value - values[-1], atol=1e-6)


def test_gae_wymaga_zgodnych_ksztaltow(segment):
    rewards, values, dones, last_value = segment
    with pytest.raises(ValueError, match="ten sam kształt"):
        compute_gae(rewards, values[:-1], dones, last_value, 0.95, 0.95)


# ------------------------------------------------------------- sklejanie grafów


@pytest.mark.parametrize("k", [1, 2, 5])
def test_sklejanie_grafow_zgadza_sie_z_pyg(graph, k):
    """`tiled_edge_index` to skrót za `Batch.from_data_list` — wolno go używać
    tylko dlatego, że topologia jest statyczna. Test pilnuje, że skrót naprawdę
    daje to samo; gdyby offsety były źle policzone, agent z kroku t dostawałby
    sąsiadów z kroku t+1 i nikt by tego nie zauważył."""
    n, ei = graph.n_nodes, graph.edge_index
    ref = Batch.from_data_list([Data(x=torch.zeros(n, 1), edge_index=ei) for _ in range(k)])
    assert torch.equal(tiled_edge_index(ei, n, k), ref.edge_index)
    assert torch.equal(batch_vector(n, k), ref.batch)


# ------------------------------------------------------------------- ValueNorm


def test_value_norm_liczy_prawdziwe_statystyki():
    vn = ValueNorm()
    g = torch.Generator().manual_seed(1)
    partie = [torch.randn(50, 7, generator=g) * 13 + 4 for _ in range(6)]
    for p in partie:
        vn.update(p)
    wszystko = torch.cat([p.reshape(-1) for p in partie])
    assert math.isclose(float(vn.mean), float(wszystko.mean()), rel_tol=1e-4)
    assert math.isclose(float(vn.var), float(wszystko.var(unbiased=False)), rel_tol=1e-3)


def test_value_norm_jest_odwracalna():
    """Krytyk uczy się w przestrzeni znormalizowanej, ale GAE liczy się na skali
    surowej — bez dokładnej odwracalności przewagi byłyby w innych jednostkach
    niż nagrody."""
    vn = ValueNorm()
    vn.update(torch.randn(100) * 5 - 2)
    x = torch.randn(20)
    assert torch.allclose(vn.denormalize(vn.normalize(x)), x, atol=1e-5)


def test_value_norm_wchodzi_do_checkpointu():
    """Statystyki to bufory, nie parametry: mają się zapisać w `state_dict`
    (wznowiony trening liczyłby inaczej), ale nie dostawać gradientu."""
    vn = ValueNorm()
    vn.update(torch.arange(10.0))
    assert set(vn.state_dict()) == {"mean", "var", "count"}
    assert list(vn.parameters()) == []


# ----------------------------------------------------------------------- bufor


def make_buffer(graph, capacity=8):
    return RolloutBuffer(capacity, graph.n_nodes, OB_LENGTH, graph.edge_index)


def fill(buf, n_steps, n_nodes, seed=0):
    g = torch.Generator().manual_seed(seed)
    for t in range(n_steps):
        buf.add(
            obs=torch.full((n_nodes, OB_LENGTH), float(t)),
            actions=torch.randint(N_PHASES, (n_nodes,), generator=g),
            log_probs=torch.randn(n_nodes, generator=g),
            values=torch.randn(n_nodes, generator=g),
            rewards=torch.randn(n_nodes, generator=g),
        )


def test_bufor_zachowuje_kolejnosc_przy_splaszczaniu(graph):
    """KONTRAKT: [t, i] → wiersz t·N + i. Tak samo układa węzły `tiled_edge_index`,
    więc pomyłka tutaj przypisałaby przewagę jednego skrzyżowania obserwacji
    drugiego — bez błędu, tylko gorszy wynik."""
    buf = make_buffer(graph)
    n = graph.n_nodes
    for t in range(4):
        znacznik = torch.arange(n, dtype=torch.float32) + 100 * t
        buf.add(
            obs=znacznik[:, None].repeat(1, OB_LENGTH),
            actions=torch.zeros(n, dtype=torch.long),
            log_probs=torch.zeros(n),
            values=torch.zeros(n),
            rewards=znacznik,
        )
    buf.compute_returns(torch.zeros(n), PPOConfig())
    (mb,) = list(buf.minibatches(1))
    kroki = mb.x[:, 0].reshape(-1, n)
    for wiersz in range(kroki.shape[0]):
        assert torch.equal(kroki[wiersz] % 100, torch.arange(n, dtype=torch.float32))


def test_bufor_pilnuje_pojemnosci_i_kolejnosci_wywolan(graph):
    buf = make_buffer(graph, capacity=3)
    fill(buf, 3, graph.n_nodes)
    assert buf.full and len(buf) == 3
    with pytest.raises(IndexError, match="pełny"):
        fill(buf, 1, graph.n_nodes)
    with pytest.raises(RuntimeError, match="compute_returns"):
        list(buf.minibatches(2))
    buf.reset()
    with pytest.raises(RuntimeError, match="pusty"):
        buf.compute_returns(torch.zeros(graph.n_nodes), PPOConfig())


def test_minibatche_pokrywaja_kazdy_krok_dokladnie_raz(graph):
    """Podział idzie po KROKACH CZASU, nie po agentach — z połowy skrzyżowań
    nie da się policzyć attention. Każdy krok ma wejść do dokładnie jednego
    minibatcha, inaczej część danych z epizodu przepada albo liczy się podwójnie."""
    buf = make_buffer(graph, capacity=7)
    fill(buf, 7, graph.n_nodes)
    buf.compute_returns(torch.zeros(graph.n_nodes), PPOConfig())

    widziane = []
    for mb in buf.minibatches(3):
        k = mb.batch.max().item() + 1
        assert mb.x.shape == (k * graph.n_nodes, OB_LENGTH)
        assert mb.edge_index.shape == (2, k * graph.n_edges)
        widziane += mb.x[:: graph.n_nodes, 0].tolist()
    assert sorted(widziane) == list(range(7))


# ------------------------------------------------------------------- update PPO


def make_ppo(graph, seed=0, **cfg):
    torch.manual_seed(seed)
    return PPO(ActorCritic(OB_LENGTH, N_PHASES), PPOConfig(**cfg))


def rollout(ppo, graph, steps=8, seed=0):
    buf = RolloutBuffer(steps, graph.n_nodes, OB_LENGTH, graph.edge_index)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        obs = torch.randn(graph.n_nodes, OB_LENGTH, generator=g)
        actions, log_probs, values = ppo.act(obs, graph.edge_index)
        buf.add(obs, actions, log_probs, values, torch.randn(graph.n_nodes, generator=g))
    buf.compute_returns(ppo.value_of(obs, graph.edge_index), ppo.config)
    return buf


def test_update_z_zerowym_lr_nie_rusza_polityki(graph):
    """Przy lr=0 ratio musi zostać 1.0 przez WSZYSTKIE epoki, więc KL = 0 i nic
    się nie clipuje. Jeśli te liczby odjeżdżają mimo zerowego kroku, znaczy że
    `act` i `evaluate` liczą różne rzeczy — a to objawia się później jako
    „PPO dziwnie się zachowuje od startu"."""
    ppo = make_ppo(graph, actor_lr=0.0, critic_lr=0.0, epochs=3)
    przed = [p.clone() for p in ppo.ac.parameters()]
    stats = ppo.update(rollout(ppo, graph))

    assert stats["approx_kl"] == pytest.approx(0.0, abs=1e-7)
    assert stats["clip_frac"] == pytest.approx(0.0, abs=1e-7)
    assert all(torch.equal(a, b) for a, b in zip(przed, ppo.ac.parameters(), strict=True))


def test_update_uczy_i_raportuje_diagnostyke(graph):
    """`approx_kl` i `clip_frac` to jedyny wczesny sygnał, że lr jest za duży;
    `entropy` — że polityka przedwcześnie się zdeterminowała. W 1.5 idą na
    krzywe uczenia obok ATT, więc muszą wychodzić z update'u."""
    ppo = make_ppo(graph, epochs=2)
    przed = [p.clone() for p in ppo.ac.parameters()]
    stats = ppo.update(rollout(ppo, graph))

    assert set(stats) >= {"policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"}
    assert stats["n_minibatches"] == 2 * 4  # epochs × n_minibatches (8 kroków / 4)
    assert any(not torch.equal(a, b) for a, b in zip(przed, ppo.ac.parameters(), strict=True))
    assert 0.0 <= stats["entropy"] <= math.log(N_PHASES) + 1e-6


def test_update_przerywa_na_target_kl(graph):
    """Hamulec awaryjny: przy absurdalnie dużym lr polityka odjeżdża w pierwszym
    minibatchu i dalsze epoki uczyłyby się już z danych zupełnie innej polityki."""
    ppo = make_ppo(graph, actor_lr=1.0, critic_lr=1.0, epochs=10, target_kl=1e-6)
    stats = ppo.update(rollout(ppo, graph))
    assert stats["early_stopped"] == 1.0
    assert stats["n_minibatches"] < 10 * 4


def test_config_odrzuca_bezsensowne_hiperparametry():
    for zle in ({"gamma": 1.5}, {"gae_lambda": -0.1}, {"epochs": 0}, {"clip_range": 0.0}):
        with pytest.raises(ValueError):
            PPOConfig(**zle)


@pytest.mark.slow
def test_ppo_faktycznie_sie_uczy(graph):
    """Test end-to-end na zadaniu, którego rozwiązanie znamy: nagroda 1 za jedną
    ustaloną fazę, 0 za resztę. Jeśli cała pętla jest poprawna, prawdopodobieństwo
    tej fazy musi wyraźnie wzrosnąć ponad start 1/8, a entropia spaść.

    To jedyny test, który łapie błędy ZNAKU — odwrócona przewaga albo `+` zamiast
    `-` przy bonusie entropijnym przechodzi wszystkie testy kształtów i dopiero
    tutaj widać, że polityka uczy się w złą stronę."""
    cel, kroki = 3, 16
    ppo = make_ppo(graph, epochs=4, n_minibatches=2, actor_lr=3e-3, critic_lr=3e-3)
    obs = torch.randn(graph.n_nodes, OB_LENGTH, generator=torch.Generator().manual_seed(5))

    torch.manual_seed(0)
    for _ in range(30):
        buf = RolloutBuffer(kroki, graph.n_nodes, OB_LENGTH, graph.edge_index)
        for _ in range(kroki):
            actions, log_probs, values = ppo.act(obs, graph.edge_index)
            buf.add(obs, actions, log_probs, values, (actions == cel).float())
        buf.compute_returns(ppo.value_of(obs, graph.edge_index), ppo.config)
        stats = ppo.update(buf)

    with torch.no_grad():
        probs = torch.softmax(ppo.ac.actor(obs, graph.edge_index), dim=-1)
    assert probs[:, cel].mean() > 0.5, f"polityka nie znalazła fazy {cel}: {probs.mean(0)}"
    assert stats["entropy"] < math.log(N_PHASES) - 0.3
