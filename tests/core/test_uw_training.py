"""Testy treningu polityki Z mechanizmem (Faza 3, §3).

Cały ten plik istnieje dla JEDNEGO błędu: sklejenia log ρ w minibatchu. `tiled_log_rho`
może zwrócić wektor o POPRAWNEJ DŁUGOŚCI i mylnym układzie — wtedy nic nie pada,
asercja w `uw_attention._forward` przechodzi, PyG przepuszcza argument do `message`
bez sprawdzenia, a trening po prostu uczy się innego sygnału, niż się wydaje. To jest
pułapka nr 3 z docstringu `uw_attention.py`, tu uderzająca realnie.

Dlatego kluczowy test nie porównuje kształtów, tylko WARTOŚCI: puszcza k kroków przez
encoder pojedynczo i porównuje z jednym przebiegiem po sklejonym grafie. Gdyby układ
był pomieszany, wyjścia się rozjadą; gdyby test sprawdzał samą długość — nie.
"""

from __future__ import annotations

import pytest
import torch

from tests.core.test_ppo import N_PHASES, OB_LENGTH, ROADNET
from tsc_marl.core.graph import RoadNetGraph
from tsc_marl.core.nets import ActorCritic, GATEncoder
from tsc_marl.core.ppo import (
    PPO,
    PPOConfig,
    RolloutBuffer,
    batch_vector,
    tiled_edge_index,
    tiled_log_rho,
)


@pytest.fixture(scope="module")
def graph():
    if not ROADNET.is_file():
        pytest.skip(f"brak roadnetu {ROADNET} (dane pobiera się osobno — patrz data/README.md)")
    return RoadNetGraph.from_roadnet(ROADNET)


def losowe_log_rho(n_steps: int, n_edges: int, n_nodes: int, seed: int = 0) -> torch.Tensor:
    """Wyraźnie UJEMNE i RÓŻNE na każdej pozycji — zero i stała są niewidoczne dla
    softmaxu (niezmienniczość na stałą wspólną dla grupy), więc na nich test
    przeszedłby także przy pomieszanym układzie."""
    g = torch.Generator().manual_seed(seed)
    return -torch.rand(n_steps, n_edges + n_nodes, generator=g) * 5.0


# ------------------------------------------------------- sklejanie log ρ w batchu


def test_tiled_log_rho_zgadza_sie_z_forwardem_per_krok(graph):
    """JĄDRO CAŁEJ FAZY 3. k kroków sklejonych w jeden graf musi dać dokładnie to,
    co k osobnych forwardów — także wtedy, gdy każdy krok ma INNE log ρ."""
    torch.manual_seed(0)
    enc = GATEncoder(OB_LENGTH).eval()
    n, e, k = graph.n_nodes, graph.edge_index.shape[1], 4

    g = torch.Generator().manual_seed(1)
    x = torch.randn(k, n, OB_LENGTH, generator=g)
    lr = losowe_log_rho(k, e, n)

    with torch.no_grad():
        osobno = torch.cat([enc(x[t], graph.edge_index, log_rho=lr[t]) for t in range(k)])
        razem = enc(
            x.reshape(k * n, OB_LENGTH),
            tiled_edge_index(graph.edge_index, n, k),
            log_rho=tiled_log_rho(lr, e, n),
        )
    assert torch.allclose(osobno, razem, atol=1e-5)


def test_naiwne_splaszczenie_ma_dobra_dlugosc_i_zly_wynik(graph):
    """Dowód, że ten test w ogóle coś sprawdza: wariant, który kusi
    (`log_rho.reshape(-1)`), przechodzi WSZYSTKIE asercje kształtu i liczy co innego.

    Gdyby ten test zaczął padać, znaczyłoby to, że zły układ daje ten sam wynik —
    czyli że pozostałe testy nie mają mocy rozróżniającej."""
    torch.manual_seed(0)
    enc = GATEncoder(OB_LENGTH).eval()
    n, e, k = graph.n_nodes, graph.edge_index.shape[1], 4
    lr = losowe_log_rho(k, e, n)

    naiwne = lr.reshape(-1)
    dobre = tiled_log_rho(lr, e, n)
    assert naiwne.shape == dobre.shape  # ← dlatego nic tego nie łapie samo z siebie
    assert not torch.equal(naiwne, dobre)

    g = torch.Generator().manual_seed(1)
    x = torch.randn(k * n, OB_LENGTH, generator=g)
    ei = tiled_edge_index(graph.edge_index, n, k)
    with torch.no_grad():
        assert not torch.allclose(enc(x, ei, log_rho=naiwne), enc(x, ei, log_rho=dobre), atol=1e-5)


def test_tiled_log_rho_odrzuca_zly_ksztalt(graph):
    n, e = graph.n_nodes, graph.edge_index.shape[1]
    with pytest.raises(ValueError, match="E\\+N"):
        tiled_log_rho(torch.zeros(3, e), e, n)  # brak pętli własnych
    with pytest.raises(ValueError, match="E\\+N"):
        tiled_log_rho(torch.zeros(e + n), e, n)  # jeden wymiar zamiast [k, E+N]


# --------------------------------------------------------------- bufor rolloutu


def bufor(graph, k: int, with_log_rho: bool) -> RolloutBuffer:
    return RolloutBuffer(k, graph.n_nodes, OB_LENGTH, graph.edge_index, with_log_rho=with_log_rho)


def wypelnij(buf: RolloutBuffer, lr: torch.Tensor | None, seed: int = 2) -> None:
    g = torch.Generator().manual_seed(seed)
    n = buf.n_nodes
    for t in range(buf.capacity):
        buf.add(
            torch.randn(n, buf.ob_length, generator=g),
            torch.zeros(n, dtype=torch.long),
            torch.zeros(n),
            torch.zeros(n),
            torch.randn(n, generator=g),
            log_rho=None if lr is None else lr[t],
        )


def test_bufor_wymaga_zgodnosci_z_with_log_rho(graph):
    """Pomyłka w tę stronę jest CICHA: bieg bez mechanizmu, który miał go mieć,
    kończy się liczbami nieodróżnialnymi od ramienia kontrolnego."""
    n, e = graph.n_nodes, graph.edge_index.shape[1]
    with pytest.raises(ValueError, match="with_log_rho"):
        bufor(graph, 2, True).add(
            torch.zeros(n, OB_LENGTH),
            torch.zeros(n, dtype=torch.long),
            torch.zeros(n),
            torch.zeros(n),
            torch.zeros(n),
        )
    with pytest.raises(ValueError, match="with_log_rho"):
        bufor(graph, 2, False).add(
            torch.zeros(n, OB_LENGTH),
            torch.zeros(n, dtype=torch.long),
            torch.zeros(n),
            torch.zeros(n),
            torch.zeros(n),
            log_rho=torch.zeros(e + n),
        )
    with pytest.raises(ValueError, match="E \\+ N"):
        bufor(graph, 2, True).add(
            torch.zeros(n, OB_LENGTH),
            torch.zeros(n, dtype=torch.long),
            torch.zeros(n),
            torch.zeros(n),
            torch.zeros(n),
            log_rho=torch.zeros(e),  # bez pętli własnych
        )


def test_minibatch_niesie_log_rho_w_kolejnosci_sklejonego_grafu(graph):
    """log ρ musi jechać przez tę samą permutację kroków co obserwacje — inaczej
    krok `t` dostałby wiarygodność krawędzi z innego momentu epizodu."""
    n, e, k = graph.n_nodes, graph.edge_index.shape[1], 6
    lr = losowe_log_rho(k, e, n)
    buf = bufor(graph, k, True)
    wypelnij(buf, lr)
    buf.compute_returns(torch.zeros(n), PPOConfig())

    widziane = 0
    for mb in buf.minibatches(2, torch.Generator().manual_seed(0)):
        m = mb.x.shape[0] // n  # kroków w tym minibatchu
        assert mb.log_rho is not None
        assert mb.log_rho.shape == (m * (e + n),)
        # odtwarzamy, z których kroków powstał minibatch, po samych obserwacjach…
        kroki = [
            next(t for t in range(k) if torch.equal(buf.obs[t], mb.x[i * n : (i + 1) * n]))
            for i in range(m)
        ]
        # …i żądamy, żeby log ρ dało się złożyć z DOKŁADNIE tych samych kroków
        assert torch.equal(mb.log_rho, tiled_log_rho(lr[kroki], e, n))
        widziane += m
    assert widziane == k


def test_bufor_bez_mechanizmu_nie_niesie_log_rho(graph):
    """None, nie zera. `log_rho=None` prowadzi w warstwie INNĄ ścieżką niż wektor
    zer (choć wynik jest ten sam) i tylko ona jest bit w bit rdzeniem z Fazy 1."""
    buf = bufor(graph, 3, False)
    wypelnij(buf, None)
    buf.compute_returns(torch.zeros(graph.n_nodes), PPOConfig())
    assert buf.log_rho is None
    assert all(mb.log_rho is None for mb in buf.minibatches(2))


# --------------------------------------------------------------- ActorCritic + PPO


def test_act_i_evaluate_widza_ten_sam_rozklad(graph):
    """Ratio w pierwszej epoce PPO musi wyjść 1.0 także z mechanizmem. Gdyby log ρ
    trafiało tylko do jednej z tych ścieżek, update liczyłby gradient innej sieci
    niż ta, która zbierała dane — bez żadnego błędu."""
    torch.manual_seed(0)
    ac = ActorCritic(OB_LENGTH, N_PHASES).eval()
    n, e = graph.n_nodes, graph.edge_index.shape[1]
    x = torch.randn(n, OB_LENGTH, generator=torch.Generator().manual_seed(3))
    lr = losowe_log_rho(1, e, n)[0]

    with torch.no_grad():
        akcje, log_prob, _ = ac.act(x, graph.edge_index, deterministic=True, log_rho=lr)
        znowu, _, _ = ac.evaluate(x, graph.edge_index, akcje, log_rho=lr)
    assert torch.allclose(torch.exp(znowu - log_prob), torch.ones(n), atol=1e-6)


def test_log_rho_naprawde_zmienia_polityke(graph):
    """Bez tego wszystkie pozostałe testy przeszłyby też wtedy, gdyby argument był
    po drodze gubiony (np. zjadany przez `**kwargs`)."""
    torch.manual_seed(0)
    ac = ActorCritic(OB_LENGTH, N_PHASES).eval()
    n, e = graph.n_nodes, graph.edge_index.shape[1]
    x = torch.randn(n, OB_LENGTH, generator=torch.Generator().manual_seed(4))
    lr = losowe_log_rho(1, e, n)[0]

    with torch.no_grad():
        bez = ac.distribution(x, graph.edge_index).logits
        z = ac.distribution(x, graph.edge_index, log_rho=lr).logits
        zera = ac.distribution(x, graph.edge_index, log_rho=torch.zeros(e + n)).logits
    assert not torch.allclose(bez, z, atol=1e-5)
    assert torch.equal(bez, zera), "log ρ = 0 musi być no-opem (A3a)"


def test_krytyk_nie_widzi_log_rho(graph):
    """Decyzja z docstringu `Critic`: modulacja opisuje WDROŻENIE (padły czujniki),
    a krytyk żyje tylko w treningu. Podanie mu wiarygodności byłoby wyciekiem od tej
    gorszej strony CTDE — actor w inferencji tej informacji nie ma."""
    torch.manual_seed(0)
    ac = ActorCritic(OB_LENGTH, N_PHASES).eval()
    n, e = graph.n_nodes, graph.edge_index.shape[1]
    x = torch.randn(n, OB_LENGTH, generator=torch.Generator().manual_seed(5))
    lr = losowe_log_rho(1, e, n)[0]

    with torch.no_grad():
        _, _, v_bez = ac.act(x, graph.edge_index, deterministic=True)
        _, _, v_z = ac.act(x, graph.edge_index, deterministic=True, log_rho=lr)
    assert torch.equal(v_bez, v_z)


def test_update_z_mechanizmem_uczy_sie(graph):
    """End-to-end: bufor z log ρ przechodzi przez pełny update i rusza politykę.
    Nie sprawdza JAKOŚCI uczenia — sprawdza, że nic po drodze nie urywa gradientu."""
    torch.manual_seed(0)
    n, e, k = graph.n_nodes, graph.edge_index.shape[1], 8
    ppo = PPO(ActorCritic(OB_LENGTH, N_PHASES), PPOConfig(epochs=2, n_minibatches=2))
    lr = losowe_log_rho(k, e, n)
    buf = bufor(graph, k, True)

    g = torch.Generator().manual_seed(7)
    x0 = torch.randn(n, OB_LENGTH, generator=g)
    przed = ppo.ac.actor(x0, graph.edge_index, log_rho=lr[0]).detach().clone()
    wypelnij(buf, lr)
    buf.compute_returns(torch.zeros(n), ppo.config)

    stats = ppo.update(buf, torch.Generator().manual_seed(0))
    assert stats["n_minibatches"] > 0
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())
    po = ppo.ac.actor(x0, graph.edge_index, log_rho=lr[0]).detach()
    assert not torch.allclose(przed, po), "update nie ruszył polityki — gradient się urwał?"


def test_batch_vector_i_log_rho_opisuja_te_same_grafy(graph):
    """Sanity: `batch_vector` (krytyk) i `tiled_log_rho` (actor) muszą numerować
    kroki tak samo, inaczej kontekst globalny opisywałby inny moment niż modulacja."""
    n, e, k = graph.n_nodes, graph.edge_index.shape[1], 3
    assert batch_vector(n, k).shape == (k * n,)
    assert tiled_log_rho(torch.zeros(k, e + n), e, n).shape == (k * (e + n),)
