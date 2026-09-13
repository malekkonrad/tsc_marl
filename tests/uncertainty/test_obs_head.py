"""Testy prototypu 2.2 — głowa obserwacyjna.

Trzy rzeczy, które potrafią cicho unieważnić wynik: okno czasowe sięgające do
poprzedniego epizodu (przeciek stanu sprzed `world.reset()`), mapowanie ruch →
krawędź na niewłaściwego sąsiada (per-edge claim stoi na tym jednym mapowaniu)
i AUROC liczony z odwróconą etykietą.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsc_marl import paths
from tsc_marl.uncertainty import obs_head as O
from tsc_marl.uncertainty.prototype import Loaded


def test_okno_nie_siega_do_poprzedniego_epizodu():
    episode = np.array([0, 0, 0, 1, 1, 1])
    win = O.window_index(episode, 3)
    assert win[0].tolist() == [0, 0, 0]
    assert win[2].tolist() == [2, 1, 0]
    assert win[3].tolist() == [3, 3, 3], "początek epizodu 1 nie widzi epizodu 0"
    assert win[5].tolist() == [5, 4, 3]


def test_macierz_sasiadow_usrednia_upstream():
    # krawędzie 0->2 i 1->2: węzeł 2 uśrednia 0 i 1; węzeł 0 nie ma nikogo
    adj = O.neighbor_matrix(np.array([[0, 1], [2, 2]]), 3)
    assert adj[2].tolist() == [0.5, 0.5, 0.0]
    assert adj[0].tolist() == [0.0, 0.0, 0.0]


def test_wejscia_okno_i_sasiedzi():
    x = np.arange(2 * 2 * 1, dtype=np.float32).reshape(2, 2, 1)  # [T=2, N=2, F=1]
    win = O.window_index(np.array([0, 0]), 2)
    adj = O.neighbor_matrix(np.array([[0], [1]]), 2)  # 0 -> 1
    z = O.build_inputs(x, win, adj)
    assert z.shape == (2, 2, 3)
    # t=1, węzeł 1: [x(1,1), x(0,1), średnia sąsiadów = x(1,0)]
    assert z[1, 1].tolist() == [3.0, 1.0, 2.0]


def test_auroc_kierunek_i_remisy():
    assert O.auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert O.auroc(np.array([0.0]), np.array([1.0])) == 0.0
    assert O.auroc(np.ones(5), np.ones(5)) == pytest.approx(0.5)


def test_krawedz_agreguje_tylko_ruchy_z_tego_sasiada():
    u = np.array([[[1.0, 5.0, 2.0]]])  # [T=1, N=1, n_mov=3]
    sources = np.array([[7, 7, -1]])  # dwa ruchy z węzła 7, jeden z brzegu
    u_max, edges = O.edge_uncertainty(u, sources, "max")
    u_mean, _ = O.edge_uncertainty(u, sources, "mean")
    assert edges == [(0, 7)]
    assert u_max[0, 0] == 5.0 and u_mean[0, 0] == 3.0


def test_zrodla_ruchow_to_prawdziwi_sasiedzi_grafu():
    """Każdy ruch z niebrzegowej drogi wjazdowej ma wskazywać węzeł, który jest
    sąsiadem upstream w grafie — inaczej wygaszalibyśmy nie tę krawędź."""
    roadnet = paths.DATA / "raw_data" / "jinan_3x4" / "roadnet_3_4_4phase.json"
    if not roadnet.is_file():
        pytest.skip("brak roadnetu")
    from tsc_marl.core.graph import RoadNetGraph

    graph = RoadNetGraph.from_roadnet(roadnet)
    src = O.movement_sources("cityflow_jinan3x4_4ph", 8)
    assert src.shape == (graph.n_nodes, 8)
    for i in range(graph.n_nodes):
        for j in src[i]:
            if j >= 0:
                assert j in graph.neighbors[i] and j != i
    # wnętrze siatki 3×4 ma sąsiadów z każdej strony, więc jakieś ruchy nie są brzegowe
    assert (src >= 0).any()


def fake(seed: int = 0) -> Loaded:
    """Zbiór, w którym `gauss1` to czysty + szum: głowa ma się nauczyć, że szum = σ↑.

    Czysty sygnał jest GŁADKI w czasie (błądzenie losowe o małym kroku), tak jak
    kolejka w prawdziwym ruchu: szum widać wtedy jako niespójność kolejnych odczytów
    w oknie. Stały poziom per węzeł NIE nadaje się na fiksturę — sieć zapamiętuje go
    z tożsamości węzła, trafia w cel bez patrzenia na odczyt i σ nie rośnie z szumem."""
    rng = np.random.default_rng(seed)
    t, n, n_mov, ph = 400, 3, 2, 2
    start = rng.uniform(1, 4, size=(1, n, 2 * n_mov))
    walk = np.cumsum(0.1 * rng.normal(size=(t, n, 2 * n_mov)), axis=0)
    xc = np.abs(start + walk).astype(np.float32)
    onehot = np.zeros((t, n, ph), np.float32)
    onehot[..., 0] = 1
    x_clean = np.concatenate([xc, onehot], -1)
    noisy = x_clean.copy()
    noisy[..., : 2 * n_mov] += rng.normal(0, 1.5, size=xc.shape).astype(np.float32)
    dead = np.zeros((t, n, n_mov), bool)
    return Loaded(
        x={"clean": x_clean, "gauss1": noisy},
        dead={"clean": dead, "gauss1": dead},
        x_clean=x_clean,
        y=np.zeros((t, n), np.int64),
        tie_margin=np.ones((t, n), np.float32),
        phase_time=np.full((t, n), 15, np.int32),
        episode=np.repeat(np.arange(2, dtype=np.int32), t // 2),
        edge_index=np.array([[0, 1], [1, 2]]),
        incidence=np.ones((n, ph, n_mov), bool),
    )


def test_glowa_podnosi_sigma_na_szumie():
    """Jedyny test łapiący odwrócony znak w NLL albo pomylone osie σ: po uczeniu
    σ na zaszumionym wejściu ma być wyraźnie większe niż na czystym."""
    data = fake()
    cfg = O.ObsConfig(
        epochs=60, train_scenarios=("clean", "gauss1"), batch=64, hidden=32, neighbors=False
    )
    fitted, history = O.fit(data, cfg)
    assert history[-1]["nll"] < history[0]["nll"]
    _, s_clean = fitted.predict(data, data.x["clean"])
    _, s_noisy = fitted.predict(data, data.x["gauss1"])
    assert s_noisy.mean() > 1.5 * s_clean.mean()


def test_cechy_trwalosci_widza_utkniecie_i_zero():
    """Detektor, który stoi w miejscu, daje „bez zmiany" = 1; zmieniający się — 0.
    Stałe zero liczy się podwójnie: bez zmiany ORAZ zero."""
    t, n = 6, 1
    x = np.zeros((t, n, 3), np.float32)
    x[:, 0, 0] = np.arange(t)  # cecha 0 zmienia się co krok
    x[:, 0, 1] = 5.0  # cecha 1 utknęła na 5
    # cecha 2 utknęła na 0
    win = O.window_index(np.zeros(t, dtype=int), 3)
    f = O.persistence_features(x, win, 3)
    assert f.shape == (t, n, 6)
    same, zero = f[-1, 0, :3], f[-1, 0, 3:]
    assert same.tolist() == [0.0, 1.0, 1.0]
    assert zero.tolist() == [0.0, 0.0, 1.0]


def test_trwalosc_zmienia_szerokosc_wejscia():
    x = np.ones((4, 2, 5), np.float32)
    win = O.window_index(np.zeros(4, dtype=int), 2)
    assert O.build_inputs(x, win, None).shape[-1] == 10
    assert O.build_inputs(x, win, None, n_persist=3).shape[-1] == 10 + 6
