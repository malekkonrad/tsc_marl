"""Testy prototypu 2.2 — głowa obserwacyjna.

Trzy rzeczy, które potrafią cicho unieważnić wynik: okno czasowe sięgające do
poprzedniego epizodu (przeciek stanu sprzed `world.reset()`), mapowanie ruch →
krawędź na niewłaściwego sąsiada (per-edge claim stoi na tym jednym mapowaniu)
i AUROC liczony z odwróconą etykietą.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

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


# ------------------------------------------------- zapis i odczyt wytrenowanej głowy


def _fitted_maly(seed: int = 0, neighbors: bool = False) -> O.Fitted:
    cfg = O.ObsConfig(
        epochs=2,
        train_scenarios=("clean", "gauss1"),
        batch=64,
        hidden=16,
        neighbors=neighbors,
        seed=seed,
    )
    fitted, _ = O.fit(fake(), cfg)
    return fitted


def test_glowa_po_zapisie_liczy_to_samo(tmp_path):
    """Roundtrip ma być BIT W BIT, nie „w przybliżeniu".

    Artefakt niesie cztery wektory standaryzacji obok wag i to one, a nie sieć,
    decydują o JEDNOSTCE, w której wychodzi u. Zgubienie `y_std` przy zapisie
    przeskalowałoby u o stały czynnik — model dalej liczy, AUROC się nie zmienia
    (monotoniczna transformacja!), a u_ref i τ z Kroku 4 przestają pasować do
    liczb, na których je dobrano. Dlatego test porównuje σ, a nie metrykę.
    """
    data = fake()
    fitted = _fitted_maly()
    sciezka = fitted.save(tmp_path / "obs_head.pt", network="cityflow_jinan3x4_4ph")

    wczytana = O.load_head(sciezka, network="cityflow_jinan3x4_4ph")
    mu, sigma = fitted.predict(data, data.x["gauss1"])
    mu2, sigma2 = wczytana.predict(data, data.x["gauss1"])
    assert np.array_equal(mu, mu2) and np.array_equal(sigma, sigma2)

    # `cfg` też jedzie w pliku: `features()` buduje okno z `cfg.window`, więc
    # wczytanie z domyślnym oknem 4 zamiast 12 dałoby inne wejście i inne u.
    assert wczytana.cfg == fitted.cfg
    assert wczytana.n_feat == fitted.n_feat


def test_glowa_z_innej_sieci_odrzucona(tmp_path):
    """Najcichsza awaria Kroku 5: głowa z JiNan wczytana na HangZhou. AUROC 0,53,
    czyli log ρ z rzutu monetą — nic się nie wywala, a T2 mierzy szum."""
    sciezka = _fitted_maly().save(tmp_path / "obs_head.pt", network="cityflow_jinan3x4_4ph")

    with pytest.raises(ValueError, match="AUROC"):
        O.load_head(sciezka, network="cityflow_hangzhou4x4_4ph")
    O.load_head(sciezka)  # bez `network` — świadome wczytanie bez kontekstu sieci


def test_stary_format_artefaktu_odrzucony(tmp_path):
    """Wersja formatu ma polec głośno. Plik bez jednego pola wczytałby się w pół
    drogi albo dał u w innych jednostkach niż te, na których dobrano τ."""
    import torch

    sciezka = _fitted_maly().save(tmp_path / "obs_head.pt", network="cityflow_jinan3x4_4ph")
    blob = torch.load(sciezka, map_location="cpu", weights_only=True)
    blob["format"] = O.HEAD_FORMAT + 1
    torch.save(blob, sciezka)

    with pytest.raises(ValueError, match="format"):
        O.load_head(sciezka)


# ------------------------------------- poziom odniesienia u na zdrowej sieci (u_ref)


def _ref_maly(tmp_path, fitted: O.Fitted, **kwargs) -> tuple[O.EdgeReference, str]:
    """Odniesienie policzone na fiksturze: 3 węzły, krawędzie 0→1 i 1→2."""
    sciezka = fitted.save(tmp_path / "obs_head.pt", network="fake")
    sources = np.array([[-1, -1], [0, 0], [1, 1]])  # ruchy węzła 1 z węzła 0, węzła 2 z 1
    ref = O.edge_reference(
        fitted, fake(), sources, network="fake", head_sha=O.file_sha(sciezka), **kwargs
    )
    return ref, str(sciezka)


def test_u_ref_ma_kolejnosc_edge_uncertainty_i_petle_na_wezel(tmp_path):
    """Kontrakt kształtu i kolejności: [E] w kolejności `edges` = (CEL, ŹRÓDŁO),
    plus [N] na pętle własne. Przepięcie na `edge_index` (ŹRÓDŁO, CEL) to Krok 4
    i ma się wydarzyć w jednym miejscu — tutaj zostaje kolejność głowy."""
    ref, _ = _ref_maly(tmp_path, _fitted_maly())
    assert ref.edges == ((1, 0), (2, 1))
    assert ref.u_edge.shape == (2,) and ref.u_self.shape == (3,)
    assert (ref.u_edge > 0).all() and (ref.u_self > 0).all()


def test_u_ref_lezy_powyzej_typowego_u_na_czystym(tmp_path):
    """`mean + 2·sd` ma zostawiać czyste kroki POD progiem, inaczej modulacja
    rusza zdrową sieć. Sprawdzane na OGONIE epizodu, bo u_ref liczy się na
    części treningowej — inaczej ta własność byłaby tautologią."""
    data, fitted = fake(), _fitted_maly()
    ref, _ = _ref_maly(tmp_path, fitted)
    sources = np.array([[-1, -1], [0, 0], [1, 1]])

    _, sigma = fitted.predict(data, data.x["clean"])
    u_mov = O.movement_uncertainty(sigma / fitted.y_std, 2)
    u_edge, _ = O.edge_uncertainty(u_mov, sources, "mean")
    tail = data.mask(fitted.cfg.split, train=False)
    assert (u_edge[tail] <= ref.u_edge).mean() > 0.9


def test_u_ref_rosnie_z_k(tmp_path):
    """`k` to jedyne pokrętło tego artefaktu i ma działać w oczywistą stronę."""
    fitted = _fitted_maly()
    maly, _ = _ref_maly(tmp_path, fitted, k=0.0)
    duzy, _ = _ref_maly(tmp_path, fitted, k=2.0)
    assert (duzy.u_edge >= maly.u_edge).all() and (duzy.u_self >= maly.u_self).all()


def test_u_ref_z_innej_glowy_odrzucone(tmp_path):
    """Rozjazd u_ref ↔ wagi jest cichy: u wychodzi w jednostkach TEJ głowy
    (przez `y_std`), więc cudze u_ref to poprawnie wyglądające liczby o innej skali."""
    ref, sciezka = _ref_maly(tmp_path, _fitted_maly())
    plik = ref.save(tmp_path / "u_ref.npz")

    wczytane = O.load_reference(plik, network="fake", head=sciezka)
    assert wczytane.edges == ref.edges
    assert np.allclose(wczytane.u_edge, ref.u_edge) and wczytane.k == ref.k

    inna = _fitted_maly(seed=1).save(tmp_path / "inna.pt", network="fake")  # inne wagi
    assert O.file_sha(inna) != O.file_sha(sciezka)
    with pytest.raises(ValueError, match="odcisk"):
        O.load_reference(plik, head=inna)

    with pytest.raises(ValueError, match="u_ref policzone na"):
        O.load_reference(plik, network="cityflow_jinan3x4_4ph")


# ------------------------------------------------------------------- u -> log ρ


def _ref(edges, u_edge, u_self) -> O.EdgeReference:
    return O.EdgeReference(
        u_edge=np.array(u_edge, np.float64),
        u_self=np.array(u_self, np.float64),
        edges=tuple(edges),
        network="fake",
        k=2.0,
        scenario="clean",
        head_sha="0" * 16,
    )


def test_log_rho_trafia_w_krawedz_o_wlasciwym_CELU():
    """NAJWAŻNIEJSZY test Kroku 4. `ref.edges` to (CEL, ŹRÓDŁO), `edge_index` to
    (ŹRÓDŁO, CEL), a zbiór krawędzi jest symetryczny — więc pomyłka w kolejności
    pary nie rzuci `KeyError`, tylko po cichu wygasi krawędź PRZECIWNĄ.

    Graf: 0↔1 w obie strony. Podnosimy u wyłącznie na 1→0 (czyli na parze
    (cel 0, źródło 1)) i sprawdzamy, że log ρ wylądowało w tej kolumnie
    `edge_index`, której CELEM jest 0."""
    ref = _ref([(0, 1), (1, 0)], [1.0, 1.0], [1.0, 1.0])
    edge_index = torch.tensor([[0, 1], [1, 0]])  # kolumna 0: 0→1, kolumna 1: 1→0
    u_edge = np.array([2.0, 1.0])  # nadmiar TYLKO na (cel 0, źródło 1)

    lr = O.log_rho(u_edge, np.ones(2), ref, edge_index, tau=1.0)
    assert lr.tolist() == [0.0, -1.0, 0.0, 0.0]  # [kol. 0→1, kol. 1→0, pętla 0, pętla 1]


def test_log_rho_jest_zerem_gdy_u_nie_przekracza_odniesienia():
    """`relu`: krawędź spokojniejsza niż zwykle zostaje przy wadze normalnej.
    Zero log ρ to dokładnie no-op warstwy (`test_uw_attention.test_no_op`)."""
    ref = _ref([(1, 0)], [1.0], [1.0, 1.0])
    ei = torch.tensor([[0], [1]])
    assert O.log_rho(np.array([1.0]), np.ones(2), ref, ei).abs().sum() == 0.0
    assert O.log_rho(np.array([0.1]), np.zeros(2), ref, ei).abs().sum() == 0.0


def test_log_rho_ma_dlugosc_E_plus_N_i_petle_na_koncu():
    """Kontrakt warstwy: pętle własne NA KOŃCU, w kolejności węzłów — tak dokłada
    je `add_self_loops`."""
    ref = _ref([(1, 0), (2, 1)], [1.0, 1.0], [1.0, 1.0, 1.0])
    ei = torch.tensor([[0, 1], [1, 2]])
    lr = O.log_rho(np.array([1.0, 1.0]), np.array([1.0, 3.0, 1.0]), ref, ei, tau=2.0)
    assert lr.shape == (2 + 3,)
    assert lr[2:].tolist() == [0.0, -1.0, 0.0]  # nadmiar 2 na węźle 1, τ = 2


def test_log_rho_skaluje_sie_odwrotnie_do_tau():
    ref = _ref([(1, 0)], [1.0], [1.0, 1.0])
    ei = torch.tensor([[0], [1]])
    u = np.array([2.0])
    maly = O.log_rho(u, np.ones(2), ref, ei, tau=0.5)
    duzy = O.log_rho(u, np.ones(2), ref, ei, tau=1.0)
    assert maly[0] == pytest.approx(2 * duzy[0])


def test_log_rho_odrzuca_krawedz_bez_odniesienia():
    """Krawędź w grafie bez własnego u (np. jedyny ruch z j do i to zawsze zielony
    prawoskręt, wycięty przez `drop_always_green`) ma być błędem, nie zerem."""
    ref = _ref([(1, 0)], [1.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="nie ma swojego u"):
        O.log_rho(np.array([1.0]), np.ones(2), ref, torch.tensor([[0, 1], [1, 0]]))


def test_log_rho_na_prawdziwym_grafie_pasuje_do_warstwy():
    """Spięcie z drugim końcem: długość i kolejność muszą pasować do tego, czego
    `UncertaintyWeightedAttention` żąda po `add_self_loops`."""
    roadnet = paths.DATA / "raw_data" / "jinan_3x4" / "roadnet_3_4_4phase.json"
    if not roadnet.is_file():
        pytest.skip("brak roadnetu")
    from tsc_marl.core.graph import RoadNetGraph

    graph = RoadNetGraph.from_roadnet(roadnet)
    src = O.movement_sources("cityflow_jinan3x4_4ph", 8)
    edges = sorted({(i, int(j)) for i in range(src.shape[0]) for j in src[i] if j >= 0})
    ref = _ref(edges, np.ones(len(edges)), np.ones(graph.n_nodes))

    u_edge = np.ones(len(edges))
    u_edge[7] = 5.0  # jedna popsuta krawędź
    lr = O.log_rho(u_edge, np.ones(graph.n_nodes), ref, graph.edge_index, tau=1.0)

    assert lr.shape == (graph.edge_index.shape[1] + graph.n_nodes,)
    (kolumna,) = (lr < 0).nonzero().flatten().tolist()
    cel, zrodlo = edges[7]
    assert graph.edge_index[:, kolumna].tolist() == [zrodlo, cel]


# ----------------------------------------------- głowa w pętli sterowania (online)

SOURCES = np.array([[-1, -1], [0, 0], [1, 1]])  # fikstura: 0→1 i 1→2
EI = torch.tensor([[0, 1], [1, 2]])


def _online_i_offline(tmp_path, tau: float = 1.0, scen: str = "gauss1", neighbors: bool = False):
    """`gauss1`, nie `clean`: na czystym u nie przekracza odniesienia, więc log ρ
    jest tożsamościowo zerowe i każde porównanie „online == offline" przechodzi
    trywialnie, także dla błędnej implementacji okna."""
    data, fitted = fake(), _fitted_maly(neighbors=neighbors)
    ref, _ = _ref_maly(tmp_path, fitted)
    _, sigma = fitted.predict(data, data.x[scen])
    u_mov = O.movement_uncertainty(sigma / fitted.y_std, 2)
    u_edge, _ = O.edge_uncertainty(u_mov, SOURCES, "mean")

    def offline(t):
        return O.log_rho(u_edge[t], u_mov[t].mean(-1), ref, EI, tau=tau)

    return data.x[scen], O.OnlineUncertainty(fitted, ref, SOURCES, EI, tau=tau), offline, data


@pytest.mark.parametrize("neighbors", [False, True])
def test_online_liczy_to_samo_co_offline(tmp_path, neighbors):
    """NAJWAŻNIEJSZY test Kroku 5: bufor musi dawać DOKŁADNIE to wejście, które
    offline zbudowałby przez `window_index`.

    Pokryte za jednym zamachem: orientacja okna (najnowszy odczyt w wierszu 0, nie
    ostatnim), przycięcie na ROZGRZEWCE (pierwsze kroki epizodu mają krótszy bufor),
    kolejność cech sąsiadów i trwałości. `neighbors=True` MUSI tu być osobnym
    przypadkiem: gałąź sąsiadów liczy się w `build_inputs` po całym wymiarze czasu,
    więc jako jedyna zależy od wysokości bufora, a nie od pojedynczego kroku.

    Pomyłka w którymkolwiek z nich daje u o innym rozkładzie niż ten, na którym
    policzone jest `u_ref` — czyli mechanizm wygaszający sąsiadów z przypadkowych
    powodów, bez jednego wyjątku po drodze.
    """
    x, online, offline, data = _online_i_offline(tmp_path, neighbors=neighbors)
    ruszone = 0
    for t in np.flatnonzero(data.episode == 0)[:20]:  # od PIERWSZEGO kroku — z rozgrzewką
        lr = online.log_rho(x[t])
        assert torch.allclose(lr, offline(t), atol=1e-6)
        ruszone += int((lr < 0).any())
    assert ruszone > 0, "log ρ było wszędzie zerem — test nie porównałby niczego"


def test_online_po_resecie_zaczyna_epizod_od_nowa(tmp_path):
    """Bufor przeniesiony przez `world.reset()` to odczyty sprzed resetu w oknie —
    inny stan ruchu, a offline `window_index` nigdy tak nie liczy."""
    x, online, offline, data = _online_i_offline(tmp_path)
    for t in np.flatnonzero(data.episode == 0)[:20]:
        online.log_rho(x[t])

    ep1 = np.flatnonzero(data.episode == 1)
    bez_resetu = online.log_rho(x[ep1[0]])
    online.reset()
    po_resecie = online.log_rho(x[ep1[0]])

    assert torch.allclose(po_resecie, offline(ep1[0]), atol=1e-6)
    assert not torch.allclose(bez_resetu, po_resecie, atol=1e-6)


def test_online_odrzuca_obserwacje_o_zlej_liczbie_wezlow(tmp_path):
    _, online, _, _ = _online_i_offline(tmp_path)
    with pytest.raises(ValueError, match="wierszy"):
        online.log_rho(np.zeros((5, 20), np.float32))


def test_rozstep_w_grupie_liczy_sie_po_CELU_i_z_petla():
    """`group_spread` ma grupować tak jak softmax w warstwie: po celu, z pętlą własną.
    Grupowanie po źródle dałoby liczbę, której attention nigdy nie widzi."""
    ei = torch.tensor([[0, 1], [1, 1]])  # obie krawędzie wchodzą do węzła 1
    lr = torch.tensor([-1.0, -3.0, 0.0, 0.0, 0.0])  # 2 krawędzie + 3 pętle
    # grupa węzła 1: krawędzie (−1, −3) i pętla (0) -> rozstęp 3; węzły 0 i 2: same pętle
    assert O.group_spread(lr, ei, 3) == pytest.approx(1.0)
