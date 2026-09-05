"""Testy kontraktu grafu sieci drogowej.

Dlaczego akurat te: błąd w `graph.py` prawie nigdy nie wywala się głośno.
Przestawiona kolejność węzłów wysyła obserwację jednego skrzyżowania w wiersz
drugiego, odwrócony kierunek krawędzi zamienia rolami W_source i W_target, a
podwójna pętla własna przesuwa wagi attention — każdy z tych błędów daje wynik
"gorszy o kilka procent", nie do odróżnienia od "no i tyle wychodzi".

Testy strukturalne nie stawiają CityFlow (`from_roadnet` czyta sam plik), więc
chodzą w milisekundach; slow jest tylko test kolejności, bo potrzebuje `world`.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch
from conftest import load_intersections

from tsc_marl import paths
from tsc_marl.core.graph import RoadNetGraph
from tsc_marl.domain import ats

# Roadnety 8-fazowe — protokół pracy (`cityflow_jinan3x4`, `cityflow_hangzhou4x4`).
# Wariant `_4phase` ma IDENTYCZNĄ topologię (patrz test niżej), różni się tylko
# `lightphases`, więc struktura grafu jest wspólna dla obu.
ROADNETS_8PH = {
    "jinan3x4": paths.DATA / "raw_data" / "jinan_3x4" / "roadnet_3_4.json",
    "hangzhou4x4": paths.DATA / "raw_data" / "hangzhou_4x4_advanced" / "roadnet_4_4.json",
}

# Oczekiwana geometria siatki M×N: 4 rogi (2 drogi wjazdowe), brzegi (3), wnętrze (4).
EXPECTED = {
    "jinan3x4": {"n_nodes": 12, "n_edges": 34, "degrees": {2: 4, 3: 6, 4: 2}},
    "hangzhou4x4": {"n_nodes": 16, "n_edges": 48, "degrees": {2: 4, 3: 8, 4: 4}},
}


@pytest.fixture(scope="session", params=sorted(ROADNETS_8PH))
def net(request):
    """(nazwa sieci, graf) dla obu sieci protokołu."""
    path = ROADNETS_8PH[request.param]
    if not path.is_file():
        pytest.skip(f"brak roadnetu {path} (dane pobiera się osobno — patrz data/README.md)")
    return request.param, RoadNetGraph.from_roadnet(path)


def test_rozmiar_i_odfiltrowanie_wirtualnych(net):
    """Węzłem jest tylko STEROWANE skrzyżowanie. Brzegowe skrzyżowania roadnetu
    (`virtual: true`) to źródła i ujścia ruchu — nie mają sygnalizacji, więc nie
    są agentami, a drogi ich dotykające nie są krawędziami grafu."""
    name, g = net
    assert g.n_nodes == EXPECTED[name]["n_nodes"]
    assert g.n_edges == EXPECTED[name]["n_edges"]
    assert all(not nid.startswith("intersection_0_") for nid in g.node_idx2id)


def test_stopnie_odpowiadaja_geometrii_siatki(net):
    """Niezależne źródło prawdy: dla siatki M×N rozkład stopni wynika z kształtu,
    a nie z pliku. Jeśli to pęknie, filtracja wirtualnych węzłów jest zepsuta."""
    name, g = net
    assert dict(Counter(g.degrees)) == EXPECTED[name]["degrees"]
    assert all(len(g.neighbors[i]) == g.degrees[i] + 1 for i in range(g.n_nodes))


def test_bijekcja_indeksow(net):
    """`node_idx2id` i `node_id2idx` muszą być wzajemnie odwrotne — na tym stoi
    cały kontrakt kolejności."""
    _, g = net
    assert all(g.node_id2idx[g.node_idx2id[i]] == i for i in range(g.n_nodes))
    assert len(set(g.node_idx2id)) == g.n_nodes


def test_edge_index_ksztalt_typ_i_brak_petli(net):
    """Kontrakt dla warstwy GAT: [2, E], long, BEZ pętli własnych (dokłada je
    `add_self_loops` w `agent/colight.py` — patrz decyzja 3 w module)."""
    _, g = net
    assert g.edge_index.shape == (2, g.n_edges)
    assert g.edge_index.dtype == torch.long
    assert not (g.edge_index[0] == g.edge_index[1]).any()
    assert g.edge_index.min() >= 0 and g.edge_index.max() < g.n_nodes


def test_krawedzie_symetryczne(net):
    """Drogi w roadnecie są jednokierunkowe, ale występują parami, więc zbiór
    krawędzi wychodzi symetryczny. Gdy to pęknie, w sieci pojawiła się droga
    jednokierunkowa i trzeba ŚWIADOMIE zdecydować, co robi z nią attention."""
    _, g = net
    pairs = {(a, b) for a, b in g.edge_index.t().tolist()}
    assert all((b, a) in pairs for a, b in pairs)


def test_sasiedztwo_zgodne_z_libsignalem(net):
    """Sąsiedztwo wyliczamy sami z `edge_index` (jedno źródło prawdy: to, co
    zobaczy GAT). Ten test sprawdza, że nie rozjechało się z gotowym `node_list`
    LibSignala, czyli że nasza interpretacja kierunku krawędzi jest ta sama."""
    name, g = net
    from common.utils import build_index_intersection_map_cityflow as build

    raw = build(str(ROADNETS_8PH[name]))
    for i, incoming in enumerate(raw["node_list"]):
        assert g.neighbors[i] == tuple(sorted({i, *incoming}))


def test_topologia_niezalezna_od_liczby_faz(net):
    """Wariant `_4phase` różni się WYŁĄCZNIE `lightphases`. Uzasadnia użycie
    4-fazowego świata w teście kolejności niżej."""
    name, g = net
    four = RoadNetGraph.from_roadnet(str(ROADNETS_8PH[name]).replace(".json", "_4phase.json"))
    assert four.node_idx2id == g.node_idx2id
    assert torch.equal(four.edge_index, g.edge_index)


def test_jednorodnosc_faz_i_ruchow(net):
    """Wszystkie skrzyżowania mają tyle samo faz i tyle samo ruchów sterowanych.
    To jest ZAŁOŻENIE, na którym stoi parameter sharing (jedna wspólna głowa
    akcji, cecha nr k znaczy to samo wszędzie) — sprawdzane, nie zakładane.
    Faza 5 (SUMO, nieregularna sieć) je złamie i ma się wtedy wywalić głośno."""
    name, _ = net
    inters = load_intersections(ROADNETS_8PH[name])
    assert {len(inter.phases) for inter, _ in inters} == {8}  # 9 lightphases minus żółta
    assert {len(ats.controlled_movements(inter)) for inter, _ in inters} == {8}


def test_post_init_odrzuca_zlamany_kontrakt():
    """Kontrakt jest egzekwowany, nie tylko opisany — inaczej pierwszy błąd
    wyjdzie dopiero jako gorsze ATT po 200 epizodach."""
    ok = dict(
        n_nodes=2,
        node_idx2id=("a", "b"),
        node_id2idx={"a": 0, "b": 1},
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        degrees=(0, 1),
        neighbors=((0,), (0, 1)),
    )
    RoadNetGraph(**ok)  # sanity: poprawny graf przechodzi

    with pytest.raises(ValueError, match="pętlę własną"):
        RoadNetGraph(**{**ok, "edge_index": torch.tensor([[1], [1]], dtype=torch.long)})
    with pytest.raises(ValueError, match="poza zakres"):
        RoadNetGraph(**{**ok, "edge_index": torch.tensor([[0], [7]], dtype=torch.long)})
    with pytest.raises(ValueError, match="wzajemnie odwrotne"):
        RoadNetGraph(**{**ok, "node_id2idx": {"a": 1, "b": 0}})
    with pytest.raises(ValueError, match="long"):
        RoadNetGraph(**{**ok, "edge_index": torch.tensor([[0], [1]], dtype=torch.float)})


@pytest.mark.slow
def test_kontrakt_kolejnosci(advanced_colight_world):
    """NAJWAŻNIEJSZY test modułu. `world.intersections` ma inną kolejność niż
    `node_idx`; pomyłka tutaj nie wywala się, tylko po cichu miesza obserwacje
    z nagrodami między skrzyżowaniami."""
    world, _ = advanced_colight_world
    g = RoadNetGraph.from_registry()

    ordered = g.order_intersections(world)
    assert [inter.id for inter in ordered] == list(g.node_idx2id)
    assert {inter.id for inter in world.intersections} == set(g.node_idx2id)

    # Na roadnetach natywnych CityFlow `world.intersections` ma DZIŚ tę samą
    # kolejność co graf, więc powyższe przeszłoby też dla kodu, który bierze
    # `world.intersections` wprost. Prawdziwym testem jest więc kolekcja o
    # kolejności JAWNIE innej — tak zachowa się Faza 5, gdzie `cf_gt_virtual`
    # rozjeżdża oba filtry (patrz docstring `graph.py`).
    shuffled = list(reversed(world.intersections))
    assert [inter.id for inter in g.order(shuffled, key=lambda i: i.id)] == list(g.node_idx2id)

    with pytest.raises(KeyError, match="brak w kolekcji"):
        g.order(world.intersections[1:], key=lambda inter: inter.id)
