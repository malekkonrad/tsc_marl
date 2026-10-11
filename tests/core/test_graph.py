"""Road-network graph contract.

Why these tests: a bug in `graph.py` almost never fails loudly. A permuted node
order sends one intersection's observation into another's row, a flipped edge
direction swaps the roles of query and key in attention, and a double self-loop
shifts the attention weights. Each gives a result "a few percent worse",
indistinguishable from "that is just what it gets".

Structural tests do not start CityFlow (`from_roadnet` reads the file only), so
they run in milliseconds; only the order test is slow, because it needs a `world`.
"""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from conftest import load_intersections

from tsc_marl import paths
from tsc_marl.core.graph import RoadNetGraph
from tsc_marl.domain import ats

# 8-phase roadnets. The `_4phase` variants have IDENTICAL topology (test below)
# and differ only in `lightphases`, so the graph structure is shared.
ROADNETS_8PH = {
    "jinan3x4": paths.DATA / "raw_data" / "jinan_3x4" / "roadnet_3_4.json",
    "hangzhou4x4": paths.DATA / "raw_data" / "hangzhou_4x4_advanced" / "roadnet_4_4.json",
}

# Expected geometry of an MxN grid: 4 corners (2 incoming roads), border (3), inner (4).
EXPECTED = {
    "jinan3x4": {"n_nodes": 12, "n_edges": 34, "degrees": {2: 4, 3: 6, 4: 2}},
    "hangzhou4x4": {"n_nodes": 16, "n_edges": 48, "degrees": {2: 4, 3: 8, 4: 4}},
}


@pytest.fixture(scope="session", params=sorted(ROADNETS_8PH))
def net(request):
    """(network name, graph) for both protocol networks."""
    path = ROADNETS_8PH[request.param]
    if not path.is_file():
        pytest.skip(f"missing roadnet {path} (data is fetched separately, see data/README.md)")
    return request.param, RoadNetGraph.from_roadnet(path)


def test_size_and_virtual_intersections_filtered(net):
    """Only a SIGNALISED intersection is a node. Border intersections
    (`virtual: true`) are traffic sources and sinks without signals, so they are
    not agents, and roads touching them are not edges."""
    name, g = net
    assert g.n_nodes == EXPECTED[name]["n_nodes"]
    assert g.n_edges == EXPECTED[name]["n_edges"]
    assert all(not nid.startswith("intersection_0_") for nid in g.node_idx2id)


def test_degrees_match_grid_geometry(net):
    """Independent source of truth: for an MxN grid the degree distribution
    follows from the shape, not from the file. If this breaks, filtering of
    virtual nodes is broken."""
    name, g = net
    assert dict(Counter(g.degrees)) == EXPECTED[name]["degrees"]
    assert all(len(g.neighbors[i]) == g.degrees[i] + 1 for i in range(g.n_nodes))


def test_index_maps_are_a_bijection(net):
    """`node_idx2id` and `node_id2idx` must be mutually inverse; the whole order
    contract rests on it."""
    _, g = net
    assert all(g.node_id2idx[g.node_idx2id[i]] == i for i in range(g.n_nodes))
    assert len(set(g.node_idx2id)) == g.n_nodes


def test_edge_index_shape_dtype_and_no_self_loops(net):
    """Contract for the GAT layer: [2, E], long, NO self-loops (added by
    `add_self_loops` in `agent/colight.py`, decision 3 in the module)."""
    _, g = net
    assert g.edge_index.shape == (2, g.n_edges)
    assert g.edge_index.dtype == torch.long
    assert not (g.edge_index[0] == g.edge_index[1]).any()
    assert g.edge_index.min() >= 0 and g.edge_index.max() < g.n_nodes


def test_grid_edges_are_symmetric(net):
    """Roads are one-way but come in pairs on the grids, so the edge set is
    symmetric. On irregular networks (Krakow) it is not, and edge direction
    decides who hears whom (decision 2)."""
    _, g = net
    pairs = {(a, b) for a, b in g.edge_index.t().tolist()}
    assert all((b, a) in pairs for a, b in pairs)


def test_neighbourhoods_match_libsignal(net):
    """Neighbourhoods are derived from `edge_index` (one source of truth: what
    GAT sees). This checks they agree with LibSignal's own `node_list`, i.e.
    that we read edge direction the same way."""
    name, g = net
    from common.utils import build_index_intersection_map_cityflow as build

    raw = build(str(ROADNETS_8PH[name]))
    for i, incoming in enumerate(raw["node_list"]):
        assert g.neighbors[i] == tuple(sorted({i, *incoming}))


def test_topology_independent_of_phase_count(net):
    """The `_4phase` variant differs ONLY in `lightphases`. Justifies using the
    4-phase world in the order test below."""
    name, g = net
    four = RoadNetGraph.from_roadnet(str(ROADNETS_8PH[name]).replace(".json", "_4phase.json"))
    assert four.node_idx2id == g.node_idx2id
    assert torch.equal(four.edge_index, g.edge_index)


def test_homogeneous_phases_and_movements(net):
    """Every intersection has the same number of phases and controlled
    movements. Parameter sharing rests on this ASSUMPTION (one shared action
    head, feature k means the same everywhere), so it is checked, not assumed."""
    name, _ = net
    inters = load_intersections(ROADNETS_8PH[name])
    assert {len(inter.phases) for inter, _ in inters} == {8}  # 9 lightphases minus yellow
    assert {len(ats.controlled_movements(inter)) for inter, _ in inters} == {8}


def test_post_init_rejects_broken_contract():
    """The contract is enforced, not only described, otherwise the first bug
    shows up as a worse ATT after 200 episodes."""
    ok = dict(
        n_nodes=2,
        node_idx2id=("a", "b"),
        node_id2idx={"a": 0, "b": 1},
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        degrees=(0, 1),
        neighbors=((0,), (0, 1)),
    )
    RoadNetGraph(**ok)  # sanity: a valid graph passes

    with pytest.raises(ValueError, match="self-loop"):
        RoadNetGraph(**{**ok, "edge_index": torch.tensor([[1], [1]], dtype=torch.long)})
    with pytest.raises(ValueError, match="outside the node range"):
        RoadNetGraph(**{**ok, "edge_index": torch.tensor([[0], [7]], dtype=torch.long)})
    with pytest.raises(ValueError, match="mutually inverse"):
        RoadNetGraph(**{**ok, "node_id2idx": {"a": 1, "b": 0}})
    with pytest.raises(ValueError, match="long"):
        RoadNetGraph(**{**ok, "edge_index": torch.tensor([[0], [1]], dtype=torch.float)})


@pytest.mark.slow
def test_order_contract(advanced_colight_world):
    """`world.intersections` is ordered independently of `node_idx`; a mistake
    here raises nothing and silently mixes observations and rewards between
    intersections."""
    world, _ = advanced_colight_world
    g = RoadNetGraph.from_registry()

    ordered = g.order_intersections(world)
    assert [inter.id for inter in ordered] == list(g.node_idx2id)
    assert {inter.id for inter in world.intersections} == set(g.node_idx2id)

    # On CityFlow roadnets `world.intersections` happens to be in graph order, so
    # the check above would also pass for code that uses it directly. The real
    # test is a collection in a deliberately DIFFERENT order.
    shuffled = list(reversed(world.intersections))
    assert [inter.id for inter in g.order(shuffled, key=lambda i: i.id)] == list(g.node_idx2id)

    with pytest.raises(KeyError, match="missing from collection"):
        g.order(world.intersections[1:], key=lambda inter: inter.id)


def test_bare_id_follows_libsignal_rule():
    """The same rule as `build_index_intersection_map_sumo`: graph and world agree."""
    assert RoadNetGraph.bare_id("GS_207432959") == "207432959"
    assert RoadNetGraph.bare_id("joinedS_1_cluster_2") == "joinedS_1_cluster_2"
    assert RoadNetGraph.bare_id("intersection_1_1") == "intersection_1_1"  # CityFlow: identity


def test_order_intersections_strips_sumo_prefix():
    """On SUMO the graph strips `GS_` (signals guessed by netconvert) and the
    world keeps it. Without the same rule in `order_intersections` the core
    fails with a KeyError on Krakow while building features."""
    path = paths.DATA / "raw_data" / "jinan_3x4" / "roadnet_3_4.json"
    if not path.is_file():
        pytest.skip(f"missing roadnet {path}")
    g = RoadNetGraph.from_roadnet(path)
    world = SimpleNamespace(
        intersections=[SimpleNamespace(id=f"GS_{nid}") for nid in reversed(g.node_idx2id)]
    )
    ordered = g.order_intersections(world)
    assert [g.bare_id(inter.id) for inter in ordered] == list(g.node_idx2id)
