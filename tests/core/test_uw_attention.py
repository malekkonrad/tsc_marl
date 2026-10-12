"""UW-attention layer (`core.uw_attention`)."""

import pytest
import torch
from agent.colight import MultiHeadAttModel

from tsc_marl import paths
from tsc_marl.core.graph import RoadNetGraph
from tsc_marl.core.uw_attention import UncertaintyWeightedAttention

ROADNET = paths.DATA / "raw_data" / "jinan_3x4" / "roadnet_3_4.json"
D, DV, D_OUT, NV = 128, 16, 128, 5


@pytest.fixture(scope="module")
def graph():
    if not ROADNET.is_file():
        pytest.skip(f"missing roadnet {ROADNET}")
    return RoadNetGraph.from_roadnet(ROADNET)


def observations(g, seed: int = 1):
    return torch.randn(g.n_nodes, D, generator=torch.Generator().manual_seed(seed))


def make_pair(seed: int = 0):
    torch.manual_seed(seed=seed)
    original = MultiHeadAttModel(d=D, dv=DV, d_out=D_OUT, nv=NV, suffix=0).eval()
    uw = UncertaintyWeightedAttention(d=D, dv=DV, d_out=D_OUT, nv=NV, suffix=0).eval()
    uw.load_state_dict(original.state_dict())

    return original, uw


def test_no_op(graph):
    """With log_rho = 0 the layer computes EXACTLY what the original does;
    otherwise no later difference could be attributed to the modulation."""
    orig, uw = make_pair()
    x = observations(graph)
    ei = graph.edge_index

    expected = orig(x, ei)
    assert torch.equal(uw(x, ei), expected)  # log_rho=None path

    n_edges = ei.shape[1] + graph.n_nodes  # E + N, after self-loops
    assert torch.equal(uw(x, ei, log_rho=torch.zeros(n_edges)), expected)


# Why -1e9, not -inf or -50: in float32 exp(-1e9 - max_i) is EXACTLY zero, so the
# edge contributes nothing and `torch.equal` works. -50 gives ~1e-22 (a weaker
# test); -inf gives NaN once ALL edges into i are suppressed (-inf - (-inf)).
SUPPRESS = -1e9


def test_suppression_cuts_the_neighbour_off(graph):
    """A very negative log_rho on j->i makes node i's output independent of x_j:
    the property the modification exists for (in the original exp(e_ij) >= 1, so a
    neighbour can only be diluted). The two control asserts make sure the test is
    not green for a row that never depended on x_j, or for an unchanged x_j."""
    _, uw = make_pair()
    ei = graph.edge_index
    n_edges = ei.shape[1] + graph.n_nodes

    # The edge is chosen by its POSITION in edge_index, which indexes log_rho.
    e = 3
    j, i = ei[:, e].tolist()
    assert [tuple(col) for col in ei.t().tolist()].count((j, i)) == 1, (
        "(j, i) repeats in edge_index; suppressing one column would not cut x_j off"
    )

    x = observations(graph)
    x_changed = x.clone()
    x_changed[j] = observations(graph, seed=2)[j]

    log_rho = torch.zeros(n_edges)
    log_rho[e] = SUPPRESS

    suppressed = uw(x, ei, log_rho=log_rho)
    suppressed_after = uw(x_changed, ei, log_rho=log_rho)
    assert torch.equal(suppressed[i], suppressed_after[i])

    # Control 1: x_changed really differs, so row j itself must move.
    assert not torch.equal(suppressed, suppressed_after)

    # Control 2: without suppression row i DOES react to x_j.
    zeros = torch.zeros(n_edges)
    assert not torch.equal(uw(x, ei, log_rho=zeros)[i], uw(x_changed, ei, log_rho=zeros)[i])


def test_invariance_to_a_common_constant(graph):
    """A constant added to log_rho of ALL edges into i leaves i's output unchanged:
    softmax sees only RELATIVE reliability, so "all neighbours equally unreliable"
    is indistinguishable from "all reliable" for this layer. `allclose`, because
    float32 arithmetic is not exactly invariant."""
    _, uw = make_pair()
    ei = graph.edge_index
    n_edges = ei.shape[1] + graph.n_nodes

    i = int(ei[1, 3])
    # i's softmax group: edges j->i AND the self-loop i->i at position E + i, which
    # is why the layer's contract is [E + N], not [E].
    incoming = torch.cat([(ei[1] == i).nonzero().flatten(), torch.tensor([ei.shape[1] + i])])

    x = observations(graph)
    reference = uw(x, ei, log_rho=torch.zeros(n_edges))

    log_rho = torch.zeros(n_edges)
    log_rho[incoming] = -3.7
    assert torch.allclose(uw(x, ei, log_rho=log_rho)[i], reference[i], atol=1e-6)

    # Control: a layer that IGNORED log_rho would also pass above. The constant on
    # all edges but the self-loop breaks the group, so row i must move.
    log_rho[ei.shape[1] + i] = 0.0
    assert not torch.allclose(uw(x, ei, log_rho=log_rho)[i], reference[i], atol=1e-6)


def test_state_dict_keys_match_the_original():
    """The layer adds no parameter (log_rho comes from outside), so trained
    weights load both ways without renaming."""
    orig, uw = make_pair()
    assert uw.state_dict().keys() == orig.state_dict().keys()
    orig.load_state_dict(uw.state_dict())  # strict=True, the other way too
