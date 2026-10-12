"""Training a policy WITH the UW mechanism.

This file exists for ONE bug: stacking log_rho in a minibatch. `tiled_log_rho`
could return a vector of the RIGHT LENGTH and the wrong layout; then nothing
fails, PyG passes the argument to `message` unchecked, and training learns
another signal than it seems. So the key test compares VALUES: k steps through
the encoder one by one against one pass over the stacked graph.
"""

from __future__ import annotations

import pytest
import torch

from tsc_marl import paths
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

ROADNET = paths.DATA / "raw_data" / "jinan_3x4" / "roadnet_3_4.json"
OB_LENGTH = 24
N_PHASES = 8


@pytest.fixture(scope="module")
def graph():
    if not ROADNET.is_file():
        pytest.skip(f"missing roadnet {ROADNET} (data is fetched separately, see data/README.md)")
    return RoadNetGraph.from_roadnet(ROADNET)


def random_log_rho(n_steps: int, n_edges: int, n_nodes: int, seed: int = 0) -> torch.Tensor:
    """Clearly NEGATIVE and DIFFERENT at every position: zeros or a constant are
    invisible to softmax, so a scrambled layout would pass with them."""
    g = torch.Generator().manual_seed(seed)
    return -torch.rand(n_steps, n_edges + n_nodes, generator=g) * 5.0


# ------------------------------------------------------- log_rho stacking


def test_tiled_log_rho_matches_per_step_forward(graph):
    """k steps stacked into one graph give exactly what k separate forward passes
    give, also when every step has a DIFFERENT log_rho."""
    torch.manual_seed(0)
    enc = GATEncoder(OB_LENGTH).eval()
    n, e, k = graph.n_nodes, graph.edge_index.shape[1], 4

    g = torch.Generator().manual_seed(1)
    x = torch.randn(k, n, OB_LENGTH, generator=g)
    lr = random_log_rho(k, e, n)

    with torch.no_grad():
        separate = torch.cat([enc(x[t], graph.edge_index, log_rho=lr[t]) for t in range(k)])
        together = enc(
            x.reshape(k * n, OB_LENGTH),
            tiled_edge_index(graph.edge_index, n, k),
            log_rho=tiled_log_rho(lr, e, n),
        )
    assert torch.allclose(separate, together, atol=1e-5)


def test_naive_flattening_has_the_right_length_and_the_wrong_result(graph):
    """Proof the test above has power: the tempting `log_rho.reshape(-1)` passes
    every shape check and computes something else."""
    torch.manual_seed(0)
    enc = GATEncoder(OB_LENGTH).eval()
    n, e, k = graph.n_nodes, graph.edge_index.shape[1], 4
    lr = random_log_rho(k, e, n)

    naive = lr.reshape(-1)
    right = tiled_log_rho(lr, e, n)
    assert naive.shape == right.shape  # which is why nothing catches it by itself
    assert not torch.equal(naive, right)

    g = torch.Generator().manual_seed(1)
    x = torch.randn(k * n, OB_LENGTH, generator=g)
    ei = tiled_edge_index(graph.edge_index, n, k)
    with torch.no_grad():
        assert not torch.allclose(enc(x, ei, log_rho=naive), enc(x, ei, log_rho=right), atol=1e-5)


def test_tiled_log_rho_rejects_a_bad_shape(graph):
    n, e = graph.n_nodes, graph.edge_index.shape[1]
    with pytest.raises(ValueError, match="E\\+N"):
        tiled_log_rho(torch.zeros(3, e), e, n)  # no self-loops
    with pytest.raises(ValueError, match="E\\+N"):
        tiled_log_rho(torch.zeros(e + n), e, n)  # one dimension instead of [k, E+N]


# --------------------------------------------------------------- rollout buffer


def make_buffer(graph, k: int, with_log_rho: bool) -> RolloutBuffer:
    return RolloutBuffer(k, graph.n_nodes, OB_LENGTH, graph.edge_index, with_log_rho=with_log_rho)


def fill(buf: RolloutBuffer, lr: torch.Tensor | None, seed: int = 2) -> None:
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


def test_buffer_requires_consistency_with_with_log_rho(graph):
    """A silent mistake: a run meant to have the mechanism would end with numbers
    indistinguishable from the control arm."""
    n, e = graph.n_nodes, graph.edge_index.shape[1]
    with pytest.raises(ValueError, match="with_log_rho"):
        make_buffer(graph, 2, True).add(
            torch.zeros(n, OB_LENGTH),
            torch.zeros(n, dtype=torch.long),
            torch.zeros(n),
            torch.zeros(n),
            torch.zeros(n),
        )
    with pytest.raises(ValueError, match="with_log_rho"):
        make_buffer(graph, 2, False).add(
            torch.zeros(n, OB_LENGTH),
            torch.zeros(n, dtype=torch.long),
            torch.zeros(n),
            torch.zeros(n),
            torch.zeros(n),
            log_rho=torch.zeros(e + n),
        )
    with pytest.raises(ValueError, match="E \\+ N"):
        make_buffer(graph, 2, True).add(
            torch.zeros(n, OB_LENGTH),
            torch.zeros(n, dtype=torch.long),
            torch.zeros(n),
            torch.zeros(n),
            torch.zeros(n),
            log_rho=torch.zeros(e),  # no self-loops
        )


def test_minibatch_carries_log_rho_in_stacked_graph_order(graph):
    """log_rho follows the same step permutation as the observations; otherwise
    step t would get edge reliability from another moment of the episode."""
    n, e, k = graph.n_nodes, graph.edge_index.shape[1], 6
    lr = random_log_rho(k, e, n)
    buf = make_buffer(graph, k, True)
    fill(buf, lr)
    buf.compute_returns(torch.zeros(n), PPOConfig())

    seen = 0
    for mb in buf.minibatches(2, torch.Generator().manual_seed(0)):
        m = mb.x.shape[0] // n  # steps in this minibatch
        assert mb.log_rho is not None
        assert mb.log_rho.shape == (m * (e + n),)
        # recover the steps of the minibatch from the observations...
        steps = [
            next(t for t in range(k) if torch.equal(buf.obs[t], mb.x[i * n : (i + 1) * n]))
            for i in range(m)
        ]
        # ...and require log_rho to be built from EXACTLY those steps
        assert torch.equal(mb.log_rho, tiled_log_rho(lr[steps], e, n))
        seen += m
    assert seen == k


def test_buffer_without_mechanism_carries_no_log_rho(graph):
    """None, not zeros: `log_rho=None` takes a DIFFERENT path in the layer, the
    only one bit-identical to the plain core."""
    buf = make_buffer(graph, 3, False)
    fill(buf, None)
    buf.compute_returns(torch.zeros(graph.n_nodes), PPOConfig())
    assert buf.log_rho is None
    assert all(mb.log_rho is None for mb in buf.minibatches(2))


# --------------------------------------------------------------- ActorCritic + PPO


def test_act_and_evaluate_see_the_same_distribution(graph):
    """The first-epoch ratio is 1 also with the mechanism; log_rho in only one
    path would compute the gradient of another network than the one that acted."""
    torch.manual_seed(0)
    ac = ActorCritic(OB_LENGTH, N_PHASES).eval()
    n, e = graph.n_nodes, graph.edge_index.shape[1]
    x = torch.randn(n, OB_LENGTH, generator=torch.Generator().manual_seed(3))
    lr = random_log_rho(1, e, n)[0]

    with torch.no_grad():
        actions, log_prob, _ = ac.act(x, graph.edge_index, deterministic=True, log_rho=lr)
        again, _, _ = ac.evaluate(x, graph.edge_index, actions, log_rho=lr)
    assert torch.allclose(torch.exp(again - log_prob), torch.ones(n), atol=1e-6)


def test_log_rho_really_changes_the_policy(graph):
    """Otherwise all other tests would pass with the argument lost on the way."""
    torch.manual_seed(0)
    ac = ActorCritic(OB_LENGTH, N_PHASES).eval()
    n, e = graph.n_nodes, graph.edge_index.shape[1]
    x = torch.randn(n, OB_LENGTH, generator=torch.Generator().manual_seed(4))
    lr = random_log_rho(1, e, n)[0]

    with torch.no_grad():
        without = ac.distribution(x, graph.edge_index).logits
        with_lr = ac.distribution(x, graph.edge_index, log_rho=lr).logits
        zeros = ac.distribution(x, graph.edge_index, log_rho=torch.zeros(e + n)).logits
    assert not torch.allclose(without, with_lr, atol=1e-5)
    assert torch.equal(without, zeros), "log_rho = 0 must be a no-op"


def test_critic_never_sees_log_rho(graph):
    """The critic exists only in training; giving it reliability would use
    information the actor lacks at deployment."""
    torch.manual_seed(0)
    ac = ActorCritic(OB_LENGTH, N_PHASES).eval()
    n, e = graph.n_nodes, graph.edge_index.shape[1]
    x = torch.randn(n, OB_LENGTH, generator=torch.Generator().manual_seed(5))
    lr = random_log_rho(1, e, n)[0]

    with torch.no_grad():
        _, _, v_without = ac.act(x, graph.edge_index, deterministic=True)
        _, _, v_with = ac.act(x, graph.edge_index, deterministic=True, log_rho=lr)
    assert torch.equal(v_without, v_with)


def test_update_with_the_mechanism_learns(graph):
    """End to end: a buffer with log_rho goes through a full update and moves the
    policy (no broken gradient on the way)."""
    torch.manual_seed(0)
    n, e, k = graph.n_nodes, graph.edge_index.shape[1], 8
    ppo = PPO(ActorCritic(OB_LENGTH, N_PHASES), PPOConfig(epochs=2, n_minibatches=2))
    lr = random_log_rho(k, e, n)
    buf = make_buffer(graph, k, True)

    g = torch.Generator().manual_seed(7)
    x0 = torch.randn(n, OB_LENGTH, generator=g)
    before = ppo.ac.actor(x0, graph.edge_index, log_rho=lr[0]).detach().clone()
    fill(buf, lr)
    buf.compute_returns(torch.zeros(n), ppo.config)

    stats = ppo.update(buf, torch.Generator().manual_seed(0))
    assert stats["n_minibatches"] > 0
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())
    after = ppo.ac.actor(x0, graph.edge_index, log_rho=lr[0]).detach()
    assert not torch.allclose(before, after), "the update did not move the policy"


def test_batch_vector_and_log_rho_describe_the_same_graphs(graph):
    """`batch_vector` (critic) and `tiled_log_rho` (actor) number steps the same way."""
    n, e, k = graph.n_nodes, graph.edge_index.shape[1], 3
    assert batch_vector(n, k).shape == (k * n,)
    assert tiled_log_rho(torch.zeros(k, e + n), e, n).shape == (k * (e + n),)
