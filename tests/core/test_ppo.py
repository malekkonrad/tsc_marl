"""MAPPO learning loop.

PPO bugs are expensive to diagnose: nothing crashes, training just does not
converge, and the cause may be one sign in GAE or a wrong [T, N] -> [T*N]
flattening order. Hence GAE is compared with an INDEPENDENT implementation (an
explicit sum from the definition, not the same recursion), graph stacking with
`torch_geometric.Batch`, and a final test checks that the whole thing learns.
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
        pytest.skip(f"missing roadnet {ROADNET} (data is fetched separately, see data/README.md)")
    return RoadNetGraph.from_roadnet(ROADNET)


# ----------------------------------------------------------------------- GAE


def gae_reference(rewards, values, dones, last_value, gamma, lam):
    """Independent implementation FROM THE DEFINITION: A_t = sum_l (gamma*lam)^l delta_{t+l}.

    Deliberately an explicit sum, not a backward recursion: if both had the same
    bug, the test would be worthless."""
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
def test_gae_matches_the_definition(segment, lam):
    rewards, values, dones, last_value = segment
    adv, ret = compute_gae(rewards, values, dones, last_value, gamma=0.95, gae_lambda=lam)
    expected = gae_reference(rewards, values, dones, last_value, 0.95, lam)
    assert torch.allclose(adv, expected, atol=1e-5)
    assert torch.allclose(ret, adv + values)


def test_gae_does_not_leak_past_a_terminal_state(segment):
    """`done` must cut the bootstrap, otherwise the last step's advantage would
    contain the value of the next episode's first (empty network) state."""
    rewards, values, dones, last_value = segment
    dones = dones.clone()
    dones[5] = 1.0
    adv, _ = compute_gae(rewards, values, dones, last_value, 0.95, 0.95)
    assert torch.allclose(
        adv, gae_reference(rewards, values, dones, last_value, 0.95, 0.95), atol=1e-5
    )
    assert torch.allclose(adv[5], rewards[5] - values[5], atol=1e-6)


def test_gae_bootstraps_the_last_step(segment):
    """The episode is cut after 3600 s, not terminated, so the last step uses
    `last_value`; a zero would lower values at the end of the episode by V(s_T)."""
    rewards, values, dones, last_value = segment
    adv, _ = compute_gae(rewards, values, dones, last_value, 0.95, 0.0)
    assert torch.allclose(adv[-1], rewards[-1] + 0.95 * last_value - values[-1], atol=1e-6)


def test_gae_requires_matching_shapes(segment):
    rewards, values, dones, last_value = segment
    with pytest.raises(ValueError, match="same shape"):
        compute_gae(rewards, values[:-1], dones, last_value, 0.95, 0.95)


# ------------------------------------------------------------- graph stacking


@pytest.mark.parametrize("k", [1, 2, 5])
def test_graph_stacking_matches_pyg(graph, k):
    """`tiled_edge_index` is a shortcut for `Batch.from_data_list`, valid because
    the topology is static. Wrong offsets would give an agent at step t the
    neighbours of step t+1, silently."""
    n, ei = graph.n_nodes, graph.edge_index
    ref = Batch.from_data_list([Data(x=torch.zeros(n, 1), edge_index=ei) for _ in range(k)])
    assert torch.equal(tiled_edge_index(ei, n, k), ref.edge_index)
    assert torch.equal(batch_vector(n, k), ref.batch)


# ------------------------------------------------------------------- ValueNorm


def test_value_norm_tracks_true_statistics():
    vn = ValueNorm()
    g = torch.Generator().manual_seed(1)
    parts = [torch.randn(50, 7, generator=g) * 13 + 4 for _ in range(6)]
    for p in parts:
        vn.update(p)
    everything = torch.cat([p.reshape(-1) for p in parts])
    assert math.isclose(float(vn.mean), float(everything.mean()), rel_tol=1e-4)
    assert math.isclose(float(vn.var), float(everything.var(unbiased=False)), rel_tol=1e-3)


def test_value_norm_is_invertible():
    """The critic learns normalised values while GAE uses the raw scale; without
    exact inversion advantages would be in other units than rewards."""
    vn = ValueNorm()
    vn.update(torch.randn(100) * 5 - 2)
    x = torch.randn(20)
    assert torch.allclose(vn.denormalize(vn.normalize(x)), x, atol=1e-5)


def test_value_norm_is_saved_in_the_checkpoint():
    """Statistics are buffers, not parameters: in `state_dict`, never trained."""
    vn = ValueNorm()
    vn.update(torch.arange(10.0))
    assert set(vn.state_dict()) == {"mean", "var", "count"}
    assert list(vn.parameters()) == []


# ----------------------------------------------------------------------- buffer


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


def test_buffer_keeps_order_when_flattening(graph):
    """CONTRACT: [t, i] -> row t*N + i, as in `tiled_edge_index`. A mistake would
    pair one intersection's advantage with another's observation."""
    buf = make_buffer(graph)
    n = graph.n_nodes
    for t in range(4):
        marker = torch.arange(n, dtype=torch.float32) + 100 * t
        buf.add(
            obs=marker[:, None].repeat(1, OB_LENGTH),
            actions=torch.zeros(n, dtype=torch.long),
            log_probs=torch.zeros(n),
            values=torch.zeros(n),
            rewards=marker,
        )
    buf.compute_returns(torch.zeros(n), PPOConfig())
    (mb,) = list(buf.minibatches(1))
    steps = mb.x[:, 0].reshape(-1, n)
    for row in range(steps.shape[0]):
        assert torch.equal(steps[row] % 100, torch.arange(n, dtype=torch.float32))


def test_buffer_guards_capacity_and_call_order(graph):
    buf = make_buffer(graph, capacity=3)
    fill(buf, 3, graph.n_nodes)
    assert buf.full and len(buf) == 3
    with pytest.raises(IndexError, match="full"):
        fill(buf, 1, graph.n_nodes)
    with pytest.raises(RuntimeError, match="compute_returns"):
        list(buf.minibatches(2))
    buf.reset()
    with pytest.raises(RuntimeError, match="empty"):
        buf.compute_returns(torch.zeros(graph.n_nodes), PPOConfig())


def test_minibatches_cover_every_step_exactly_once(graph):
    """The split is over TIME STEPS, not agents (attention needs every node).
    Each step must land in exactly one minibatch."""
    buf = make_buffer(graph, capacity=7)
    fill(buf, 7, graph.n_nodes)
    buf.compute_returns(torch.zeros(graph.n_nodes), PPOConfig())

    seen = []
    for mb in buf.minibatches(3):
        k = mb.batch.max().item() + 1
        assert mb.x.shape == (k * graph.n_nodes, OB_LENGTH)
        assert mb.edge_index.shape == (2, k * graph.n_edges)
        seen += mb.x[:: graph.n_nodes, 0].tolist()
    assert sorted(seen) == list(range(7))


# ------------------------------------------------------------------- PPO update


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


def test_update_with_zero_lr_leaves_the_policy(graph):
    """With lr = 0 the ratio stays 1 through all epochs: KL = 0, nothing clipped.
    Otherwise `act` and `evaluate` compute different things."""
    ppo = make_ppo(graph, actor_lr=0.0, critic_lr=0.0, epochs=3)
    before = [p.clone() for p in ppo.ac.parameters()]
    stats = ppo.update(rollout(ppo, graph))

    assert stats["approx_kl"] == pytest.approx(0.0, abs=1e-7)
    assert stats["clip_frac"] == pytest.approx(0.0, abs=1e-7)
    assert all(torch.equal(a, b) for a, b in zip(before, ppo.ac.parameters(), strict=True))


def test_update_learns_and_reports_diagnostics(graph):
    ppo = make_ppo(graph, epochs=2)
    before = [p.clone() for p in ppo.ac.parameters()]
    stats = ppo.update(rollout(ppo, graph))

    assert set(stats) >= {"policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"}
    assert stats["n_minibatches"] == 2 * 4  # epochs x n_minibatches (8 steps / 4)
    assert any(not torch.equal(a, b) for a, b in zip(before, ppo.ac.parameters(), strict=True))
    assert 0.0 <= stats["entropy"] <= math.log(N_PHASES) + 1e-6


def test_update_stops_at_target_kl(graph):
    """Emergency brake: with an absurd lr the policy drifts in the first minibatch,
    and later epochs would learn from another policy's data."""
    ppo = make_ppo(graph, actor_lr=1.0, critic_lr=1.0, epochs=10, target_kl=1e-6)
    stats = ppo.update(rollout(ppo, graph))
    assert stats["early_stopped"] == 1.0
    assert stats["n_minibatches"] < 10 * 4


def test_config_rejects_nonsense_hyperparameters():
    for bad in ({"gamma": 1.5}, {"gae_lambda": -0.1}, {"epochs": 0}, {"clip_range": 0.0}):
        with pytest.raises(ValueError):
            PPOConfig(**bad)


@pytest.mark.slow
def test_ppo_actually_learns(graph):
    """End to end on a task with a known solution: reward 1 for one fixed phase,
    0 otherwise. The probability of that phase must rise well above 1/8 and the
    entropy must fall. The only test that catches SIGN errors (a flipped
    advantage or entropy bonus passes every shape test)."""
    target, steps = 3, 16
    ppo = make_ppo(graph, epochs=4, n_minibatches=2, actor_lr=3e-3, critic_lr=3e-3)
    obs = torch.randn(graph.n_nodes, OB_LENGTH, generator=torch.Generator().manual_seed(5))

    torch.manual_seed(0)
    for _ in range(30):
        buf = RolloutBuffer(steps, graph.n_nodes, OB_LENGTH, graph.edge_index)
        for _ in range(steps):
            actions, log_probs, values = ppo.act(obs, graph.edge_index)
            buf.add(obs, actions, log_probs, values, (actions == target).float())
        buf.compute_returns(ppo.value_of(obs, graph.edge_index), ppo.config)
        stats = ppo.update(buf)

    with torch.no_grad():
        probs = torch.softmax(ppo.ac.actor(obs, graph.edge_index), dim=-1)
    assert probs[:, target].mean() > 0.5, f"policy did not find phase {target}: {probs.mean(0)}"
    assert stats["entropy"] < math.log(N_PHASES) - 0.3
