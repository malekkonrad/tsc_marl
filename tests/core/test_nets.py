"""Actor, critic and encoder networks.

A network almost never fails on a structural bug, it just learns worse. These
tests target such silent failures: layers hidden from the optimiser, an encoder
that depends on node numbering, time steps mixed in a batched graph, and a
mismatch between the rollout and update paths. No engine needed: the graph is
read from the roadnet file, observations are random.
"""

from __future__ import annotations

import math

import pytest
import torch
from agent.colight import MultiHeadAttModel
from torch_geometric.data import Batch, Data

from tsc_marl import paths
from tsc_marl.core.graph import RoadNetGraph
from tsc_marl.core.nets import Actor, ActorCritic, Critic, GATEncoder

ROADNET = paths.DATA / "raw_data" / "jinan_3x4" / "roadnet_3_4.json"

# ATS row width with 8 phases: 8 pressure + 8 demand + 8 one-hot.
OB_LENGTH = 24


@pytest.fixture(scope="module")
def graph():
    if not ROADNET.is_file():
        pytest.skip(f"missing roadnet {ROADNET} (data is fetched separately, see data/README.md)")
    return RoadNetGraph.from_roadnet(ROADNET)


def make(n_layers: int = 1, seed: int = 0) -> GATEncoder:
    torch.manual_seed(seed)
    return GATEncoder(OB_LENGTH, n_layers=n_layers).eval()


def observations(g, seed: int = 1) -> torch.Tensor:
    return torch.randn(g.n_nodes, OB_LENGTH, generator=torch.Generator().manual_seed(seed))


def test_output_shape(graph):
    """[N, ob_length] -> [N, out_dim]; `out_dim` is OUTPUT_DIM of the last used
    layer, not of the last entry in the config list."""
    enc = make()
    out = enc(observations(graph), graph.edge_index)
    assert out.shape == (graph.n_nodes, 128)
    assert enc.out_dim == 128


def test_gradient_reaches_every_parameter(graph):
    """Attention layers in a plain list would be missing from `.parameters()`:
    the optimiser would train the embedding only, with no error."""
    enc = make(n_layers=2)
    names = [n for n, _ in enc.named_parameters()]
    assert any("blocks.0" in n for n in names) and any("blocks.1" in n for n in names)

    enc(observations(graph), graph.edge_index).sum().backward()
    no_grad = [n for n, p in enc.named_parameters() if p.grad is None or not p.grad.any()]
    assert no_grad == []


def test_permutation_equivariance(graph):
    """With parameter sharing the encoder must not know which node it is:
    renumbering the intersections renumbers the output and nothing else."""
    enc = make()
    x, ei = observations(graph), graph.edge_index
    out = enc(x, ei)

    perm = torch.randperm(graph.n_nodes, generator=torch.Generator().manual_seed(7))
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(graph.n_nodes)  # old index -> new

    out_perm = enc(x[perm], inverse[ei])
    assert torch.allclose(out_perm, out[perm], atol=1e-5)


def test_receptive_field_grows_with_layers(graph):
    """One layer = 1 hop, two = 2 hops; this is why the critic has two layers."""
    center = max(range(graph.n_nodes), key=lambda i: len(graph.neighbors[i]))
    one_hop = set(graph.neighbors[center])
    two_hop = {j for i in one_hop for j in graph.neighbors[i]} - one_hop
    assert two_hop, "no node at distance 2, the test would be meaningless"
    far = min(two_hop)

    x, ei = observations(graph), graph.edge_index
    perturbed = x.clone()
    perturbed[far] += 5.0

    shallow = make(n_layers=1)
    assert torch.allclose(shallow(x, ei)[center], shallow(perturbed, ei)[center], atol=1e-6)

    deep = make(n_layers=2)
    assert not torch.allclose(deep(x, ei)[center], deep(perturbed, ei)[center], atol=1e-6)


def test_pyg_batch_does_not_mix_graphs(graph):
    """Time steps are batched as one block-diagonal graph: two samples computed
    together must equal the two computed separately."""
    enc = make()
    x1, x2 = observations(graph, seed=1), observations(graph, seed=2)
    separate = torch.cat([enc(x1, graph.edge_index), enc(x2, graph.edge_index)])

    batch = Batch.from_data_list([Data(x=x, edge_index=graph.edge_index) for x in (x1, x2)])
    assert torch.allclose(enc(batch.x, batch.edge_index), separate, atol=1e-5)


# ------------------------------------------------------------------ UW-attention


def test_without_log_rho_bit_identical_to_original(graph):
    """Without log_rho the encoder computes exactly what CoLight's original layers
    compute, with the same weights. Together with the next test this closes both
    sides: without log_rho nothing changed, with it exactly the right thing does.
    Two layers, so the loop over `self.blocks` is exercised."""
    torch.manual_seed(0)
    enc = GATEncoder(OB_LENGTH, n_layers=2).eval()
    x = observations(graph)

    # Reference: the same weights in LibSignal's ORIGINAL layer.
    h = enc.embedding_mlp(x, True)
    for block in enc.blocks:
        ref = MultiHeadAttModel(d=block.d, dv=block.dv, d_out=block.d_out, nv=block.nv, suffix=0)
        ref.load_state_dict(block.state_dict())  # strict=True: keys must match
        h = ref.eval()(h, graph.edge_index, True)

    assert torch.equal(enc(x, graph.edge_index), h)


def test_log_rho_reaches_the_layer_through_the_actor(graph):
    """log_rho must survive `Actor.forward` -> `GATEncoder.forward` -> layer.
    Dropping it on the way raises nothing; the policy runs without modulation."""
    torch.manual_seed(0)
    actor = Actor(OB_LENGTH, 8).eval()  # 1 hop, as in the core
    ei = graph.edge_index

    j, i = ei[:, 3].tolist()
    x = observations(graph)
    x_changed = x.clone()
    x_changed[j] = observations(graph, seed=2)[j]

    log_rho = torch.zeros(ei.shape[1] + graph.n_nodes)
    log_rho[3] = -1e9
    assert torch.equal(actor(x, ei, log_rho=log_rho)[i], actor(x_changed, ei, log_rho=log_rho)[i])

    # Control: without log_rho the same row DOES react to the neighbour (otherwise
    # the equality above would only show that i never looked at j).
    assert not torch.equal(actor(x, ei)[i], actor(x_changed, ei)[i])


def test_third_dimension_rejected(graph):
    """[B, N, F] would pass the embedding and produce garbage in `message()`,
    which reads the number of EDGES from it. Fail loudly instead."""
    enc = make()
    with pytest.raises(ValueError, match="Batch"):
        enc(observations(graph).unsqueeze(0), graph.edge_index)
    with pytest.raises(ValueError, match="ob_length"):
        enc(torch.randn(graph.n_nodes, OB_LENGTH + 1), graph.edge_index)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_layers": 3}, "n_layers=3"),  # lists in colight.yml have 2 entries
        ({"emb_dims": (64, 64)}, "embedding output"),
        ({"n_layers": 2, "output_dim": (64, 128)}, "do not chain"),
        ({"n_layers": 0}, "n_layers"),
    ],
)
def test_rejects_inconsistent_config(kwargs, match):
    """Mismatched per-layer sizes would otherwise fail later with a message that
    does not name the wrong YAML key."""
    with pytest.raises(ValueError, match=match):
        GATEncoder(OB_LENGTH, **kwargs)


# ------------------------------------------------------- ablation encoders


@pytest.mark.parametrize("kind", ["gat", "mean", "none"])
def test_encoder_variants_same_shape_and_gradient(graph, kind):
    """Same contract [N, ob_length] -> [N, 128] and a gradient to every parameter."""
    from tsc_marl.core.nets import build_encoder

    torch.manual_seed(0)
    enc = build_encoder(kind, OB_LENGTH)
    out = enc(observations(graph), graph.edge_index)
    assert out.shape == (graph.n_nodes, 128)
    out.sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())


@pytest.mark.parametrize("kind", ["gat", "mean", "none"])
def test_encoder_variants_permutation_equivariance(graph, kind):
    from tsc_marl.core.nets import build_encoder

    torch.manual_seed(0)
    enc = build_encoder(kind, OB_LENGTH).eval()
    x = observations(graph)
    perm = torch.randperm(graph.n_nodes, generator=torch.Generator().manual_seed(3))
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(graph.n_nodes)
    ei = inv[graph.edge_index]
    assert torch.allclose(enc(x, graph.edge_index)[perm], enc(x[perm], ei), atol=1e-5)


def test_none_ignores_neighbours_mean_does_not(graph):
    """What the ablation separates: `none` depends only on its own row, `mean`
    and `gat` change when a neighbour changes."""
    from tsc_marl.core.nets import build_encoder

    x = observations(graph)
    src, dst = graph.edge_index[:, 0]
    x2 = x.clone()
    x2[src] += 5.0
    for kind, expected in (("none", False), ("mean", True), ("gat", True)):
        torch.manual_seed(0)
        enc = build_encoder(kind, OB_LENGTH).eval()
        changed = not torch.allclose(enc(x, graph.edge_index)[dst], enc(x2, graph.edge_index)[dst])
        assert changed == expected, kind


@pytest.mark.parametrize("kind", ["gat", "mean", "none"])
def test_actor_works_with_every_encoder(graph, kind):
    """Regression: `Actor.forward` always passes `log_rho` to the encoder, so a
    variant without that argument fails on every call. The other variant tests
    call `build_encoder` directly and skip this path."""
    torch.manual_seed(0)
    actor = Actor(OB_LENGTH, 8, encoder=kind).eval()
    assert actor(observations(graph), graph.edge_index).shape == (graph.n_nodes, 8)


@pytest.mark.parametrize("kind", ["mean", "none"])
def test_variants_without_attention_reject_log_rho(graph, kind):
    """Refusal must be loud: silently dropping log_rho would let a run
    "UW-attention + mean" finish with plausible numbers."""
    torch.manual_seed(0)
    actor = Actor(OB_LENGTH, 8, encoder=kind).eval()
    log_rho = torch.zeros(graph.edge_index.shape[1] + graph.n_nodes)
    with pytest.raises(ValueError, match="attention"):
        actor(observations(graph), graph.edge_index, log_rho=log_rho)


@pytest.mark.parametrize("kind", ["gat", "mean", "none"])
def test_variants_survive_to_device(graph, kind):
    """Regression: a method named `_apply` overrode `nn.Module._apply`, which
    `.to(device)` uses; the other tests passed because they never moved the net."""
    ac = ActorCritic(OB_LENGTH, 8, encoder=kind).to("cpu").to(torch.float32)
    assert ac.actor(observations(graph), graph.edge_index).shape == (graph.n_nodes, 8)


def test_unknown_encoder():
    from tsc_marl.core.nets import build_encoder

    with pytest.raises(ValueError, match="encoder"):
        build_encoder("rocket", OB_LENGTH)


def test_actorcritic_variant_only_in_the_actor():
    """The critic stays GAT for every actor variant; GAT actor state_dict keys are
    unchanged, so existing checkpoints still load."""
    from tsc_marl.core.nets import LocalEncoder

    ac = ActorCritic(OB_LENGTH, 8, encoder="none")
    assert isinstance(ac.actor.encoder, LocalEncoder)
    assert isinstance(ac.critic.encoder, GATEncoder)
    gat = ActorCritic(OB_LENGTH, 8)
    plain = Actor(OB_LENGTH, 8)
    assert gat.actor.state_dict().keys() == plain.state_dict().keys()


@pytest.mark.slow
def test_from_registry_reads_the_baseline_config(advanced_colight_world):
    """A real run takes hyperparameters from `colight.yml`, not from the defaults,
    otherwise the comparison with Advanced-CoLight is not fair."""
    world, model = advanced_colight_world
    enc = GATEncoder.from_registry(model.ob_length)
    assert enc.n_layers == 1
    assert enc.ob_length == model.ob_length
    assert enc.out_dim == 128


# ---------------------------------------------------------------------- Actor


def make_actor(n_phases: int = 8, seed: int = 0, **kwargs) -> Actor:
    torch.manual_seed(seed)
    return Actor(OB_LENGTH, n_phases, **kwargs).eval()


def test_actor_shape_and_single_layer_head(graph):
    """Head = one `nn.Linear`, as `OUTPUT_LAYERS: []` in colight.yml."""
    actor = make_actor()
    logits = actor(observations(graph), graph.edge_index)
    assert logits.shape == (graph.n_nodes, 8)
    assert len([m for m in actor.head if isinstance(m, torch.nn.Linear)]) == 1
    assert len([m for m in make_actor(hidden=64).head if isinstance(m, torch.nn.Linear)]) == 2


def test_actor_starts_almost_uniform(graph):
    """gain 0.01 on the last layer: initial entropy close to log(n_phases)."""
    logits = make_actor()(observations(graph), graph.edge_index)
    assert logits.std() < 0.1, "initial logits too wide, gain 0.01 did not apply"

    entropy = torch.distributions.Categorical(logits=logits).entropy()
    assert torch.allclose(entropy, torch.full_like(entropy, math.log(8)), atol=0.01)


def test_actor_returns_logits_not_probabilities(graph):
    """A `softmax` in `forward` would make `Categorical(logits=...)` apply
    log-softmax twice: a flatter distribution and no error."""
    logits = make_actor()(observations(graph), graph.edge_index)
    assert not torch.allclose(logits.sum(-1), torch.ones(graph.n_nodes), atol=1e-3)
    assert (logits < 0).any(), "logits must be able to be negative"


def test_actor_gradient_reaches_encoder_and_head(graph):
    actor = make_actor(hidden=64)
    actor(observations(graph), graph.edge_index).sum().backward()
    no_grad = [n for n, p in actor.named_parameters() if p.grad is None or not p.grad.any()]
    assert no_grad == []
    assert any(n.startswith("encoder.blocks") for n, _ in actor.named_parameters())


def test_actor_permutation_equivariance(graph):
    """As for the encoder, checked at the end of the path: the node number must
    not affect its distribution over phases."""
    actor = make_actor()
    x, ei = observations(graph), graph.edge_index
    perm = torch.randperm(graph.n_nodes, generator=torch.Generator().manual_seed(7))
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(graph.n_nodes)
    assert torch.allclose(actor(x[perm], inverse[ei]), actor(x, ei)[perm], atol=1e-5)


def test_actor_rejects_a_single_phase():
    """A distribution over one phase is degenerate (zero entropy and gradient)."""
    with pytest.raises(ValueError, match="n_phases"):
        Actor(OB_LENGTH, 1)


@pytest.mark.slow
def test_actor_from_registry(advanced_colight_world):
    """The actor encoder reads the baseline's config (`GATEncoder.registry_kwargs`)."""
    _, model = advanced_colight_world
    actor = Actor.from_registry(model.ob_length, 8)
    assert actor.encoder.n_layers == 1
    assert actor.encoder.ob_length == model.ob_length
    assert actor.n_phases == 8


# --------------------------------------------------------------- Critic / A2C


def make_critic(seed: int = 0, **kwargs) -> Critic:
    torch.manual_seed(seed)
    return Critic(OB_LENGTH, **kwargs).eval()


def test_critic_returns_a_value_per_agent(graph):
    """Factorised critic: V_i per agent, not one scalar per step; 2 hops by default."""
    critic = make_critic()
    v = critic(observations(graph), graph.edge_index)
    assert v.shape == (graph.n_nodes,)
    assert critic.encoder.n_layers == 2


def test_critic_batch_does_not_mix_time_steps(graph):
    """The critic's most important test. With `h.mean(0)` instead of
    `global_mean_pool(h, batch)` a batched minibatch would average nodes from
    different moments: the critic sees the future and nothing reports it."""
    critic = make_critic()
    x1, x2 = observations(graph, seed=1), observations(graph, seed=2)
    separate = torch.cat([critic(x1, graph.edge_index), critic(x2, graph.edge_index)])

    batch = Batch.from_data_list([Data(x=x, edge_index=graph.edge_index) for x in (x1, x2)])
    together = critic(batch.x, batch.edge_index, batch.batch)
    assert torch.allclose(together, separate, atol=1e-5)


def test_critic_no_batch_means_one_graph(graph):
    """`batch=None` must equal all zeros, otherwise rollout (one graph) and update
    (Batch) would compute different things."""
    critic = make_critic()
    x, ei = observations(graph), graph.edge_index
    zeros = torch.zeros(graph.n_nodes, dtype=torch.long)
    assert torch.allclose(critic(x, ei), critic(x, ei, zeros))

    with pytest.raises(ValueError, match="batch"):
        critic(x, ei, torch.zeros(graph.n_nodes + 1, dtype=torch.long))


def test_critic_permutation_equivariance(graph):
    """Mean pooling is permutation invariant, so the critic is equivariant."""
    critic = make_critic()
    x, ei = observations(graph), graph.edge_index
    perm = torch.randperm(graph.n_nodes, generator=torch.Generator().manual_seed(7))
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(graph.n_nodes)
    assert torch.allclose(critic(x[perm], inverse[ei]), critic(x, ei)[perm], atol=1e-5)


def test_critic_gradient_reaches_everything(graph):
    critic = make_critic()
    critic(observations(graph), graph.edge_index).sum().backward()
    assert [n for n, p in critic.named_parameters() if p.grad is None or not p.grad.any()] == []


def make_ac(seed: int = 0, **kwargs) -> ActorCritic:
    torch.manual_seed(seed)
    return ActorCritic(OB_LENGTH, 8, **kwargs).eval()


def test_actorcritic_has_disjoint_weights():
    """Accidental sharing raises nothing and only hurts training, so it is tested."""
    ac = make_ac()
    assert not {id(p) for p in ac.actor.parameters()} & {id(p) for p in ac.critic.parameters()}
    assert ac.actor.encoder.n_layers == 1 and ac.critic.encoder.n_layers == 2


def test_evaluate_matches_act(graph):
    """In the first PPO epoch the ratio exp(log_prob - old_log_prob) must be
    exactly 1; otherwise `act` and `evaluate` disagree."""
    ac = make_ac()
    x, ei = observations(graph), graph.edge_index
    torch.manual_seed(3)
    actions, logp, values = ac.act(x, ei)

    logp2, entropy, values2 = ac.evaluate(x, ei, actions)
    assert torch.allclose(torch.exp(logp2 - logp), torch.ones_like(logp), atol=1e-6)
    assert torch.allclose(values, values2)
    assert entropy.shape == (graph.n_nodes,)
    assert torch.allclose(ac.value(x, ei), values)


def test_act_shapes_and_modes(graph):
    """`deterministic=True` is evaluation; training samples, which is where
    PPO's exploration comes from."""
    ac = make_ac()
    x, ei = observations(graph), graph.edge_index

    a1, logp, v = ac.act(x, ei, deterministic=True)
    a2, _, _ = ac.act(x, ei, deterministic=True)
    assert a1.shape == logp.shape == v.shape == (graph.n_nodes,)
    assert torch.equal(a1, a2)
    assert a1.max() < 8 and a1.min() >= 0

    torch.manual_seed(11)
    samples = torch.stack([ac.act(x, ei)[0] for _ in range(5)])
    assert not (samples == samples[0]).all(), "the policy does not sample, check Categorical"


def heterogeneous_mask(n_nodes: int, n_phases: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """(phase counts, mask) as on Krakow: intersections with 2 up to 8 phases."""
    counts = torch.tensor([2 + i % (n_phases - 1) for i in range(n_nodes)])
    return counts, torch.arange(n_phases)[None, :] < counts[:, None]


def test_phase_mask_removes_missing_phases(graph):
    """The policy must never pick a phase the intersection lacks, sampling or
    argmax, and -inf in the logits must not leak into log_prob, entropy or grads."""
    counts, mask = heterogeneous_mask(graph.n_nodes)
    torch.manual_seed(0)
    ac = ActorCritic(OB_LENGTH, 8, phase_mask=mask)
    x, ei = observations(graph), graph.edge_index

    logits = ac.actor(x, ei)
    assert torch.isneginf(logits[~mask]).all()
    assert torch.isfinite(logits[mask]).all()

    torch.manual_seed(3)
    for _ in range(50):
        actions, log_probs, _ = ac.act(x, ei)
        assert (actions < counts).all()
        assert torch.isfinite(log_probs).all()
    assert (ac.act(x, ei, deterministic=True)[0] < counts).all()

    log_probs, entropy, _ = ac.evaluate(x, ei, actions)
    assert torch.isfinite(log_probs).all() and torch.isfinite(entropy).all()
    (log_probs.sum() + entropy.sum()).backward()
    for name, p in ac.actor.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_phase_mask_in_minibatch_and_not_in_state_dict(graph):
    """A PPO minibatch stacks k time steps into k*N rows, so the mask is tiled k
    times; and it is not saved, so checkpoints without a mask still load."""
    from tsc_marl.core.ppo import tiled_edge_index

    _, mask = heterogeneous_mask(graph.n_nodes)
    k = 3
    torch.manual_seed(0)
    ac = ActorCritic(OB_LENGTH, 8, phase_mask=mask)
    x = torch.randn(k * graph.n_nodes, OB_LENGTH)
    logits = ac.actor(x, tiled_edge_index(graph.edge_index, graph.n_nodes, k))
    assert torch.equal(torch.isneginf(logits), ~mask.repeat(k, 1))

    assert not any("phase_mask" in key for key in ac.state_dict())
    ActorCritic(OB_LENGTH, 8).load_state_dict(ac.state_dict())  # and the other way round


@pytest.mark.slow
def test_actorcritic_from_registry(advanced_colight_world):
    """The critic does not take `n_layers` from the config, which holds the
    actor's value (`N_LAYERS: 1`)."""
    _, model = advanced_colight_world
    ac = ActorCritic.from_registry(model.ob_length, 8)
    assert ac.actor.encoder.n_layers == 1
    assert ac.critic.encoder.n_layers == 2
    assert Critic.from_registry(model.ob_length).encoder.n_layers == 2
