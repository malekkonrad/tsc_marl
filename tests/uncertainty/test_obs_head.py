"""Observation uncertainty head (`uncertainty.obs_head`).

Three things can silently invalidate the result: a time window reaching into the
previous episode (state from before `world.reset()`), a movement -> edge mapping
to the wrong neighbour (the per-edge signal rests on it), and AUROC computed
with a flipped label.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tsc_marl import paths
from tsc_marl.core.graph import RoadNetGraph
from tsc_marl.uncertainty import obs_head as O
from tsc_marl.uncertainty.prototype import Loaded

JINAN_4PH = paths.DATA / "raw_data" / "jinan_3x4" / "roadnet_3_4_4phase.json"


def test_window_does_not_reach_the_previous_episode():
    episode = np.array([0, 0, 0, 1, 1, 1])
    win = O.window_index(episode, 3)
    assert win[0].tolist() == [0, 0, 0]
    assert win[2].tolist() == [2, 1, 0]
    assert win[3].tolist() == [3, 3, 3], "the start of episode 1 must not see episode 0"
    assert win[5].tolist() == [5, 4, 3]


def test_neighbour_matrix_averages_upstream():
    # edges 0->2 and 1->2: node 2 averages 0 and 1; node 0 has nobody
    adj = O.neighbor_matrix(np.array([[0, 1], [2, 2]]), 3)
    assert adj[2].tolist() == [0.5, 0.5, 0.0]
    assert adj[0].tolist() == [0.0, 0.0, 0.0]


def test_inputs_window_and_neighbours():
    x = np.arange(2 * 2 * 1, dtype=np.float32).reshape(2, 2, 1)  # [T=2, N=2, F=1]
    win = O.window_index(np.array([0, 0]), 2)
    adj = O.neighbor_matrix(np.array([[0], [1]]), 2)  # 0 -> 1
    z = O.build_inputs(x, win, adj)
    assert z.shape == (2, 2, 3)
    # t=1, node 1: [x(1,1), x(0,1), neighbour mean = x(1,0)]
    assert z[1, 1].tolist() == [3.0, 1.0, 2.0]


def test_auroc_direction_and_ties():
    assert O.auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert O.auroc(np.array([0.0]), np.array([1.0])) == 0.0
    assert O.auroc(np.ones(5), np.ones(5)) == pytest.approx(0.5)


def test_edge_aggregates_only_movements_from_that_neighbour():
    u = np.array([[[1.0, 5.0, 2.0]]])  # [T=1, N=1, n_mov=3]
    sources = np.array([[7, 7, -1]])  # two movements from node 7, one from the border
    u_max, edges = O.edge_uncertainty(u, sources, "max")
    u_mean, _ = O.edge_uncertainty(u, sources, "mean")
    assert edges == [(0, 7)]
    assert u_max[0, 0] == 5.0 and u_mean[0, 0] == 3.0


def test_movement_sources_are_real_graph_neighbours():
    """Every movement from a non-border road points to an upstream graph neighbour;
    otherwise the wrong edge would be muted."""
    if not JINAN_4PH.is_file():
        pytest.skip("missing roadnet")
    graph = RoadNetGraph.from_roadnet(JINAN_4PH)
    src = O.movement_sources("cityflow_jinan3x4_4ph", 8)
    assert src.shape == (graph.n_nodes, 8)
    for i in range(graph.n_nodes):
        for j in src[i]:
            if j >= 0:
                assert j in graph.neighbors[i] and j != i
    # inner nodes of the 3x4 grid have neighbours on every side
    assert (src >= 0).any()


def fake(seed: int = 0) -> Loaded:
    """`gauss1` = clean + noise: the head must learn that noise raises sigma.

    The clean signal is SMOOTH in time (a small-step random walk), like a real
    queue, so noise shows as inconsistency within the window. A constant level
    per node would not do: the network would memorise it from node identity and
    sigma would not grow with noise."""
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


def test_head_raises_sigma_on_noise():
    """The only test catching a flipped NLL sign or swapped sigma axes: after
    training, sigma on noisy input must be clearly larger than on clean input."""
    data = fake()
    cfg = O.ObsConfig(
        epochs=60, train_scenarios=("clean", "gauss1"), batch=64, hidden=32, neighbors=False
    )
    fitted, history = O.fit(data, cfg)
    assert history[-1]["nll"] < history[0]["nll"]
    _, s_clean = fitted.predict(data, data.x["clean"])
    _, s_noisy = fitted.predict(data, data.x["gauss1"])
    assert s_noisy.mean() > 1.5 * s_clean.mean()


def test_persistence_features_see_stuck_and_zero():
    """A stuck detector gives "no change" = 1, a changing one 0; a constant zero
    counts twice: no change AND zero."""
    t, n = 6, 1
    x = np.zeros((t, n, 3), np.float32)
    x[:, 0, 0] = np.arange(t)  # feature 0 changes every step
    x[:, 0, 1] = 5.0  # feature 1 stuck at 5
    # feature 2 stuck at 0
    win = O.window_index(np.zeros(t, dtype=int), 3)
    f = O.persistence_features(x, win, 3)
    assert f.shape == (t, n, 6)
    same, zero = f[-1, 0, :3], f[-1, 0, 3:]
    assert same.tolist() == [0.0, 1.0, 1.0]
    assert zero.tolist() == [0.0, 0.0, 1.0]


def test_persistence_changes_the_input_width():
    x = np.ones((4, 2, 5), np.float32)
    win = O.window_index(np.zeros(4, dtype=int), 2)
    assert O.build_inputs(x, win, None).shape[-1] == 10
    assert O.build_inputs(x, win, None, n_persist=3).shape[-1] == 10 + 6


# ------------------------------------------------- saving and loading a trained head


def _small_fitted(seed: int = 0, neighbors: bool = False) -> O.Fitted:
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


def test_head_after_saving_computes_the_same(tmp_path):
    """The round trip is BIT-IDENTICAL. The artefact carries standardisation
    vectors next to the weights, and they set the UNIT of u: losing `y_std` would
    rescale u by a constant, leave AUROC unchanged (monotone) and break u_ref and
    tau. Hence the test compares sigma, not a metric."""
    data = fake()
    fitted = _small_fitted()
    path = fitted.save(tmp_path / "obs_head.pt", network="cityflow_jinan3x4_4ph")

    loaded = O.load_head(path, network="cityflow_jinan3x4_4ph")
    mu, sigma = fitted.predict(data, data.x["gauss1"])
    mu2, sigma2 = loaded.predict(data, data.x["gauss1"])
    assert np.array_equal(mu, mu2) and np.array_equal(sigma, sigma2)

    # `cfg` is saved too: `features()` builds the window from `cfg.window`
    assert loaded.cfg == fitted.cfg
    assert loaded.n_feat == fitted.n_feat


def test_head_from_another_network_is_rejected(tmp_path):
    """A JiNan head loaded on HangZhou is near chance: nothing fails, the
    evaluation measures noise."""
    path = _small_fitted().save(tmp_path / "obs_head.pt", network="cityflow_jinan3x4_4ph")

    with pytest.raises(ValueError, match="AUROC"):
        O.load_head(path, network="cityflow_hangzhou4x4_4ph")
    O.load_head(path)  # without `network`: a deliberate context-free load


def test_old_artefact_format_is_rejected(tmp_path):
    """A format version mismatch fails loudly instead of half-loading."""
    path = _small_fitted().save(tmp_path / "obs_head.pt", network="cityflow_jinan3x4_4ph")
    blob = torch.load(path, map_location="cpu", weights_only=True)
    blob["format"] = O.HEAD_FORMAT + 1
    torch.save(blob, path)

    with pytest.raises(ValueError, match="format"):
        O.load_head(path)


# ------------------------------------- reference level of u on a healthy network (u_ref)


def _small_ref(tmp_path, fitted: O.Fitted, **kwargs) -> tuple[O.EdgeReference, str]:
    """Reference on the fixture: 3 nodes, edges 0->1 and 1->2."""
    path = fitted.save(tmp_path / "obs_head.pt", network="fake")
    sources = np.array([[-1, -1], [0, 0], [1, 1]])  # node 1's movements from 0, node 2's from 1
    ref = O.edge_reference(
        fitted, fake(), sources, network="fake", head_sha=O.file_sha(path), **kwargs
    )
    return ref, str(path)


def test_u_ref_follows_edge_uncertainty_order_with_self_loops(tmp_path):
    """Shape and order contract: [E] in `edges` order = (TARGET, SOURCE), plus [N]
    self-loops; the permutation to `edge_index` happens in `log_rho` only."""
    ref, _ = _small_ref(tmp_path, _small_fitted())
    assert ref.edges == ((1, 0), (2, 1))
    assert ref.u_edge.shape == (2,) and ref.u_self.shape == (3,)
    assert (ref.u_edge > 0).all() and (ref.u_self > 0).all()


def test_u_ref_lies_above_typical_clean_u(tmp_path):
    """`mean + 2*sd` leaves clean steps BELOW the threshold. Checked on the episode
    TAIL, since u_ref uses the training part (otherwise a tautology)."""
    data, fitted = fake(), _small_fitted()
    ref, _ = _small_ref(tmp_path, fitted)
    sources = np.array([[-1, -1], [0, 0], [1, 1]])

    _, sigma = fitted.predict(data, data.x["clean"])
    u_mov = O.movement_uncertainty(sigma / fitted.y_std, 2)
    u_edge, _ = O.edge_uncertainty(u_mov, sources, "mean")
    tail = data.mask(fitted.cfg.split, train=False)
    assert (u_edge[tail] <= ref.u_edge).mean() > 0.9


def test_u_ref_grows_with_k(tmp_path):
    fitted = _small_fitted()
    small, _ = _small_ref(tmp_path, fitted, k=0.0)
    large, _ = _small_ref(tmp_path, fitted, k=2.0)
    assert (large.u_edge >= small.u_edge).all() and (large.u_self >= small.u_self).all()


def test_u_ref_from_another_head_is_rejected(tmp_path):
    """u is in units of THAT head (through `y_std`), so another head's u_ref gives
    plausible numbers on another scale."""
    ref, head = _small_ref(tmp_path, _small_fitted())
    file = ref.save(tmp_path / "u_ref.npz")

    loaded = O.load_reference(file, network="fake", head=head)
    assert loaded.edges == ref.edges
    assert np.allclose(loaded.u_edge, ref.u_edge) and loaded.k == ref.k

    other = _small_fitted(seed=1).save(tmp_path / "other.pt", network="fake")  # other weights
    assert O.file_sha(other) != O.file_sha(head)
    with pytest.raises(ValueError, match="hash"):
        O.load_reference(file, head=other)

    with pytest.raises(ValueError, match="u_ref was computed on"):
        O.load_reference(file, network="cityflow_jinan3x4_4ph")


# ------------------------------------------------------------------- u -> log_rho


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


def test_log_rho_lands_on_the_edge_with_the_right_target():
    """The key test of `log_rho`: `ref.edges` are (TARGET, SOURCE), `edge_index`
    (SOURCE, TARGET), the edge set is symmetric, so a swapped pair raises no
    `KeyError` and silently mutes the OPPOSITE edge. Graph 0<->1; u is raised only
    on 1->0 (pair (target 0, source 1)), which must land in the column with target 0."""
    ref = _ref([(0, 1), (1, 0)], [1.0, 1.0], [1.0, 1.0])
    edge_index = torch.tensor([[0, 1], [1, 0]])  # column 0: 0->1, column 1: 1->0
    u_edge = np.array([2.0, 1.0])  # excess ONLY on (target 0, source 1)

    lr = O.log_rho(u_edge, np.ones(2), ref, edge_index, tau=1.0)
    assert lr.tolist() == [0.0, -1.0, 0.0, 0.0]  # [col 0->1, col 1->0, loop 0, loop 1]


def test_log_rho_is_zero_when_u_does_not_exceed_the_reference():
    """`relu`: a quieter edge keeps its normal weight; zero log_rho is exactly the
    layer's no-op."""
    ref = _ref([(1, 0)], [1.0], [1.0, 1.0])
    ei = torch.tensor([[0], [1]])
    assert O.log_rho(np.array([1.0]), np.ones(2), ref, ei).abs().sum() == 0.0
    assert O.log_rho(np.array([0.1]), np.zeros(2), ref, ei).abs().sum() == 0.0


def test_log_rho_has_length_e_plus_n_with_loops_last():
    """Layer contract: self-loops LAST, in node order, as `add_self_loops` adds them."""
    ref = _ref([(1, 0), (2, 1)], [1.0, 1.0], [1.0, 1.0, 1.0])
    ei = torch.tensor([[0, 1], [1, 2]])
    lr = O.log_rho(np.array([1.0, 1.0]), np.array([1.0, 3.0, 1.0]), ref, ei, tau=2.0)
    assert lr.shape == (2 + 3,)
    assert lr[2:].tolist() == [0.0, -1.0, 0.0]  # excess 2 on node 1, tau = 2


def test_log_rho_scales_inversely_with_tau():
    ref = _ref([(1, 0)], [1.0], [1.0, 1.0])
    ei = torch.tensor([[0], [1]])
    u = np.array([2.0])
    small_tau = O.log_rho(u, np.ones(2), ref, ei, tau=0.5)
    large_tau = O.log_rho(u, np.ones(2), ref, ei, tau=1.0)
    assert small_tau[0] == pytest.approx(2 * large_tau[0])


def test_log_rho_rejects_an_edge_without_reference():
    """A graph edge without its own u (e.g. the only movement from j to i is an
    always-green right turn, dropped by `drop_always_green`) is an error, not zero."""
    ref = _ref([(1, 0)], [1.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="has no u"):
        O.log_rho(np.array([1.0]), np.ones(2), ref, torch.tensor([[0, 1], [1, 0]]))


def test_log_rho_on_a_real_graph_fits_the_layer():
    """Length and order match what `UncertaintyWeightedAttention` expects."""
    if not JINAN_4PH.is_file():
        pytest.skip("missing roadnet")
    graph = RoadNetGraph.from_roadnet(JINAN_4PH)
    src = O.movement_sources("cityflow_jinan3x4_4ph", 8)
    edges = sorted({(i, int(j)) for i in range(src.shape[0]) for j in src[i] if j >= 0})
    ref = _ref(edges, np.ones(len(edges)), np.ones(graph.n_nodes))

    u_edge = np.ones(len(edges))
    u_edge[7] = 5.0  # one corrupted edge
    lr = O.log_rho(u_edge, np.ones(graph.n_nodes), ref, graph.edge_index, tau=1.0)

    assert lr.shape == (graph.edge_index.shape[1] + graph.n_nodes,)
    (column,) = (lr < 0).nonzero().flatten().tolist()
    target, source = edges[7]
    assert graph.edge_index[:, column].tolist() == [source, target]


# ----------------------------------------------- the head in the control loop (online)

SOURCES = np.array([[-1, -1], [0, 0], [1, 1]])  # fixture: 0->1 and 1->2
EI = torch.tensor([[0, 1], [1, 2]])


def _online_and_offline(tmp_path, tau: float = 1.0, scen: str = "gauss1", neighbors: bool = False):
    """`gauss1`, not `clean`: on clean data u stays below the reference, log_rho is
    identically zero and "online == offline" would pass trivially."""
    data, fitted = fake(), _small_fitted(neighbors=neighbors)
    ref, _ = _small_ref(tmp_path, fitted)
    _, sigma = fitted.predict(data, data.x[scen])
    u_mov = O.movement_uncertainty(sigma / fitted.y_std, 2)
    u_edge, _ = O.edge_uncertainty(u_mov, SOURCES, "mean")

    def offline(t):
        return O.log_rho(u_edge[t], u_mov[t].mean(-1), ref, EI, tau=tau)

    return data.x[scen], O.OnlineUncertainty(fitted, ref, SOURCES, EI, tau=tau), offline, data


@pytest.mark.parametrize("neighbors", [False, True])
def test_online_computes_the_same_as_offline(tmp_path, neighbors):
    """The buffer must give EXACTLY the input `window_index` would build offline:
    window orientation (newest in row 0), warm-up clipping, order of neighbour and
    persistence features. `neighbors=True` separately, since that branch depends on
    the buffer height."""
    x, online, offline, data = _online_and_offline(tmp_path, neighbors=neighbors)
    moved = 0
    for t in np.flatnonzero(data.episode == 0)[:20]:  # from the FIRST step, with warm-up
        lr = online.log_rho(x[t])
        assert torch.allclose(lr, offline(t), atol=1e-6)
        moved += int((lr < 0).any())
    assert moved > 0, "log_rho was zero everywhere; the test compared nothing"


def test_online_restarts_the_episode_after_reset(tmp_path):
    """A buffer carried across `world.reset()` holds readings of other traffic."""
    x, online, offline, data = _online_and_offline(tmp_path)
    for t in np.flatnonzero(data.episode == 0)[:20]:
        online.log_rho(x[t])

    ep1 = np.flatnonzero(data.episode == 1)
    without_reset = online.log_rho(x[ep1[0]])
    online.reset()
    after_reset = online.log_rho(x[ep1[0]])

    assert torch.allclose(after_reset, offline(ep1[0]), atol=1e-6)
    assert not torch.allclose(without_reset, after_reset, atol=1e-6)


def test_online_rejects_an_observation_with_the_wrong_node_count(tmp_path):
    _, online, _, _ = _online_and_offline(tmp_path)
    with pytest.raises(ValueError, match="rows"):
        online.log_rho(np.zeros((5, 20), np.float32))


def test_group_spread_groups_by_target_with_the_loop():
    """`group_spread` groups like the layer's softmax: by target, self-loop included."""
    ei = torch.tensor([[0, 1], [1, 1]])  # both edges go into node 1
    lr = torch.tensor([-1.0, -3.0, 0.0, 0.0, 0.0])  # 2 edges + 3 loops
    # node 1's group: edges (-1, -3) and loop (0) -> spread 3; nodes 0 and 2: loops only
    assert O.group_spread(lr, ei, 3) == pytest.approx(1.0)


# ------------------------------------------- edge side: target vs source


def _zero_ref(side: str) -> O.EdgeReference:
    """Zero reference on the 0->1, 1->2 fixture: log_rho = -u/tau directly."""
    return O.EdgeReference(
        u_edge=np.zeros(2),
        u_self=np.zeros(3),
        edges=((1, 0), (2, 1)),
        network="fake",
        k=0.0,
        scenario="clean",
        head_sha="x",
        side=side,
    )


def test_source_aggregates_all_source_movements_with_the_same_edges():
    """`source` takes ALL movements of node j (border ones too), while the set and
    order of `edges` stay as with `target`."""
    u = np.array([[[1.0, 3.0], [10.0, 20.0], [0.0, 0.0]]])  # [T=1, N=3, n_mov=2]
    sources = np.array([[-1, -1], [0, 0], [1, 1]])  # 0->1, 1->2
    u_target, e_target = O.edge_uncertainty(u, sources, "mean", side="target")
    u_source, e_source = O.edge_uncertainty(u, sources, "mean", side="source")
    assert e_target == e_source == [(1, 0), (2, 1)]
    # edge 0->1: target = node 1's movements from node 0 (15), source = node 0's (2)
    assert u_target[0].tolist() == [15.0, 0.0]
    assert u_source[0].tolist() == [2.0, 15.0]


def test_dead_source_mutes_its_edge_only_on_the_source_side():
    """Node 0 has corrupted readings. The message on 0->1 is `hidden_embedding(x_0)`,
    so IT must be muted. On the target side u of 0->1 comes from node 1's healthy
    movements and stays zero; on the source side exactly that edge is muted. Node
    0's self-loop is the same in both variants."""
    u_mov = np.array([[4.0, 4.0], [0.0, 0.0], [0.0, 0.0]])  # [N, n_mov]
    sources = np.array([[-1, -1], [0, 0], [1, 1]])
    ei = torch.tensor([[0, 1], [1, 2]])  # (source, target): 0->1, 1->2
    result = {}
    for side in O.EDGE_SIDES:
        u_edge, _ = O.edge_uncertainty(u_mov[None], sources, "mean", side=side)
        result[side] = O.log_rho(u_edge[0], u_mov.mean(-1), _zero_ref(side), ei, tau=1.0)
    # layout: [0->1, 1->2, loop 0, loop 1, loop 2]
    assert result["target"].tolist() == [0.0, 0.0, -4.0, 0.0, 0.0]
    assert result["source"].tolist() == [-4.0, 0.0, -4.0, 0.0, 0.0]


def test_dead_intersection_on_the_target_side_is_invariant():
    """All of node 1's movements corrupted give the SAME penalty to every edge in
    its softmax group (0->1 and loop 1), which softmax ignores. On the source side
    node 1's group has a spread: healthy neighbour 0 stays, its own loop drops."""
    u_mov = np.array([[0.0, 0.0], [4.0, 4.0], [0.0, 0.0]])
    sources = np.array([[-1, -1], [0, 0], [1, 1]])
    ei = torch.tensor([[0, 1], [1, 2]])
    spread = {}
    for side in O.EDGE_SIDES:
        u_edge, _ = O.edge_uncertainty(u_mov[None], sources, "mean", side=side)
        lr = O.log_rho(u_edge[0], u_mov.mean(-1), _zero_ref(side), ei, tau=1.0)
        group_1 = lr[[0, 3]]  # edge 0->1 and loop 1: everything into 1
        spread[side] = float(group_1.max() - group_1.min())
    assert spread["target"] == 0.0
    assert spread["source"] == 4.0


def test_unknown_side_is_rejected():
    u = np.zeros((1, 3, 2))
    sources = np.array([[-1, -1], [0, 0], [1, 1]])
    with pytest.raises(ValueError, match="side"):
        O.edge_uncertainty(u, sources, "mean", side="src")
    with pytest.raises(ValueError, match="side"):
        O.reference_path("fake", "src")


def test_reference_keeps_its_side_through_saving(tmp_path):
    """The side is stored IN THE FILE; files from before the variant lack the field
    and load as `target`, the side they were computed on."""
    fitted = _small_fitted()
    ref, head = _small_ref(tmp_path, fitted, side="source")
    assert ref.side == "source"
    loaded = O.load_reference(ref.save(tmp_path / "u_ref.npz"), head=head)
    assert loaded.side == "source"
    assert np.array_equal(loaded.u_edge, ref.u_edge)

    blob = dict(np.load(tmp_path / "u_ref.npz"))
    del blob["side"]
    np.savez(tmp_path / "old.npz", **blob)
    assert O.load_reference(tmp_path / "old.npz").side == "target"


def test_source_reference_differs_from_target(tmp_path):
    """Same head, same edges, another u_ref."""
    fitted = _small_fitted()
    target, _ = _small_ref(tmp_path, fitted)
    source, _ = _small_ref(tmp_path, fitted, side="source")
    assert target.edges == source.edges and target.head_sha == source.head_sha
    assert np.array_equal(target.u_self, source.u_self)
    assert not np.allclose(target.u_edge, source.u_edge)


def test_target_reference_path_is_unchanged():
    """Already computed plans read the original file name."""
    assert O.reference_path("net").name == "u_ref__net.npz"
    assert O.reference_path("net", "source").name == "u_ref__net__source.npz"


def test_online_on_the_source_side_computes_the_same_as_offline(tmp_path):
    """`OnlineUncertainty` takes the side from the REFERENCE."""
    data, fitted = fake(), _small_fitted()
    ref, _ = _small_ref(tmp_path, fitted, side="source")
    _, sigma = fitted.predict(data, data.x["gauss1"])
    u_mov = O.movement_uncertainty(sigma / fitted.y_std, 2)
    u_edge, _ = O.edge_uncertainty(u_mov, SOURCES, "mean", side="source")
    online = O.OnlineUncertainty(fitted, ref, SOURCES, EI, tau=1.0)

    x = data.x["gauss1"]
    for t in np.flatnonzero(data.episode == 0)[:20]:
        offline = O.log_rho(u_edge[t], u_mov[t].mean(-1), ref, EI, tau=1.0)
        assert torch.allclose(online.log_rho(x[t]), offline, atol=1e-6)
