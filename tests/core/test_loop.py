"""Environment loop.

The loop reproduces LibSignal's training protocol, and any deviation gives
numbers that cannot be compared with the baselines rather than an error. The
tests target action permutation (one intersection's action executed on another
raises nothing), metrics matching LibSignal's `Metrics`, and evaluation that
never touches the critic. Tests with a real CityFlow are `slow` and use a short
episode: they check mechanics, not policy quality.
"""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tsc_marl import paths
from tsc_marl.core.graph import RoadNetGraph
from tsc_marl.core.loop import CoreTrainer, LoopConfig, TrafficEnv, world_order
from tsc_marl.core.nets import ActorCritic
from tsc_marl.core.ppo import PPO, PPOConfig

ROADNET = paths.DATA / "raw_data" / "jinan_3x4" / "roadnet_3_4.json"

# Short episode: 10 decisions instead of 240. Loop mechanics, not convergence.
SHORT = LoopConfig(steps=150, test_steps=150, action_interval=15, episodes=1)


@pytest.fixture(scope="module")
def graph():
    if not ROADNET.is_file():
        pytest.skip(f"missing roadnet {ROADNET} (data is fetched separately, see data/README.md)")
    return RoadNetGraph.from_roadnet(ROADNET)


# ------------------------------------------------------------------- fast


def test_loop_config_counts_decisions():
    """15 s per decision and a 3600 s episode give 240 decisions."""
    cfg = LoopConfig()
    assert cfg.decisions == 240 and cfg.test_decisions == 240


def test_world_order_permutes_actions(graph):
    """The most important fast test here. `World.step` indexes its OWN list of
    intersections, our action vector is in `node_idx` order. The fake world is
    in REVERSED order, because on the real CityFlow networks both orders happen
    to agree and a no-op permutation would pass."""
    fake = SimpleNamespace(
        intersections=[SimpleNamespace(id=nid) for nid in reversed(graph.node_idx2id)]
    )
    perm = world_order(graph, fake)
    actions = np.arange(graph.n_nodes)  # action of node i == i
    for_world = actions[perm]

    for position, inter in enumerate(fake.intersections):
        assert for_world[position] == graph.node_id2idx[inter.id]
    assert list(perm) == list(reversed(range(graph.n_nodes)))


def test_world_order_strips_the_sumo_prefix(graph):
    """`world_sumo` prefixes ids with `GS_`, the graph stores them without it."""
    fake = SimpleNamespace(
        intersections=[SimpleNamespace(id=f"GS_{nid}") for nid in graph.node_idx2id]
    )
    assert list(world_order(graph, fake)) == list(range(graph.n_nodes))


def test_checkpoint_restores_policy_and_value_scale(graph, tmp_path):
    """The checkpoint must include `ValueNorm`, otherwise a resumed run computes
    values on another scale than the saved critic."""
    torch.manual_seed(0)
    ob_length = 24
    stub_env = SimpleNamespace(graph=graph, features=SimpleNamespace(ob_length=ob_length))
    trainer = CoreTrainer(stub_env, PPO(ActorCritic(ob_length, 8), PPOConfig()), SHORT)
    trainer.ppo.value_norm.update(torch.randn(100) * 7 + 3)

    x = torch.randn(graph.n_nodes, ob_length)
    with torch.no_grad():
        before = trainer.ppo.ac.actor(x, graph.edge_index).clone()
    scale_before = (float(trainer.ppo.value_norm.mean), float(trainer.ppo.value_norm.var))
    path = tmp_path / "core.pt"
    trainer.save(path)

    with torch.no_grad():  # spoil weights and statistics so load has work to do
        for p in trainer.ppo.ac.actor.parameters():
            p.add_(1.0)
    trainer.ppo.value_norm.update(torch.full((1000,), 500.0))
    trainer.load(path)

    with torch.no_grad():
        assert torch.allclose(trainer.ppo.ac.actor(x, graph.edge_index), before)
    scale_after = (float(trainer.ppo.value_norm.mean), float(trainer.ppo.value_norm.var))
    assert scale_after == pytest.approx(scale_before)


# ------------------------------------------------- tail metrics, without CityFlow


def _metrics_stub(n_phases, legal, n_nodes=1, waiting=None):
    """`TrafficEnv` with stubbed collaborators, to test the tail ARITHMETIC, not
    the engine. `legal[i]` = number of phases node i really has. `waiting` maps
    lane -> waiting vehicles (default empty: no demand anywhere)."""
    waiting = defaultdict(int) if waiting is None else defaultdict(int, waiting)

    m = object.__new__(TrafficEnv)
    m.graph = SimpleNamespace(n_nodes=n_nodes)
    # One movement per phase, movement f served by phase f; lanes named `i<node>_<f>`
    # so that demand is controlled through `waiting`.
    incidence = np.zeros((n_nodes, n_phases, n_phases), dtype=bool)
    for i in range(n_nodes):
        for f in range(n_phases):
            incidence[i, f, f] = True
    m.features = SimpleNamespace(
        n_phases=n_phases,
        n_mov=n_phases,
        intersections=[SimpleNamespace(phases=list(range(k))) for k in legal],
        structs=[
            SimpleNamespace(
                reward_lanes=[],
                movements=[([f"i{i}_{f}"], []) for f in range(n_phases)],
            )
            for i in range(n_nodes)
        ],
        phase_incidence=lambda: incidence,
        rewards=lambda: np.zeros(n_nodes),
        observe=lambda: np.zeros((n_nodes, 1)),
    )
    m.world = SimpleNamespace(
        step=lambda _a: None,
        get_info=lambda _k: waiting,
        get_average_travel_time=lambda: 300.0,
        get_cur_throughput=lambda: 1000.0,
    )
    m.action_interval = 15
    m._to_world = np.arange(n_nodes)
    m._delay = [SimpleNamespace(generate=lambda: 0.0) for _ in range(n_nodes)]
    m._reset_stats()
    return m


def test_wait_max_measures_phase_starvation():
    """A node held in phase 0 starves the others: after k decisions the longest
    unserved phase has waited k x 15 s, which `att` and `queue` do not show."""
    m = _metrics_stub(n_phases=3, legal=[3])
    for _ in range(10):
        m.step(np.array([0]))
    assert m.metrics()["wait_max"] == pytest.approx(10 * 15)


def test_wait_max_ignores_missing_phases():
    """`n_phases` is the network maximum, so a 2-phase node has a dead third
    column that never gets green. Without the legal-phase mask `wait_max` would
    grow with episode length on every heterogeneous network. Here the node
    alternates its two phases, so the real starvation is one decision."""
    m = _metrics_stub(n_phases=3, legal=[2])
    for i in range(10):
        m.step(np.array([i % 2]))
    assert m.metrics()["wait_max"] == pytest.approx(15), "phase 2 does not exist"


def test_wait_max_demand_ignores_an_empty_phase():
    """The point of having two metrics: phases 1 and 2 are starved but nobody
    waits there. `wait_max` shows the gap, `wait_max_demand` stays silent."""
    m = _metrics_stub(n_phases=3, legal=[3])
    for _ in range(10):
        m.step(np.array([0]))
    result = m.metrics()
    assert result["wait_max"] == pytest.approx(10 * 15), "the raw gap must grow"
    assert result["wait_max_demand"] == 0.0, "an empty phase is not starvation"


def test_wait_max_demand_counts_a_phase_with_someone_waiting():
    """The same, but a vehicle waits on phase 1's movement: now it IS starvation."""
    m = _metrics_stub(n_phases=3, legal=[3], waiting={"i0_1": 4})
    for _ in range(10):
        m.step(np.array([0]))
    result = m.metrics()
    assert result["wait_max"] == pytest.approx(10 * 15)
    assert result["wait_max_demand"] == pytest.approx(10 * 15)


def test_wait_max_demand_ignores_missing_phases():
    """A dead column never gets green; with "demand" on it the metric would report
    growing starvation on a phase that does not exist. The node alternates its two
    phases while 9 vehicles wait on (non-existent) phase 2."""
    m = _metrics_stub(n_phases=3, legal=[2], waiting={"i0_2": 9})
    for i in range(10):
        m.step(np.array([i % 2]))
    assert m.metrics()["wait_max_demand"] == 0.0


def test_queue_tail_is_the_max_over_nodes_not_the_mean():
    """`queue` averages over the network, so one congested intersection among
    three empty ones disappears; `queue_max` shows it."""
    m = _metrics_stub(n_phases=2, legal=[2, 2, 2, 2], n_nodes=4)
    m._queue_sum = np.array([120.0, 0.0, 0.0, 0.0])
    m._delay_sum = np.zeros(4)
    m._decisions = 10
    result = m.metrics()
    assert result["queue"] == pytest.approx(3.0)  # 120 / (10 decisions x 4 nodes)
    assert result["queue_max"] == pytest.approx(12.0)  # 120 / 10 on one node
    assert result["queue_p90"] > result["queue"]


# ----------------------------------------------------------- with a real CityFlow


@pytest.fixture(scope="module")
def trainer(advanced_colight_world):
    world, _ = advanced_colight_world
    torch.manual_seed(0)
    return CoreTrainer.from_world(world, loop_config=SHORT)


@pytest.mark.slow
def test_training_episode_fills_the_buffer_and_gives_metrics(trainer):
    """End to end: 10 decisions x 15 engine steps, a full buffer, GAE computed,
    finite metrics; the queue reward is non-positive and ATT positive (the
    engine really ran)."""
    buffer, metrics = trainer.collect()

    assert len(buffer) == SHORT.decisions == 10
    assert buffer.obs.shape == (10, trainer.env.graph.n_nodes, trainer.env.features.ob_length)
    assert torch.isfinite(buffer.advantages).all() and torch.isfinite(buffer.returns).all()
    assert set(metrics) == {
        "att",
        "throughput",
        "reward",
        "queue",
        "delay",
        "queue_max",
        "queue_p90",
        "wait_max",
        "wait_max_demand",
    }
    assert metrics["queue_max"] >= metrics["queue_p90"] >= 0
    assert metrics["wait_max"] >= metrics["wait_max_demand"] >= 0, (
        "age with demand is a subset of the raw age"
    )
    assert all(np.isfinite(v) for v in metrics.values())
    assert metrics["att"] > 0 and metrics["queue"] >= 0
    assert metrics["reward"] <= 0, "the queue reward cannot be positive"


@pytest.mark.slow
def test_actions_reach_the_right_intersections(trainer):
    """Behavioural permutation test: every node gets a different phase and the
    world is read back. 15 s > 5 s of yellow, so the target phase is active
    after one decision."""
    env = trainer.env
    env.reset()
    n_phases = env.features.n_phases
    actions = np.array([(i % (n_phases - 1)) + 1 for i in range(env.graph.n_nodes)])  # never 0

    env.step(actions)

    for inter in env.world.intersections:
        node_idx = env.graph.node_id2idx[inter.id]
        assert inter.current_phase == actions[node_idx], (
            f"{inter.id}: world has phase {inter.current_phase}, "
            f"node {node_idx} was given {actions[node_idx]}"
        )


@pytest.mark.slow
def test_evaluation_never_calls_the_critic_and_is_repeatable(trainer):
    """Inference uses the actor only: the critic is replaced by one that raises.
    The same deterministic policy on a reset engine must give the same ATT,
    otherwise no before/after-failure comparison would make sense."""

    def explode(*args, **kwargs):
        raise AssertionError("critic called during evaluation")

    original = trainer.ppo.ac.critic.forward
    trainer.ppo.ac.critic.forward = explode
    try:
        first = trainer.evaluate()
        second = trainer.evaluate()
    finally:
        trainer.ppo.ac.critic.forward = original

    assert first["att"] == pytest.approx(second["att"], rel=1e-9)
    assert first["throughput"] == second["throughput"]


# ----------------------------------------------------------------- device


@pytest.mark.slow
def test_core_defaults_to_cpu(trainer):
    """Without an explicit `device`, network, edge index and ValueNorm are all on
    the CPU. A mismatch would surface only at the first update."""
    core = trainer

    assert core.device.type == "cpu"
    assert next(core.ppo.ac.parameters()).device == core.device
    assert core.edge_index.device == core.device
    assert core.ppo.value_norm.mean.device == core.device


@pytest.mark.slow
def test_edge_index_is_moved_once(trainer):
    """The trainer keeps its own copy of the static edge index on the device."""
    core = trainer

    assert core.edge_index.device == core.device
    assert torch.equal(core.edge_index, core.env.graph.edge_index)


@pytest.mark.slow
def test_numpy_boundary_works_both_ways(trainer):
    """`_obs` and `_actions` are the only device crossings; `.cpu()` in `_actions`
    is a no-op on CPU and required on CUDA."""
    core = trainer

    obs = core._obs(core.env.reset())
    assert obs.device == core.device

    actions = core.ppo.ac.actor(obs, core.edge_index).argmax(-1)
    numpy_actions = core._actions(actions)
    assert isinstance(numpy_actions, np.ndarray)
    assert numpy_actions.shape == (core.env.graph.n_nodes,)
