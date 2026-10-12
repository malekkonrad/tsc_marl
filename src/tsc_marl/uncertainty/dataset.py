"""Behaviour-cloning dataset (state -> expert action) for training the uncertainty heads.

Supervised learning on dumped pairs isolates the uncertainty mechanism from
reinforcement learning and runs on a CPU in minutes.

The expert is Advanced-MP: its decision is a function of exactly the information
in the ATS state (p(s) sums the EP block over a phase's movements, d(s) the
demand block), so the task is REALIZABLE on clean input and any error under
failure can be blamed on the failure, not on model capacity. The one exception
is the minimum-green latch, which the state does not carry; `phase_time` is
dumped next to the input and the latch frequency reported (`hold_min_green`).

The expert decides from the clean engine and the CORRUPTED observation is
stored: the right target ("what should have been done"). It comes for free,
because `advanced_mp.py` reads the engine and the failure layer lives only in
`StateBuilder`, so no scenario can change the trajectory.

All scenarios in one pass: one world, one expert, one `StateBuilder` per
scenario plus a clean one, so trajectories are identical across scenarios by
construction, and the whole dataset costs one run.

    tsc dataset -n cityflow_jinan3x4_4ph --episodes 3
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tsc_marl.domain.sensors import FailureSpec
from tsc_marl.paths import RUNS

EXPERIMENT = "bc"
CLEAN = "clean"

#: Scenario grid. The head trains on `gauss` 1.0/2.0 and is tested on 3.5/4.0 and
#: `urand`, i.e. on noise UNSEEN in training; otherwise an uncertainty estimator
#: could not be told apart from a detector calibrated on known levels.
DEFAULT_SCENARIOS: tuple[FailureSpec, ...] = (
    FailureSpec(),  # clean: reference and head training set
    FailureSpec(kind="gauss", level=1.0),
    FailureSpec(kind="gauss", level=2.0),
    FailureSpec(kind="gauss", level=3.5),
    FailureSpec(kind="gauss", level=4.0),
    FailureSpec(kind="urand", level=3.5),
    FailureSpec(kind="mask", level=0.25),
    FailureSpec(kind="mask", level=0.5),
    FailureSpec(kind="mask", level=0.25, unit="intersection"),
    FailureSpec(kind="drift", level=0.2),
    FailureSpec(kind="drift", level=0.3),
)


@dataclass
class Dataset:
    """Dump in GRAPH NODE order (`graph.py` contract), the order the core uses."""

    x_clean: np.ndarray  # [T, N, F] float32, observation without failure
    x: dict[str, np.ndarray]  # slug -> [T, N, F] float32
    dead: dict[str, np.ndarray]  # slug -> [T, N, n_mov] bool, movement with a dead detector
    y: np.ndarray  # [T, N] int64, phase chosen by the expert
    phase_time: np.ndarray  # [T, N] int32, NOT an input (see the module docstring)
    episode: np.ndarray  # [T] int32
    edge_index: np.ndarray  # [2, E] int64
    #: [N, n_phases, n_mov] bool: movements served by each phase (`ats.phase_incidence`);
    #: needed to recompute p(s) and d(s), i.e. to reproduce the expert from the state.
    incidence: np.ndarray
    #: [T, N] float32: p(s) of the best phase minus the second, from the CLEAN state.
    #: A measure of AMBIVALENCE: near zero two phases are equally good, and decision
    #: uncertainty has nothing to do with sensor failure; kept as a measured variable.
    tie_margin: np.ndarray
    meta: dict[str, Any]

    def save(self, path: Path) -> Path:
        arrays: dict[str, np.ndarray] = {
            "x_clean": self.x_clean,
            "y": self.y,
            "phase_time": self.phase_time,
            "episode": self.episode,
            "edge_index": self.edge_index,
            "incidence": self.incidence,
            "tie_margin": self.tie_margin,
        }
        for name, value in self.x.items():
            arrays[f"x__{name}"] = value
        for name, value in self.dead.items():
            arrays[f"dead__{name}"] = value
        np.savez_compressed(path, **arrays)
        return path


def collect(
    network: str = "cityflow_jinan3x4_4ph",
    scenarios: tuple[FailureSpec, ...] = DEFAULT_SCENARIOS,
    *,
    expert: str = "advancedmp",
    episodes: int = 3,
    decisions: int | None = None,
) -> Dataset:
    """One pass, all scenarios. -> `Dataset` with statistics in `meta`."""
    from tsc_marl import libsignal

    libsignal.bootstrap()  # `core.loop` pulls in `agent.colight` from the submodule

    from tsc_marl.core.features import StateBuilder
    from tsc_marl.core.graph import RoadNetGraph
    from tsc_marl.core.loop import LoopConfig, TrafficEnv, world_order
    from tsc_marl.libsignal import session

    # `graphic: True` is forced: rows must follow graph node order, and LibSignal
    # builds the graph only when asked (`advancedmp.yml` does not need one).
    patch = {"model": {"graphic": True}}
    with session.standalone_world(expert, network, config_patch=patch) as (world, agents, config):
        cfg = LoopConfig.from_registry()
        n_dec = decisions if decisions is not None else cfg.decisions
        t_min = config["model"].get("t_min", 0)

        graph = RoadNetGraph.from_registry()
        order = graph.order_intersections(world)
        to_world = world_order(graph, world)

        clean_builder = StateBuilder.from_registry(world, order, failure=None)
        builders = {
            s.slug(): StateBuilder.from_registry(world, order, failure=s) for s in scenarios
        }
        env = TrafficEnv(world, graph, clean_builder, cfg.action_interval)

        incidence = clean_builder.phase_incidence()
        acc = _Accumulator(builders, incidence)
        att: list[float] = []
        for ep in range(episodes):
            env.reset(ep)
            for agent in agents:
                agent.reset()
            for b in builders.values():
                b.reset_sensors(ep)
            acc.run_episode(agents, env, to_world, n_dec, ep, t_min)
            att.append(float(world.get_average_travel_time()))

        w1 = config["model"].get("w1")
        return acc.finish(graph, network, expert, scenarios, episodes, n_dec, att, t_min, w1)


class _Accumulator:
    """Collects rows episode by episode; the decision loop holds this module's only
    ordering pitfall."""

    def __init__(self, builders: dict[str, Any], incidence: np.ndarray) -> None:
        self.builders = builders
        self.incidence = incidence
        self.n_phases = builders[CLEAN].n_phases
        self.x_clean: list[np.ndarray] = []
        self.x: dict[str, list[np.ndarray]] = {name: [] for name in builders}
        self.dead: dict[str, list[np.ndarray]] = {name: [] for name in builders}
        self.y: list[np.ndarray] = []
        self.phase_time: list[np.ndarray] = []
        self.episode: list[np.ndarray] = []
        self.held = 0
        self.total = 0

    def run_episode(self, agents, env, to_world, n_dec, ep, t_min) -> None:
        n = len(to_world)
        for _ in range(n_dec):
            # 1. Observations, ALL from the same engine state, before the action.
            self.x_clean.append(self.builders[CLEAN].observe(corrupt=False))
            for name, b in self.builders.items():
                self.x[name].append(b.observe())
                self.dead[name].append(_dead_movements(b))

            # 2. The expert reads the engine, not our features: failure does not apply.
            y_world = np.array([a.get_action(None, None) for a in agents], dtype=np.int64)
            pt = np.array([a.inter_obj.current_phase_time for a in agents], dtype=np.int32)
            self.held += int((pt < t_min).sum())
            self.total += n

            # 3. Renumbering: agent rank k serves `world.intersections[k]`, tensor
            #    rows follow `node_idx`; `to_world[k] = node_idx`, so assign by index.
            y_node = np.empty(n, dtype=np.int64)
            pt_node = np.empty(n, dtype=np.int32)
            y_node[to_world] = y_world
            pt_node[to_world] = pt
            self.y.append(y_node)
            self.phase_time.append(pt_node)
            self.episode.append(np.full(1, ep, dtype=np.int32))

            env.step(y_node)

    def finish(self, graph, network, expert, scenarios, episodes, n_dec, att, t_min, w1) -> Dataset:
        x = {k: np.stack(v).astype(np.float32) for k, v in self.x.items()}
        x_clean = np.stack(self.x_clean).astype(np.float32)
        body = x_clean.shape[-1] - self.n_phases
        y = np.stack(self.y).astype(np.int64)
        check = expert_check(x_clean, y, self.incidence, w1, self.n_phases)
        meta = {
            "network": network,
            "expert": expert,
            "episodes": episodes,
            "decisions_per_episode": n_dec,
            "n_nodes": int(graph.n_nodes),
            "t_min": t_min,
            "w1": w1,
            # How often the minimum-green latch fired: the part of the expert's
            # decision NOT recoverable from the state (a ceiling, not a model flaw).
            "hold_min_green": self.held / max(self.total, 1),
            # the task is realizable from the state alone, see `expert_check`
            "agreement": check["agreement"],
            "agreement_decidable": check["agreement_decidable"],
            "tie_rate": check["tie_rate"],
            "boundary_rate": check["boundary_rate"],
            "att_per_episode": att,
            "scenarios": {s.slug(): s.as_dict() for s in scenarios},
            # Phase block excluded: the one-hot never changes and would dilute the share.
            "changed_cells": {
                name: float((np.abs(v[..., :body] - x_clean[..., :body]) > 1e-9).mean())
                for name, v in x.items()
            },
            "dead_movements": {name: float(np.stack(v).mean()) for name, v in self.dead.items()},
        }
        return Dataset(
            x_clean=x_clean,
            x=x,
            dead={k: np.stack(v) for k, v in self.dead.items()},
            y=y,
            tie_margin=check["margin"],
            phase_time=np.stack(self.phase_time).astype(np.int32),
            episode=np.concatenate(self.episode).astype(np.int32),
            edge_index=np.asarray(graph.edge_index, dtype=np.int64),
            incidence=self.incidence,
            meta=meta,
        )


def expert_check(
    x_clean: np.ndarray, y: np.ndarray, incidence: np.ndarray, w1: float, n_phases: int
) -> dict[str, Any]:
    """Recompute the Advanced-MP decision from the STATE ALONE and compare.

    A self-check of every dump: one number checks feature order (EP before demand,
    one-hot last), phase incidence, the `rank -> node_idx` renumbering and that
    labels come from the CLEAN state. A shifted column raises nothing and only
    lowers this number. Agreement is counted on DECIDABLE samples; the rest are
    ties decided by the last bit of a sum:
      * `tie`      two phases with equal p(s), `argmax` picks by order,
      * `boundary` d(a_cur) * W1 exactly equal to max p(s) under a strict condition.
    """
    eps = 1e-6
    x = x_clean.astype(np.float64)
    inc = incidence.astype(np.float64)
    n_mov = incidence.shape[-1]
    p_s = np.einsum("npm,tnm->tnp", inc, x[..., :n_mov])
    d_s = np.einsum("npm,tnm->tnp", inc, x[..., n_mov : 2 * n_mov])
    cur = x[..., -n_phases:].argmax(-1)

    top = np.sort(p_s, -1)
    margin = top[..., -1] - top[..., -2]
    d_cur = np.take_along_axis(d_s, cur[..., None], -1)[..., 0] * w1
    pred = np.where(d_cur > p_s.max(-1), cur, p_s.argmax(-1))

    tie = margin <= eps
    boundary = np.abs(d_cur - p_s.max(-1)) <= eps
    decidable = ~tie & ~boundary
    return {
        "agreement": float((pred == y).mean()),
        # NaN, not 1.0, when everything is a tie: "nothing to check" differs from "agrees"
        "agreement_decidable": (
            float((pred[decidable] == y[decidable]).mean()) if decidable.any() else float("nan")
        ),
        "tie_rate": float(tie.mean()),
        "boundary_rate": float(boundary.mean()),
        "margin": margin.astype(np.float32),
    }


def _dead_movements(builder) -> np.ndarray:
    """[N, n_mov]: whether a MOVEMENT is served by a dead detector (any of its in
    or out lanes, since e(l,m) looks both ways). Ground truth for localisation.
    Not the same as "the feature changed": a masked detector on an empty lane
    reports 0, the true value, and is still dead.
    """
    out = np.zeros((builder.n_nodes, builder.n_mov), dtype=bool)
    sensors = builder.sensors
    if sensors is None or not sensors.broken:
        return out
    broken = sensors.broken
    for i, struct in enumerate(builder.structs):
        for m, (ins, outs) in enumerate(struct.movements):
            out[i, m] = any(lane in broken for lane in (*ins, *outs))
    return out


def build(
    network: str = "cityflow_jinan3x4_4ph",
    *,
    expert: str = "advancedmp",
    episodes: int = 3,
    replace: bool = False,
) -> Path:
    """Dump under a registered `runlog` run, so the dataset has provenance."""
    from runlog import Experiment
    from tsc_marl.libsignal import experiment as exp_module

    data = collect(network, expert=expert, episodes=episodes)
    params = {k: data.meta[k] for k in ("network", "expert", "episodes", "t_min", "w1")}
    params["scenarios"] = sorted(data.meta["scenarios"])
    run_id = f"{expert}__{network}__ep{episodes}"

    exp = Experiment(EXPERIMENT, root=RUNS)
    with exp.start(run_id, params, repos=exp_module.repos(), replace=replace) as run:
        path = data.save(run.path / "dataset.npz")
        run.result(
            samples=int(data.y.size),
            hold_min_green=data.meta["hold_min_green"],
            agreement=data.meta["agreement"],
            agreement_decidable=data.meta["agreement_decidable"],
            tie_rate=data.meta["tie_rate"],
            boundary_rate=data.meta["boundary_rate"],
            att_per_episode=data.meta["att_per_episode"],
            changed_cells=data.meta["changed_cells"],
            dead_movements=data.meta["dead_movements"],
            dataset=str(path.relative_to(run.path)),
        )
    return run.path
