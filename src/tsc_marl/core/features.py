"""Observation and reward of the whole network.

Shared by the core, Advanced-CoLight and the max-pressure controllers; do not
duplicate this logic elsewhere.

State variants, simplest first (per movement):

    queue     mean queue of the movement's in lanes
    pressure  sum q(in) - sum q(out)          (Varaiya, 2013)
    ep        mean q(in) - mean q(out)        (Wu et al., 2021)
    ats       {ep, effective running}         (Zhang et al., 2022)

Every variant ends with a one-hot of the current phase. Blocks are zero-padded
separately, so feature k means the same on every intersection.

Reward variants: `queue` (minus the sum of queues on the incoming roads, as in
Advanced-CoLight; default) and `pressure` (PressLight).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from tsc_marl.domain import ats
from tsc_marl.domain.sensors import FailureSpec, SensorModel, failure_units

if TYPE_CHECKING:  # type-only: the submodule is importable only after `libsignal.bootstrap()`
    from world.world_cityflow import Intersection, World

DEFAULT_MAX_SPEED = 11.11  # m/s, world_cityflow's fallback for lanes without maxSpeed

STATE_VARIANTS = ("queue", "pressure", "ep", "ats")
REWARD_VARIANTS = ("queue", "pressure")


def engine_api(world: World) -> Any:
    """Engine with the CityFlow API: `world.eng` on CityFlow, the `cityflow_api`
    view on SUMO (`libsignal/sumo_world.py`), so queues and vehicle positions are
    read the same way on both engines."""
    return getattr(world, "cityflow_api", None) or world.eng


def phase_mask(phase_counts: Sequence[int], n_phases: int) -> np.ndarray | None:
    """[N, n_phases] bool: which actions exist at each intersection.

    None when all intersections have `n_phases` phases (JiNan, HangZhou): an
    all-true mask changes nothing, and skipping it keeps the computation
    bit-identical. On a heterogeneous network (Krakow, 2 to 7 phases) the policy
    could otherwise pick a phase the intersection does not have.
    """
    counts = np.asarray(phase_counts)
    if (counts >= n_phases).all():
        return None
    return np.arange(n_phases)[None, :] < counts[:, None]


class Structs(NamedTuple):
    """Static data of one intersection, computed once at construction."""

    movements: list[ats.Movement]
    in_lanes: list[str]  # in lanes of the controlled movements: demand domain
    all_lanes: list[str]  # in and out lanes of the controlled movements: pressure domain
    reward_lanes: list[str]  # all lanes of the incoming roads: reward domain (as CoLight)
    effective_range: dict[str, float]  # lane -> L = V_max * action interval (Def 8)


class StateBuilder:
    """Builds observations [N, F] and rewards [N] for a given intersection order.

    The order is an input: the core passes `graph.order_intersections(world)`,
    Advanced-CoLight passes `world.intersections` (the order its parent class
    uses). See the order contract in `core/graph.py`.
    """

    def __init__(
        self,
        world: World,
        intersections: Sequence[Intersection],
        *,
        state: str = "ats",
        reward: str = "queue",
        drop_always_green: bool = True,
        obs_length: float | None = None,
        action_interval: int = 15,
        n_phases: int | None = None,
        failure: FailureSpec | None = None,
    ) -> None:
        if state not in STATE_VARIANTS:
            raise ValueError(f"state={state!r}, allowed: {STATE_VARIANTS}")
        if reward not in REWARD_VARIANTS:
            raise ValueError(f"reward={reward!r}, allowed: {REWARD_VARIANTS}")

        self.world = world
        # `get_info` reads a cache refreshed only for subscribed functions.
        # LibSignal agents subscribe implicitly through `LaneVehicleGenerator`;
        # without this line `rewards()` raises KeyError on the first step.
        # Subscribing is idempotent.
        world.subscribe("lane_waiting_count")
        self.intersections = list(intersections)
        self.state = state
        self.reward = reward
        self.obs_length = obs_length
        self.drop_always_green = drop_always_green
        self.structs = [
            self._build_structs(inter, drop_always_green, action_interval, obs_length)
            for inter in self.intersections
        ]
        self.n_mov = max(len(s.movements) for s in self.structs)
        self.n_phases = n_phases or max(len(inter.phases) for inter in self.intersections)

        # One sensor model for the whole network: the same physical detector is
        # read by two neighbouring intersections and must show both the same value.
        self.failure = failure or FailureSpec()
        self.sensors: SensorModel | None = None
        if self.failure.active:
            lanes = sorted({lane for st in self.structs for lane in st.all_lanes})
            self.sensors = SensorModel(
                self.failure, failure_units(self.intersections, lanes, self.failure.unit)
            )

    @classmethod
    def from_registry(
        cls, world: World, intersections: Sequence[Intersection], **overrides: Any
    ) -> StateBuilder:
        """Configured like the rest of the repo, through LibSignal's YAML configs."""
        from common.registry import Registry

        param = Registry.mapping["model_mapping"]["setting"].param
        kwargs: dict[str, Any] = {
            "state": param.get("state", "ats"),
            "reward": param.get("reward", "queue"),
            "drop_always_green": param.get("drop_always_green", True),
            "obs_length": param.get("obs_length", None),
            "failure": FailureSpec.from_registry(),
            "action_interval": Registry.mapping["trainer_mapping"]["setting"].param[
                "action_interval"
            ],
        }
        return cls(world, intersections, **{**kwargs, **overrides})

    def _build_structs(
        self,
        inter: Intersection,
        drop_always_green: bool,
        action_interval: int,
        obs_length: float | None,
    ) -> Structs:
        movements = ats.controlled_movements(inter, drop_always_green)
        lanes_in = sorted(ats.in_lanes(movements))
        return Structs(
            movements=movements,
            in_lanes=lanes_in,
            all_lanes=sorted(ats.all_lanes(movements)),
            # The reward counts all lanes of the incoming roads, right turns
            # included, as `LaneVehicleGenerator(in_only=True)` does for CoLight.
            reward_lanes=ats.in_road_lanes(inter),
            effective_range={
                lane: (
                    obs_length
                    if obs_length is not None
                    else self.world.all_lanes_speed.get(lane, DEFAULT_MAX_SPEED) * action_interval
                )
                for lane in lanes_in
            },
        )

    @property
    def n_nodes(self) -> int:
        return len(self.intersections)

    @property
    def ob_length(self) -> int:
        """Row width; the only place where the observation size enters the
        architecture (the encoder's first `nn.Linear`)."""
        blocks = 2 if self.state == "ats" else 1
        return blocks * self.n_mov + self.n_phases

    def phase_mask(self) -> np.ndarray | None:
        """[N, n_phases] bool in this builder's order (see module-level `phase_mask`)."""
        return phase_mask([len(inter.phases) for inter in self.intersections], self.n_phases)

    def movement_mask(self) -> np.ndarray | None:
        """[N, n_mov] bool: which feature columns are real movements and which are
        zero padding up to `n_mov`. None on homogeneous networks, as `phase_mask`.

        Needed by the frozen-reading rule (`RuleGate`, `RuleUncertainty`): a
        padded zero never changes, so without the mask a small intersection would
        look mostly broken even on clean traffic."""
        return phase_mask([len(st.movements) for st in self.structs], self.n_mov)

    def _snapshot(self) -> tuple[dict, dict, dict]:
        """One fetch per call: these return the state of the whole network, so
        calling them per intersection would copy the same dicts N times."""
        eng = engine_api(self.world)
        return eng.get_lane_vehicles(), eng.get_vehicle_distance(), eng.get_vehicle_speed()

    def _queue(self, struct: Structs, snapshot, full_queue) -> Mapping[str, float]:
        """Queue within `obs_length` of the stop line (as in the authors' code),
        or over the whole lane when no window is set."""
        if self.obs_length is None:
            return full_queue
        lane_vehicles, distances, speeds = snapshot
        return ats.windowed_queue(
            struct.all_lanes,
            lane_vehicles,
            distances,
            speeds,
            self.world.lane_length,
            self.obs_length,
        )

    def observe(self, *, corrupt: bool = True) -> np.ndarray:
        """[N, ob_length] float32 in the order given at construction.

        `corrupt=False` bypasses the failure layer without touching its state (no
        sampling, no `hold` update), so two calls at the same simulation moment
        give a (clean, corrupted) pair from exactly the same engine state.
        """
        sensors = self.sensors if corrupt else None
        if sensors is not None:
            sensors.begin_decision()
        snapshot = self._snapshot()
        lane_vehicles, distances, speeds = snapshot
        full_queue = (
            None if self.obs_length is not None else self.world.get_info("lane_waiting_count")
        )

        rows = np.zeros((self.n_nodes, self.ob_length), dtype=np.float32)
        for idx, (inter, s) in enumerate(zip(self.intersections, self.structs, strict=True)):
            q = self._queue(s, snapshot, full_queue)
            if sensors is not None:
                q = sensors.read(q, s.all_lanes, channel="queue")

            if self.state == "queue":
                head = [sum(q[lane] for lane in ins) / len(ins) for ins, _ in s.movements]
            elif self.state == "pressure":
                head = ats.movement_pressure(s.movements, q, normalize=False)
            else:  # "ep" and the first block of "ats"
                head = ats.movement_pressure(s.movements, q)
            rows[idx, : len(head)] = head

            if self.state == "ats":
                running = ats.effective_running(
                    s.in_lanes,
                    lane_vehicles,
                    distances,
                    speeds,
                    self.world.lane_length,
                    s.effective_range,
                )
                if sensors is not None:
                    running = sensors.read(running, s.in_lanes, channel="running")
                demand = ats.movement_running(s.movements, running)
                rows[idx, self.n_mov : self.n_mov + len(demand)] = demand

            # Current phase as an action index (yellow excluded). During a
            # transition it holds the target phase: the observation describes
            # the decision, not the lamp.
            rows[idx, self.ob_length - self.n_phases + inter.current_phase] = 1.0
        return rows

    def phase_incidence(self) -> np.ndarray:
        """[N, n_phases, n_mov] bool: which movements (feature columns) each phase serves.

        With it a max-pressure decision is computed from the observation alone,
        including a corrupted one, so max-pressure controllers face the same
        sensor failure as the policy (`robustness/controllers.py`).
        """
        out = np.zeros((self.n_nodes, self.n_phases, self.n_mov), dtype=bool)
        for i, inter in enumerate(self.intersections):
            for phase, movements in enumerate(ats.phase_incidence(inter, self.drop_always_green)):
                out[i, phase, movements] = True
        return out

    def reset_sensors(self, episode: int) -> None:
        """Redraw the persistent detector state for a new episode. Otherwise the
        same lanes would stay broken for the whole training and `hold` would leak
        values from before `world.reset()`."""
        if self.sensors is not None:
            self.sensors.reset(episode)

    def rewards(self) -> np.ndarray:
        """[N] float32, always from the whole-lane queue, never the `obs_length`
        window, so the reward stays comparable with CoLight.

        The failure layer never touches it, and not by a flag: `get_info` reads
        the world's cache, not the observation path. Failures affect only the
        policy input."""
        queue = self.world.get_info("lane_waiting_count")
        out = np.zeros(self.n_nodes, dtype=np.float32)
        for idx, s in enumerate(self.structs):
            if self.reward == "queue":
                out[idx] = -sum(queue[lane] for lane in s.reward_lanes)
            else:  # "pressure" (PressLight): penalise imbalance, hence abs
                out[idx] = -abs(sum(ats.movement_pressure(s.movements, queue, normalize=False)))
        return out
