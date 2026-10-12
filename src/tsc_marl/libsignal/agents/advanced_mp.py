"""Efficient-MP (Wu et al., 2021) and Advanced-MP (Zhang et al., 2022).

Implemented from the papers' definitions, because LibSignal has neither, and
its own MaxPressure computes pressure from `lane_count` (all vehicles, no
per-lane normalisation) instead of queues. The formulas live in `domain/ats.py`:

    Def 5   phase pressure  p(s) = sum of e(l,m) over the movements of s
    Def 10  phase demand    d(s) = sum of r(l,m) over the movements of s
    Alg 1   Advanced-MP     keep the current phase if d(a_cur) * W1 > max p(s),
                            otherwise argmax p(s)
"""

import gym.spaces  # not `import gym`: its __init__ does not re-export `spaces` for type checkers
import numpy as np
from agent import BaseAgent
from common.registry import Registry
from generator import IntersectionPhaseGenerator, LaneVehicleGenerator

from tsc_marl.core.features import DEFAULT_MAX_SPEED, engine_api
from tsc_marl.domain import ats


class _PressureBase(BaseAgent):
    """Shared part: metric generators and the phase -> movement split."""

    def __init__(self, world, rank):
        super().__init__(world)
        self.world = world
        self.rank = rank
        self.model = None
        param = Registry.mapping["model_mapping"]["setting"].param
        # skip links green in every phase (right turns), see ats.always_green_lanelinks
        self.drop_always_green = param.get("drop_always_green", False)
        self.obs_length = param.get("obs_length", None)
        self._build_generators()
        self.action_space = gym.spaces.Discrete(len(self.inter_obj.phases))
        self.t_min = Registry.mapping["model_mapping"]["setting"].param["t_min"]
        self._movements = ats.phase_movements(self.inter_obj, self.drop_always_green)
        # the same movements as a flat list: the domain of the lane sets
        self._flat = ats.controlled_movements(self.inter_obj, self.drop_always_green)

    def _build_generators(self):
        inter_id = self.world.intersection_ids[self.rank]
        self.inter_obj = self.world.id2intersection[inter_id]
        self.ob_generator = LaneVehicleGenerator(
            self.world, self.inter_obj, ["lane_count"], in_only=True, average=None
        )
        # LibSignal's defaults `("phase")` and `("cur_phase")` are strings, not tuples
        # (missing comma), so pyright infers `str`; lists are what the code expects.
        self.phase_generator = IntersectionPhaseGenerator(
            self.world,
            self.inter_obj,
            ["phase"],  # pyright: ignore[reportArgumentType]
            targets=["cur_phase"],  # pyright: ignore[reportArgumentType]
            negative=False,
        )
        self.reward_generator = LaneVehicleGenerator(
            self.world, self.inter_obj, ["lane_count"], in_only=True, average="all", negative=True
        )
        self.queue = LaneVehicleGenerator(
            self.world, self.inter_obj, ["lane_waiting_count"], in_only=True, negative=False
        )
        self.delay = LaneVehicleGenerator(
            self.world, self.inter_obj, ["lane_delay"], in_only=True, negative=False
        )

    def _phase_pressure(self, q):
        return ats.phase_pressure(self._movements, q)

    def _phase_demand(self, running):
        return ats.phase_demand(self._movements, running)

    def _windowed_queue(self, lane_vehicles, distances, speeds, obs_length):
        return ats.windowed_queue(
            ats.all_lanes(self._flat),
            lane_vehicles,
            distances,
            speeds,
            self.world.lane_length,
            obs_length,
        )

    def _effective_running(self, lane_vehicles, distances, speeds, unlimited=False):
        t_duration = Registry.mapping["trainer_mapping"]["setting"].param["action_interval"]
        lanes = ats.in_lanes(self._flat)
        # obs_length = the authors' fixed window; without it L = V_max(lane) * interval (Def 8)
        rng = (
            self.obs_length
            if self.obs_length is not None
            else {
                l: self.world.all_lanes_speed.get(l, DEFAULT_MAX_SPEED) * t_duration for l in lanes
            }
        )
        return ats.effective_running(
            lanes,
            lane_vehicles,
            distances,
            speeds,
            self.world.lane_length,
            rng,
            unlimited=unlimited,
        )

    def reset(self):
        self._build_generators()

    def get_ob(self):
        return np.array([self.ob_generator.generate()], dtype=np.float32)

    def get_reward(self):
        return np.squeeze(np.array([self.reward_generator.generate()])) * 12

    def get_phase(self):
        return (np.concatenate([self.phase_generator.generate()])).astype(np.int8)

    def get_queue(self):
        return sum(self.queue.generate())

    def get_delay(self):
        return sum(self.delay.generate())


@Registry.register_model("efficientmp")
class EfficientMaxPressureAgent(_PressureBase):
    """Picks the phase with the largest efficient pressure (Wu et al., 2021)."""

    def __repr__(self):
        return "Efficient-MP Agent has no Network model"

    def get_action(self, ob, phase, test=True):
        if self.inter_obj.current_phase_time < self.t_min:
            return self.inter_obj.current_phase
        q = self.world.get_info("lane_waiting_count")
        return int(np.argmax(self._phase_pressure(q)))


@Registry.register_model("advancedmp")
class AdvancedMaxPressureAgent(_PressureBase):
    """Advanced-MP (Zhang et al., 2022, Alg 1): the current phase defends itself with
    its demand (d * W1), the others compete on efficient pressure."""

    def __init__(self, world, rank):
        super().__init__(world, rank)
        self.w1 = Registry.mapping["model_mapping"]["setting"].param["w1"]

    def __repr__(self):
        return "Advanced-MP Agent has no Network model"

    def get_action(self, ob, phase, test=True):
        if self.inter_obj.current_phase_time < self.t_min:
            return self.inter_obj.current_phase

        # straight from the engine (world.get_info caches only subscribed keys);
        # `engine_api` so that SUMO goes through the CityFlow view of `sumo_world.py`
        eng = engine_api(self.world)
        lane_veh = eng.get_lane_vehicles()
        dist = eng.get_vehicle_distance()
        spd = eng.get_vehicle_speed()

        if self.obs_length is not None:
            q = self._windowed_queue(lane_veh, dist, spd, self.obs_length)
        else:
            q = self.world.get_info("lane_waiting_count")
        pressures = self._phase_pressure(q)
        running = self._effective_running(lane_veh, dist, spd)
        demands = self._phase_demand(running)

        cur = self.inter_obj.current_phase
        keep = demands[cur] * self.w1 > max(pressures)

        if keep:
            return cur
        return int(np.argmax(pressures))
