"""
Efficient-MP (Wu et al. 2021) i Advanced-MP (Zhang et al. 2022) — implementacja
wg definicji z papieru "Expression might be enough" (Advanced-CoLight), bo
LibSignal nie ma żadnej z tych metod, a jego własny MaxPressure liczy pressure
z lane_count (wszystkie pojazdy, bez normalizacji per pas) zamiast z kolejek.

Def 4  efficient pressure:  e(l,m) = (1/M)·Σ q(l') − (1/N)·Σ q(m')
Def 5  phase pressure:      p(s)   = Σ e(l,m), (l,m) ∈ s
Def 8  effective range:     L      = V_max · t_duration      (165 m przy Δt=15)
Def 9  effective running:   r(l,m) = Σ r_e(l'),  l' ∈ l
Def 10 phase demand:        d(s)   = Σ r(l,m), (l,m) ∈ s
Alg. 1 Advanced-MP:  jeśli d(a_cur)·W1 > max p(s) → utrzymaj fazę, inaczej argmax p(s)
"""

import gym.spaces  # NIE `import gym`: gym ma py.typed, a jego __init__ nie re-eksportuje
import numpy as np  # podmodułu `spaces`, więc statycznie `gym.spaces` nie istnieje
from agent import BaseAgent
from common.registry import Registry
from generator import IntersectionPhaseGenerator, LaneVehicleGenerator

from tsc_marl.core.features import engine_api
from tsc_marl.domain import ats


class _PressureBase(BaseAgent):
    """Wspólna część: generatory metryk + rozbicie faz na ruchy (l,m) z użytymi pasami."""

    def __init__(self, world, rank):
        super().__init__(world)
        self.world = world
        self.rank = rank
        self.model = None
        param = Registry.mapping["model_mapping"]["setting"].param
        # pomijaj ruchy zielone w KAŻDEJ fazie (skręty w prawo) — patrz ats.always_green_lanelinks
        self.drop_always_green = param.get("drop_always_green", False)
        self.obs_length = param.get("obs_length", None)
        self._build_generators()
        self.action_space = gym.spaces.Discrete(len(self.inter_obj.phases))
        self.t_min = Registry.mapping["model_mapping"]["setting"].param["t_min"]
        self._movements = ats.phase_movements(self.inter_obj, self.drop_always_green)
        # płaska lista tych samych ruchów — dziedzina zbiorów pasów (bez duplikatów)
        self._flat = ats.controlled_movements(self.inter_obj, self.drop_always_green)

    def _build_generators(self):
        inter_id = self.world.intersection_ids[self.rank]
        self.inter_obj = self.world.id2intersection[inter_id]
        self.ob_generator = LaneVehicleGenerator(
            self.world, self.inter_obj, ["lane_count"], in_only=True, average=None
        )
        # pyright widzi tu `fns: str` / `targets: str`, bo IntersectionPhaseGenerator ma
        # w LibSignalu domyślne `("phase")` i `("cur_phase")` — to nie krotki, tylko stringi
        # w nawiasach (brak przecinka). Listy są tym, czego ten kod naprawdę oczekuje;
        # domyślna wartość jest niedziałająca i nikt jej nie używa.
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
        # obs_length = stałe okno autorów; bez niego L liczone per pas z jego V_max (Def 8)
        rng = (
            self.obs_length
            if self.obs_length is not None
            else {l: self.world.all_lanes_speed.get(l, 11.11) * t_duration for l in lanes}
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
    """Wybiera fazę o największym efficient pressure (Wu 2021)."""

    def __repr__(self):
        return "Efficient-MP Agent has no Network model"

    def get_action(self, ob, phase, test=True):
        if self.inter_obj.current_phase_time < self.t_min:
            return self.inter_obj.current_phase
        q = self.world.get_info("lane_waiting_count")
        return int(np.argmax(self._phase_pressure(q)))


@Registry.register_model("advancedmp")
class AdvancedMaxPressureAgent(_PressureBase):
    """Advanced-MP (Zhang 2022, Algorytm 1): bieżąca faza broni się swoim demand
    (d·W1), reszta konkuruje efficient pressure."""

    def __init__(self, world, rank):
        super().__init__(world, rank)
        self.w1 = Registry.mapping["model_mapping"]["setting"].param["w1"]

    def __repr__(self):
        return "Advanced-MP Agent has no Network model"

    def get_action(self, ob, phase, test=True):
        if self.inter_obj.current_phase_time < self.t_min:
            return self.inter_obj.current_phase

        # prosto z silnika — world.get_info buforuje tylko klucze subskrybowane przez generatory;
        # `engine_api`, żeby na SUMO szło przez widok CityFlow z `libsignal/sumo_world.py`
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
