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
from agent import BaseAgent  # z submodułu external/LibSignal (patrz tsc_marl.libsignal)
from common.registry import Registry
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator
import numpy as np
import gym

WAITING_SPEED = 0.1  # próg "stoi" — ten sam, którego używa world_cityflow


class _PressureBase(BaseAgent):
    """Wspólna część: generatory metryk + rozbicie faz na ruchy (l,m) z użytymi pasami."""

    def __init__(self, world, rank):
        super().__init__(world)
        self.world = world
        self.rank = rank
        self.model = None
        param = Registry.mapping['model_mapping']['setting'].param
        # pomijaj ruchy zielone w KAŻDEJ fazie (skręty w prawo) — patrz _controlled_lanelinks
        self.drop_always_green = param.get('drop_always_green', False)
        self.obs_length = param.get('obs_length', None)
        self._build_generators()
        self.action_space = gym.spaces.Discrete(len(self.inter_obj.phases))
        self.t_min = Registry.mapping['model_mapping']['setting'].param['t_min']
        self._movements = self._phase_movements()

    def _build_generators(self):
        inter_id = self.world.intersection_ids[self.rank]
        self.inter_obj = self.world.id2intersection[inter_id]
        self.ob_generator = LaneVehicleGenerator(self.world, self.inter_obj, ['lane_count'],
                                                 in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(self.world, self.inter_obj, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, self.inter_obj, ["lane_count"],
                                                     in_only=True, average='all', negative=True)
        self.queue = LaneVehicleGenerator(self.world, self.inter_obj, ["lane_waiting_count"],
                                          in_only=True, negative=False)
        self.delay = LaneVehicleGenerator(self.world, self.inter_obj, ["lane_delay"],
                                          in_only=True, negative=False)

    @staticmethod
    def _road_of(lane):
        return lane.rsplit('_', 1)[0]

    def _controlled_lanelinks(self):
        """Lanelinki STEROWANE sygnałem = te, których NIE ma we wszystkich fazach.
        W roadnetach CityFlow (JiNan/HangZhou) skręty w prawo są dozwolone w KAŻDEJ
        fazie — nie konkurują o zielone, więc nie należą ani do pressure, ani do demand.
        Referencyjna implementacja autorów sumuje wyłącznie ruchy przelotowe i w lewo
        (`phase_1 = tr_mo_pr[1] + tr_mo_pr[4]` itd.)."""
        per_phase = [set(map(tuple, ll)) for ll in self.inter_obj.phase_available_lanelinks]
        always = set.intersection(*per_phase) if per_phase else set()
        return always

    def _phase_movements(self):
        """Faza → lista ruchów (l,m); dla każdego zbiór UŻYWANYCH pasów wjazdowych
        i wyjazdowych ("used lane numbers M, N" z Def 4)."""
        always = self._controlled_lanelinks() if self.drop_always_green else set()
        per_phase = []
        for phase_id in range(len(self.inter_obj.phases)):
            moves = {}
            for start, end in self.inter_obj.phase_available_lanelinks[phase_id]:
                if (start, end) in always:
                    continue
                key = (self._road_of(start), self._road_of(end))
                ins, outs = moves.setdefault(key, (set(), set()))
                ins.add(start)
                outs.add(end)
            per_phase.append(list(moves.values()))
        return per_phase

    def _windowed_queue(self, lane_vehicles, distances, speeds, obs_length):
        """Kolejka liczona TYLKO w oknie obs_length od stop-linii — tak jak liczy ją
        referencyjna implementacja autorów (`_get_part_observations`, OBS_LENGTH=167).
        Papier tego nie mówi: Def. 4 sugeruje kolejkę całego pasa."""
        lanes = {l for moves in self._movements for ins, outs in moves for l in (*ins, *outs)}
        q = {}
        for lane in lanes:
            length = self.world.lane_length.get(lane, 0.0)
            n = 0
            for vid in lane_vehicles.get(lane, []):
                if speeds.get(vid, 1.0) > WAITING_SPEED:
                    continue
                if length - distances.get(vid, 0.0) <= obs_length:
                    n += 1
            q[lane] = n
        return q

    def _phase_pressure(self, q):
        """p(s) dla każdej fazy — Def 4 + Def 5."""
        out = []
        for moves in self._movements:
            p = 0.0
            for ins, outs in moves:
                up = sum(q[l] for l in ins) / len(ins)
                down = sum(q[m] for m in outs) / len(outs)
                p += up - down
            out.append(p)
        return out

    def _effective_running(self, lane_vehicles, distances, speeds, unlimited=False):
        """r_e(l') dla pasów wjazdowych — Def 8 + Def 9. Liczy tylko pojazdy JADĄCE
        i będące w zasięgu L = V_max·Δt od skrzyżowania."""
        t_duration = Registry.mapping['trainer_mapping']['setting'].param['action_interval']
        in_lanes = {l for moves in self._movements for ins, _ in moves for l in ins}
        counts = {}
        for lane in in_lanes:
            L = (self.obs_length if self.obs_length is not None
                 else self.world.all_lanes_speed.get(lane, 11.11) * t_duration)
            length = self.world.lane_length.get(lane, 0.0)
            n = 0
            for vid in lane_vehicles.get(lane, []):
                if speeds.get(vid, 0.0) < WAITING_SPEED:
                    continue                        # stoi → to kolejka, nie "demand"
                if unlimited or length - distances.get(vid, 0.0) <= L:
                    n += 1
            counts[lane] = n
        return counts

    def _phase_demand(self, running):
        """d(s) dla każdej fazy — Def 10."""
        out = []
        for moves in self._movements:
            out.append(sum(running.get(l, 0) for ins, _ in moves for l in ins))
        return out

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


@Registry.register_model('efficientmp')
class EfficientMaxPressureAgent(_PressureBase):
    """Wybiera fazę o największym efficient pressure (Wu 2021)."""

    def __repr__(self):
        return 'Efficient-MP Agent has no Network model'

    def get_action(self, ob, phase, test=True):
        if self.inter_obj.current_phase_time < self.t_min:
            return self.inter_obj.current_phase
        q = self.world.get_info("lane_waiting_count")
        return int(np.argmax(self._phase_pressure(q)))


@Registry.register_model('advancedmp')
class AdvancedMaxPressureAgent(_PressureBase):
    """Advanced-MP (Zhang 2022, Algorytm 1): bieżąca faza broni się swoim demand
    (d·W1), reszta konkuruje efficient pressure."""

    def __init__(self, world, rank):
        super().__init__(world, rank)
        self.w1 = Registry.mapping['model_mapping']['setting'].param['w1']

    def __repr__(self):
        return 'Advanced-MP Agent has no Network model'

    def get_action(self, ob, phase, test=True):
        if self.inter_obj.current_phase_time < self.t_min:
            return self.inter_obj.current_phase

        # prosto z silnika — world.get_info buforuje tylko klucze subskrybowane przez generatory
        lane_veh = self.world.eng.get_lane_vehicles()
        dist = self.world.eng.get_vehicle_distance()
        spd = self.world.eng.get_vehicle_speed()

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
