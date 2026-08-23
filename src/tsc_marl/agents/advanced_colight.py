"""Advanced-CoLight (Zhang i in. 2022) — CoLight z ATS zamiast `lane_count`.

Architektura sieci BEZ ZMIAN: `ColightNet` dostaje tylko inny wymiar wejścia
(jedyne miejsce, gdzie wymiar obserwacji wchodzi do modelu, to pierwszy
`nn.Linear` w `Embedding_MLP`). Cały wkład metody siedzi w reprezentacji stanu
— i to jest teza papieru: "expression might be enough".

Stan (sekcja 4.3): bieżąca faza + ATS (Def. 11), czyli dla KAŻDEGO ruchu (l,m)
efficient pressure e(l,m) i effective running r(l,m). Nie sumy po fazie — te są
decyzją Advanced-MP, nie stanem RL.

Układ wiersza (stały dla całej sieci, żeby cecha nr k znaczyła to samo na
każdym skrzyżowaniu — CoLight dzieli jeden model między wszystkie):

    [ e(l,m) × n_mov | r(l,m) × n_mov | one-hot fazy × n_faz ]

Bloki są dopychane zerami OSOBNO, nie na końcu wiersza: przy sieci z różnymi
skrzyżowaniami (NY) doklejenie paddingu na końcu przesunęłoby one-hot na inne
kolumny na różnych skrzyżowaniach.

Reward zostaje CoLightowy (−Σ kolejek) — Advanced-XLight zmienia tylko stan.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from agent.base import Registry
from agent.colight import CoLightAgent

from tsc_marl.agents import ats

DEFAULT_MAX_SPEED = 11.11  # fallback jak w world_cityflow, gdy pas nie ma maxSpeed


class _Structs(NamedTuple):
    """Statyka jednego skrzyżowania — nie zmienia się w trakcie symulacji."""

    movements: list[ats.Movement]
    in_lanes: list[str]
    all_lanes: list[str]
    effective_range: dict[str, float]  # pas → L = V_max·Δt (Def. 8)


@Registry.register_model("advancedcolight")
class AdvancedCoLight(CoLightAgent):
    def __init__(self, world, rank):
        # MUSI powstać przed super(): __init__ rodzica woła _build_model(), a ten
        # sięga po struktury. CoLight sam mapuje rank → intersection_id, więc tu
        # przekazujemy rank surowy (podwójne mapowanie = TypeError na liście).
        self._structs: list[_Structs] | None = None
        self._n_mov = 0
        super().__init__(world, rank)

    # ------------------------------------------------------------------ statyka

    def _ats_structs(self) -> list[_Structs]:
        """Liczone raz i cache'owane: roadnet się nie zmienia, a `reset()` między
        epizodami przebudowuje tylko generatory. Leniwe, bo wołane z DWÓCH stron
        — z `_build_model()` (w środku `__init__` rodzica, zanim istnieje
        optymalizator) i z `get_ob()`."""
        if self._structs is None:
            param = Registry.mapping["model_mapping"]["setting"].param
            drop = param.get("drop_always_green", True)
            t_duration = Registry.mapping["trainer_mapping"]["setting"].param["action_interval"]
            obs_length = param.get("obs_length", None)

            structs = []
            for inter in self.world.intersections:
                movements = ats.controlled_movements(inter, drop)
                lanes_in = sorted(ats.in_lanes(movements))
                structs.append(
                    _Structs(
                        movements=movements,
                        in_lanes=lanes_in,
                        all_lanes=sorted(ats.all_lanes(movements)),
                        effective_range={
                            lane: (
                                obs_length
                                if obs_length is not None
                                else self.world.all_lanes_speed.get(lane, DEFAULT_MAX_SPEED)
                                * t_duration
                            )
                            for lane in lanes_in
                        },
                    )
                )
            self._structs = structs
            self._n_mov = max(len(s.movements) for s in structs)
        return self._structs

    def _build_model(self):
        """Nadpisane WYŁĄCZNIE po to, żeby podać właściwy `ob_length` — sieć
        buduje rodzic. Wołane dwa razy (model + target), więc idempotentne."""
        self._ats_structs()
        self.ob_length = 2 * self._n_mov + self.action_space.n
        return super()._build_model()

    # ---------------------------------------------------------------- obserwacja

    def get_ob(self):
        """[skrzyżowania, ob_length], w kolejności `world.intersections` — tej
        samej, w której rodzic zbudował `ob_generator` i w której indeksowany
        jest graf (`node_id2idx` powstaje z tego samego pliku roadnetu)."""
        # JEDNO pobranie na wywołanie: te trzy zwracają stan CAŁEJ sieci, nie
        # skrzyżowania. Wołanie ich w pętli po skrzyżowaniach kopiuje ten sam
        # słownik N razy. Subskrypcja przez world.get_info nie pomoże — cache
        # odświeża się co KROK symulacji (1 s), a decyzja zapada raz na Δt.
        eng = self.world.eng
        lane_vehicles = eng.get_lane_vehicles()
        distances = eng.get_vehicle_distance()
        speeds = eng.get_vehicle_speed()

        obs_length = Registry.mapping["model_mapping"]["setting"].param.get("obs_length", None)
        full_queue = None if obs_length is not None else self.world.get_info("lane_waiting_count")

        rows = np.zeros((len(self.world.intersections), self.ob_length), dtype=np.float32)
        for idx, (inter, s) in enumerate(
            zip(self.world.intersections, self._ats_structs(), strict=True)
        ):
            if obs_length is not None:
                q = ats.windowed_queue(
                    s.all_lanes,
                    lane_vehicles,
                    distances,
                    speeds,
                    self.world.lane_length,
                    obs_length,
                )
            else:
                q = full_queue
            running = ats.effective_running(
                s.in_lanes,
                lane_vehicles,
                distances,
                speeds,
                self.world.lane_length,
                s.effective_range,
            )

            pressure = ats.movement_pressure(s.movements, q)
            demand = ats.movement_running(s.movements, running)
            rows[idx, : len(pressure)] = pressure
            rows[idx, self._n_mov : self._n_mov + len(demand)] = demand
            # bieżąca faza: indeks w inter.phases, czyli w przestrzeni AKCJI (bez
            # żółtej). W trakcie przejścia trzyma fazę docelową — i o to chodzi,
            # bo obserwacja opisuje decyzję, nie stan lampy.
            rows[idx, 2 * self._n_mov + inter.current_phase] = 1.0
        return rows
