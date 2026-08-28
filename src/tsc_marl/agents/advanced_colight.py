"""Advanced-CoLight (Zhang i in. 2022) — CoLight z ATS zamiast `lane_count`.

Architektura sieci BEZ ZMIAN: `ColightNet` dostaje tylko inny wymiar wejścia
(jedyne miejsce, gdzie wymiar obserwacji wchodzi do modelu, to pierwszy
`nn.Linear` w `Embedding_MLP`). Cały wkład metody siedzi w reprezentacji stanu
— i to jest teza papieru: "expression might be enough".

Stan (sekcja 4.3): bieżąca faza + ATS (Def. 11), czyli dla KAŻDEGO ruchu (l,m)
efficient pressure e(l,m) i effective running r(l,m). Nie sumy po fazie — te są
decyzją Advanced-MP, nie stanem RL. Liczy to `tsc_marl.features.StateBuilder`,
ten sam, którego używa rdzeń Fazy 1: stan jest CAŁYM wkładem tej metody, więc
dwie jego kopie byłyby najgorszym możliwym miejscem na rozjazd (patrz docstring
`features.py`).

Reward zostaje CoLightowy (−Σ kolejek) — Advanced-XLight zmienia tylko stan.
"""

from __future__ import annotations

from agent.base import Registry
from agent.colight import CoLightAgent

from tsc_marl.features import StateBuilder, Structs


@Registry.register_model("advancedcolight")
class AdvancedCoLight(CoLightAgent):
    def __init__(self, world, rank):
        # MUSI powstać przed super(): __init__ rodzica woła _build_model(), a ten
        # sięga po builder. CoLight sam mapuje rank → intersection_id, więc tu
        # przekazujemy rank surowy (podwójne mapowanie = TypeError na liście).
        self._features: StateBuilder | None = None
        super().__init__(world, rank)

    # ------------------------------------------------------------------ statyka

    def _builder(self) -> StateBuilder:
        """Leniwy i cache'owany: roadnet się nie zmienia, a `reset()` między
        epizodami przebudowuje tylko generatory. Leniwość jest konieczna, bo
        wołane z DWÓCH stron — z `_build_model()` (w środku `__init__` rodzica,
        zanim istnieje optymalizator) i z `get_ob()`.

        Kolejność wierszy to `world.intersections` — ta sama, w której rodzic
        zbudował `ob_generator`. Rdzeń Fazy 1 poda tu `graph.order_intersections`;
        na roadnetach CityFlow obie są dziś identyczne (patrz `graph.py`).
        """
        if self._features is None:
            # state/reward na sztywno: Advanced-CoLight JEST zdefiniowany przez ATS,
            # więc nie podlega przełącznikowi ablacyjnemu — ten należy do rdzenia.
            self._features = StateBuilder.from_registry(
                self.world, self.world.intersections, state="ats", reward="queue"
            )
        return self._features

    def _ats_structs(self) -> list[Structs]:
        """Statyka per skrzyżowanie — używane przez testy kontraktu obserwacji."""
        return self._builder().structs

    def _build_model(self):
        """Nadpisane WYŁĄCZNIE po to, żeby podać właściwy `ob_length` — sieć
        buduje rodzic. Wołane dwa razy (model + target), więc idempotentne."""
        self.ob_length = self._builder().ob_length
        return super()._build_model()

    # ---------------------------------------------------------------- obserwacja

    def get_ob(self):
        """[skrzyżowania, ob_length] — układ wiersza i padding opisuje `features.py`."""
        return self._builder().observe()
