"""Advanced-CoLight (Zhang et al., 2022): CoLight with the ATS state instead of `lane_count`.

The network architecture is unchanged: `ColightNet` only gets another input
size (the first `nn.Linear` of `Embedding_MLP`). The method's whole
contribution is the state representation, which is the paper's claim
("expression might be enough").

State: the current phase plus ATS (Def 11), i.e. efficient pressure and
effective running per movement. It is computed by `core.features.StateBuilder`,
the same code the core uses, because two copies of the state would be the worst
possible place for a discrepancy. The reward stays CoLight's (minus the queues).
"""

from __future__ import annotations

from agent.base import Registry
from agent.colight import CoLightAgent

from tsc_marl.core.features import StateBuilder, Structs


@Registry.register_model("advancedcolight")
class AdvancedCoLight(CoLightAgent):
    def __init__(self, world, rank):
        # Must exist before super(): the parent's __init__ calls _build_model(),
        # which needs the builder. CoLight maps rank -> intersection id itself, so
        # the raw rank is passed on.
        self._features: StateBuilder | None = None
        super().__init__(world, rank)

    # ------------------------------------------------------------------ static

    def _builder(self) -> StateBuilder:
        """Lazy and cached (the roadnet does not change), because it is needed both
        by `_build_model()`, inside the parent's `__init__`, and by `get_ob()`.

        Rows follow `world.intersections`, the order the parent built its
        generators in; on CityFlow roadnets it equals the graph order
        (`core/graph.py`).
        """
        if self._features is None:
            # state and reward fixed: Advanced-CoLight IS defined by ATS
            self._features = StateBuilder.from_registry(
                self.world, self.world.intersections, state="ats", reward="queue"
            )
        return self._features

    def _ats_structs(self) -> list[Structs]:
        """Per-intersection static data, used by the observation contract tests."""
        return self._builder().structs

    def _build_model(self):
        """Overridden only to set `ob_length`; the parent builds the network.
        Called twice (model and target), so idempotent."""
        self.ob_length = self._builder().ob_length
        return super()._build_model()

    # ---------------------------------------------------------------- observation

    def get_ob(self):
        """[N, ob_length]; row layout and padding as in `core/features.py`."""
        return self._builder().observe()
