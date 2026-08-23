from agent.base import Registry
from agent.colight import CoLightAgent


@Registry.register_model("advancedcolight")
class AdvancedCoLight(CoLightAgent):
    def __init__(self, world, rank):
        super().__init__(world, world.intersection_ids[rank])


class MyOwn(AdvancedCoLight):
    pass
