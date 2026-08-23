from agent.base import Registry
from agent.colight import CoLightAgent
from world.world_cityflow import World


def calc_effective_vehicle_running():
    pass


@Registry.register_model("advancedcolight")
class AdvancedCoLight(CoLightAgent):
    def __init__(self, world: World, rank):
        super().__init__(world, world.intersection_ids[rank])
