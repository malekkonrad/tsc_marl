"""Shared test fixtures.

ATS tests need no engine: LibSignal's `Intersection` parses the roadnet itself and
touches the outside only through `world.eng.set_tl_phase` in `reset()`. We build the
REAL `Intersection` on a stub world, so the tests check the code that runs in the
simulation, not our own copy of the roadnet parsing, in ~0.1 s instead of starting CityFlow.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tsc_marl import libsignal, paths

libsignal.bootstrap()  # submodule path + SUMO_HOME (world/__init__ imports world_sumo)

ROADNETS = {
    "jinan3x4": paths.DATA / "raw_data" / "jinan_3x4" / "roadnet_3_4_4phase.json",
    "hangzhou4x4": paths.DATA / "raw_data" / "hangzhou_4x4_advanced" / "roadnet_4_4_4phase.json",
}


class _FakeEngine:
    def set_tl_phase(self, *args):
        pass


class _FakeWorld:
    RIGHT = True

    def __init__(self):
        self.eng = _FakeEngine()


def load_intersections(roadnet: Path):
    """(Intersection object, raw roadnet dict) for every non-virtual intersection."""
    from world.world_cityflow import Intersection  # importable only after bootstrap

    raw = json.loads(roadnet.read_text())
    world = _FakeWorld()
    return [(Intersection(d, world), d) for d in raw["intersections"] if not d["virtual"]]


@pytest.fixture(scope="session", params=sorted(ROADNETS))
def roadnet(request):
    path = ROADNETS[request.param]
    if not path.is_file():
        pytest.skip(f"no roadnet {path} (see data/README.md)")
    return load_intersections(path)


# A real CityFlow world + agent for tests marked `slow`. It goes through the SAME
# interface sequence as production (`session.build_interfaces`): LibSignal fills its
# global Registry in a fixed order, and a private copy of that sequence would let the
# test pass after production drifts. The Registry has no clean reset, so the world is
# built ONCE per session.

NETWORK = "cityflow_jinan3x4_4ph"


def _build(agent_name: str, network: str):
    # lazy: importing LibSignal fills its Registry, which only the slow tests need
    from common.registry import Registry
    from utils.logger import build_config

    from tsc_marl.libsignal import session
    from tsc_marl.libsignal.experiment import parse_args

    session.register_engine_components()
    args = parse_args(["-a", agent_name, "-n", network])
    config, _ = build_config(args)
    session.build_interfaces(config)

    from world.world_cityflow import World

    world = World(f"configs/sim/{network}.cfg", thread_num=4, interface="libsumo")
    model = Registry.mapping["model_mapping"][agent_name](world, 0)
    return world, model


@pytest.fixture(scope="session")
def advanced_colight_world():
    """(world, agent) for `advancedcolight` on JiNan 3x4; ~15 s, hence `slow`."""
    cfg = paths.CONFIGS / "sim" / f"{NETWORK}.cfg"
    snapshot = cfg.read_bytes()  # LibSignal writes replay log paths here
    try:
        yield _build("advancedcolight", NETWORK)
    finally:
        Path(cfg).write_bytes(snapshot)
