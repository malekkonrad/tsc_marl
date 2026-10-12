"""SUMO adapter (`libsignal/sumo_world.py`): backend parity with CityFlow.

Fast tests check the adapter's pure functions on stubs. One slow test builds a real
SUMO world on the Krakow network in a SEPARATE process (`sumo_probe.py`): LibSignal
keeps its configuration in a global Registry, and the CityFlow world from
`conftest.py` lives for the whole session.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from common.utils import build_index_intersection_map_sumo
from sumolib import checkBinary

from tsc_marl import paths
from tsc_marl.core.features import engine_api, phase_mask
from tsc_marl.core.graph import RoadNetGraph
from tsc_marl.libsignal.bridge import child_env
from tsc_marl.libsignal.sumo_world import (
    build_graph,
    cityflow_transition,
    serving_lanelinks,
    travel_time,
)

PROBE = Path(__file__).with_name("sumo_probe.py")
NETWORK = "sumo_krakow3w_mid"
NO_KRAKOW = "no Krakow scenario; .venv/bin/python data/raw_data/krakow_3w/build.py"


def test_serving_lanelinks_takes_permissive_green():
    """Upstream takes only `G` and `s`, so a yielding turn (`g`) fell out of the phases."""
    links = [[("a_0", "x_0", ":v0")], [("a_1", "y_0", ":v1")], [], [("b_0", "x_0", ":v2")]]
    inter = SimpleNamespace(
        lanelinks=links,
        green_phases=[SimpleNamespace(state="GgrG"), SimpleNamespace(state="rrGs")],
    )
    lanelinks, starts = serving_lanelinks(inter)
    assert lanelinks[0] == [["a_0", "x_0"], ["a_1", "y_0"], ["b_0", "x_0"]]  # `g` at index 1
    assert lanelinks[1] == [["b_0", "x_0"]]  # `G` at an index without a link drops, `s` stays
    assert starts == [["a_0", "a_1", "b_0"], ["b_0"]]


def test_serving_lanelinks_truncates_like_upstream():
    """A phase state longer than the link list: upstream breaks, we zip."""
    inter = SimpleNamespace(
        lanelinks=[[("a_0", "x_0", ":v")]], green_phases=[SimpleNamespace(state="GG")]
    )
    assert serving_lanelinks(inter)[0] == [[["a_0", "x_0"]]]


def test_travel_time_counts_running_vehicles_like_cityflow():
    """`Engine::getAverageTravelTime`: finished + running since entry. Upstream would
    give 150 here (finished only) and reward a method that blocks traffic."""
    finished = {"v1": 100.0, "v2": 200.0}
    entered = {"v1": 0.0, "v2": 50.0, "v3": 300.0, "v4": 500.0}
    assert travel_time(finished, entered, now=700.0) == (100 + 200 + 400 + 200) / 4
    assert travel_time({}, {}, now=10.0) == 0.0


class _Lights:
    def __init__(self):
        self.calls: list[int] = []

    def setPhase(self, _tl_id, index):  # noqa: N802 - SUMO API name
        self.calls.append(index)


def _inter(yellow_phase_time: int, yellow_dict: dict[str, int]) -> SimpleNamespace:
    return SimpleNamespace(
        id="tl",
        eng=SimpleNamespace(trafficlight=_Lights()),
        current_phase=0,
        current_phase_time=0,
        transition=None,
        yellow_phase_time=yellow_phase_time,
        yellow_dict=yellow_dict,
    )


def test_phase_transition_like_cityflow():
    """A phase change first runs the transition phase for `yellow_phase_time` steps,
    actions meanwhile are ignored and `current_phase` is still the OLD phase."""
    inter = _inter(yellow_phase_time=3, yellow_dict={"0_1": 5})
    seen = []
    for action in [0, 0, 1, 1, 0, 1, 1]:  # the `0` mid-transition must be ignored
        cityflow_transition(inter, action)
        seen.append((inter.current_phase, inter.current_phase_time))
    assert inter.eng.trafficlight.calls == [0, 0, 5, 5, 5, 1, 1]
    assert seen == [(0, 1), (0, 2), (0, 1), (0, 2), (0, 3), (1, 4), (1, 5)]


def test_no_yellow_when_nothing_turns_red_or_in_benchmark_mode():
    for inter in (_inter(3, yellow_dict={}), _inter(0, yellow_dict={"0_1": 5})):
        cityflow_transition(inter, 1)
        assert inter.eng.trafficlight.calls == [1]
        assert (inter.current_phase, inter.current_phase_time, inter.transition) == (1, 1, None)


def test_engine_api_picks_the_sumo_view():
    eng, api = object(), object()
    assert engine_api(SimpleNamespace(eng=eng)) is eng
    assert engine_api(SimpleNamespace(eng=eng, cityflow_api=api)) is api


def test_phase_mask_only_on_a_heterogeneous_network():
    assert phase_mask([8, 8, 8], 8) is None  # JiNan/HangZhou: the GATE path unchanged
    mask = phase_mask([2, 7, 3], 7)
    assert mask.shape == (3, 7) and mask.dtype == bool
    assert mask.sum(axis=1).tolist() == [2, 7, 3]
    assert mask[0].tolist() == [True, True, False, False, False, False, False]


def _grid(tmp_path: Path, length: float) -> str:
    """A 3x3 grid with a signal at every node, where neighbours are known."""
    net = tmp_path / "grid.net.xml"
    subprocess.run(
        [
            checkBinary("netgenerate"),
            "--grid",
            "--grid.number", "3",
            "--grid.length", str(length),
            "-j", "traffic_light",
            "--no-turnarounds",
            "-o", str(net),
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip
    return str(net)


def test_sumo_graph_on_a_grid(tmp_path):
    """3x3: 12 neighbourhoods both ways = 24 edges; a corner has 2 in-edges, a side 3,
    the centre 4. With a reach shorter than a segment, none."""
    net = _grid(tmp_path, length=200)
    g = RoadNetGraph._from_libsignal(build_graph(net))  # also checks the RoadNetGraph contract
    assert (g.n_nodes, g.n_edges) == (9, 24)
    assert sorted(g.degrees) == [2, 2, 2, 2, 3, 3, 3, 3, 4]
    assert build_graph(net, reach_m=150)["sparse_adj"].shape == (0, 2)


def test_sumo_graph_on_krakow_instead_of_libsignals():
    """LibSignal's graph on an OSM network: 1 edge for 25 nodes, a GAT without neighbours."""
    net = paths.DATA / "raw_data" / "krakow_3w" / "3w.net.xml"
    if not net.is_file():
        pytest.skip(NO_KRAKOW)
    ours = build_graph(str(net))
    upstream = build_index_intersection_map_sumo(str(net))
    assert ours["node_idx2id"] == upstream["node_idx2id"]  # same nodes, same order
    g = RoadNetGraph._from_libsignal(ours)
    assert g.n_edges >= g.n_nodes > 0
    assert len(upstream["sparse_adj"]) < g.n_edges // 10
    assert len({tuple(e) for e in ours["sparse_adj"].tolist()}) == g.n_edges  # no duplicates


@pytest.mark.slow
def test_core_on_the_krakow_sumo_world():
    """A real SUMO world + the GAT+PPO core on Krakow (25 signals, 2-7 phases)."""
    if not (paths.DATA / "raw_data" / "krakow_3w" / "3w_mid.rou.xml").is_file():
        pytest.skip(NO_KRAKOW)
    out = subprocess.run(
        [sys.executable, str(PROBE), NETWORK],
        env=child_env(),
        cwd=paths.REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    r = json.loads(out.stdout.strip().splitlines()[-1])

    assert r["world_class"] == "tsc_marl.libsignal.sumo_world.World"
    assert r["identity_kept"], "reset() replaced the Intersection objects"
    assert r["heterogeneous"], "Krakow has 2-7 phases, the mask should exist"
    assert r["permissive_links"] > 0, "permissive green `g` added no movement"
    assert r["lanes_known"], "controlled movement lanes without a length in lane_length"
    assert r["actions_valid"], "the policy chose a phase the intersection lacks"
    assert r["one_hot_matches_phase"], "the phase one-hot disagrees with inter.current_phase"
    assert r["transition_after_start"], "a phase change without transition after the start"
    assert r["att"] > 0 and r["throughput"] > 0
