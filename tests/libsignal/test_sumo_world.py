"""Adapter SUMO (`libsignal/sumo_world.py`) — parytet backendu z CityFlow.

Szybkie testy sprawdzają czyste funkcje adaptera na atrapach. Jeden wolny stawia
prawdziwy świat SUMO na sieci Krakowa w OSOBNYM procesie (`sumo_probe.py`): LibSignal
trzyma konfigurację w globalnym Registry, a świat CityFlow z `conftest.py` żyje przez
całą sesję — drugi świat w tym samym procesie nadpisałby mu `graph_setting`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tsc_marl import paths
from tsc_marl.core.features import engine_api, phase_mask
from tsc_marl.libsignal.bridge import child_env
from tsc_marl.libsignal.sumo_world import cityflow_transition, serving_lanelinks, travel_time

PROBE = Path(__file__).with_name("sumo_probe.py")
NETWORK = "sumo_krakow3w_mid"


def test_serving_lanelinks_bierze_zielone_warunkowe():
    """Upstream bierze tylko `G` i `s` — skręt z ustąpieniem (`g`) wypadał z faz."""
    links = [[("a_0", "x_0", ":v0")], [("a_1", "y_0", ":v1")], [], [("b_0", "x_0", ":v2")]]
    inter = SimpleNamespace(
        lanelinks=links,
        green_phases=[SimpleNamespace(state="GgrG"), SimpleNamespace(state="rrGs")],
    )
    lanelinks, starts = serving_lanelinks(inter)
    assert lanelinks[0] == [["a_0", "x_0"], ["a_1", "y_0"], ["b_0", "x_0"]]  # `g` na indeksie 1
    assert lanelinks[1] == [["b_0", "x_0"]]  # `G` na indeksie bez linku odpada, `s` zostaje
    assert starts == [["a_0", "a_1", "b_0"], ["b_0"]]


def test_serving_lanelinks_ucina_jak_upstream():
    """Stan fazy dłuższy niż lista linków — upstream robi `break`, my `zip`."""
    inter = SimpleNamespace(
        lanelinks=[[("a_0", "x_0", ":v")]], green_phases=[SimpleNamespace(state="GG")]
    )
    assert serving_lanelinks(inter)[0] == [[["a_0", "x_0"]]]


def test_travel_time_liczy_tez_jadace_jak_cityflow():
    """`Engine::getAverageTravelTime`: dojechane + jadące od wjazdu. Upstream
    liczyłby tu 150 (same dojechane) i nagradzał metodę, która blokuje ruch."""
    finished = {"v1": 100.0, "v2": 200.0}
    entered = {"v1": 0.0, "v2": 50.0, "v3": 300.0, "v4": 500.0}
    assert travel_time(finished, entered, now=700.0) == (100 + 200 + 400 + 200) / 4
    assert travel_time({}, {}, now=10.0) == 0.0


class _Lights:
    def __init__(self):
        self.calls: list[int] = []

    def setPhase(self, _tl_id, index):  # noqa: N802 - nazwa z API SUMO
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


def test_przejscie_faz_jak_cityflow():
    """Zmiana fazy = najpierw faza przejściowa przez `yellow_phase_time` kroków, akcje
    w tym czasie ignorowane, `current_phase` to wciąż STARA faza. Upstream po
    pierwszych sekundach epizodu przełączał od razu, bez przejścia."""
    inter = _inter(yellow_phase_time=3, yellow_dict={"0_1": 5})
    seen = []
    for action in [0, 0, 1, 1, 0, 1, 1]:  # `0` w środku przejścia ma zostać zignorowane
        cityflow_transition(inter, action)
        seen.append((inter.current_phase, inter.current_phase_time))
    assert inter.eng.trafficlight.calls == [0, 0, 5, 5, 5, 1, 1]
    assert seen == [(0, 1), (0, 2), (0, 1), (0, 2), (0, 3), (1, 4), (1, 5)]


def test_przejscie_bez_zoltego_gdy_nic_nie_gasnie_albo_tryb_benchmark():
    for inter in (_inter(3, yellow_dict={}), _inter(0, yellow_dict={"0_1": 5})):
        cityflow_transition(inter, 1)
        assert inter.eng.trafficlight.calls == [1]
        assert (inter.current_phase, inter.current_phase_time, inter.transition) == (1, 1, None)


def test_engine_api_wybiera_widok_sumo():
    eng, api = object(), object()
    assert engine_api(SimpleNamespace(eng=eng)) is eng
    assert engine_api(SimpleNamespace(eng=eng, cityflow_api=api)) is api


def test_phase_mask_tylko_na_sieci_niejednorodnej():
    assert phase_mask([8, 8, 8], 8) is None  # JiNan/HangZhou — ścieżka GATE bez zmian
    mask = phase_mask([2, 7, 3], 7)
    assert mask.shape == (3, 7) and mask.dtype == bool
    assert mask.sum(axis=1).tolist() == [2, 7, 3]
    assert mask[0].tolist() == [True, True, False, False, False, False, False]


def _grid(tmp_path: Path, length: float) -> str:
    """Siatka 3×3 z sygnalizacją na każdym węźle — tu wiadomo, kto jest czyim sąsiadem."""
    from sumolib import checkBinary  # noqa: PLC0415

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


def test_graf_sumo_na_siatce(tmp_path):
    """3×3: 12 sąsiedztw w obie strony = 24 krawędzie; róg ma 2 wjazdy, bok 3,
    środek 4. Z zasięgiem krótszym niż odcinek — ani jednej."""
    from tsc_marl.core.graph import RoadNetGraph  # noqa: PLC0415
    from tsc_marl.libsignal.sumo_world import build_graph  # noqa: PLC0415

    net = _grid(tmp_path, length=200)
    g = RoadNetGraph._from_libsignal(build_graph(net))  # kontrakt RoadNetGraph też sprawdza
    assert (g.n_nodes, g.n_edges) == (9, 24)
    assert sorted(g.degrees) == [2, 2, 2, 2, 3, 3, 3, 3, 4]
    assert build_graph(net, reach_m=150)["sparse_adj"].shape == (0, 2)


def test_graf_sumo_na_krakowie_zamiast_grafu_libsignala():
    """Graf LibSignala na sieci z OSM: 1 krawędź na 25 węzłów, czyli GAT bez sąsiadów."""
    from common.utils import build_index_intersection_map_sumo  # noqa: PLC0415

    from tsc_marl.core.graph import RoadNetGraph  # noqa: PLC0415
    from tsc_marl.libsignal.sumo_world import build_graph  # noqa: PLC0415

    net = paths.DATA / "raw_data" / "krakow_3w" / "3w.net.xml"
    if not net.is_file():
        pytest.skip("brak scenariusza Krakowa — .venv/bin/python data/raw_data/krakow_3w/build.py")
    ours = build_graph(str(net))
    upstream = build_index_intersection_map_sumo(str(net))
    assert ours["node_idx2id"] == upstream["node_idx2id"]  # te same węzły, ta sama kolejność
    g = RoadNetGraph._from_libsignal(ours)
    assert g.n_edges >= g.n_nodes > 0
    assert len(upstream["sparse_adj"]) < g.n_edges // 10
    assert len({tuple(e) for e in ours["sparse_adj"].tolist()}) == g.n_edges  # bez duplikatów


@pytest.mark.slow
def test_rdzen_na_swiecie_sumo_krakowa():
    """Prawdziwy świat SUMO + rdzeń GAT+PPO na Krakowie (25 sygnalizacji, 2–7 faz)."""
    if not (paths.DATA / "raw_data" / "krakow_3w" / "3w_mid.rou.xml").is_file():
        pytest.skip("brak scenariusza Krakowa — .venv/bin/python data/raw_data/krakow_3w/build.py")
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
    assert r["identity_kept"], "reset() podmienił obiekty Intersection"
    assert r["heterogeneous"], "Kraków ma 2–7 faz — maska powinna istnieć"
    assert r["permissive_links"] > 0, "zielone warunkowe `g` nie dołożyło żadnego ruchu"
    assert r["lanes_known"], "pasy ruchów sterowanych bez długości w lane_length"
    assert r["actions_valid"], "polityka wybrała fazę, której skrzyżowanie nie ma"
    assert r["one_hot_matches_phase"], "one-hot fazy nie zgadza się z inter.current_phase"
    assert r["transition_after_start"], "po starcie epizodu zmiana fazy bez przejścia"
    assert r["att"] > 0 and r["throughput"] > 0
