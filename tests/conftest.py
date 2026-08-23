"""Wspólna infrastruktura testów.

Testy ATS nie potrzebują silnika: `Intersection` z LibSignala parsuje roadnet
sam, a jedyne, czego dotyka na zewnątrz, to `world.eng.set_tl_phase` w `reset()`.
Budujemy więc PRAWDZIWY obiekt `Intersection` na atrapie świata — dzięki temu
test sprawdza kod, który naprawdę biegnie w symulacji, a nie własną kopię
parsowania roadnetu, i chodzi w ~0,1 s zamiast stawiać CityFlow.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tsc_marl import libsignal

libsignal.bootstrap()  # ścieżka do submodułu + SUMO_HOME (world/__init__ ciągnie world_sumo)

ROADNETS = {
    "jinan3x4": libsignal.DATA / "raw_data" / "jinan_3x4" / "roadnet_3_4_4phase.json",
    "hangzhou4x4": libsignal.DATA
    / "raw_data"
    / "hangzhou_4x4_advanced"
    / "roadnet_4_4_4phase.json",
}


class _FakeEngine:
    def set_tl_phase(self, *args):
        pass


class _FakeWorld:
    RIGHT = True

    def __init__(self):
        self.eng = _FakeEngine()


def load_intersections(roadnet: Path):
    """(obiekt Intersection, surowy dict z roadnetu) dla każdego niewirtualnego skrzyżowania."""
    from world.world_cityflow import Intersection  # noqa: PLC0415 - dopiero po bootstrapie

    raw = json.loads(roadnet.read_text())
    world = _FakeWorld()
    return [(Intersection(d, world), d) for d in raw["intersections"] if not d["virtual"]]


@pytest.fixture(scope="session", params=sorted(ROADNETS))
def roadnet(request):
    path = ROADNETS[request.param]
    if not path.is_file():
        pytest.skip(f"brak roadnetu {path} (dane pobiera się osobno — patrz data/README.md)")
    return load_intersections(path)
