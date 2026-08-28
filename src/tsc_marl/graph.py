"""Graf sieci drogowej — kanoniczna topologia dla rdzenia Fazy 1 (GAT + PPO + CTDE).

Węzeł = sterowane skrzyżowanie (agent), krawędź = droga między dwoma sterowanymi
skrzyżowaniami. Parsowania roadnetu NIE powtarzamy: robi je LibSignal
(`common/utils.py:build_index_intersection_map_cityflow`) i to jego wynik czyta
`agent/colight.py`, czyli implementacja, do której liczb się porównujemy. Ten
moduł tylko opakowuje ten wynik w zamrożony obiekt i pilnuje kontraktu.

TRZY DECYZJE PROJEKTOWE zapisane tutaj, żeby za pół roku nie zgadywać, czy to był
wybór, czy niedopatrzenie:

1. SĄSIEDZTWO TOPOLOGICZNE, nie geo top-k. Sąsiad = skrzyżowanie połączone drogą
   wjazdową, plus self. CoLight ([[wei-2019-colight]]) bierze k najbliższych
   euklidesowo, więc ma zawsze |N_i| = 5; my mamy |N_i| ∈ {3, 4, 5}, bo róg siatki
   ma dwóch sąsiadów. Na regularnej siatce dla węzłów wewnętrznych oba warianty
   dają ten sam zbiór, a topologiczny jest index-free i przenosi się na
   nieregularną sieć (Faza 5). Wariant geo to slot ablacyjny 1.6.

2. KIERUNEK KRAWĘDZI bez odwracania. `sparse_adj` to pary [start, end] drogi, więc
   po transpozycji dostajemy konwencję PyG: wiersz 0 = źródło, wiersz 1 = cel, a
   agregacja leci DO celu. Węzeł zbiera zatem od tych, którzy w niego wjeżdżają
   (upstream). Tak robi `agent/colight.py`; stary `agent/colight_pytorch_agent.py`
   robił `torch.flip`, czyli odwrotnie. Na naszych sieciach zbiór krawędzi jest
   symetryczny, więc różnica dotyczy wyłącznie tego, kto dostaje W_target, a kto
   W_source w scorze attention — ale attention jest asymetryczna z premedytacją,
   więc konwencja musi być jedna i jawna.

3. edge_index BEZ PĘTLI WŁASNYCH. Dokłada je warstwa GAT (`add_self_loops`
   w `agent/colight.py`). Trzymanie ich również tutaj dałoby podwójną pętlę
   własną — błąd, który niczego nie wywala, tylko po cichu przesuwa wagi.

KONTRAKT KOLEJNOŚCI (najważniejsze w tym module): wiersz i tensora obserwacji
⟺ `node_idx == i` ⟺ `node_idx2id[i]` ⟺ pozycja i w wektorze akcji i nagród.

Na roadnetach natywnych CityFlow `world.intersections` ma DZIŚ dokładnie tę samą
kolejność (oba filtry przechodzą tę samą listę z JSON-a, odsiewając `virtual`),
więc `order_intersections()` jest w tej chwili no-opem. Nie wolno na tym polegać:
`world_cityflow.py:258` odsiewa wirtualne również po kluczu `cf_gt_virtual`,
którego `build_index_intersection_map_cityflow` w ogóle nie zna — na sieciach
konwertowanych z SUMO (Faza 5) oba filtry mogą więc zwrócić inny ZBIÓR węzłów,
nie tylko inną kolejność. Przepuszczaj wszystko przez `order_intersections()`;
kosztuje jeden słownik, a ratuje przed błędem, który niczego nie wywala:
obserwacja jednego skrzyżowania trafia w wiersz drugiego i sieć uczy się gorzej.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:  # tylko dla typów — inaczej moduł nie zaimportuje się bez
    from world.world_cityflow import Intersection, World  # bootstrapu ścieżki do submodułu


# eq=False celowo: pole `edge_index` jest tensorem, a wygenerowane `__eq__`
# porównywałoby je operatorem `==`, który zwraca tensor, nie bool ("truth value of
# an array with more than one element is ambiguous"). Graf porównuje się przez
# pola skalarne, nie przez cały obiekt.
@dataclass(frozen=True, eq=False)
class RoadNetGraph:
    """Statyczna topologia sieci. Zamrożona, bo graf nie zmienia się w trakcie
    biegu (LibSignal ma w tym miejscu `# TODO: support dynamic graph later`),
    a zamrożenie kupuje pewność, że nikt go nie zmutuje w połowie epizodu.

    Zależy WYŁĄCZNIE od roadnetu i torcha — nic z `agents/` ani z treningu. Dzięki
    temu 1.3 (encoder), 1.4 (krytyk) i Faza 3 (fallback potrzebuje sąsiedztwa,
    żeby wiedzieć, czyje obserwacje odpadły) używają go niezależnie od siebie.
    """

    n_nodes: int
    node_idx2id: tuple[str, ...]
    node_id2idx: Mapping[str, int]
    edge_index: torch.Tensor  # [2, E], long; wiersz 0 = źródło, 1 = cel; bez self-loopów
    degrees: tuple[int, ...]  # liczba dróg WJAZDOWYCH, czyli |N_i| - 1
    neighbors: tuple[tuple[int, ...], ...]  # sąsiedzi ZE self, rosnąco

    # ------------------------------------------------------------------ budowa

    @classmethod
    def from_roadnet(cls, path: str | Path) -> RoadNetGraph:
        """Zbuduj z pliku roadnetu, bez stawiania silnika i bez `Registry`.

        To jest wejście dla testów: graf da się sprawdzić w milisekundach, zamiast
        czekać ~15 s na CityFlow. Bootstrap jest tutaj, bo `common/utils.py`
        importuje `world` i `sumolib` na poziomie modułu.
        """
        from tsc_marl import libsignal  # noqa: PLC0415 - unikamy cyklu przy imporcie pakietu

        libsignal.bootstrap()
        from common.utils import build_index_intersection_map_cityflow  # noqa: PLC0415

        return cls._from_libsignal(build_index_intersection_map_cityflow(str(path)))

    @classmethod
    def from_registry(cls) -> RoadNetGraph:
        """Zbuduj z grafu, który LibSignal policzył już przy starcie biegu
        (`Graph_World_Interface` w `common/interface.py`). Ta sama treść co
        `from_roadnet`, ale bez drugiego parsowania roadnetu.
        """
        from common.registry import Registry  # noqa: PLC0415 - dopiero po bootstrapie

        return cls._from_libsignal(Registry.mapping["world_mapping"]["graph_setting"].graph)

    @classmethod
    def _from_libsignal(cls, graph: Mapping[str, Any]) -> RoadNetGraph:
        """Przepisz dict LibSignala na nasz kontrakt.

        `sparse_adj` bywa pustą listą, a nie tablicą (sieć bez ważnych dróg), stąd
        jawne `reshape` zamiast polegania na kształcie z numpy.
        """
        node_idx2id = graph["node_idx2id"]
        n_nodes = len(node_idx2id)
        ids = tuple(node_idx2id[i] for i in range(n_nodes))

        adj = torch.as_tensor(graph["sparse_adj"], dtype=torch.long).reshape(-1, 2)
        edge_index = adj.t().contiguous()  # [E, 2] -> [2, E]: wiersz 0 źródło, 1 cel

        # Sąsiedztwo liczymy z `edge_index`, nie z gotowego `node_list`, żeby mieć
        # JEDNO źródło prawdy: to, co zobaczy warstwa GAT po `add_self_loops`.
        # Zgodność obu jest sprawdzana testem.
        incoming: list[set[int]] = [{i} for i in range(n_nodes)]
        for src, dst in edge_index.t().tolist():
            incoming[dst].add(src)

        return cls(
            n_nodes=n_nodes,
            node_idx2id=ids,
            node_id2idx=MappingProxyType(dict(graph["node_id2idx"])),
            edge_index=edge_index,
            degrees=tuple(int(d) for d in graph["node_degrees"]),
            neighbors=tuple(tuple(sorted(n)) for n in incoming),
        )

    def __post_init__(self) -> None:
        """Kontrakt egzekwowany, nie tylko opisany w docstringu."""
        if len(self.node_idx2id) != self.n_nodes:
            raise ValueError(
                f"node_idx2id ma {len(self.node_idx2id)} pozycji, n_nodes={self.n_nodes}"
            )
        if len(self.degrees) != self.n_nodes or len(self.neighbors) != self.n_nodes:
            raise ValueError("degrees/neighbors muszą mieć po jednej pozycji na węzeł")
        if any(self.node_id2idx[nid] != i for i, nid in enumerate(self.node_idx2id)):
            raise ValueError("node_id2idx i node_idx2id nie są wzajemnie odwrotne")
        if self.edge_index.dtype != torch.long or self.edge_index.dim() != 2:
            raise ValueError(
                f"edge_index musi być long [2, E], jest {self.edge_index.dtype} "
                f"{tuple(self.edge_index.shape)}"
            )
        if self.edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index musi mieć 2 wiersze (źródło, cel), ma {self.edge_index.shape[0]}"
            )
        if self.n_edges and (self.edge_index.min() < 0 or self.edge_index.max() >= self.n_nodes):
            raise ValueError("edge_index wskazuje poza zakres węzłów")
        if bool((self.edge_index[0] == self.edge_index[1]).any()):
            raise ValueError(
                "edge_index zawiera pętlę własną — dokłada je warstwa GAT (patrz decyzja 3)"
            )

    # ------------------------------------------------------------------ użycie

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def order_intersections(self, world: World) -> list[Intersection]:
        """`world.intersections` w kanonicznej kolejności `node_idx`.

        JEDYNE dopuszczalne wejście do świata przy budowie obserwacji, nagród
        i akcji — patrz KONTRAKT KOLEJNOŚCI w docstringu modułu. Ten sam wzorzec,
        który `agent/colight.py` powtarza ręcznie przy każdym generatorze.
        """
        return self.order(world.intersections, key=lambda inter: inter.id)

    def order(self, items: Sequence[Any], key=lambda x: x) -> list[Any]:
        """Ustaw dowolną kolekcję zaadresowaną id skrzyżowań w kolejności `node_idx`."""
        by_id = {key(item): item for item in items}
        missing = [nid for nid in self.node_idx2id if nid not in by_id]
        if missing:
            raise KeyError(f"brak w kolekcji skrzyżowań grafu: {missing[:5]} (z {len(missing)})")
        return [by_id[nid] for nid in self.node_idx2id]

    def __repr__(self) -> str:
        return f"RoadNetGraph(n_nodes={self.n_nodes}, n_edges={self.n_edges})"
