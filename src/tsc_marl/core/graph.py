"""Road-network graph shared by the state builder, the networks and the loop.

A node is a signalised intersection (an agent), an edge is a road between two
of them. The roadnet is parsed by LibSignal (`build_index_intersection_map_cityflow`),
the same code CoLight uses; this module wraps the result and enforces its contract.

Design decisions:

1. Neighbours are topological (connected by an incoming road), not the k nearest
   intersections as in CoLight (Wei et al., 2019).
2. Edges keep the roadnet direction (row 0 = source, row 1 = target), so a node
   aggregates from its upstream intersections.
3. `edge_index` has no self-loops. The GAT layer adds them; a second loop here
   would silently double a node's own weight.

Order contract: row i of the observation, node i of the graph and position i of
the action and reward vectors are the same intersection. The simulator orders
`world.intersections` on its own, so every access to the world goes through
`order_intersections()`. A mismatch raises nothing, the network just learns
from the wrong rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:  # type-only: the submodule is importable only after `libsignal.bootstrap()`
    from world.world_cityflow import Intersection, World


# eq=False on purpose: `edge_index` is a tensor, and a generated `__eq__` would
# compare it with `==`, which returns a tensor, not a bool.
@dataclass(frozen=True, eq=False)
class RoadNetGraph:
    """Static network topology, frozen so that nothing mutates it mid-episode.

    Depends only on the roadnet and torch, nothing from training, so the state
    builder, the networks and the wrappers use it independently.
    """

    n_nodes: int
    node_idx2id: tuple[str, ...]
    node_id2idx: Mapping[str, int]
    edge_index: torch.Tensor  # [2, E] long; row 0 = source, row 1 = target; no self-loops
    degrees: tuple[int, ...]  # number of incoming roads
    neighbors: tuple[tuple[int, ...], ...]  # upstream neighbours plus the node itself, ascending

    @classmethod
    def from_roadnet(cls, path: str | Path) -> RoadNetGraph:
        """Build from a roadnet file, without an engine or `Registry`.

        The entry point for tests and offline analysis: the graph is checked in
        milliseconds instead of waiting for CityFlow. Requires the submodule on
        `sys.path` (`libsignal.bootstrap()`), like every other import from it.
        """
        from common.utils import build_index_intersection_map_cityflow

        return cls._from_libsignal(build_index_intersection_map_cityflow(str(path)))

    @classmethod
    def from_registry(cls) -> RoadNetGraph:
        """Build from the graph LibSignal computed at the start of the run
        (`Graph_World_Interface`): same content as `from_roadnet`, parsed once.
        """
        from common.registry import Registry

        return cls._from_libsignal(Registry.mapping["world_mapping"]["graph_setting"].graph)

    @classmethod
    def _from_libsignal(cls, graph: Mapping[str, Any]) -> RoadNetGraph:
        """Translate LibSignal's dict into this contract.

        `sparse_adj` can be an empty list rather than an array (a network without
        valid roads), hence the explicit `reshape`.
        """
        node_idx2id = graph["node_idx2id"]
        n_nodes = len(node_idx2id)
        ids = tuple(node_idx2id[i] for i in range(n_nodes))

        adj = torch.as_tensor(graph["sparse_adj"], dtype=torch.long).reshape(-1, 2)
        edge_index = adj.t().contiguous()  # [E, 2] -> [2, E]

        # Neighbourhoods come from `edge_index`, not from LibSignal's `node_list`,
        # so there is one source of truth: what the GAT layer sees. A test checks
        # that both agree.
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
        """The contract is enforced here, not only described."""
        if len(self.node_idx2id) != self.n_nodes:
            raise ValueError(
                f"node_idx2id has {len(self.node_idx2id)} entries, n_nodes={self.n_nodes}"
            )
        if len(self.degrees) != self.n_nodes or len(self.neighbors) != self.n_nodes:
            raise ValueError("degrees/neighbors need exactly one entry per node")
        if any(self.node_id2idx[nid] != i for i, nid in enumerate(self.node_idx2id)):
            raise ValueError("node_id2idx and node_idx2id are not mutually inverse")
        if self.edge_index.dtype != torch.long or self.edge_index.dim() != 2:
            raise ValueError(
                f"edge_index must be long [2, E], got {self.edge_index.dtype} "
                f"{tuple(self.edge_index.shape)}"
            )
        if self.edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must have 2 rows (source, target), got {self.edge_index.shape[0]}"
            )
        if self.n_edges and (self.edge_index.min() < 0 or self.edge_index.max() >= self.n_nodes):
            raise ValueError("edge_index points outside the node range")
        if bool((self.edge_index[0] == self.edge_index[1]).any()):
            raise ValueError("edge_index contains a self-loop; GAT adds them (decision 3)")

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @staticmethod
    def bare_id(intersection_id: str) -> str:
        """World intersection id -> graph node id.

        On SUMO, LibSignal's graph builder strips the `GS_` prefix (signals
        guessed by netconvert) while `world_sumo` keeps it in `inter.id`. The rule
        is copied from that builder character for character, including matching
        `GS_` anywhere in the id, because what matters is agreement with the
        graph. On CityFlow this is the identity.
        """
        return intersection_id if "GS_" not in intersection_id else intersection_id[3:]

    def order_intersections(self, world: World) -> list[Intersection]:
        """`world.intersections` in canonical `node_idx` order.

        The only allowed way into the world when building observations, rewards
        and actions (see ORDER CONTRACT in the module docstring).
        """
        return self.order(world.intersections, key=lambda inter: self.bare_id(inter.id))

    def order(self, items: Sequence[Any], key=lambda x: x) -> list[Any]:
        """Arrange any collection keyed by intersection id in `node_idx` order."""
        by_id = {key(item): item for item in items}
        missing = [nid for nid in self.node_idx2id if nid not in by_id]
        if missing:
            raise KeyError(f"graph nodes missing from collection: {missing[:5]} ({len(missing)})")
        return [by_id[nid] for nid in self.node_idx2id]

    def __repr__(self) -> str:
        return f"RoadNetGraph(n_nodes={self.n_nodes}, n_edges={self.n_edges})"
