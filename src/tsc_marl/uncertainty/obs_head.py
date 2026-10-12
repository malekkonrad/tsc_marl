"""Heteroscedastic OBSERVATION head: uncertainty of the input features.

Decision uncertainty (a Dirichlet over phases, `prototype.py`) does not estimate
observation uncertainty: outside the training range it turns back, and it does
not see drift. This head works on the INPUT FEATURES, per movement, aggregated
to graph edges.

Input: a single reading of 7 instead of 4 is perfectly plausible; corruption
shows only as INCONSISTENCY. So the head sees (1) a window of the node's last
readings (a queue cannot grow faster than the saturation flow; under red it can
only grow, and the phase is in the one-hot), (2) optionally the mean of upstream
neighbours' readings (flow conservation), and (3) optionally persistence
features (how long a value has not changed, how often it is exactly zero).

Output: for each movement feature (EP and demand blocks) the mean mu and scale
sigma of the TRUE value; Gaussian NLL against `x_clean`, which only the
simulator knows. Self-calibrating: a too small sigma is punished by the squared
error, a too large one by log sigma.

Reading: u of a movement = max sigma over its features (in units of the
feature's clean standard deviation); u of edge j->i = max or mean over the
movements of i whose incoming road starts at j (or over all of j's movements,
`side="source"`). Detection = AUROC of u on CORRUPTED cells vs all CLEAN cells.

    tsc obshead            # -> results/gate_2_2.md
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from tsc_marl.uncertainty.prototype import TRAIN_SCENARIOS, Loaded, load

#: Reading error (vehicles, per feature) above which a cell counts as corrupted.
#: A dead detector is ALWAYS corrupted, even when it happens to report the truth
#: (0 on an empty lane): localisation ground truth comes from the failure layer.
CORRUPT_TAU = 0.5
UNSEEN = ("gauss3p5", "gauss4", "urand3p5", "mask0p5", "mask0p25_intersection")


@dataclass(frozen=True)
class ObsConfig:
    window: int = 4
    neighbors: bool = True
    #: PERSISTENCE features over the window: the fraction a feature stood still and the
    #: fraction it was exactly zero. A dead detector leaves its trace in time (stuck),
    #: not in a single value; zero is the modal value on clean traffic.
    persistence: bool = False
    hidden: int = 128
    epochs: int = 60
    lr: float = 1e-3
    batch: int = 512
    split: float = 0.67  # split BY TIME, as in `prototype.py`
    train_scenarios: tuple[str, ...] = TRAIN_SCENARIOS
    seed: int = 0
    min_sigma: float = 0.02  # lower bound of sigma, in standardised units


# ---------------------------------------------------------------- inputs


def window_index(episode: np.ndarray, k: int) -> np.ndarray:
    """[T, k]: indices t, t-1, ..., t-k+1, clipped to the EPISODE START (after
    `world.reset()` the previous episode's readings describe other traffic)."""
    t = np.arange(len(episode))
    first: dict[int, int] = {}
    for i, e in enumerate(episode.tolist()):
        first.setdefault(e, i)
    start = np.array([first[e] for e in episode.tolist()])
    return np.maximum(t[:, None] - np.arange(k)[None, :], start[:, None])


def full_window(episode: np.ndarray, k: int) -> np.ndarray:
    """[T] bool: steps whose window `k` FITS within the episode.

    During warm-up `window_index` repeats the episode start, so every feature
    "does not change" and every movement looks suspicious. For threshold
    calibration this matters: the clean tail fills with ones and the 99.5%
    quantile becomes 1.0, a threshold that never fires. The gate in the control
    loop follows the same rule (a full buffer).
    """
    t = np.arange(len(episode))
    first: dict[int, int] = {}
    for i, e in enumerate(episode.tolist()):
        first.setdefault(e, i)
    start = np.array([first[e] for e in episode.tolist()])
    return (t - start) >= (k - 1)


def neighbor_matrix(edge_index: np.ndarray, n: int) -> np.ndarray:
    """[N, N]: row i averages the nodes with an edge INTO i (upstream); a node
    without neighbours gets zeros."""
    a = np.zeros((n, n), dtype=np.float32)
    a[edge_index[1], edge_index[0]] = 1.0
    deg = a.sum(1, keepdims=True)
    return np.divide(a, deg, out=np.zeros_like(a), where=deg > 0)


def persistence_features(x: np.ndarray, win: np.ndarray, n_feat: int) -> np.ndarray:
    """[T, N, 2*n_feat]: per movement feature, the fraction of window steps WITHOUT
    CHANGE and the fraction of readings EXACTLY ZERO. At episode start the window is
    clipped, so "no change" grows artificially, equally on clean and corrupted input."""
    xs = np.stack([x[win[:, j]][..., :n_feat] for j in range(win.shape[1])])  # [k, T, N, f]
    same = (np.abs(np.diff(xs, axis=0)) < 1e-6).mean(0) if len(xs) > 1 else np.zeros_like(xs[0])
    zero = (np.abs(xs) < 1e-6).mean(0)
    return np.concatenate([same, zero], -1)


def build_inputs(
    x: np.ndarray, win: np.ndarray, adj: np.ndarray | None, n_persist: int = 0
) -> np.ndarray:
    """[T, N, F] -> [T, N, k*F (+F) (+2*n_persist)]: window of own readings, mean of
    upstream neighbours and (optionally) persistence of the first `n_persist` features."""
    parts = [x[win[:, j]] for j in range(win.shape[1])]
    if adj is not None:
        parts.append(np.einsum("ij,tjf->tif", adj, x))
    if n_persist:
        parts.append(persistence_features(x, win, n_persist))
    return np.concatenate(parts, -1).astype(np.float32)


# ------------------------------------------------------------- geometry


def _intersections(roadnet: Path):
    """LibSignal `Intersection` objects from the file alone, without an engine (the
    trick of `tests/conftest.py`): movements are parsed by the simulation's code."""
    from tsc_marl import libsignal

    libsignal.bootstrap()
    from world.world_cityflow import Intersection

    class _Eng:
        def set_tl_phase(self, *a):
            pass

    class _World:
        RIGHT = True
        eng = _Eng()

    raw = json.loads(roadnet.read_text())
    return raw, {d["id"]: Intersection(d, _World()) for d in raw["intersections"]}


def roadnet_of(network: str) -> Path:
    from tsc_marl.paths import CONFIGS, REPO_ROOT

    cfg = json.loads((CONFIGS / "sim" / f"{network}.cfg").read_text())
    return REPO_ROOT / "data" / cfg["roadnetFile"]


def movement_sources(network: str, n_mov: int, drop_always_green: bool = True) -> np.ndarray:
    """[N, n_mov]: index of the node a movement comes from; -1 = network border.

    The movement -> edge mapping: the movement's incoming road starts at
    neighbour j, so it belongs to edge j->i. Movement order as in `StateBuilder`
    (`ats.controlled_movements`), node order as in the graph."""
    from tsc_marl import libsignal
    from tsc_marl.core.graph import RoadNetGraph
    from tsc_marl.domain import ats

    libsignal.bootstrap()
    path = roadnet_of(network)
    graph = RoadNetGraph.from_roadnet(path)
    raw, inters = _intersections(path)
    start = {r["id"]: r["startIntersection"] for r in raw["roads"]}
    out = np.full((graph.n_nodes, n_mov), -1, dtype=np.int64)
    for i, nid in enumerate(graph.node_idx2id):
        for m, (ins, _outs) in enumerate(ats.controlled_movements(inters[nid], drop_always_green)):
            out[i, m] = graph.node_id2idx.get(start[ats.road_of(ins[0])], -1)
    return out


# ------------------------------------------------------------------ model


class ObsHead(nn.Module):
    """MLP shared by all nodes: [.., D] -> (mu, sigma) [.., n_feat]."""

    def __init__(self, d_in: int, n_feat: int, hidden: int, min_sigma: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.mu = nn.Linear(hidden, n_feat)
        self.sigma = nn.Linear(hidden, n_feat)
        self.min_sigma = min_sigma

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(z)
        return self.mu(h), torch.nn.functional.softplus(self.sigma(h)) + self.min_sigma


def gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (0.5 * ((y - mu) / sigma) ** 2 + torch.log(sigma)).mean()


@dataclass
class Fitted:
    model: ObsHead
    in_mean: np.ndarray
    in_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    n_feat: int
    adj: np.ndarray | None
    cfg: ObsConfig

    def features(self, data: Loaded, x: np.ndarray) -> np.ndarray:
        win = window_index(data.episode, self.cfg.window)
        adj = neighbor_matrix(data.edge_index, data.n_nodes) if self.cfg.neighbors else None
        n_persist = self.n_feat if self.cfg.persistence else 0
        return (build_inputs(x, win, adj, n_persist) - self.in_mean) / self.in_std

    @torch.no_grad()
    def predict(self, data: Loaded, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """-> (mu, sigma) in RAW feature units, [T, N, n_feat]."""
        self.model.eval()
        z = torch.from_numpy(self.features(data, x).astype(np.float32))
        mu, sigma = self.model(z)
        return mu.numpy() * self.y_std + self.y_mean, sigma.numpy() * self.y_std

    def save(self, path: str | Path, *, network: str) -> Path:
        """Save EVERYTHING inference needs: weights, standardisation and config.

        Weights alone are not enough: sigma leaves the model standardised, and
        only `y_std` turns it into "clean feature deviations", the unit u_ref and
        tau are measured in. Loading weights with another standardisation raises
        nothing and rescales u by a constant. `network` is stored because the head
        BELONGS TO A NETWORK: one trained on JiNan is near chance on HangZhou.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": HEAD_FORMAT,
                "network": network,
                "cfg": asdict(self.cfg),
                "n_feat": int(self.n_feat),
                "in_mean": torch.as_tensor(self.in_mean),
                "in_std": torch.as_tensor(self.in_std),
                "y_mean": torch.as_tensor(self.y_mean),
                "y_std": torch.as_tensor(self.y_std),
                "adj": None if self.adj is None else torch.as_tensor(self.adj),
                "state_dict": self.model.state_dict(),
            },
            path,
        )
        return path


def fit(data: Loaded, cfg: ObsConfig | None = None) -> tuple[Fitted, list[dict]]:
    cfg = cfg or ObsConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    n_feat = data.ob_length - data.n_phases  # EP (+ demand) blocks, no phase one-hot

    win = window_index(data.episode, cfg.window)
    adj = neighbor_matrix(data.edge_index, data.n_nodes) if cfg.neighbors else None
    train = data.mask(cfg.split, train=True)
    n_persist = n_feat if cfg.persistence else 0
    zs = [build_inputs(data.x[s], win, adj, n_persist)[train] for s in cfg.train_scenarios]
    z = np.concatenate(zs).reshape(-1, zs[0].shape[-1])
    y = np.concatenate([data.x_clean[train][..., :n_feat]] * len(zs)).reshape(-1, n_feat)

    # standardise on the CLEAN training set: sigma comes out in "clean feature
    # deviations", comparable between features
    clean_z = build_inputs(data.x_clean, win, adj, n_persist)[train].reshape(-1, z.shape[-1])
    in_mean, in_std = clean_z.mean(0), np.maximum(clean_z.std(0), 1e-3)
    clean_y = data.x_clean[train][..., :n_feat].reshape(-1, n_feat)
    y_mean, y_std = clean_y.mean(0), np.maximum(clean_y.std(0), 1e-3)

    zt = torch.from_numpy(((z - in_mean) / in_std).astype(np.float32))
    yt = torch.from_numpy(((y - y_mean) / y_std).astype(np.float32))
    model = ObsHead(zt.shape[1], n_feat, cfg.hidden, cfg.min_sigma)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(zt))
        losses = []
        for i in range(0, len(order), cfg.batch):
            idx = torch.from_numpy(order[i : i + cfg.batch])
            mu, sigma = model(zt[idx])
            loss = gaussian_nll(mu, sigma, yt[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))
        history.append({"epoch": epoch, "nll": float(np.mean(losses))})
    return Fitted(model, in_mean, in_std, y_mean, y_std, n_feat, adj, cfg), history


# ------------------------------------------------- saving and loading a trained head

#: Artefact format version; bump when the file CONTENT changes, so an old file fails
#: at load instead of half-loading and giving u in other units.
HEAD_FORMAT = 1


def head_path(network: str) -> Path:
    """Default artefact location, one file per network (see `Fitted.save`)."""
    from tsc_marl.paths import RUNS

    return RUNS / "heads" / f"obs_head__{network}.pt"


def load_head(path: str | Path, *, network: str | None = None) -> Fitted:
    """Load a head saved by `Fitted.save`. -> `Fitted` in eval mode.

    With `network` the network is checked; controllers always pass it."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=True)
    if blob.get("format") != HEAD_FORMAT:
        raise ValueError(
            f"artefact format {blob.get('format')!r}, expected {HEAD_FORMAT}; "
            "retrain the head (`tsc obshead --save`)"
        )
    if network is not None and blob["network"] != network:
        raise ValueError(
            f"head trained on {blob['network']!r}, the network is {network!r}; "
            "a head is near chance AUROC on another network (results/gate_2_2.md)"
        )

    raw = dict(blob["cfg"])
    raw["train_scenarios"] = tuple(raw["train_scenarios"])
    cfg = ObsConfig(**raw)
    n_feat = int(blob["n_feat"])
    model = ObsHead(int(blob["in_mean"].shape[0]), n_feat, cfg.hidden, cfg.min_sigma)
    model.load_state_dict(blob["state_dict"])  # strict=True: shapes must match
    model.eval()
    adj = None if blob["adj"] is None else blob["adj"].numpy()
    return Fitted(
        model,
        blob["in_mean"].numpy(),
        blob["in_std"].numpy(),
        blob["y_mean"].numpy(),
        blob["y_std"].numpy(),
        n_feat,
        adj,
        cfg,
    )


# --------------------------------------------- reference level of u on a healthy network

#: How many deviations above the clean mean still count as "normal" u for an edge.
#: Not a tuning knob: chosen once, on the BC dataset, before evaluation.
REFERENCE_K = 2.0
REFERENCE_FORMAT = 1


@dataclass(frozen=True)
class EdgeReference:
    """u an edge shows when NOTHING is broken: `mean + k*sd` over a clean window.

    u does not drop to zero on healthy detectors (a queue is unpredictable by
    itself), and it differs between intersections (sigma scales with load).
    Without subtracting this level `log_rho = -u/tau` would punish an edge for
    being busy, not for a failure, and break the no-op on clean traffic.
    `mean + 2*sd` was chosen over the mean and a quantile because it is the
    quietest on clean traffic and gives the largest failure/clean contrast (the
    comparison table is in the git history of this file). tau scales all of it
    linearly, so the reference and tau are chosen independently.

    ORDER: `u_edge` follows `edges`, i.e. (TARGET i, SOURCE j) as returned by
    `edge_uncertainty`, NOT `graph.edge_index`, which is (source, target) and
    sorted differently; `log_rho` is the only place that permutes one into the other.
    """

    u_edge: np.ndarray  # [E], in `edges` order
    u_self: np.ndarray  # [N], in node order: self-loops
    edges: tuple[tuple[int, int], ...]
    network: str
    k: float
    scenario: str
    head_sha: str
    #: Which end of an edge carries its u (`EDGE_SIDES`); part of the reference,
    #: because target- and source-side u differ for the SAME `edges` and a mix-up
    #: would only shift the threshold.
    side: str = "target"

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            format=np.array(REFERENCE_FORMAT),
            u_edge=self.u_edge,
            u_self=self.u_self,
            edges=np.array(self.edges, dtype=np.int64).reshape(-1, 2),
            network=np.array(self.network),
            k=np.array(self.k),
            scenario=np.array(self.scenario),
            head_sha=np.array(self.head_sha),
            side=np.array(self.side),
        )
        return path


def file_sha(path: str | Path) -> str:
    """Head artefact hash (16 chars): not security, a detector of "u_ref from
    another head than the weights"."""
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def reference_path(network: str, side: str = "target") -> Path:
    """`target` keeps the original file name, read by already computed plans."""
    from tsc_marl.paths import RUNS

    _check_side(side)
    suffix = "" if side == "target" else f"__{side}"
    return RUNS / "heads" / f"u_ref__{network}{suffix}.npz"


def edge_reference(
    fitted: Fitted,
    data: Loaded,
    sources: np.ndarray,
    *,
    network: str,
    head_sha: str,
    k: float = REFERENCE_K,
    scenario: str = "clean",
    side: str = "target",
) -> EdgeReference:
    """Compute the reference on the TRAINING part of the clean window, so the
    no-op check on the episode tail is not a tautology."""
    train = data.mask(fitted.cfg.split, train=True)
    _, sigma = fitted.predict(data, data.x[scenario])
    u_mov = movement_uncertainty(sigma / fitted.y_std, sources.shape[1])  # [T, N, n_mov]
    u_edge, edges = edge_uncertainty(u_mov, sources, "mean", side=side)

    # self-loop: MEAN over all of the node's movements, border ones included
    e, s = u_edge[train], u_mov[train].mean(-1)
    return EdgeReference(
        u_edge=e.mean(0) + k * e.std(0),
        u_self=s.mean(0) + k * s.std(0),
        edges=tuple((int(i), int(j)) for i, j in edges),
        network=network,
        k=float(k),
        scenario=scenario,
        head_sha=head_sha,
        side=side,
    )


def load_reference(
    path: str | Path, *, network: str | None = None, head: str | Path | None = None
) -> EdgeReference:
    """Load a reference; with `head`, check it belongs to THOSE weights."""
    blob = np.load(Path(path), allow_pickle=False)
    if int(blob["format"]) != REFERENCE_FORMAT:
        raise ValueError(f"u_ref format {int(blob['format'])}, expected {REFERENCE_FORMAT}")
    ref = EdgeReference(
        u_edge=blob["u_edge"],
        u_self=blob["u_self"],
        edges=tuple((int(i), int(j)) for i, j in blob["edges"]),
        network=str(blob["network"]),
        k=float(blob["k"]),
        scenario=str(blob["scenario"]),
        head_sha=str(blob["head_sha"]),
        # files from before the `source` variant lack the field: target side
        side=str(blob["side"]) if "side" in blob.files else "target",
    )
    if network is not None and ref.network != network:
        raise ValueError(f"u_ref was computed on {ref.network!r}, the network is {network!r}")
    if head is not None and (sha := file_sha(head)) != ref.head_sha:
        raise ValueError(
            f"u_ref computed with another head (hash {ref.head_sha}, file has {sha}); "
            "u is in units of THAT head, so the threshold means nothing; "
            "recompute: `tsc obshead --save`"
        )
    return ref


# ------------------------------------------------------- u -> log_rho (layer input)

#: Modulation scale: how many units of EXCESS u (above `EdgeReference`) make one nat
#: in the attention logit. Chosen on the BC dataset with a rule written BEFORE
#: evaluation: "scenario gauss2 gives a mean log_rho spread within a softmax group of
#: ln 10" (the most and least reliable neighbours differ tenfold). It came out 0.245
#: and was rounded. The median spread is zero in every scenario, failures included:
#: the mechanism is sparse in time and reacts to spikes above an edge's own norm.
DEFAULT_TAU = 0.25


def log_rho(
    u_edge: np.ndarray,
    u_self: np.ndarray,
    ref: EdgeReference,
    edge_index: torch.Tensor,
    *,
    tau: float = DEFAULT_TAU,
) -> torch.Tensor:
    """u -> log_rho [E + N] in `edge_index` order AFTER `add_self_loops`:

        log_rho = -relu(u - u_ref) / tau

    `relu`: an edge quieter than usual keeps its normal weight instead of a bonus,
    which would be another channel changing behaviour on clean traffic.

    ORDER PITFALL, the reason this function exists: `ref.edges` are pairs
    (TARGET i, SOURCE j) sorted by target, `edge_index` columns are (SOURCE,
    TARGET) in LibSignal's order. On symmetric edge sets a swapped pair raises no
    `KeyError`: the lookup succeeds and puts each edge's uncertainty on the
    opposite edge. Hence an explicit map and a test that sets u on ONE edge.
    Self-loops come last in node order (`add_self_loops`).
    """
    if u_edge.shape != ref.u_edge.shape or u_self.shape != ref.u_self.shape:
        raise ValueError(
            f"u has shapes {u_edge.shape}/{u_self.shape}, the reference "
            f"{ref.u_edge.shape}/{ref.u_self.shape}"
        )
    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must be [2, E], got {tuple(edge_index.shape)}")
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")

    columns = [(int(src), int(dst)) for src, dst in edge_index.t().tolist()]
    position = {(i, j): k for k, (i, j) in enumerate(ref.edges)}
    try:  # (source, target) -> look up u of edge (target, source)
        order = [position[(dst, src)] for src, dst in columns]
    except KeyError as missing:
        raise ValueError(
            f"edge {missing.args[0]!r} of edge_index has no u in the reference "
            f"({len(ref.edges)} edges); u_ref was computed on another network or another "
            "movement set (see `drop_always_green` in `movement_sources`)"
        ) from missing
    if len(set(order)) != len(ref.edges):
        raise ValueError(f"edge_index covers {len(set(order))} of {len(ref.edges)} reference edges")

    excess_e = np.maximum(u_edge - ref.u_edge, 0.0)[order]
    excess_s = np.maximum(u_self - ref.u_self, 0.0)
    out = -np.concatenate([excess_e, excess_s]) / tau
    return torch.from_numpy(out.astype(np.float32))


def suspicious_movements(x: np.ndarray, win: np.ndarray, n_mov: int) -> np.ndarray:
    """[T, N, n_mov] bool: movements whose features ALL stand still for the whole window.

    One criterion: "dead with `zero` substitution" is a special case of "constant",
    and the same catches `hold` (a frozen last reading), which leaves no trace in
    the VALUE. Built-in weakness, not fixable by a threshold: an empty low-traffic
    approach reports zero for many decisions and is HEALTHY. Requires a FULL window;
    the warm-up guard lives in the gate.
    """
    n_feat = (x.shape[-1] // n_mov) * n_mov  # without the phase one-hot
    xs = np.stack([x[win[:, j]][..., :n_feat] for j in range(win.shape[1])])  # [k, T, N, f]
    stale = (xs.max(0) - xs.min(0)) < 1e-6  # [T, N, f]
    blocks = [stale[..., b * n_mov : (b + 1) * n_mov] for b in range(n_feat // n_mov)]
    return np.logical_and.reduce(blocks)  # suspicious = ALL features of the movement stand


def group_spread(log_rho: torch.Tensor, edge_index: torch.Tensor, n_nodes: int) -> float:
    """Mean spread (max - min) of log_rho within a node's SOFTMAX GROUP (incoming
    edges plus its self-loop): the honest measure of how much the mechanism does.
    The level of log_rho says nothing, since softmax ignores a constant per group;
    0 = attention sees no difference, ln 10 = best and worst neighbour differ
    tenfold. tau is chosen on this measure.
    """
    lr = log_rho.detach().numpy()
    target = np.concatenate([edge_index[1].numpy(), np.arange(n_nodes)])
    groups = [lr[target == n] for n in range(n_nodes)]
    return float(np.mean([g.max() - g.min() for g in groups if len(g)]))


class OnlineUncertainty:
    """The head in the control loop: window buffer -> u -> log_rho, one step at a time.

    Offline the head gets a whole episode and windows it with `window_index`; in
    the loop there is only a buffer of recent readings. This class builds EXACTLY
    the offline input from the buffer (newest reading in row 0), which its tests
    check. During warm-up the window is clipped to the episode start index for
    index, as offline; otherwise early steps would have another u distribution
    than `u_ref`. Clearing the buffer at episode start is correctness: a reading
    from before `world.reset()` describes other traffic.
    """

    def __init__(
        self,
        head: Fitted,
        ref: EdgeReference,
        sources: np.ndarray,
        edge_index: torch.Tensor,
        *,
        tau: float = DEFAULT_TAU,
    ) -> None:
        if sources.shape[0] != int(edge_index.max()) + 1:
            raise ValueError(
                f"sources has {sources.shape[0]} nodes, edge_index sees {int(edge_index.max()) + 1}"
            )
        self.head, self.ref, self.sources, self.edge_index, self.tau = (
            head,
            ref,
            sources,
            edge_index,
            tau,
        )
        self.n_nodes = int(sources.shape[0])
        self.n_mov = int(sources.shape[1])
        self.adj = neighbor_matrix(edge_index.numpy(), self.n_nodes) if head.cfg.neighbors else None
        self.n_persist = head.n_feat if head.cfg.persistence else 0
        self._buf: list[np.ndarray] = []

    def reset(self) -> None:
        self._buf.clear()

    @property
    def ready(self) -> bool:
        """Whether the buffer holds a FULL window. During warm-up persistence signals
        are distorted, so the gate stays silent; log_rho modulation may go on (the
        head learned with the same artefact)."""
        return len(self._buf) >= self.head.cfg.window

    @torch.no_grad()
    def movements(self, obs: np.ndarray) -> np.ndarray:
        """Add the reading to the window; u PER MOVEMENT [N, n_mov] for THIS step."""
        return self._u_mov(obs)

    @torch.no_grad()
    def log_rho(self, obs: np.ndarray) -> torch.Tensor:
        """Add the reading to the window; log_rho [E + N] for THIS step."""
        u_mov = self._u_mov(obs)
        # the side comes from the REFERENCE: u and u_ref must use the same side
        u_edge, _ = edge_uncertainty(u_mov[None], self.sources, "mean", side=self.ref.side)
        return log_rho(u_edge[0], u_mov.mean(-1), self.ref, self.edge_index, tau=self.tau)

    def _u_mov(self, obs: np.ndarray) -> np.ndarray:
        """Buffer -> features -> sigma -> u per movement; shared by the UQ gate and
        the log_rho modulation, which must not drift apart."""
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape[0] != self.n_nodes:
            raise ValueError(f"observation has {obs.shape[0]} rows, the graph {self.n_nodes}")
        self._buf.insert(0, obs)  # newest first, as `window_index` indexes
        del self._buf[self.head.cfg.window :]

        hist = np.stack(self._buf)  # [m, N, F], m <= window, row 0 = newest
        # The FULL window matrix [m, k], not one row: the neighbour branch of
        # `build_inputs` uses the whole `x`, so with one `win` row the parts would
        # differ in height. All m rows are computed and row 0 is taken.
        m, k = len(self._buf), self.head.cfg.window
        win = np.minimum(np.arange(m)[:, None] + np.arange(k)[None, :], m - 1)
        z = build_inputs(hist, win, self.adj, self.n_persist)[:1]  # [1, N, D]
        if z.shape[-1] != self.head.in_mean.shape[0]:
            raise ValueError(
                f"head input has width {z.shape[-1]}, the head was trained on "
                f"{self.head.in_mean.shape[0]}: another window/neighbour/persistence setup"
            )

        z = (z - self.head.in_mean) / self.head.in_std
        self.head.model.eval()
        _, sigma = self.head.model(torch.from_numpy(z.astype(np.float32)))
        # through `y_std` and back, EXACTLY as `Fitted.predict` + `evaluate`, so u is
        # in the same units as `u_ref` to the last bit
        s_std = (sigma.numpy() * self.head.y_std) / self.head.y_std
        return movement_uncertainty(s_std, self.n_mov)[0]  # [N, n_mov]


#: Allowed share of FALSE switches on a clean window. A gate threshold is the
#: (1 - this) quantile of the signal on clean traffic, one criterion for the rule
#: and UQ, so the ablation compares signal QUALITY, not eagerness to fire.
GATE_FALSE_ALARM = 0.005
GATE_FORMAT = 1


@dataclass(frozen=True)
class GateReference:
    """Gate thresholds calibrated on the CLEAN window of the BC dataset, not on
    evaluation results.

    The criterion (<= `false_alarm` decisions with a false switch) is written
    before measuring; thresholds come from the training part, checked on the tail.
    Both signals have the SAME shape, a node's fraction of suspicious movements,
    differing only in the criterion: `rule` asks "do this movement's features
    stand still" (`suspicious_movements`), `uq` asks "does its u exceed its own
    clean norm" (`u_mov_ref`).
    """

    rule: float
    uq: float
    #: [N, n_mov]: `mean + k*sd` of a MOVEMENT's u on the clean window; per movement,
    #: since the gate asks "how many of my detectors went wrong".
    u_mov_ref: np.ndarray
    false_alarm: float
    window: int
    network: str
    head_sha: str

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            format=np.array(GATE_FORMAT),
            rule=np.array(self.rule),
            uq=np.array(self.uq),
            u_mov_ref=self.u_mov_ref,
            false_alarm=np.array(self.false_alarm),
            window=np.array(self.window),
            network=np.array(self.network),
            head_sha=np.array(self.head_sha),
        )
        return path


def gate_path(network: str) -> Path:
    from tsc_marl.paths import RUNS

    return RUNS / "heads" / f"gate_ref__{network}.npz"


def load_gates(
    path: str | Path, *, network: str | None = None, head: str | Path | None = None
) -> GateReference:
    blob = np.load(Path(path), allow_pickle=False)
    if int(blob["format"]) != GATE_FORMAT:
        raise ValueError(f"gate threshold format {int(blob['format'])}, expected {GATE_FORMAT}")
    ref = GateReference(
        rule=float(blob["rule"]),
        uq=float(blob["uq"]),
        u_mov_ref=blob["u_mov_ref"],
        false_alarm=float(blob["false_alarm"]),
        window=int(blob["window"]),
        network=str(blob["network"]),
        head_sha=str(blob["head_sha"]),
    )
    if network is not None and ref.network != network:
        raise ValueError(f"thresholds were computed on {ref.network!r}, the network is {network!r}")
    if head is not None and (sha := file_sha(head)) != ref.head_sha:
        raise ValueError(
            f"UQ threshold computed with another head (hash {ref.head_sha}, file has {sha}); "
            "u is in units of THAT head; recompute: `tsc obshead --save`"
        )
    return ref


def calibrate_gates(
    fitted: Fitted,
    ref: EdgeReference,
    data: Loaded,
    sources: np.ndarray,
    *,
    network: str,
    head_sha: str,
    false_alarm: float = GATE_FALSE_ALARM,
    scenario: str = "clean",
) -> GateReference:
    """Both gate thresholds from one quantile of the clean signal distribution.
    `ref` is unused here but kept in the signature for neighbour-edge variants."""
    # FULL-window steps only; otherwise warm-up fills the tail with ones
    train = data.mask(fitted.cfg.split, train=True) & full_window(data.episode, fitted.cfg.window)
    win = window_index(data.episode, fitted.cfg.window)
    n_mov = sources.shape[1]

    rule = suspicious_movements(data.x[scenario], win, n_mov).mean(-1)  # [T, N]

    _, sigma = fitted.predict(data, data.x[scenario])
    u_mov = movement_uncertainty(sigma / fitted.y_std, n_mov)  # [T, N, n_mov]
    clean = u_mov[train]
    u_mov_ref = clean.mean(0) + REFERENCE_K * clean.std(0)  # [N, n_mov]
    uq = (u_mov >= u_mov_ref).mean(-1)  # fraction of suspicious movements, as in the rule

    return GateReference(
        rule=float(np.quantile(rule[train], 1.0 - false_alarm)),
        uq=float(np.quantile(uq[train], 1.0 - false_alarm)),
        u_mov_ref=u_mov_ref,
        false_alarm=float(false_alarm),
        window=int(fitted.cfg.window),
        network=network,
        head_sha=head_sha,
    )


def train_head(
    network: str = "cityflow_jinan3x4_4ph",
    cfg: ObsConfig | None = None,
    out: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """All artefacts at once -> (head, u_ref, gate thresholds); a mismatch between
    them would be silent (numbers in other units)."""
    from tsc_marl.paths import RUNS
    from tsc_marl.uncertainty import prototype as P

    P.cap_threads()
    base = load(RUNS / "bc" / f"advancedmp__{network}__ep3" / "dataset.npz")
    fitted, _ = fit(base, cfg or ObsConfig())
    head = fitted.save(out or head_path(network), network=network)

    sources = movement_sources(network, base.dead["clean"].shape[-1])
    sha = file_sha(head)
    ref = edge_reference(fitted, base, sources, network=network, head_sha=sha)
    gates = calibrate_gates(fitted, ref, base, sources, network=network, head_sha=sha)
    return head, ref.save(reference_path(network)), gates.save(gate_path(network))


def save_reference(network: str, side: str) -> Path:
    """`u_ref` alone for an EXISTING head, without retraining: retraining would
    change the head file and its hash, and `target` vs `source` would no longer
    be paired (another head = another u)."""
    from tsc_marl.paths import RUNS

    head = head_path(network)
    if not head.is_file():
        raise SystemExit(f"no head {head}; first run: tsc obshead -n {network} --save ...")
    fitted = load_head(head, network=network)
    base = load(RUNS / "bc" / f"advancedmp__{network}__ep3" / "dataset.npz")
    sources = movement_sources(network, base.dead["clean"].shape[-1])
    ref = edge_reference(fitted, base, sources, network=network, head_sha=file_sha(head), side=side)
    return ref.save(reference_path(network, side))


# ---------------------------------------------------------------- metrics


def auroc(pos: np.ndarray, neg: np.ndarray, cap: int = 200_000, seed: int = 0) -> float:
    """AUROC via ranks (Mann-Whitney), ties averaged."""
    pos, neg = np.asarray(pos, float).ravel(), np.asarray(neg, float).ravel()
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    if len(pos) > cap:
        pos = rng.choice(pos, cap, replace=False)
    if len(neg) > cap:
        neg = rng.choice(neg, cap, replace=False)
    x = np.concatenate([pos, neg])
    ranks = np.empty(len(x))
    ranks[x.argsort(kind="mergesort")] = np.arange(1, len(x) + 1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / counts)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def movement_uncertainty(sigma_std: np.ndarray, n_mov: int) -> np.ndarray:
    """[.., n_feat] -> [.., n_mov]: max sigma over a movement's features (EP, demand)."""
    blocks = [
        sigma_std[..., b * n_mov : (b + 1) * n_mov] for b in range(sigma_std.shape[-1] // n_mov)
    ]
    return np.max(np.stack(blocks), axis=0)


#: Which end of edge j->i carries its uncertainty:
#:
#:   target  movements of TARGET i whose incoming road starts at j (the physical
#:           lanes of road j->i); the original variant.
#:   source  ALL movements of SOURCE j: the reliability of the MESSAGE, since
#:           attention on j->i weighs `hidden_embedding(x_j)`, built from j's readings.
#:
#: The difference is largest for a dead intersection j: on the target side road
#: j->i belongs to healthy i, so neighbours keep trusting j's data; on the source
#: side every neighbour mutes exactly the edge from j.
EDGE_SIDES = ("target", "source")


def _check_side(side: str) -> None:
    if side not in EDGE_SIDES:
        raise ValueError(f"side={side!r}, allowed: {EDGE_SIDES}")


def edge_uncertainty(
    u_mov: np.ndarray, sources: np.ndarray, how: str = "max", *, side: str = "target"
):
    """Movement u -> edge u (j->i). -> ([T, E], list of (i, j)). Border movements
    (-1) belong to no edge. The set and ORDER of `edges` do not depend on `side`,
    so `log_rho` permutes both variants the same way. With `side="source"` the
    aggregation covers ALL of j's movements, border ones included."""
    _check_side(side)
    edges = sorted({(i, int(j)) for i in range(sources.shape[0]) for j in sources[i] if j >= 0})
    agg = np.max if how == "max" else np.mean
    if side == "source":
        cols = [agg(u_mov[:, j, :], axis=-1) for _i, j in edges]
    else:
        cols = [agg(u_mov[:, i, sources[i] == j], axis=-1) for i, j in edges]
    return np.stack(cols, axis=-1), edges


def corrupted_movements(x, x_clean, dead, n_mov: int) -> np.ndarray:
    """[T, N, n_mov]: a movement with a dead detector OR a feature error > CORRUPT_TAU;
    `x`, `x_clean` are the movement feature blocks only (no phase one-hot)."""
    err = np.abs(x - x_clean)
    big = np.zeros(x.shape[:-1] + (n_mov,), dtype=bool)
    for b in range(err.shape[-1] // n_mov):
        big |= err[..., b * n_mov : (b + 1) * n_mov] > CORRUPT_TAU
    return big | dead.astype(bool)


def evaluate(fitted: Fitted, data: Loaded, sources: np.ndarray, scenarios=None) -> dict[str, Any]:
    """Metrics on the episode TAIL for every scenario of the dataset."""
    n_mov = data.dead["clean"].shape[-1]
    tail = data.mask(fitted.cfg.split, train=False)
    nf = fitted.n_feat
    res: dict[str, Any] = {}
    clean_u = clean_edge = None
    for name in ["clean", *sorted(s for s in data.x if s != "clean")]:
        if scenarios and name not in scenarios and name != "clean":
            continue
        x = data.x[name]
        mu, sigma = fitted.predict(data, x)
        mu, sigma = mu[tail], sigma[tail]
        xc, xt = data.x_clean[tail][..., :nf], x[tail][..., :nf]
        s_std = sigma / fitted.y_std
        u_mov = movement_uncertainty(s_std, n_mov)
        bad = corrupted_movements(xt, xc, data.dead[name][tail], n_mov)
        # A WRONG reading = the feature really moved away from the truth. A dead
        # detector on an empty lane reports the true 0: "corrupted", but nothing can
        # detect it (zero is modal), so detection is also computed on wrong ones only.
        wrong = corrupted_movements(xt, xc, np.zeros_like(data.dead[name][tail]), n_mov)
        u_max, _ = edge_uncertainty(u_mov, sources, "max")
        u_mean, edges = edge_uncertainty(u_mov, sources, "mean")
        bad_edge = np.stack([bad[:, i, sources[i] == j].any(-1) for i, j in edges], -1)
        if name == "clean":
            clean_u, clean_edge = u_mov, (u_max, u_mean)
        err_mu = mu - xc
        row = {
            "u_mean": float(u_mov.mean()),
            "coverage90": float((np.abs(err_mu) <= 1.645 * sigma).mean()),
            "calib": float(np.sqrt((err_mu**2).mean()) / np.sqrt((sigma**2).mean())),
            "rmse_input": float(np.sqrt(((xt - xc) ** 2).mean())),
            "rmse_mu": float(np.sqrt((err_mu**2).mean())),
            "corrupt_share": float(bad.mean()),
            "wrong_share": float(wrong.mean()),
            "auroc_wrong": auroc(u_mov[wrong], clean_u) if name != "clean" else float("nan"),
            "auroc_mov": auroc(u_mov[bad], clean_u) if name != "clean" else float("nan"),
            "auroc_edge_max": (
                auroc(u_max[bad_edge], clean_edge[0]) if name != "clean" else float("nan")
            ),
            "auroc_edge_mean": (
                auroc(u_mean[bad_edge], clean_edge[1]) if name != "clean" else float("nan")
            ),
        }
        res[name] = row
    return res


# ------------------------------------------------------------------ gate


def _gauss_series(res: dict[str, Any]) -> list[tuple[float, float]]:
    pts = [(0.0, res["clean"]["u_mean"])]
    for name, r in res.items():
        if name.startswith("gauss"):
            pts.append((float(name[5:].replace("p", ".")), r["u_mean"]))
    return sorted(pts)


def gate(res: dict[str, Any], sep: dict[str, Any]) -> dict[str, Any]:
    gs = _gauss_series(res)
    # criteria on WRONG readings (see `evaluate`): "does the head see a bad reading",
    # not "does it know a detector is dead while it reports the truth"
    unseen = [res[n]["auroc_wrong"] for n in UNSEEN if n in res]
    seen = [res[n]["auroc_wrong"] for n in ("gauss1", "gauss2", "mask0p25") if n in res]
    drift = [res[n]["auroc_wrong"] for n in res if n.startswith("drift")]
    out = {
        "a_series": gs,
        "a_monotonic": [v for _, v in gs] == sorted(v for _, v in gs),
        "c_coverage_clean": res["clean"]["coverage90"],
        "c_coverage_seen": float(np.mean([res[n]["coverage90"] for n in ("gauss1", "gauss2")])),
        "d_auroc_seen_min": min(seen) if seen else float("nan"),
        "e_auroc_unseen_min": min(unseen) if unseen else float("nan"),
        "e_auroc_drift_max": max(drift) if drift else float("nan"),
        "f_edge_max_vs_mean": float(
            np.nanmean(
                [res[n]["auroc_edge_max"] - res[n]["auroc_edge_mean"] for n in res if n != "clean"]
            )
        ),  # fmt: skip
        "sep": sep,
    }
    out["passed"] = bool(
        out["a_monotonic"] and out["d_auroc_seen_min"] >= 0.8 and out["e_auroc_unseen_min"] >= 0.7
    )
    return out


def separation(fitted: Fitted, base: Loaded, ood: dict[str, Loaded], ens_fn) -> dict[str, Any]:
    """2x2 table {clean, corrupted} x {training traffic, OOD}.

    `u_obs` should react to the ROW (failure), `u_ep` to the COLUMN (unknown
    traffic); if both react to everything, there is one signal in two disguises."""
    out: dict[str, Any] = {}
    sets = {"training": base, **ood}
    for label, data in sets.items():
        tail = data.mask(fitted.cfg.split, train=False)
        n_mov = data.dead["clean"].shape[-1]
        for scen in ("clean", "gauss2", "mask0p5"):
            if scen not in data.x:
                continue
            _, sigma = fitted.predict(data, data.x[scen])
            u_obs = movement_uncertainty(sigma[tail] / fitted.y_std, n_mov).mean()
            u_ep = ens_fn(data, data.x[scen][tail]) if ens_fn else float("nan")
            out[f"{label}|{scen}"] = {"u_obs": float(u_obs), "u_ep": float(u_ep)}
    return out


def _fmt(v: float, f: str = ".3f") -> str:
    return "-" if v is None or not np.isfinite(v) else format(v, f)


def to_markdown(res: dict[str, Any], g: dict[str, Any], cfg: ObsConfig, ood_res: dict) -> str:
    ok = {True: "YES", False: "NO"}
    lines = [
        "# Prototype 2.2 gate: OBSERVATION head (uncertainty of input features)",
        "",
        f"Window of {cfg.window} readings{' + upstream neighbour mean' if cfg.neighbors else ''}, "
        f"{cfg.epochs} epochs, Gaussian NLL against `x_clean`, trained on: "
        f"{', '.join(cfg.train_scenarios)}. Evaluated on the episode tail. u = sigma in units "
        "of the clean feature deviation; movement u = max over its features (EP, demand).",
        "",
        "| scenario | train | corrupted movements | wrong movements | mean u "
        "| AUROC movement (corrupted) | AUROC movement (wrong) | AUROC edge (max) "
        "| AUROC edge (mean) | 90% coverage | input RMSE | mu RMSE |",
        "|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in res.items():
        lines.append(
            f"| {name} | {'x' if name in cfg.train_scenarios else ''} | {r['corrupt_share']:.1%} "
            f"| {r['wrong_share']:.1%} | {r['u_mean']:.3f} | {_fmt(r['auroc_mov'])} "
            f"| {_fmt(r['auroc_wrong'])} | {_fmt(r['auroc_edge_max'])} "
            f"| {_fmt(r['auroc_edge_mean'])} | {r['coverage90']:.1%} | {r['rmse_input']:.3f} "
            f"| {r['rmse_mu']:.3f} |"
        )
    if ood_res:
        lines += ["", "### The same metrics on OOD sets (traffic outside training)", "",
                  "| set | scenario | mean u | AUROC movement | 90% coverage |",
                  "|---|---|---:|---:|---:|"]  # fmt: skip
        for label, rr in ood_res.items():
            for name in ("clean", "gauss2", "mask0p5", "drift0p3"):
                if name in rr:
                    r = rr[name]
                    lines.append(
                        f"| {label} | {name} | {r['u_mean']:.3f} | {_fmt(r['auroc_mov'])} "
                        f"| {r['coverage90']:.1%} |"
                    )
    sep = g["sep"]
    labels = sorted({k.split("|")[0] for k in sep}, key=lambda s: (s != "training", s))
    lines += [
        "",
        "## 2x2 separation: do the two channels measure two different things",
        "",
        "`u_obs` = mean u of the observation head, `u_ep` = deep-ensemble epistemic "
        "(MI between members, prototype 2.0). Row = state of readings, column = traffic.",
        "",
        "| readings \\ traffic | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    for scen in ("clean", "gauss2", "mask0p5"):
        cells = []
        for lab in labels:
            v = sep.get(f"{lab}|{scen}")
            cells.append(
                "-" if v is None else f"u_obs {v['u_obs']:.3f}, u_ep {_fmt(v['u_ep'], '.4f')}"
            )
        lines.append(f"| {scen} | " + " | ".join(cells) + " |")
    lines += ["", *_separation_findings(sep, labels)]
    gs = g["a_series"]
    lines += [
        "",
        "## Exit criteria",
        "",
        f"* **(a) monotonicity**: u over noise sigma ({', '.join(f'{s:g}' for s, _ in gs)}): "
        f"{', '.join(f'{v:.3f}' for _, v in gs)} -> **{ok[g['a_monotonic']]}** "
        "(3.5 and 4.0 UNSEEN in training).",
        f"* **(c) calibration**: 90% interval coverage on clean "
        f"{g['c_coverage_clean']:.1%}, on seen noise {g['c_coverage_seen']:.1%}.",
        f"* **(d) detection on seen**: min movement AUROC (gauss1, gauss2, mask0p25): "
        f"**{_fmt(g['d_auroc_seen_min'])}** (threshold 0.8).",
        f"* **(e) detection on UNSEEN**: min AUROC over {', '.join(UNSEEN)}: "
        f"**{_fmt(g['e_auroc_unseen_min'])}** (threshold 0.7); `drift` (another family): max "
        f"AUROC **{_fmt(g['e_auroc_drift_max'])}**.",
        f"* **(f) movement -> edge aggregation**: AUROC(max) - AUROC(mean), mean: "
        f"{g['f_edge_max_vs_mean']:+.3f} -> in attention "
        f"**{'max' if g['f_edge_max_vs_mean'] >= 0 else 'mean'}**.",
        "",
        f"**Gate 2.2 (observation head): {'PASSED' if g['passed'] else 'NOT PASSED'}.**",
        "",
        "## What this table does NOT say",
        "",
        "* The prototype is SUPERVISED (the simulator knows `x_clean`). In the control loop the "
        "head works the same way: trained offline, inference without the truth.",
        "* AUROC compares corrupted cells with the CLEAN scenario; it does not measure "
        "localisation within one episode (where all detectors are noisy there is nothing to "
        "localise).",
        f'* "Corrupted movement" = a dead detector or a feature error > {CORRUPT_TAU} vehicles. '
        "Drift of small counts rarely crosses it, so its AUROC uses few cells.",
    ]
    return "\n".join(lines) + "\n"


def _separation_findings(sep: dict[str, Any], labels: list[str]) -> list[str]:
    def val(lab, scen, key):
        v = sep.get(f"{lab}|{scen}")
        return float("nan") if v is None else v[key]

    ood = [lab for lab in labels if lab != "training"]
    if not ood:
        return []
    row_obs = np.nanmean([val(lab, "gauss2", "u_obs") - val(lab, "clean", "u_obs")
                          for lab in ["training", *ood]])  # fmt: skip
    col_obs = np.nanmean(
        [val(lab, "clean", "u_obs") - val("training", "clean", "u_obs") for lab in ood]
    )
    row_ep = np.nanmean([val(lab, "gauss2", "u_ep") - val(lab, "clean", "u_ep")
                         for lab in ["training", *ood]])  # fmt: skip
    col_ep = np.nanmean(
        [val(lab, "clean", "u_ep") - val("training", "clean", "u_ep") for lab in ood]
    )
    out = [
        f"* `u_obs`: gain from failure (row) **{row_obs:+.3f}**, from OOD (column) "
        f"**{col_obs:+.3f}**.",
        f"* `u_ep`: gain from failure **{row_ep:+.4f}**, from OOD **{col_ep:+.4f}**.",
    ]
    if np.isfinite(col_obs) and col_obs < 0:
        out.append(
            "* `u_obs` DROPS on traffic outside training: the head's sigma scales with load "
            "(fewer vehicles, smaller possible reading error) and the OOD sets are lighter. "
            "Harmless for a fallback (no false alarms), but a threshold on `u_obs` must be "
            "relative to the sigma predicted on clean traffic, not an absolute constant."
        )
    obs_ok = np.isfinite(row_obs) and row_obs > 3 * max(abs(col_obs), 1e-9)
    ep_ok = np.isfinite(col_ep) and col_ep > max(row_ep, 0)
    out.append(
        "* Separation verdict: "
        + ("`u_obs` reacts to failure, not to OOD; " if obs_ok else "`u_obs` is NOT clean; ")
        + (
            "`u_ep` reacts more to OOD than to failure."
            if ep_ok
            else "`u_ep` does NOT react more to OOD than to failure, so a two-level fallback "
            "has no empirical basis here."
        )  # fmt: skip
    )
    return out


# ------------------------------------------------------- variants for masked detectors

#: Variants compared when trying to improve the 2.2 gate on dead detectors. List and
#: criterion written BEFORE the first run; the result cannot move the threshold.
#: Variant 0 = the 2.2 gate head, the reference.
MASK_VARIANTS = (
    ("window 4 (gate 2.2)", ObsConfig()),
    ("window 12", ObsConfig(window=12)),
    ("window 12 + persistence", ObsConfig(window=12, persistence=True)),
)
MASK_UNSEEN = ("mask0p5", "mask0p25_intersection")
NOISE_UNSEEN = ("gauss3p5", "gauss4", "urand3p5")
MASK_THRESHOLD = 0.70  # AUROC on wrong readings, BOTH unseen masks
NOISE_FLOOR = 0.80  # unseen noise must not lose because of it


def mask_variants(network: str = "cityflow_jinan3x4_4ph") -> Path:
    """Three head variants on the same dataset -> `results/gate_2_2_masks.md`."""
    from tsc_marl.paths import RESULTS, RUNS
    from tsc_marl.uncertainty import prototype as P

    P.cap_threads()
    base = load(RUNS / "bc" / f"advancedmp__{network}__ep3" / "dataset.npz")
    sources = movement_sources(network, base.dead["clean"].shape[-1])
    results = []
    for label, cfg in MASK_VARIANTS:
        fitted, _ = fit(base, cfg)
        results.append((label, cfg, evaluate(fitted, base, sources)))

    cols = ("gauss1", "gauss2", "gauss4", "urand3p5", "mask0p25", "mask0p5",
            "mask0p25_intersection", "drift0p3")  # fmt: skip
    out = [
        "# Gate 2.2: attempt to improve on dead detectors",
        "",
        f"Criterion fixed BEFORE running: a variant passes if AUROC on wrong readings is "
        f">= {MASK_THRESHOLD} on BOTH unseen masks ({', '.join(MASK_UNSEEN)}) and "
        f">= {NOISE_FLOOR} on every unseen noise ({', '.join(NOISE_UNSEEN)}). "
        "Trained on clean/gauss1/gauss2/mask0p25, evaluated on the episode tail.",
        "",
        "| variant | " + " | ".join(cols) + " | 90% coverage (clean) | passes |",
        "|---|" + "---:|" * len(cols) + "---:|:-:|",
    ]
    passing = []
    for label, _cfg, res in results:
        ok = (
            min(res[n]["auroc_wrong"] for n in MASK_UNSEEN) >= MASK_THRESHOLD
            and min(res[n]["auroc_wrong"] for n in NOISE_UNSEEN) >= NOISE_FLOOR
        )
        if ok:
            passing.append(label)
        cells = " | ".join(_fmt(res[c]["auroc_wrong"]) for c in cols)
        out.append(
            f"| {label} | {cells} | {res['clean']['coverage90']:.1%} | {'YES' if ok else 'NO'} |"
        )
    best = max(results, key=lambda r: min(r[2][n]["auroc_wrong"] for n in MASK_UNSEEN))
    out += [
        "",
        "## Verdict",
        "",
        (
            f"**Gate 2.2 PASSED** with: {', '.join(passing)}. This variant replaces the "
            "gate 2.2 head in the mechanism."
            if passing
            else "**Gate 2.2 CLOSED AS PARTIALLY PASSED.** No variant reaches "
            f"{MASK_THRESHOLD} on both unseen masks. Best on masks: "
            f'"{best[0]}" (min AUROC '
            f"{min(best[2][n]['auroc_wrong'] for n in MASK_UNSEEN):.3f}). The head estimates "
            "uncertainty for NOISE failures and works poorly on SILENT ones (dead detectors)."
        ),
    ]
    path = RESULTS / "gate_2_2_masks.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


def run(cfg: ObsConfig | None = None, *, network: str = "cityflow_jinan3x4_4ph") -> Path:
    from tsc_marl.paths import RESULTS, RUNS
    from tsc_marl.uncertainty import prototype as P

    P.cap_threads()
    cfg = cfg or ObsConfig()
    base = load(RUNS / "bc" / f"advancedmp__{network}__ep3" / "dataset.npz")
    n_mov = base.dead["clean"].shape[-1]
    fitted, history = fit(base, cfg)
    res = evaluate(fitted, base, movement_sources(network, n_mov))

    ood: dict[str, Loaded] = {}
    ood_res: dict[str, Any] = {}
    for label in ("r2000", "hz", "d0p25"):
        net = P.OOD_NETWORKS[label]
        path = RUNS / "bc" / f"advancedmp__{net}__ep3" / "dataset.npz"
        if path.is_file():
            ood[label] = load(path)
            ood_res[label] = evaluate(
                fitted, ood[label], movement_sources(net, n_mov),
                scenarios=("gauss2", "mask0p5", "drift0p3"),
            )  # fmt: skip

    pcfg = P.TrainConfig()
    ensemble = P.fit_ensemble(base, pcfg)

    def ens_fn(data, x):
        return float(P.ensemble_uncertainty(ensemble, x, data, pcfg)["ens_epistemic"].mean())

    g = gate(res, separation(fitted, base, ood, ens_fn))
    out = RESULTS / "gate_2_2.md"
    out.write_text(to_markdown(res, g, cfg, ood_res), encoding="utf-8")
    (RESULTS / "gate_2_2.json").write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "gate": g,
                "rows": res,
                "ood": ood_res,
                "history": history[-5:],
            },
            indent=1,
            default=str,
        ),  # fmt: skip
        encoding="utf-8",
    )
    return out
