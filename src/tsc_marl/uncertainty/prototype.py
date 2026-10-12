"""Decision-uncertainty prototype: supervised training of the uncertainty head and its gate.

Cheaply checks the premise ("uncertainty grows with sensor failure") and picks a
method (EDL vs MC-dropout) offline, on the dataset from `dataset.py`, on a CPU.

"One intersection" = explicit self-loops, not another model: the GAT layer adds
the loops itself, so without edges each node sees only itself, with exactly the
core's architecture.

The data split is BY TIME: traffic in `flow.json` is deterministic, so episodes
share the same trajectory (they differ only in the failure draw); a split by
episode would leak, and a random split leaks through correlated neighbouring
decisions. The first `split` of each episode trains, the tail evaluates.

Noise levels in training are SEPARATE from test levels: the head trains on
`gauss` 1.0/2.0 and is tested on 3.5/4.0, `urand` and masks, i.e. on noise it has
not seen; otherwise a calibrated detector of known levels would pass as an
uncertainty estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

#: Head training scenarios. The 25% mask IS in training (without it the decision
#: head does not react to masking at all). Held out: `gauss` 3.5/4.0, `urand`, the
#: 50% mask, the intersection mask and ALL of `drift`, i.e. another severity,
#: another failure unit and another failure family.
TRAIN_SCENARIOS = ("clean", "gauss1", "gauss2", "mask0p25")
#: OOD sets for epistemic uncertainty, three kinds of distance from training:
#:   d*  the same traffic pattern scaled DOWN (`domain/demand.py`); at low demand
#:       phase ties become frequent, so decision uncertainty may be ambivalence,
#:       hence columns computed on decidable samples only;
#:   r*  ANOTHER measured traffic pattern (`_2000`/`_2500` files of Advanced_XLight);
#:   hz  another TOPOLOGY and load (HangZhou 4x4), same feature semantics.
OOD_NETWORKS = {
    "d0p75": "cityflow_jinan3x4_4ph_d0p75",
    "d0p5": "cityflow_jinan3x4_4ph_d0p5",
    "d0p25": "cityflow_jinan3x4_4ph_d0p25",
    "r2000": "cityflow_jinan3x4_4ph_r2000",
    "r2500": "cityflow_jinan3x4_4ph_r2500",
    "hz": "cityflow_hangzhou4x4_4ph",
}
OOD_GROUPS = {
    "demand x": ("d0p75", "d0p5", "d0p25"),
    "pattern": ("r2000", "r2500"),
    "topology": ("hz",),
}
TIE_EPS = 1e-6


def cap_threads(limit: int = 4) -> int:
    """Torch threads for CPU prototypes. With tens of thousands of weights and
    32-step batches, synchronisation costs more than arithmetic at the default
    thread count, and two such processes side by side spin in OpenMP busy-waiting.
    An explicit OMP_NUM_THREADS wins."""
    import os

    n = int(os.environ.get("OMP_NUM_THREADS") or min(limit, os.cpu_count() or 1))
    torch.set_num_threads(n)
    return n


@dataclass(frozen=True)
class TrainConfig:
    mode: str = "edl"  # "edl" (deployed) or "softmax" (base for MC-dropout)
    epochs: int = 150
    lr: float = 1e-3
    batch: int = 32  # in TIME STEPS, not rows: one step is the whole graph
    dropout: float = 0.1
    #: The KL regulariser's `anneal` grows linearly over this many epochs; switched
    #: on at once it kills evidence before the model learns (Sensoy et al., 2018).
    anneal_epochs: int = 10
    split: float = 0.67  # fraction of each episode for training (split BY TIME)
    train_scenarios: tuple[str, ...] = TRAIN_SCENARIOS
    hop: bool = False  # False = 0-hop (one intersection)
    mc_passes: int = 10  # K for MC-dropout
    #: Deep ensemble as the epistemic REFERENCE (Lakshminarayanan et al., 2017):
    #: last-layer MC-dropout gives near-zero epistemic values. 0 = off.
    ensemble: int = 5
    seed: int = 0
    weight_decay: float = 0.0


@dataclass
class Loaded:
    """Dataset loaded from npz. `times` is the decision index WITHIN an episode,
    the basis of the time split."""

    x: dict[str, np.ndarray]
    dead: dict[str, np.ndarray]
    x_clean: np.ndarray
    y: np.ndarray
    tie_margin: np.ndarray
    phase_time: np.ndarray
    episode: np.ndarray
    edge_index: np.ndarray
    incidence: np.ndarray
    times: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        _, counts = np.unique(self.episode, return_counts=True)
        self.times = np.concatenate([np.arange(c) for c in counts])

    @property
    def n_nodes(self) -> int:
        return self.y.shape[1]

    @property
    def ob_length(self) -> int:
        return self.x_clean.shape[-1]

    @property
    def n_phases(self) -> int:
        return int(self.incidence.shape[1])

    def mask(self, split: float, train: bool) -> np.ndarray:
        cut = int(round(split * (self.times.max() + 1)))
        return self.times < cut if train else self.times >= cut


def load(path: Path | str) -> Loaded:
    d = np.load(Path(path))
    return Loaded(
        x={k[3:]: d[k] for k in d.files if k.startswith("x__")},
        dead={k[6:]: d[k] for k in d.files if k.startswith("dead__")},
        x_clean=d["x_clean"],
        y=d["y"],
        tie_margin=d["tie_margin"],
        phase_time=d["phase_time"],
        episode=d["episode"],
        edge_index=d["edge_index"],
        incidence=d["incidence"],
    )


# ------------------------------------------------------------------ training


def fit(data: Loaded, cfg: TrainConfig | None = None) -> tuple[Any, list[dict]]:
    """-> (model, history), on the CPU."""
    from tsc_marl.uncertainty.heads import UncertainPolicy, edl_loss

    cfg = cfg or TrainConfig()

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model = UncertainPolicy(data.ob_length, data.n_phases, mode=cfg.mode, dropout=cfg.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    edges = _edges(data, cfg)

    train_t = np.flatnonzero(data.mask(cfg.split, train=True))
    pairs = [(s, t) for s in cfg.train_scenarios for t in train_t]
    y = torch.from_numpy(data.y)

    history: list[dict] = []
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(pairs))
        anneal = min(1.0, (epoch + 1) / max(cfg.anneal_epochs, 1))
        losses, correct, seen = [], 0, 0
        for start in range(0, len(order), cfg.batch):
            batch = [pairs[i] for i in order[start : start + cfg.batch]]
            xb = torch.from_numpy(np.stack([data.x[s][t] for s, t in batch]))
            yb = torch.stack([y[t] for _, t in batch])
            k = len(batch)
            out = model(xb.reshape(-1, data.ob_length), _tiled(edges, data.n_nodes, k))
            target = yb.reshape(-1)

            if cfg.mode == "edl":
                loss = edl_loss(out, target, anneal)
                pred = out.argmax(-1)
            else:
                loss = torch.nn.functional.cross_entropy(out, target)
                pred = out.argmax(-1)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

            losses.append(float(loss))
            correct += int((pred == target).sum())
            seen += target.numel()

        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "acc": correct / seen,
                "anneal": anneal,
            }
        )
    return model, history


def fit_ensemble(data: Loaded, cfg: TrainConfig) -> list[Any]:
    """K `softmax` members with different seeds; they differ ONLY in initialisation
    and minibatch order, so their disagreement measures model uncertainty."""
    from dataclasses import replace

    return [
        fit(data, replace(cfg, mode="softmax", seed=cfg.seed + 1000 + k))[0]
        for k in range(cfg.ensemble)
    ]


@torch.no_grad()
def ensemble_uncertainty(models: list[Any], x: np.ndarray, data: Loaded, cfg: TrainConfig):
    """Same decomposition as MC-BALD: total = H(mean p), aleatoric = E H(p), epistemic = MI."""
    k = x.shape[0]
    xb = torch.from_numpy(x.reshape(-1, data.ob_length))
    ei = _tiled(_edges(data, cfg), data.n_nodes, k)
    probs = []
    for m in models:
        m.eval()
        probs.append(torch.softmax(m(xb, ei), -1))
    probs = torch.stack(probs)  # [M, rows, C]
    mean = probs.mean(0)
    total = -(mean * torch.log(mean + 1e-10)).sum(-1)
    aleatoric = -(probs * torch.log(probs + 1e-10)).sum(-1).mean(0)
    shape = (k, data.n_nodes)
    return {
        "ens_total": total.reshape(shape).numpy(),
        "ens_aleatoric": aleatoric.reshape(shape).numpy(),
        "ens_epistemic": (total - aleatoric).clamp_min(0).reshape(shape).numpy(),
    }


def _edges(data: Loaded, cfg: TrainConfig) -> torch.Tensor:
    """0-hop = EXPLICIT self-loops, NOT an empty `edge_index`.

    An empty one looks like the obvious way to cut neighbours off and is a trap:
    `add_self_loops` without `num_nodes` infers zero nodes, the aggregation returns
    ZEROS and the model trains only its bias, without an error. The double
    self-loop (ours plus the layer's) is harmless: softmax over two identical
    entries gives 0.5 + 0.5, the same as one loop with weight 1."""
    if not cfg.hop:
        loop = torch.arange(data.n_nodes, dtype=torch.long)
        return torch.stack([loop, loop])
    return torch.from_numpy(data.edge_index)


def _tiled(edge_index: torch.Tensor, n_nodes: int, k: int) -> torch.Tensor:
    from tsc_marl.core.ppo import tiled_edge_index

    return tiled_edge_index(edge_index, n_nodes, k)


# ---------------------------------------------------------------- evaluation


@torch.no_grad()
def predict(
    model, x: np.ndarray, data: Loaded, cfg: TrainConfig, ensemble: list[Any] | None = None
) -> dict[str, np.ndarray]:
    """EDL (single pass) and MC-dropout (K passes) uncertainty on the same inputs."""
    from tsc_marl.uncertainty.heads import dirichlet_uncertainty, mc_dropout

    model.eval()
    edges = _edges(data, cfg)
    k = x.shape[0]
    xb = torch.from_numpy(x.reshape(-1, data.ob_length))
    ei = _tiled(edges, data.n_nodes, k)

    out = model(xb, ei)
    single = dirichlet_uncertainty(out) if cfg.mode == "edl" else _softmax_uncertainty(out)
    mc = mc_dropout(model, xb, ei, cfg.mc_passes)

    shape = (k, data.n_nodes)
    result = {
        f"edl_{name}": single[name].reshape(shape).numpy()
        for name in ("total", "aleatoric", "epistemic")
    }
    if "vacuity" in single:
        result["edl_vacuity"] = single["vacuity"].reshape(shape).numpy()
    result |= {
        f"mc_{name}": mc[name].reshape(shape).numpy()
        for name in ("total", "aleatoric", "epistemic")
    }
    result["prob"] = single["prob"].reshape(k, data.n_nodes, -1).numpy()
    result["pred"] = single["prob"].argmax(-1).reshape(shape).numpy()
    if ensemble:
        result |= ensemble_uncertainty(ensemble, x, data, cfg)
    return result


def _softmax_uncertainty(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """A single `softmax` pass does not split uncertainty: all of it is aleatoric
    and epistemic is zero, stated so the table does not suggest otherwise."""
    p = torch.softmax(logits, -1)
    total = -(p * torch.log(p + 1e-10)).sum(-1)
    return {"prob": p, "total": total, "aleatoric": total, "epistemic": torch.zeros_like(total)}


# ------------------------------------------------------------------- gate


def scenario_table(
    model,
    data: Loaded,
    cfg: TrainConfig,
    *,
    label: str = "",
    only: tuple[str, ...] = (),
    ensemble: list[Any] | None = None,
) -> list[dict]:
    """One row per scenario: accuracy, calibration and uncertainty channels on the
    episode TAIL (unseen in training).

    `*_dec` columns use DECIDABLE samples only (the best phase's p(s) strictly
    above the second). A tie is correct, high DECISION uncertainty unrelated to
    failure or OOD, and its share depends on load."""
    from tsc_marl.uncertainty.heads import expected_calibration_error

    mask = data.mask(cfg.split, train=False)
    y = data.y[mask]
    dec = data.tie_margin[mask] > TIE_EPS
    nan = float("nan")
    rows = []
    for name, x in sorted(data.x.items()):
        if only and name not in only:
            continue
        out = predict(model, x[mask], data, cfg, ensemble)
        prob = torch.from_numpy(out["prob"].reshape(-1, data.n_phases))
        rows.append(
            {
                "dataset": label,
                "scenario": name,
                "trained": name in cfg.train_scenarios and not label,
                "acc": float((out["pred"] == y).mean()),
                "ece": expected_calibration_error(prob, torch.from_numpy(y.reshape(-1))),
                "edl_total": float(out["edl_total"].mean()),
                "edl_aleatoric": float(out["edl_aleatoric"].mean()),
                "edl_epistemic": float(out["edl_epistemic"].mean()),
                "edl_vacuity": float(out.get("edl_vacuity", np.zeros(1)).mean()),
                "mc_aleatoric": float(out["mc_aleatoric"].mean()),
                "mc_epistemic": float(out["mc_epistemic"].mean()),
                "ens_total": float(out["ens_total"].mean()) if ensemble else nan,
                "ens_epistemic": float(out["ens_epistemic"].mean()) if ensemble else nan,
                "edl_total_dec": float(out["edl_total"][dec].mean()) if dec.any() else nan,
                "edl_epistemic_dec": (
                    float(out["edl_epistemic"][dec].mean()) if dec.any() else nan
                ),
                "ens_epistemic_dec": (
                    float(out["ens_epistemic"][dec].mean()) if ensemble and dec.any() else nan
                ),
                "tie_share": float(1.0 - dec.mean()),
                "dead": float(data.dead[name][mask].mean()),
            }
        )
    return rows


def _corr(a: list[float], b: list[float]) -> float:
    x, y = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def gate(rows: list[dict]) -> dict[str, Any]:
    """Exit criteria computed from the table, so "passed" is a statement about
    numbers, not an impression from a plot."""
    nan = float("nan")
    by = {(r["dataset"], r["scenario"]): r for r in rows}
    base = [k for k in by if not k[0]]

    def series(prefix: str, key: str) -> list[float]:
        names = [s for _, s in base if s.startswith(prefix)]
        names.sort(key=lambda n: float(n[len(prefix) :].replace("p", ".") or 0))
        return [by[("", n)][key] for n in names]

    # the severity series starts at sigma = 0 (the clean state)
    gauss = [by[("", "clean")]["edl_total"], *series("gauss", "edl_total")]
    clean = by[("", "clean")]
    ood = {r["dataset"]: r for r in rows if r["dataset"] and r["scenario"] == "clean"}

    def gain(key: str, labels) -> float:
        vals = [ood[lab][key] - clean[key] for lab in labels if lab in ood]
        vals = [v for v in vals if np.isfinite(v)]
        return max(vals) if vals else nan

    keys = ("edl_total", "edl_total_dec", "edl_epistemic", "edl_epistemic_dec",
            "ens_epistemic", "ens_epistemic_dec")  # fmt: skip
    groups = {g: {k: gain(k, labels) for k in keys} for g, labels in OOD_GROUPS.items()}
    ratios = [r["edl_epistemic"] / r["edl_total"] for r in rows if r["edl_total"] > 0]
    spread = (max(ratios) - min(ratios)) / float(np.mean(ratios)) if ratios else nan
    has_ens = any(np.isfinite(r.get("ens_epistemic", nan)) for r in rows)
    edl = [r["edl_epistemic"] for r in rows]

    out = {
        # (a) monotonicity: uncertainty grows with noise severity
        "a_monotonic_gauss": gauss == sorted(gauss),
        "a_gauss_series": gauss,
        # (b) separation: aleatoric on noise, epistemic on OOD
        "b_aleatoric_gain_seen": by[("", "gauss2")]["edl_aleatoric"] - clean["edl_aleatoric"],
        "b_aleatoric_gain_unseen": by[("", "gauss4")]["edl_aleatoric"] - clean["edl_aleatoric"],
        "b_epistemic_gain_ood": max(
            (v["edl_epistemic"] for v in groups.values() if np.isfinite(v["edl_epistemic"])),
            default=nan,
        ),
        "b_epistemic_gain_noise": by[("", "gauss2")]["edl_epistemic"] - clean["edl_epistemic"],
        "b_ood_groups": groups,
        "b_ens_gain_noise": by[("", "gauss2")].get("ens_epistemic", nan)
        - clean.get("ens_epistemic", nan),
        "b_ens_gain_unseen": by[("", "gauss4")].get("ens_epistemic", nan)
        - clean.get("ens_epistemic", nan),
        # accuracy keeps FALLING while uncertainty stops growing: "confident and wrong"
        "f_acc_drop_unseen": clean["acc"] - by[("", "gauss4")]["acc"],
        "f_ece_unseen": by[("", "gauss4")]["ece"],
        # (c) calibration
        "c_ece_clean": clean["ece"],
        "c_ece_worst": max(r["ece"] for r in rows),
        # (d) the cheap EDL agrees with the expensive reference
        "d_edl_vs_mc_corr": _corr(edl, [r["mc_epistemic"] for r in rows]),
        "d_edl_vs_ens_corr": _corr(edl, [r.get("ens_epistemic", nan) for r in rows]),
        "d_reference": "ensemble" if has_ens else "mc-dropout (last layer)",
        # (e) generalisation to what the head did NOT see in training
        "e_unseen_noise_above_clean": (
            min(by[("", n)]["edl_total"] for n in ("gauss3p5", "gauss4", "urand3p5"))
            > clean["edl_total"]
        ),
        "e_unseen_mask_above_clean": (
            min(by[("", n)]["edl_total"] for n in ("mask0p5", "mask0p25_intersection"))
            > clean["edl_total"]
        ),
        "e_drift_above_clean": by[("", "drift0p3")]["edl_total"] > clean["edl_total"],
        "e_drift_delta": by[("", "drift0p3")]["edl_total"] - clean["edl_total"],
        "e_drift_acc_drop": clean["acc"] - by[("", "drift0p3")]["acc"],
        # (g) does EDL's epistemic channel carry its OWN information, or is it a
        # constant fraction of total (then epistemic, vacuity and total are one signal)
        "g_ratio_min": min(ratios) if ratios else nan,
        "g_ratio_max": max(ratios) if ratios else nan,
        "g_ratio_spread": spread,
        "g_ratio_constant": bool(np.isfinite(spread) and spread < 0.1),
    }
    corr_d = out["d_edl_vs_ens_corr"] if has_ens else out["d_edl_vs_mc_corr"]
    out["passed"] = bool(
        out["a_monotonic_gauss"]
        and out["e_drift_above_clean"]
        and np.isfinite(corr_d)
        and corr_d > 0.5
        and out["b_epistemic_gain_ood"] > out["b_epistemic_gain_noise"]
        and not out["g_ratio_constant"]
    )
    return out


def _f(v: float, fmt: str = "+.4f") -> str:
    return "-" if v is None or not np.isfinite(v) else format(v, fmt)


def to_markdown(rows: list[dict], g: dict[str, Any], cfg: TrainConfig, meta: dict) -> str:
    ok = {True: "YES", False: "NO"}
    ref = g["d_reference"]
    out = [
        "# Prototype 2.0 gate: DECISION uncertainty vs sensor failure and OOD",
        "",
        f"Mode `{cfg.mode}`, {'1-hop' if cfg.hop else '0-hop (one intersection)'}, "
        f"{cfg.epochs} epochs, trained on: {', '.join(cfg.train_scenarios)}.",
        f"Evaluated on the episode TAIL (last {100 * (1 - cfg.split):.0f}% of decisions). "
        f"Epistemic reference: **{ref}** (K={cfg.ensemble if cfg.ensemble else cfg.mc_passes}).",
        "",
        "| set | scenario | train | acc | ECE | EDL total | EDL epist | epist/total "
        "| vacuity | ens epist | MC epist | ties | EDL total (decidable) | dead movements |",
        "|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        ratio = r["edl_epistemic"] / r["edl_total"] if r["edl_total"] > 0 else float("nan")
        out.append(
            f"| {r['dataset'] or 'base'} | {r['scenario']} | {'x' if r['trained'] else ''} "
            f"| {r['acc']:.3f} | {r['ece']:.3f} | {r['edl_total']:.3f} "
            f"| {r['edl_epistemic']:.4f} | {ratio:.3f} | {r['edl_vacuity']:.3f} "
            f"| {_f(r.get('ens_epistemic'), '.4f')} | {r['mc_epistemic']:.4f} "
            f"| {r.get('tie_share', float('nan')):.1%} | {_f(r.get('edl_total_dec'), '.3f')} "
            f"| {r['dead']:.1%} |"
        )

    out += [
        "",
        "## Uncertainty gain on OOD (relative to clean base), maximum per group",
        "",
        "| group | EDL total | EDL total (decidable) | EDL epist | EDL epist (decidable) "
        "| ens epist | ens epist (decidable) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, v in g["b_ood_groups"].items():
        out.append(
            f"| {name} | {_f(v['edl_total'], '+.3f')} | {_f(v['edl_total_dec'], '+.3f')} "
            f"| {_f(v['edl_epistemic'])} | {_f(v['edl_epistemic_dec'])} "
            f"| {_f(v['ens_epistemic'])} | {_f(v['ens_epistemic_dec'])} |"
        )
    out += [
        f"| *noise gauss2 (for comparison)* | - | - | {_f(g['b_epistemic_gain_noise'])} | - "
        f"| {_f(g['b_ens_gain_noise'])} | - |",
        "",
        "## Exit criteria",
        "",
        f"* **(a) monotonicity**: EDL total over `gauss` (sigma = 0, 1, 2, 3.5, 4): "
        f"{', '.join(f'{v:.3f}' for v in g['a_gauss_series'])} -> "
        f"**{ok[g['a_monotonic_gauss']]}**.",
        f"* **(b) separation**: EDL aleatoric on seen noise "
        f"{_f(g['b_aleatoric_gain_seen'], '+.3f')}, unseen "
        f"{_f(g['b_aleatoric_gain_unseen'], '+.3f')}; EDL epistemic on OOD "
        f"{_f(g['b_epistemic_gain_ood'])} vs on noise {_f(g['b_epistemic_gain_noise'])}.",
        f"* **(c) calibration**: ECE clean {g['c_ece_clean']:.3f}, worst scenario "
        f"{g['c_ece_worst']:.3f}.",
        f"* **(d) EDL vs reference**: epistemic correlation over rows with the ensemble "
        f"**{_f(g['d_edl_vs_ens_corr'], '.3f')}**, with last-layer MC-dropout "
        f"{_f(g['d_edl_vs_mc_corr'], '.3f')}.",
        f"* **(e) unseen**: noise 3.5/4.0 and `urand` above clean: "
        f"**{ok[g['e_unseen_noise_above_clean']]}**; 50% mask and intersection mask: "
        f"**{ok[g['e_unseen_mask_above_clean']]}**; `drift`: **{ok[g['e_drift_above_clean']]}** "
        f"(delta {g['e_drift_delta']:+.3f} with an accuracy drop of {g['e_drift_acc_drop']:.3f}).",
        f"* **(g) is EDL epistemic a separate channel**: epistemic/total in "
        f"[{g['g_ratio_min']:.3f}, {g['g_ratio_max']:.3f}] over all {len(rows)} rows "
        f"(spread {g['g_ratio_spread']:.1%} of the mean) -> "
        f"**{'ONE SIGNAL' if g['g_ratio_constant'] else 'separate channel'}**.",
        "",
        "## Verdict",
        "",
        f"**Gate 2.0 for the DECISION head: {'PASSED' if g['passed'] else 'NOT PASSED'}.**",
        "",
    ]
    out += _verdict(g)
    out += [
        "",
        "## What this table does NOT say",
        "",
        f"Base dataset `tie_rate` = **{meta.get('tie_rate', float('nan')):.1%}**: in that many "
        "decisions two phases have equal p(s) and high DECISION uncertainty is correct, "
        "unrelated to failure. The *ties* column shows their share per set; it grows at low "
        "demand, which is why OOD gains are also given on decidable samples.",
        "",
        "MC-dropout drops the LAST LAYER only; the reference for (d) is a deep ensemble "
        "(K `softmax` members, different seeds), because MC epistemic is of order 0.004.",
    ]
    return "\n".join(out) + "\n"


def _verdict(g: dict[str, Any]) -> list[str]:
    """Conclusions derived from the numbers; the text changes when the data do."""
    lines = []
    gs = g["a_gauss_series"]
    if not g["a_monotonic_gauss"]:
        peak = int(np.argmax(gs))
        lines.append(
            f"1. **Outside the training range decision uncertainty stops growing** (peak at "
            f"index {peak} of {', '.join(f'{v:.3f}' for v in gs)}), while accuracy keeps "
            f"falling (by {g['f_acc_drop_unseen']:.3f}) and ECE rises to {g['f_ece_unseen']:.3f}: "
            "confident and wrong exactly where the fallback should act."
        )
    if not g["e_drift_above_clean"]:
        lines.append(
            f"2. **`drift` is invisible** to decision uncertainty (delta {g['e_drift_delta']:+.3f} "
            f"with an accuracy drop of {g['e_drift_acc_drop']:.3f})."
        )
    if g["g_ratio_constant"]:
        lines.append(
            f"3. **EDL epistemic is not a separate channel**: epistemic/total stays in "
            f"[{g['g_ratio_min']:.3f}, {g['g_ratio_max']:.3f}] in every row, so epistemic, "
            'vacuity and total are one signal on different scales. A gain in "epistemic on '
            'OOD" is a gain in total, not separate information about an unknown state.'
        )
    ens_ood = {k: v["ens_epistemic_dec"] for k, v in g["b_ood_groups"].items()}
    ens_noise = g["b_ens_gain_noise"]
    if all(not np.isfinite(v) for v in ens_ood.values()):
        lines.append("4. Ensemble reference not computed: the epistemic channel is undecided.")
    else:
        best = max((k for k in ens_ood if np.isfinite(ens_ood[k])), key=lambda k: ens_ood[k])
        if ens_ood[best] > max(ens_noise, 0) * 1.5:
            lines.append(
                f"4. **An epistemic fallback channel is supported by the ENSEMBLE, not EDL**: "
                f'ensemble epistemic grows most on group "{best}" '
                f"({ens_ood[best]:+.4f} on decidable samples) vs {ens_noise:+.4f} on noise."
            )
        else:
            lines.append(
                f"4. **Even the ensemble does not separate OOD from noise** (best group "
                f'"{best}": {ens_ood[best]:+.4f} vs noise {ens_noise:+.4f}); an epistemic '
                "fallback channel has no empirical basis on these sets."
            )
    lines.append(
        "5. **Conclusion for 2.2:** the per-edge signal comes from a heteroscedastic head on "
        "the INPUT FEATURES (`results/gate_2_2.md`), not from the Dirichlet spread."
    )
    return lines


def run(cfg: TrainConfig | None = None, *, network: str = "cityflow_jinan3x4_4ph") -> Path:
    """Full gate 2.0: training, table and criteria -> `results/gate_2_0.md`."""
    import json

    from tsc_marl import libsignal
    from tsc_marl.paths import RESULTS, RUNS

    libsignal.bootstrap()  # `heads` -> `core.nets` pulls in `agent.colight`
    cap_threads()
    cfg = cfg or TrainConfig()
    base = RUNS / "bc" / f"advancedmp__{network}__ep3"
    data = load(base / "dataset.npz")
    meta = json.loads((base / "result.json").read_text())

    model, history = fit(data, cfg)
    ensemble = fit_ensemble(data, cfg) if cfg.ensemble else None
    rows = scenario_table(model, data, cfg, ensemble=ensemble)
    for label, net in OOD_NETWORKS.items():
        path = RUNS / "bc" / f"advancedmp__{net}__ep3" / "dataset.npz"
        if not path.is_file():
            continue
        rows += scenario_table(
            model, load(path), cfg, label=label, only=("clean",), ensemble=ensemble
        )

    g = gate(rows)
    out = RESULTS / "gate_2_0.md"
    out.write_text(to_markdown(rows, g, cfg, meta), encoding="utf-8")
    (RESULTS / "gate_2_0.json").write_text(
        json.dumps(
            {"config": cfg.__dict__, "gate": g, "rows": rows, "history": history[-5:]},
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )
    return out
