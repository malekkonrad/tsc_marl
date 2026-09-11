"""Prototyp 2.2 — heteroscedastyczna głowa OBSERWACYJNA: niepewność cech wejściowych.

Po co osobno od 2.0. Bramka 2.0 zmierzyła, że niepewność DECYZJI (Dirichlet nad
fazami) nie estymuje niepewności obserwacji: poza zakresem treningu zawraca,
a dryfu nie widzi. Plan od początku stawiał mechanizm 2.2 gdzie indziej — na
głowie liczonej z CECH WEJŚCIOWYCH, per ruch (l,m), agregowanej na krawędź grafu.
Ten moduł to tani prototyp tego mechanizmu na tym samym zrzucie BC, przed
wpięciem go w attention.

CO GŁOWA DOSTAJE (Załącznik A2 ze spotkania 18.08): pojedynczy odczyt 7 zamiast 4
jest w pełni wiarygodny — skażenie widać wyłącznie przez NIESPÓJNOŚĆ. Wejście to
więc (1) okno `window` ostatnich odczytów węzła (kolejka nie rośnie szybciej niż
saturation flow; na czerwonym może tylko rosnąć — faza jest w one-hocie) i
(2) opcjonalnie średnia odczytów sąsiadów upstream (zachowanie przepływu).

CO GŁOWA ZWRACA: dla każdej cechy ruchu (blok EP i blok demand) średnią μ i skalę σ
rozkładu PRAWDZIWEJ wartości; strata Gaussian NLL względem `x_clean`, który zna
wyłącznie symulator. Człon jest samokalibrujący: zaniżone σ eksploduje kwadratem
błędu, zawyżone kosztuje log σ.

JAK CZYTAĆ WYNIK: u_ruchu = max σ po dwóch cechach ruchu (w jednostkach odchylenia
cechy na czystym), u_krawędzi(j→i) = max albo średnia po ruchach skrzyżowania i,
których droga wjazdowa zaczyna się w j (mini-ablacja z planu). Detekcja = AUROC
u na komórkach POPSUTYCH w scenariuszu vs u na wszystkich komórkach CZYSTYCH —
czyli czy sygnał odróżnia zepsuty odczyt od zdrowego, a nie czy rośnie średnio.

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

#: Błąd odczytu (w pojazdach, na cesze), od którego komórkę uznajemy za popsutą.
#: Martwy detektor jest popsuty ZAWSZE, nawet gdy przypadkiem raportuje prawdę (0 na
#: pustym pasie) — ground truth lokalizacji idzie z warstwy awarii, nie z różnicy cech.
CORRUPT_TAU = 0.5
UNSEEN = ("gauss3p5", "gauss4", "urand3p5", "mask0p5", "mask0p25_intersection")


@dataclass(frozen=True)
class ObsConfig:
    window: int = 4
    neighbors: bool = True
    hidden: int = 128
    epochs: int = 60
    lr: float = 1e-3
    batch: int = 512
    split: float = 0.67  # jak 2.0 — podział PO CZASIE (patrz `prototype.py`)
    train_scenarios: tuple[str, ...] = TRAIN_SCENARIOS
    seed: int = 0
    min_sigma: float = 0.02  # w jednostkach standaryzowanych — dolna granica σ


# ---------------------------------------------------------------- wejścia


def window_index(episode: np.ndarray, k: int) -> np.ndarray:
    """[T, k]: indeksy t, t−1, …, t−k+1, przycięte do POCZĄTKU EPIZODU.

    Okno nie może sięgać do poprzedniego epizodu: po `world.reset()` ostatni
    odczyt poprzedniego epizodu opisuje zupełnie inny stan ruchu."""
    t = np.arange(len(episode))
    first: dict[int, int] = {}
    for i, e in enumerate(episode.tolist()):
        first.setdefault(e, i)
    start = np.array([first[e] for e in episode.tolist()])
    return np.maximum(t[:, None] - np.arange(k)[None, :], start[:, None])


def neighbor_matrix(edge_index: np.ndarray, n: int) -> np.ndarray:
    """[N, N] — wiersz i uśrednia węzły, z których krawędź WCHODZI do i (upstream).
    Węzeł bez sąsiadów dostaje zera, nie dzielenie przez zero."""
    a = np.zeros((n, n), dtype=np.float32)
    a[edge_index[1], edge_index[0]] = 1.0
    deg = a.sum(1, keepdims=True)
    return np.divide(a, deg, out=np.zeros_like(a), where=deg > 0)


def build_inputs(x: np.ndarray, win: np.ndarray, adj: np.ndarray | None) -> np.ndarray:
    """[T, N, F] -> [T, N, k·F (+F)] — okno własnych odczytów (+ średnia sąsiadów)."""
    parts = [x[win[:, j]] for j in range(win.shape[1])]
    if adj is not None:
        parts.append(np.einsum("ij,tjf->tif", adj, x))
    return np.concatenate(parts, -1).astype(np.float32)


# ------------------------------------------------------------- geometria


def _intersections(roadnet: Path):
    """Obiekty `Intersection` LibSignala z samego pliku, bez silnika — ten sam trik
    co `tests/conftest.py`: parsowanie ruchów idzie kodem, który biegnie w symulacji."""
    from tsc_marl import libsignal  # noqa: PLC0415

    libsignal.bootstrap()
    from world.world_cityflow import Intersection  # noqa: PLC0415

    class _Eng:
        def set_tl_phase(self, *a):
            pass

    class _World:
        RIGHT = True
        eng = _Eng()

    raw = json.loads(roadnet.read_text())
    return raw, {d["id"]: Intersection(d, _World()) for d in raw["intersections"]}


def roadnet_of(network: str) -> Path:
    from tsc_marl.paths import CONFIGS, REPO_ROOT  # noqa: PLC0415

    cfg = json.loads((CONFIGS / "sim" / f"{network}.cfg").read_text())
    return REPO_ROOT / "data" / cfg["roadnetFile"]


def movement_sources(network: str, n_mov: int, drop_always_green: bool = True) -> np.ndarray:
    """[N, n_mov] — indeks węzła, z którego wjeżdża ruch (l,m); −1 = brzeg sieci.

    To jest mapowanie ruch → krawędź z planu: droga wjazdowa ruchu zaczyna się
    w sąsiedzie j, więc ruch należy do krawędzi j→i. Kolejność ruchów taka jak
    w `StateBuilder` (`ats.controlled_movements`), kolejność węzłów — grafu."""
    from tsc_marl.core.graph import RoadNetGraph  # noqa: PLC0415
    from tsc_marl.domain import ats  # noqa: PLC0415

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
    """MLP wspólny dla wszystkich węzłów: [.., D] -> (μ, σ) [.., n_feat]."""

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
        return (build_inputs(x, win, adj) - self.in_mean) / self.in_std

    @torch.no_grad()
    def predict(self, data: Loaded, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """-> (μ, σ) w jednostkach SUROWYCH cech, [T, N, n_feat]."""
        self.model.eval()
        z = torch.from_numpy(self.features(data, x).astype(np.float32))
        mu, sigma = self.model(z)
        return mu.numpy() * self.y_std + self.y_mean, sigma.numpy() * self.y_std


def fit(data: Loaded, cfg: ObsConfig | None = None) -> tuple[Fitted, list[dict]]:
    cfg = cfg or ObsConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    n_feat = data.ob_length - data.n_phases  # blok EP (+ demand) — bez one-hotu fazy

    win = window_index(data.episode, cfg.window)
    adj = neighbor_matrix(data.edge_index, data.n_nodes) if cfg.neighbors else None
    train = data.mask(cfg.split, train=True)
    zs = [build_inputs(data.x[s], win, adj)[train] for s in cfg.train_scenarios]
    z = np.concatenate(zs).reshape(-1, zs[0].shape[-1])
    y = np.concatenate([data.x_clean[train][..., :n_feat]] * len(zs)).reshape(-1, n_feat)

    # standaryzacja z CZYSTEGO zbioru treningowego — σ wychodzi wtedy w jednostkach
    # „odchylenia cechy na czystym", porównywalnych między cechami
    clean_z = build_inputs(data.x_clean, win, adj)[train].reshape(-1, z.shape[-1])
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


# ---------------------------------------------------------------- metryki


def auroc(pos: np.ndarray, neg: np.ndarray, cap: int = 200_000, seed: int = 0) -> float:
    """AUROC przez rangi (Mann–Whitney), remisy uśredniane."""
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
    """[.., n_feat] -> [.., n_mov]: max σ po cechach ruchu (EP i, jeśli jest, demand)."""
    blocks = [
        sigma_std[..., b * n_mov : (b + 1) * n_mov] for b in range(sigma_std.shape[-1] // n_mov)
    ]
    return np.max(np.stack(blocks), axis=0)


def edge_uncertainty(u_mov: np.ndarray, sources: np.ndarray, how: str = "max"):
    """u ruchów -> u krawędzi (j→i). -> ([T, E], lista (i, j)). Ruchy z brzegu
    sieci (−1) nie należą do żadnej krawędzi grafu."""
    edges = sorted({(i, int(j)) for i in range(sources.shape[0]) for j in sources[i] if j >= 0})
    agg = np.max if how == "max" else np.mean
    cols = [agg(u_mov[:, i, sources[i] == j], axis=-1) for i, j in edges]
    return np.stack(cols, axis=-1), edges


def corrupted_movements(x, x_clean, dead, n_mov: int) -> np.ndarray:
    """[T, N, n_mov] — ruch z martwym detektorem ALBO z błędem cechy > CORRUPT_TAU.
    `x`, `x_clean` to już same bloki cech ruchów (bez one-hotu fazy)."""
    err = np.abs(x - x_clean)
    big = np.zeros(x.shape[:-1] + (n_mov,), dtype=bool)
    for b in range(err.shape[-1] // n_mov):
        big |= err[..., b * n_mov : (b + 1) * n_mov] > CORRUPT_TAU
    return big | dead.astype(bool)


def evaluate(fitted: Fitted, data: Loaded, sources: np.ndarray, scenarios=None) -> dict[str, Any]:
    """Metryki na OGONIE epizodu dla każdego scenariusza zbioru."""
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


# ------------------------------------------------------------------ bramka


def _gauss_series(res: dict[str, Any]) -> list[tuple[float, float]]:
    pts = [(0.0, res["clean"]["u_mean"])]
    for name, r in res.items():
        if name.startswith("gauss"):
            pts.append((float(name[5:].replace("p", ".")), r["u_mean"]))
    return sorted(pts)


def gate(res: dict[str, Any], sep: dict[str, Any]) -> dict[str, Any]:
    gs = _gauss_series(res)
    unseen = [res[n]["auroc_mov"] for n in UNSEEN if n in res]
    seen = [res[n]["auroc_mov"] for n in ("gauss1", "gauss2", "mask0p25") if n in res]
    drift = [res[n]["auroc_mov"] for n in res if n.startswith("drift")]
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
    """Tabela 2×2 z Załącznika A7: {czyste, popsute} × {ruch treningowy, OOD}.

    `u_obs` ma reagować na WIERSZ (awaria), `u_ep` na KOLUMNĘ (nieznany ruch).
    Jeśli oba reagują na wszystko — nie ma dekompozycji, tylko jeden sygnał
    w dwóch przebraniach, a dwupoziomowy fallback traci uzasadnienie."""
    out: dict[str, Any] = {}
    sets = {"trening": base, **ood}
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
    return "—" if v is None or not np.isfinite(v) else format(v, f)


def to_markdown(res: dict[str, Any], g: dict[str, Any], cfg: ObsConfig, ood_res: dict) -> str:
    ok = {True: "TAK", False: "NIE"}
    lines = [
        "# Bramka prototypu 2.2 — głowa OBSERWACYJNA (niepewność cech wejściowych)",
        "",
        f"Okno {cfg.window} odczytów{' + średnia sąsiadów upstream' if cfg.neighbors else ''}, "
        f"{cfg.epochs} epok, Gaussian NLL względem `x_clean`, trening na: "
        f"{', '.join(cfg.train_scenarios)}. Ewaluacja na ogonie epizodu. u = σ w jednostkach "
        "odchylenia cechy na czystym; u ruchu = max po jego cechach (EP, demand).",
        "",
        "| scenariusz | tren. | popsute ruchy | u średnie | AUROC ruch | AUROC krawędź (max) "
        "| AUROC krawędź (mean) | pokrycie 90% | RMSE wejścia | RMSE μ |",
        "|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in res.items():
        lines.append(
            f"| {name} | {'x' if name in cfg.train_scenarios else ''} | {r['corrupt_share']:.1%} "
            f"| {r['u_mean']:.3f} | {_fmt(r['auroc_mov'])} | {_fmt(r['auroc_edge_max'])} "
            f"| {_fmt(r['auroc_edge_mean'])} | {r['coverage90']:.1%} | {r['rmse_input']:.3f} "
            f"| {r['rmse_mu']:.3f} |"
        )
    if ood_res:
        lines += ["", "### Te same metryki na zbiorach OOD (ruch spoza treningu)", "",
                  "| zbiór | scenariusz | u średnie | AUROC ruch | pokrycie 90% |",
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
    labels = sorted({k.split("|")[0] for k in sep}, key=lambda s: (s != "trening", s))
    lines += [
        "",
        "## Separacja 2×2 — czy dwa kanały mierzą dwie różne rzeczy",
        "",
        "`u_obs` = średnie u głowy obserwacyjnej, `u_ep` = epistemic deep ensemble "
        "(MI między członkami, prototyp 2.0). Wiersz = stan odczytów, kolumna = ruch.",
        "",
        "| odczyty \\ ruch | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    for scen in ("clean", "gauss2", "mask0p5"):
        cells = []
        for lab in labels:
            v = sep.get(f"{lab}|{scen}")
            cells.append(
                "—" if v is None else f"u_obs {v['u_obs']:.3f} · u_ep {_fmt(v['u_ep'], '.4f')}"
            )
        lines.append(f"| {scen} | " + " | ".join(cells) + " |")
    lines += ["", *_separation_findings(sep, labels)]
    gs = g["a_series"]
    lines += [
        "",
        "## Kryteria wyjścia",
        "",
        f"* **(a) monotoniczność** — u po σ szumu ({', '.join(f'{s:g}' for s, _ in gs)}): "
        f"{', '.join(f'{v:.3f}' for _, v in gs)} → **{ok[g['a_monotonic']]}** "
        "(3,5 i 4,0 NIEWIDZIANE w treningu).",
        f"* **(c) kalibracja** — pokrycie przedziału 90% na czystym "
        f"{g['c_coverage_clean']:.1%}, na szumie widzianym {g['c_coverage_seen']:.1%}.",
        f"* **(d) detekcja na widzianym** — min AUROC ruchu (gauss1, gauss2, mask0p25): "
        f"**{_fmt(g['d_auroc_seen_min'])}** (próg 0,8).",
        f"* **(e) detekcja na NIEWIDZIANYM** — min AUROC po {', '.join(UNSEEN)}: "
        f"**{_fmt(g['e_auroc_unseen_min'])}** (próg 0,7); `drift` (inna rodzina): max AUROC "
        f"**{_fmt(g['e_auroc_drift_max'])}**.",
        f"* **(f) agregacja ruch→krawędź** — AUROC(max) − AUROC(mean), średnio: "
        f"{g['f_edge_max_vs_mean']:+.3f} → w attention "
        f"**{'max' if g['f_edge_max_vs_mean'] >= 0 else 'mean'}**.",
        "",
        f"**Bramka 2.2 (głowa obserwacyjna): {'ZALICZONA' if g['passed'] else 'NIEZALICZONA'}.**",
        "",
        "## Czego ta tabela NIE mówi",
        "",
        "* Prototyp jest NADZOROWANY (symulator zna `x_clean`). W pętli sterowania głowa działa "
        "tak samo — trening offline na trajektoriach zamrożonej polityki, inferencja bez prawdy.",
        "* AUROC porównuje komórki popsute z CZYSTEGO scenariusza — nie mierzy lokalizacji "
        "w obrębie jednego epizodu (tam, gdzie wszystkie detektory szumią, nie ma czego "
        "lokalizować).",
        f"* „Popsuty ruch” = martwy detektor albo błąd cechy > {CORRUPT_TAU} pojazdu. Dryf małych "
        "zliczeń rzadko przekracza próg, więc jego AUROC liczy się na nielicznych komórkach.",
    ]
    return "\n".join(lines) + "\n"


def _separation_findings(sep: dict[str, Any], labels: list[str]) -> list[str]:
    def val(lab, scen, key):
        v = sep.get(f"{lab}|{scen}")
        return float("nan") if v is None else v[key]

    ood = [lab for lab in labels if lab != "trening"]
    if not ood:
        return []
    row_obs = np.nanmean([val(lab, "gauss2", "u_obs") - val(lab, "clean", "u_obs")
                          for lab in ["trening", *ood]])  # fmt: skip
    col_obs = np.nanmean(
        [val(lab, "clean", "u_obs") - val("trening", "clean", "u_obs") for lab in ood]
    )
    row_ep = np.nanmean([val(lab, "gauss2", "u_ep") - val(lab, "clean", "u_ep")
                         for lab in ["trening", *ood]])  # fmt: skip
    col_ep = np.nanmean(
        [val(lab, "clean", "u_ep") - val("trening", "clean", "u_ep") for lab in ood]
    )
    out = [
        f"* `u_obs`: przyrost od awarii (wiersz) **{row_obs:+.3f}**, od OOD (kolumna) "
        f"**{col_obs:+.3f}**.",
        f"* `u_ep`: przyrost od awarii **{row_ep:+.4f}**, od OOD **{col_ep:+.4f}**.",
    ]
    obs_ok = np.isfinite(row_obs) and row_obs > 3 * max(abs(col_obs), 1e-9)
    ep_ok = np.isfinite(col_ep) and col_ep > max(row_ep, 0)
    out.append(
        "* Werdykt separacji: "
        + ("`u_obs` reaguje na awarię, nie na OOD — " if obs_ok else "`u_obs` NIE jest czysty — ")
        + (
            "`u_ep` reaguje mocniej na OOD niż na awarię."
            if ep_ok
            else "`u_ep` NIE reaguje na OOD mocniej niż na awarię — dwupoziomowy fallback nie ma "
            "tu empirycznej podstawy na poziomie (a)."
        )  # fmt: skip
    )
    return out


def run(cfg: ObsConfig | None = None, *, network: str = "cityflow_jinan3x4_4ph") -> Path:
    from tsc_marl.paths import RESULTS, RUNS  # noqa: PLC0415
    from tsc_marl.uncertainty import prototype as P  # noqa: PLC0415

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
