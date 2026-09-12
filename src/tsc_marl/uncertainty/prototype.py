"""Prototyp walidacyjny 2.0 — uczenie nadzorowane głowy niepewności i BRAMKA.

Cel z planu: TANIO potwierdzić premisę („niepewność rośnie z awarią sensora")
i rozstrzygnąć metodę (EDL vs MC-dropout) zanim powstanie cały system. Wszystko
offline, na zrzucie z `dataset.py`, na CPU, w minutach.

„JEDNO SKRZYŻOWANIE" = PUSTY `edge_index`, nie inny model. Warstwa GAT dokłada
pętle własne sama, więc bez krawędzi każdy węzeł widzi wyłącznie siebie — mamy
izolację mechanizmu z 2.0 przy DOKŁADNIE tej architekturze, którą trenuje rdzeń.
Druga iteracja 2.0 (uncertainty-weighted attention) to `--hop`, nie nowy plik.

PODZIAŁ DANYCH IDZIE PO CZASIE, i to nie jest kosmetyka. Ruch w `flow.json` jest
deterministyczny, więc epizody mają IDENTYCZNĄ trajektorię (różnią się tylko
losowaniem awarii) — podział „po epizodach" byłby przeciekiem co do bitu,
a podział losowy przeciekiem przez korelację sąsiednich decyzji (15 s odstępu).
Zostaje pierwsze `split` epizodu na trening, ogon na ewaluację; to jest też
uczciwsze, bo ogon epizodu ma inne obciążenie niż początek.

POZIOMY SZUMU W TRENINGU SĄ ROZDZIELONE OD TESTOWYCH (bramka 2.0e): głowa uczy
się na `gauss` 1,0/2,0, a bramkę zdaje na 3,5/4,0, na `urand` i na maskach —
czyli na szumie, którego nie widziała. Bez tego mierzylibyśmy kalibrowany
detektor znanych poziomów i nazywali go estymatorem niepewności.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

#: Scenariusze treningowe głowy. Maska 25% JEST w treningu, i to jest wynik
#: prototypu, nie założenie: bez niej głowa decyzyjna nie reaguje na maskowanie
#: w ogóle (patrz `results/gate_2_0.md`). Trzymamy poza treningiem `gauss` 3,5/4,0,
#: `urand`, maskę 50%, maskę całego skrzyżowania i CAŁY `drift` — czyli inną
#: dotkliwość, inną jednostkę awarii i inną RODZINĘ awarii.
TRAIN_SCENARIOS = ("clean", "gauss1", "gauss2", "mask0p25")
#: Zbiory OOD dla epistemic (bramka 2.0 b), trzy RODZAJE oddalenia od treningu:
#:   d*  — ten sam wzorzec ruchu, przeskalowany W DÓŁ (JiNan 4-fazowy korkuje się
#:         już przy 1,1×, patrz `domain/demand.py`). Uwaga: przy niskim popycie
#:         rośnie liczba REMISÓW faz, więc przyrost niepewności decyzji może być
#:         ambiwalencją, nie OOD — stąd kolumny liczone tylko na próbkach rozstrzygalnych;
#:   r*  — INNY wzorzec ruchu z pomiarów (pliki `_2000`/`_2500` z Advanced_XLight);
#:   hz  — inna TOPOLOGIA i obciążenie (HangZhou 4×4), te same semantyki cech.
OOD_NETWORKS = {
    "d0p75": "cityflow_jinan3x4_4ph_d0p75",
    "d0p5": "cityflow_jinan3x4_4ph_d0p5",
    "d0p25": "cityflow_jinan3x4_4ph_d0p25",
    "r2000": "cityflow_jinan3x4_4ph_r2000",
    "r2500": "cityflow_jinan3x4_4ph_r2500",
    "hz": "cityflow_hangzhou4x4_4ph",
}
OOD_GROUPS = {
    "popyt ×": ("d0p75", "d0p5", "d0p25"),
    "wzorzec": ("r2000", "r2500"),
    "topologia": ("hz",),
}
TIE_EPS = 1e-6


def cap_threads(limit: int = 4) -> int:
    """Wątki torcha dla prototypów na CPU. Sieci mają kilkadziesiąt tysięcy wag, a batch
    to 32 kroki — przy domyślnych N wątkach synchronizacja kosztuje więcej niż rachunek,
    a dwa takie procesy obok siebie (albo obok treningu) kręcą się w aktywnym oczekiwaniu
    OpenMP: 2026-09-11 bramki, które liczą się w minuty, stały 15 h na 800% CPU.
    Jawne OMP_NUM_THREADS w środowisku wygrywa."""
    import os  # noqa: PLC0415

    n = int(os.environ.get("OMP_NUM_THREADS") or min(limit, os.cpu_count() or 1))
    torch.set_num_threads(n)
    return n


@dataclass(frozen=True)
class TrainConfig:
    mode: str = "edl"  # "edl" (deployowana) albo "softmax" (baza pod MC-dropout)
    epochs: int = 150
    lr: float = 1e-3
    batch: int = 32  # w KROKACH CZASU, nie w wierszach — jeden krok to cały graf
    dropout: float = 0.1
    #: `anneal` regulatora KL rośnie liniowo przez tyle epok. Włączony od razu
    #: zabija evidence, zanim model zdąży się czegokolwiek nauczyć (Sensoy 2018).
    anneal_epochs: int = 10
    split: float = 0.67  # ułamek epizodu na trening (podział PO CZASIE)
    train_scenarios: tuple[str, ...] = TRAIN_SCENARIOS
    hop: bool = False  # False = 0-hop („jedno skrzyżowanie" z 2.0)
    mc_passes: int = 10  # K dla MC-dropout ([[kirthan-2026-ua-gnn]])
    #: Deep ensemble — WZORZEC epistemic dla bramki (d). MC-dropout ostatniej warstwy
    #: dawał epistemic rzędu 0,004, czyli korelacja z nim była korelacją z szumem.
    #: Ensemble to akceptowany gold standard (Lakshminarayanan 2017) i kosztuje K
    #: treningów po ~40 s CPU. 0 = wyłączony.
    ensemble: int = 5
    seed: int = 0
    weight_decay: float = 0.0


@dataclass
class Loaded:
    """Zrzut wczytany z npz. `times` to indeks decyzji WEWNĄTRZ epizodu — na nim
    stoi podział po czasie."""

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


# ------------------------------------------------------------------ trening


def fit(data: Loaded, cfg: TrainConfig | None = None) -> tuple[Any, list[dict]]:
    """-> (model, historia). Wszystko na CPU: sieć ma kilkadziesiąt tysięcy wag."""
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
    """K członków `softmax` o różnych ziarnach — różnią się TYLKO inicjalizacją
    i kolejnością minibatchy, więc ich rozbieżność mierzy niepewność modelu."""
    from dataclasses import replace  # noqa: PLC0415

    return [
        fit(data, replace(cfg, mode="softmax", seed=cfg.seed + 1000 + k))[0]
        for k in range(cfg.ensemble)
    ]


@torch.no_grad()
def ensemble_uncertainty(models: list[Any], x: np.ndarray, data: Loaded, cfg: TrainConfig):
    """Ta sama dekompozycja co MC-BALD: total = H(p̄), aleatoric = E H(p), epistemic = MI."""
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
    """0-hop = JAWNE pętle własne, NIE pusty `edge_index`.

    Pusty wygląda na oczywisty sposób odcięcia sąsiadów i jest pułapką:
    `MultiHeadAttModel` woła `add_self_loops(edge_index)` BEZ `num_nodes`, więc
    z pustego wejścia wnioskuje zero węzłów, nie dokłada niczego i agregacja
    zwraca SAME ZERA — model przestaje zależeć od wejścia, a mimo to trenuje się
    bez błędu (uczy się samego biasu). Kosztowało to jeden pełny przebieg bramki,
    w którym wszystkie scenariusze miały identyczną niepewność.

    Podwójna pętla własna (nasza + dołożona przez warstwę) jest tu nieszkodliwa:
    softmax po dwóch identycznych wpisach daje 0,5 + 0,5, czyli dokładnie to samo
    co jedna pętla o wadze 1."""
    if not cfg.hop:
        loop = torch.arange(data.n_nodes, dtype=torch.long)
        return torch.stack([loop, loop])
    return torch.from_numpy(data.edge_index)


def _tiled(edge_index: torch.Tensor, n_nodes: int, k: int) -> torch.Tensor:
    from tsc_marl.core.ppo import tiled_edge_index

    return tiled_edge_index(edge_index, n_nodes, k)


# ---------------------------------------------------------------- ewaluacja


@torch.no_grad()
def predict(
    model, x: np.ndarray, data: Loaded, cfg: TrainConfig, ensemble: list[Any] | None = None
) -> dict[str, np.ndarray]:
    """Niepewność EDL (single-pass) i MC-dropout (K przejść) na tych samych
    wejściach — bramka 2.0(d) porównuje DOKŁADNIE te dwa słowniki."""
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
    """Dla trybu `softmax` pojedyncze przejście nie rozdziela niepewności —
    całość idzie w aleatoric, epistemic zostaje zerem. Zapisane wprost, żeby
    tabela nie sugerowała, że softmax coś rozdziela."""
    p = torch.softmax(logits, -1)
    total = -(p * torch.log(p + 1e-10)).sum(-1)
    return {"prob": p, "total": total, "aleatoric": total, "epistemic": torch.zeros_like(total)}


# ------------------------------------------------------------------- bramka


def scenario_table(
    model,
    data: Loaded,
    cfg: TrainConfig,
    *,
    label: str = "",
    only: tuple[str, ...] = (),
    ensemble: list[Any] | None = None,
) -> list[dict]:
    """Jeden wiersz na scenariusz: trafność, kalibracja i kanały niepewności,
    liczone na OGONIE epizodu (dane niewidziane w treningu).

    Kolumny `*_dec` liczone WYŁĄCZNIE na próbkach rozstrzygalnych (najlepsza faza
    ma p(s) ostro większe od drugiej). Remis to poprawna, wysoka niepewność
    DECYZJI bez żadnego związku z awarią czy OOD — a jego udział zależy od
    obciążenia sieci, więc bez tego rozbicia przyrost niepewności na niskim
    popycie mógłby być samym przyrostem remisów."""
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
                "trenowany": name in cfg.train_scenarios and not label,
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
    """Kryteria wyjścia 2.0 policzone z tabeli — jawnie, żeby „bramka zaliczona"
    było stwierdzeniem o liczbach, a nie wrażeniem z wykresu."""
    nan = float("nan")
    by = {(r["dataset"], r["scenario"]): r for r in rows}
    base = [k for k in by if not k[0]]

    def series(prefix: str, key: str) -> list[float]:
        names = [s for _, s in base if s.startswith(prefix)]
        names.sort(key=lambda n: float(n[len(prefix) :].replace("p", ".") or 0))
        return [by[("", n)][key] for n in names]

    # ciąg dotkliwości zaczyna się od sigma=0, inaczej „monotoniczność" pomijałaby
    # najważniejszy punkt odniesienia — czysty stan
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
        # (a) monotoniczność: niepewność rośnie z dotkliwością szumu
        "a_monotonic_gauss": gauss == sorted(gauss),
        "a_gauss_series": gauss,
        # (b) rozdzielność: aleatoric na szumie, epistemic na OOD
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
        # trafność SPADA dalej, choć niepewność już nie rośnie — miara „pewny i w błędzie"
        "f_acc_drop_unseen": clean["acc"] - by[("", "gauss4")]["acc"],
        "f_ece_unseen": by[("", "gauss4")]["ece"],
        # (c) kalibracja
        "c_ece_clean": clean["ece"],
        "c_ece_worst": max(r["ece"] for r in rows),
        # (d) tania separacja EDL zgadza się z drogim wzorcem
        "d_edl_vs_mc_corr": _corr(edl, [r["mc_epistemic"] for r in rows]),
        "d_edl_vs_ens_corr": _corr(edl, [r.get("ens_epistemic", nan) for r in rows]),
        "d_reference": "ensemble" if has_ens else "mc-dropout (ostatnia warstwa)",
        # (e) generalizacja na to, czego głowa NIE widziała w treningu
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
        # (g) czy kanał epistemic EDL niesie WŁASNĄ informację, czy jest stałym
        # ułamkiem total — wtedy epistemic, vacuity i total to jeden sygnał
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
    return "—" if v is None or not np.isfinite(v) else format(v, fmt)


def to_markdown(rows: list[dict], g: dict[str, Any], cfg: TrainConfig, meta: dict) -> str:
    ok = {True: "TAK", False: "NIE"}
    ref = g["d_reference"]
    out = [
        "# Bramka prototypu 2.0 — niepewność DECYZJI vs awaria sensora i OOD",
        "",
        f"Tryb `{cfg.mode}`, {'1-hop' if cfg.hop else '0-hop (jedno skrzyżowanie)'}, "
        f"{cfg.epochs} epok, trening na: {', '.join(cfg.train_scenarios)}.",
        f"Ewaluacja na OGONIE epizodu (ostatnie {100 * (1 - cfg.split):.0f}% decyzji). "
        f"Wzorzec epistemic: **{ref}** (K={cfg.ensemble if cfg.ensemble else cfg.mc_passes}).",
        "",
        "| zbiór | scenariusz | tren. | acc | ECE | EDL total | EDL epist | epist/total "
        "| vacuity | ens epist | MC epist | remisy | EDL total (rozstrz.) | martwe ruchy |",
        "|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        ratio = r["edl_epistemic"] / r["edl_total"] if r["edl_total"] > 0 else float("nan")
        out.append(
            f"| {r['dataset'] or 'bazowy'} | {r['scenario']} | {'x' if r['trenowany'] else ''} "
            f"| {r['acc']:.3f} | {r['ece']:.3f} | {r['edl_total']:.3f} "
            f"| {r['edl_epistemic']:.4f} | {ratio:.3f} | {r['edl_vacuity']:.3f} "
            f"| {_f(r.get('ens_epistemic'), '.4f')} | {r['mc_epistemic']:.4f} "
            f"| {r.get('tie_share', float('nan')):.1%} | {_f(r.get('edl_total_dec'), '.3f')} "
            f"| {r['dead']:.1%} |"
        )

    out += [
        "",
        "## Przyrost niepewności na OOD (względem czystego bazowego), maksimum w grupie",
        "",
        "| grupa | EDL total | EDL total (rozstrz.) | EDL epist | EDL epist (rozstrz.) "
        "| ens epist | ens epist (rozstrz.) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, v in g["b_ood_groups"].items():
        out.append(
            f"| {name} | {_f(v['edl_total'], '+.3f')} | {_f(v['edl_total_dec'], '+.3f')} "
            f"| {_f(v['edl_epistemic'])} | {_f(v['edl_epistemic_dec'])} "
            f"| {_f(v['ens_epistemic'])} | {_f(v['ens_epistemic_dec'])} |"
        )
    out += [
        f"| *szum gauss2 (dla porównania)* | — | — | {_f(g['b_epistemic_gain_noise'])} | — "
        f"| {_f(g['b_ens_gain_noise'])} | — |",
        "",
        "## Kryteria wyjścia",
        "",
        f"* **(a) monotoniczność** — EDL total po `gauss` (σ = 0, 1, 2, 3,5, 4): "
        f"{', '.join(f'{v:.3f}' for v in g['a_gauss_series'])} → "
        f"**{ok[g['a_monotonic_gauss']]}**.",
        f"* **(b) rozdzielność** — aleatoric EDL na szumie widzianym "
        f"{_f(g['b_aleatoric_gain_seen'], '+.3f')}, niewidzianym "
        f"{_f(g['b_aleatoric_gain_unseen'], '+.3f')}; epistemic EDL na OOD "
        f"{_f(g['b_epistemic_gain_ood'])} vs na szumie {_f(g['b_epistemic_gain_noise'])}.",
        f"* **(c) kalibracja** — ECE na czystym {g['c_ece_clean']:.3f}, najgorszy scenariusz "
        f"{g['c_ece_worst']:.3f}.",
        f"* **(d) EDL ≈ wzorzec** — korelacja epistemic po wierszach: z ensemble "
        f"**{_f(g['d_edl_vs_ens_corr'], '.3f')}**, z MC-dropout ostatniej warstwy "
        f"{_f(g['d_edl_vs_mc_corr'], '.3f')}.",
        f"* **(e) niewidziane** — szum 3,5/4,0 i `urand` powyżej czystego: "
        f"**{ok[g['e_unseen_noise_above_clean']]}**; maska 50% i maska skrzyżowania: "
        f"**{ok[g['e_unseen_mask_above_clean']]}**; `drift`: **{ok[g['e_drift_above_clean']]}** "
        f"(Δ {g['e_drift_delta']:+.3f} przy spadku trafności o {g['e_drift_acc_drop']:.3f}).",
        f"* **(g) czy epistemic EDL to osobny kanał** — stosunek epistemic/total w "
        f"[{g['g_ratio_min']:.3f}, {g['g_ratio_max']:.3f}] we wszystkich {len(rows)} wierszach "
        f"(rozrzut {g['g_ratio_spread']:.1%} średniej) → "
        f"**{'JEDEN SYGNAŁ' if g['g_ratio_constant'] else 'osobny kanał'}**.",
        "",
        "## Werdykt",
        "",
        f"**Bramka 2.0 dla głowy DECYZYJNEJ: {'ZALICZONA' if g['passed'] else 'NIEZALICZONA'}.**",
        "",
    ]
    out += _verdict(g)
    out += [
        "",
        "## Czego ta tabela NIE mówi",
        "",
        f"`tie_rate` zrzutu bazowego = **{meta.get('tie_rate', float('nan')):.1%}** — w tylu "
        "decyzjach dwie fazy mają równe p(s) i wysoka niepewność DECYZJI jest poprawna, bez "
        "związku z awarią. Kolumna *remisy* pokazuje ich udział w każdym zbiorze: na niskim "
        "popycie rośnie, dlatego przyrosty OOD podajemy też na próbkach rozstrzygalnych.",
        "",
        "MC-dropout jest dropoutem OSTATNIEJ WARSTWY; jako wzorzec dla (d) służy deep ensemble "
        "(K członków `softmax`, różne ziarna), bo epistemic MC jest rzędu 0,004.",
    ]
    return "\n".join(out) + "\n"


def _verdict(g: dict[str, Any]) -> list[str]:
    """Wnioski wyprowadzone z liczb — tekst zmienia się, gdy zmienią się dane."""
    lines = []
    gs = g["a_gauss_series"]
    if not g["a_monotonic_gauss"]:
        peak = int(np.argmax(gs))
        lines.append(
            f"1. **Poza zakresem treningowym niepewność decyzji przestaje rosnąć** (szczyt przy "
            f"indeksie {peak} ciągu {', '.join(f'{v:.3f}' for v in gs)}), choć trafność spada "
            f"dalej (o {g['f_acc_drop_unseen']:.3f}), a ECE rośnie do {g['f_ece_unseen']:.3f}: "
            "pewny i w błędzie dokładnie tam, gdzie fallback ma zadziałać."
        )
    if not g["e_drift_above_clean"]:
        lines.append(
            f"2. **`drift` jest niewidoczny** dla niepewności decyzji (Δ {g['e_drift_delta']:+.3f} "
            f"przy spadku trafności o {g['e_drift_acc_drop']:.3f})."
        )
    if g["g_ratio_constant"]:
        lines.append(
            f"3. **Epistemic EDL nie jest osobnym kanałem**: epistemic/total mieści się w "
            f"[{g['g_ratio_min']:.3f}, {g['g_ratio_max']:.3f}] w każdym wierszu, więc epistemic, "
            "vacuity i total to ten sam sygnał w różnej skali. Przyrost „epistemic na OOD” "
            "jest przyrostem total, nie osobną informacją o nieznanym stanie."
        )
    ens_ood = {k: v["ens_epistemic_dec"] for k, v in g["b_ood_groups"].items()}
    ens_noise = g["b_ens_gain_noise"]
    if all(not np.isfinite(v) for v in ens_ood.values()):
        lines.append("4. Wzorzec ensemble niepoliczony — kanał epistemic bez rozstrzygnięcia.")
    else:
        best = max((k for k in ens_ood if np.isfinite(ens_ood[k])), key=lambda k: ens_ood[k])
        if ens_ood[best] > max(ens_noise, 0) * 1.5:
            lines.append(
                f"4. **Kanał epistemic → MaxPressure ma podstawę w ENSEMBLE, nie w EDL**: "
                f"epistemic ensemble rośnie najmocniej na grupie „{best}” "
                f"({ens_ood[best]:+.4f} na próbkach rozstrzygalnych) wobec {ens_noise:+.4f} "
                "na szumie. Rozdział źródeł działa dla wzorca, więc do fallbacku (a) idzie "
                "sygnał rozbieżności modeli, a nie vacuity."
            )
        else:
            lines.append(
                f"4. **Nawet ensemble nie oddziela OOD od szumu** (najlepsza grupa „{best}”: "
                f"{ens_ood[best]:+.4f} vs szum {ens_noise:+.4f}). Kanał epistemic → MaxPressure "
                "nie ma podstawy empirycznej na tych zbiorach — fallback jednopoziomowy "
                "(tylko obserwacyjny) jest uczciwszym planem."
            )
    lines.append(
        "5. **Wniosek dla 2.2 (bez zmian):** sygnał per-edge idzie z heteroscedastycznej głowy "
        "na CECHACH WEJŚCIOWYCH (`results/gate_2_2.md`), nie ze spreadu Dirichleta."
    )
    return lines


def run(cfg: TrainConfig | None = None, *, network: str = "cityflow_jinan3x4_4ph") -> Path:
    """Pełna bramka 2.0: trening + tabela + kryteria -> `results/gate_2_0.md`."""
    import json

    from tsc_marl import libsignal
    from tsc_marl.paths import RESULTS, RUNS

    libsignal.bootstrap()  # `heads` -> `core.nets` ciągnie `agent.colight` z submodułu
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
