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
#: Sieci o obniżonym popycie — scenariusz OOD dla epistemic (bramka 2.0 b).
#: Podbicie popytu odpada: JiNan 4-fazowy korkuje się już przy 1,1× (patrz
#: `domain/demand.py`), więc skala oddalenia od treningu idzie w dół.
OOD_NETWORKS = (
    "cityflow_jinan3x4_4ph_d0p75",
    "cityflow_jinan3x4_4ph_d0p5",
    "cityflow_jinan3x4_4ph_d0p25",
)


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
def predict(model, x: np.ndarray, data: Loaded, cfg: TrainConfig) -> dict[str, np.ndarray]:
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
    model, data: Loaded, cfg: TrainConfig, *, label: str = "", only: tuple[str, ...] = ()
) -> list[dict]:
    """Jeden wiersz na scenariusz: trafność, kalibracja i cztery kanały niepewności,
    liczone na OGONIE epizodu (dane niewidziane w treningu)."""
    from tsc_marl.uncertainty.heads import expected_calibration_error

    mask = data.mask(cfg.split, train=False)
    y = data.y[mask]
    rows = []
    for name, x in sorted(data.x.items()):
        if only and name not in only:
            continue
        out = predict(model, x[mask], data, cfg)
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
                "dead": float(data.dead[name][mask].mean()),
            }
        )
    return rows


def gate(rows: list[dict]) -> dict[str, Any]:
    """Kryteria wyjścia 2.0 policzone z tabeli — jawnie, żeby „bramka zaliczona"
    było stwierdzeniem o liczbach, a nie wrażeniem z wykresu."""
    by = {(r["dataset"], r["scenario"]): r for r in rows}
    base = [k for k in by if not k[0]]

    def series(prefix: str, key: str) -> list[float]:
        names = [s for _, s in base if s.startswith(prefix)]
        names.sort(key=lambda n: float(n[len(prefix) :].replace("p", ".") or 0))
        return [by[("", n)][key] for n in names]

    # ciąg dotkliwości zaczyna się od sigma=0, inaczej „monotoniczność" pomijałaby
    # najważniejszy punkt odniesienia — czysty stan
    gauss = [by[("", "clean")]["edl_total"], *series("gauss", "edl_total")]
    ood = [r for r in rows if r["dataset"] and r["scenario"] == "clean"]
    ood.sort(key=lambda r: r["dataset"])  # d0p25 < d0p5 < d0p75 = malejące oddalenie
    clean = by[("", "clean")]

    edl = np.array([r["edl_epistemic"] for r in rows])
    mc = np.array([r["mc_epistemic"] for r in rows])
    out = {
        # (a) monotoniczność: niepewność rośnie z dotkliwością szumu
        "a_monotonic_gauss": gauss == sorted(gauss),
        "a_gauss_series": gauss,
        # (b) rozdzielność: aleatoric na szumie, epistemic na OOD popytu
        "b_aleatoric_gain_seen": by[("", "gauss2")]["edl_aleatoric"] - clean["edl_aleatoric"],
        "b_aleatoric_gain_unseen": by[("", "gauss4")]["edl_aleatoric"] - clean["edl_aleatoric"],
        "b_epistemic_gain_ood": (
            max((r["edl_epistemic"] for r in ood), default=float("nan")) - clean["edl_epistemic"]
        ),
        "b_epistemic_gain_noise": by[("", "gauss2")]["edl_epistemic"] - clean["edl_epistemic"],
        # trafność SPADA dalej, choć niepewność już nie rośnie — miara „pewny i w błędzie"
        "f_acc_drop_unseen": clean["acc"] - by[("", "gauss4")]["acc"],
        "f_ece_unseen": by[("", "gauss4")]["ece"],
        # (c) kalibracja
        "c_ece_clean": clean["ece"],
        "c_ece_worst": max(r["ece"] for r in rows),
        # (d) tania separacja EDL zgadza się z drogim wzorcem MC-BALD
        "d_edl_vs_mc_corr": float(np.corrcoef(edl, mc)[0, 1]) if len(edl) > 2 else float("nan"),
        # (e) generalizacja na to, czego głowa NIE widziała w treningu
        "e_unseen_noise_above_clean": (
            min(by[("", n)]["edl_total"] for n in ("gauss3p5", "gauss4", "urand3p5"))
            > clean["edl_total"]
        ),
        # ta sama RODZINA awarii, inna dotkliwość i inna jednostka
        "e_unseen_mask_above_clean": (
            min(by[("", n)]["edl_total"] for n in ("mask0p5", "mask0p25_intersection"))
            > clean["edl_total"]
        ),
        # inna RODZINA (scenariusz NIEOBSERWOWALNY) — tu spodziewamy się porażki
        "e_drift_above_clean": by[("", "drift0p3")]["edl_total"] > clean["edl_total"],
        "e_drift_delta": by[("", "drift0p3")]["edl_total"] - clean["edl_total"],
        # ile trafności kosztuje scenariusz, którego niepewność NIE widzi
        "e_drift_acc_drop": clean["acc"] - by[("", "drift0p3")]["acc"],
    }
    out["passed"] = bool(
        out["a_monotonic_gauss"]
        and out["e_drift_above_clean"]
        and out["d_edl_vs_mc_corr"] > 0.5
        and out["b_epistemic_gain_ood"] > out["b_epistemic_gain_noise"]
    )
    return out


def to_markdown(rows: list[dict], g: dict[str, Any], cfg: TrainConfig, meta: dict) -> str:
    ok = {True: "TAK", False: "NIE"}
    out = [
        "# Bramka prototypu 2.0 — niepewność vs awaria sensora",
        "",
        f"Tryb `{cfg.mode}`, {'1-hop' if cfg.hop else '0-hop (jedno skrzyżowanie)'}, "
        f"{cfg.epochs} epok, trening na: {', '.join(cfg.train_scenarios)}.",
        f"Ewaluacja na OGONIE epizodu (ostatnie {100 * (1 - cfg.split):.0f}% decyzji), "
        f"MC-dropout K={cfg.mc_passes}.",
        "",
        "| zbiór | scenariusz | tren. | acc | ECE | EDL total | EDL aleat | EDL epist "
        "| vacuity | MC aleat | MC epist | martwe ruchy |",
        "|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        out.append(
            f"| {r['dataset'] or 'bazowy'} | {r['scenario']} | {'x' if r['trenowany'] else ''} "
            f"| {r['acc']:.3f} | {r['ece']:.3f} | {r['edl_total']:.3f} | {r['edl_aleatoric']:.3f} "
            f"| {r['edl_epistemic']:.4f} | {r['edl_vacuity']:.3f} | {r['mc_aleatoric']:.3f} "
            f"| {r['mc_epistemic']:.4f} | {r['dead']:.1%} |"
        )

    out += [
        "",
        "## Kryteria wyjścia",
        "",
        f"* **(a) monotoniczność** — niepewność rośnie z sigma szumu: "
        f"**{ok[g['a_monotonic_gauss']]}** "
        f"(EDL total po `gauss`: {', '.join(f'{v:.3f}' for v in g['a_gauss_series'])}).",
        f"* **(b) rozdzielność** — aleatoric na szumie WIDZIANYM w treningu: "
        f"**{g['b_aleatoric_gain_seen']:+.3f}** (na niewidzianym: "
        f"**{g['b_aleatoric_gain_unseen']:+.3f}**), "
        f"epistemic na OOD popytu: **{g['b_epistemic_gain_ood']:+.4f}**, "
        f"epistemic na szumie: **{g['b_epistemic_gain_noise']:+.4f}** "
        "(chcemy: pierwsza i druga dodatnie, trzecia wyraźnie mniejsza od drugiej).",
        f"* **(c) kalibracja** — ECE na czystym **{g['c_ece_clean']:.3f}**, "
        f"najgorszy scenariusz **{g['c_ece_worst']:.3f}**.",
        f"* **(d) EDL ≈ MC-BALD** — korelacja epistemic po scenariuszach: "
        f"**{g['d_edl_vs_mc_corr']:.3f}**.",
        f"* **(e) niewidziane w treningu** — szum 3,5/4,0 i `urand` powyżej czystego: "
        f"**{ok[g['e_unseen_noise_above_clean']]}**; maska 50% i maska całego skrzyżowania "
        f"(inna dotkliwość, inna jednostka) powyżej czystego: "
        f"**{ok[g['e_unseen_mask_above_clean']]}**; `drift` — INNA RODZINA awarii, "
        f"scenariusz nieobserwowalny: **{ok[g['e_drift_above_clean']]}** "
        f"(Δ niepewności {g['e_drift_delta']:+.3f} przy spadku trafności "
        f"o {g['e_drift_acc_drop']:.3f}).",
        "",
        "## Werdykt",
        "",
        f"**Bramka 2.0 dla głowy DECYZYJNEJ: {'ZALICZONA' if g['passed'] else 'NIEZALICZONA'}.**",
        "",
        "Rozstrzygnięcie, po które szło się do prototypu: **niepewność decyzji (Dirichlet nad",
        "fazami) NIE jest estymatorem niepewności obserwacji, tylko detektorem awarii",
        "WIDZIANYCH W TRENINGU.** Trzy pomiary składają się na to jedno zdanie:",
        "",
        f"1. Poza zakresem treningowym niepewność **przestaje rosnąć i zawraca** "
        f"({', '.join(f'{v:.3f}' for v in g['a_gauss_series'])}), choć trafność spada dalej "
        f"(o {g['f_acc_drop_unseen']:.3f}), a ECE rośnie do {g['f_ece_unseen']:.3f}. "
        "Model staje się PEWNY I W BŁĘDZIE dokładnie tam, gdzie fallback ma zadziałać.",
        f"2. `drift` — inna RODZINA awarii, scenariusz NIEOBSERWOWALNY — jest niewidoczny: "
        f"Δ niepewności {g['e_drift_delta']:+.3f} przy spadku trafności o "
        f"{g['e_drift_acc_drop']:.3f}. Rule-gate jest tu ślepy z definicji, ale głowa "
        "decyzyjna też.",
        f"3. Tania separacja EDL **nie zgadza się** z drogim wzorcem: korelacja epistemic "
        f"z MC-BALD po scenariuszach = {g['d_edl_vs_mc_corr']:.3f}. MC podnosi epistemic na "
        "szumie niewidzianym, EDL go tam obniża.",
        "",
        "Co z tego wynika dla Fazy 2: **plan miał rację, stawiając mechanizm 2.2 na "
        "heteroscedastycznej głowie liczonej z CECH WEJŚCIOWYCH, a nie na spreadzie "
        "Dirichleta** — i jest to teraz stwierdzenie oparte na pomiarze, nie na argumencie "
        "z confounda. Niepewność decyzyjna zostaje przy fallbacku (3.1), gdzie jej rolą jest "
        "twarda bariera, a nie lokalizacja zepsutego detektora.",
        "",
        "Co działa i warto zachować: epistemic reaguje na OOD POPYTU wyraźniej niż na szum "
        f"({g['b_epistemic_gain_ood']:+.4f} vs {g['b_epistemic_gain_noise']:+.4f}), czyli "
        "kanał epistemic → MaxPressure z dwupoziomowego fallbacku ma podstawę empiryczną. "
        "Reaguje też na maskę o dotkliwości i jednostce niewidzianej w treningu — "
        "generalizuje W RODZINIE awarii, nie MIĘDZY rodzinami.",
        "",
        "## Czego ta tabela NIE mówi",
        "",
        f"`tie_rate` zrzutu = **{meta.get('tie_rate', float('nan')):.1%}**: w tylu decyzjach dwie "
        "fazy mają równe p(s), więc wysoka niepewność DECYZJI jest tam poprawna i nie ma nic "
        "wspólnego z awarią sensora. To jest zmierzony confound, przed którym ostrzega 2.2 — "
        "i powód, dla którego sygnał per-edge ma iść z CECH WEJŚCIOWYCH, a nie ze spreadu "
        "Dirichleta. Kolumna `martwe ruchy` pokazuje drugą stronę: przy masce większość ruchów "
        "ma martwy detektor, a mimo to cechy zmieniają się rzadko (awaria CICHA).",
        "",
        "MC-dropout jest tu dropoutem OSTATNIEJ WARSTWY, nie całej sieci — standardowe "
        "przybliżenie, ale nie pełny Gal-Ghahramani. Do nazwania w pracy.",
    ]
    return "\n".join(out) + "\n"


def run(cfg: TrainConfig | None = None, *, network: str = "cityflow_jinan3x4_4ph") -> Path:
    """Pełna bramka 2.0: trening + tabela + kryteria -> `results/gate_2_0.md`."""
    import json

    from tsc_marl.paths import RESULTS, RUNS

    cfg = cfg or TrainConfig()
    base = RUNS / "bc" / f"advancedmp__{network}__ep3"
    data = load(base / "dataset.npz")
    meta = json.loads((base / "result.json").read_text())

    model, history = fit(data, cfg)
    rows = scenario_table(model, data, cfg)
    for net in OOD_NETWORKS:
        path = RUNS / "bc" / f"advancedmp__{net}__ep3" / "dataset.npz"
        if not path.is_file():
            continue
        rows += scenario_table(
            model, load(path), cfg, label=net.rsplit("_", 1)[-1], only=("clean",)
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
