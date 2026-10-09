"""Głowy niepewności: EDL (single-pass) i MC-dropout (gold standard offline).

Ta sama DEKOMPOZYCJA dla obu, i to jest cały sens tego modułu. MC-dropout/BALD
rozkłada niepewność na:

    total     = H[E_θ p]          entropia uśrednionego rozkładu
    aleatoric = E_θ H[p]          średnia entropia pojedynczych przejść
    epistemic = total − aleatoric  mutual information (BALD)

Dirichlet daje dokładnie te trzy wielkości ANALITYCZNIE, w jednym przejściu:

    total     = H(α/S)
    aleatoric = −Σ (α_k/S)·(ψ(α_k+1) − ψ(S+1))     = E_{p~Dir(α)} H(p)
    epistemic = total − aleatoric

Dzięki temu bramka 2.0(d) („tania separacja EDL zgadza się z drogim wzorcem")
jest porównaniem TYCH SAMYCH liczb, a nie dwóch różnych definicji niepewności —
inaczej rozjazd między metodami byłby nie do odróżnienia od rozjazdu między
wzorami. Obok raportujemy `vacuity = K/S` (subjective logic), bo to ją plan
wskazuje jako sygnał epistemic dla fallbacku.

Ograniczenie zapisane wprost: MC-dropout jest tu dropoutem OSTATNIEJ WARSTWY
(na wyjściu encodera), nie całej sieci. To standardowe przybliżenie i wystarcza
jako punkt odniesienia w prototypie, ale nie jest pełnym Gal-Ghahramani —
w pracy trzeba to nazwać, a nie przemilczeć.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tsc_marl.core.nets import GATEncoder, orthogonal_

EPS = 1e-10


class UncertainPolicy(nn.Module):
    """Encoder z Fazy 1 + głowa: `edl` (Dirichlet) albo `softmax` (MC-dropout).

    Encoder jest DOKŁADNIE tym samym `GATEncoder`, którego używa rdzeń — prototyp
    ma mierzyć głowę niepewności, a nie różnicę architektur.

    „Jedno skrzyżowanie" z 2.0 realizujemy JAWNYMI PĘTLAMI WŁASNYMI, nie innym
    modelem i NIE pustym `edge_index`: `add_self_loops` woła się bez `num_nodes`,
    więc z pustej listy krawędzi wnioskuje zero węzłów i nie dokłada niczego —
    agregacja zwraca zera, a model uczy się samego biasu (kosztowało to jeden
    przebieg bramki 2.2; buduje je `_edges` w `prototype.py`, a od Kroku 1
    UW-attention pusty `edge_index` jest wyjątkiem, nie cichą stałą).
    Druga iteracja 2.0 (uncertainty-weighted attention) to wtedy podanie
    prawdziwego `edge_index` — ta sama sieć, te same wagi.
    """

    def __init__(
        self,
        ob_length: int,
        n_phases: int,
        *,
        mode: str = "edl",
        dropout: float = 0.1,
        n_layers: int = 1,
        **encoder_kwargs,
    ) -> None:
        super().__init__()
        if mode not in ("edl", "softmax"):
            raise ValueError(f"mode={mode!r}, dozwolone: ('edl', 'softmax')")
        self.mode = mode
        self.n_phases = n_phases
        self.encoder = GATEncoder(ob_length, n_layers=n_layers, **encoder_kwargs)
        self.dropout = nn.Dropout(dropout)
        # gain=0.01 jak w `Actor`: startujemy blisko rozkładu jednostajnego, czyli
        # przy EDL blisko ZERA evidence — model zaczyna od „nic nie wiem", a nie
        # od pewnego siebie zgadywania, które trzeba potem odkręcać
        self.head = orthogonal_(nn.Linear(self.encoder.out_dim, n_phases), gain=0.01)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """-> [N, K]: `alpha` (mode='edl') albo logity (mode='softmax')."""
        h = self.dropout(self.encoder(x, edge_index))
        out = self.head(h)
        if self.mode == "softmax":
            return out
        # softplus, nie exp: exp saturuje i wysadza gradient, gdy sieć nabierze
        # pewności — a evidence ma rosnąć bez ograniczenia
        return torch.nn.functional.softplus(out) + 1.0  # alpha = evidence + 1


# --------------------------------------------------------------------- EDL


def edl_loss(alpha: Tensor, target: Tensor, anneal: float = 1.0) -> Tensor:
    """Bayes risk of cross-entropy + regularyzator KL (Sensoy 2018).

    Człon pierwszy: E_{p~Dir(α)}[CE] = Σ y_k (ψ(S) − ψ(α_k)) — uczy evidence
    tam, gdzie jest właściwa klasa. Sam w sobie NIE karze evidence przy klasach
    błędnych: model może rozdać dowody wszystkim i nadal mieć małą stratę.

    Człon drugi: KL[Dir(α̃) ‖ Dir(1)], gdzie α̃ = y + (1−y)⊙α (dowody prawidłowej
    klasy usunięte) — ściąga do jednostajnego wszystko, co NIE jest właściwą
    klasą. `anneal` rośnie od 0 do 1 przez pierwsze epoki: włączony od razu
    zabija evidence, zanim model zdąży się czegokolwiek nauczyć.
    """
    y = torch.nn.functional.one_hot(target, alpha.shape[-1]).to(alpha.dtype)
    s = alpha.sum(-1, keepdim=True)
    risk = (y * (torch.digamma(s) - torch.digamma(alpha))).sum(-1)
    return (risk + anneal * _kl_uniform(y + (1 - y) * alpha)).mean()


def _kl_uniform(alpha: Tensor) -> Tensor:
    """KL[Dir(α) ‖ Dir(1)] — dywergencja od rozkładu jednostajnego.

    KL = log B(1) − log B(α) + Σ (α_k − 1)·(ψ(α_k) − ψ(S)),  log B(α) = Σ logΓ(α_k) − logΓ(S).
    """
    ones = torch.ones_like(alpha)
    s = alpha.sum(-1, keepdim=True)
    log_beta_alpha = torch.lgamma(alpha).sum(-1) - torch.lgamma(s).squeeze(-1)
    log_beta_ones = torch.lgamma(ones).sum(-1) - torch.lgamma(ones.sum(-1))  # = −logΓ(K)
    digamma = torch.digamma(alpha) - torch.digamma(s)
    return (log_beta_ones - log_beta_alpha) + ((alpha - ones) * digamma).sum(-1)


def dirichlet_uncertainty(alpha: Tensor) -> dict[str, Tensor]:
    """Rozkład niepewności z jednego przejścia — te same trzy wielkości co BALD."""
    s = alpha.sum(-1, keepdim=True)
    p = alpha / s
    total = -(p * torch.log(p + EPS)).sum(-1)
    aleatoric = -(p * (torch.digamma(alpha + 1) - torch.digamma(s + 1))).sum(-1)
    return {
        "prob": p,
        "total": total,
        "aleatoric": aleatoric,
        "epistemic": (total - aleatoric).clamp_min(0.0),
        # K/S — im mniej dowodów, tym bliżej 1. To ten sygnał, którym plan steruje
        # fallbackiem (epistemic → MaxPressure), więc raportujemy go osobno.
        "vacuity": alpha.shape[-1] / s.squeeze(-1),
    }


# -------------------------------------------------------------- MC-dropout


@torch.no_grad()
def mc_dropout(model: UncertainPolicy, x: Tensor, edge_index: Tensor, k: int = 10):
    """K przejść z WŁĄCZONYM dropoutem. -> dekompozycja BALD.

    `model.train()` tylko na czas próbkowania: dropout musi być aktywny, ale
    gradientów nie liczymy. Reszta modelu (encoder LibSignala) nie ma warstw
    zależnych od trybu, więc przełączenie nie zmienia niczego poza dropoutem.
    """
    was_training = model.training
    model.train()
    try:
        probs = torch.stack(
            [torch.softmax(_logits(model, x, edge_index), -1) for _ in range(k)]
        )  # [K, N, C]
    finally:
        model.train(was_training)

    mean = probs.mean(0)
    total = -(mean * torch.log(mean + EPS)).sum(-1)
    aleatoric = -(probs * torch.log(probs + EPS)).sum(-1).mean(0)
    return {
        "prob": mean,
        "total": total,
        "aleatoric": aleatoric,
        "epistemic": (total - aleatoric).clamp_min(0.0),  # mutual information (BALD)
    }


def _logits(model: UncertainPolicy, x: Tensor, edge_index: Tensor) -> Tensor:
    out = model(x, edge_index)
    # przy EDL log(alpha) odtwarza logity zgodne z p = alpha/S, więc ta sama
    # funkcja obsługuje obie głowy i porównanie EDL vs MC nie zależy od trybu
    return torch.log(out + EPS) if model.mode == "edl" else out


def expected_calibration_error(prob: Tensor, target: Tensor, bins: int = 10) -> float:
    """ECE — średnia |trafność − pewność| ważona liczebnością koszyka.

    Reliability diagram z bramki 2.0(c) rysuje się z tych samych koszyków."""
    conf, pred = prob.max(-1)
    correct = (pred == target).float()
    edges = torch.linspace(0, 1, bins + 1, device=prob.device)
    ece = torch.zeros((), device=prob.device)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += mask.float().mean() * (correct[mask].mean() - conf[mask].mean()).abs()
    return float(ece)
