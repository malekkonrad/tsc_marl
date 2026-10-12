"""Decision uncertainty heads: EDL (single pass) and MC-dropout (offline reference).

Both use the SAME decomposition. MC-dropout / BALD splits uncertainty into

    total     = H[E_theta p]           entropy of the mean distribution
    aleatoric = E_theta H[p]           mean entropy of single passes
    epistemic = total - aleatoric      mutual information (BALD)

and a Dirichlet gives the same three quantities ANALYTICALLY in one pass:

    total     = H(alpha / S)
    aleatoric = -sum (alpha_k / S) * (psi(alpha_k + 1) - psi(S + 1))
    epistemic = total - aleatoric

so comparing the cheap EDL with the expensive reference compares the same
numbers, not two definitions. `vacuity = K / S` (subjective logic) is reported
too. Limitation, stated: MC-dropout here drops only the LAST layer (the
encoder output), a standard approximation, not full Gal-Ghahramani.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tsc_marl.core.nets import GATEncoder, orthogonal_

EPS = 1e-10


class UncertainPolicy(nn.Module):
    """The core's encoder plus a head: `edl` (Dirichlet) or `softmax` (MC-dropout).

    The encoder is EXACTLY the core's `GATEncoder`, so the prototype measures the
    head, not an architecture difference. "One intersection" is done with
    EXPLICIT self-loops (`_edges` in `prototype.py`), not an empty `edge_index`:
    `add_self_loops` without `num_nodes` infers zero nodes from an empty list, and
    the model would learn only the bias (now an error in the attention layer).
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
            raise ValueError(f"mode={mode!r}, allowed: ('edl', 'softmax')")
        self.mode = mode
        self.n_phases = n_phases
        self.encoder = GATEncoder(ob_length, n_layers=n_layers, **encoder_kwargs)
        self.dropout = nn.Dropout(dropout)
        # gain 0.01 as in `Actor`: start near uniform, i.e. near ZERO evidence for EDL
        self.head = orthogonal_(nn.Linear(self.encoder.out_dim, n_phases), gain=0.01)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """-> [N, K]: `alpha` (mode='edl') albo logity (mode='softmax')."""
        h = self.dropout(self.encoder(x, edge_index))
        out = self.head(h)
        if self.mode == "softmax":
            return out
        # softplus, not exp: exp blows up the gradient once the network is confident
        return torch.nn.functional.softplus(out) + 1.0  # alpha = evidence + 1


# --------------------------------------------------------------------- EDL


def edl_loss(alpha: Tensor, target: Tensor, anneal: float = 1.0) -> Tensor:
    """Bayes risk of cross-entropy plus a KL regulariser (Sensoy et al., 2018).

    First term, E_{p~Dir(alpha)}[CE] = sum y_k (psi(S) - psi(alpha_k)), builds
    evidence for the right class, but does NOT penalise evidence for wrong ones.
    Second term, KL[Dir(alpha~) || Dir(1)] with alpha~ = y + (1 - y) * alpha,
    pulls everything except the right class to uniform. `anneal` grows from 0 to
    1 over the first epochs; switched on at once it kills evidence before learning.
    """
    y = torch.nn.functional.one_hot(target, alpha.shape[-1]).to(alpha.dtype)
    s = alpha.sum(-1, keepdim=True)
    risk = (y * (torch.digamma(s) - torch.digamma(alpha))).sum(-1)
    return (risk + anneal * _kl_uniform(y + (1 - y) * alpha)).mean()


def _kl_uniform(alpha: Tensor) -> Tensor:
    """KL[Dir(alpha) || Dir(1)], the divergence from uniform:

    KL = log B(1) - log B(alpha) + sum (alpha_k - 1) * (psi(alpha_k) - psi(S)),
    log B(alpha) = sum log Gamma(alpha_k) - log Gamma(S).
    """
    ones = torch.ones_like(alpha)
    s = alpha.sum(-1, keepdim=True)
    log_beta_alpha = torch.lgamma(alpha).sum(-1) - torch.lgamma(s).squeeze(-1)
    log_beta_ones = torch.lgamma(ones).sum(-1) - torch.lgamma(ones.sum(-1))  # = -log Gamma(K)
    digamma = torch.digamma(alpha) - torch.digamma(s)
    return (log_beta_ones - log_beta_alpha) + ((alpha - ones) * digamma).sum(-1)


def dirichlet_uncertainty(alpha: Tensor) -> dict[str, Tensor]:
    """Uncertainty decomposition from one pass: the same three quantities as BALD."""
    s = alpha.sum(-1, keepdim=True)
    p = alpha / s
    total = -(p * torch.log(p + EPS)).sum(-1)
    aleatoric = -(p * (torch.digamma(alpha + 1) - torch.digamma(s + 1))).sum(-1)
    return {
        "prob": p,
        "total": total,
        "aleatoric": aleatoric,
        "epistemic": (total - aleatoric).clamp_min(0.0),
        # K/S: closer to 1 with less evidence; reported separately
        "vacuity": alpha.shape[-1] / s.squeeze(-1),
    }


# -------------------------------------------------------------- MC-dropout


@torch.no_grad()
def mc_dropout(model: UncertainPolicy, x: Tensor, edge_index: Tensor, k: int = 10):
    """K passes with dropout ON. -> BALD decomposition.

    `model.train()` only while sampling (no gradients); the encoder has no other
    mode-dependent layers, so only dropout changes.
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
    # for EDL log(alpha) gives logits consistent with p = alpha / S, so one
    # function serves both heads
    return torch.log(out + EPS) if model.mode == "edl" else out


def expected_calibration_error(prob: Tensor, target: Tensor, bins: int = 10) -> float:
    """ECE: mean |accuracy - confidence| weighted by bin size (the same bins as a
    reliability diagram)."""
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
