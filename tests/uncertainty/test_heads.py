"""Decision uncertainty heads: formulas, and one trap that already cost a full run.

Formulas are checked at LIMIT values with a known analytic result (alpha = 1 ->
entropy log K, no evidence -> vacuity 1), because an error in the decomposition
raises nothing and produces a plausible, wrong table.
"""

from __future__ import annotations

import math

import pytest
import torch

from tsc_marl.uncertainty.heads import (
    UncertainPolicy,
    _kl_uniform,
    dirichlet_uncertainty,
    edl_loss,
    expected_calibration_error,
    mc_dropout,
)

K = 4


def test_kl_from_uniform_is_zero_without_evidence():
    assert float(_kl_uniform(torch.ones(1, K))) == pytest.approx(0.0, abs=1e-6)


def test_kl_grows_with_evidence():
    """The regulariser penalises evidence given to wrong classes."""
    kl = [float(_kl_uniform(torch.full((1, K), a))) for a in (1.0, 2.0, 5.0, 20.0)]
    assert kl == sorted(kl) and kl[-1] > kl[0]


def test_no_evidence_is_maximal_uncertainty():
    u = dirichlet_uncertainty(torch.ones(1, K))
    assert float(u["total"]) == pytest.approx(math.log(K), abs=1e-5)
    assert float(u["vacuity"]) == pytest.approx(1.0)


def test_much_evidence_for_one_class_is_certainty():
    u = dirichlet_uncertainty(torch.tensor([[100.0, 1.0, 1.0, 1.0]]))
    assert float(u["total"]) < 0.3
    assert float(u["vacuity"]) < 0.05


def test_ambivalence_is_aleatoric_not_epistemic():
    """Two equally good phases with plenty of evidence are DECISION uncertainty
    (aleatoric), not missing knowledge (epistemic); otherwise ties would pose as
    sensor failures."""
    u = dirichlet_uncertainty(torch.tensor([[50.0, 50.0, 1.0, 1.0]]))
    assert float(u["aleatoric"]) > 0.6
    assert float(u["epistemic"]) < 0.05


def test_epistemic_is_high_with_little_evidence():
    weak = dirichlet_uncertainty(torch.tensor([[2.0, 2.0, 1.0, 1.0]]))
    strong = dirichlet_uncertainty(torch.tensor([[50.0, 50.0, 1.0, 1.0]]))
    assert float(weak["epistemic"]) > float(strong["epistemic"])
    assert float(weak["vacuity"]) > float(strong["vacuity"])


def test_edl_loss_drops_when_evidence_goes_to_the_right_class():
    target = torch.tensor([0])
    good = torch.tensor([[20.0, 1.0, 1.0, 1.0]])
    bad = torch.tensor([[1.0, 20.0, 1.0, 1.0]])
    assert float(edl_loss(good, target, 0.0)) < float(edl_loss(bad, target, 0.0))


def test_ece_is_zero_for_perfect_calibration():
    prob = torch.eye(K)
    assert expected_calibration_error(prob, torch.arange(K)) == pytest.approx(0.0, abs=1e-6)


def test_ece_detects_overconfidence():
    prob = torch.zeros(4, K)
    prob[:, 0] = 1.0
    ece = expected_calibration_error(prob, torch.tensor([0, 1, 1, 1]))
    assert ece == pytest.approx(0.75, abs=1e-5)  # confidence 1.0, accuracy 0.25


# ------------------------------------------------------------------- 0-hop


def loops(n: int) -> torch.Tensor:
    idx = torch.arange(n)
    return torch.stack([idx, idx])


def test_zero_hop_isolates_nodes():
    """One intersection via self-loops: a node's output must not depend on others."""
    torch.manual_seed(0)
    model = UncertainPolicy(20, K, dropout=0.0).eval()
    x = torch.randn(5, 20)
    a = model(x, loops(5))
    x2 = x.clone()
    x2[1:] = torch.randn(4, 20)
    b = model(x2, loops(5))
    assert torch.allclose(a[0], b[0], atol=1e-6)
    assert not torch.allclose(a[1:], b[1:], atol=1e-6)


def test_empty_edge_index_is_an_error():
    """`add_self_loops` without `num_nodes` infers zero nodes from an empty
    `edge_index`, the aggregation returns zeros and the model trains only its bias,
    silently. The attention layer turns this into an error; 0-hop uses self-loops
    (`_edges` in `prototype.py`)."""
    torch.manual_seed(0)
    model = UncertainPolicy(20, K, dropout=0.0).eval()
    empty = torch.zeros((2, 0), dtype=torch.long)
    with pytest.raises(ValueError, match="isolated"):
        model(torch.randn(5, 20), empty)


def test_one_hop_sees_the_neighbour():
    torch.manual_seed(0)
    model = UncertainPolicy(20, K, dropout=0.0).eval()
    edges = torch.tensor([[1], [0]])  # 1 -> 0
    x = torch.randn(2, 20)
    a = model(x, edges)
    x2 = x.clone()
    x2[1] = torch.randn(20)
    assert not torch.allclose(a[0], model(x2, edges)[0], atol=1e-6)


def test_mc_dropout_samples_and_restores_the_mode():
    """K passes must DIFFER (otherwise epistemic is zero by definition) and the
    model must return to its previous mode."""
    torch.manual_seed(0)
    model = UncertainPolicy(20, K, dropout=0.5).eval()
    # the head starts with gain 0.01, i.e. almost uniform, where dropout moves
    # nothing; scale the weights so the test measures sampling, not initialisation
    with torch.no_grad():
        model.head.weight.mul_(50.0)
    out = mc_dropout(model, torch.randn(8, 20), loops(8), k=10)
    assert not model.training
    assert float(out["epistemic"].mean()) > 0
    assert torch.allclose(out["prob"].sum(-1), torch.ones(8), atol=1e-5)


def test_softmax_and_edl_give_the_same_shape():
    torch.manual_seed(0)
    x, ei = torch.randn(3, 20), loops(3)
    for mode in ("edl", "softmax"):
        out = UncertainPolicy(20, K, mode=mode, dropout=0.0).eval()(x, ei)
        assert out.shape == (3, K)
    assert (UncertainPolicy(20, K, dropout=0.0).eval()(x, ei) >= 1.0).all()  # alpha = evidence + 1


def test_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="mode="):
        UncertainPolicy(20, K, mode="dirichlet")
