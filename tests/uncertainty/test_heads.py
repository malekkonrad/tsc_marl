"""Testy głów niepewności — wzory i jedna pułapka, która już raz kosztowała bramkę.

Wzory sprawdzamy na wartościach GRANICZNYCH, dla których znamy wynik analitycznie
(α=1 → entropia log K, brak dowodów → vacuity 1), bo błąd w dekompozycji nie
wywala się: daje tabelę, która wygląda sensownie i mówi nieprawdę.
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


def test_kl_od_jednostajnego_zeruje_sie_przy_braku_dowodow():
    assert float(_kl_uniform(torch.ones(1, K))) == pytest.approx(0.0, abs=1e-6)


def test_kl_rosnie_z_dowodami():
    """Regulator ma karać dowody rozdane klasom błędnym — inaczej model może
    rozdać evidence wszystkim i nadal mieć małą stratę."""
    kl = [float(_kl_uniform(torch.full((1, K), a))) for a in (1.0, 2.0, 5.0, 20.0)]
    assert kl == sorted(kl) and kl[-1] > kl[0]


def test_brak_dowodow_to_maksymalna_niepewnosc():
    u = dirichlet_uncertainty(torch.ones(1, K))
    assert float(u["total"]) == pytest.approx(math.log(K), abs=1e-5)
    assert float(u["vacuity"]) == pytest.approx(1.0)


def test_duzo_dowodow_na_jedna_klase_to_pewnosc():
    u = dirichlet_uncertainty(torch.tensor([[100.0, 1.0, 1.0, 1.0]]))
    assert float(u["total"]) < 0.3
    assert float(u["vacuity"]) < 0.05


def test_ambiwalencja_jest_ALEATORYCZNA_nie_epistemiczna():
    """Sedno mechanizmu i powód, dla którego 2.2 NIE bierze sygnału ze spreadu
    Dirichleta: dwie fazy równie dobre przy mnóstwie dowodów to niepewność
    DECYZJI (aleatoric), a nie brak wiedzy (epistemic). Gdyby dekompozycja tego
    nie rozdzielała, `tie_rate` = 14% zrzutu udawałoby awarie sensorów."""
    u = dirichlet_uncertainty(torch.tensor([[50.0, 50.0, 1.0, 1.0]]))
    assert float(u["aleatoric"]) > 0.6
    assert float(u["epistemic"]) < 0.05


def test_epistemic_wysoki_przy_malej_liczbie_dowodow():
    slaby = dirichlet_uncertainty(torch.tensor([[2.0, 2.0, 1.0, 1.0]]))
    mocny = dirichlet_uncertainty(torch.tensor([[50.0, 50.0, 1.0, 1.0]]))
    assert float(slaby["epistemic"]) > float(mocny["epistemic"])
    assert float(slaby["vacuity"]) > float(mocny["vacuity"])


def test_strata_edl_maleje_gdy_dowody_ida_do_wlasciwej_klasy():
    target = torch.tensor([0])
    dobre = torch.tensor([[20.0, 1.0, 1.0, 1.0]])
    zle = torch.tensor([[1.0, 20.0, 1.0, 1.0]])
    assert float(edl_loss(dobre, target, 0.0)) < float(edl_loss(zle, target, 0.0))


def test_ece_zeruje_sie_dla_idealnej_kalibracji():
    prob = torch.eye(K)
    assert expected_calibration_error(prob, torch.arange(K)) == pytest.approx(0.0, abs=1e-6)


def test_ece_wykrywa_nadmierna_pewnosc():
    prob = torch.zeros(4, K)
    prob[:, 0] = 1.0
    ece = expected_calibration_error(prob, torch.tensor([0, 1, 1, 1]))
    assert ece == pytest.approx(0.75, abs=1e-5)  # pewność 1.0, trafność 0.25


# ------------------------------------------------------------------- 0-hop


def loops(n: int) -> torch.Tensor:
    idx = torch.arange(n)
    return torch.stack([idx, idx])


def test_zerohop_izoluje_wezly():
    """„Jedno skrzyżowanie" z 2.0 realizujemy pętlami własnymi — wyjście węzła
    nie może zależeć od cech pozostałych."""
    torch.manual_seed(0)
    model = UncertainPolicy(20, K, dropout=0.0).eval()
    x = torch.randn(5, 20)
    a = model(x, loops(5))
    x2 = x.clone()
    x2[1:] = torch.randn(4, 20)
    b = model(x2, loops(5))
    assert torch.allclose(a[0], b[0], atol=1e-6)
    assert not torch.allclose(a[1:], b[1:], atol=1e-6)


def test_PUSTY_edge_index_zeruje_wyjscie():
    """Pułapka, która kosztowała pełny przebieg bramki: `MultiHeadAttModel` woła
    `add_self_loops` BEZ `num_nodes`, więc z pustego `edge_index` wnioskuje zero
    węzłów, nie dokłada pętli i agregacja zwraca zera — model przestaje zależeć
    od wejścia i trenuje sam bias, bez żadnego błędu. Test istnieje po to, żeby
    nikt nie „uprościł" 0-hopu z powrotem do pustej listy krawędzi."""
    torch.manual_seed(0)
    model = UncertainPolicy(20, K, dropout=0.0).eval()
    pusty = torch.zeros((2, 0), dtype=torch.long)
    assert torch.allclose(model(torch.randn(5, 20), pusty), model(torch.randn(5, 20), pusty))


def test_jeden_hop_widzi_sasiada():
    torch.manual_seed(0)
    model = UncertainPolicy(20, K, dropout=0.0).eval()
    edges = torch.tensor([[1], [0]])  # 1 -> 0
    x = torch.randn(2, 20)
    a = model(x, edges)
    x2 = x.clone()
    x2[1] = torch.randn(20)
    assert not torch.allclose(a[0], model(x2, edges)[0], atol=1e-6)


def test_mc_dropout_probkuje_i_przywraca_tryb():
    """K przejść ma się RÓŻNIĆ (inaczej epistemic jest zerem z definicji),
    a model ma wrócić do trybu, w którym był — inaczej kolejna ewaluacja
    liczyłaby się z włączonym dropoutem."""
    torch.manual_seed(0)
    model = UncertainPolicy(20, K, dropout=0.5).eval()
    # głowa startuje z gain=0.01, czyli praktycznie na rozkładzie jednostajnym —
    # wtedy dropout niczym nie porusza i epistemic wychodzi zerem Z DEFINICJI.
    # Skalujemy wagi, żeby test mierzył próbkowanie, a nie inicjalizację.
    with torch.no_grad():
        model.head.weight.mul_(50.0)
    out = mc_dropout(model, torch.randn(8, 20), loops(8), k=10)
    assert not model.training
    assert float(out["epistemic"].mean()) > 0
    assert torch.allclose(out["prob"].sum(-1), torch.ones(8), atol=1e-5)


def test_softmax_i_edl_daja_ten_sam_ksztalt():
    torch.manual_seed(0)
    x, ei = torch.randn(3, 20), loops(3)
    for mode in ("edl", "softmax"):
        out = UncertainPolicy(20, K, mode=mode, dropout=0.0).eval()(x, ei)
        assert out.shape == (3, K)
    assert (UncertainPolicy(20, K, dropout=0.0).eval()(x, ei) >= 1.0).all()  # alpha = evidence + 1


def test_odrzuca_nieznany_tryb():
    with pytest.raises(ValueError, match="mode="):
        UncertainPolicy(20, K, mode="dirichlet")
