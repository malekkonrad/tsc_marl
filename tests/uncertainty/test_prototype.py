"""Testy prototypu 2.0 — podział danych i pętla ucząca.

Podział jest tu jedyną rzeczą, która potrafi cicho unieważnić CAŁĄ bramkę: ruch
w `flow.json` jest deterministyczny, więc epizody mają identyczną trajektorię
i „podział po epizodach" byłby przeciekiem co do bitu.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsc_marl.uncertainty import prototype as P

N, F, PH, DEC, EPS = 3, 8, 2, 10, 2


def fake(seed: int = 0) -> P.Loaded:
    """Zbiór o kształcie zrzutu, z etykietą WYZNACZALNĄ z wejścia (argmax pierwszych
    PH kolumn) — dzięki temu można sprawdzić, czy pętla ucząca w ogóle się uczy."""
    rng = np.random.default_rng(seed)
    t = DEC * EPS
    x = rng.normal(size=(t, N, F)).astype(np.float32)
    y = x[..., :PH].argmax(-1).astype(np.int64)
    inc = np.zeros((N, PH, F // 2), dtype=bool)
    inc[:, np.arange(PH), np.arange(PH)] = True
    return P.Loaded(
        x={"clean": x, "gauss1": x + 0.1 * rng.normal(size=x.shape).astype(np.float32)},
        dead={"clean": np.zeros((t, N, F // 2), bool), "gauss1": np.zeros((t, N, F // 2), bool)},
        x_clean=x,
        y=y,
        tie_margin=np.ones((t, N), np.float32),
        phase_time=np.full((t, N), 15, np.int32),
        episode=np.repeat(np.arange(EPS, dtype=np.int32), DEC),
        edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
        incidence=inc,
    )


def test_czas_liczy_sie_wewnatrz_epizodu():
    """`times` musi wracać do zera na każdym epizodzie — inaczej podział po czasie
    wyciąłby cały drugi epizod zamiast ogona każdego."""
    data = fake()
    assert data.times.tolist() == list(range(DEC)) * EPS


def test_podzial_po_czasie_nie_zachodzi_i_tnie_kazdy_epizod():
    data = fake()
    train, test = data.mask(0.7, True), data.mask(0.7, False)
    assert not (train & test).any() and (train | test).all()
    for ep in range(EPS):
        block = data.episode == ep
        assert train[block].sum() == 7 and test[block].sum() == 3


def test_zerohop_ma_petle_wlasne_a_nie_pusty_graf():
    """Regresja do pułapki z `heads`: 0-hop to N pętli własnych, NIE zero krawędzi."""
    data = fake()
    edges = P._edges(data, P.TrainConfig(hop=False))
    assert edges.shape == (2, N)
    assert (edges[0] == edges[1]).all()
    assert P._edges(data, P.TrainConfig(hop=True)).shape[1] == data.edge_index.shape[1]


@pytest.mark.parametrize("mode", ["edl", "softmax"])
def test_petla_uczaca_faktycznie_sie_uczy(mode):
    """Zadanie jest wyznaczalne z wejścia, więc trafność ma wyraźnie przebić
    losowanie. Test łapie błędy ZNAKU straty i pomylone osie — przechodzą przez
    wszystkie testy kształtów."""
    data = fake()
    cfg = P.TrainConfig(mode=mode, epochs=60, train_scenarios=("clean",), batch=8)
    _, history = P.fit(data, cfg)
    assert history[-1]["acc"] > 0.8
    assert history[-1]["loss"] < history[0]["loss"]


def test_predict_zwraca_oba_zrodla_niepewnosci():
    data = fake()
    cfg = P.TrainConfig(epochs=5, train_scenarios=("clean",), batch=8, mc_passes=3)
    model, _ = P.fit(data, cfg)
    mask = data.mask(0.7, False)
    out = P.predict(model, data.x["clean"][mask], data, cfg)
    for key in ("edl_total", "edl_aleatoric", "edl_epistemic", "edl_vacuity", "mc_epistemic"):
        assert out[key].shape == (mask.sum(), N), key
    assert np.allclose(out["edl_total"], out["edl_aleatoric"] + out["edl_epistemic"], atol=1e-5)


def test_tabela_zna_scenariusze_treningowe():
    data = fake()
    cfg = P.TrainConfig(epochs=5, train_scenarios=("clean",), batch=8, mc_passes=3)
    model, _ = P.fit(data, cfg)
    rows = P.scenario_table(model, data, cfg)
    assert {r["scenario"] for r in rows} == {"clean", "gauss1"}
    assert [r["trenowany"] for r in rows] == [True, False]
    # zbiór OOD nigdy nie jest „trenowany", nawet jeśli nazwa scenariusza się zgadza
    assert all(not r["trenowany"] for r in P.scenario_table(model, data, cfg, label="d0p5"))
