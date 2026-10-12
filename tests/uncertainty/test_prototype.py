"""Decision-uncertainty prototype: data split and training loop.

The split is the one thing that can silently invalidate the whole gate: traffic
in `flow.json` is deterministic, so episodes share the same trajectory and a split
"by episode" would leak bit for bit.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsc_marl.uncertainty import prototype as P

N, F, PH, DEC, EPS = 3, 8, 2, 10, 2


def fake(seed: int = 0) -> P.Loaded:
    """A dataset shaped like a dump, with a label COMPUTABLE from the input (argmax
    of the first PH columns), so one can check that training learns at all."""
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


def test_time_restarts_within_each_episode():
    """`times` restarts at zero every episode; otherwise the time split would cut
    out the whole second episode instead of each episode's tail."""
    data = fake()
    assert data.times.tolist() == list(range(DEC)) * EPS


def test_time_split_is_disjoint_and_cuts_every_episode():
    data = fake()
    train, test = data.mask(0.7, True), data.mask(0.7, False)
    assert not (train & test).any() and (train | test).all()
    for ep in range(EPS):
        block = data.episode == ep
        assert train[block].sum() == 7 and test[block].sum() == 3


def test_zero_hop_has_self_loops_not_an_empty_graph():
    """Regression for the `heads` trap: 0-hop is N self-loops, NOT zero edges."""
    data = fake()
    edges = P._edges(data, P.TrainConfig(hop=False))
    assert edges.shape == (2, N)
    assert (edges[0] == edges[1]).all()
    assert P._edges(data, P.TrainConfig(hop=True)).shape[1] == data.edge_index.shape[1]


@pytest.mark.parametrize("mode", ["edl", "softmax"])
def test_training_loop_actually_learns(mode):
    """The task is computable from the input, so accuracy must clearly beat chance;
    catches loss SIGN errors and swapped axes that pass every shape test."""
    data = fake()
    cfg = P.TrainConfig(mode=mode, epochs=60, train_scenarios=("clean",), batch=8)
    _, history = P.fit(data, cfg)
    assert history[-1]["acc"] > 0.8
    assert history[-1]["loss"] < history[0]["loss"]


def test_predict_returns_both_uncertainty_sources():
    data = fake()
    cfg = P.TrainConfig(epochs=5, train_scenarios=("clean",), batch=8, mc_passes=3)
    model, _ = P.fit(data, cfg)
    mask = data.mask(0.7, False)
    out = P.predict(model, data.x["clean"][mask], data, cfg)
    for key in ("edl_total", "edl_aleatoric", "edl_epistemic", "edl_vacuity", "mc_epistemic"):
        assert out[key].shape == (mask.sum(), N), key
    assert np.allclose(out["edl_total"], out["edl_aleatoric"] + out["edl_epistemic"], atol=1e-5)


def test_table_knows_the_training_scenarios():
    data = fake()
    cfg = P.TrainConfig(epochs=5, train_scenarios=("clean",), batch=8, mc_passes=3)
    model, _ = P.fit(data, cfg)
    rows = P.scenario_table(model, data, cfg)
    assert {r["scenario"] for r in rows} == {"clean", "gauss1"}
    assert [r["trained"] for r in rows] == [True, False]
    # an OOD set is never "trained", even when the scenario name matches
    assert all(not r["trained"] for r in P.scenario_table(model, data, cfg, label="d0p5"))
