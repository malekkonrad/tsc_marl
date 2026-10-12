"""Ensemble controller and plans with weights from the training network.

Two silent failures to catch: (1) MI computed the wrong way round or over the
wrong axis (agreeing actors must give ZERO); (2) a parent template taking the
weights from the EVALUATION network instead of the training one (`{base}` must
strip the traffic-variant suffix).
"""

from __future__ import annotations

import textwrap

import numpy as np
import pytest
import torch
from torch import nn

from tsc_marl.robustness import controllers as c
from tsc_marl.robustness import degrade as d


class Fixed(nn.Module):
    """Actor with constant logits independent of the input."""

    def __init__(self, logits):
        super().__init__()
        self.logits = torch.tensor(logits, dtype=torch.float32)

    def forward(self, x, edge_index):
        return self.logits.expand(x.shape[0], -1)


EI = torch.zeros((2, 0), dtype=torch.long)
OBS = np.zeros((3, 4), np.float32)


def test_agreeing_actors_give_zero_disagreement():
    ens = c.EnsembleController([Fixed([2.0, 0.0]), Fixed([2.0, 0.0])], EI)
    actions = ens.act(OBS)
    stats = ens.episode_stats()
    assert actions.tolist() == [0, 0, 0]
    assert stats["ens_mi"] == pytest.approx(0.0, abs=1e-6)
    assert stats["ens_disagree"] == 0.0


def test_different_actors_disagree_and_the_first_decides():
    ens = c.EnsembleController([Fixed([3.0, 0.0]), Fixed([0.0, 3.0]), Fixed([0.0, 3.0])], EI)
    actions = ens.act(OBS)
    stats = ens.episode_stats()
    assert actions.tolist() == [0, 0, 0], "the first actor decides, not the majority"
    assert stats["ens_mi"] > 0.1
    assert stats["ens_disagree"] == pytest.approx(1.0)  # both others would choose differently


def test_stats_reset_per_episode():
    ens = c.EnsembleController([Fixed([3.0, 0.0]), Fixed([0.0, 3.0])], EI)
    ens.act(OBS)
    ens.reset_stats()
    assert ens.episode_stats() == {}


@pytest.mark.parametrize(
    ("network", "base"),
    [
        ("cityflow_hangzhou4x4_4ph_r5816", "cityflow_hangzhou4x4_4ph"),
        ("cityflow_hangzhou4x4_4ph_d1p5", "cityflow_hangzhou4x4_4ph"),
        ("cityflow_jinan3x4_4ph_d0p25", "cityflow_jinan3x4_4ph"),
        ("cityflow_jinan3x4_4ph", "cityflow_jinan3x4_4ph"),
        ("cityflow_hangzhou4x4", "cityflow_hangzhou4x4"),
    ],
)
def test_base_strips_only_the_traffic_suffix(network, base):
    assert d.base_network(network) == base


def test_ensemble_plan_one_unit_five_parents(tmp_path):
    p = tmp_path / "ood.yml"
    p.write_text(
        textwrap.dedent(
            """
            networks: [cityflow_hangzhou4x4_4ph, cityflow_hangzhou4x4_4ph_r5816]
            controllers:
              gatppo: {seeds: [0, 1], parent: "{agent}__{base}__s{seed}_y{yellow}"}
              gatppo_ens:
                agent: gatppo
                ensemble: [0, 1, 2, 3, 4]
                parent: "{agent}__{base}__s{seed}_y{yellow}"
            """
        )
    )
    plan = d.load_plan(p)
    ens_units = [u for u in plan.units() if u.controller == "gatppo_ens"]
    assert len(ens_units) == 2 and {u.seed for u in ens_units} == {0}
    heavy = next(u for u in ens_units if u.network.endswith("r5816"))
    ids = plan.parent_ids(heavy)
    assert ids == [f"gatppo__cityflow_hangzhou4x4_4ph__s{s}_y5" for s in range(5)]
    assert plan.kind("gatppo_ens") == "ensemble" and plan.kind("gatppo") == "policy"


def test_ensemble_only_from_core_actors(tmp_path):
    p = tmp_path / "bad.yml"
    p.write_text("networks: [x]\ncontrollers:\n  e: {agent: efficientmp, ensemble: [0, 1]}\n")
    with pytest.raises(SystemExit, match="ensemble"):
        d.load_plan(p)
