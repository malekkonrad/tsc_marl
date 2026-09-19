"""Testy warstwy treningu z awariami (domain randomization) z Fazy 3, §3.

Dwie rzeczy, których zepsucie jest CICHE i unieważnia cały eksperyment:

1. Harmonogram awarii ma zależeć wyłącznie od (seed mixu, numer epizodu) — nie od
   `--seed` polityki i nie od stanu globalnego RNG. Inaczej ramiona 2 i 3 widzą różne
   awarie i różnica między nimi miesza mechanizm z innym losowaniem.
2. Mieszanka w `p1_fail.yml` i `p1_uw.yml` ma być IDENTYCZNA. Rozjazd w jednej wadze
   nie wywala niczego — daje tabelę, która odpowiada na inne pytanie.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import yaml

from tsc_marl.domain.sensors import FailureSpec
from tsc_marl.paths import CONFIGS
from tsc_marl.robustness.training import FailureMix

MIESZANKA = {
    "seed": 0,
    "scenarios": [
        {"weight": 4, "kind": "none"},
        {"weight": 2, "kind": "mask", "level": 0.25},
        {"weight": 1, "kind": "gauss", "level": 2.0},
    ],
}


def mix() -> FailureMix:
    return FailureMix.from_config(MIESZANKA)


# ------------------------------------------------------------------- losowanie


def test_scenariusz_zalezy_tylko_od_numeru_epizodu():
    """Wznowiony trening od epizodu 137 ma dostać ten sam scenariusz co bieg ciągły,
    a dwa ramiona — ten sam harmonogram. Stąd RNG kluczowany epizodem, nie strumień
    ciągnięty po kolei."""
    m = mix()
    assert [m.sample(e) for e in range(30)] == [m.sample(e) for e in range(30)]
    assert m.sample(137) == m.sample(137)


def test_losowanie_nie_zalezy_od_globalnego_rng():
    """`--seed` steruje inicjalizacją sieci przez `set_global_seed`. Gdyby wzorzec
    awarii szedł z tego samego źródła, seedy różniłyby się JEDNOCZEŚNIE polityką
    i tym, które detektory padły — i rozrzutu w T2 nie dałoby się rozdzielić.
    Ta sama zasada, co w `domain/sensors.py`."""
    m = mix()
    random.seed(1234)
    np.random.seed(1234)
    pierwsze = [m.sample(e) for e in range(20)]
    random.seed(9)
    np.random.seed(9)
    assert [m.sample(e) for e in range(20)] == pierwsze


def test_seed_mieszanki_zmienia_harmonogram():
    a = FailureMix.from_config(MIESZANKA)
    b = FailureMix.from_config({**MIESZANKA, "seed": 1})
    assert [a.sample(e) for e in range(40)] != [b.sample(e) for e in range(40)]


def test_wagi_sa_respektowane():
    """4:2:1 — sprawdzamy rząd wielkości, nie dokładny ułamek: to jest kontrola,
    że wagi w ogóle działają, a nie test generatora numpy."""
    m = mix()
    proby = [m.sample(e) for e in range(3000)]
    czyste = sum(1 for s in proby if not s.active)
    assert m.probabilities.tolist() == pytest.approx([4 / 7, 2 / 7, 1 / 7])
    assert 0.5 < czyste / len(proby) < 0.65


def test_mieszanka_zawiera_epizody_czyste_i_z_awaria():
    """Ramię z samymi awariami zapomniałoby czystego ruchu, a kolumna `clean` w T2
    jest odniesieniem dla wszystkich degradacji (A3a)."""
    m = mix()
    proby = {m.sample(e) for e in range(200)}
    assert any(not s.active for s in proby)
    assert any(s.active for s in proby)


# ----------------------------------------------------------------- parsowanie


def test_brak_mieszanki_to_none():
    """Bieg bez `failure_mix` ma być DOKŁADNIE rdzeniem z Fazy 1 — None, nie
    mieszanka z jednym czystym scenariuszem (tamta przebudowuje `StateBuilder`)."""
    assert FailureMix.from_config(None) is None
    assert FailureMix.from_config({}) is None


def test_literowka_w_scenariuszu_wywala_sie_przy_budowie():
    """Ma paść przy starcie, nie po 200 epizodach dziwnych wyników."""
    with pytest.raises(TypeError):
        FailureMix.from_config({"scenarios": [{"kind": "mask", "lvl": 0.25}]})
    with pytest.raises(ValueError, match="kind"):
        FailureMix.from_config({"scenarios": [{"kind": "maska", "level": 0.25}]})


def test_odrzuca_bezsensowne_wagi():
    with pytest.raises(ValueError, match="nieujemne"):
        FailureMix((FailureSpec(),), (-1.0,))
    with pytest.raises(ValueError, match="dodatnia"):
        FailureMix((FailureSpec(),), (0.0,))
    with pytest.raises(ValueError, match="pusta"):
        FailureMix((), ())


def test_as_dict_opisuje_caly_harmonogram():
    """Bez tego z `runs/` nie wynika, na czym uczyła się polityka — a dwa ramiona
    z różnymi mieszankami wyglądałyby w katalogu tak samo."""
    d = mix().as_dict()
    assert d["seed"] == 0
    assert [s["weight"] for s in d["scenarios"]] == [4.0, 2.0, 1.0]
    assert [s["kind"] for s in d["scenarios"]] == ["none", "mask", "gauss"]


# ------------------------------------------------------ spójność obu ramion


def suite_mix(name: str) -> dict:
    raw = yaml.safe_load((CONFIGS / "suites" / f"{name}.yml").read_text())
    return raw["overrides"]["model"]["failure_mix"]


def test_ramiona_maja_te_sama_mieszanke():
    """`p1_fail` (bez UW) i `p1_uw` (z UW) różnią się WYŁĄCZNIE mechanizmem —
    to jest cały sens ramienia kontrolnego. Rozjazd w jednej wadze nie wywala
    niczego, tylko unieważnia porównanie 2↔3."""
    assert suite_mix("p1_fail") == suite_mix("p1_uw")


def test_tylko_ramie_uw_wlacza_mechanizm():
    bez = yaml.safe_load((CONFIGS / "suites" / "p1_fail.yml").read_text())["overrides"]["model"]
    z = yaml.safe_load((CONFIGS / "suites" / "p1_uw.yml").read_text())["overrides"]["model"]
    assert "uw" not in bez
    assert z["uw"]["tau"] == 0.25
    assert set(z) - set(bez) == {"uw"}


def test_mieszanka_suite_ow_sie_parsuje():
    """Plik YAML ma dawać zbudowaną mieszankę, a nie wyjątek w 3. godzinie biegu."""
    m = FailureMix.from_config(suite_mix("p1_uw"))
    assert m is not None
    slugs = {s.suffix() or "clean" for s in m.specs}
    assert "clean" in slugs
    # trzy poziomy obserwowalności z `domain/sensors.py` mają być reprezentowane
    assert any(s.kind == "gauss" for s in m.specs)  # obserwowalne
    assert any(s.kind == "mask" and s.substitution == "hold" for s in m.specs)  # wnioskowalne
    assert any(s.kind == "drift" for s in m.specs)  # nieobserwowalne
