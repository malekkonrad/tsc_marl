"""Trening polityki Z mechanizmem — awarie w treningu (domain randomization) + log ρ.

Odpowiedź na pytanie otwarte z `tsc_marl_research/faza3-po-t2.md` §3: T2 pokazało,
że UW-attention DOKLEJONE do gotowej polityki jest wobec ATT neutralne. To NIE jest
to samo, co „architektura trenowana z nim nic by nie dała" — i tego dotyczy ten
moduł.

DWIE RZECZY, KTÓRE MUSZĄ WEJŚĆ RAZEM (i dlaczego osobno nie mają sensu):

1. AWARIE W TRENINGU. Rdzeń Fazy 1 uczy się na CZYSTYM ruchu, gdzie zmierzone
   `uw_active` ≈ 3%, a `uw_spread` ≈ 0,39 nata. Polityka przez cały trening prawie
   nie zobaczyłaby modulacji — eksperyment wyszedłby pusty z powodu trywialnego
   i nie dałoby się go odróżnić od wyniku merytorycznego. Żeby mechanizm miał się
   czego nauczyć, awarie muszą być w rozkładzie treningowym.

2. TRZY RAMIONA, NIE DWA. Sam punkt (1) rodzi natychmiastowy zarzut „poprawa jest
   od treningu z awariami, nie od mechanizmu". Uczciwy układ to:

       rdzeń na czystym (jest — `p1_gate`)  ·  rdzeń z awariami BEZ UW (`p1_fail`)
                                            ·  rdzeń z awariami Z UW  (`p1_uw`)

   Ramiona 2 i 3 MUSZĄ widzieć ten sam harmonogram awarii, inaczej różnica między
   nimi miesza mechanizm z innym losowaniem. Stąd `FailureMix` losuje scenariusz
   ze strumienia zależnego WYŁĄCZNIE od (`seed` mixu, numer epizodu) — nie od
   `--seed` polityki. To ta sama zasada, co w `domain/sensors.py`: gdyby wzorzec
   awarii szedł z ziarna polityki, seedy różniłyby się jednocześnie inicjalizacją
   sieci I tym, które detektory padły, i rozrzutu nie dałoby się rozdzielić.

OGRANICZENIE DO NAZWANIA WPROST W ROZDZIALE 4. `u_ref` jest skalibrowane na zrzucie
BC z Advanced-MP (ATT ≈ 270 s), a w trakcie treningu rozkład stanów dryfuje wraz
z polityką, więc punkt odniesienia przestaje pasować. Obrona: głowa i `u_ref` zostają
ZAMROŻONE (liczą się pod `no_grad`, nie są w `parameters()`, nie ma ich
w checkpoincie). Przeliczanie `u_ref` w trakcie treningu wprowadziłoby sprzężenie
polityka↔odniesienie, którego potem nie da się rozplątać — i to jest powód, nie wygoda.

JAK TO WYGLĄDA W LICZBACH, bo to widać od pierwszego epizodu. Nieuczona polityka
korkuje sieć (ATT ≈ 615 s wobec 270 s w zrzucie), kolejki rosną poza zakres zrzutu,
σ głowy idzie w górę WSZĘDZIE i `relu(u − u_ref)` przestaje selekcjonować: na
`mask0p25` zmierzone `uw_spread` = 6,85 i `uw_active` = 47%, wobec 0,83 / kilku
procent na tym samym scenariuszu przy polityce wytrenowanej (`results/t2_uw.md`).
Na starcie treningu mechanizm jest więc NASYCONY, a nie precyzyjny. Dlatego
`uw_spread` i `uw_active` lecą do `metrics.csv` per epizod: ich opadanie wraz z ATT
jest jedyną obserwacją, która odróżnia „mechanizm stopniowo zaczyna coś znaczyć"
od „mechanizm przez cały trening był szumem".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from tsc_marl.domain.sensors import FailureSpec

if TYPE_CHECKING:
    import torch
    from torch import Tensor

CLEAN = FailureSpec()

#: Skąd UW-attention bierze u: z głowy obserwacyjnej, z prawdy warstwy awarii
#: (wyrocznia modulacji — górne ograniczenie, `controllers.OracleUncertainty`) albo
#: z licznika zamrożonych detektorów (`controllers.RuleUncertainty`, wdrażalny).
UW_SOURCES = ("head", "oracle", "rule")
#: Kanały modulacji po stronie źródła — opis przy `controllers.source_log_rho`.
UW_CHANNELS = ("all", "self", "edges")


def check_uw(uw: dict[str, Any]) -> str | None:
    """Błąd w spec `uw` albo None. JEDEN walidator dla planu T2 i suite'a treningowego.

    Każda literówka ma tu być błędem, a nie wartością domyślną: kolumna podpisana
    „reguła" liczona po cichu głową (albo trening „z regułą" uczony na głowie po
    stronie celu) powtarzałaby stary pomiar pod nową nazwą — i nic by tego nie zgłosiło.
    """
    from tsc_marl.uncertainty.obs_head import EDGE_SIDES  # noqa: PLC0415 - ciągnie torcha

    nieznane = set(uw) - {"tau", "side", "u", "channels", "threshold"}
    if nieznane:
        return f"nieznane klucze {sorted(nieznane)}"
    if "side" in uw and uw["side"] not in EDGE_SIDES:
        return f"side={uw['side']!r}, dozwolone: {EDGE_SIDES}"
    zrodlo = uw.get("u", "head")
    if zrodlo not in UW_SOURCES:
        return f"u={zrodlo!r}, dozwolone: {UW_SOURCES}"
    # wyrocznia i reguła istnieją tylko po stronie źródła (`source_log_rho`) —
    # wymagamy tego JAWNIE, żeby w planie nie stało coś, czego sterownik nie liczy
    if zrodlo in ("oracle", "rule") and uw.get("side") != "source":
        return f"modulacja u={zrodlo} wymaga side: source"
    if "channels" in uw:
        if uw["channels"] not in UW_CHANNELS:
            return f"channels={uw['channels']!r}, dozwolone: {UW_CHANNELS}"
        if zrodlo == "head":
            return "channels tylko z u: oracle albo u: rule (głowa nie rozdziela kanałów)"
    if "threshold" in uw and zrodlo != "rule":
        return "threshold tylko z u: rule (próg licznika zamrożeń)"
    return None


# ------------------------------------------------------------------ mieszanka awarii


@dataclass(frozen=True)
class FailureMix:
    """Rozkład scenariuszy awarii na epizody treningowe.

    `sample(episode)` losuje z WŁASNEGO strumienia kluczowanego numerem epizodu,
    a nie z globalnego generatora: dzięki temu harmonogram nie zależy ani od
    `--seed`, ani od tego, ile epizodów już poleciało (wznowiony trening od epizodu
    137 dostaje ten sam scenariusz co bieg ciągły). Ten sam wzorzec, co
    `SensorModel.reset`, i z tego samego powodu.

    KTÓRE detektory padną, wybiera już `SensorModel` z (`spec.seed`, `episode`) —
    więc dwa epizody z tym samym scenariuszem i tak psują inne pasy.
    """

    specs: tuple[FailureSpec, ...]
    weights: tuple[float, ...]
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.specs:
            raise ValueError("pusta mieszanka awarii — podaj co najmniej jeden scenariusz")
        if len(self.specs) != len(self.weights):
            raise ValueError(f"{len(self.specs)} scenariuszy, {len(self.weights)} wag")
        if any(w < 0 for w in self.weights):
            raise ValueError(f"wagi muszą być nieujemne, są {self.weights}")
        if sum(self.weights) <= 0:
            raise ValueError("suma wag musi być dodatnia")

    @property
    def probabilities(self) -> np.ndarray:
        w = np.asarray(self.weights, dtype=np.float64)
        return w / w.sum()

    def sample(self, episode: int) -> FailureSpec:
        rng = np.random.default_rng([self.seed, int(episode)])
        return self.specs[int(rng.choice(len(self.specs), p=self.probabilities))]

    def as_dict(self) -> dict[str, Any]:
        """Do parametrów biegu — bez tego z `runs/` nie wynika, na czym uczyła się
        polityka, a dwa ramiona z różnymi mieszankami wyglądałyby tak samo."""
        return {
            "seed": self.seed,
            "scenarios": [
                {"weight": float(w), **s.as_dict()}
                for s, w in zip(self.specs, self.weights, strict=True)
            ],
        }

    def describe(self) -> str:
        from tsc_marl.robustness.degrade import slug  # noqa: PLC0415 - cykl importu

        return ", ".join(
            f"{slug(s)}={p:.0%}" for s, p in zip(self.specs, self.probabilities, strict=True)
        )

    @classmethod
    def from_config(cls, raw: Any) -> FailureMix | None:
        """`model.failure_mix` z YAML-a -> mieszanka albo None (bieg bez awarii).

        Format (patrz `configs/suites/p1_fail.yml`):

            failure_mix:
              seed: 0
              scenarios:
                - {weight: 3, kind: none}
                - {weight: 1, kind: gauss, level: 2.0}

        Nieznane pola scenariusza wywalają się w `FailureSpec.__post_init__` —
        literówka w YAML-u ma paść przy budowie, nie po 200 epizodach.
        """
        if not raw:
            return None
        if not isinstance(raw, dict) or "scenarios" not in raw:
            raise ValueError("failure_mix musi być mapą z kluczem `scenarios`")
        specs, weights = [], []
        for entry in raw["scenarios"]:
            item = dict(entry)
            weights.append(float(item.pop("weight", 1.0)))
            specs.append(FailureSpec(**item))
        return cls(tuple(specs), tuple(weights), seed=int(raw.get("seed", 0)))

    @classmethod
    def from_registry(cls) -> FailureMix | None:
        from common.registry import Registry  # noqa: PLC0415 - dopiero po bootstrapie

        param = Registry.mapping["model_mapping"]["setting"].param
        return cls.from_config(param.get("failure_mix"))


# --------------------------------------------------------- głowa w pętli treningowej


def online_uncertainty(
    network: str,
    n_mov: int,
    edge_index: Tensor,
    *,
    tau: float | None = None,
    side: str = "target",
) -> Any:
    """`OnlineUncertainty` z ZAMROŻONYCH artefaktów `runs/heads/` dla tej sieci.

    Wspólne dla ewaluacji (`runner/degrade.build_uncertainty`) i dla treningu:
    dwie kopie tego ładowania rozjechałyby się przy pierwszej zmianie nazw plików,
    a rozjazd „u_ref z innej głowy niż wagi" jest cichy — to liczby w innych
    jednostkach, nie błąd. `load_reference` sprawdza odcisk pliku głowy i odmawia.

    `network` ma być siecią BAZOWĄ (bez sufiksu wariantu popytu): artefakty są
    związane z siecią, głowa z JiNan na HangZhou ma AUROC 0,53
    (`results/gate_2_2.md`).

    `side` wybiera PLIK odniesienia (`oh.reference_path`), a strona zapisana w nim
    musi się z nim zgadzać — plik przemianowany ręcznie dałby u po jednej stronie
    krawędzi i próg po drugiej, bez żadnego wyjątku.
    """
    from tsc_marl.uncertainty import obs_head as oh  # noqa: PLC0415

    head, ref = oh.head_path(network), oh.reference_path(network, side)
    if not head.is_file():
        raise SystemExit(
            f"brak głowy obserwacyjnej dla '{network}' ({head.name}) — "
            f"policz ją: tsc obshead -n {network} --save --window 12 --persistence"
        )
    if not ref.is_file():
        raise SystemExit(
            f"brak odniesienia '{ref.name}' dla '{network}' — "
            + (
                f"policz je: tsc obshead -n {network} --save --window 12 --persistence"
                if side == "target"
                else f"policz je dla istniejącej głowy: tsc obshead -n {network} --reference {side}"
            )
        )
    fitted = oh.load_head(head, network=network)
    reference = oh.load_reference(ref, network=network, head=head)
    if reference.side != side:
        raise SystemExit(f"{ref.name} ma side={reference.side!r}, a plan chce {side!r}")
    sources = oh.movement_sources(network, n_mov)
    return oh.OnlineUncertainty(
        fitted, reference, sources, edge_index, tau=oh.DEFAULT_TAU if tau is None else float(tau)
    )


def gate_reference(network: str) -> Any:
    """Progi bramek realnych dla sieci BAZOWEJ (`runs/heads/gate_ref__<sieć>.npz`).

    Wspólne dla bramki-reguły w ewaluacji, modulacji z reguły w ewaluacji i modulacji
    z reguły w TRENINGU — jedno ładowanie, żeby polityka nie uczyła się na innym
    oknie albo progu, niż potem mierzy ją `tsc degrade`."""
    from tsc_marl.robustness import rule_ref  # noqa: PLC0415
    from tsc_marl.uncertainty import obs_head as oh  # noqa: PLC0415

    sciezka = oh.gate_path(network)
    if sciezka.is_file():
        return oh.load_gates(sciezka, network=network, head=oh.head_path(network))
    # Sieć bez głowy (Kraków, SUMO): sam próg reguły, tym samym przepisem — patrz
    # `robustness/rule_ref.py`. `uq` jest wtedy None i bramka UQ się nie zbuduje.
    regula = rule_ref.rule_path(network)
    if regula.is_file():
        return rule_ref.load_rule(regula, network=network)
    raise SystemExit(
        f"brak progów bramki dla '{network}' — policz je: "
        f"tsc obshead -n {network} --save --window 12 --persistence (CityFlow, z głową) "
        f"albo tsc rulecal -n {network} (sama reguła, także SUMO)"
    )


def uw_signal(uw: dict[str, Any], network: str, env: Any) -> Any:
    """Źródło log ρ dla UW-attention ze spec `uw` — JEDNO dla treningu i ewaluacji.

    `u: head` (domyślnie) — zamrożona głowa obserwacyjna po stronie `side`;
    `u: oracle` — prawda z warstwy awarii (sufit, `controllers.OracleUncertainty`);
    `u: rule` — licznik zamrożonych detektorów z oknem i progiem bramki-reguły
    (`controllers.RuleUncertainty`). `channels` dotyczy tylko oracle/rule (strona
    źródła). `network` = sieć BAZOWA, `env` = `TrafficEnv` (czytany na bieżąco:
    `env.features` podmienia się per scenariusz i per epizod mieszanki awarii).

    Przed tą funkcją trening znał wyłącznie głowę po stronie celu (`p1_uw`), a
    ewaluacja wszystkie trzy źródła — więc sygnał z reguły dało się mierzyć, ale
    nie dało się z nim uczyć."""
    from tsc_marl.robustness import controllers as c  # noqa: PLC0415

    tau = uw.get("tau")
    zrodlo = uw.get("u", "head")
    channels = uw.get("channels", "all")
    edge_index = env.graph.edge_index
    if zrodlo == "oracle":
        return c.OracleUncertainty(
            env, edge_index, tau=float(0.25 if tau is None else tau), channels=channels
        )
    if zrodlo == "rule":
        progi = gate_reference(network)
        return c.RuleUncertainty(
            edge_index,
            window=progi.window,
            threshold=float(uw.get("threshold", progi.rule)),
            n_mov=env.features.n_mov,
            tau=float(0.25 if tau is None else tau),
            channels=channels,
            real=env.features.movement_mask(),
        )
    return online_uncertainty(
        network, env.features.n_mov, edge_index, tau=tau, side=uw.get("side", "target")
    )


class UWStream:
    """log ρ krok po kroku + diagnostyka epizodu — to samo, co robi `UWPolicyController`.

    Wydzielone, bo trening potrzebuje DOKŁADNIE tej samej ścieżki co ewaluacja:
    ten sam bufor okna, ten sam `reset()` na starcie epizodu, ten sam `uw_spread`.
    Gdyby trening liczył log ρ inaczej niż `tsc degrade`, polityka uczyłaby się
    jednego sygnału, a była mierzona drugim — i nic by tego nie zgłosiło.

    URZĄDZENIE. Głowa liczy na CPU (jeden mały MLP na 12 wierszach, `no_grad`),
    bo `OnlineUncertainty` trzyma bufor w numpy, a `group_spread` i tak wraca na
    CPU. Na urządzenie polityki jedzie dopiero gotowy wektor [E+N] — raz na
    decyzję, 240 razy na epizod, więc transfer jest szumem wobec kroku symulatora.
    """

    def __init__(
        self,
        uncertainty: Any,
        edge_index: Tensor,
        device: torch.device | str = "cpu",
        *,
        n_nodes: int | None = None,
    ) -> None:
        import torch  # noqa: PLC0415

        self.uncertainty = uncertainty
        self.edge_index = edge_index  # CPU — kolejność krawędzi, nie obliczenia
        self.device = torch.device(device)
        self.n_nodes = int(n_nodes if n_nodes is not None else int(edge_index.max()) + 1)
        self.reset()

    def reset(self) -> None:
        """Czyszczenie bufora okna to POPRAWNOŚĆ, nie higiena: odczyt sprzed
        `world.reset()` opisuje zupełnie inny stan ruchu, a głowa policzyłaby
        z niego wysokie σ i mechanizm zaczynałby epizod od wygaszania losowych
        sąsiadów."""
        self.uncertainty.reset()
        self._spread: list[float] = []
        self._active: list[float] = []

    def log_rho(self, obs: np.ndarray) -> Tensor:
        """[E+N] na urządzeniu polityki; przy okazji zbiera diagnostykę epizodu."""
        from tsc_marl.uncertainty.obs_head import group_spread  # noqa: PLC0415

        lr = self.uncertainty.log_rho(obs)  # CPU
        self._spread.append(group_spread(lr, self.edge_index, self.n_nodes))
        self._active.append(float((lr < 0).float().mean()))
        return lr.to(self.device)

    def episode_stats(self) -> dict[str, float]:
        """`uw_spread` = średni rozstęp log ρ w GRUPIE SOFTMAXU węzła, czyli jedyna
        część sygnału, którą attention widzi (na stałą wspólną dla grupy jest
        niezmienniczy). Bez tej liczby krzywa uczenia ramienia 3 jest nieodróżnialna
        od ramienia 2, które po prostu nie ma mechanizmu."""
        if not self._spread:
            return {}
        return {
            "uw_spread": float(np.mean(self._spread)),
            "uw_active": float(np.mean(self._active)),
        }


def uw_config_from_registry() -> dict[str, Any] | None:
    """`model.uw` z YAML-a -> {'tau': ...} albo None (bieg bez mechanizmu).

    `uw: True` też działa i znaczy „domyślne τ" — τ jest dobrane raz, na tabeli
    rozstępu log ρ (`obs_head.DEFAULT_TAU`), a nie strojone per bieg.
    """
    from common.registry import Registry  # noqa: PLC0415 - dopiero po bootstrapie

    raw = Registry.mapping["model_mapping"]["setting"].param.get("uw")
    if not raw:
        return None
    uw = {} if raw is True else dict(raw)
    blad = check_uw(uw)
    if blad:
        raise SystemExit(f"model.uw w suite: {blad}")
    return uw
