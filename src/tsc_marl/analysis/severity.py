"""Przelicznik dotkliwości awarii: poziom w SPECYFIKACJI → odchylenie na CESZE.

Po co to istnieje. RobustLight ([[li-2025-robustlight]]) ma stan EP + running,
czyli nasz ATS, i podaje skalę 3,5–4,0 dla `gauss`/`urand` — ale wstrzykuje szum
na poziomie STANU, a `domain.sensors` na poziomie DETEKTORA (patrz uzasadnienie
tam). To nie jest ta sama liczba: e(l,m) to RÓŻNICA ŚREDNICH po pasach
wjazdowych i wyjazdowych (Def. 4), więc sigma per pas wychodzi na cesze
osłabiona o ~1/sqrt(M) na każdą stronę i wzmocniona przez odejmowanie dwóch
niezależnych szumów. Bez zmierzonego przelicznika krzywe degradacji T2 nie są
porównywalne z liczbami z papieru, a to jest główny wynik pracy.

Mierzymy EMPIRYCZNIE, nie wyprowadzamy: M różni się między ruchami, zliczenie
jest ucinane w zerze (`max(0, ...)`), a maska i rozkalibrowanie w ogóle nie mają
„sigmy" — ich dotkliwość widać dopiero na cesze.

    tsc severity                      # JiNan, siatka domyślna -> results/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tsc_marl import paths
from tsc_marl.domain.sensors import FailureSpec

#: Siatka scenariuszy = to, co pojawi się w T2. `gauss` w kilku poziomach, żeby
#: dało się zobaczyć, czy przelicznik jest liniowy (dla maski i dryfu nie będzie).
DEFAULT_SPECS = (
    FailureSpec(kind="gauss", level=1.0),
    FailureSpec(kind="gauss", level=2.0),
    FailureSpec(kind="gauss", level=3.5),
    FailureSpec(kind="gauss", level=4.0),
    FailureSpec(kind="urand", level=3.5),
    FailureSpec(kind="urand", level=4.0),
    FailureSpec(kind="mask", level=0.25),
    FailureSpec(kind="mask", level=0.5),
    FailureSpec(kind="mask", level=0.25, unit="intersection"),
    FailureSpec(kind="drift", level=0.2),
    FailureSpec(kind="drift", level=0.3),
)


@dataclass(frozen=True)
class Severity:
    """Dotkliwość jednego scenariusza, osobno dla obu bloków cech ATS."""

    spec: FailureSpec
    ep_std: float  # odchylenie standardowe (popsute − czyste) na bloku EP
    ep_mad: float  # średnie odchylenie bezwzględne
    demand_std: float
    demand_mad: float
    touched: float  # ułamek komórek, które w ogóle się zmieniły

    @property
    def transfer(self) -> float | None:
        """ep_std / poziom — przelicznik „sigma na detektorze → sigma na cesze".
        Sensowny tylko dla szumu addytywnego; dla maski i dryfu `level` znaczy
        co innego (ułamek jednostek, względne rozkalibrowanie)."""
        if self.spec.kind not in ("gauss", "urand") or not self.spec.level:
            return None
        return self.ep_std / self.spec.level


def measure(
    network: str = "cityflow_jinan3x4_4ph",
    specs: tuple[FailureSpec, ...] = DEFAULT_SPECS,
    *,
    decisions: int = 60,
    warmup: int = 300,
    action_interval: int = 15,
) -> tuple[list[Severity], tuple[float, float]]:
    """Wszystkie scenariusze na TYCH SAMYCH stanach silnika.

    Jeden świat, jeden `StateBuilder` na scenariusz plus jeden czysty: dzięki
    temu różnica między wierszami tabeli to wyłącznie scenariusz, a nie inny
    ruch. Sterowanie w trakcie pomiaru to FixedTime (faza cyklowana co decyzję) —
    nie chodzi o jakość sterowania, tylko o zapełnioną, zmienną sieć.
    """
    from tsc_marl.libsignal import session

    with session.standalone_world("advancedcolight", network, n_agents=1) as (world, _, _):
        return _measure(world, specs, decisions, warmup, action_interval)


def _measure(world, specs, decisions, warmup, action_interval):
    from tsc_marl.core.features import StateBuilder

    inters = world.intersections
    n_phases = max(len(i.phases) for i in inters)
    clean_builder = StateBuilder.from_registry(world, inters, failure=None)
    builders = [StateBuilder.from_registry(world, inters, failure=s) for s in specs]
    n_mov = clean_builder.n_mov

    diffs: list[list[np.ndarray]] = [[] for _ in specs]
    clean_rows: list[np.ndarray] = []
    total = warmup + decisions * action_interval
    for step in range(total):
        if step >= warmup and (step - warmup) % action_interval == 0:
            clean = clean_builder.observe()
            clean_rows.append(clean)
            for acc, b in zip(diffs, builders, strict=True):
                acc.append(b.observe() - clean)
        world.step([(step // action_interval) % len(i.phases) for i in inters])

    rows = [
        _summarize(spec, np.concatenate(acc), n_mov, n_phases)
        for spec, acc in zip(specs, diffs, strict=True)
    ]
    return rows, _zero_share(np.concatenate(clean_rows), n_mov)


def _zero_share(clean: np.ndarray, n_mov: int) -> tuple[float, float]:
    """Ułamek DOKŁADNYCH ZER w obu blokach cech na czystym biegu.

    Liczba decyzyjna dla semantyki podstawienia: zero jest wartością MODALNĄ,
    więc martwy detektor raportujący 0 wygląda dokładnie jak zdrowy detektor na
    pustym wlocie — i dlatego maska zmienia znacznie mniej komórek, niż wynosi
    ułamek zepsutych pasów. Rośnie też z tego wniosek do T2: dotkliwość maski
    zależy od OBCIĄŻENIA sieci, więc ten sam procent maski nie znaczy tego
    samego na JiNan i na HangZhou."""
    ep, demand = clean[:, :n_mov], clean[:, n_mov : 2 * n_mov]
    return float((ep == 0).mean()), float((demand == 0).mean())


def _summarize(spec: FailureSpec, diffs: np.ndarray, n_mov: int, n_phases: int) -> Severity:
    ep, demand = diffs[:, :n_mov], diffs[:, n_mov : 2 * n_mov]
    body = diffs[:, :-n_phases]
    return Severity(
        spec=spec,
        ep_std=float(ep.std()),
        ep_mad=float(np.abs(ep).mean()),
        demand_std=float(demand.std()),
        demand_mad=float(np.abs(demand).mean()),
        touched=float((np.abs(body) > 1e-9).mean()),
    )


def to_markdown(rows: list[Severity], zero_share: tuple[float, float], network: str) -> str:
    out = [
        "# Przelicznik dotkliwości awarii (poziom w specyfikacji → odchylenie na cesze)",
        "",
        f"Sieć: `{network}`. Stan: ATS (blok EP + blok demand), okno `obs_length`.",
        "Kolumna **przelicznik** = `ep_std / level` — o tyle szum wstrzyknięty NA DETEKTORZE",
        "jest słabszy/mocniejszy na cesze, którą widzi polityka. Bez tego poziomy",
        "z [[li-2025-robustlight]] (3,5–4,0 na STANIE) nie są porównywalne z naszymi.",
        "",
        "| scenariusz | poziom | jedn. | EP std | EP MAD | demand std | demand MAD |"
        " zmienione | przelicznik |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        t = f"{r.transfer:.3f}" if r.transfer is not None else "—"
        out.append(
            f"| {r.spec.kind} | {r.spec.level:g} | {r.spec.unit} | {r.ep_std:.3f} | "
            f"{r.ep_mad:.3f} | {r.demand_std:.3f} | {r.demand_mad:.3f} | "
            f"{r.touched:.1%} | {t} |"
        )
    ep_zero, demand_zero = zero_share
    out += [
        "",
        "## Jak to czytać",
        "",
        f"Na czystym biegu **{ep_zero:.0%} komórek bloku EP i {demand_zero:.0%} bloku "
        "demand to dokładnie zero**. Stąd dwa wnioski, oba istotne dla T2:",
        "",
        "1. **Maska jest w dużej części no-opem.** Zero jest wartością MODALNĄ, więc martwy",
        "   detektor raportujący 0 wygląda jak zdrowy detektor na pustym wlocie — kolumna",
        "   *zmienione* jest znacznie niższa niż ułamek zepsutych pasów. To nie jest usterka",
        "   pomiaru, tylko własność podstawienia `zero`, i dokładnie ona czyni maskę awarią",
        "   CICHĄ (patrz `domain/sensors.py`: podstawienie nie jest neutralne).",
        "2. **Dotkliwość maski zależy od OBCIĄŻENIA sieci.** Ten sam procent maski nie znaczy",
        "   tego samego na JiNan i na HangZhou (6295 vs 2983 poj./h) — w pracy trzeba podawać",
        "   ułamek zmienionych komórek obok procentu maski, inaczej krzywe degradacji obu",
        "   sieci nie są porównywalne między sobą.",
        "",
        "Dla szumu addytywnego przelicznik ma przewidywalne źródło: e(l,m) to różnica ŚREDNICH",
        "po pasach wjazdowych i wyjazdowych, więc przy M pasach z każdej strony sigma na cesze",
        "wynosi sigma·sqrt(1/M_in + 1/M_out) — dla M=3 daje to 0,816. Odchylenia od tej wartości",
        "biorą się z ruchów o innym M i z ucięcia w zerze (`max(0, ...)`), które przy większych",
        "sigma zjada coraz więcej masy — stąd przelicznik MALEJE z poziomem.",
        "`urand` o tym samym `level` jest istotnie SŁABSZY (szerokość a to sigma = a/sqrt(3)),",
        'więc porównywanie obu ataków „po tej samej liczbie" jest mylące.',
    ]
    return "\n".join(out) + "\n"


def main(network: str = "cityflow_jinan3x4_4ph") -> Path:
    rows, zero_share = measure(network)
    path = paths.RESULTS / "sensor_severity.md"
    path.write_text(to_markdown(rows, zero_share, network), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(main())
