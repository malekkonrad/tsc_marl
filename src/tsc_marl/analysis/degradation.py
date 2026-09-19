"""Raport T2 bez mechanizmu: tabela i krzywe degradacji zamrożonych sterowników.

Czyta biegi `runs/t2/*__<plan>` (nic nie uruchamia). Dwie decyzje, które zmieniają
liczby, więc są tu jawnie:

* UŚREDNIANIE DWUSTOPNIOWE — najpierw po LOSOWANIACH awarii w obrębie seeda, potem
  mean±std po SEEDACH polityki. Std w tabeli to więc rozrzut między politykami, nie
  między losowaniami; płaska średnia po wszystkich wierszach mieszałaby oba źródła
  (patrz `domain/sensors.py`: RNG awarii jest celowo niezależny od `--seed`).
* ODNIESIENIE — degradacja Δ liczona względem CZYSTEGO wyniku TEGO SAMEGO sterownika
  w tym samym ewaluatorze, nie względem T1. Na czystym ewaluator zgadza się z T1
  co do decyzji poza remisami (`robustness/controllers.py`), ale remisy potrafią
  przesunąć ATT o ~1%, a to jest rząd małych degradacji.

Kontrola spójności: czysty wynik uczonego sterownika MUSI równać się ostatniemu
testowi z jego treningu (te same wagi, argmax, deterministyczny CityFlow). Raport
liczy tę różnicę — niezerowa znaczy, że ewaluator nie mierzy tej polityki, którą
mierzy tabela T1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tsc_marl.domain.sensors import FailureSpec
from tsc_marl.paths import RESULTS, RUNS
from tsc_marl.robustness.degrade import EXPERIMENT, LEARNED, Plan, slug

# Kolory: sloty 1–4 palety kategorycznej (walidacja: CVD ΔE ≥ 9,1, normal ≥ 22,9).
# Kolor idzie za STEROWNIKIEM, nie za pozycją — brakujący sterownik nie przemalowuje
# pozostałych.
COLORS = {
    "gatppo": "#2a78d6",
    "advancedcolight": "#eb6834",
    "efficientmp": "#1baf7a",
    "advancedmp": "#eda100",
    # warianty actora z ablacji 1.6 (plan t2_ablations): sloty 5 i 7 palety, żeby
    # nie pożyczać koloru Advanced-CoLighta ani reguł z t2_nomech (walidacja
    # z rdzeniem: CVD ΔE ≥ 13,0, normal ≥ 27,5)
    "gatppo_mean": "#e87ba4",
    "gatppo_nognn": "#4a3aa7",
    # mechanizmy Fazy 3 (plany t2_uw, t2_gate). Kolejne sloty tej samej palety;
    # ΔE nie przeliczane, bo te kolumny nie pojawiają się na jednym wykresie
    # z wariantami ablacyjnymi — w swoich planach sąsiadują tylko z rdzeniem.
    "gatppo_uw": "#d63f6e",
    "gatppo_gate": "#7a4fd6",
    # SWEEP PROGU bramki (t2_gate) to dane UPORZĄDKOWANE — jeden parametr rosnący —
    # więc rampa jednej barwy od jasnej do ciemnej, a nie cztery kolory kategorialne.
    # Kategorialne sugerowałyby cztery różne mechanizmy zamiast czterech nastawień
    # jednego; kolejność progów przestałaby być czytelna z samego wykresu.
    "gatppo_gate_t0p25": "#b49ae8",
    "gatppo_gate_t0p50": "#8f66dd",
    "gatppo_gate_t0p75": "#6a3fc4",
    "gatppo_gate_t1p00": "#452a8a",
    # bramki realne (t2_gate_real): wyrocznia zostaje fioletowa jak sweep, a dwa
    # sygnały wdrażalne dostają własne barwy — tu to SĄ różne mechanizmy, nie
    # nastawienia jednego, więc kategorialnie, a nie rampą.
    "gatppo_gate_oracle": "#7a4fd6",
    "gatppo_gate_rule": "#1baf7a",
    "gatppo_gate_uq": "#eda100",
}
LABELS = {
    "gatppo": "rdzeń GAT+PPO+CTDE",
    "advancedcolight": "Advanced-CoLight",
    "efficientmp": "Efficient-MP",
    "advancedmp": "Advanced-MP",
    "fixedtime": "FixedTime",
    "gatppo_mean": "actor mean-pool",
    "gatppo_nognn": "actor bez GNN",
    "gatppo_ens": "ensemble 5 actorów (decyduje seed 0)",
    "gatppo_uw": "rdzeń + UW-attention",
    "gatppo_gate": "rdzeń + bramka (wyrocznia)",
    "gatppo_gate_t0p25": "bramka, próg 0,25",
    "gatppo_gate_t0p50": "bramka, próg 0,50",
    "gatppo_gate_t0p75": "bramka, próg 0,75",
    "gatppo_gate_t1p00": "bramka, próg 1,00",
    "gatppo_gate_oracle": "bramka: wyrocznia (sufit)",
    "gatppo_gate_rule": "bramka: reguła (trwałość)",
    "gatppo_gate_uq": "bramka: UQ (głowa)",
}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

#: Panele wykresu = rodziny awarii, w kolejności skali obserwowalności (2.0a).
FAMILIES = (
    ("szum gaussowski σ", lambda s: s.kind == "gauss"),
    ("martwe detektory, zero", lambda s: s.kind == "mask" and s.unit == "lane"
     and s.substitution == "zero"),
    ("martwe skrzyżowania", lambda s: s.kind == "mask" and s.unit == "intersection"),
    ("martwe bez śladu, hold", lambda s: s.kind == "mask" and s.substitution == "hold"),
    ("rozkalibrowanie", lambda s: s.kind == "drift"),
)  # fmt: skip


def load(plan: Plan, metrics: tuple[str, ...] = ("att", "throughput")):
    """Długa ramka: controller, network, seed, scenario, draw + kolumny `metrics`."""
    import pandas as pd  # noqa: PLC0415

    from runlog import Experiment  # noqa: PLC0415

    exp = Experiment(EXPERIMENT, root=RUNS)
    frames = []
    for run in exp.runs():
        if run.status != "ok" or not run.id.endswith(f"__{plan.name}"):
            continue
        path = run.path / "metrics.csv"
        if not path.is_file():
            continue
        m = pd.read_csv(path)
        m = m[m["metric"].isin(list(metrics))]
        wide = m.pivot_table(index=["split", "step"], columns="metric", values="value")
        wide = wide.reset_index().rename(columns={"split": "scenario", "step": "draw"})
        p = run.params
        # Tylko kolumny TEGO planu. Filtr po sufiksie `__<plan>` nie wystarcza:
        # przemianowanie kolumny (np. `gatppo_gate` -> `gatppo_gate_t0p50` przy
        # sweepie progu) zostawia w `runs/t2` biegi ze starą nazwą i pasującym
        # sufiksem. Wchodziły do ramki jako widmowa kolumna i wywalały wykres na
        # `plan.kind()` — a w tabelach były niewidoczne, bo te iterują po planie.
        if p["controller"] not in plan.controllers:
            continue
        wide["controller"], wide["network"], wide["seed"] = p["controller"], p["network"], p["seed"]
        wide["parent_run"] = p.get("parent_run")
        frames.append(wide)
    if not frames:
        return pd.DataFrame(
            columns=["controller", "network", "seed", "scenario", "draw", "att", "throughput"]
        )
    return pd.concat(frames, ignore_index=True)


def aggregate(df):
    """(controller, network, scenario) -> att_mean, att_std (po seedach), n_seeds, n_draws."""
    if df.empty:
        return df
    per_seed = (
        df.groupby(["controller", "network", "scenario", "seed"])
        .agg(att=("att", "mean"), draws=("draw", "nunique"))
        .reset_index()
    )
    out = (
        per_seed.groupby(["controller", "network", "scenario"])
        .agg(
            att_mean=("att", "mean"),
            att_std=("att", "std"),
            n_seeds=("seed", "nunique"),
            n_draws=("draws", "min"),
        )
        .reset_index()
    )
    out["att_std"] = out["att_std"].fillna(0.0)
    clean = out[out["scenario"] == "clean"].set_index(["controller", "network"])["att_mean"]
    out["delta"] = [
        r.att_mean / clean.get((r.controller, r.network), np.nan) - 1.0 for r in out.itertuples()
    ]
    return out


#: Tytuł raportu zależy od planu — „T2 bez mechanizmu" było zaszyte na sztywno i
#: kłamało w raportach `t2_ablations` (ablacje agregacji) oraz `t2_uw` (plan istnieje
#: WYŁĄCZNIE po to, żeby zmierzyć mechanizm).
TITLES = {
    "t2_nomech": "T2 bez mechanizmu",
    "t2_ablations": "T2 dla ablacji agregacji",
    "t2_uw": "T2 z UW-attention",
    "t2_ood": "T2 na ruchu spoza treningu",
    "t2_gate": "T2 z bramką fallbacku",
    "t2_gate_real": "T2 z bramką realną (reguła vs UQ)",
}

#: Próg |t| dla różnicy parowanej. 2,8 ≈ poprawka Bonferroniego na ~26 porównań
#: (13 scenariuszy × 2 sieci) przy α = 0,05 — bo tabelę czyta się CAŁĄ i wybranie
#: z niej najładniejszego wiersza po fakcie jest testowaniem wielokrotnym.
T_THRESHOLD = 2.8


def mechanism_pairs(plan: Plan) -> list[tuple[str, str]]:
    """[(kolumna z mechanizmem, kolumna odniesienia)] — pary liczone Z TEGO SAMEGO
    checkpointu. Dla UW-attention modulacja jest bezparametrowa, więc `agent` obu
    kolumn jest ten sam i różni je wyłącznie mechanizm."""
    out = []
    for label in plan.controllers:
        if plan.kind(label) not in ("policy_uw", "policy_gate"):
            continue
        baza = [
            c
            for c in plan.controllers
            if plan.kind(c) == "policy" and plan.agent(c) == plan.agent(label)
        ]
        if baza:
            out.append((label, baza[0]))
    return out


#: rodzaj mechanizmu -> (metryki z `result.json`, zdanie tłumaczące je w raporcie).
#: Każdy mechanizm musi mieć diagnostykę „czy w ogóle zadziałał", bo bez niej pusty
#: wynik jest nieodróżnialny od mechanizmu, który przez cały epizod milczał.
DIAGNOSTICS = {
    "policy_uw": (
        ("uw_spread", "uw_active"),
        "`uw_spread` = średni rozstęp log ρ w GRUPIE SOFTMAXU węzła, czyli jedyna część "
        "sygnału, którą attention widzi (na stałą wspólną dla grupy jest niezmienniczy). "
        "0 = warstwa nie widzi różnicy między sąsiadami, ln 10 ≈ 2,3 = różnią się 10×. "
        "`uw_active` = odsetek krawędzi z niezerowym log ρ.",
    ),
    "policy_gate": (
        ("gate_share", "gate_switches", "gate_precision", "gate_recall"),
        "`gate_share` = odsetek par (decyzja, węzeł), w których sterowanie przejęła "
        "reguła. `gate_switches` = liczba ZMIAN trybu na węzłach w epizodzie — to one "
        "psują ruch (przełączenie resetuje fazę), a nie sam udział. `gate_precision` "
        "i `gate_recall` liczą się względem WYROCZNI (prawda z warstwy awarii, "
        "wyłącznie diagnostycznie) i rozdzielają „detektor jest słaby” od "
        "„przełączanie mało daje” — dwie rzeczy, które samo ATT skleja.",
    ),
}


def mechanism_diagnostics(
    plan: Plan, controller: str, metrics: tuple[str, ...]
) -> dict[tuple[str, str], dict[str, float]]:
    """(sieć, scenariusz) -> średnie `metrics` po seedach.

    Te liczby siedzą w `result.json` (metryki epizodu sterownika), nie w `metrics.csv`,
    i bez nich raport NIE ODRÓŻNIA „mechanizm nie pomógł" od „mechanizm nie wystartował".
    To jest różnica między wynikiem a brakiem wyniku.
    """
    from runlog import Experiment  # noqa: PLC0415

    zebrane: dict[tuple[str, str], dict[str, list[float]]] = {}
    exp = Experiment(EXPERIMENT, root=RUNS)
    for run in exp.runs():
        if run.status != "ok" or not run.id.endswith(f"__{plan.name}"):
            continue
        if run.params.get("controller") != controller:
            continue
        net = run.params["network"]
        for key, value in run.result.items():
            for metric in metrics:
                if key.startswith(f"{metric}__"):
                    scen = key[len(metric) + 2 :]
                    zebrane.setdefault((net, scen), {}).setdefault(metric, []).append(float(value))
    return {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in zebrane.items()}


def paired_delta(df, mechanizm: str, baza: str):
    """Różnica ATT sparowana po (sieć, scenariusz, seed, losowanie).

    Dlaczego parowanie, a nie ±std z tabeli: obie kolumny jadą TYM SAMYM checkpointem
    i TYMI SAMYMI losowaniami awarii (`draw` wybiera, które detektory padły), więc
    rozrzut między seedami i między losowaniami jest WSPÓLNY dla obu i w różnicy się
    skraca. Porównanie nieparowane topi efekt rzędu 1 s w odchyleniu rzędu 30 s —
    i odwrotnie, potrafi pokazać różnicę 3%, która po sparowaniu jest szumem.
    """
    import pandas as pd  # noqa: PLC0415
    from scipy.stats import ttest_1samp  # noqa: PLC0415

    sub = df[df["controller"].isin([mechanizm, baza])]
    wide = sub.pivot_table(
        index=["network", "scenario", "seed", "draw"], columns="controller", values="att"
    ).dropna()
    if wide.empty:
        return pd.DataFrame()
    wide["delta"] = wide[mechanizm] - wide[baza]
    out = []
    for (net, scen), g in wide.groupby(["network", "scenario"]):
        d = g["delta"].to_numpy()
        t, pval = (
            ttest_1samp(d, 0.0) if len(d) > 1 and d.std() > 0 else (float("nan"), float("nan"))
        )
        # REMISY liczone osobno i WYŁĄCZONE z „lepszy w". Przy bramce fallbacku remis
        # to najczęstszy wynik — mechanizm nie strzelił, epizod poszedł identycznie —
        # a wliczony do mianownika udawałby przegraną: wiersz z Δ = +0,0 pokazywał
        # „lepszy w 0/15", co czyta się jak porażkę zamiast jak brak zmiany.
        out.append({
            "network": net, "scenario": scen, "n": len(d), "remis": int((d == 0).sum()),
            "delta": float(d.mean()), "sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            "t": float(t), "p": float(pval), "lepszy": int((d < 0).sum()),
        })  # fmt: skip
    return pd.DataFrame(out)


def parent_consistency(df, skip: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    """Czysty wynik ewaluatora vs ostatni test treningu — dla każdego uczonego biegu.

    `skip` = etykiety sterowników, dla których różnica jest OCZEKIWANA i nie jest
    rozjazdem: kolumna UW-attention jedzie tym samym checkpointem, ale z modulacją
    attention, więc na czystym z definicji NIE MUSI dać liczby z T1 co do setnej.
    To, jak bardzo się różni, jest wynikiem (A3a) i stoi w tabeli wyżej — kontrola
    „ewaluator mierzy tę samą politykę" dotyczy sterowników bez mechanizmu.
    """
    import pandas as pd  # noqa: PLC0415

    from tsc_marl.libsignal import experiment as t1  # noqa: PLC0415

    out = []
    runs = {r.id: r for r in t1.experiment().runs()}
    clean = df[(df["scenario"] == "clean") & df["parent_run"].notna()]
    for r in clean.itertuples():
        if r.controller in skip:
            continue
        parent = runs.get(r.parent_run)
        path = parent.path / "metrics.csv" if parent else None
        if path is None or not path.is_file():
            continue
        # tylko gdy ewaluacja jedzie po ruchu TRENINGOWYM rodzica — na innym pliku ruchu
        # (plan t2_ood) różnica z ostatnim testem treningu jest oczekiwana, nie jest błędem
        if parent.params.get("network") != r.network:
            continue
        m = pd.read_csv(path)
        test = m[(m["split"] == "test") & (m["metric"] == "att")].sort_values("step")
        if test.empty:
            continue
        last = float(test["value"].iloc[-1])
        out.append({"run": r.parent_run, "eval": float(r.att), "last_test": last,
                    "diff": float(r.att) - last})  # fmt: skip
    return out


# ------------------------------------------------------------------ markdown


def _spec_by_slug(plan: Plan) -> dict[str, FailureSpec]:
    return {slug(s): s for s in plan.scenarios}


def _mechanism_section(plan: Plan, df, mechanizm: str, baza: str) -> list[str]:
    """Dwie tabele, które rozstrzygają o wyniku planu z mechanizmem: CZY zadziałał
    i CZY zmienił ATT. Bez pierwszej druga jest nieinterpretowalna."""
    metryki, wyjasnienie = DIAGNOSTICS[plan.kind(mechanizm)]
    diag = mechanism_diagnostics(plan, mechanizm, metryki)
    pary = paired_delta(df, mechanizm, baza) if df is not None and not df.empty else None
    if not diag and (pary is None or pary.empty):
        return []

    etykieta = LABELS.get(mechanizm, mechanizm)
    lines = [
        f"## Mechanizm: {etykieta} vs {LABELS.get(baza, baza)}",
        "",
        "Obie kolumny liczą się z TEGO SAMEGO checkpointu i tych samych losowań awarii "
        "(`draw` wybiera, które detektory padły), więc porównujemy je **parami**, nie przez "
        "±std z tabeli wyżej: wspólny rozrzut między seedami i losowaniami skraca się "
        "w różnicy. `Δ` < 0 = mechanizm lepszy. `|t| ≥ "
        f"{T_THRESHOLD}` to próg po poprawce na tyle porównań, ile ma ta tabela.",
        "",
        wyjasnienie,
        "",
    ]
    for net in plan.networks:
        wiersze = pary[pary["network"] == net] if pary is not None and not pary.empty else None
        if wiersze is None or wiersze.empty:
            continue
        kolejnosc = [s for s in (slug(x) for x in plan.scenarios) if s in set(wiersze["scenario"])]
        lines += [
            f"### {net}",
            "",
            f"| scenariusz | {' | '.join(metryki)} | Δ ATT [s] | sd | t | lepszy w |",
            "|---|" + "---:|" * (len(metryki) + 4),
        ]
        for scen in kolejnosc:
            r = wiersze[wiersze["scenario"] == scen].iloc[0]
            d = diag.get((net, scen), {})
            # udziały procentowo, reszta (rozstęp log ρ, liczba przełączeń) liczbowo
            komorki = [
                (
                    "—"
                    if m not in d
                    else f"{d[m]:.1%}"
                    if m.endswith(("_active", "_share"))
                    else f"{d[m]:.2f}"
                )
                for m in metryki
            ]
            mocny = "**" if abs(r["t"]) >= T_THRESHOLD else ""
            # sd = 0 (scenariusz deterministyczny, np. `clean`) -> t nieokreślone.
            # „—" zamiast `nan`: brak rozrzutu to nie jest brak wyniku.
            t = "—" if not np.isfinite(r["t"]) else f"{r['t']:+.2f}"
            lines.append(
                f"| `{scen}` | {' | '.join(komorki)} | {mocny}{r['delta']:+.2f}{mocny} | "
                f"{r['sd']:.1f} | {t} | {int(r['lepszy'])}/{int(r['n'] - r['remis'])} |"
            )
        istotne = wiersze[wiersze["t"].abs() >= T_THRESHOLD]
        lines += [
            "",
            (
                "* Żaden scenariusz nie przekracza progu — mechanizm jest wobec ATT **neutralny**."
                if istotne.empty
                else "* Powyżej progu: "
                + ", ".join(
                    f"`{r.scenario}` ({r.delta:+.2f} s, t = {r.t:+.2f}, lepszy w "
                    f"{int(r.lepszy)}/{int(r.n)} parach)"
                    for r in istotne.itertuples()
                )
            ),
            "",
        ]
    return lines


def _sweep_section(plan: Plan, df, pary: list[tuple[str, str]]) -> list[str]:
    """KRZYWA ZAPASU: próg bramki × scenariusz -> Δ ATT (i `gate_share`).

    Zastępuje osobne sekcje per kolumna, gdy plan jest sweepem jednego parametru —
    cztery prawie identyczne tabele nie są czytelne, a to właśnie ta jedna jest
    wynikiem do rozdziału 4: mówi, jak agresywna ma być realna bramka.

    Czytać po KIERUNKU, nie po pojedynczym `t`: przy kilkunastu komórkach jedna
    przekroczy próg przypadkiem, a spójność znaku w kolumnie „lepszy w" nie.
    """
    progi = []
    for mechanizm, _ in pary:
        spec = plan.controllers[mechanizm].get("gate", {})
        progi.append((float(spec.get("threshold", 0.5)), mechanizm))
    progi.sort()
    baza = pary[0][1]

    lines = [
        "## Krzywa zapasu: próg bramki × scenariusz",
        "",
        f"Δ ATT [s] względem `{LABELS.get(baza, baza)}`, sparowana po (seed, losowanie); "
        "w nawiasie odsetek par (decyzja, węzeł) przejętych przez regułę. "
        f"**pogrubienie** = |t| ≥ {T_THRESHOLD}. Δ < 0 = bramka lepsza.",
        "",
        "Próg to ułamek pasów wjazdowych węzła w awarii, od którego oddaje sterowanie "
        "regule — im wyżej, tym bramka ostrożniejsza.",
        "",
    ]
    for net in plan.networks:
        kolejnosc, wiersze = None, {}
        for prog, mechanizm in progi:
            pary_df = paired_delta(df, mechanizm, baza)
            if pary_df.empty:
                continue
            sub = pary_df[pary_df["network"] == net]
            if sub.empty:
                continue
            wiersze[prog] = (
                sub.set_index("scenario"),
                mechanism_diagnostics(plan, mechanizm, ("gate_share",)),
            )
            if kolejnosc is None:
                kolejnosc = [
                    s for s in (slug(x) for x in plan.scenarios) if s in sub["scenario"].values
                ]
        if not wiersze:
            continue
        lines += [
            f"### {net}",
            "",
            "| próg | " + " | ".join(f"`{s}`" for s in kolejnosc) + " |",
            "|---|" + "---:|" * len(kolejnosc),
        ]
        for prog, (sub, diag) in sorted(wiersze.items()):
            komorki = []
            for scen in kolejnosc:
                r = sub.loc[scen]
                mocny = "**" if abs(r["t"]) >= T_THRESHOLD else ""
                share = diag.get((net, scen), {}).get("gate_share")
                udzial = "" if share is None else f" ({share:.0%})"
                komorki.append(f"{mocny}{r['delta']:+.1f}{mocny}{udzial}")
            lines.append(f"| {prog:.2f} | " + " | ".join(komorki) + " |")
        lines.append("")
    return lines


def to_markdown(
    plan: Plan, agg, checks: list[dict[str, Any]], figures: dict[str, Path], df=None
) -> str:
    specs = _spec_by_slug(plan)
    order = [slug(s) for s in plan.scenarios]
    ctrls = [c for c in plan.controllers if c != "fixedtime"]
    lines = [
        f"# {TITLES.get(plan.name, 'T2')} — plan `{plan.name}`",
        "",
        f"Tryb {plan.mode} (yellow={plan.yellow}), {plan.draws} losowań awarii na scenariusz, "
        "ewaluacja ZAMROŻONYCH sterowników (`tsc degrade`). Wszystkie czytają tę samą warstwę "
        "detektorów: MaxPressure liczy decyzję z popsutego wektora stanu, nie z silnika.",
        "",
        "Komórka = ATT [s], średnia po losowaniach, potem po seedach polityki (± std po "
        "seedach); w nawiasie Δ względem czystego wyniku tego samego sterownika. "
        "**▲ = gorzej niż FixedTime** — tylko tam dolna warstwa fallbacku ma pole do zysku.",
        "",
    ]
    for net in plan.networks:
        sub = agg[agg["network"] == net] if not agg.empty else agg
        ft = sub[(sub["controller"] == "fixedtime") & (sub["scenario"] == "clean")]
        ft_att = float(ft["att_mean"].iloc[0]) if len(ft) else np.nan
        lines += [f"## {net}", "", f"FixedTime (niezależny od odczytów): **{ft_att:.1f} s**.", ""]
        present = [c for c in ctrls if len(sub) and (sub["controller"] == c).any()]
        missing = [c for c in ctrls if c not in present]
        if not present:
            lines += ["_Brak policzonych sterowników dla tej sieci._", ""]
            continue
        head = "| scenariusz | " + " | ".join(
            f"{LABELS.get(c, c)} (n={int(sub[sub['controller'] == c]['n_seeds'].max())})"
            for c in present
        )
        lines += [head + " |", "|---|" + "---:|" * len(present)]
        for name in order:
            cells = []
            for c in present:
                row = sub[(sub["controller"] == c) & (sub["scenario"] == name)]
                if row.empty:
                    cells.append("—")
                    continue
                r = row.iloc[0]
                std = f" ± {r.att_std:.1f}" if r.n_seeds > 1 else ""
                mark = " ▲" if r.att_mean > ft_att else ""
                delta = "" if name == "clean" else f" ({r.delta:+.1%})"
                cells.append(f"{r.att_mean:.1f}{std}{delta}{mark}")
            lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
        lines.append("")
        if missing:
            lines += [
                f"_Niepoliczone na tej maszynie: {', '.join(LABELS.get(c, c) for c in missing)} "
                "(brak checkpointów rodzica — patrz `tsc degrade status`)._",
                "",
            ]
        lines += _findings(sub, present, ft_att, specs)
        if net in figures:
            lines += ["", f"![krzywe degradacji {net}](figures/{figures[net].name})", ""]

    pary = mechanism_pairs(plan)
    if len(pary) > 1:  # sweep jednego parametru: jedna tabela zamiast N sekcji
        lines += _sweep_section(plan, df, pary)
    else:
        for mechanizm, baza in pary:
            lines += _mechanism_section(plan, df, mechanizm, baza)

    if checks:
        worst = max(abs(c["diff"]) for c in checks)
        lines += [
            "## Kontrola: ewaluator mierzy tę samą politykę co T1",
            "",
            f"Czysty wynik ewaluatora vs ostatni test treningu, {len(checks)} biegów uczonych: "
            f"największa różnica **{worst:.3f} s**"
            + (" — zgodne co do precyzji logu." if worst < 0.05 else " — ROZJAZD, sprawdzić!"),
            "",
        ]
    lines += [
        "## Czego ta tabela NIE mówi",
        "",
        "* Poziomy są poziomami DETEKTORA. Na cechę `gauss` przenosi się ze współczynnikiem "
        "~0,8–0,9, `urand` jest ~1,6× słabszy przy tym samym `level` "
        "(`results/sensor_severity.md`).",
        "* Maska z podstawieniem `zero` zmienia znacznie mniej komórek cech, niż wynosi ułamek "
        "martwych pasów (zero jest wartością modalną) — ta sama liczba nie znaczy tego samego "
        "na sieciach o różnym obciążeniu.",
        "* Scenariusze `hold` zaczynają się w decyzji 60 (15. minuta), więc pierwsza ćwiartka "
        "epizodu jest czysta — ich Δ nie jest porównywalne wprost z maską od początku.",
    ]
    return "\n".join(lines) + "\n"


def _findings(sub, present, ft_att, specs) -> list[str]:
    """Wnioski liczone z tabeli, nie pisane z góry."""
    out = []
    for c in present:
        rows = sub[(sub["controller"] == c) & (sub["scenario"] != "clean")]
        if rows.empty:
            continue
        worst = rows.loc[rows["att_mean"].idxmax()]
        above = rows[rows["att_mean"] > ft_att]["scenario"].tolist()
        text = (
            f"* **{LABELS.get(c, c)}** — najgorszy scenariusz `{worst.scenario}`: "
            f"{worst.att_mean:.1f} s ({worst.delta:+.1%}). "
        )
        if above:
            text += (
                f"Gorzej niż FixedTime w {len(above)} scenariuszach: "
                + ", ".join(f"`{a}`" for a in above)
                + "."
            )
        else:
            text += (
                "Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do "
                "FixedTime nie ma z czego zarobić."
            )
        out.append(text)
    return out + [""]


# --------------------------------------------------------------------- wykres


def plot(plan: Plan, agg, network: str, path: Path) -> Path | None:
    """Małe wielokrotności: jeden panel na rodzinę awarii, wspólna oś Y (ATT)."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    sub = agg[agg["network"] == network]
    if sub.empty:
        return None
    specs = _spec_by_slug(plan)
    families = [(t, [s for s in plan.scenarios if f(s)]) for t, f in FAMILIES]
    families = [(t, s) for t, s in families if s]
    if not families:
        return None
    ft = sub[(sub["controller"] == "fixedtime") & (sub["scenario"] == "clean")]
    ft_att = float(ft["att_mean"].iloc[0]) if len(ft) else None

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    fig, axes = plt.subplots(
        1, len(families), figsize=(3.1 * len(families), 3.4), sharey=True, facecolor=SURFACE
    )
    axes = np.atleast_1d(axes)
    for ax, (title, fam) in zip(axes, families, strict=True):
        ax.set_facecolor(SURFACE)
        for c in (c for c in COLORS if (sub["controller"] == c).any()):
            xs, ys, sd = [0.0], [], []
            clean = sub[(sub["controller"] == c) & (sub["scenario"] == "clean")]
            if clean.empty:
                continue
            ys.append(float(clean["att_mean"].iloc[0]))
            sd.append(float(clean["att_std"].iloc[0]))
            for s in sorted(fam, key=lambda s: s.level):
                row = sub[(sub["controller"] == c) & (sub["scenario"] == slug(s))]
                if row.empty:
                    continue
                xs.append(s.level)
                ys.append(float(row["att_mean"].iloc[0]))
                sd.append(float(row["att_std"].iloc[0]))
            xs, ys, sd = np.array(xs), np.array(ys), np.array(sd)
            ax.plot(xs, ys, color=COLORS[c], lw=2, marker="o", ms=5.5, mec=SURFACE, mew=1.5,
                    solid_capstyle="round", label=LABELS[c], zorder=3)  # fmt: skip
            if plan.kind(c) in LEARNED and (sd > 0).any():
                ax.fill_between(xs, ys - sd, ys + sd, color=COLORS[c], alpha=0.10, lw=0)
        if ft_att is not None:
            ax.axhline(ft_att, color=INK2, lw=1, zorder=2)
        ax.set_title(title, color=INK, fontsize=9, loc="left")
        ax.set_xlabel("poziom awarii (detektor)", color=INK2)
        ax.grid(True, color=GRID, lw=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=INK2)
    axes[0].set_ylabel("ATT [s]", color=INK2)
    if ft_att is not None:
        axes[-1].annotate("FixedTime", (1.0, ft_att), xycoords=("axes fraction", "data"),
                          xytext=(-2, 3), textcoords="offset points", ha="right", va="bottom",
                          color=INK2, fontsize=8)  # fmt: skip
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               labelcolor=INK, bbox_to_anchor=(0.5, 1.02))  # fmt: skip
    fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    _ = specs
    return path


def main(plan: Plan) -> Path:
    df = load(plan)
    agg = aggregate(df)
    figures = {}
    for net in plan.networks:
        p = plot(plan, agg, net, RESULTS / "figures" / f"{plan.name}_{net}.png")
        if p is not None:
            figures[net] = p
    z_mechanizmem = frozenset(
        c for c in plan.controllers if plan.kind(c) in ("policy_uw", "policy_gate")
    )
    checks = parent_consistency(df, z_mechanizmem) if not df.empty else []
    out = RESULTS / f"{plan.name}.md"
    out.write_text(to_markdown(plan, agg, checks, figures, df), encoding="utf-8")
    if not agg.empty:
        agg.to_csv(RESULTS / f"{plan.name}.csv", index=False)
    return out
