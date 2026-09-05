"""Tabele i raporty do pracy — z zapisanych biegów, bez uruchamiania czegokolwiek.

Rozdzielone od `suite.py` świadomie: ten moduł CZYTA `runlog` i pisze markdown do
`results/`, i nie ma żadnej drogi, którą mógłby odpalić symulację. Dopóki jedno
i drugie siedziało w `bench.py`, przeliczenie tabeli wymagało wejścia do modułu
umiejącego uruchomić trzydzieści biegów.

    tsc table          # przelicz tabele z runs/
"""

from __future__ import annotations

import sys
from datetime import datetime

from tsc_marl.libsignal import RESULTS

# Liczby z literatury dla TEGO protokołu (4 fazy, Δt=15): Zhang 2022 (Advanced-XLight),
# Tabela 2, kolumny JiNan-1 (`anon_3_4_jinan_real.json`) i HangZhou-1.
# advancedmp/efficientmp: te same definicje, które implementujemy w advanced_mp.py.
LITERATURE = {
    "cityflow_jinan3x4_4ph": {
        "fixedtime": 428.11,
        "maxpressure": 273.96,
        "efficientmp": 269.87,
        "advancedmp": 253.61,
        "colight": 272.06,
        "mplight": 297.46,
        "advancedcolight": 245.73,
    },
    "cityflow_hangzhou4x4_4ph": {
        "fixedtime": 495.57,
        "maxpressure": 288.54,
        "efficientmp": 284.44,
        "advancedmp": 279.47,
        "colight": 297.02,
        "mplight": 314.60,
        "advancedcolight": 270.45,
    },
}


def build_baseline_report(rows):
    """T1-baseline: nasze liczby obok literatury, osobno dla obu trybów raportowania.
    To jest punkt odniesienia całej pracy — GATE Fazy 0."""
    idx = {}
    for r in rows:
        # yellow_length jest liczbą (parametry biegu mają typy, w odróżnieniu od CSV)
        idx[(r["network"], r["agent"], r.get("yellow_length"))] = r
    out = [
        "# T1-baseline — punkt odniesienia pracy",
        "",
        f"Wygenerowane: {datetime.now():%Y-%m-%d %H:%M}. "
        "Protokół: `configs/tsc/thesis_protocol.yml` "
        "(Δt=15 s, epizod 3600 s, `PYTHONHASHSEED=0`), sieci w wariancie 4-fazowym "
        "— jak protokół papierów.",
        "",
        "ATT = średnia z 10 ostatnich epizodów testowych (metody bez treningu: jeden przebieg).",
        "Δ liczone względem literatury w trybie **benchmark**, bo w nim liczą swoje "
        "wyniki papiery.",
        "",
    ]
    for net, lit in LITERATURE.items():
        out += [
            f"## {net}",
            "",
            "| agent | benchmark (yellow=0) | realistyczny (yellow=5) | literatura | Δ benchmark |",
            "|---|---|---|---|---|",
        ]
        for agent in (
            "fixedtime",
            "maxpressure",
            "sotl",
            "efficientmp",
            "advancedmp",
            "presslight",
            "colight",
            "advancedcolight",
        ):
            b = idx.get((net, agent, 0))
            r5 = idx.get((net, agent, 5))
            if not b and not r5:
                continue

            def val(x):
                return f"{float(x['att_last10'] or x['att']):.1f}" if x else "—"

            ref = lit.get(agent)
            d = "—"
            if b and ref:
                got = float(b["att_last10"] or b["att"])
                d = f"{(got - ref) / ref * 100:+.1f}%"
            out.append(f"| {agent} | {val(b)} | {val(r5)} | {ref if ref else '—'} | {d} |")
        out.append("")
    out += [
        "## Jak czytać Δ",
        "",
        "- **`maxpressure` to implementacja z LibSignal**, licząca pressure z `lane_count`",
        "  (wszystkie pojazdy, bez normalizacji per pas), a nie wg definicji z papieru —",
        "  jej odchyłka NIE jest kryterium bramki. Rolę poprawnego MP pełni `efficientmp`.",
        "- **Odtwarzalność zależy od sieci**: JiNan trafia w literaturę co do ~1%",
        "  (FixedTime −0,8%, Efficient-MP +0,9%), HangZhou nie (Efficient-MP +10,4%).",
        "  Ten sam protokół, ten sam kod, dane zweryfikowane jako identyczne z plikami",
        "  autorów (`hangzhou_4x4_advanced/` vs `hangzhou_4x4_gudang_18041610_1h/`:",
        "  wyniki zgodne co do 0,01 s). Do opisania w pracy jako ograniczenie porównań.",
        "- **`advancedmp` raportowany jest przy globalnym W1 = 1,0.** W trybie benchmark",
        "  człon demand nie działa (bez kosztu przejścia utrzymanie fazy nic nie daje),",
        "  więc metoda degeneruje się do Efficient-MP i +8,2% względem literatury jest",
        "  OCZEKIWANE. Liczby z papieru dla Advanced-MP nie da się odtworzyć w trybie",
        "  benchmark przy żadnym W1 — to najmocniejszy argument, że ten tryb gubi",
        "  mechanizm metody, którą opisuje papier.",
        "- Kolumna `realistyczny` to nasz protokół (5 s przejścia fazowego). Różnica",
        "  między trybami jest samodzielnym wynikiem — patrz `t1_w1_sweep.md`.",
        "",
    ]
    (RESULTS / "t1_baseline.md").write_text("\n".join(out))
    print("\n".join(out))


def training_results(exp):
    """Udane biegi TRENINGOWE/pomiarowe — bez ewaluacji zamrożonych polityk.

    Biegi z `tsc_marl.evaluate` mają w parametrach `parent_run` i opisują scenariusz
    (np. awarię sensorów), a nie metodę. Wpuszczone do tabeli T1 udawałyby dodatkowe
    seedy tej samej metody i zaniżałyby albo zawyżały jej średnią."""
    df = exp.results(status="ok")
    if df.empty or "parent_run" not in df.columns:
        return df
    return df[df["parent_run"].isna()]


def _records(exp):
    """Wiersze do tabel jako słowniki (params + result)."""
    df = training_results(exp)
    return [] if df.empty else df.to_dict("records")


def build_table(exp):
    """Tabela mean±std po seedach. std=0 przy metodach bez treningu jest OCZEKIWANE:
    symulacja jest deterministyczna, seed rusza tylko inicjalizację sieci/eksplorację."""
    df = training_results(exp)
    if df.empty:
        print("Brak policzonych biegów w runs/ — najpierw odpal runy.", file=sys.stderr)
        return

    # `advancedmp` przy różnych W1 to różne metody, nie ten sam agent z rozrzutem
    df["label"] = df.apply(
        lambda r: (
            f"{r['agent']} (W1={r['w1']})"
            if r["agent"] == "advancedmp" and r.get("w1")
            else r["agent"]
        ),
        axis=1,
    )
    # tryb (yellow_length) MUSI być w kluczu — inaczej tabela uśrednia realistyczny
    # i benchmarkowy razem, co daje liczby bez sensu
    agg = exp.aggregate(
        by=["network", "label", "yellow_length"],
        over="seed",
        metrics=["att", "att_last10", "throughput", "queue"],
        df=df,  # kolumna `label` jest pochodna, nie ma jej w zapisanych parametrach
    ).sort_values(["network", "label", "yellow_length"])

    lines = [
        "| sieć | agent | yellow | ATT (10 ost. testów) | ATT (ostatni) | throughput | queue | n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in agg.to_dict("records"):
        sd = f" ± {r['att_std']:.1f}" if r["n"] > 1 and r["att_std"] == r["att_std"] else ""
        main = (
            f"{r['att_last10_mean']:.1f}" if r["att_last10_mean"] == r["att_last10_mean"] else "—"
        )
        lines.append(
            f"| {r['network']} | {r['label']} | {r['yellow_length']} | {main} | "
            f"{r['att_mean']:.1f}{sd} | {r['throughput_mean']:.0f} | {r['queue_mean']:.1f} | "
            f"{int(r['n'])} |"
        )
    build_baseline_report(_records(exp))
    table = "\n".join(lines)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "t1_table.md").write_text(
        f"# T1 — baseline'y na czystym benchmarku\n\n"
        f"Wygenerowane: {datetime.now():%Y-%m-%d %H:%M}\n\n{table}\n"
    )
    print(table)
