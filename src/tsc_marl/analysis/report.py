"""Tabele i raporty do pracy — z zapisanych biegów, bez uruchamiania czegokolwiek.

Rozdzielone od `runner/suite.py` świadomie: ten moduł CZYTA `runlog` i pisze markdown do
`results/`, i nie ma żadnej drogi, którą mógłby odpalić symulację. Dopóki jedno
i drugie siedziało w `bench.py`, przeliczenie tabeli wymagało wejścia do modułu
umiejącego uruchomić trzydzieści biegów.

    tsc table          # przelicz tabele z runs/
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime

from tsc_marl.paths import RESULTS

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

# GATE Fazy 1 (`configs/suites/p1_gate.yml`): rdzeń ma dojść do ATT Advanced-CoLighta
# „w granicach kilku procent”, na tych samych sieciach i seedach, w trybie realistycznym.
# Próg to GÓRNA granica tego sformułowania — zapisana tutaj, żeby werdykt nie zależał
# od tego, kto czyta tabelę. Raport podaje też najmniejszy próg, przy którym werdykt się
# utrzymuje, bo wartość ustalono po obejrzeniu wyników.
GATE_REFERENCE = "advancedcolight"
GATE_CORE = "gatppo"
GATE_YELLOW = 5
GATE_TOLERANCE = 0.05
GATE_NETWORKS = ("cityflow_jinan3x4_4ph", "cityflow_hangzhou4x4_4ph")
# Okno raportowe jak `REPORT_WINDOW` w `core/loop.py` (import ciągnąłby torcha do tabel).
WINDOW = 10
# Stabilność: najgorszy test w ostatnich TAIL epizodach i pierwszy epizod, od którego
# kroczące WINDOW testów mieści się w CONVERGED od wyniku końcowego.
TAIL = 50
CONVERGED = 0.02
# Krzywe uczenia bramki — obok raportu, żeby markdown linkował je ścieżką względną.
FIGURES = RESULTS / "figures"

# WARIANT biegu = sufiks `run_id` po `__s<seed>_y<yellow>`. Siatki z `overrides` (ablacje,
# `p1_abl_*`) mają TEGO SAMEGO agenta, sieć, seed i yellow co rdzeń — różnią się tylko
# odciskiem configu i sufiksem. Bez tej kolumny ablacje wmieszałyby się do wiersza rdzenia
# w T1 i do strony `gatppo` w GATE, a werdykt bramki liczyłby się z 36 biegów zamiast 6.
RUN_SUFFIX = re.compile(r"__s\d+_y\d+(?:__(?P<variant>.+))?$")

# Ablacje rdzenia 1.6: wariant -> etykieta. "" = rdzeń z GATE (GAT, stan ATS).
AGGREGATION = {"": "GAT (rdzeń)", "p1_abl_meanpool": "mean-pool (GraphSAGE)",
               "p1_abl_nognn": "bez GNN (MLP)"}  # fmt: skip
STATE = {"": "ATS (rdzeń)", "p1_abl_state_ep": "EP", "p1_abl_state_pressure": "pressure",
         "p1_abl_state_queue": "queue"}  # fmt: skip


def variant_of(run_id: str) -> str:
    m = RUN_SUFFIX.search(run_id)
    return (m.group("variant") or "") if m else ""


def build_baseline_report(rows):
    """T1-baseline: nasze liczby obok literatury, osobno dla obu trybów raportowania.
    To jest punkt odniesienia całej pracy — GATE Fazy 0.

    Metody uczone mają kilka seedów: komórka to ŚREDNIA po seedach (wcześniej słownik
    trzymał ostatni wczytany bieg, czyli losowy seed). Biegi z wariantem (ablacje,
    sanity) nie wchodzą — to nie jest ta sama metoda."""
    acc = {}
    for r in rows:
        if r.get("variant"):
            continue
        v = r.get("att_last10")
        v = r.get("att") if v is None or v != v else v
        # yellow_length jest liczbą (parametry biegu mają typy, w odróżnieniu od CSV)
        acc.setdefault((r["network"], r["agent"], r.get("yellow_length")), []).append(float(v))
    idx = {k: {"att_last10": sum(v) / len(v), "att": sum(v) / len(v), "n": len(v)}
           for k, v in acc.items()}  # fmt: skip
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
                if not x:
                    return "—"
                n = f" (n={x['n']})" if x["n"] > 1 else ""
                return f"{float(x['att_last10']):.1f}{n}"

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

    Biegi z `tsc_marl.runner.evaluate` mają w parametrach `parent_run` i opisują scenariusz
    (np. awarię sensorów), a nie metodę. Wpuszczone do tabeli T1 udawałyby dodatkowe
    seedy tej samej metody i zaniżałyby albo zawyżały jej średnią."""
    df = exp.results(status="ok")
    if df.empty:
        return df
    if "parent_run" in df.columns:
        df = df[df["parent_run"].isna()]
    df = df.copy()
    df["variant"] = df["run_id"].map(variant_of)
    return df


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

    # `advancedmp` przy różnych W1 to różne metody, nie ten sam agent z rozrzutem;
    # wariant (sufiks run_id) to też inna metoda — patrz RUN_SUFFIX
    df["label"] = df.apply(
        lambda r: (
            (
                f"{r['agent']} (W1={r['w1']})"
                if r["agent"] == "advancedmp" and r.get("w1")
                else r["agent"]
            )
            + (f" [{r['variant']}]" if r["variant"] else "")
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
    build_gate_report(exp)
    build_ablation_report(exp)
    table = "\n".join(lines)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "t1_table.md").write_text(
        f"# T1 — baseline'y na czystym benchmarku\n\n"
        f"Wygenerowane: {datetime.now():%Y-%m-%d %H:%M}\n\n{table}\n"
    )
    print(table)


def _curve_stats(curves, run_id):
    """Najgorszy test w ogonie i epizod zbieżności — z krzywej testowego ATT jednego biegu."""
    att = (
        curves[(curves["run_id"] == run_id) & (curves["split"] == "test")]
        .loc[lambda c: c["metric"] == "att"]
        .sort_values("step")["value"]
        .reset_index(drop=True)
    )
    final = att.iloc[-WINDOW:].mean()
    rolling = att.rolling(WINDOW).mean()
    hit = rolling[rolling <= final * (1 + CONVERGED)]
    return {
        "tail_worst": att.iloc[-TAIL:].max(),
        "converged_at": float(hit.index[0]) if len(hit) else float("nan"),
    }


def _device(run_path, provenance):
    """Urządzenie biegu: z logu rdzenia („| device cuda:0”), a gdy go nie ma — z `--ngpu`
    w argv. Samo `--ngpu 0` nie wystarcza: bez CUDA bieg schodzi do CPU z ostrzeżeniem."""
    for log in sorted(run_path.glob("engine/**/*.log")):
        hit = re.search(r"\| device (\S+)", log.read_text(errors="ignore"))
        if hit:
            return hit.group(1)
    argv = provenance.get("argv", [])
    ngpu = argv[argv.index("--ngpu") + 1] if "--ngpu" in argv else "-1"
    return "cpu" if ngpu == "-1" else f"ngpu={ngpu} (bez potwierdzenia w logu)"


def _p(value):
    if value != value:
        return "—"
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def build_gate_report(exp):
    """GATE Fazy 1: rdzeń vs Advanced-CoLight -> `results/p1_gate.md`.

    Bez kompletu OBU stron na OBU sieciach nie pisze nic — pół bramki w `results/`
    wyglądałoby jak werdykt."""
    import pandas as pd
    from scipy.stats import ttest_ind

    df = training_results(exp)
    if df.empty:
        return
    df = df[
        df["agent"].isin([GATE_REFERENCE, GATE_CORE])
        & (df["yellow_length"] == GATE_YELLOW)
        & df["network"].isin(GATE_NETWORKS)
        & (df["variant"] == "")  # ablacje mają tego samego agenta — patrz RUN_SUFFIX
    ]
    missing = [
        f"{agent}@{net}"
        for net in GATE_NETWORKS
        for agent in (GATE_REFERENCE, GATE_CORE)
        if df[(df["network"] == net) & (df["agent"] == agent)].empty
    ]
    if missing:
        print(f"GATE Fazy 1: brak biegów {', '.join(missing)} — bez raportu.", file=sys.stderr)
        return

    curves = exp.metrics(run_ids=list(df["run_id"]))
    stats = pd.DataFrame([{"run_id": rid, **_curve_stats(curves, rid)} for rid in df["run_id"]])
    df = df.merge(stats, on="run_id")

    # Proweniencja z `meta.json`: urządzenie, wersja kodu, maszyna. Czasy biegów są
    # porównywalne tylko przy tym samym sprzęcie — raport ma to stwierdzić, nie zakładać.
    devices, code, hosts = {}, set(), set()
    wanted = set(df["run_id"])
    for run in exp.runs():
        if run.id not in wanted:
            continue
        prov = json.loads((run.path / "meta.json").read_text()).get("provenance") or {}
        devices[run.id] = _device(run.path, prov)
        code.add(
            tuple((name, g.get("sha"), g.get("dirty")) for name, g in sorted(prov["git"].items()))
            if prov.get("git")
            else ()
        )
        hosts.add(prov.get("hostname") or "?")
    df["device"] = df["run_id"].map(devices)

    labels = {GATE_REFERENCE: "Advanced-CoLight", GATE_CORE: "rdzeń (gatppo)"}
    att_rows, seed_rows, curve_rows, verdicts, ref_att = [], [], [], [], {}
    for net in GATE_NETWORKS:
        side = {
            agent: df[(df["network"] == net) & (df["agent"] == agent)].sort_values("seed")
            for agent in labels
        }
        ref = side[GATE_REFERENCE]["att_last10"].astype(float)
        core = side[GATE_CORE]["att_last10"].astype(float)
        ref_att[net] = ref.mean()
        delta = (core.mean() - ref.mean()) / ref.mean()
        pvalue = (
            ttest_ind(core, ref, equal_var=False).pvalue
            if min(len(ref), len(core)) > 1
            else float("nan")
        )
        ok = delta <= GATE_TOLERANCE
        verdicts.append((net, delta, pvalue, ok))
        att_rows.append(
            f"| {net} | {ref.mean():.1f} ± {ref.std():.1f} | {core.mean():.1f} ± {core.std():.1f} "
            f"| {delta * 100:+.2f}% | {_p(pvalue)} | {len(ref)} / {len(core)} "
            f"| {'TAK' if ok else 'NIE'} |"
        )
        for agent, label in labels.items():
            s = side[agent]
            per_seed = " · ".join(
                f"s{int(r['seed'])} {float(r['att_last10']):.1f}" for r in s.to_dict("records")
            )
            seed_rows.append(f"| {net} | {label} | {per_seed} |")
            conv = s["converged_at"].dropna()
            conv_txt = f"{int(conv.min())}–{int(conv.max())}" if len(conv) else "—"
            curve_rows.append(
                f"| {net} | {label} | {s['tail_worst'].max():.1f} | {conv_txt} "
                f"| {s['throughput'].astype(float).mean():.0f} "
                f"| {s['queue'].astype(float).mean():.2f} | {s['duration_s'].mean():.0f} "
                f"| {', '.join(sorted(set(s['device'])))} |"
            )

    # matplotlib dopiero tutaj — tabele T1 nie powinny go potrzebować
    from tsc_marl.analysis.plot import gate_figures

    figures = gate_figures(
        curves, df[["run_id", "network", "agent"]], labels, ref_att, GATE_TOLERANCE, FIGURES
    )

    device_of = {
        agent: ", ".join(sorted(set(df[df["agent"] == agent]["device"]))) for agent in labels
    }
    same_hw = device_of[GATE_REFERENCE] == device_of[GATE_CORE]
    if len(code) == 1 and next(iter(code)):
        code_line = "Kod jednakowy dla wszystkich biegów: " + ", ".join(
            f"{name} `{sha}`" + (" (DIRTY)" if dirty else "")
            for name, sha, dirty in next(iter(code))
        )
    else:
        code_line = f"**UWAGA: biegi z {len(code)} różnych wersji kodu** — porównanie niepewne"
    code_line += f"; maszyna: `{', '.join(sorted(hosts))}`."

    passed = all(ok for *_, ok in verdicts)
    worst = max(delta for _, delta, _, _ in verdicts)
    reading = []
    for net, delta, pvalue, _ in verdicts:
        if pvalue == pvalue and pvalue < 0.05:
            better = labels[GATE_REFERENCE] if delta > 0 else labels[GATE_CORE]
            reading.append(
                f"* **{net}** — różnica rozróżnialna statystycznie (Welch, p = {_p(pvalue)}) "
                f"na korzyść: {better}; praktycznie {abs(delta) * 100:.2f}% ATT."
            )
        else:
            reading.append(
                f"* **{net}** — metody nierozróżnialne (Welch, p = {_p(pvalue)}); "
                f"Δ {delta * 100:+.2f}% mieści się w rozrzucie po seedach."
            )

    out = [
        "# GATE Fazy 1 — rdzeń (GAT + PPO + CTDE) vs Advanced-CoLight",
        "",
        f"Wygenerowane: {datetime.now():%Y-%m-%d %H:%M}. Suity `p1_reference` + `p1_gate`, "
        f"tryb realistyczny (yellow={GATE_YELLOW}), protokół `configs/tsc/thesis_protocol.yml`.",
        f"ATT = średnia z {WINDOW} ostatnich epizodów testowych biegu; niżej średnia ± std "
        "po seedach.",
        code_line,
        "",
        "| sieć | Advanced-CoLight | rdzeń (gatppo) | Δ | p (Welch) | n | w progu |",
        "|---|---:|---:|---:|---:|:-:|:-:|",
        *att_rows,
        "",
        "## Po seedach",
        "",
        "| sieć | agent | ATT (10 ost. testów) |",
        "|---|---|---|",
        *seed_rows,
        "",
        "## Stabilność i koszt",
        "",
        f"| sieć | agent | najgorszy test (ost. {TAIL}) | zbieżność [epizod] "
        "| throughput | queue | czas biegu [s] | urządzenie |",
        "|---|---|---:|---:|---:|---:|---:|---|",
        *curve_rows,
        "",
        f"Zbieżność = pierwszy epizod, od którego kroczące {WINDOW} testów mieści się w "
        f"{CONVERGED * 100:.0f}% od wyniku końcowego (zakres po seedach). Najgorszy test w "
        "ogonie łapie jednorazowe załamania polityki, których średnia z 10 nie pokazuje.",
        "",
        "## Krzywe uczenia",
        "",
        f"Rysunki w `results/{FIGURES.name}/` (PNG tutaj, PDF do pracy), przeliczane razem "
        "z tym raportem.",
        "",
        *(
            line
            for name, caption in figures
            for line in (f"![{caption}]({FIGURES.name}/{name})", "", f"*{caption}*", "")
        ),
        "## Kryterium wyjścia",
        "",
        f"Rdzeń w granicach **{GATE_TOLERANCE * 100:.0f}%** ATT Advanced-CoLighta na każdej "
        "sieci (Δ > 0 = rdzeń gorszy). Próg to górna granica sformułowania z "
        "`p1_gate.yml` („w granicach kilku procent”), zapisana jako `GATE_TOLERANCE` w "
        "`analysis/report.py` PO obejrzeniu wyników — dlatego niżej jest też najmniejszy "
        "próg, przy którym werdykt się utrzymuje.",
        "",
        *reading,
        "",
        "## Werdykt",
        "",
        f"**GATE Fazy 1: {'ZALICZONA' if passed else 'NIEZALICZONA'}.** Największe Δ = "
        f"{worst * 100:+.2f}%"
        + (
            f"; werdykt utrzymuje się przy każdym progu ≥ {max(worst, 0) * 100:.2f}%."
            if passed
            else "."
        ),
        "",
        "## Skąd różnica czasu biegu",
        "",
        (
            "Symulacja jest w obu ścieżkach TA SAMA (epizod treningowy + testowy po 3600 s, "
            "`test_when_train` po obu stronach) i sprzęt też, więc różnica siedzi w uczeniu:"
            if same_hw
            else f"**Czasów NIE da się porównać wprost: {labels[GATE_REFERENCE]} liczył na "
            f"`{device_of[GATE_REFERENCE]}`, rdzeń na `{device_of[GATE_CORE]}`.** Symulacja "
            "jest w obu ścieżkach ta sama (epizod treningowy + testowy po 3600 s), ale różnica "
            "czasu miesza dwie przyczyny, których z tych biegów nie da się rozdzielić:"
        ),
        "",
        "* **Advanced-CoLight** — w `external/LibSignal/trainer/tsc_trainer.py` blok treningu "
        "(l. 172) stoi na poziomie `while`, POZA `if i % action_interval == 0` (l. 139). Przy "
        "`update_model_rate: 1` (`configs/tsc/base.yml`) warunek jest prawdziwy w każdym "
        "takcie symulatora: po `learning_start` to 3600 kroków optymalizatora na epizod, "
        "każdy z trzema przebiegami GAT w przód (target na b_tp, model dwa razy na b_t).",
        "* **rdzeń** — jeden update PPO na epizod: `epochs × n_minibatches` = 10 × 4 = 40 "
        "kroków optymalizatora (`configs/tsc/gatppo.yml`).",
        *(
            []
            if same_hw
            else [
                "* **sprzęt** — Advanced-CoLight nie może zejść na GPU (`p1_reference.yml`: "
                "submoduł nie ma `.cuda()`), rdzeń poszedł na GPU zgodnie z `ngpu` w "
                "`p1_gate.yml`. Ile z różnicy czasu to karta, a ile liczba kroków — nie wiadomo."
            ]
        ),
        "",
        "Wniosek do pracy: twierdzenie odporne na sprzęt to ~90× mniej kroków optymalizatora "
        "na epizod przy tym samym ATT. Czasu biegu NIE cytować jako przewagi metody"
        + ("" if same_hw else " bez biegu rdzenia na CPU (`--ngpu -1`)")
        + ". Część różnicy to też artefakt pętli LibSignala, której świadomie nie ruszamy, "
        "bo zmiana kadencji zmieniłaby baseline dopasowany do literatury (`t1_baseline.md`).",
        "",
        "## Czego ta tabela NIE mówi",
        "",
        "* Czas biegu to `duration_s` z `meta.json` — CAŁOŚĆ. Rozbicie na symulację i update "
        "nie jest logowane w żadnej ze ścieżek; powyższy podział wynika z kodu, nie z pomiaru.",
        "* HangZhou słabo różnicuje metody w trybie realistycznym: klasyczne baseline'y lądują "
        "blisko (`t1_baseline.md`, kolumna „realistyczny”), a odtwarzalność literatury na tej "
        "sieci jest słaba. Ciężar dowodu bramki niesie JiNan.",
        "",
    ]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "p1_gate.md").write_text("\n".join(out))
    print("\n".join(out))


# ------------------------------------------------------------------ ablacje 1.6


def _ablation_rows(df, variants: dict[str, str], net: str):
    """(etykieta, wartości att_last10 po seedach) dla wariantów obecnych w danych."""
    out = []
    for variant, label in variants.items():
        vals = df[(df["network"] == net) & (df["variant"] == variant)]["att_last10"]
        vals = vals.astype(float).dropna().tolist()
        if vals:
            out.append((label, variant, vals))
    return out


def build_ablation_report(exp):
    """Ablacje rdzenia 1.6 -> `results/p1_ablations.md`. Odniesienie = biegi GATE
    (wariant pusty), więc Δ mierzy dokładnie jedną zmianę względem rdzenia."""
    import numpy as np
    from scipy.stats import ttest_ind

    df = training_results(exp)
    if df.empty:
        return
    core = df[(df["agent"] == GATE_CORE) & (df["yellow_length"] == GATE_YELLOW)]
    known = set(AGGREGATION) | set(STATE)
    if not (core["variant"].isin(known - {""})).any():
        return
    refs = df[(df["yellow_length"] == GATE_YELLOW) & (df["variant"] == "")
              & df["agent"].isin(["advancedcolight", "efficientmp", "advancedmp"])]  # fmt: skip

    def fmt(vals):
        sd = f" ± {np.std(vals, ddof=1):.1f}" if len(vals) > 1 else ""
        return f"{np.mean(vals):.1f}{sd}"

    out = [
        "# Ablacje rdzenia (1.6) — GAT+PPO+CTDE",
        "",
        f"Wygenerowane: {datetime.now():%Y-%m-%d %H:%M}. Tryb realistyczny (yellow=5), "
        "ATT = średnia z 10 ostatnich testów, mean ± std po seedach. Każdy wariant zmienia "
        "JEDNĄ rzecz względem rdzenia z GATE; krytyk zostaje GAT-em we wszystkich "
        "(ablacja pyta o decyzję actora). Δ > 0 = gorzej niż rdzeń.",
        "",
    ]
    for net in GATE_NETWORKS:
        sub = core[core["network"] == net]
        base = sub[sub["variant"] == ""]["att_last10"].astype(float).dropna().tolist()
        if not base:
            continue
        out += [f"## {net}", ""]
        for title, variants in (("Agregacja sąsiadów w actorze", AGGREGATION),
                                ("Reprezentacja stanu", STATE)):  # fmt: skip
            rows = _ablation_rows(sub, variants, net)
            out += [f"### {title}", "", "| wariant | ATT | Δ vs rdzeń | p (Welch) | n |",
                    "|---|---:|---:|---:|:-:|"]  # fmt: skip
            for label, variant, vals in rows:
                if variant == "":
                    out.append(f"| **{label}** | {fmt(vals)} | — | — | {len(vals)} |")
                    continue
                d = np.mean(vals) / np.mean(base) - 1
                p = ttest_ind(vals, base, equal_var=False).pvalue if len(vals) > 1 else float("nan")
                out.append(f"| {label} | {fmt(vals)} | {d:+.2%} | {_p(p)} | {len(vals)} |")
            out.append("")
        ref_rows = []
        for a in ("advancedcolight", "efficientmp", "advancedmp"):
            vals = refs[(refs["network"] == net) & (refs["agent"] == a)]["att_last10"]
            if not vals.empty:
                ref_rows.append(f"{a}: {fmt(vals.astype(float).tolist())}")
        if ref_rows:
            out += ["Odniesienie (ten sam protokół): " + " · ".join(ref_rows) + ".", ""]
    out += [
        "## Jak czytać",
        "",
        "* p z testu Welcha przy n=3 to orientacja, nie dowód — ta sama konwencja co w GATE.",
        "* Efficient-MP i Advanced-MP obok: odpowiedź na „po co GNN, skoro MaxPressure "
        "wystarcza” — wariant bez GNN powinien lądować między nimi a rdzeniem.",
    ]
    (RESULTS / "p1_ablations.md").write_text("\n".join(out) + "\n")
