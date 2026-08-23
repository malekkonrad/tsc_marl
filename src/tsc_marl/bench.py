#!/usr/bin/env python3
"""Runner tabeli T1 — siatka {agent} × {sieć} × {seed}, jeden punkt = jeden bieg.

Cienki z założenia: przechowywanie, wyszukiwanie i agregacja wyników to zadanie
`runlog` (`src/runlog/`). Tutaj zostaje to, co dziedzinowe — jakie punkty siatki
policzyć, jak odpalić proces i jak zrenderować tabele do pracy.

Każdy run leci w osobnym podprocesie (`python -m tsc_marl.run`), bo LibSignal trzyma
stan w globalnym Registry i nie da się go czysto zresetować między konfiguracjami.
Wynik czytamy z `result.json` biegu, a nie z regexa na stdout — log jest artefaktem,
stdout bywa uciętym buforem.

Użycie (z korzenia repo):
    python -m tsc_marl.bench                             # baseline'y bez treningu
    python -m tsc_marl.bench -a fixedtime maxpressure sotl
    python -m tsc_marl.bench -n cityflow_jinan3x4 -s 0 1 2
    python -m tsc_marl.bench -m benchmark                # tryb papierów (yellow=0)
    python -m tsc_marl.bench -a colight -s 0 --timeout 86400   # trening
    python -m tsc_marl.bench --table-only                # przelicz tabele z runs/
    python -m tsc_marl.bench -a colight --replace        # policz ponownie mimo wyniku
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime

from tsc_marl import experiment, libsignal
from tsc_marl.libsignal import REPO_ROOT, RESULTS

# Dwa tryby raportowania to jedna wartość yellow_length podawana flagą, a nie
# dwa drzewa configów jak w sandboxie (patrz tsc_marl/run.py, --yellow).
YELLOW_BY_MODE = {"realistic": 5, "benchmark": 0}


def run_one(agent, world, network, seed, yellow, timeout, replace=False):
    """Odpala jeden bieg w podprocesie. Zwraca (status, sekundy)."""
    cmd = [
        sys.executable,
        "-m",
        "tsc_marl.run",
        "-a",
        agent,
        "-w",
        world,
        "-n",
        network,
        "--seed",
        str(seed),
        "--yellow",
        str(yellow),
    ]
    if replace:
        cmd.append("--replace")
    t0 = time.time()
    try:
        p = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=libsignal.child_env(),
            timeout=timeout,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return "timeout", time.time() - t0
    if p.returncode != 0:
        tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-3:])
        print(f"    ! bieg nieudany:\n      {tail}", file=sys.stderr)
        return "failed", time.time() - t0
    return "ok", time.time() - t0


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--agents", nargs="+", default=["fixedtime", "maxpressure", "sotl"])
    ap.add_argument(
        "-n", "--networks", nargs="+", default=["cityflow_jinan3x4", "cityflow_hangzhou4x4"]
    )
    ap.add_argument("-w", "--world", default="cityflow")
    ap.add_argument("-s", "--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--timeout", type=int, default=3600, help="sekundy na jeden run")
    ap.add_argument("--table-only", action="store_true")
    ap.add_argument(
        "--replace",
        action="store_true",
        help="policz ponownie także te punkty siatki, które już mają wynik",
    )
    ap.add_argument(
        "-m",
        "--mode",
        choices=sorted(YELLOW_BY_MODE),
        default="realistic",
        help="realistic = yellow 5 (nasz protokół), benchmark = yellow 0 (jak papiery)",
    )
    args = ap.parse_args()

    libsignal.check_ready()
    libsignal.bootstrap()  # potrzebne do czytania configów przy planowaniu siatki
    exp = experiment.experiment()

    if args.table_only:
        build_table(exp)
        return

    yellow = YELLOW_BY_MODE[args.mode]
    done = skipped = 0
    for net in args.networks:
        for agent in args.agents:
            for seed in args.seeds:
                _, params, rid = experiment.plan(agent, args.world, net, seed, yellow)
                # skip-if-done po ODCISKU PARAMETRÓW, nie po samej nazwie: zmiana
                # czegokolwiek w configu metody unieważnia poprzedni wynik
                if not args.replace and exp.is_done(rid, params):
                    print(f"  {net:22s} {agent:14s} seed={seed}  pomijam (jest wynik)")
                    skipped += 1
                    continue
                print(f"  {net:22s} {agent:14s} seed={seed} ...", end=" ", flush=True)
                status, dt = run_one(
                    agent, args.world, net, seed, yellow, args.timeout, args.replace
                )
                print(f"{status} ({dt:.0f}s)")
                done += 1

    print(f"\nPoliczone: {done}, pominięte: {skipped}. Biegi: {exp.path}")

    # meta trzyma OBA tryby obok siebie — inaczej drugi przebieg kasuje protokół pierwszego
    meta_path = RESULTS / "t1_meta.json"
    RESULTS.mkdir(exist_ok=True)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.setdefault("modes", {})[args.mode] = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "yellow_length": yellow,
        "runs_root": str(exp.path),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    build_table(exp)


if __name__ == "__main__":
    main()
