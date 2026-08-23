#!/usr/bin/env python3
"""
Runner tabeli T1 — jeden config → jeden wynik, z zapisem na dysk.

Uruchamia siatkę {agent} × {sieć} × {seed}, parsuje metryki z logu LibSignal
i zapisuje:
  results/t1_raw.csv       — jeden wiersz na run (surowe liczby + metadane)
  results/t1_table.md      — tabela mean±std, gotowa do pracy
  results/t1_meta.json     — protokół, wersje, data (żeby wynik dało się odtworzyć)

Wyniki są DOPISYWANE do t1_raw.csv, więc kolejne wywołania rozszerzają tabelę
(np. baseline'y dziś, CoLight jutro po treningu).

Każdy run leci w osobnym podprocesie (`python -m tsc_marl.run`), bo LibSignal trzyma
stan w globalnym Registry i nie da się go czysto zresetować między konfiguracjami.

Użycie (z korzenia repo):
    python -m tsc_marl.bench                             # baseline'y bez treningu
    python -m tsc_marl.bench -a fixedtime maxpressure sotl
    python -m tsc_marl.bench -n cityflow_jinan3x4 -s 0 1 2
    python -m tsc_marl.bench -m benchmark                # tryb papierów (yellow=0)
    python -m tsc_marl.bench -a colight -s 0 --timeout 86400   # trening
    python -m tsc_marl.bench --table-only                # przelicz tabelę z CSV
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from statistics import mean, pstdev

from tsc_marl import libsignal
from tsc_marl.libsignal import LIBSIGNAL, REPO_ROOT, RESULTS, RUNS

# Dwa tryby raportowania to teraz jedna wartość yellow_length podawana flagą, a nie
# dwa drzewa configów jak w sandboxie (patrz tsc_marl/run.py, --yellow).
YELLOW_BY_MODE = {"realistic": 5, "benchmark": 0}
MODE = "realistic"

# Parsowana linia LibSignala, np.:
#   "Final Travel Time is 514.1358, mean rewards: -723.7992, queue: 34.9205,
#    delay: 3.5311, throughput: 5768"
FINAL_RE = re.compile(
    r"Final Travel Time is ([\d.]+), mean rewards: (-?[\d.]+), queue: ([\d.]+), "
    r"delay: ([\d.]+), throughput: (\d+)"
)
FIELDS = [
    "timestamp",
    "agent",
    "world",
    "network",
    "seed",
    "att",
    "att_last10",
    "att_best",
    "reward",
    "queue",
    "delay",
    "throughput",
    "action_interval",
    "yellow_length",
    "episodes",
    "runtime_s",
    "status",
    # parametry agenta — bez nich wiersz NIE jest odtwarzalny (Advanced-MP zmienia
    # wynik o 10% w zależności od w1/obs_length/drop_always_green)
    "t_min",
    "w1",
    "obs_length",
    "drop_always_green",
]

# parametry z sekcji `model`, które trafiają do CSV (reszta to hiperparametry sieci)
MODEL_FIELDS = ["t_min", "w1", "obs_length", "drop_always_green"]


def git_sha(repo):
    """Krótki SHA + znacznik `-dirty`, jeśli w drzewie są niezacommitowane zmiany."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def att_from_training_log(agent, world, network):
    """Protokół papieru (Advanced-CoLight 5.1): raport = ŚREDNIA Z OSTATNICH 10
    epizodów testowych, nie pojedynczy test po treningu. Przy niestabilnym uczeniu
    (CoLight) różnica jest ogromna: 1204,5 (ostatni) vs 766,1 (średnia z 10)
    vs 314,6 (najlepszy). Zwraca (mean_last10, best) albo (None, None)."""
    # _DTL = metody uczone (wiersze TRAIN/TEST per epizod); metody bez treningu
    # zostawiają tylko _BRF z jedną liczbą — dla nich last10 == att i liczy je caller
    logdir = RUNS / "tsc" / f"{world}_{agent}" / network / "test/logger"
    logs = (
        sorted(logdir.glob("*_DTL.log"), key=lambda p: p.stat().st_mtime) if logdir.is_dir() else []
    )
    if not logs:
        return None, None
    vals = []
    for line in logs[-1].read_text().splitlines():
        parts = line.split("\t")
        if len(parts) > 3 and parts[1] == "TEST":
            try:
                vals.append(float(parts[3]))
            except ValueError:
                pass
    if not vals:
        return None, None
    last10 = vals[-10:]
    return sum(last10) / len(last10), min(vals)


def protocol_params(agent):
    """Wyciąga zamrożony protokół + parametry agenta z połączonych configów —
    to samo, co zobaczy run.py."""
    sys.path.insert(0, str(LIBSIGNAL))
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        from utils.logger import load_config

        cfg, _ = load_config(f"./configs/tsc/{agent}.yml")
        t = cfg["trainer"]
        out = {
            k: t[k] for k in ("action_interval", "yellow_length", "steps", "test_steps", "episodes")
        }
        # tryb nadpisuje protokół tą samą drogą, co run.py — CSV ma zapisać to,
        # co REALNIE poszło do symulatora, nie to, co stoi w pliku
        out["yellow_length"] = YELLOW_BY_MODE[MODE]
        out.update({k: cfg["model"].get(k, "") for k in MODEL_FIELDS})
        return out
    finally:
        os.chdir(cwd)


def run_one(agent, world, network, seed, timeout):
    env = libsignal.child_env()
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
        str(YELLOW_BY_MODE[MODE]),
    ]
    t0 = time.time()
    try:
        p = subprocess.run(
            cmd, cwd=REPO_ROOT, env=env, timeout=timeout, capture_output=True, text=True
        )
        out = p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return None, time.time() - t0, "timeout"
    hits = FINAL_RE.findall(out)
    if not hits:
        tail = "\n".join(out.strip().splitlines()[-3:])
        print(f"    ! brak wyniku:\n      {tail}", file=sys.stderr)
        return None, time.time() - t0, "no-result"
    att, rew, q, d, thr = hits[-1]
    return (
        dict(
            att=float(att), reward=float(rew), queue=float(q), delay=float(d), throughput=int(thr)
        ),
        time.time() - t0,
        "ok",
    )


def append_rows(rows, tag=""):
    RESULTS.mkdir(exist_ok=True)
    # osobny CSV na równoległe przebiegi (treningi lecą po 4 naraz) — build_table
    # i tak scala wszystkie t1_raw*.csv, a nie ma wyścigu o dopisanie do jednego pliku
    path = RESULTS / (f"t1_raw_{tag}.csv" if tag else "t1_raw.csv")
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
    return path


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
    },
    "cityflow_hangzhou4x4_4ph": {
        "fixedtime": 495.57,
        "maxpressure": 288.54,
        "efficientmp": 284.44,
        "advancedmp": 279.47,
        "colight": 297.02,
        "mplight": 314.60,
    },
}


def build_baseline_report(rows):
    """T1-baseline: nasze liczby obok literatury, osobno dla obu trybów raportowania.
    To jest punkt odniesienia całej pracy — GATE Fazy 0."""
    idx = {}
    for r in rows:
        idx[(r["network"], r["agent"], r.get("yellow_length", "?"))] = r
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
        ):
            b = idx.get((net, agent, "0"))
            r5 = idx.get((net, agent, "5"))
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


def build_table():
    """Tabela mean±std po seedach. std=0 przy metodach bez treningu jest OCZEKIWANE:
    symulacja jest deterministyczna, seed rusza tylko inicjalizację sieci/eksplorację."""
    paths = sorted(RESULTS.glob("t1_raw*.csv"))
    if not paths:
        print("Brak results/t1_raw*.csv — najpierw odpal runy.", file=sys.stderr)
        return
    rows = [r for p in paths for r in csv.DictReader(open(p)) if r["status"] == "ok"]
    groups = {}
    for r in rows:
        # tryb (yellow_length) MUSI być częścią klucza — inaczej tabela uśrednia
        # realistyczny i benchmarkowy razem, co daje liczby bez sensu
        agent = r["agent"]
        if r.get("w1") and agent == "advancedmp":
            agent = f"{agent} (W1={r['w1']})"
        groups.setdefault((r["network"], agent, r.get("yellow_length", "?")), []).append(r)

    lines = [
        "| sieć | agent | yellow | ATT (10 ost. testów) | ATT (ostatni) | throughput | queue | n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for net, agent, yl in sorted(groups):
        g = groups[(net, agent, yl)]
        att = [float(r["att"]) for r in g]
        thr = [float(r["throughput"]) for r in g]
        q = [float(r["queue"]) for r in g]
        l10 = [float(r["att_last10"]) for r in g if r.get("att_last10")]
        sd = f" ± {pstdev(att):.1f}" if len(att) > 1 else ""
        # ATT wg protokołu papieru; dla metod bez treningu (1 epizod) obie kolumny są równe
        main = f"{mean(l10):.1f}" if l10 else "—"
        lines.append(
            f"| {net} | {agent} | {yl} | {main} | {mean(att):.1f}{sd} | {mean(thr):.0f} | "
            f"{mean(q):.1f} | {len(att)} |"
        )
    build_baseline_report(rows)
    table = "\n".join(lines)
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
    ap.add_argument("--tag", default="", help="sufiks pliku CSV — do przebiegów równoległych")
    ap.add_argument(
        "-m",
        "--mode",
        choices=sorted(YELLOW_BY_MODE),
        default="realistic",
        help="realistic = yellow 5 (nasz protokół), benchmark = yellow 0 (jak papiery)",
    )
    args = ap.parse_args()

    global MODE
    MODE = args.mode
    libsignal.check_ready()

    if args.table_only:
        build_table()
        return

    rows = []
    for net in args.networks:
        for agent in args.agents:
            proto = protocol_params(agent)
            for seed in args.seeds:
                print(f"  {net:22s} {agent:12s} seed={seed} ...", end=" ", flush=True)
                res, dt, status = run_one(agent, args.world, net, seed, args.timeout)
                print(
                    f"{status} ATT={res['att']:.2f} ({dt:.0f}s)" if res else f"{status} ({dt:.0f}s)"
                )
                last10, best = (
                    att_from_training_log(agent, args.world, net) if res else (None, None)
                )
                if res and last10 is None:  # metoda bez treningu — jeden przebieg
                    last10 = best = res["att"]
                rows.append(
                    dict(
                        timestamp=datetime.now().isoformat(timespec="seconds"),
                        agent=agent,
                        world=args.world,
                        network=net,
                        seed=seed,
                        att=res["att"] if res else "",
                        att_last10=round(last10, 2) if last10 is not None else "",
                        att_best=round(best, 2) if best is not None else "",
                        reward=res["reward"] if res else "",
                        queue=res["queue"] if res else "",
                        delay=res["delay"] if res else "",
                        throughput=res["throughput"] if res else "",
                        action_interval=proto["action_interval"],
                        yellow_length=proto["yellow_length"],
                        episodes=proto["episodes"],
                        runtime_s=round(dt, 1),
                        status=status,
                        **{k: proto[k] for k in MODEL_FIELDS},
                    )
                )

    print(f"\nZapisano: {append_rows(rows, args.tag)}")
    # meta trzyma OBA tryby obok siebie — inaczej drugi przebieg kasuje protokół pierwszego
    meta_path = RESULTS / "t1_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.setdefault("modes", {})[args.mode] = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "protocol": protocol_params(args.agents[0]),
    }
    # Bez wersji kodu wiersz w CSV nie jest odtwarzalny: te same argumenty na innym
    # commicie potrafią dać inną liczbę (patrz fix update_target_rate).
    meta["code_version"] = {"tsc_marl": git_sha(REPO_ROOT), "LibSignal": git_sha(LIBSIGNAL)}
    meta["pythonhashseed"] = "0 (wymuszone — bez tego world_sumo niedeterministyczny)"
    meta["note"] = (
        "Δt z configs/tsc/thesis_protocol.yml; yellow_length nadpisywany "
        "trybem (--mode) przez tsc_marl.run --yellow"
    )
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    build_table()


if __name__ == "__main__":
    main()
