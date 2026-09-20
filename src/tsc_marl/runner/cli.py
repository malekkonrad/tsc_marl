"""Jedno wejście do projektu: `tsc <komenda>`.

Warstwa cienka z założenia — komendy tylko rozdzielają ruch, cała logika siedzi
w `train.py`, `train_core.py`, `evaluate.py`, `suite.py` i `analysis/report.py`. Dzięki temu
testy, `suite` i podprocesy wołają te same funkcje, a nie CLI.

`train`, `core` i `eval` przepuszczają argumenty BEZ ZMIAN do argparse'owych
parserów tamtych modułów (`--help` też — obsłuży go argparse). Powtórzenie
piętnastu flag w typerze byłoby dokładnie tym rozjazdem, który dopiero co
usunęliśmy z sekwencji interfejsów.
"""

from __future__ import annotations

import os
import sys

import typer

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    # Bez tego rich opakowuje KAŻDY wyjątek w ramkę z kodem źródłowym. Nasze błędy
    # dziedzinowe (RunExists, RunIdCollision, SystemExit z suite'a) są komunikatami
    # dla człowieka, a nie awarią — ramka z traceback tylko je zakrywa.
    pretty_exceptions_enable=False,
    help="Sterowanie sygnalizacją świetlną — treningi, siatki eksperymentów, raporty.",
)
suite_app = typer.Typer(no_args_is_help=True, help="Deklaratywne siatki biegów (configs/suites).")
app.add_typer(suite_app, name="suite")

# Argumenty lecą do argparse'a docelowego modułu — łącznie z `--help`, dlatego
# `help_option_names` jest puste (inaczej click przechwyciłby je u siebie).
PASSTHROUGH = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
    "help_option_names": [],
}


def ensure_hashseed() -> None:
    """Zrestartuj interpreter z PYTHONHASHSEED=0, jeśli go nie ma.

    PYTHONHASHSEED działa TYLKO ustawiony przed startem interpretera — z Pythona
    nie da się go „ustawić wstecz", a bez niego `world_sumo` jest
    niedeterministyczny (te same argumenty, inny wynik). To jest jedyny powód,
    dla którego istniały `scripts/*.sh`; tutaj załatwia to jedno `execv`.

    `sys.orig_argv` odtwarza wywołanie dokładnie, więc działa tak samo dla
    `tsc ...` i dla `python -m tsc_marl ...`. Strażnik na wejściu wyklucza pętlę,
    a podprocesy `suite` dostają zmienną z `libsignal.child_env()` — dla nich to
    no-op.
    """
    if os.environ.get("PYTHONHASHSEED") == "0" or not getattr(sys, "orig_argv", None):
        return
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, sys.orig_argv)


@app.callback()
def _przed_komenda() -> None:
    ensure_hashseed()


# ------------------------------------------------------------------ pojedyncze biegi


@app.command(context_settings=PASSTHROUGH)
def train(ctx: typer.Context) -> None:
    """Trening/pomiar agenta LibSignala (`tsc train -a colight -n ... --seed 0`)."""
    from tsc_marl.runner import train as train_module

    train_module.main(ctx.args)


@app.command()
def prototype(
    network: str = typer.Option("cityflow_jinan3x4_4ph", "-n", help="sieć bazowa zrzutu"),
    mode: str = typer.Option("edl", help="'edl' albo 'softmax'"),
    epochs: int = typer.Option(150, help="epoki uczenia głowy"),
    hop: bool = typer.Option(False, help="1-hop zamiast 0-hop (druga iteracja 2.0)"),
    ensemble: int = typer.Option(5, help="członkowie deep ensemble (wzorzec epistemic); 0 = bez"),
) -> None:
    """Bramka prototypu 2.0 (uczenie + tabela + kryteria) -> results/gate_2_0.md."""
    from tsc_marl.uncertainty import prototype as module

    cfg = module.TrainConfig(mode=mode, epochs=epochs, hop=hop, ensemble=ensemble)
    print(module.run(cfg, network=network))


@app.command()
def obshead(
    network: str = typer.Option("cityflow_jinan3x4_4ph", "-n", help="sieć bazowa zrzutu"),
    window: int = typer.Option(4, help="ile ostatnich odczytów widzi głowa"),
    neighbors: bool = typer.Option(True, help="dołóż średnią odczytów sąsiadów upstream"),
    epochs: int = typer.Option(60, help="epoki uczenia głowy"),
    persistence: bool = typer.Option(False, help="dołóż cechy trwałości odczytu (utknięcie)"),
    mask_variants: bool = typer.Option(
        False, help="porównaj warianty na martwych detektorach -> results/gate_2_2_masks.md"
    ),
    save: bool = typer.Option(
        False, help="ucz SAMĄ głowę (bez bramki) i zapisz artefakt -> runs/heads/"
    ),
) -> None:
    """Bramka prototypu 2.2 — głowa obserwacyjna (per ruch → krawędź) -> results/gate_2_2.md."""
    from tsc_marl.uncertainty import obs_head as module

    if mask_variants:
        print(module.mask_variants(network=network))
        return
    cfg = module.ObsConfig(
        window=window, neighbors=neighbors, epochs=epochs, persistence=persistence
    )
    if save:
        for path in module.train_head(network, cfg):
            print(path)
        return
    print(module.run(cfg, network=network))


@app.command()
def demand(
    network: str = typer.Option("cityflow_jinan3x4_4ph", "-n", help="sieć bazowa"),
    factors: str = typer.Option("0.5,1.5", help="mnożniki natężenia, po przecinku"),
) -> None:
    """Warianty popytu (scenariusz OOD dla epistemic) -> configs/sim/<sieć>_d*.cfg."""
    from tsc_marl.domain import demand as module

    for f in [float(x) for x in factors.split(",")]:
        flow, cfg = module.write_variant(network, f)
        print(f"x{f:g}: {cfg}  ({flow.name})")


@app.command()
def dataset(
    network: str = typer.Option("cityflow_jinan3x4_4ph", "-n", help="sieć"),
    expert: str = typer.Option("advancedmp", "-a", help="ekspert do behavior cloningu"),
    episodes: int = typer.Option(3, help="liczba epizodów (różnią się LOSOWANIEM AWARII)"),
    replace: bool = typer.Option(False, help="nadpisz istniejący bieg"),
) -> None:
    """Zbiór (stan -> akcja eksperta) dla prototypu 2.0 -> runs/bc/."""
    from tsc_marl.uncertainty import dataset as module

    print(module.build(network, expert=expert, episodes=episodes, replace=replace))


@app.command()
def severity(
    network: str = typer.Option("cityflow_jinan3x4_4ph", help="sieć, na której mierzymy"),
) -> None:
    """Przelicznik dotkliwości awarii sensora -> results/sensor_severity.md."""
    from tsc_marl.analysis import severity as module

    print(module.main(network))


@app.command(context_settings=PASSTHROUGH)
def core(ctx: typer.Context) -> None:
    """Rdzeń Fazy 1 — GAT + PPO + CTDE, własna pętla (`tsc core -n ... --seed 0`)."""
    from tsc_marl.runner import train_core

    train_core.main(ctx.args)


@app.command("eval", context_settings=PASSTHROUGH)
def evaluate_(ctx: typer.Context) -> None:
    """Ewaluacja zamrożonej polityki z checkpointu (`tsc eval --load-run ...`)."""
    from tsc_marl.runner import evaluate

    evaluate.main(ctx.args)


# ------------------------------------------------------------------------- siatki


@suite_app.command("run")
def suite_run(
    nazwa: str = typer.Argument(..., help="nazwa albo ścieżka pliku z configs/suites/"),
    replace: bool = typer.Option(False, "--replace", help="policz ponownie także gotowe punkty"),
    jobs: int = typer.Option(1, "--jobs", "-j", help="ile punktów liczyć naraz (podprocesy)"),
    stagger: float = typer.Option(15.0, help="sekundy między startami równoległych punktów"),
) -> None:
    """Policz wszystkie punkty siatki (pomija te, które mają aktualny wynik)."""
    from tsc_marl.runner import suite

    licznik = suite.run(suite.load(nazwa), replace=replace, jobs=jobs, stagger=stagger)
    if licznik.get("failed") or licznik.get("timeout"):
        raise typer.Exit(1)


@suite_app.command("status")
def suite_status(
    nazwa: str = typer.Argument(..., help="nazwa albo ścieżka pliku z configs/suites/"),
) -> None:
    """Stan każdego punktu siatki — także tych jeszcze niepoliczonych."""
    from tsc_marl.runner import suite

    wiersze = suite.status(suite.load(nazwa))
    print(f"{'run_id':52s} {'status':9s} {'aktualny':9s} att_last10")
    for w in wiersze:
        att = f"{w['att_last10']:.1f}" if w["att_last10"] is not None else "—"
        print(f"{w['run_id']:52s} {w['status']:9s} {str(w['aktualny']):9s} {att}")
    brakuje = sum(1 for w in wiersze if not w["aktualny"])
    print(f"\n{len(wiersze) - brakuje}/{len(wiersze)} punktów ma aktualny wynik.")


@suite_app.command("list")
def suite_list() -> None:
    """Wypisz dostępne suite'y."""
    from tsc_marl.runner import suite

    for path in sorted(suite.SUITES.glob("*.yml")):
        s = suite.load(path)
        print(f"  {s.name:16s} {s.module:20s} yellow={s.yellow}  {len(s.points())} punktów")


# ------------------------------------------------------------------- awarie (T2)

degrade_app = typer.Typer(
    no_args_is_help=True, help="Zamrożone sterowniki pod awarią sensorów (configs/robustness)."
)
app.add_typer(degrade_app, name="degrade")


@degrade_app.command("run")
def degrade_run(
    plan: str = typer.Argument(..., help="nazwa albo ścieżka pliku z configs/robustness/"),
    jobs: int = typer.Option(1, "--jobs", "-j", help="ile punktów liczyć naraz"),
    replace: bool = typer.Option(False, "--replace", help="policz ponownie także gotowe"),
    stagger: float = typer.Option(15.0, help="sekundy między startami podprocesów"),
    trace: bool = typer.Option(
        False,
        "--trace",
        help="zapisz ślad bramki per (decyzja, węzeł) do gate_trace.csv — do wyjaśniania "
        "pojedynczych przełączeń; NIE zmienia decyzji ani odcisku biegu",
    ),
) -> None:
    """Policz brakujące punkty planu (sterownik × sieć × seed, wszystkie scenariusze)."""
    from tsc_marl.robustness.degrade import load_plan
    from tsc_marl.runner import degrade

    licznik = degrade.run_plan(
        load_plan(plan), jobs=jobs, replace=replace, stagger=stagger, trace=trace
    )
    if licznik.get("failed") or licznik.get("timeout"):
        raise typer.Exit(1)


@degrade_app.command("status")
def degrade_status(plan: str = typer.Argument(...)) -> None:
    """Stan punktów planu — także tych, którym brakuje checkpointu rodzica."""
    from tsc_marl.robustness.degrade import load_plan
    from tsc_marl.runner import degrade

    for w in degrade.status(load_plan(plan)):
        att = f"{w['att_clean']:.1f}" if w["att_clean"] is not None else "—"
        # marker TYLKO dla biegu, który istnieje, ale ma inne parametry — brakujący
        # policzy się sam, bez `--replace`, i mylące byłoby sugerowanie inaczej
        przestarzaly = w["status"] != "brak" and not w["aktualny"]
        aktualny = "  NIEAKTUALNY (--replace)" if przestarzaly else ""
        print(f"{w['run_id']:64s} {w['status']:8s} {att:>7s}  ← {w['rodzic']}{aktualny}")


@degrade_app.command("ood")
def degrade_ood(plan: str = typer.Argument("t2_ood")) -> None:
    """Test gałęzi epistemic: polityka vs MaxPressure na nowym ruchu + rozbieżność ensemble."""
    from tsc_marl import libsignal
    from tsc_marl.analysis import ood
    from tsc_marl.robustness.degrade import load_plan

    libsignal.bootstrap()
    print(ood.main(load_plan(plan)))


@degrade_app.command("report")
def degrade_report(plan: str = typer.Argument("t2_nomech")) -> None:
    """Tabela i krzywe degradacji z policzonych biegów -> results/<plan>.md."""
    from tsc_marl.analysis import degradation
    from tsc_marl.robustness.degrade import load_plan

    print(degradation.main(load_plan(plan)))


# ------------------------------------------------------------------------ raporty


@app.command()
def table() -> None:
    """Przelicz tabele T1 i GATE Fazy 1 z zapisanych biegów do `results/` (nic nie uruchamia)."""
    from tsc_marl import libsignal
    from tsc_marl.analysis import report
    from tsc_marl.libsignal import experiment

    libsignal.bootstrap()
    report.build_table(experiment.experiment())


@app.command()
def runs() -> None:
    """Wypisz policzone biegi — status, parametry, ATT."""
    import pandas as pd

    from tsc_marl.libsignal import experiment

    pd.set_option("display.width", 200)
    df = experiment.experiment().results()
    print("Brak biegów w runs/." if df.empty else df.to_string(index=False))


@app.command(context_settings=PASSTHROUGH)
def plot(ctx: typer.Context) -> None:
    """Krzywe uczenia z runs/ → runs/plots/ (działa też dla biegu w toku)."""
    from tsc_marl.analysis import plot as plot_module

    plot_module.main(ctx.args)


@app.command()
def doctor() -> None:
    """Sprawdź środowisko: submoduł, SUMO_HOME, wersje, PYTHONHASHSEED."""
    from tsc_marl import libsignal, paths

    libsignal.check_ready()
    libsignal.bootstrap()
    print(f"repo:            {paths.REPO_ROOT}")
    print(f"LibSignal:       {paths.LIBSIGNAL}")
    print(f"SUMO_HOME:       {os.environ.get('SUMO_HOME')}")
    print(f"PYTHONHASHSEED:  {os.environ.get('PYTHONHASHSEED')}")
    for modul in ("torch", "cityflow", "torch_geometric"):
        try:
            m = __import__(modul)
            print(f"{modul + ':':16s} {getattr(m, '__version__', 'ok')}")
        except ImportError as e:
            print(f"{modul + ':':16s} BRAK ({e})")


if __name__ == "__main__":
    app()
