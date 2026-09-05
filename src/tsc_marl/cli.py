"""Jedno wejście do projektu: `tsc <komenda>`.

Warstwa cienka z założenia — komendy tylko rozdzielają ruch, cała logika siedzi
w `run.py`, `run_core.py`, `evaluate.py`, `suite.py` i `report.py`. Dzięki temu
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
    from tsc_marl import run

    run.main(ctx.args)


@app.command(context_settings=PASSTHROUGH)
def core(ctx: typer.Context) -> None:
    """Rdzeń Fazy 1 — GAT + PPO + CTDE, własna pętla (`tsc core -n ... --seed 0`)."""
    from tsc_marl import run_core

    run_core.main(ctx.args)


@app.command("eval", context_settings=PASSTHROUGH)
def evaluate_(ctx: typer.Context) -> None:
    """Ewaluacja zamrożonej polityki z checkpointu (`tsc eval --load-run ...`)."""
    from tsc_marl import evaluate

    evaluate.main(ctx.args)


# ------------------------------------------------------------------------- siatki


@suite_app.command("run")
def suite_run(
    nazwa: str = typer.Argument(..., help="nazwa albo ścieżka pliku z configs/suites/"),
    replace: bool = typer.Option(False, "--replace", help="policz ponownie także gotowe punkty"),
) -> None:
    """Policz wszystkie punkty siatki (pomija te, które mają aktualny wynik)."""
    from tsc_marl import suite

    licznik = suite.run(suite.load(nazwa), replace=replace)
    if licznik.get("failed") or licznik.get("timeout"):
        raise typer.Exit(1)


@suite_app.command("status")
def suite_status(
    nazwa: str = typer.Argument(..., help="nazwa albo ścieżka pliku z configs/suites/"),
) -> None:
    """Stan każdego punktu siatki — także tych jeszcze niepoliczonych."""
    from tsc_marl import suite

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
    from tsc_marl import suite

    for path in sorted(suite.SUITES.glob("*.yml")):
        s = suite.load(path)
        print(f"  {s.name:16s} {s.module:20s} yellow={s.yellow}  {len(s.points())} punktów")


# ------------------------------------------------------------------------ raporty


@app.command()
def table() -> None:
    """Przelicz tabele T1 z zapisanych biegów do `results/` (nic nie uruchamia)."""
    from tsc_marl import experiment, libsignal, report

    libsignal.bootstrap()
    report.build_table(experiment.experiment())


@app.command()
def runs() -> None:
    """Wypisz policzone biegi — status, parametry, ATT."""
    import pandas as pd

    from tsc_marl import experiment

    pd.set_option("display.width", 200)
    df = experiment.experiment().results()
    print("Brak biegów w runs/." if df.empty else df.to_string(index=False))


@app.command(context_settings=PASSTHROUGH)
def plot(ctx: typer.Context) -> None:
    """Krzywe uczenia z runs/ → runs/plots/ (działa też dla biegu w toku)."""
    from tsc_marl import plot as plot_module

    plot_module.main(ctx.args)


@app.command()
def doctor() -> None:
    """Sprawdź środowisko: submoduł, SUMO_HOME, wersje, PYTHONHASHSEED."""
    from tsc_marl import libsignal

    libsignal.check_ready()
    libsignal.bootstrap()
    print(f"repo:            {libsignal.REPO_ROOT}")
    print(f"LibSignal:       {libsignal.LIBSIGNAL}")
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
