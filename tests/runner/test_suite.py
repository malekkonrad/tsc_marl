"""Testy warstwy deklaratywnej — siatka z YAML-a zamiast z historii powłoki.

Sedno jest w jednym miejscu: `overrides` suite'a MUSZĄ wejść do configu przed
policzeniem odcisku biegu. Nałożone później zmieniałyby zachowanie, nie zmieniając
`config_hash` — a wtedy `skip-if-done` pomija bieg z innymi hiperparametrami jako
„już policzony". Reszta testów pilnuje granicy procesu: rodzic i dziecko muszą
policzyć ten sam config, choć są osobnymi interpreterami.
"""

from __future__ import annotations

import textwrap

import pytest

from tsc_marl import libsignal
from tsc_marl.runner import suite

NETWORK = "cityflow_jinan3x4_4ph"


def zapisz(tmp_path, tresc: str, nazwa: str = "probny.yml"):
    p = tmp_path / nazwa
    p.write_text(textwrap.dedent(tresc))
    return p


# ------------------------------------------------------------------- siatka


def test_punkty_to_iloczyn_kartezjanski(tmp_path):
    s = suite.load(
        zapisz(
            tmp_path,
            """
            grid:
              agent: [fixedtime, maxpressure]
              network: [jinan, hangzhou]
              seed: [0, 1, 2]
            """,
        )
    )
    punkty = s.points()
    assert len(punkty) == 12
    assert len({(p.agent, p.network, p.seed) for p in punkty}) == 12
    # `world` nie jest w siatce — wchodzi z domyślnych
    assert {p.world for p in punkty} == {"cityflow"}


def test_pole_spoza_siatki_bierze_wartosc_z_defaults(tmp_path):
    s = suite.load(
        zapisz(
            tmp_path,
            """
            defaults: {agent: colight, seed: 7}
            grid: {network: [jinan]}
            """,
        )
    )
    (punkt,) = s.points()
    assert (punkt.agent, punkt.seed) == ("colight", 7)


def test_brak_pola_wszedzie_to_blad_od_razu(tmp_path):
    """Lepszy komunikat przy planowaniu niż KeyError w podprocesie po pół godzinie."""
    s = suite.load(zapisz(tmp_path, "grid: {network: [jinan]}\n"))
    with pytest.raises(SystemExit, match="agent"):
        s.points()


def test_entry_wybiera_modul_wejsciowy(tmp_path):
    """To jest powód istnienia `entry`: przed nim runner siatki miał zaszyte
    `tsc_marl.runner.train` i rdzeń Fazy 1 był z niego nieosiągalny."""
    baseline = suite.load(zapisz(tmp_path, "grid: {agent: [x], network: [y], seed: [0]}\n"))
    rdzen = suite.load(
        zapisz(tmp_path, "entry: core\ngrid: {network: [y], seed: [0]}\n", "rdzen.yml")
    )
    assert baseline.module == "tsc_marl.runner.train"
    assert rdzen.module == "tsc_marl.runner.train_core"
    assert rdzen.points()[0].agent == "gatppo", "`entry: core` domyślnie liczy rdzeń"


def test_mode_ustawia_yellow(tmp_path):
    wspolne = "grid: {agent: [x], network: [y], seed: [0]}\n"
    assert suite.load(zapisz(tmp_path, wspolne, "a.yml")).yellow == 5
    assert suite.load(zapisz(tmp_path, "mode: benchmark\n" + wspolne, "b.yml")).yellow == 0


# --------------------------------------------------------- granica procesu


def test_argv_przekazuje_sciezke_a_nie_slownik(tmp_path):
    """Nadpisania jadą do podprocesu jako `--suite <plik>`, nie jako serializacja
    słownika. Dzięki temu typy YAML-owe zostają YAML-owe, a komendę z logu da się
    skopiować i powtórzyć ręcznie."""
    s = suite.load(
        zapisz(
            tmp_path,
            """
            grid: {agent: [gatppo], network: [jinan], seed: [0]}
            overrides:
              model: {NUM_HEADS: [8, 8]}
            """,
        )
    )
    argv = s.argv(s.points()[0])
    assert "--suite" in argv and str(s.path) in argv
    assert not any(a.startswith("--set") for a in argv)
    assert argv[:2] == ["-a", "gatppo"]
    assert "--replace" not in argv
    assert "--replace" in s.argv(s.points()[0], replace=True)


def test_suffix_tylko_dla_suite_ow_z_nadpisaniami(tmp_path):
    """Suite BEZ nadpisań ma trafiać w te same `run_id`, co uruchomienia ręczne —
    inaczej nie zobaczyłby wcześniejszych wyników jako gotowych. Suite Z nadpisaniami
    musi mieć własne, bo inaczej dwa różne configi biją się o jeden katalog."""
    goly = suite.load(zapisz(tmp_path, "grid: {agent: [x], network: [y], seed: [0]}\n", "g.yml"))
    zmieniony = suite.load(
        zapisz(
            tmp_path,
            "grid: {agent: [x], network: [y], seed: [0]}\noverrides: {model: {a: 1}}\n",
            "z.yml",
        )
    )
    assert goly.suffix == ""
    assert zmieniony.suffix == "z"


def test_overrides_zmieniaja_odcisk_i_run_id(tmp_path):
    """Test za całą warstwę: `plan()` — czyli to, czym runner sprawdza skip-if-done —
    musi zobaczyć nadpisania. Gdyby wchodziły dopiero w podprocesie, rodzic uznałby
    punkt za policzony, a dziecko policzyłoby zupełnie inny eksperyment."""
    libsignal.bootstrap()
    wspolne = f"grid: {{agent: [advancedcolight], network: [{NETWORK}], seed: [0]}}\n"
    goly = suite.load(zapisz(tmp_path, wspolne, "goly.yml"))
    zmieniony = suite.load(
        zapisz(tmp_path, wspolne + "overrides:\n  model: {NUM_HEADS: [8, 8]}\n", "zmiana.yml")
    )

    _, params_goly, rid_goly = suite.plan(goly, goly.points()[0])
    _, params_zm, rid_zm = suite.plan(zmieniony, zmieniony.points()[0])

    assert params_goly["config_hash"] != params_zm["config_hash"]
    assert rid_goly != rid_zm and rid_zm.endswith("__zmiana")


def test_plan_i_proces_liczacy_ida_ta_sama_droga(tmp_path):
    """`plan()` (rodzic) i `session.resolve()` (dziecko) muszą dać ten sam config.
    Dwie osobne ścieżki to gwarancja cichego rozjazdu przy pierwszej zmianie."""
    from tsc_marl.libsignal import session
    from tsc_marl.runner.train import parse_args

    libsignal.bootstrap()
    s = suite.load(
        zapisz(
            tmp_path,
            f"grid: {{agent: [advancedcolight], network: [{NETWORK}], seed: [0]}}\n"
            "overrides:\n  trainer: {episodes: 3}\n",
        )
    )
    punkt = s.points()[0]
    args_rodzica, params, _ = suite.plan(s, punkt)
    config_dziecka, suffix = session.resolve(parse_args(s.argv(punkt)))

    assert config_dziecka["trainer"]["episodes"] == 3
    assert params["episodes"] == 3
    assert suffix == s.suffix
    assert args_rodzica.suite == str(s.path)


# ------------------------------------------------------------------- walidacja


def test_literowka_w_polu_pada_zamiast_byc_ignorowana(tmp_path):
    """Ciche zignorowanie `overides:` znaczyłoby bieg policzony bez nadpisań —
    i tabelę, w której nie widać, że eksperymentu nie było."""
    with pytest.raises(SystemExit, match="overides"):
        suite.load(zapisz(tmp_path, "overides: {model: {a: 1}}\ngrid: {network: [y]}\n"))


def test_nieznany_tryb_i_nieznane_wejscie_padaja(tmp_path):
    with pytest.raises(SystemExit, match="mode"):
        suite.load(zapisz(tmp_path, "mode: papierowy\ngrid: {network: [y]}\n", "m.yml"))
    with pytest.raises(SystemExit, match="entry"):
        suite.load(zapisz(tmp_path, "entry: rakieta\ngrid: {network: [y]}\n", "e.yml"))


def test_nieistniejacy_suite_wypisuje_dostepne():
    with pytest.raises(SystemExit, match="Dostępne"):
        suite.load("nie_ma_takiego")


# ----------------------------------------------------------------- urządzenie


def test_ngpu_leci_flaga_do_dziecka(tmp_path):
    """`ngpu` z YAML-a musi trafić do argv podprocesu — inaczej suite liczyłby
    na CPU niezależnie od tego, co deklaruje siatka (tak było do Fazy 1)."""
    p = zapisz(
        tmp_path,
        """
        entry: core
        ngpu: "2"
        grid:
          network: [cityflow_jinan3x4_4ph]
          seed: [0]
        """,
    )
    s = suite.load(p)
    argv = s.argv(s.points()[0])
    assert argv[argv.index("--ngpu") + 1] == "2"


def test_domyslnie_cpu(tmp_path):
    """Brak `ngpu` w YAML-u = CPU. Maszyna deweloperska nie ma karty, a siatka
    bez tego pola ma działać tak jak przed dodaniem obsługi GPU."""
    p = zapisz(
        tmp_path,
        """
        grid:
          agent: [efficientmp]
          network: [cityflow_jinan3x4_4ph]
          seed: [0]
        """,
    )
    s = suite.load(p)
    argv = s.argv(s.points()[0])
    assert argv[argv.index("--ngpu") + 1] == "-1"


@pytest.mark.slow
def test_ngpu_nie_wchodzi_do_odcisku(tmp_path):
    """NAJWAŻNIEJSZY test tej sekcji: ten sam bieg na CPU i na GPU to JEDEN
    eksperyment, nie dwa.

    `ngpu` opisuje maszynę, nie treść naukową biegu, więc ląduje w sekcji
    `command` — jedynej, której `build_params` nie bierze do odcisku. Gdyby
    wpadło do `model`/`trainer`, przeniesienie siatki na mocniejszą maszynę
    unieważniłoby wszystko, co już policzone, i kazało liczyć od zera.
    """
    from runlog import params_hash

    libsignal.bootstrap()
    p = zapisz(
        tmp_path,
        """
        entry: core
        grid:
          network: [cityflow_jinan3x4_4ph]
          seed: [0]
        """,
    )
    s = suite.load(p)
    punkt = s.points()[0]

    odciski, identyfikatory = set(), set()
    for ngpu in ("-1", "0", "3"):
        s.ngpu = ngpu
        _, params, rid = suite.plan(s, punkt)
        odciski.add(params_hash(params))
        identyfikatory.add(rid)

    assert len(odciski) == 1, f"ngpu zmienia odcisk parametrów: {odciski}"
    assert len(identyfikatory) == 1, f"ngpu zmienia run_id: {identyfikatory}"
