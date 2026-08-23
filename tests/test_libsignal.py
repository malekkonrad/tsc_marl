"""Testy mostu do submodułu — na razie jedna rzecz, ale kosztowna, gdy zawiedzie."""

from __future__ import annotations

from tsc_marl import libsignal


def test_sim_configs_wracaja_do_stanu_sprzed_biegu(tmp_path, monkeypatch):
    """LibSignal wpisuje do `configs/sim/*.cfg` świeże ścieżki replaya przy każdym biegu.

    Kluczowe: przywracamy MIGAWKĘ, a nie wersję z gita. Poprzednia implementacja robiła
    `git checkout -- configs/sim`, więc kasowała też niezacommitowane poprawki w tych
    plikach — cicho, po każdym uruchomieniu. Kosztowało to poprawkę ścieżki do roadnetu
    HangZhou, którą trzeba było nanieść drugi raz.
    """
    sim = tmp_path / "sim"
    sim.mkdir()
    cfg = sim / "siec.cfg"
    cfg.write_text('{"roadnetFile": "moja/poprawiona/sciezka.json"}')
    inny = sim / "drugi.cfg"
    inny.write_text("{}")
    monkeypatch.setattr(libsignal, "CONFIGS", tmp_path)

    with libsignal.sim_configs_untouched():
        cfg.write_text('{"roadnetFile": "moja/poprawiona/sciezka.json", "replayLogFile": "smiec"}')
        inny.write_text('{"replayLogFile": "smiec"}')

    assert cfg.read_text() == '{"roadnetFile": "moja/poprawiona/sciezka.json"}'
    assert inny.read_text() == "{}"


def test_migawka_przywraca_tez_po_wyjatku(tmp_path, monkeypatch):
    """Bieg, który padł, nie może zostawić configów zabrudzonych ścieżkami replaya."""
    sim = tmp_path / "sim"
    sim.mkdir()
    cfg = sim / "siec.cfg"
    cfg.write_text("oryginał")
    monkeypatch.setattr(libsignal, "CONFIGS", tmp_path)

    try:
        with libsignal.sim_configs_untouched():
            cfg.write_text("zabrudzone przez silnik")
            raise RuntimeError("symulator padł")
    except RuntimeError:
        pass

    assert cfg.read_text() == "oryginał"
