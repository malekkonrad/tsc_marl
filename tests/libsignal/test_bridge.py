"""Bridge to the submodule: one thing so far, but costly when it fails."""

from __future__ import annotations

from tsc_marl import libsignal, paths


def test_sim_configs_return_to_their_state_before_the_run(tmp_path, monkeypatch):
    """LibSignal writes fresh replay paths into `configs/sim/*.cfg` on every run.
    The SNAPSHOT is restored, not the git version: `git checkout -- configs/sim`
    would also delete uncommitted edits to these files, silently."""
    sim = tmp_path / "sim"
    sim.mkdir()
    cfg = sim / "net.cfg"
    cfg.write_text('{"roadnetFile": "my/fixed/path.json"}')
    other = sim / "other.cfg"
    other.write_text("{}")
    monkeypatch.setattr(paths, "CONFIGS", tmp_path)

    with libsignal.sim_configs_untouched():
        cfg.write_text('{"roadnetFile": "my/fixed/path.json", "replayLogFile": "junk"}')
        other.write_text('{"replayLogFile": "junk"}')

    assert cfg.read_text() == '{"roadnetFile": "my/fixed/path.json"}'
    assert other.read_text() == "{}"


def test_the_snapshot_is_restored_after_an_exception_too(tmp_path, monkeypatch):
    """A failed run must not leave configs dirtied with replay paths."""
    sim = tmp_path / "sim"
    sim.mkdir()
    cfg = sim / "net.cfg"
    cfg.write_text("original")
    monkeypatch.setattr(paths, "CONFIGS", tmp_path)

    try:
        with libsignal.sim_configs_untouched():
            cfg.write_text("dirtied by the engine")
            raise RuntimeError("simulator crashed")
    except RuntimeError:
        pass

    assert cfg.read_text() == "original"
