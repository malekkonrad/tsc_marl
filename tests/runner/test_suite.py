"""Declarative grids: the grid comes from YAML, not shell history.

The core of it: a suite's `overrides` MUST enter the config before the run
fingerprint is computed; applied later they would change behaviour without
changing `config_hash`, and skip-if-done would treat a run with other
hyperparameters as already done. The other tests guard the process boundary:
parent and child must compute the same config in separate interpreters.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time

import pytest

from runlog import params_hash
from tsc_marl import libsignal
from tsc_marl.libsignal import session
from tsc_marl.libsignal.experiment import parse_args
from tsc_marl.runner import suite

NETWORK = "cityflow_jinan3x4_4ph"


def write(tmp_path, content: str, name: str = "trial.yml"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(content))
    return p


# ------------------------------------------------------------------- grid


def test_points_are_the_cartesian_product(tmp_path):
    s = suite.load(
        write(
            tmp_path,
            """
            grid:
              agent: [fixedtime, maxpressure]
              network: [jinan, hangzhou]
              seed: [0, 1, 2]
            """,
        )
    )
    points = s.points()
    assert len(points) == 12
    assert len({(p.agent, p.network, p.seed) for p in points}) == 12
    # `world` is not in the grid: it comes from the defaults
    assert {p.world for p in points} == {"cityflow"}


def test_a_field_outside_the_grid_comes_from_defaults(tmp_path):
    s = suite.load(
        write(
            tmp_path,
            """
            defaults: {agent: colight, seed: 7}
            grid: {network: [jinan]}
            """,
        )
    )
    (point,) = s.points()
    assert (point.agent, point.seed) == ("colight", 7)


def test_a_field_missing_everywhere_fails_at_once(tmp_path):
    """A clear message while planning beats a KeyError in a subprocess later."""
    s = suite.load(write(tmp_path, "grid: {network: [jinan]}\n"))
    with pytest.raises(SystemExit, match="agent"):
        s.points()


def test_entry_picks_the_entry_module(tmp_path):
    baseline = suite.load(write(tmp_path, "grid: {agent: [x], network: [y], seed: [0]}\n"))
    core = suite.load(write(tmp_path, "entry: core\ngrid: {network: [y], seed: [0]}\n", "core.yml"))
    assert baseline.module == "tsc_marl.runner.train"
    assert core.module == "tsc_marl.runner.train_core"
    assert core.points()[0].agent == "gatppo", "`entry: core` computes the core by default"


def test_mode_sets_yellow(tmp_path):
    common = "grid: {agent: [x], network: [y], seed: [0]}\n"
    assert suite.load(write(tmp_path, common, "a.yml")).yellow == 5
    assert suite.load(write(tmp_path, "mode: benchmark\n" + common, "b.yml")).yellow == 0


# --------------------------------------------------------- process boundary


def test_argv_passes_a_path_not_a_dict(tmp_path):
    """Overrides reach the subprocess as `--suite <file>`, not a serialised dict:
    YAML types stay YAML and a command from a log can be repeated by hand."""
    s = suite.load(
        write(
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


def test_suffix_only_for_suites_with_overrides(tmp_path):
    """A suite WITHOUT overrides hits the run ids of manual runs; one WITH overrides
    needs its own, or two configs fight over one directory."""
    plain = suite.load(write(tmp_path, "grid: {agent: [x], network: [y], seed: [0]}\n", "g.yml"))
    changed = suite.load(
        write(
            tmp_path,
            "grid: {agent: [x], network: [y], seed: [0]}\noverrides: {model: {a: 1}}\n",
            "z.yml",
        )
    )
    assert plain.suffix == ""
    assert changed.suffix == "z"


def test_overrides_change_fingerprint_and_run_id(tmp_path):
    """`plan()`, which the runner uses for skip-if-done, must see the overrides;
    otherwise the parent would treat a point as done and the child compute
    another experiment."""
    libsignal.bootstrap()
    common = f"grid: {{agent: [advancedcolight], network: [{NETWORK}], seed: [0]}}\n"
    plain = suite.load(write(tmp_path, common, "plain.yml"))
    changed = suite.load(
        write(tmp_path, common + "overrides:\n  model: {NUM_HEADS: [8, 8]}\n", "change.yml")
    )

    _, params_plain, rid_plain = suite.plan(plain, plain.points()[0])
    _, params_changed, rid_changed = suite.plan(changed, changed.points()[0])

    assert params_plain["config_hash"] != params_changed["config_hash"]
    assert rid_plain != rid_changed and rid_changed.endswith("__change")


def test_plan_and_the_computing_process_take_the_same_path(tmp_path):
    """`plan()` (parent) and `session.resolve()` (child) give the same config."""
    libsignal.bootstrap()
    s = suite.load(
        write(
            tmp_path,
            f"grid: {{agent: [advancedcolight], network: [{NETWORK}], seed: [0]}}\n"
            "overrides:\n  trainer: {episodes: 3}\n",
        )
    )
    point = s.points()[0]
    parent_args, params, _ = suite.plan(s, point)
    child_config, suffix = session.resolve(parse_args(s.argv(point)))

    assert child_config["trainer"]["episodes"] == 3
    assert params["episodes"] == 3
    assert suffix == s.suffix
    assert parent_args.suite == str(s.path)


# ------------------------------------------------------------------- validation


def test_a_typo_in_a_field_fails_instead_of_being_ignored(tmp_path):
    """Silently ignoring `overides:` would mean a run computed without overrides."""
    with pytest.raises(SystemExit, match="overides"):
        suite.load(write(tmp_path, "overides: {model: {a: 1}}\ngrid: {network: [y]}\n"))


def test_unknown_mode_and_entry_fail(tmp_path):
    with pytest.raises(SystemExit, match="mode"):
        suite.load(write(tmp_path, "mode: paper\ngrid: {network: [y]}\n", "m.yml"))
    with pytest.raises(SystemExit, match="entry"):
        suite.load(write(tmp_path, "entry: rocket\ngrid: {network: [y]}\n", "e.yml"))


def test_a_missing_suite_lists_the_available_ones():
    with pytest.raises(SystemExit, match="Available"):
        suite.load("no_such_suite")


# ----------------------------------------------------------------- device


def test_ngpu_reaches_the_child_as_a_flag(tmp_path):
    """`ngpu` from YAML reaches the subprocess argv."""
    p = write(
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


def test_cpu_by_default(tmp_path):
    p = write(
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


# ------------------------------------------------------------------ parallel


def _launcher(tmp_path, codes: dict, durations: dict | None = None):
    """`launch` for `schedule` on plain Python processes: point -> exit code (a list
    = successive attempts). Records how often each point started."""
    starts: dict = {}
    durations = durations or {}

    def launch(point):
        n = starts.get(point, 0)
        starts[point] = n + 1
        code = codes[point][min(n, len(codes[point]) - 1)]
        sleep = durations.get(point, 0.05)
        log = open(tmp_path / f"{point}.log", "a")  # noqa: SIM115
        cmd = [sys.executable, "-c", f"import time,sys; time.sleep({sleep}); sys.exit({code})"]
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log

    return launch, starts


def test_schedule_computes_everything_and_returns_statuses(tmp_path):
    launch, starts = _launcher(tmp_path, {"a": [0], "b": [0], "c": [0]})
    result = suite.schedule(["a", "b", "c"], launch, jobs=2, timeout=30, stagger=0)
    assert result == {"a": "ok", "b": "ok", "c": "ok"}
    assert starts == {"a": 1, "b": 1, "c": 1}


def test_an_early_failure_is_retried_once(tmp_path):
    """A failure in the first seconds is most likely the `configs/sim/*.cfg` race
    (see STAGGER_S): retried ONCE. A second failure is a real error."""
    launch, starts = _launcher(tmp_path, {"transient": [1, 0], "broken": [1]})
    result = suite.schedule(["transient", "broken"], launch, jobs=2, timeout=30, stagger=0)
    assert result == {"transient": "ok", "broken": "failed"}
    assert starts == {"transient": 2, "broken": 2}


def test_timeout_kills_the_process(tmp_path):
    launch, _ = _launcher(tmp_path, {"slow": [0]}, {"slow": 30})
    result = suite.schedule(["slow"], launch, jobs=1, timeout=0.5, stagger=0)
    assert result == {"slow": "timeout"}


def test_stagger_spaces_the_starts(tmp_path):
    """Starts at least `stagger` apart, even with free slots."""
    launch, _ = _launcher(tmp_path, {"a": [0], "b": [0]})
    t0 = time.time()
    suite.schedule(["a", "b"], launch, jobs=2, timeout=30, stagger=0.6)
    assert time.time() - t0 >= 0.6


@pytest.mark.slow
def test_ngpu_is_not_part_of_the_fingerprint(tmp_path):
    """The same run on CPU and GPU is ONE experiment: `ngpu` describes the machine
    and lands in the `command` section, outside the fingerprint. Otherwise moving a
    grid to a stronger machine would invalidate everything computed so far."""
    libsignal.bootstrap()
    p = write(
        tmp_path,
        """
        entry: core
        grid:
          network: [cityflow_jinan3x4_4ph]
          seed: [0]
        """,
    )
    s = suite.load(p)
    point = s.points()[0]

    hashes, ids = set(), set()
    for ngpu in ("-1", "0", "3"):
        s.ngpu = ngpu
        _, params, rid = suite.plan(s, point)
        hashes.add(params_hash(params))
        ids.add(rid)

    assert len(hashes) == 1, f"ngpu changes the params fingerprint: {hashes}"
    assert len(ids) == 1, f"ngpu changes the run_id: {ids}"
