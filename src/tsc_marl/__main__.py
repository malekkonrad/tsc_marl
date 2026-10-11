"""`python -m tsc_marl` is equivalent to `tsc`; works before the console script
is installed into .venv/bin."""

from tsc_marl.runner.cli import app

app()
