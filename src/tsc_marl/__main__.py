"""`python -m tsc_marl` = `tsc`. Istnieje, żeby CLI działało również zanim
console script trafi do .venv/bin (czyli przed `uv pip install -e .`)."""

from tsc_marl.runner.cli import app

app()
