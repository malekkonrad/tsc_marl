"""Process entry points and run orchestration.

Top layer: may import everything below, nothing imports it. `cli.py` only
dispatches; the logic lives in `train.py`, `train_core.py`, `evaluate.py`,
`suite.py` and `degrade.py`, so tests and `suite` subprocesses call the same
functions rather than the CLI.
"""
