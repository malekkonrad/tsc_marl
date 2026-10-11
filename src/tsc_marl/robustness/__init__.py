"""Controller behaviour under sensor failure.

`controllers` — controllers and wrappers that all read the same detector layer
(policy, MaxPressure, FixedTime, gate/mask/UW wrappers); `degrade` — evaluation plans
and the loop running a frozen controller over failure scenarios; `training` —
failure randomization and uncertainty signals used during training. Entry point:
`runner/degrade.py` (`tsc degrade`).
"""
