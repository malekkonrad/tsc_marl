"""Agents implemented for this thesis, registered in LibSignal's Registry.

The Registry is a plain dict filled by decorators at import time
(`common/registry.py`), so importing this package after `import agent` makes our
methods available as `-a <name>` while the code stays in this repository.

  efficientmp     — Efficient-MP (Wu et al., 2021)
  advancedmp      — Advanced-MP (Zhang et al., 2022)
  advancedcolight — Advanced-CoLight (Zhang et al., 2022)
"""

from tsc_marl.libsignal.agents.advanced_colight import AdvancedCoLight  # noqa: F401
from tsc_marl.libsignal.agents.advanced_mp import (  # noqa: F401 - imported for registration
    AdvancedMaxPressureAgent,
    EfficientMaxPressureAgent,
)

__all__ = ["AdvancedCoLight", "AdvancedMaxPressureAgent", "EfficientMaxPressureAgent"]
