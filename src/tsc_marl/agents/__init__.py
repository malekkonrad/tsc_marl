"""Agenci zaimplementowani na potrzeby pracy.

Registry LibSignala to zwykły słownik zapełniany dekoratorami przy imporcie
(`common/registry.py`), więc wystarczy zaimportować ten pakiet PO `import agent`,
żeby nasze metody były widoczne pod swoimi nazwami w `-a {nazwa}`. Dzięki temu nasz
kod zostaje w naszym repo zamiast wsiąkać w drzewo submodułu.

Zaimplementowane:
  efficientmp  — Efficient-MP (Wu i in. 2021); odtwarza literaturę z dokładnością 0,9%
  advancedmp   — Advanced-MP (Zhang i in. 2022)

TODO (baseline'y z planu, jeszcze nie zaimplementowane):
  advancedcolight — Advanced-CoLight (Zhang i in. 2022), linia SOTA
  bgat            — kontrybucja pracy (Faza 2)
"""

from tsc_marl.agents.advanced_mp import (  # noqa: F401 - import dla efektu rejestracji
    AdvancedMaxPressureAgent,
    EfficientMaxPressureAgent,
)

__all__ = ["AdvancedMaxPressureAgent", "EfficientMaxPressureAgent"]
