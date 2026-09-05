"""Sterowanie sygnalizacją świetlną — MARL nad grafem sieci drogowej.

Pakiet dzieli się na warstwy, a zależności idą TYLKO w dół tej listy:

* `paths`, `seeding` — fundament, bez zależności wewnętrznych,
* `domain/`          — wzory z papierów (ATS), bez frameworków,
* `core/`            — rdzeń Fazy 1: GAT + PPO + CTDE, nasza implementacja,
* `libsignal/`       — most do submodułu `external/LibSignal` i wtyczki do ich Registry,
* `runner/`          — wejścia procesu (`tsc ...`) i orkiestracja biegów,
* `analysis/`        — tabele i wykresy z zapisanych biegów.

Kluczowa granica: `core/` NIE importuje `libsignal/`. Świat symulacji wstrzykuje
mu `runner/train_core.py`, dzięki czemu rdzeń testuje się bez silnika.
"""
