"""Reproducible seeding for RL experiments.

Seed EVERY source of randomness, including the simulator. Call
``set_global_seed(seed)`` once at the start of a run, and pass the same seed to
the simulator:

    from src.seeding import set_global_seed
    set_global_seed(1)
    # SUMO:     add  --seed 1   (or seed=1 via sumo-rl / TraCI)
    # CityFlow: set  "seed": 1  in the engine config JSON

Note on determinism: PyTorch does not guarantee bit-identical results across
library versions, hardware, or CPU-vs-GPU. The container pins versions; combine
that with fixed seeds and report mean +/- std over several seeds (e.g. 1..5)
rather than relying on a single "reproducible" number.
"""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int, deterministic: bool = True) -> int:
    """Seed Python, NumPy, and PyTorch. Returns the seed for convenience."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Required for deterministic cuBLAS matmuls (also set in the image).
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:  # noqa: BLE001 - older torch may not support this
                pass
    except ImportError:
        pass

    return seed


def seed_everything(seed: int, deterministic: bool = True) -> int:
    """Alias for set_global_seed (matches common PyG/Lightning naming)."""
    return set_global_seed(seed, deterministic=deterministic)
