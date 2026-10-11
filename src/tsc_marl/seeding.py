"""Global seeding of Python, NumPy and PyTorch.

The simulator is not seeded here: traffic is fixed by the flow file, and
`PYTHONHASHSEED` must be set before the interpreter starts (`runner/cli.py`).
PyTorch does not guarantee bit-identical results across versions, hardware or
CPU vs GPU, so results are reported as mean ± std over seeds.
"""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int, deterministic: bool = True) -> int:
    """Seed Python, NumPy, and PyTorch. Returns the seed for convenience."""
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
