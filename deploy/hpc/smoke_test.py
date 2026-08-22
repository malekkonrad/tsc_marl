#!/usr/bin/env python3
"""Health check for the tsc-marl environment.

Run inside the container to confirm every layer works:
    docker run --rm tsc-marl python /opt/smoke_test.py
    docker run --rm --gpus all tsc-marl python /opt/smoke_test.py   # + GPU
    apptainer exec --nv tsc-marl.sif python /opt/smoke_test.py      # on HPC

Exit code is non-zero if any REQUIRED check fails.
"""
import importlib
import os
import shutil
import sys

OK = "\033[92mPASS\033[0m"
NO = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

required_failures = 0


def check(label, fn, required=True):
    global required_failures
    try:
        detail = fn()
        print(f"[{OK}] {label}: {detail}")
    except Exception as e:  # noqa: BLE001
        tag = NO if required else WARN
        print(f"[{tag}] {label}: {e}")
        if required:
            required_failures += 1


def _import(mod):
    def _f():
        m = importlib.import_module(mod)
        return f"v{getattr(m, '__version__', '?')}"
    return _f


print("=" * 64)
print(" tsc-marl environment smoke test")
print("=" * 64)
print(f"Python: {sys.version.split()[0]}  ({sys.executable})")

# --- core scientific stack ---
check("numpy", _import("numpy"))
check("torch", _import("torch"))


def _torch_cuda():
    import torch
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        names = ", ".join(torch.cuda.get_device_name(i) for i in range(n))
        return f"CUDA available - {n} GPU(s): {names} (torch built for CUDA {torch.version.cuda})"
    return "CPU-only (no CUDA visible - expected on the laptop / when --gpus is omitted)"


check("torch CUDA", _torch_cuda, required=False)
check("torch_geometric", _import("torch_geometric"))


def _pyg_ext():
    import torch_scatter  # noqa: F401
    return "compiled extensions present (accelerated scatter/sparse ops)"


check("torch_geometric C++ ext (optional)", _pyg_ext, required=False)

# --- SUMO ---
def _sumo_home():
    home = os.environ.get("SUMO_HOME")
    if not home or not os.path.isdir(home):
        raise RuntimeError(f"SUMO_HOME unset or missing: {home!r}")
    for sub in ("bin", "tools", "data"):
        if not os.path.isdir(os.path.join(home, sub)):
            raise RuntimeError(f"SUMO_HOME missing {sub}/ ({home})")
    return home


check("SUMO_HOME", _sumo_home)


def _sumo_binary():
    exe = shutil.which("sumo") or os.path.join(os.environ.get("SUMO_HOME", ""), "bin", "sumo")
    if not exe or not os.path.exists(exe):
        raise RuntimeError("sumo binary not found on PATH or in $SUMO_HOME/bin")
    return exe


check("sumo binary", _sumo_binary)
check("traci", _import("traci"))
check("libsumo", _import("libsumo"))
check("sumolib", _import("sumolib"))


def _sumo_rl():
    import sumo_rl  # noqa: F401
    return f"v{getattr(sumo_rl, '__version__', '?')}"


check("sumo-rl", _sumo_rl)

# --- CityFlow ---
def _cityflow():
    import cityflow
    # instantiating requires config files; just confirm the Engine symbol exists.
    assert hasattr(cityflow, "Engine")
    return "import OK, cityflow.Engine present"


check("cityflow", _cityflow)

# --- LibSignal on PYTHONPATH ---
def _libsignal():
    path = "/opt/LibSignal"
    if not os.path.isdir(path):
        raise RuntimeError("/opt/LibSignal missing")
    if path not in sys.path and path not in os.environ.get("PYTHONPATH", ""):
        raise RuntimeError("LibSignal not on PYTHONPATH")
    return f"{path} on PYTHONPATH"


check("LibSignal", _libsignal, required=False)

print("=" * 64)
if required_failures == 0:
    print(f"All required checks {OK}. Environment is ready.")
    sys.exit(0)
else:
    print(f"{required_failures} required check(s) {NO}. See above.")
    sys.exit(1)
