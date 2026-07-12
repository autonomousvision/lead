"""Bootstrap sys.path and environment variables once."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from lead.common.dotenv import read_dotenv

# Insert PYTHONPATHS into sys.path
for _p in [
    Path(read_dotenv("CARLA_ROOT")) / "PythonAPI/carla",
    Path(read_dotenv("LEADERBOARD_ROOT")),
    Path(read_dotenv("SCENARIO_RUNNER_ROOT")),
]:
    _s = str(_p)
    if _p.exists():
        if _s not in sys.path:
            sys.path.insert(-1, _s)
    else:
        print(
            f"PYTHON path not found: {_p}. Please ensure{_p} exists.",
        )

# Set environment variables from .env file if not already set
for env in ["PY123D_DATA_ROOT"]:
    if not os.environ.get(env):
        os.environ[env] = read_dotenv(env)

if os.environ.get("LEAD_RUNTIME_TYPE_CHECKING", "0") == "1":
    import importlib

    from jaxtyping import install_import_hook

    # Modules of numba @njit(cache=True) kernels. The hook would wrap each kernel's
    # Python function, and numba then compiles the wrapper and tries to pickle its
    # closure for the on-disk cache -- which fails on the wrapper's weakrefs. Import
    # them first: the hook only instruments modules imported after it is installed.
    # Their signatures are enforced by numba itself, so nothing is lost.
    for _numba_module in [
        "lead.common.sensors.ransac",
        "lead.common.sensors.ransac_fast",
        "lead.lead.planning.forecast_kernels",
    ]:
        importlib.import_module(_numba_module)

    # Applies @jaxtyped(typechecker=beartype) to every function and dataclass
    # in lead.* imported after this point.
    install_import_hook("lead", "beartype.beartype")
