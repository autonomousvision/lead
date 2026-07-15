"""Bootstrap sys.path and environment variables once.

Simulator paths are optional: reading 123D logs needs none of them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from lead.common.dotenv import read_dotenv

# Put the vendored CARLA client, leaderboard and scenario runner on the path.
for _key, _suffix in [
    ("CARLA_ROOT", "PythonAPI/carla"),
    ("LEADERBOARD_ROOT", ""),
    ("SCENARIO_RUNNER_ROOT", ""),
]:
    try:
        _root = read_dotenv(_key)
    except KeyError:
        continue
    _p = Path(_root) / _suffix if _suffix else Path(_root)
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(-1, str(_p))

# Set environment variables from .env file if not already set
for env in ["PY123D_DATA_ROOT"]:
    if not os.environ.get(env):
        try:
            os.environ[env] = read_dotenv(env)
        except KeyError:
            pass

if os.environ.get("LEAD_RUNTIME_TYPE_CHECKING", "0") == "1":
    import importlib

    from jaxtyping import install_import_hook

    # Modules of numba @njit(cache=True) kernels: import them before the hook is
    # installed so they stay uninstrumented. Wrapping a kernel makes numba pickle
    # the wrapper's closure for the on-disk cache, which fails on its weakrefs.
    for _numba_module in [
        "lead.common.sensors.ransac",
        "lead.expert.driving.forecast_kernels",
    ]:
        importlib.import_module(_numba_module)

    # Applies @jaxtyped(typechecker=beartype) to every function and dataclass
    # in lead.* imported after this point.
    install_import_hook("lead", "beartype.beartype")
