"""Bootstrap sys.path and environment variables once.

Simulator paths are optional: reading 123D logs needs none of them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from lead.common.env import read_dotenv

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

if os.environ.get("LEAD_RUNTIME_TYPE_CHECKING", "true").lower() == "true":
    import importlib

    from jaxtyping import install_import_hook

    # lead.common.sensors.ransac is a numba @njit(cache=True) kernel module:
    # import it before the hook is installed so it stays uninstrumented.
    # Wrapping a kernel makes numba pickle the wrapper's closure for the
    # on-disk cache, which fails on its weakrefs.
    importlib.import_module("lead.common.sensors.ransac")

    # The hook resolves the checker by dotted path, so bind it as an attribute
    # of its package first. lead.common has an empty __init__, so nothing else
    # is pulled in ahead of the hook and left uninstrumented.
    from lead.common import runtime_typing

    # Applies @jaxtyped(typechecker=...) to every function and dataclass in
    # lead.* imported after this point. lead.expert.driving.forecast_kernels
    # has the same numba-cache concern as ransac above, but it also needs
    # CARLA, and importing it here -- before we know whether this process
    # will ever touch the expert -- would make CARLA load as a side effect
    # of importing anything under lead at all. lead/expert/__init__.py pauses
    # this hook for that one import instead, once lead.expert is actually
    # used; see lead.common.runtime_typing.import_unwrapped.
    runtime_typing.hook_manager = install_import_hook(
        "lead",
        "lead.common.runtime_typing.typechecker",
    )
