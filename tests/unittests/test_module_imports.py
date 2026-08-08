"""Every module imports cleanly, including under the runtime type checker.

The jaxtyping import hook builds its checks at import time, so an annotation it
cannot resolve — a forward reference to a ``TYPE_CHECKING``-only name, say —
raises when the module is first imported rather than when the function runs. A
module no test imports would otherwise reach CI broken.
"""

import importlib
import pkgutil
import subprocess
import sys

import pytest

import lead

# Only installed with the simulator, which unit tests do not have. A module
# that needs one of these is skipped rather than excluded by name, so the check
# still covers it wherever the simulator is on the path.
_SIMULATOR_PACKAGES = frozenset({"carla", "leaderboard", "srunner"})


def _importable_modules() -> list[str]:
    """Every ``lead.*`` module.

    Returns:
        The module names, sorted.
    """
    return sorted(
        module.name for module in pkgutil.walk_packages(lead.__path__, "lead.")
    )


@pytest.mark.parametrize("module_name", _importable_modules())
def test_module_imports(module_name: str) -> None:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        root = (error.name or "").split(".")[0]
        if root not in _SIMULATOR_PACKAGES:
            raise
        pytest.skip(f"{module_name} needs the simulator package {root!r}")


def test_offline_stack_does_not_import_carla_or_expert() -> None:
    """Importing the offline stack must not pull in CARLA or lead.expert.

    Runs in a subprocess with a fresh interpreter: within this process other
    tests legitimately import lead.expert directly (e.g. kinematic bicycle
    model tests), so an in-process ``"carla" not in sys.modules`` check would
    be polluted by whichever tests already ran.
    """
    check = (
        "import lead.training.run_config, sys; "
        "assert 'carla' not in sys.modules, 'pulled in carla'; "
        "assert not any(m == 'lead.expert' or m.startswith('lead.expert.') "
        "for m in sys.modules), 'pulled in lead.expert'"
    )
    result = subprocess.run(
        [sys.executable, "-c", check],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
