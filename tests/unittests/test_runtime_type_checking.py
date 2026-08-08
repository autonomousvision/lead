"""Guard that the unit suite runs with runtime type checking active.

`LEAD_RUNTIME_TYPE_CHECKING=true` makes `lead.__init__` install the jaxtyping
import hook, wrapping every `lead.*` function with beartype. Without it, argument
and shape violations pass silently and the suite stops catching type regressions.
"""

import os

import pytest
from jaxtyping import TypeCheckError

from lead.common.geometry import normalize_angle_rad


def test_runtime_type_checking_flag_enabled() -> None:
    """The env flag that installs the beartype import hook must be set."""
    assert os.environ.get("LEAD_RUNTIME_TYPE_CHECKING", "false").lower() == "true", (
        "Run the suite with LEAD_RUNTIME_TYPE_CHECKING=true so beartype checks "
        "every lead.* call against its annotations."
    )


def test_beartype_rejects_wrong_argument_type() -> None:
    """A wrong-typed argument to an instrumented lead.* function is rejected."""
    with pytest.raises(TypeCheckError):
        normalize_angle_rad("not a float")


def test_expert_stays_instrumented_except_its_numba_kernels() -> None:
    """lead.expert.driving.forecast_kernels dodges the hook; the rest of lead.expert doesn't.

    Regression guard for lead.common.runtime_typing.import_unwrapped: if it
    ever failed to restore the hook after pausing it for that one module,
    every lead.* import after the first lead.expert import would silently
    stop being type-checked, with no error anywhere to notice.
    """
    import numba

    try:
        from lead.expert.driving import forecast_kernels
        from lead.expert.utils.weather import get_night_mode
    except ModuleNotFoundError as error:
        root = (error.name or "").split(".")[0]
        pytest.skip(f"needs the simulator package {root!r}")

    kernels = [
        obj
        for obj in vars(forecast_kernels).values()
        if isinstance(obj, numba.core.dispatcher.Dispatcher)
    ]
    assert kernels, "expected at least one numba kernel in forecast_kernels"

    with pytest.raises(TypeCheckError):
        get_night_mode("not a weather param")

    # The hook must still be active for code imported after lead.expert.
    with pytest.raises(TypeCheckError):
        normalize_angle_rad("not a float either")
