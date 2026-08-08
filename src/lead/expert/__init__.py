"""Import the expert's numba kernels before the jaxtyping hook can wrap them.

lead.expert.driving.forecast_kernels holds numba @njit(cache=True) kernels:
wrapping one with @jaxtyped makes numba pickle the wrapper's closure for the
on-disk cache, which fails on its weakrefs. Unlike lead.common.sensors.ransac,
this can't dodge the hook by importing before it installs -- that runs in
lead/__init__.py, for every consumer of lead, and would make CARLA (which
this module needs) load as a side effect of importing anything under lead at
all. Pausing the hook for this one import, once lead.expert is actually used,
keeps the rest of the expert -- and everything else -- instrumented.
"""

from lead.common.runtime_typing import import_unwrapped

import_unwrapped("lead.expert.driving.forecast_kernels")
