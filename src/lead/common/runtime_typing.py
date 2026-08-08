"""The type checker the jaxtyping import hook applies to every ``lead`` module.

Kept out of ``lead/__init__.py`` because the hook resolves it by dotted path,
so it must be importable before the hook is installed.
"""

import importlib
import sys

from beartype import BeartypeConf, beartype

# PEP 484's implicit numeric tower: an int satisfies a float annotation, and a
# float satisfies a complex one, as they do for every static checker. Beartype
# is strict without it, which would reject passing an int-valued config field
# to a function whose parameter is a distance in metres.
typechecker = beartype(conf=BeartypeConf(is_pep484_tower=True))

# Set by lead/__init__.py once the jaxtyping import hook is installed; stays
# None when LEAD_RUNTIME_TYPE_CHECKING is off, so import_unwrapped is then a
# plain import.
hook_manager = None


def import_unwrapped(module_name: str) -> None:
    """Import a module with the jaxtyping import hook paused, then restore it.

    For numba ``@njit(cache=True)`` kernel modules: wrapping a kernel with
    ``@jaxtyped`` makes numba pickle the wrapper's closure for the on-disk
    cache, which fails on its weakrefs.

    Args:
        module_name: Dotted path of the module to import unwrapped.
    """
    if hook_manager is None:
        importlib.import_module(module_name)
        return
    sys.meta_path.remove(hook_manager.hook)
    try:
        importlib.import_module(module_name)
    finally:
        sys.meta_path.insert(0, hook_manager.hook)
