"""Bootstrap sys.path for bundled 3rd_party directories that lack pip installs.

Runs once on the first import of `lead.*`. Prepends paths to sys.path without
touching os.environ, so any caller-set PYTHONPATH is preserved. This avoids
needing PYTHONPATH exports for notebooks / IDEs that spawn a Python process
directly (Jupyter, PyCharm, VSCode, plain `python`, pytest, etc.).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_THIRD_PARTY_PATHS = [
    _REPO_ROOT / "3rd_party/CARLA_0915/PythonAPI/carla",
]
for _p in _THIRD_PARTY_PATHS:
    _s = str(_p)
    if _p.exists():
        if _s not in sys.path:
            sys.path.insert(-1, _s)
    else:
        print(
            f"3rd-party path not found: {_p}. Please ensure{_p} exists.",
        )
