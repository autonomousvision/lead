from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DOTENV_PATH = REPO_ROOT / ".env"
DOTENV_EXAMPLE_PATH = REPO_ROOT / ".env.example"


def read_dotenv(key: str, default: str | None = None) -> str:
    """Read one value from ``.env``, freshly from disk on every call.

    Args:
        key: Variable name, e.g. ``"MAX_NUM_PARALLEL_JOBS_TOWN13"``.
        default: Value returned when the key is missing.

    Returns:
        The raw string value.

    Raises:
        KeyError: If the key is missing and no default is given.
    """
    path = DOTENV_PATH if DOTENV_PATH.exists() else DOTENV_EXAMPLE_PATH
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            values[k.strip()] = v.strip().strip("'\"")
    if key in values:
        return values[key]
    if default is not None:
        return default
    raise KeyError(f"{key} not found in {path}")


def read_dotenv_int(key: str, default: int | None = None) -> int:
    """Read one integer value from ``.env``, freshly from disk on every call.

    Args:
        key: Variable name, e.g. ``"COLLECT_DATA_CARLA_BOOT_TIMEOUT"``.
        default: Value returned when the key is missing.

    Returns:
        The value parsed as int.
    """
    return int(read_dotenv(key, None if default is None else str(default)))
