import os
from pathlib import Path

from setuptools import setup

ROOT: Path = Path(__file__).resolve().parent


def create_dotenv() -> None:
    example = (ROOT / ".env.example").read_text().replace("<LEAD directory>", str(ROOT))
    dotenv = ROOT / ".env"
    if not dotenv.exists():
        dotenv.write_text(example)
        return

    # Append keys present in .env.example but missing from .env; local values are never touched.
    def keys(text: str) -> set[str]:
        lines = (line.strip() for line in text.splitlines())
        return {
            line.partition("=")[0].strip()
            for line in lines
            if line and not line.startswith("#") and "=" in line
        }

    missing = keys(example) - keys(dotenv.read_text())
    if missing:
        added = [
            line
            for line in example.splitlines()
            if line.strip().partition("=")[0].strip() in missing
        ]
        with dotenv.open("a") as f:
            f.write("\n# --- Added from .env.example ---\n")
            f.write("\n".join(added) + "\n")


def install_path_hook() -> None:
    path_dirs = f"{ROOT}:{ROOT}/scripts/cli:{ROOT}/scripts/common"
    export_line = (
        f'case ":$PATH:" in *":{ROOT}/scripts/cli:"*) ;; '
        f'*) export PATH="{path_dirs}:$PATH" ;; esac'
    )
    conda_prefix = os.environ.get("CONDA_PREFIX")
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if conda_prefix:
        hook = Path(conda_prefix) / "etc" / "conda" / "activate.d" / "lead_path.sh"
        hook.parent.mkdir(parents=True, exist_ok=True)
        hook.write_text(
            export_line + "\n" + 'export UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"\n',
        )
        unhook = Path(conda_prefix) / "etc" / "conda" / "deactivate.d" / "lead_path.sh"
        unhook.parent.mkdir(parents=True, exist_ok=True)
        unhook.write_text("unset UV_PROJECT_ENVIRONMENT\n")
    elif virtual_env:
        activate = Path(virtual_env) / "bin" / "activate"
        if activate.exists() and export_line not in activate.read_text():
            with activate.open("a") as f:
                f.write("\n" + export_line + "\n")


create_dotenv()
install_path_hook()
setup()
