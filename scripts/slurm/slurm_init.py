"""Python counterpart of slurm_init.sh: lets a Python script submit itself to Slurm.

Declare resources as regular #SBATCH comment lines at the top of the script —
after the shebang, before the docstring (sbatch stops parsing directives at the
first non-comment line) — and call ``slurm_init()`` first thing in ``__main__``:

    #!/usr/bin/env python3
    #SBATCH --time=1-00:00:00
    #SBATCH --cpus-per-task=24
    #SBATCH --mem=128gb
    '''Module docstring.'''
    from slurm.slurm_init import slurm_init

    if __name__ == "__main__":
        slurm_init()
        ...

Run ``python3 the_script.py`` and it submits itself (the .py file is the batch
script; sbatch reads the #SBATCH lines from it). Inside the job, or on a
machine without sbatch, ``slurm_init()`` just sets up OUTPUT_DIR, cds to the
repo root, and returns so the script keeps running.
"""

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from lead.common.dotenv import REPO_ROOT, read_dotenv


def slurm_init(job_name: str | None = None) -> None:
    """Submit the calling script to Slurm, or set up the job-side/local run.

    Mirrors sourcing scripts/slurm/slurm_init.sh: on the submit side (not
    already inside a Slurm job, sbatch available) this execs
    ``sbatch <script> <argv>`` — resource requests come from the script's own
    #SBATCH lines — and does not return. On the job side, or locally when
    sbatch is unavailable, it changes to the repo root, exports OUTPUT_DIR
    (outputs/<script dir>/<script name>/<timestamp>), prints run info, and
    returns.

    Args:
        job_name: Slurm ``--job-name`` (default: the script file name).
    """
    script = Path(sys.argv[0]).resolve()
    os.chdir(REPO_ROOT)

    if "SLURM_JOB_ID" not in os.environ:
        # Submit side (or plain local run): compute the per-script output dir
        # once here; the job side inherits it via --export=ALL (Slurm runs a
        # spooled copy of the script, which can't recompute it from its own
        # path).
        script_name = script.stem
        job_date = datetime.now().strftime("%y%m%d_%H%M%S")
        rel_dir = os.path.relpath(script.parent, REPO_ROOT)
        output_dir = REPO_ROOT / "outputs" / rel_dir / script_name / job_date
        os.environ["LEAD_SCRIPT"] = str(script)
        os.environ["SCRIPT_NAME"] = script_name
        os.environ["SLURM_JOB_DATE"] = job_date
        os.environ["OUTPUT_DIR"] = str(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if shutil.which("sbatch"):
            os.execvp(
                "sbatch",
                [
                    "sbatch",
                    f"--job-name={job_name or script_name}",
                    f"--partition={read_dotenv('SLURM_PARTITION')}",
                    f"--mail-type={read_dotenv('SLURM_MAIL_TYPE', 'FAIL,END')}",
                    f"--mail-user={read_dotenv('SLURM_EMAIL_USER', os.environ['USER'])}",
                    "--export=ALL",
                    f"--chdir={REPO_ROOT}",
                    f"--output={output_dir}/stdout_%j.log",
                    f"--error={output_dir}/stderr_%j.log",
                    str(script),
                    *sys.argv[1:],
                ],
            )
    else:
        # Job side (real Slurm job): print job info now that we're on the node.
        subprocess.run(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"]])

    print(f"OUTPUT_DIR: {os.environ.get('OUTPUT_DIR', '')}")
    print(f"Using env: {os.environ.get('CONDA_DEFAULT_ENV', 'none')}")
    print(f"Using python at: {sys.executable}")
