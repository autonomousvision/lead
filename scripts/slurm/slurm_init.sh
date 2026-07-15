#!/usr/bin/bash
# Sourced Slurm bootstrap: sets up the per-script output dir and resubmits the script to Slurm.
# Source it after any #SBATCH directives; it resolves itself to an absolute path, then cds to
# the repo root.

set -e
shopt -s globstar

LEAD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" # Repo root: this file lives at <root>/scripts/slurm/
export LEAD_ROOT

# Slurm runs a spooled copy of the script on the job side, so $0 is unreliable there; the real
# script path is computed once on the submit side and inherited by the job via --export=ALL.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
	LEAD_SCRIPT="$(realpath "$0")"
	export LEAD_SCRIPT

	SCRIPT_NAME=$(basename "$LEAD_SCRIPT" ".sh") # Script name
	export SCRIPT_NAME
	SLURM_JOB_DATE="$(date +"%y%m%d_%H%M%S")"
	export SLURM_JOB_DATE

	# Per-script output dir: the script's own path relative to LEAD_ROOT, plus its name and a timestamp.
	rel_dir="$(realpath --relative-to="$LEAD_ROOT" "$(dirname "$LEAD_SCRIPT")")"
	export OUTPUT_DIR="$LEAD_ROOT/outputs/$rel_dir/$SCRIPT_NAME/$SLURM_JOB_DATE"
fi

# Every script in this repo (this one included) uses paths relative to the repo root
# (scripts/..., src/lead/..., 3rd_party/...), so run as if invoked from there.
cd "$LEAD_ROOT"

# Repo-level tunables (SLURM_PARTITION, SLURM_EMAIL_USER, ...), re-read live on every job.
if [[ -f "$LEAD_ROOT/.env" ]]; then
	set -a
	source "$LEAD_ROOT/.env"
	set +a
fi

# Resubmit this script and its #SBATCH directives to Slurm, only when not already inside a job
# and sbatch is available; otherwise a no-op and the caller keeps running.
# Usage: submit [output_dir] [job_name]
function submit() {
	local dir="${1:-$OUTPUT_DIR}"
	local name="${2:-$SCRIPT_NAME}"
	mkdir -p "$dir"
	if [[ -z "${SLURM_JOB_ID:-}" ]] && command -v sbatch >/dev/null 2>&1; then
		exec sbatch \
			--job-name="${name}" \
			--partition="${SLURM_PARTITION:?Set SLURM_PARTITION in .env (see .env.example)}" \
			--mail-type="${SLURM_MAIL_TYPE:-FAIL,END}" \
			--mail-user="${SLURM_EMAIL_USER:-$USER}" \
			--export=ALL,SLURM_JOB_DATE="${SLURM_JOB_DATE}" \
			--output="${dir}/stdout_%j.log" \
			--error="${dir}/stderr_%j.log" \
			"$LEAD_SCRIPT"
	fi
}

if [[ -z "${LEAD_SLURM_MANUAL_SUBMIT:-}" ]]; then
	submit
fi

# Runs the Python file with the same path/name as this shell script (foo.sh -> foo.py),
# forwarding any arguments to it verbatim.
# Usage: run_python_counterpart [args-for-python-script...]
function run_python_counterpart() {
	python3 "${LEAD_SCRIPT%.sh}.py" "$@"
}

# Everything below runs on the job side or on a local run without sbatch; the submit-side pass
# above already exited via exec.

# On a real Slurm job, print job info from the compute node.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
	scontrol show job "${SLURM_JOB_ID}"
fi

echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "Using env: ${CONDA_DEFAULT_ENV:-none}"
echo "Using python at: $(which python)"
