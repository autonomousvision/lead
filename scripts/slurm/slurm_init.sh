#!/usr/bin/bash
# Generic Slurm bootstrap, shared by every script this repo submits to Slurm. Source it after
# any #SBATCH directives — sourcing alone is enough, it resubmits the script to Slurm itself:
#   source scripts/slurm/slurm_init.sh
# Works no matter what directory you run the script from: it resolves itself to an absolute
# path before doing anything else, then cds to the repo root (every script in this repo assumes
# paths relative to there).
#
# Training/evaluation experiment scripts (scripts/slurm/experiments/**) should source
# scripts/slurm/slurm_experiment_init.sh instead — it sources this file and adds the
# experiment-run bookkeeping (EXPERIMENT_RUN_ID, TRAINING_OUTPUT_DIR, train(), evaluate(), ...)
# on top.
#
# Mirrors the pretraining-workspace slurm_helper.sh pattern: resubmission (exec sbatch ... "$0")
# happens automatically as soon as this file is sourced. The one exception is
# slurm_experiment_init.sh, which sets LEAD_SLURM_MANUAL_SUBMIT before sourcing this file: its
# evaluate_* drivers must keep running on the login node (they launch a screen session that
# orchestrates its own per-route sbatch jobs) instead of becoming a batch job themselves, so
# submission there is an explicit `submit` call made by `train`, not automatic on sourcing.

set -e
shopt -s globstar

LEAD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" # Repo root: this file lives at <root>/scripts/slurm/
export LEAD_ROOT

# Once a job actually starts, Slurm runs a spooled copy of the submitted script (e.g.
# /var/spool/slurmd/.../slurm_script), not the original file — so on the job side, $0 no longer
# points at the real script and can't be used to recompute these. Compute them once, on the
# submit side (or a plain local run without Slurm), and let the job side inherit them via
# `submit`'s --export=ALL instead of recomputing from a now-unreliable $0.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
	LEAD_SCRIPT="$(realpath "$0")"
	export LEAD_SCRIPT

	SCRIPT_NAME=$(basename "$LEAD_SCRIPT" ".sh") # Script name
	export SCRIPT_NAME
	SLURM_JOB_DATE="$(date +"%y%m%d_%H%M%S")"
	export SLURM_JOB_DATE

	# Generic per-script output dir, mirroring slurm_helper.sh's OUTPUT_DIR: same path (relative
	# to LEAD_ROOT) as the script itself, plus its name and a timestamp. Experiment scripts get
	# a nicer, experiment-scoped TRAINING_OUTPUT_DIR/EVALUATION_OUTPUT_DIR instead (see
	# slurm_experiment_init.sh) but OUTPUT_DIR is still set here for anything that wants it.
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

# Resubmit this script (its resolved path, see LEAD_SCRIPT above), with its #SBATCH directives,
# to Slurm — only if we're not already inside a job and sbatch is available. On the job side, or
# when sbatch isn't available at all, it's a no-op and the caller just keeps running
# locally/in the current job.
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

# Runs the Python file with the same path/name as this shell script (its "counterpart"), e.g.
# some/dir/foo.sh -> some/dir/foo.py. (Pure-Python jobs don't need a .sh wrapper at all — they
# can call slurm_init() from scripts/slurm/slurm_init.py instead.)
# Any arguments are forwarded to it verbatim.
# Usage: run_python_counterpart [args-for-python-script...]
function run_python_counterpart() {
	python3 "${LEAD_SCRIPT%.sh}.py" "$@"
}

# Everything below only runs on the job side (or a local run without sbatch, or a caller that
# opted into manual submission) — the submit-side pass above already exited via exec.

# Job side (real Slurm job): print job info once we're actually on the compute node.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
	scontrol show job "${SLURM_JOB_ID}"
fi

echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "Using env: ${CONDA_DEFAULT_ENV:-none}"
echo "Using python at: $(which python)"
