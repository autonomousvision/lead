"""Configuration class for SLURM evaluation environment variables."""

import os

from lead.common import runtime


class ConfigSlurm:
    """Configuration for SLURM evaluation jobs and environment."""

    @property
    def experiment_run_id(self) -> str:
        """Unique identifier for the experiment run."""
        return os.environ["EXPERIMENT_RUN_ID"]

    @property
    def experiment_name(self) -> str:
        """Name of the experiment."""
        return os.environ["EXPERIMENT_NAME"]

    @property
    def script_name(self) -> str:
        """Name of the evaluation script being run."""
        return os.environ["SCRIPT_NAME"]

    @property
    def evaluation_dataset(self) -> str:
        """Dataset being evaluated (e.g., 'Town13', 'longest6', 'bench2drive')."""
        return os.environ["EVALUATION_DATASET"]

    @property
    def evaluation_output_dir(self) -> str:
        """Base output directory for evaluation results."""
        return os.environ["EVALUATION_OUTPUT_DIR"]

    @property
    def save_path(self) -> str | None:
        """Path where simulator outputs are saved (set per route during execution)."""
        return os.environ.get("SAVE_PATH")

    @property
    def user(self) -> str:
        """Current system user running the job."""
        return os.environ["USER"]

    @property
    def is_tcml(self) -> bool:
        """True if running on TCML cluster."""
        return os.environ.get("TCML") is not None

    @property
    def slurm_job_id(self) -> str | None:
        """Current SLURM job ID if running in a SLURM job."""
        return os.environ.get("SLURM_JOB_ID")

    @property
    def slurm_array_task_id(self) -> str | None:
        """Task ID for SLURM array jobs."""
        return os.environ.get("SLURM_ARRAY_TASK_ID")

    @property
    def merged_results_path(self) -> str:
        """Path to merged evaluation results JSON file."""
        return os.path.join(self.evaluation_output_dir, "merged.json")

    @property
    def route_files_by_dataset(self) -> dict[str, str]:
        """Map dataset names to their route XML files."""
        project_root = runtime.project_root()
        return {
            "Town13": os.path.join(
                project_root,
                "data/routes/Town13/Town13.xml",
            ),
            "longest6": os.path.join(
                project_root,
                "3rd_party/leaderboard/standard/leaderboard/data/longest6.xml",
            ),
            "bench2drive": os.path.join(
                project_root,
                "3rd_party/leaderboard/bench2drive/leaderboard/data/bench2drive220.xml",
            ),
        }

    @property
    def current_route_file(self) -> str:
        """Route file for the current evaluation dataset."""
        routes = self.route_files_by_dataset
        if self.evaluation_dataset in routes:
            return routes[self.evaluation_dataset]
        raise ValueError(f"Unknown EVALUATION_DATASET: {self.evaluation_dataset}")

    def get_env_exports(self) -> str:
        """Generate shell export commands for all current environment variables."""
        return "\n".join(
            [f"export {key}='{value}'" for key, value in os.environ.items()],
        )
