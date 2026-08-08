"""Lightning Trainer arguments that are not otherwise part of the experiment."""

from lead.config.node import ConfigNode


class LightningConfig(ConfigNode):
    """How the Lightning Trainer runs a training loop.

    These map onto Trainer arguments one for one and carry Lightning's own
    defaults. Knobs that describe the model rather than the loop, such as the
    precision or whether batch norm syncs, live in
    :class:`~lead.config.training.optimization_config.OptimizationConfig`
    because evaluation reads them too.
    """

    # --- Hardware ---
    # Device type to train on: "auto" takes a GPU where there is one, else the CPU.
    accelerator: str = "auto"

    # Devices to train on, in Lightning's spelling: "auto", "-1" for all, a count
    # ("2"), or specific ids ("0,1"). Under Slurm this must match the task count.
    devices: str = "auto"

    # Nodes of a multi-node job; the launcher decides how many tasks each runs.
    num_nodes: int = 1

    # Distribution strategy; "auto" picks DDP when more than one device is used.
    strategy: str = "auto"

    # The three below configure the DDP strategy "auto" builds; a named strategy
    # ignores them.
    # If true synchronize buffers across ranks every forward, which the running
    # statistics do not need.
    ddp_broadcast_buffers: bool = False

    # If true promise the graph never changes, letting DDP reorder its buckets
    # once. A model with control flow that skips parameters must turn this off.
    ddp_static_graph: bool = True

    # If true have gradients point into DDP's communication buckets. This changes
    # the gradient strides, which makes the attention backward materialize a
    # matching copy.
    ddp_gradient_as_bucket_view: bool = True

    # Seconds a collective may block before the watchdog fails it, which is how
    # long the surviving ranks wait for one that died. Torch's own default.
    ddp_timeout_seconds: int = 1800

    # --- Loop ---
    # Optimizer steps before training stops; -1 runs the configured epochs out.
    max_steps: int = -1

    # Fraction of the training batches an epoch uses; 1.0 is all of them.
    limit_train_batches: float = 1.0

    # Batches whose gradients accumulate into one optimizer step, trading steps
    # for a larger effective batch at the same memory.
    accumulate_grad_batches: int = 1

    # Validation batches run before training starts; there is no validation loop.
    num_sanity_val_steps: int = 0

    # --- Gradients ---
    # Gradient norm to clip to; 0 disables clipping.
    gradient_clip_val: float = 0.0

    # What ``gradient_clip_val`` measures: "norm" or "value".
    gradient_clip_algorithm: str = "norm"

    # --- Performance ---
    # If true let cuDNN benchmark its algorithms, which pays off once the input
    # shapes stop changing.
    benchmark: bool = True

    # If true restrict every op to a deterministic implementation, which costs
    # speed and makes a run reproducible.
    deterministic: bool = False

    # --- Output ---
    # If true draw the per-epoch progress bar, one line per step in a job log.
    enable_progress_bar: bool = True

    # If true print the layer/parameter table before training.
    enable_model_summary: bool = True

    # Per-step timing report to print at the end: "simple", "advanced", or empty
    # for none.
    profiler: str = ""

    # --- Debugging ---
    # Batches to run one loop of as a smoke test, skipping loggers and
    # checkpoints; 0 trains normally.
    fast_dev_run: int = 0

    # Batches to train and validate on repeatedly to prove the model can fit
    # them; 0 trains normally.
    overfit_batches: float = 0.0

    # If true have autograd locate the op that produced a NaN, at a large cost.
    detect_anomaly: bool = False
