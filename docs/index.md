# LEAD

LEAD produces an as-generic-as-possible driving dataset, plus the generic tooling
around it.

Most CARLA codebases are written around one model; LEAD is written around the data,
in the [Py123D](https://github.com/kesai-labs/py123d) standard format.

Reading the data is model-agnostic, while featurization and label building are
entirely the policy's responsibility. This split is enforced by import contracts,
not by convention.

## Basic documents

To understand the data layout and how the data is processed:

- [Data access](data_access.md): understand the data layout.

To understand how TransFuser is trained, and how to implement your own policy:

- [Training](training.md): building the cache, pretraining, post-training, config overrides.
- [Add your own policy](add_a_policy.md): a walkthrough of the smallest policy in the repo.

To drive a benchmark with a trained policy:

- [Evaluation](eval.md): closed-loop runs, scaling them on SLURM, aggregating scores.

To generate your own data, see:

- [Data collection](data_generation.md): running the expert to generate a dataset, changing the sensor rig, adding a modality offline.

## Further documents

- [Architecture](architecture.md): understand the design of the repository and how data is loaded and processed.
- [Training speed](speed.md): options for faster training.
