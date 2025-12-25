# Tutorial Jupyter Notebooks

The notebooks were tested on VSCode. This [setting](https://github.com/autonomousvision/lead/blob/main/.vscode/settings.json) makes sure the notebooks are started in project root.

## Pipline Verification Notebooks

### 1. Inspect Expert's Output

**Notebook:** [notebooks/inspect_expert_output.ipynb](https://github.com/autonomousvision/lead/blob/main/notebooks/inspect_expert_output.ipynb)

Run expert, produce data and verify that everything works.

### 2. Load Pre-trained Model, Data and Example Inference

**Notebook:** [notebooks/carla_offline_inference.ipynb](https://github.com/autonomousvision/lead/blob/main/notebooks/carla_offline_inference.ipynb)

Load model checkpoints. Load data, visualize a random sample, and run offline inference.

## Data Format Explained Interactively

**Notebook:** [notebooks/data_format.ipynb](https://github.com/autonomousvision/lead/blob/main/notebooks/data_format.ipynb)

Understand what the dataloader outputs and how to fit them to your model.

## Debug Closed-Loop Evaluation Interactively

**Notebook:** [notebooks/inspect_sensor_agent_io.ipynb](https://github.com/autonomousvision/lead/blob/main/notebooks/inspect_sensor_agent_io.ipynb)

Debug model inputs/outputs during closed-loop evaluation.
