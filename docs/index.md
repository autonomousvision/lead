# LEAD

LEAD produces an as-generic-as-possible driving dataset, plus the generic tooling
around it. Policy learning is only one application of the dataset — TransFuser is,
in turn, only one possible policy.

Every application reads the dataset through one py123d-native, model-agnostic
generic loader (`lead.dataloader.py123d_data_loader`), so all downstream consumers
see the same unified format:

```
                              ┌─► World model
                              ├─► Video generator
py123d logs ─► generic loader ├─► RL on logs-replay
                              ├─► Perception model
                              └─► Policy learning
```

Policy learning — the application LEAD ships tooling for — adds two more layers on
top of the generic loader: a model-specific loader that turns generic output into
one policy's tensors, and a policy that consumes those tensors. Offline training and
closed-loop evaluation both flow through the same model-specific loader, so they
can't drift apart:

```
offline:      py123d logs    ─► generic loader  ─┐
                                                 ├─► model-specific loader ─► tensors ─► policy
closed-loop:  CARLA sensors  ─► inference API   ─┘
```

Each policy gets its own model-specific loader, e.g. `lead.policy.transfuser.dataloader`
for TransFuser. Policies are swappable: `lead.training.train` only depends on the
`AbstractPolicy` contract (`build_dataset`, `build_features`, `compute_loss`, …), not
on any specific policy —

```
train.py ─► AbstractPolicy ─┬─► TransfuserPolicy
                            └─► <your policy, loader, visualizer, loss functions>
```

— so adding a new policy means implementing that contract, not modifying the training loop.

## Further guides

- [Data access](data_access.md): reading py123d logs directly, or via `Py123DDataLoader`.
- [Setup for development](setup.md): environment, CARLA, tests.
- [Data collection](data_generation.md): running the expert to generate a dataset.
- [Training](training.md): Work in progress.
- [Evaluation](eval.md): Work in progress.
