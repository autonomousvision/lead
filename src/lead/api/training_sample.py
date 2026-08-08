"""The training sample of a policy: the parts that fill it, and the batch it collates into.

The two sides of :func:`~torch.utils.data.default_collate` are different types: a
sample holds numpy arrays with no batch axis, a batch holds tensors with one — and
a ``str`` field becomes ``list[str]``, a ``float`` field a float64 tensor. A policy
declares both, one as a dataclass and one as a ``TypedDict``.
"""

import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields

from torch.utils.data import default_collate

from lead.api.scene_data import SceneData
from lead.api.scene_loading_spec import SceneLoadingSpec


@dataclass
class PolicyTrainingSample:
    """One unbatched sample of a policy's dataset, with typed principal fields.

    Subclasses are dataclasses declaring one field per principal array, plus the
    inherited ``extras``. The jaxtyping import hook instruments dataclasses, so
    every field's dtype and shape is checked as the sample is built.
    """

    # Everything the parts emit beyond the principal fields (meta lifts,
    # identity fields, diagnostics).
    extras: dict[str, typing.Any] = field(default_factory=dict)

    @classmethod
    def from_outputs(
        cls,
        outputs: dict[str, typing.Any],
    ) -> "PolicyTrainingSample":
        """Sort the merged part outputs into typed fields and extras.

        Args:
            outputs: The merged output dict of every part of one sample.

        Returns:
            The typed sample.
        """
        field_names = {f.name for f in fields(cls) if f.name != "extras"}
        principal = {name: outputs[name] for name in field_names if name in outputs}
        extras = {
            name: value for name, value in outputs.items() if name not in field_names
        }
        return cls(**principal, extras=extras)

    def to_batch_dict(self) -> dict[str, typing.Any]:
        """Flatten into the model's key-value layout, dropping unset fields.

        Returns:
            The sample as one flat dict.
        """
        flat: dict[str, typing.Any] = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "extras" and getattr(self, f.name) is not None
        }
        flat.update(self.extras)
        return flat

    @classmethod
    def collate(
        cls,
        samples: Sequence["PolicyTrainingSample"],
    ) -> dict[str, typing.Any]:
        """Collate samples into one training batch dict.

        Args:
            samples: The samples of one batch.

        Returns:
            The batch, with every field stacked along a new leading dimension.
        """
        return default_collate([sample.to_batch_dict() for sample in samples])


@dataclass(frozen=True)
class SamplePart:
    """One part of a policy's sample: what it reads, how it builds, how it caches.

    ``builds`` must be deterministic in the scene data; randomness belongs to
    augmentation, which runs once every part is in.
    """

    # Which modalities of the scene ``builds`` needs.
    reads: SceneLoadingSpec
    # The derivation: one scene into named arrays/values of the sample.
    builds: Callable[[SceneData], Mapping[str, typing.Any]]
    # The outputs a prebuilt cache store may hold, each with its codec spec.
    # Empty means the part always computes; when every output is stored, the
    # dataset skips both the computation and this part's scene reads.
    caches: Mapping[str, str | dict] = field(default_factory=dict)
