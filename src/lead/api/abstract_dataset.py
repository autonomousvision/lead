"""The shared dataset over a policy's sample parts, with the per-part cache.

A policy's dataset subclasses :class:`AbstractPolicyDataset` and supplies only
model-specific pieces: its sample parts, its batch-item class, and an optional
output post-processing hook. Everything generic — resolving which parts the
cache serves, the cached-vs-computed dispatch, the cache-build interface —
lives here once.
"""

import abc
import logging
import typing
from collections.abc import Sized
from typing import ClassVar

from py123d.common.io.camera.jpeg_camera_io import set_jpeg_decoder
from py123d.common.io.camera.png_camera_io import set_png_decoder
from torch.utils.data import Dataset

from lead.api.py123d_log_api import NORMAL_SENSOR_VIEW, PERTURBATED_SENSOR_VIEW
from lead.api.scene_loading_spec import SceneLoadingSpec
from lead.api.training_sample import PolicyTrainingSample, SamplePart
from lead.cache.cache_store import CacheStoreReader, SampleCacheAddress
from lead.common.image_decoders import decode_jpeg, decode_png
from lead.config import LeadConfig
from lead.config.policy.abstract_policy_config import AbstractPolicyConfig
from lead.log_reader.scene_loader import SceneLoader

LOG = logging.getLogger(__name__)


class SizedDataset(Dataset, Sized, abc.ABC):
    """A map-style :class:`Dataset` reporting its length, as training requires."""


class AbstractPolicyDataset(SizedDataset, abc.ABC):
    """Training dataset over a policy's sample parts, with a per-part opt-in cache.

    Parts whose declared outputs are all in the cache store are served from it;
    the rest are computed from scene data read with exactly the union of their
    loading specs. Which parts fall on which side is resolved once, at
    construction: the store's contents do not change while training runs. A
    single sample the store does not hold is computed in full, so a store built
    under a different scene enumeration stays usable.
    """

    # The typed sample this dataset yields; set by the subclass.
    sample_class: ClassVar[type[PolicyTrainingSample]]

    def __init__(
        self,
        lead_config: LeadConfig,
        scene_loader: SceneLoader,
        policy_config: AbstractPolicyConfig,
    ) -> None:
        """Construct the dataset, opening the cache store if configured.

        Args:
            lead_config: Root config tree.
            scene_loader: Loader over the scenes this policy trains on, built
                by the policy (see ``AbstractPolicy.build_dataset``). Which
                scenes to read and how is the policy's decision; the dataset
                only turns what comes back into tensors.
            policy_config: Config section of the policy this dataset serves,
                naming the cache store it may read.
        """
        self.lead_config = lead_config
        self.policy_config = policy_config
        self._scene_loader = scene_loader
        self._sample_parts: dict[str, SamplePart] = self.get_sample_parts()

        # Whether this process already warned about a sample the store lacks.
        self._warned_about_an_uncached_sample = False
        self._cache_reader: CacheStoreReader | None = None
        # A policy with no cacheable part has no store to open; reading live
        # is not a configuration error then.
        if lead_config.training.data.read_from_cache_store and any(
            part.caches for part in self._sample_parts.values()
        ):
            self._cache_reader = CacheStoreReader(
                policy_config.cache_store_root,
                self.cache_finger_print,
            )
        served_part_names = self._part_names_the_store_serves()
        self._cached_part_names = tuple(
            part_name
            for part_name in self._sample_parts
            if part_name in served_part_names
        )
        self._computed_part_names = tuple(
            part_name
            for part_name in self._sample_parts
            if part_name not in served_part_names
        )

    # --- The model-specific parts a subclass supplies ---
    @abc.abstractmethod
    def get_sample_parts(self) -> dict[str, SamplePart]:
        """Build the policy's sample parts, keyed by part name."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def cache_finger_print(self) -> dict[str, str]:
        """Named config values this dataset's cacheable parts read, stringified."""
        raise NotImplementedError

    def prepare_read(self) -> None:
        """Per-read worker preparation, run in every data loader worker.

        Injects our image decoders into py123d: both decode faster than the
        OpenCV path py123d ships with, which wraps the same libraries but
        adds a colour conversion pass.
        """
        set_jpeg_decoder(decode_jpeg)
        set_png_decoder(decode_png)

    def postprocess_outputs(
        self,
        outputs: dict[str, typing.Any],
        scene_index: int,
        loading_seconds: float,
    ) -> dict[str, typing.Any]:
        """Hook over the merged part outputs of one sample; attaches the timing.

        Runs once every part is in (cached and computed), so randomness such as
        augmentation belongs here, never in a part. Every sample carries
        ``loading_time``, which is what the batch size is counted from.

        Args:
            outputs: The merged output dict of every part of one sample.
            scene_index: Sample index into the loader's scene enumeration.
            loading_seconds: Wall-clock seconds this sample took to assemble.

        Returns:
            The outputs handed to the batch item.
        """
        del scene_index
        outputs["loading_time"] = loading_seconds
        return outputs

    # --- Generic machinery ---
    @property
    def collate_fn(self) -> typing.Callable:
        """The collate the training DataLoader uses for this dataset."""
        return self.sample_class.collate

    def __len__(self) -> int:
        return len(self._scene_loader)

    def __getitem__(self, scene_index: int) -> PolicyTrainingSample:
        import time

        self.prepare_read()
        start_loading_time = time.time()

        sensor_view = self._scene_loader.draw_sensor_view(scene_index)
        # Empty when no store is configured, when this draw landed on a sensor
        # view the store does not hold, and when the store does not hold this
        # sample; in every case each part computes.
        cached_part_outputs = self._read_cached_part_outputs(scene_index, sensor_view)
        part_names_to_build = (
            self._computed_part_names
            if cached_part_outputs
            else tuple(self._sample_parts)
        )
        outputs = cached_part_outputs | self._assemble(
            part_names_to_build,
            scene_index,
            use_perturbated_view=sensor_view == PERTURBATED_SENSOR_VIEW,
        )
        outputs = self.postprocess_outputs(
            outputs,
            scene_index,
            time.time() - start_loading_time,
        )
        return self.sample_class.from_outputs(outputs)

    def _assemble(
        self,
        part_names: tuple[str, ...],
        scene_index: int,
        *,
        use_perturbated_view: bool,
    ) -> dict[str, typing.Any]:
        """Read one scene for these parts and run their builders over it.

        Args:
            part_names: Parts to build, in declaration order.
            scene_index: Sample index into the loader's scene enumeration.
            use_perturbated_view: Whether to read the perturbated rig view.

        Returns:
            The merged outputs of those parts.
        """
        scene_data = self._scene_loader.read(
            scene_index,
            use_perturbated_view,
            SceneLoadingSpec.union(
                self._sample_parts[part_name].reads for part_name in part_names
            ),
        )
        part_outputs: dict[str, typing.Any] = {}
        for part_name in part_names:
            part_outputs.update(self._sample_parts[part_name].builds(scene_data))
        return part_outputs

    def _read_cached_part_outputs(
        self,
        scene_index: int,
        sensor_view: str,
    ) -> dict[str, typing.Any]:
        """Read every output of the store-served parts of one sample.

        Args:
            scene_index: Sample index into the loader's scene enumeration.
            sensor_view: The view this draw landed on.

        Returns:
            The stored outputs by tensor name; empty when this draw's view is
            not in the store, when the store does not hold the sample, or when
            the store serves no part at all.
        """
        if not self._cached_part_names or self._cache_reader is None:
            return {}
        if not self._cache_reader.has_sensor_view(sensor_view):
            return {}
        log_name, sample_key = self.sample_cache_address(scene_index)
        tensor_names = [
            tensor_name
            for part_name in self._cached_part_names
            for tensor_name in self._sample_parts[part_name].caches
        ]
        outputs = self._cache_reader.read(
            sensor_view,
            log_name,
            sample_key,
            tensor_names,
        )
        if outputs is None:
            # A store built under a different scene enumeration covers most of
            # this run's samples but not all; the rest compute live. Warning
            # once keeps a store that covers almost nothing from going unnoticed.
            if not self._warned_about_an_uncached_sample:
                self._warned_about_an_uncached_sample = True
                LOG.warning(
                    f"Sample {sample_key} of log {log_name} is not in the "
                    f"{sensor_view} store under {self.policy_config.cache_store_root}; "
                    f"computing it live. Rebuild the store with "
                    f"lead.training.build_cache if this is not rare.",
                )
            return {}
        return outputs

    def _part_names_the_store_serves(self) -> set[str]:
        """Read side: the parts this store already holds, so reads can skip them.

        A part counts only when *every* output it declares is in the store; a
        half-written part has to be computed like any other.

        Returns:
            The names of the parts the store can serve.
        """
        if self._cache_reader is None:
            return set()
        stored_tensor_names = set(self._cache_reader.tensor_names)
        return {
            part_name
            for part_name, part in self._sample_parts.items()
            if part.caches and set(part.caches) <= stored_tensor_names
        }

    def _part_names_to_write(self) -> tuple[str, ...]:
        """Write side: the parts a cache build computes, in declaration order.

        Every part that declares outputs, whatever any store currently holds.

        Returns:
            The names of the parts a build writes.
        """
        return tuple(
            part_name for part_name, part in self._sample_parts.items() if part.caches
        )

    # --- Cache-building interface (see lead.training.build_cache) ---
    def cached_tensor_codecs(self) -> dict[str, typing.Any]:
        """Which sample tensors the store holds and the codec each is encoded with.

        Merged from the cacheable parts of this dataset, so the manifest never
        declares tensors the build cannot write; a part gated off by config
        computes live.

        Returns:
            The tensor-name to codec-spec map.
        """
        return {
            tensor_name: codec
            for part in self._sample_parts.values()
            for tensor_name, codec in part.caches.items()
        }

    def sample_sensor_views(self, scene_index: int) -> list[str]:
        """Sensor views a sample is cached into: normal, plus perturbated where recorded.

        Args:
            scene_index: Sample index into the loader's scene enumeration.

        Returns:
            The view subdir names.
        """
        views = [NORMAL_SENSOR_VIEW]
        if self._scene_loader.has_perturbated_sensor_view(scene_index):
            views.append(PERTURBATED_SENSOR_VIEW)
        return views

    def sample_log_name(self, scene_index: int) -> str:
        """The log a sample belongs to; cheap, unlike the full cache address.

        Args:
            scene_index: Sample index into the loader's scene enumeration.

        Returns:
            The log name.
        """
        return self._scene_loader.scenes[scene_index].log_name

    def sample_cache_address(self, scene_index: int) -> SampleCacheAddress:
        """The store address a sample's tensors live under, the same in every view.

        Args:
            scene_index: Sample index into the loader's scene enumeration.

        Returns:
            The log the sample belongs to and its per-log sample key.
        """
        scene = self._scene_loader.scenes[scene_index]
        return SampleCacheAddress(
            log_name=scene.log_name,
            sample_key=str(scene.get_timestamp_at_iteration(0).time_us),
        )

    def precompute_cached_tensors(
        self,
        scene_index: int,
        sensor_view: str,
    ) -> dict[str, typing.Any]:
        """Assemble every cacheable part of one ``(sample, view)``.

        Args:
            scene_index: Sample index into the loader's scene enumeration.
            sensor_view: One of :meth:`sample_sensor_views`.

        Returns:
            The cacheable tensors by name, merged across parts.
        """
        self.prepare_read()
        return self._assemble(
            self._part_names_to_write(),
            scene_index,
            use_perturbated_view=sensor_view == PERTURBATED_SENSOR_VIEW,
        )
