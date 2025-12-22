import abc
import logging
import lzma
import os
import pathlib
import pickle
import random
import sys

import ujson
from beartype import beartype
from tqdm import tqdm

from lead.training.config_training import TrainingConfig
from lead.training.data_loader.buckets.bucket import Bucket

LOG = logging.getLogger(__name__)


class AbstractBucketCollection(abc.ABC):
    buckets: list[Bucket] = []

    @beartype
    def __init__(self, root: str | list[str], config: TrainingConfig):
        if isinstance(root, str):
            root = [str(p) for p in pathlib.Path(root).iterdir() if p.is_dir()]
        self.root = sorted(root)
        if config.randomize_route_order:
            random.shuffle(self.root)
        self.config = config
        self.total_routes = 0
        self.trainable_routes = 0
        self.all_frames = 0
        self.trainable_frames = 0

        # Try to load from cache first
        if self._does_cache_exist() and not self.config.force_rebuild_bucket:
            LOG.info(f"Loading collection from cache: {self.cache_file_path()}")
            self._load_from_cache()
        else:
            LOG.info("Building collection from scratch...")
            self._build_buckets()
            for bucket in self.buckets:
                bucket.finalize()
            if not self.config.visualize_dataset:
                self._save_to_cache()

    @beartype
    def _does_cache_exist(self) -> bool:
        """Check if the cache file exists."""
        return os.path.exists(self.cache_file_path())

    @beartype
    def iter_root(self):
        """Iterate over the root directories and yield route directories.

        Yields:
            str: Path to every route directory.
        """
        for scenario_dir in tqdm(
            self.root,
            file=sys.stdout,
            desc="Iterating scenario types",
        ):
            routes = os.listdir(scenario_dir)
            LOG.info(f"Found {len(routes)} routes in scenario {scenario_dir}")
            if self.config.randomize_route_order:
                random.shuffle(routes)
            else:
                routes = sorted(routes)
            for route in routes:
                self.total_routes += 1
                route_dir = os.path.join(scenario_dir, route)
                yield route_dir

    @beartype
    def iter_route(self, route_path: str):
        """Iterate over the frames in a route directory.

        Args:
            route_dir: Path to the route directory.

        Yields:
            int: Frame indices in the route directory.
        """
        lidar_dir = route_path + "/bboxes"
        num_seq = len(os.listdir(lidar_dir))

        for seq in range(self.config.skip_first, num_seq - self.config.skip_last):
            self.all_frames += 1
            yield seq

    # Abstract methods that subclasses MUST implement
    @abc.abstractmethod
    def _build_buckets(self):
        """Build the curriculum from scratch - to be implemented by subclasses"""
        pass

    @abc.abstractmethod
    def cache_file_path(self):
        """Return path for cache file - to be implemented by subclasses"""
        pass

    @abc.abstractmethod
    def buckets_mixture_per_epoch(self, epoch) -> dict[int, float]:
        pass

    def __len__(self):
        return len(self.buckets)

    def __getitem__(self, index) -> Bucket:
        return self.buckets[index]

    def _load_from_cache(self):
        """Load the curriculum from cache file."""
        LOG.info(f"Loading bucket collection from cache: {self.cache_file_path()}")
        with lzma.open(self.cache_file_path(), "rb") as f:
            cached_data = pickle.load(f)

        self.buckets = cached_data["buckets"]
        self.total_routes = cached_data["total_routes"]
        self.trainable_routes = cached_data["trainable_routes"]
        self.all_frames = cached_data["all_frames"]
        self.trainable_frames = cached_data["trainable_frames"]
        LOG.info("Bucket collection loaded from cache.")

    def _save_to_cache(self):
        """Save the curriculum to cache file."""
        if not self.config.visualize_dataset:
            LOG.info(f"Saving bucket collection to cache: {self.cache_file_path()}")
            cache_data = {
                "buckets": self.buckets,
                "total_routes": self.total_routes,
                "trainable_routes": self.trainable_routes,
                "all_frames": self.all_frames,
                "trainable_frames": self.trainable_frames,
            }

            # Create cache directory if it doesn't exist
            os.makedirs(os.path.dirname(self.cache_file_path()), exist_ok=True)

            with lzma.open(self.cache_file_path(), "wb") as f:
                pickle.dump(cache_data, f)
            LOG.info("Bucket collection saved to cache.")


@beartype
def route_failed(route_path: str) -> bool:
    """Check if a route has failed or should be skipped.

    Args:
        route_path: Directory of the route.

    Returns:
        bool: True if the route has failed or should be skipped, False otherwise.
    """
    if not os.path.isfile(route_path + "/results.json"):
        LOG.info(f"\tSkipping route {route_path} due to missing results file.")
        return True

    if not os.path.isfile(route_path + "/metas/0000.pkl"):
        LOG.info(f"\tSkipping route {route_path} due to missing meta files right now.")
        return True

    # We skip data where the expert did not achieve perfect driving score (except for min speed infractions)
    with open(route_path + "/results.json", encoding="utf-8") as f:
        results_route = ujson.load(f)
    infraction = True
    if results_route["scores"]["score_composed"] == 100.0:
        infraction = False
    if results_route["scores"]["score_composed"] < 100.0 and results_route["num_infractions"] == len(
        results_route["infractions"]["min_speed_infractions"]
    ):
        infraction = False
    if results_route["status"] == "Completed" and results_route["num_infractions"] == (
        len(results_route["infractions"]["min_speed_infractions"]) + len(results_route["infractions"]["outside_route_lanes"])
    ):  # We ignore minor out of lanes if we still complete the route
        infraction = False
    if results_route["status"] == "Perfect":
        infraction = False

    agent_not_setup = results_route["status"] == "Failed - Agent couldn't be set up"
    failed = results_route["status"] == "Failed"
    simulation_crashed = results_route["status"] == "Failed - Simulation crashed"
    agent_crashed = results_route["status"] == "Failed - Agent crashed"

    if infraction or agent_not_setup or failed or simulation_crashed or agent_crashed:
        LOG.info(
            f"\tSkipping {route_path}: results {results_route['status']}, scores {results_route['scores']['score_composed']}."
        )
        return True
    return False


@beartype
def route_not_finished(route_path: str) -> bool:
    if not os.path.isfile(route_path + "/results.json"):
        LOG.info(f"\tSkipping route {route_path} due to missing results file.")
        return True
    return False


@beartype
def route_completed_but_fail(route_path: str) -> bool:
    if not os.path.isfile(route_path + "/results.json"):
        LOG.info(f"\tSkipping route {route_path} due to missing results file.")
        return False

    if not os.path.isfile(route_path + "/metas/0000.pkl"):
        LOG.info(f"\tSkipping route {route_path} due to missing meta files right now.")
        return False

    with open(route_path + "/results.json", encoding="utf-8") as f:
        results_route = ujson.load(f)

    agent_not_setup = results_route["status"] == "Failed - Agent couldn't be set up"
    simulation_crashed = results_route["status"] == "Failed - Simulation crashed"
    agent_crashed = results_route["status"] == "Failed - Agent crashed"

    if agent_not_setup or simulation_crashed or agent_crashed:
        LOG.info(
            f"\tSkipping {route_path}: results {results_route['status']}, scores {results_route['scores']['score_composed']}."
        )
        return False
    if not os.path.exists(route_path + "/metas"):
        LOG.info(f"\tSkipping route {route_path} due to missing metas data.")
        return False
    if results_route["scores"]["score_composed"] == 100.0:
        LOG.info(f"\tSkipping route {route_path} due to perfect score.")
        return False
    if (
        results_route["status"] == "Completed"
        and results_route["scores"]["score_composed"] < 100.0
        and results_route["num_infractions"] == len(results_route["infractions"]["min_speed_infractions"])
    ):
        LOG.info(f"\tSkipping route {route_path} due to only min speed infractions.")
        return False
    if results_route["status"] == "Completed" and results_route["num_infractions"] == (
        len(results_route["infractions"]["min_speed_infractions"]) + len(results_route["infractions"]["outside_route_lanes"])
    ):
        LOG.info(f"\tSkipping route {route_path} due to only min speed and outside lane infractions.")
        return False

    return True
