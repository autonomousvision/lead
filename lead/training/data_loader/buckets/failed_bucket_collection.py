"""
Perception curriculum for pretraining on CARLA datasets. Just load everything, no filtering, no skipping failed routes.
"""

import os

from lead.training.config_training import TrainingConfig
from lead.training.data_loader.buckets.abstract_bucket_collection import (
    AbstractBucketCollection,
    route_completed_but_fail,
    route_not_finished,
)
from lead.training.data_loader.buckets.bucket import Bucket


class FailedBucketCollection(AbstractBucketCollection):
    def __init__(self, root: str | list[str], config: TrainingConfig):
        self.buckets = [Bucket(config)]
        super().__init__(root, config)
        print("Using failed bucket collection")

    def _build_buckets(self):
        """Build bucket collection from scratch"""
        for route_dir in self.iter_root():
            if route_not_finished(route_dir):
                print(f"Skipping unfinished route {route_dir}")
                continue
            if route_completed_but_fail(route_dir):
                self.trainable_routes += 1
                for seq in self.iter_route(route_dir):
                    self.buckets[0].add(route_dir, seq)
                    self.trainable_frames += 1

    def cache_file_path(self):
        """Return path for cache file"""
        return os.path.join(self.config.bucket_collection_path, "failed_buckets.gz")

    def buckets_mixture_per_epoch(self, _):
        return {0: 1.0}
