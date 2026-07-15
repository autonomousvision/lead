"""Convert all bundled CARLA OpenDRIVE maps to 123D Arrow maps.

Run once before launching a job array.
"""

import argparse

from py123d.api.map.arrow.arrow_map_writer import ArrowMapWriter
from py123d.common.runtime.dataset_paths import get_dataset_paths
from py123d.datatypes import MapMetadata
from py123d.parser.opendrive.opendrive_map_parser import iter_xodr_map_objects

from lead.expert.logs_writing.py123d_logging import CARLA_MAPS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="carla",
        help="py123d dataset name, must match lead_config.expert.data_collection.py123d_dataset",
    )
    args = parser.parse_args()

    dataset_paths = get_dataset_paths()
    assert dataset_paths.py123d_maps_root is not None, "PY123D_DATA_ROOT is not set"
    assert dataset_paths.py123d_logs_root is not None, "PY123D_DATA_ROOT is not set"
    map_writer = ArrowMapWriter(
        force_map_conversion=False,
        maps_root=dataset_paths.py123d_maps_root,
        logs_root=dataset_paths.py123d_logs_root,
    )

    xodr_files = sorted(CARLA_MAPS_DIR.glob("*.xodr.gz"))
    for xodr_file in xodr_files:
        town_name = xodr_file.name.removesuffix(".xodr.gz")
        map_metadata = MapMetadata(
            dataset=args.dataset,
            location=town_name.lower(),
            map_has_z=True,
            map_is_per_log=False,
        )
        if map_writer.reset(map_metadata):
            for map_object in iter_xodr_map_objects(xodr_file):
                map_writer.write_map_object(map_object)
            # close() is what flushes the buffered map to disk; the next reset()
            # would silently discard it otherwise
            map_writer.close()
            print(f"Converted {town_name}")
        else:
            print(f"Already converted: {town_name}")


if __name__ == "__main__":
    main()
