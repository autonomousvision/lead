#!/usr/bin/env python3
"""Split scenario-swapped route XML files into individual route files."""

import os
import xml.etree.ElementTree as ET
from pathlib import Path


def split_route_file(input_file: Path, output_dir: Path, start_idx: int) -> int:
    """Split a route XML file into individual route files, continuing from global counter."""
    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
    except ET.ParseError:
        print(f"Failed to parse {input_file}, skipping.")
        return 0

    route_count = 0
    for route in root.findall("route"):
        global_id = f"route_{start_idx + route_count:06d}"
        route.set("id", global_id)

        routes_root = ET.Element("routes")
        routes_root.append(route)

        output_file = output_dir / f"{global_id}.xml"
        ET.ElementTree(routes_root).write(output_file, encoding="utf-8", xml_declaration=False)

        route_count += 1

    return route_count



def main():
    input_base = Path("data/routes/scenario_swapped_routes")
    output_base = Path("data/routes/supplementary_data")

    if not input_base.exists():
        print(f"Input directory not found: {input_base}")
        return

    total_routes = 0
    total_files = 0
    global_counter = 0

    # Iterate through town/src/tgt structure
    for town_dir in input_base.iterdir():
        if not town_dir.is_dir():
            continue

        for src_dir in town_dir.iterdir():
            if not src_dir.is_dir():
                continue

            for tgt_dir in src_dir.iterdir():
                if not tgt_dir.is_dir():
                    continue

                tgt_scenario = tgt_dir.name

                # Create output directory for this target scenario
                output_dir = output_base / tgt_scenario
                output_dir.mkdir(parents=True, exist_ok=True)

                # Process all XML files in this directory
                xml_files = list(tgt_dir.glob("*.xml"))
                if not xml_files:
                    continue

                for xml_file in xml_files:
                    route_count = split_route_file(xml_file, output_dir, global_counter)
                    global_counter += route_count
                    total_routes += route_count
                    total_files += 1

                if xml_files:
                    print(f"Processed {len(xml_files)} files from {town_dir.name}/{src_dir.name}/{tgt_scenario} -> {output_dir}")

    print(f"\n✓ Split complete!")
    print(f"  Total files processed: {total_files}")
    print(f"  Total routes created: {total_routes}")
    print(f"  Output directory: {output_base}")


if __name__ == "__main__":
    main()
