#!/usr/bin/env python3
import os
import random
import xml.etree.ElementTree as ET


def process_routes(src_route_dir, tgt_dir, tgt_scenario, src_scenario, route_id):
    total_written = 0
    for filename in os.listdir(src_route_dir):
        if not filename.endswith(".xml"):
            continue

        file_path = os.path.join(src_route_dir, filename)
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            for route in root.findall("route"):
                route.set("id", str(route_id))

                for scenario in route.findall(".//scenarios/scenario"):
                    scenario.attrib["name"] = f"{tgt_scenario}_duplicated_from_{src_scenario}"
                    scenario.attrib["type"] = tgt_scenario

                    # Reduce flow_speed by 5
                    flow_speed_elem = scenario.find("flow_speed")
                    if flow_speed_elem is not None and "value" in flow_speed_elem.attrib:
                        current_speed = int(flow_speed_elem.attrib["value"])
                        flow_speed_elem.attrib["value"] = str(current_speed - 5)

                # Write individual file
                routes_root = ET.Element("routes")
                routes_root.append(route)

                tgt_file = os.path.join(tgt_dir, f"{route_id}.xml")
                ET.ElementTree(routes_root).write(tgt_file, encoding="utf-8", xml_declaration=False)

                route_id += 1
                total_written += 1

        except ET.ParseError:
            print(f"Failed to parse {file_path}, skipping.")

    return total_written, route_id


def main():
    new_count = 0
    route_id = 700000

    for town, town_path in [("Town12", "50x38_Town12"), ("Town13", "50x36_Town13")]:
        src_route_dir = f"data/routes/{town_path}/SignalizedJunctionLeftTurn"
        src_scenario = "SignalizedJunctionLeftTurn"
        tgt_scenario = "SignalizedJunctionLeftTurnEnterFlow"
        tgt_dir = f"data/routes/scenario_swapped_routes/{town}/{src_scenario}/{tgt_scenario}"

        os.makedirs(tgt_dir, exist_ok=True)
        written, route_id = process_routes(src_route_dir, tgt_dir, tgt_scenario, src_scenario, route_id)
        print(f"Written {written} routes to {tgt_dir}/")
        new_count += written

    print(f"Total new routes created: {new_count}")


if __name__ == "__main__":
    main()
