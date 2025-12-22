import os
import random
import xml.etree.ElementTree as ET
import carla

# Ranges (change here)
DISTANCE_MIN, DISTANCE_MAX = 35, 40
FREQ_FROM_MIN, FREQ_FROM_MAX = 20, 60
FREQ_TO_MIN, FREQ_TO_MAX = 100, 190

BACKWARD_STEP = 2.0  # meters to step backward each time
MAX_BACKWARD_DIST = 200.0  # don’t go more than this

def extend_start_backward(carla_map, x, y, z):
    """Go backward until just before intersection or lane start."""
    wp = carla_map.get_waypoint(carla.Location(x=x, y=y, z=z))
    prev_wp = wp  # keep track of last non-junction waypoint
    dist = 0.0
    while wp and not wp.is_junction and dist < MAX_BACKWARD_DIST:
        prev_wp = wp
        nxt = wp.previous(BACKWARD_STEP)
        if not nxt:
            break
        wp = nxt[0]
        dist += BACKWARD_STEP
    return prev_wp.transform.location if prev_wp else carla.Location(x=x, y=y, z=z)



def process_routes(src_route_dir, tgt_route, tgt_scenario, src_scenario, carla_map, route_id):
    all_routes = []
    for filename in os.listdir(src_route_dir):
        if not filename.endswith(".xml"):
            continue

        file_path = os.path.join(src_route_dir, filename)
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            for route in root.findall("route"):
                route.set("id", str(route_id))
                route_id += 1

                # waypoints
                waypoints = route.find("waypoints")
                if waypoints is None:
                    continue
                first_pos = waypoints[0]
                start_x = float(first_pos.attrib["x"])
                start_y = float(first_pos.attrib["y"])
                start_z = float(first_pos.attrib["z"])

                # extend backward
                new_loc = extend_start_backward(carla_map, start_x, start_y, start_z)
                first_pos.set("x", str(new_loc.x))
                first_pos.set("y", str(new_loc.y))
                first_pos.set("z", str(new_loc.z))

                for scenario in route.findall(".//scenarios/scenario"):
                    scenario.attrib["name"] = f"{tgt_scenario}_duplicated_from_{src_scenario}"
                    scenario.attrib["type"] = tgt_scenario

                    # update trigger point to new start
                    trig = scenario.find("trigger_point")
                    if trig is not None:
                        trig.set("x", str(new_loc.x))
                        trig.set("y", str(new_loc.y))
                        trig.set("z", str(new_loc.z))
                        trig.set("yaw", str(carla_map.get_waypoint(new_loc).transform.rotation.yaw))

                    # Randomize <distance>
                    dist_el = scenario.find("distance")
                    if dist_el is None:
                        dist_el = ET.SubElement(scenario, "distance")
                    dist_el.set("value", str(random.randint(DISTANCE_MIN, DISTANCE_MAX)))

                    # Randomize <frequency>
                    freq_el = scenario.find("frequency")
                    if freq_el is None:
                        freq_el = ET.SubElement(scenario, "frequency")
                    freq_el.set("from", str(random.randint(FREQ_FROM_MIN, FREQ_FROM_MAX)))
                    freq_el.set("to", str(random.randint(FREQ_TO_MIN, FREQ_TO_MAX)))

                all_routes.append(route)
        except ET.ParseError:
            print(f"Failed to parse {file_path}, skipping.")

    # Write
    routes_root = ET.Element("routes")
    for route in all_routes:
        routes_root.append(route)

    ET.ElementTree(routes_root).write(tgt_route, encoding="utf-8", xml_declaration=False)
    return len(all_routes), route_id


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(180.0)

    new_count = 0
    route_id = 400000

    for town, town_path in [("Town12", "50x38_Town12"), ("Town13", "50x36_Town13")]:
        world = client.load_world(town)
        carla_map = world.get_map()

        for tgt_scenario in ["ConstructionObstacleTwoWays", "AccidentTwoWays"]:
            for src_route_dir in [f"data/routes/{town_path}/ParkingExit"]:
                src_scenario = os.path.basename(src_route_dir)
                tgt_route = f"data/routes/scenario_swapped_routes/{town}/{src_scenario}/{tgt_scenario}/{route_id}.xml"

                if tgt_scenario == src_scenario:
                    print(f"Skipping identical scenario swap for {tgt_scenario} from {src_scenario}")
                    continue

                os.makedirs(os.path.dirname(tgt_route), exist_ok=True)
                written, route_id = process_routes(src_route_dir, tgt_route, tgt_scenario, src_scenario, carla_map, route_id)
                print(f"Written {written} routes to {tgt_route}")
                new_count += written

    print(f"Total new routes created: {new_count}")


if __name__ == "__main__":
    main()
