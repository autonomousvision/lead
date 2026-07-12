"""
This script extracts and saves various map data from the CARLA simulator for later use.
It connects to the CARLA server, loads a selected map, and collects the following information:

- Road waypoints: A set of 3D coordinates representing the drivable paths on the map.
- Parking waypoints: Coordinates of parking spots along the roads.
- Stop sign locations: Positions of stop signs on the map.
- Traffic light locations: Positions of traffic lights.

The collected data is saved in a pickle file for each map, which can be loaded and used
by other parts of the application. This preprocessing step enhances efficiency by avoiding
the need to query the CARLA server for map data during runtime, speeding up route and
scenario creation processes.

Note: Make sure the CARLA server is running before executing this script.
"""

import numpy as np
import carla
import argparse
import pickle
import pathlib
import os
from tqdm import tqdm
import time

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="localhost", help="The host IP of the CARLA Server.")
parser.add_argument("--port", type=int, default=2000, help="The port of the CARLA Server.")
parser.add_argument("--output-dir", type=str, default="carla_map_data", help="The path to save the map data.")
args = parser.parse_args()

try:
    # Connect to the CARLA server
    client = carla.Client(args.host, args.port)
    client.set_timeout(1)  # Set a short timeout for the initial connection test

    # Check if the CARLA server is running
    client.get_server_version()

except RuntimeError as e:
    # If the server is not running, print an error message and exit
    print(f"Error: {e}")
    print("Make sure the CARLA server is running before executing this script.")
    exit(1)

# Set a longer timeout for subsequent operations
client.set_timeout(120)

# Get the list of available maps
available_maps = client.get_available_maps()
available_maps = sorted([x.split("/")[-1] for x in available_maps])

# Create the save path if it doesn't exist
pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)

for map_name in tqdm(available_maps):

    # Load the world and get the map
    carla_world = client.load_world(map_name)
    time.sleep(5)  # sometimes CARLA takes a bit time to load the world properly
    carla_map = carla_world.get_map()

    # Get map waypoints and parking waypoints
    road_waypoints = carla_map.generate_waypoints(1)
    road_waypoints_np = np.array(
        [[wp.transform.location.x, wp.transform.location.y, wp.transform.location.z] for wp in road_waypoints]
    )
    road_waypoints_np = road_waypoints_np[:, :3]  # Keep only x and y coordinates

    # Collect parking waypoints
    parking_waypoints = []
    for wp in road_waypoints:
        left_waypoint = wp.get_left_lane()
        right_waypoint = wp.get_right_lane()
        if (
            left_waypoint is not None
            and left_waypoint.lane_type == carla.LaneType.Parking
            and not left_waypoint.is_junction
        ):
            parking_waypoints.append(left_waypoint)
        if (
            right_waypoint is not None
            and right_waypoint.lane_type == carla.LaneType.Parking
            and not right_waypoint.is_junction
        ):
            parking_waypoints.append(right_waypoint)
    parking_waypoints_np = np.array(
        [[wp.transform.location.x, wp.transform.location.y, wp.transform.location.z] for wp in parking_waypoints]
    )
    parking_waypoints_np = parking_waypoints_np.reshape((-1, 3)).astype("float")

    # Collect stop signs
    stop_signs = carla_world.get_actors().filter("*traffic.stop*")
    stop_sign_centers = [x.get_transform().transform(x.trigger_volume.location) for x in stop_signs]
    stop_sign_wps = [carla_map.get_waypoint(x) for x in stop_sign_centers]
    stop_sign_centers_np = np.array(
        [[x.transform.location.x, x.transform.location.y, x.transform.location.z] for x in stop_sign_wps]
    ).astype("float")

    # Collect traffic lights
    traffic_lights = carla_world.get_actors().filter("*traffic.traffic_light*")
    traffic_light_wps = [x.get_affected_lane_waypoints() for x in traffic_lights]
    traffic_light_wps = [item for sublist in traffic_light_wps for item in sublist]
    traffic_light_centers_np = np.array(
        [[x.transform.location.x, x.transform.location.y, x.transform.location.z] for x in traffic_light_wps]
    ).astype("float")

    # Calculate map dimensions
    all_waypoints_np = np.concatenate([road_waypoints_np, parking_waypoints_np], axis=0)
    num_road_waypoints, num_parking_waypoints = road_waypoints_np.shape[0], parking_waypoints_np.shape[0]

    # Save the data to a pickle file
    data = {
        "stop_sign_centers_np": stop_sign_centers_np,
        "traffic_light_centers_np": traffic_light_centers_np,
        "all_waypoints_np": all_waypoints_np,
        "num_road_waypoints": num_road_waypoints,
        "num_parking_waypoints": num_parking_waypoints,
    }

    with open(os.path.join(args.output_dir, f"{map_name}.pkl"), "wb") as f:
        pickle.dump(data, f)
