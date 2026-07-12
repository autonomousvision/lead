#!/usr/bin/env python
import argparse
import time

import carla


def load_town_and_set_position(town_name, position=None):
    """
    Connect to the CARLA server, load the specified town, and optionally move the spectator to a specific position.
    """
    CARLA_HOST = "localhost"
    CARLA_PORT = 2000
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(360.0)

    print(f"Loading {town_name}...")
    world = client.load_world(town_name)
    print(f"{town_name} loaded successfully!")

    # Set the spectator to the specified position if provided
    if position is not None:
        spectator = world.get_spectator()
        transform = carla.Transform(
            carla.Location(x=position[0], y=position[1], z=position[2] + 20),  # Z is height
            carla.Rotation(pitch=-90, yaw=0, roll=0),  # Look downwards
        )
        spectator.set_transform(transform)
        print(f"Moved spectator to position: {position}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a CARLA town and optionally set spectator position")
    parser.add_argument("--town", type=str, default="Town12", help="Town name to load (default: Town12)")
    parser.add_argument(
        "--position", type=float, nargs=3, metavar=("X", "Y", "Z"),
        help="Spectator position as three floats (x, y, z)",
    )
    args = parser.parse_args()

    try:
        position = tuple(args.position) if args.position else None
        load_town_and_set_position(args.town, position)
        while True:
            time.sleep(1)  # Keep the script running to maintain connection
    except KeyboardInterrupt:
        print("Exiting...")
