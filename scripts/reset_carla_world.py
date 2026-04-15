#!/usr/bin/env python3
"""
Clean up CARLA simulator without restarting it.
This script removes all actors (vehicles, walkers, sensors) from the current world.
"""

import argparse
import sys
import time
import carla



def clean_carla(host='localhost', port=2000, timeout=10.0, reload_map=False):
    """
    Connect to CARLA and destroy all actors.

    Args:
        host: CARLA server host
        port: CARLA server port
        timeout: Connection timeout in seconds
        reload_map: If True, reload the current map for complete cleanup
    """
    try:
        # Connect to the CARLA server
        print(f"Connecting to CARLA server at {host}:{port}...")
        client = carla.Client(host, port)
        client.set_timeout(timeout)

        # Get the world
        world = client.get_world()
        print(f"Connected to world: {world.get_map().name}")

        # Get all actors
        actors = world.get_actors()
        print(f"Found {len(actors)} actors in the world")

        # Categorize actors
        vehicles = actors.filter('vehicle.*')
        walkers = actors.filter('walker.*')
        sensors = actors.filter('sensor.*')
        traffic_lights = actors.filter('traffic.traffic_light')

        print(f"  - Vehicles: {len(vehicles)}")
        print(f"  - Walkers: {len(walkers)}")
        print(f"  - Sensors: {len(sensors)}")
        print(f"  - Traffic lights: {len(traffic_lights)}")

        # Destroy all actors except traffic lights
        destroyed_count = 0

        # First destroy sensors (attached to vehicles)
        print("\nDestroying sensors...")
        for sensor in sensors:
            try:
                sensor.destroy()
                destroyed_count += 1
            except Exception as e:
                print(f"  Warning: Could not destroy sensor {sensor.id}: {e}")

        # Then destroy walkers and their controllers
        print("Destroying walkers...")
        for walker in walkers:
            try:
                walker.destroy()
                destroyed_count += 1
            except Exception as e:
                print(f"  Warning: Could not destroy walker {walker.id}: {e}")

        # Finally destroy vehicles
        print("Destroying vehicles...")
        for vehicle in vehicles:
            try:
                vehicle.destroy()
                destroyed_count += 1
            except Exception as e:
                print(f"  Warning: Could not destroy vehicle {vehicle.id}: {e}")

        print(f"\nSuccessfully destroyed {destroyed_count} actors")

        # Optional: Reset traffic lights to green
        print("\nResetting traffic lights to green...")
        for traffic_light in traffic_lights:
            try:
                traffic_light.set_state(carla.TrafficLightState.Green)
                traffic_light.set_green_time(100.0)
            except Exception as e:
                print(f"  Warning: Could not reset traffic light {traffic_light.id}: {e}")

        # CRITICAL: Tick the world multiple times to ensure physics state is cleared
        print("\nClearing physics state (ticking world)...")
        for i in range(10):
            world.tick()
            time.sleep(0.05)

        print("Physics state cleared")

        # Optional: Reload the map for complete cleanup
        if reload_map:
            print(f"\nReloading map: {world.get_map().name}...")
            client.reload_world()
            print("Map reloaded successfully")
            time.sleep(1.0)
            world = client.get_world()

        # Verify cleanup
        remaining_actors = world.get_actors()
        remaining_vehicles = remaining_actors.filter('vehicle.*')
        remaining_walkers = remaining_actors.filter('walker.*')
        remaining_sensors = remaining_actors.filter('sensor.*')

        # Sometimes actors appear after physics clearing - destroy them too
        if len(remaining_vehicles) > 0 or len(remaining_walkers) > 0 or len(remaining_sensors) > 0:
            print(f"\nWarning: Found {len(remaining_vehicles) + len(remaining_walkers) + len(remaining_sensors)} actors after physics clear")
            print(f"  - Vehicles: {len(remaining_vehicles)}")
            print(f"  - Walkers: {len(remaining_walkers)}")
            print(f"  - Sensors: {len(remaining_sensors)}")

            print("\nPerforming second cleanup pass...")
            second_pass_count = 0

            for sensor in remaining_sensors:
                try:
                    sensor.destroy()
                    second_pass_count += 1
                except Exception as e:
                    print(f"  Warning: Could not destroy sensor {sensor.id}: {e}")

            for walker in remaining_walkers:
                try:
                    walker.destroy()
                    second_pass_count += 1
                except Exception as e:
                    print(f"  Warning: Could not destroy walker {walker.id}: {e}")

            for vehicle in remaining_vehicles:
                try:
                    vehicle.destroy()
                    second_pass_count += 1
                except Exception as e:
                    print(f"  Warning: Could not destroy vehicle {vehicle.id}: {e}")

            print(f"Destroyed {second_pass_count} additional actors")

            # Tick again to finalize
            print("Ticking world again...")
            for i in range(5):
                world.tick()
                time.sleep(0.05)

            # Final verification
            final_actors = world.get_actors()
            final_vehicles = final_actors.filter('vehicle.*')
            final_walkers = final_actors.filter('walker.*')
            final_sensors = final_actors.filter('sensor.*')

            if len(final_vehicles) > 0 or len(final_walkers) > 0 or len(final_sensors) > 0:
                print(f"\n⚠ Warning: Still {len(final_vehicles) + len(final_walkers) + len(final_sensors)} actors remain")
                print(f"  - Vehicles: {len(final_vehicles)}")
                print(f"  - Walkers: {len(final_walkers)}")
                print(f"  - Sensors: {len(final_sensors)}")
                print("\nTry running with --reload-map for complete cleanup")
            else:
                print("\n✓ World cleaned successfully! All actors removed.")
        else:
            print("\n✓ World cleaned successfully! All actors removed.")

        return True

    except RuntimeError as e:
        print(f"Error: Could not connect to CARLA server: {e}")
        print("Make sure CARLA is running and accessible.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Clean up CARLA simulator without restarting it',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Connect to localhost:2000
  %(prog)s --host 192.168.1.100     # Connect to remote server
  %(prog)s --port 2002               # Use different port
  %(prog)s --reload-map             # Reload map for thorough cleanup
        """
    )
    parser.add_argument('--host', default='localhost',
                        help='CARLA server host (default: localhost)')
    parser.add_argument('--port', type=int, default=2000,
                        help='CARLA server port (default: 2000)')
    parser.add_argument('--timeout', type=float, default=10.0,
                        help='Connection timeout in seconds (default: 10.0)')
    parser.add_argument('--reload-map', action='store_true',
                        help='Reload the current map for complete cleanup (takes longer)')

    args = parser.parse_args()

    success = clean_carla(args.host, args.port, args.timeout, args.reload_map)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
