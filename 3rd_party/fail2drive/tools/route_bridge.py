"""
This scripts automatically changes the given Leaderboard 1.0
routes file into the new format used by the Leaderboard 2.0.

Disclaimer: The Leaderboard 2.0 improves the Leaderboard 1.0 scenarios so while this
script will maintain the route structure, the scenarios themselves will be different.
"""

import json
import xml.etree.ElementTree as ET
import argparse
import math
from numpy import random

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption

import carla

TRIGGER_THRESHOLD = 2.0
TRIGGER_ANGLE_THRESHOLD = 10

NUMBER_TO_NAME = {
    "Scenario1": "ControlLoss",
    "Scenario2": "HardBreakRoute",
    "Scenario3": "DynamicObjectCrossing",
    "Scenario4": "VehicleTurningRoute",
    "Scenario7": "SignalizedJunctionLeftTurn",
    "Scenario8": "OppositeVehicleRunningRedLight",
    "Scenario9": "SignalizedJunctionRightTurn",
}

def get_route(tmap, trajectory):

    grp = GlobalRoutePlanner(tmap, 1)
    route = []
    for i in range(len(trajectory) - 1):
        waypoint = trajectory[i]
        waypoint_next = trajectory[i + 1]
        interpolated_trace = grp.trace_route(waypoint, waypoint_next)
        for wp_tuple in interpolated_trace:
            route.append((wp_tuple[0].transform, wp_tuple[1]))

    return route

def get_scenarios(scenario_path):

    with open(scenario_path, 'r') as fd:
        annotations = json.load(fd)
        return annotations['available_scenarios'][0]

def scan_route_for_scenarios(route_name, trajectory, scenarios):

    existent_triggers = {}
    possible_scenarios = {}
    latest_trigger_id = 0

    for town_name in scenarios.keys():
        if town_name != route_name:
            continue

        scenarios = scenarios[town_name]
        for scenario in scenarios:  # For each existent scenario
            scenario_name = scenario["scenario_type"]
            for event in scenario["available_event_configurations"]:
                waypoint = event['transform']  # trigger point of this scenario
                waypoint['x'] = float(waypoint['x'])
                waypoint['y'] = float(waypoint['y'])
                waypoint['z'] = float(waypoint['z'])
                waypoint['yaw'] = float(waypoint['yaw'])
                # We match trigger point to the  route, now we need to check if the route affects
                match_position = match_world_location_to_route(waypoint, trajectory)
                if match_position is not None:

                    other_vehicles = event['other_actors'] if 'other_actors' in event else None
                    scenario_subtype = get_scenario_type(scenario_name, match_position, trajectory)
                    if scenario_subtype is None: continue
                    scenario_description = {
                        'name': scenario_name,
                        'other_actors': other_vehicles,
                        'trigger_position': waypoint,
                    }

                    trigger_id = check_trigger_position(waypoint, existent_triggers)
                    if trigger_id is None:
                        existent_triggers.update({latest_trigger_id: waypoint})
                        possible_scenarios.update({latest_trigger_id: []})
                        trigger_id = latest_trigger_id
                        latest_trigger_id += 1

                    possible_scenarios[trigger_id].append(scenario_description)

    return possible_scenarios


def match_world_location_to_route(world_location, route_description):

    def match_waypoints(waypoint1, wtransform):

        dx = float(waypoint1['x']) - wtransform.location.x
        dy = float(waypoint1['y']) - wtransform.location.y
        dz = float(waypoint1['z']) - wtransform.location.z
        dpos = math.sqrt(dx * dx + dy * dy + dz * dz)
        dyaw = (float(waypoint1['yaw']) - wtransform.rotation.yaw) % 360

        return dpos < TRIGGER_THRESHOLD \
            and (dyaw < TRIGGER_ANGLE_THRESHOLD or dyaw > (360 - TRIGGER_ANGLE_THRESHOLD))

    match_position = 0
    for route_waypoint in route_description:
        if match_waypoints(world_location, route_waypoint[0]):
            return match_position
        match_position += 1

    return None

def get_scenario_type(scenario, match_position, trajectory):

    def check_this_waypoint(tuple_wp_turn):
        if RoadOption.LANEFOLLOW == tuple_wp_turn[1]: return False
        elif RoadOption.CHANGELANELEFT == tuple_wp_turn[1]: return False
        elif RoadOption.CHANGELANERIGHT == tuple_wp_turn[1]: return False
        return True

    subtype = 'valid'
    if scenario == 'Scenario4':
        for tuple_wp_turn in trajectory[match_position:]:
            if check_this_waypoint(tuple_wp_turn):
                if RoadOption.LEFT == tuple_wp_turn[1]: subtype = 'S4left'
                elif RoadOption.RIGHT == tuple_wp_turn[1]: subtype = 'S4right'
                else: subtype = None
                break
            subtype = None

    if scenario == 'Scenario7':
        for tuple_wp_turn in trajectory[match_position:]:
            if check_this_waypoint(tuple_wp_turn):
                if RoadOption.LEFT == tuple_wp_turn[1]: subtype = 'S7left'
                elif RoadOption.RIGHT == tuple_wp_turn[1]: subtype = 'S7right'
                elif RoadOption.STRAIGHT == tuple_wp_turn[1]: subtype = 'S7opposite'
                else: subtype = None
                break
            subtype = None

    if scenario == 'Scenario8':
        for tuple_wp_turn in trajectory[match_position:]:
            if check_this_waypoint(tuple_wp_turn):
                if RoadOption.LEFT == tuple_wp_turn[1]: subtype = 'S8left'
                else: subtype = None
                break
            subtype = None

    if scenario == 'Scenario9':
        for tuple_wp_turn in trajectory[match_position:]:
            if check_this_waypoint(tuple_wp_turn):
                if RoadOption.RIGHT == tuple_wp_turn[1]: subtype = 'S9right'
                else: subtype = None
                break
            subtype = None

    return subtype


def check_trigger_position(new_trigger, existing_triggers):
    """
    Check if this trigger position already exists or if it is a new one.
    :param new_trigger:
    :param existing_triggers:
    :return:
    """

    for trigger_id in existing_triggers.keys():
        trigger = existing_triggers[trigger_id]
        dx = trigger['x'] - new_trigger['x']
        dy = trigger['y'] - new_trigger['y']
        distance = math.sqrt(dx * dx + dy * dy)

        dyaw = (trigger['yaw'] - new_trigger['yaw']) % 360
        if distance < TRIGGER_THRESHOLD \
            and (dyaw < TRIGGER_ANGLE_THRESHOLD or dyaw > (360 - TRIGGER_ANGLE_THRESHOLD)):
            return trigger_id

    return None


def scenario_sampling(potential_scenarios_definitions, random_seed=0):
    """
    The function used to sample the scenarios that are going to happen for this route.
    """
    rng = random.RandomState(random_seed)

    def position_sampled(scenario_choice, sampled_scenarios):
        """
        Check if a position was already sampled, i.e. used for another scenario
        """
        for existent_scenario in sampled_scenarios:
            # If the scenarios have equal positions then it is true.
            if compare_scenarios(scenario_choice, existent_scenario):
                return True

        return False

    def select_scenario(list_scenarios):
        # priority to the scenarios with higher number: 10 has priority over 9, etc.
        higher_id = -1
        selected_scenario = None
        for scenario in list_scenarios:
            try:
                scenario_number = int(scenario['name'].split('Scenario')[1])
            except:
                scenario_number = -1

            if scenario_number >= higher_id:
                higher_id = scenario_number
                selected_scenario = scenario

        return selected_scenario

    # The idea is to randomly sample a scenario per trigger position.
    sampled_scenarios = []
    for trigger in potential_scenarios_definitions.keys():
        possible_scenarios = potential_scenarios_definitions[trigger]

        scenario_choice = select_scenario(possible_scenarios)
        del possible_scenarios[possible_scenarios.index(scenario_choice)]
        # We keep sampling and testing if this position is present on any of the scenarios.
        while position_sampled(scenario_choice, sampled_scenarios):
            if possible_scenarios is None or not possible_scenarios:
                scenario_choice = None
                break
            scenario_choice = rng.choice(possible_scenarios)
            del possible_scenarios[possible_scenarios.index(scenario_choice)]

        if scenario_choice is not None:
            sampled_scenarios.append(scenario_choice)

    return sampled_scenarios

def compare_scenarios(scenario_choice, existent_scenario):

    def transform_to_pos_vec(scenario):

        position_vec = [scenario['trigger_position']]
        if scenario['other_actors'] is not None:
            if 'left' in scenario['other_actors']:
                position_vec += scenario['other_actors']['left']
            if 'front' in scenario['other_actors']:
                position_vec += scenario['other_actors']['front']
            if 'right' in scenario['other_actors']:
                position_vec += scenario['other_actors']['right']

        return position_vec

    # put the positions of the scenario choice into a vec of positions to be able to compare

    choice_vec = transform_to_pos_vec(scenario_choice)
    existent_vec = transform_to_pos_vec(existent_scenario)
    for pos_choice in choice_vec:
        for pos_existent in existent_vec:

            dx = float(pos_choice['x']) - float(pos_existent['x'])
            dy = float(pos_choice['y']) - float(pos_existent['y'])
            dz = float(pos_choice['z']) - float(pos_existent['z'])
            dist_position = math.sqrt(dx * dx + dy * dy + dz * dz)
            dyaw = float(pos_choice['yaw']) - float(pos_choice['yaw'])
            dist_angle = math.sqrt(dyaw * dyaw)
            if dist_position < TRIGGER_THRESHOLD and dist_angle < TRIGGER_ANGLE_THRESHOLD:
                return True

    return False

def save_new_route(new_routes, route_id, route_town, route, scenarios_data):

    # Create the base elements
    route_elem = ET.SubElement(new_routes, 'route')
    route_elem.set('id', route_id)
    route_elem.set('town', route_town)

    weathers = ET.SubElement(route_elem, "weathers")
    waypoints = ET.SubElement(route_elem, "waypoints")
    scenarios = ET.SubElement(route_elem, "scenarios")

    # Add the weather
    weather_params = [
        'cloudiness', 'precipitation', 'precipitation_deposits', 'wetness',
        'wind_intensity', 'sun_azimuth_angle', 'sun_altitude_angle', 'fog_density', 'fog_distance'
    ]

    for old_weather in route.iter('weather'):
        weather = ET.SubElement(weathers, "weather")
        weather.set('route_percentage', '0')
        for param in weather_params:
            if param in old_weather.attrib:
                weather.set(param, old_weather.attrib[param])

    # Add the waypoints
    for w in route.iter('waypoint'):
        position = ET.SubElement(waypoints, "position")
        position.set('x', w.attrib['x'])
        position.set('y', w.attrib['y'])
        position.set('z', w.attrib['z'])

    # Add the scenarios

    scenarios_number = {}
    for name in list(NUMBER_TO_NAME):
        scenarios_number[name] = 1

    for scenario in scenarios_data:

        # This aren't scenarios in the Leaderboard 2.0 per se, instead using the BackgroundActivity
        if scenario['name'] in ("Scenario5", "Scenario6", "Scenario10"):
            continue

        scenario_elem = ET.SubElement(scenarios, "scenario")
        scenario_elem.set("type", NUMBER_TO_NAME[scenario['name']])
        number = scenarios_number[scenario['name']]
        scenario_elem.set("name", NUMBER_TO_NAME[scenario['name']] + "_" + str(number))
        scenarios_number[scenario['name']] += 1

        trigger_point = ET.SubElement(scenario_elem, 'trigger_point')
        trigger_point.set('x', str(scenario['trigger_position']['x']))
        trigger_point.set('y', str(scenario['trigger_position']['y']))
        trigger_point.set('z', str(scenario['trigger_position']['z']))
        trigger_point.set('yaw', str(scenario['trigger_position']['yaw']))


def prettify_and_save_tree(filename, tree):
    def indent(elem, spaces=3, level=0):
        i = "\n" + level * spaces * " "
        j = "\n" + (level + 1) * spaces * " "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = j
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for subelem in elem:
                indent(subelem, spaces, level+1)
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
        return elem

    spaces = 3
    weather_spaces = spaces*4*" "
    indent(tree.getroot(), spaces)
    tree.write(filename)

    with open(filename, 'r') as f:
        data = f.read()

    temp = data.replace("   </", "</")
    temp = temp.replace(" />", "/>")
    temp = temp.replace(" cloudiness", "\n" + weather_spaces + "cloudiness")
    new_data = temp.replace(" wind_intensity", "\n" + weather_spaces + "wind_intensity")

    with open(filename, 'w') as f:
        f.write(new_data)

def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument('-r', '--routes', required=True, help='routes file path')
    argparser.add_argument('-s', '--scenarios', required=True, help='scenarios file path')
    argparser.add_argument('-e', '--endpoint', required=True, help='path to the endpoint file')
    args = argparser.parse_args()

    client = carla.Client()
    client.set_timeout(20)

    # New xml data
    new_routes = ET.Element('routes')
    new_tree = ET.ElementTree(new_routes)

    routes = ET.parse(args.routes)
    for route in routes.iter("route"):

        route_id = route.attrib['id']
        route_town = route.attrib['town']
        trajectory = []
        for w in route.iter('waypoint'):
            trajectory.append(carla.Location(float(w.attrib['x']), float(w.attrib['y']), float(w.attrib['z'])))

        world = client.load_world(route_town)

        # Get the route
        route_data = get_route(world.get_map(), trajectory)

        # Get the scenarios data
        scenarios_data = get_scenarios(args.scenarios)

        # Get the potential scenarios
        potential_scenarios_definitions = scan_route_for_scenarios(route_town, route_data, scenarios_data)

        # Get the real scenarios
        scenarios_data = scenario_sampling(potential_scenarios_definitions)

        # Save the scenarios
        save_new_route(new_routes, route_id, route_town, route, scenarios_data)

    prettify_and_save_tree(args.endpoint, new_tree)

if __name__ == '__main__':
    main()
