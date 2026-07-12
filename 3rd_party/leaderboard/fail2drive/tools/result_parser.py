'''
File that aggregates results of a carla evaluation run into a csv file.
'''

import os
import argparse
import re
import csv
import sys
import xml.etree.ElementTree as ET
import numpy as np
import ujson
import math

scale_factor = 0.2
PENALTY_VALUE_DICT = {
    # Traffic events that substract a set amount of points.
    'collisions_pedestrian': 0.5 * scale_factor,
    'collisions_vehicle': 0.6 * scale_factor,
    'collisions_layout': 0.65 * scale_factor,
    'red_light': 0.7 * scale_factor,
    'scenario_timeouts': 0.7 * scale_factor,
    'yield_emergency_vehicle_infractions': 0.7 * scale_factor,
    'stop_infraction': 0.8 * scale_factor,
}

def min_speed_penalty(percentage):
  score_penalty = (1 - (1 - 0.7) * (1 - percentage / 100))
  return score_penalty

def outside_route_lanes_penalty(percentage):
  score_penalty = (1 - (1 - 0.0) * percentage / 100)
  return score_penalty

def main():
  # available arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--xml', type=str, default='../leaderboard/data/longest6.xml', help='Routes file.')
  parser.add_argument('--results', type=str, required=True, help='Folder with json files to be parsed')
  parser.add_argument('--strict',
                      action='store_true',
                      default=False,
                      help='If set only creates the results file if all routes finished correctly.')

  args = parser.parse_args()

  infraction_names = [
                    "collisions_layout",
                    "collisions_pedestrian",
                    "collisions_vehicle",
                    "red_light",
                    "stop_infraction",
                    "outside_route_lanes",
                    "min_speed_infractions",
                    "yield_emergency_vehicle_infractions",
                    "scenario_timeouts",
                    "route_dev",
                    "vehicle_blocked",
                    "route_timeout"
  ]
  labels = [
        "Avg. driving score",
        "Avg. route completion",
        "Avg. infraction penalty",
        "Collisions with pedestrians",
        "Collisions with vehicles",
        "Collisions with layout",
        "Red lights infractions",
        "Stop sign infractions",
        "Off-road infractions",
        "Route deviations",
        "Route timeouts",
        "Agent blocked",
        "Yield emergency vehicles infractions",
        "Scenario timeouts",
        "Min speed infractions"
  ]

  total_score_labels = []

  driving_scores = []
  route_completions = []
  infraction_scores = []
  normalized_driving_scores = []
  normalized_infraction_scores = []
  route_ids = []
  duration_games = []
  route_lengths = []
  individual_infractions = []

  total_km_driven = 0.0
  total_driven_hours = 0.0
  total_number_of_routes = 0
  total_infractions = {}
  total_infractions_per_km = {}

  root = ET.parse(args.xml).getroot()

  # build route matching dict
  route_matching = {}
  for route in root.iter('route'):
    route_matching[route.attrib["id"]] = {'town': route.attrib["town"]}

  filenames = []
  for foldername, _, files in os.walk(args.results):
    paths = []
    for filename in files:
      if filename.endswith('.json'):
        paths.append(os.path.join(foldername, filename))
    filenames += paths

  abort = False
  # aggregate files
  for f in filenames:
    with open(f, encoding='utf-8') as json_file:
      evaluation_data = ujson.load(json_file)

      if len(total_infractions) == 0:
        for infraction_name in infraction_names:
          total_infractions[infraction_name] = 0

      for record in evaluation_data['_checkpoint']['records']:
        if record['scores']['score_route'] <= 1e-7:
          print('Warning: There is a route where the agent did not start to drive.' + ' Route ID: ' +
                record['route_id'],
                file=sys.stderr)
        if record['status'] == 'Failed - Agent couldn\'t be set up':
          print('Error: There is at least one route where the agent could not be set up.' + ' Route ID: ' +
                record['route_id'],
                file=sys.stderr)
          abort = True
        if record['status'] == 'Failed':
          print('Error: There is at least one route that failed.' + ' Route ID: ' + record['route_id'],
                file=sys.stderr)
          abort = True
        if record['status'] == 'Failed - Simulation crashed':
          print('Error: There is at least one route where the simulation crashed.' + ' Route ID: ' +
                record['route_id'],
                file=sys.stderr)
          abort = True
        if record['status'] == 'Failed - Agent crashed':
          print('Error: There is at least one route where the agent crashed.' + ' Route ID: ' + record['route_id'],
                file=sys.stderr)
          abort = True

        percentage_of_route_completed = record['scores']['score_route'] / 100.0
        route_length_km = record['meta']['route_length'] / 1000.0
        driven_km = percentage_of_route_completed * route_length_km
        route_time_hours = record['meta']['duration_game'] / 3600.0  # conversion from seconds to hours
        total_driven_hours += route_time_hours
        total_km_driven += driven_km
        if route_time_hours > 0.0:
          avg_speed_km_h = driven_km / route_time_hours
        else:
          avg_speed_km_h = 0.0

        total_number_of_routes += 1
        local_infractions = {}
        for infraction_name in infraction_names:
          local_infractions[infraction_name] = 0
          if infraction_name == 'outside_route_lanes':
            if len(record['infractions'][infraction_name]) > 0:
              meters_off_road = re.findall(r'\d+\.\d+', record['infractions'][infraction_name][0])[0]
              km_off_road = float(meters_off_road) / 1000.0
              total_infractions[infraction_name] += km_off_road
              local_infractions[infraction_name] += km_off_road
          elif infraction_name == 'min_speed_infractions':
            if len(record['infractions'][infraction_name]) == 0:
              total_infractions[infraction_name] += 0
              local_infractions[infraction_name] += 0
            else:
              perc_speed_of_traffic = []
              for min_speed_inf in record['infractions'][infraction_name]:
                pattern = r"(\d+(\.\d+)?)%"
                min_speed_section = float(re.findall(pattern, min_speed_inf)[0][0]) / 100.0
                # clip to 0 to 1
                min_speed_section = min(1.0, max(0.0, min_speed_section))
                perc_speed_of_traffic.append(min_speed_section)
              avg_min_speed = np.mean(perc_speed_of_traffic)
              # We log percentage of route where the min speed was violated.
              total_infractions[infraction_name] += 1.0 - avg_min_speed
              local_infractions[infraction_name] += 1.0 - avg_min_speed

          else:
            num_infraction = len(record['infractions'][infraction_name])
            total_infractions[infraction_name] += num_infraction
            local_infractions[infraction_name] += num_infraction

        # Compute normalized driving score.
        score_penalty = 1.0
        score_route = record['scores']['score_route']

        # Standard infractions
        for inf_name in local_infractions:
          if inf_name in PENALTY_VALUE_DICT:
            if driven_km > 0.0:
                score_penalty *= math.pow(PENALTY_VALUE_DICT[inf_name], (local_infractions[inf_name] / driven_km))

        # Special infraction min speed
        if len(record['infractions']['min_speed_infractions']) > 0:
          for min_speed_inf in record['infractions']['min_speed_infractions']:
            min_speed_section = float(re.findall(r'\d+\.\d+', min_speed_inf)[0])
            score_penalty *= min_speed_penalty(min_speed_section)

        # Special infraction outside route lanes
        if len(record['infractions']['outside_route_lanes']) > 0:
          for outside_route_lanes in record['infractions']['outside_route_lanes']:
            outside_route_lanes_perc = float(re.findall(r'\d+\.\d+', outside_route_lanes)[1])
            score_penalty *= outside_route_lanes_penalty(outside_route_lanes_perc)

        normalied_ds = score_route * score_penalty

        normalized_driving_scores.append(normalied_ds)
        normalized_infraction_scores.append(score_penalty)

        driving_scores.append(record['scores']['score_composed'])
        route_completions.append(score_route)
        infraction_scores.append(record['scores']['score_penalty'])
        route_ids.append(record['route_id'])
        duration_games.append(record['meta']['duration_game'])
        route_lengths.append(record['meta']['route_length'])
        individual_infractions.append(local_infractions)

      total_score_labels = labels[:]
      total_score_labels.append('Avg. speed km/h')
      total_score_labels.append('Avg. Normalized DS')
      total_score_labels.append('Avg. Normalized IS')

  for key, value in total_infractions.items():
    if key == 'min_speed_infractions':
      # Since this infraction is a percentage, we put it in rage [0.0, 100.0]
      total_infractions_per_km[key] = (value / total_number_of_routes) * 100.0
    else:
      total_infractions_per_km[key] = value / total_km_driven
      if key == 'outside_route_lanes':
        # Since this infraction is a percentage, we put it in rage [0.0, 100.0]
        total_infractions_per_km[key] = total_infractions_per_km[key] * 100.0

  avg_km_h_speed = total_km_driven / total_driven_hours


  if total_number_of_routes % len(route_matching) != 0:
    print('Error: The number of completed routes (' + str(total_number_of_routes) +
          ') is not a multiple of the total routes (' + str(len(route_matching)) +
          '). Check if there are missing results.',
          file=sys.stderr)
    abort = True

  if abort and args.strict:
    print('Don not create result file because not all routes were completed successfully and strict is set.',
          file=sys.stderr)
    sys.exit()

  total_score_values = np.zeros(18)

  for idx, value in enumerate(total_score_labels):
    if value == 'Avg. driving score':
      total_score_values[idx] = np.sum(driving_scores) / len(driving_scores)
    elif value == 'Avg. route completion':
      total_score_values[idx] = np.sum(route_completions) / len(route_completions)
    elif value == 'Avg. infraction penalty':
      total_score_values[idx] = np.sum(infraction_scores) / len(infraction_scores)
    elif value == 'Collisions with pedestrians':
      total_score_values[idx] = total_infractions_per_km['collisions_pedestrian']
    elif value == 'Collisions with vehicles':
      total_score_values[idx] = total_infractions_per_km['collisions_vehicle']
    elif value == 'Collisions with layout':
      total_score_values[idx] = total_infractions_per_km['collisions_layout']
    elif value == 'Red lights infractions':
      total_score_values[idx] = total_infractions_per_km['red_light']
    elif value == 'Stop sign infractions':
      total_score_values[idx] = total_infractions_per_km['stop_infraction']
    elif value == 'Off-road infractions':
      total_score_values[idx] = total_infractions_per_km['outside_route_lanes']
    elif value == 'Route deviations':
      total_score_values[idx] = total_infractions_per_km['route_dev']
    elif value == 'Route timeouts':
      total_score_values[idx] = total_infractions_per_km['route_timeout']
    elif value == 'Agent blocked':
      total_score_values[idx] = total_infractions_per_km['vehicle_blocked']
    elif value == 'Yield emergency vehicles infractions':
      total_score_values[idx] = total_infractions_per_km['yield_emergency_vehicle_infractions']
    elif value == 'Scenario timeouts':
      total_score_values[idx] = total_infractions_per_km['scenario_timeouts']
    elif value == 'Min speed infractions':
      total_score_values[idx] = total_infractions_per_km['min_speed_infractions']
    elif value == 'Avg. speed km/h':
      total_score_values[idx] = avg_km_h_speed
    elif value == 'Avg. Normalized DS':
      total_score_values[idx] = np.sum(normalized_driving_scores) / len(normalized_driving_scores)
    elif value == 'Avg. Normalized IS':
      total_score_values[idx] = np.sum(normalized_infraction_scores) / len(normalized_infraction_scores)

  # dict to extract unique identity of route in case of repetitions
  route_to_id = {}
  for route_id in route_ids:
    route_to_id[route_id] = str(re.search('_(\\d+)_', route_id).group(1))

  # build table of relevant information
  total_score_info = [{"label": label, "value": value} for label, value in zip(total_score_labels, total_score_values)]
  route_scenarios = [{"route": route_id,
                      "town": route_matching[route_to_id[route_id]]["town"],
                      "duration": duration_games[idx],
                      "length": route_lengths[idx],
                      "DS": driving_scores[idx],
                      "RC": route_completions[idx],
                      "NDS": normalized_driving_scores[idx],
                      "infractions": [(key, item)
                                      for key, item in individual_infractions[idx].items()]}
                     for idx, route_id in enumerate(route_ids)]



  # compute aggregated statistics and table for each filter
  filters = ["route", "town"]
  evaluation_filtered = {}

  for filter in filters:
    subcategories = np.unique(np.array([scenario[filter] for scenario in route_scenarios]))
    route_scenarios_per_subcategory = {}
    evaluation_per_subcategory = {}
    for subcategory in subcategories:
      route_scenarios_per_subcategory[subcategory] = []
      evaluation_per_subcategory[subcategory] = {}
    for scenario in route_scenarios:
      route_scenarios_per_subcategory[scenario[filter]].append(scenario)
    for subcategory in subcategories:
      scores = np.array([scenario["DS"] for scenario in route_scenarios_per_subcategory[subcategory]])
      completions = np.array([scenario["RC"] for scenario in route_scenarios_per_subcategory[subcategory]])
      n_scores = np.array([scenario["NDS"] for scenario in route_scenarios_per_subcategory[subcategory]])
      durations = np.array([scenario["duration"] for scenario in route_scenarios_per_subcategory[subcategory]])
      lengths = np.array([scenario["length"] for scenario in route_scenarios_per_subcategory[subcategory]])

      infractions = np.array([[infraction[1] for infraction in scenario["infractions"]]
                              for scenario in route_scenarios_per_subcategory[subcategory]])

      scores_combined = (scores.mean(), scores.std())
      completions_combined = (completions.mean(), completions.std())
      n_scores_combined = (n_scores.mean(), n_scores.std())

      durations_combined = (durations.mean(), durations.std())
      lengths_combined = (lengths.mean(), lengths.std())
      infractions_combined = [(mean, std) for mean, std in zip(infractions.mean(axis=0), infractions.std(axis=0))]

      evaluation_per_subcategory[subcategory] = {"DS": scores_combined,
                                                 "RC": completions_combined,
                                                 "NDS": n_scores_combined,
                                                 "duration": durations_combined,
                                                 "length": lengths_combined,
                                                 "infractions": infractions_combined}
    evaluation_filtered[filter] = evaluation_per_subcategory

  # write output csv file
  if not os.path.isdir(args.results):
    os.mkdir(args.save_dir)
  with open(os.path.join(args.results, 'results.csv'), 'w') as f:  # Make file object first
    csv_writer_object = csv.writer(f)  # Make csv writer object
    # writerow writes one row of data given as list object
    for info in total_score_info:
      csv_writer_object.writerow([item for _, item in info.items()])
    csv_writer_object.writerow([""])

    for filter in filters:
      infractions_types = []
      for infraction in route_scenarios[0]["infractions"]:
        infractions_types.append(infraction[0] + " mean")
        infractions_types.append(infraction[0] + " std")

      # route aggregation table has additional columns
      if filter == "route":
        csv_writer_object.writerow(
          [filter, "town", "DS mean", "DS std", "RC mean", "RC std", "NDS mean", "NDS std",
           "duration mean", "duration std", "length mean", "length std"] +
          infractions_types)
      else:
        csv_writer_object.writerow(
          [filter, "DS mean", "DS std", "RC mean", "RC std", "NDS mean", "NDS std", "duration mean", "duration std",
           "length mean", "length std"] +
          infractions_types)

      try:
        sorted_keys = sorted(evaluation_filtered[filter].keys(),
                             key=lambda fil: int(re.search('_(\\d+)_', fil).group(1)))
      except AttributeError:
        sorted_keys = sorted(evaluation_filtered[filter].keys())

      for key in sorted_keys:
        item = evaluation_filtered[filter][key]
        infractions_output = []
        for infraction in item["infractions"]:
          infractions_output.append(infraction[0])
          infractions_output.append(infraction[1])
        if filter == "route":
          csv_writer_object.writerow([key,
                                      route_matching[route_to_id[key]]["town"],
                                      item["DS"][0], item["DS"][1],
                                      item["RC"][0], item["RC"][1],
                                      item["NDS"][0], item["NDS"][1],
                                      item["duration"][0], item["duration"][1],
                                      item["length"][0], item["length"][1]] +
                                     infractions_output)
        else:
          csv_writer_object.writerow([key,
                                      item["DS"][0], item["DS"][1],
                                      item["RC"][0], item["RC"][1],
                                      item["NDS"][0], item["NDS"][1],
                                      item["duration"][0], item["duration"][1],
                                      item["length"][0], item["length"][1]] +
                                     infractions_output)
      csv_writer_object.writerow([""])

if __name__ == '__main__':
  main()
