# CARLA Route Generator

<p align="center">
  <img src="images/demo.gif" alt="Alt Text" width="75%">
</p>

The CARLA Route Generator is a Python application that provides a graphical user interface for creating and editing routes, as well as defining scenarios within the CARLA simulator. It can also be used in conjunction with CARLA Leaderboard 2.0.

**Features**
- **Route Management:** Load, save, and manipulate routes by adding or removing waypoints.
- **Scenario Definition:** Define scenarios along routes, and specifying their attributes.
- **Visualization:** Visualize the map, routes, waypoints, stop signs, and traffic lights.
- **Multi-Route Support:** Add, remove, and switch between multiple routes within the same map.

**Note:** When loading files containing many routes, such as the training or validation routes released by the CARLA team, the program can take quite a while to load them. This is because each route is planned using the CARLA route planner, which is computationally intensive. As workaround you can use the script [split_big_route_files.py](scripts/split_big_route_files.py) to first split the routes into multiple files and then open the desired route. We recommend saving your created routes frequently, as CARLA can occasionally crash, which may result in loss of unsaved routes.

## Instructions
The code has been tested on Linux and requires Python 3.10+.

## Setup

1. **Clone the repository:**
```Shell
git clone git@github.com:autonomousvision/carla_route_generator.git
```

**Set up CARLA:** Choose one of the following options:
1. Use an existing CARLA simulator (at least version 0.9.14)
2. Run the automated setup script:
```Shell
bash setup_carla.sh
```
3. Follow the official CARLA documentation to set up CARLA version 0.9.14 or 0.9.15 (https://carla.readthedocs.io/en/0.9.15/start_quickstart/#carla-0912)

**Build the Conda Environment:**
```Shell
cd carla_route_generator
conda env create -f environment.yml
conda activate carla_route_generator
```

## Usage
**Start the CARLA simulator:**
```Shell
cd path/to/CARLA
bash CarlaUE4.sh (-carla-rpc-port=<carla-port>)
```

**Start the program:**
```Another shell
bash start_window.sh (--host=<carla-host> --port=<carla-port>)
```

**Tool Usage:**
- **Add a route point** Left-click on the route.
- **Remove a Route Point:** Left-click close (<10 m) to an existing route point.
- **Add a Scenario:** Right-click close to the route.
- **Remove a Scenario:** Right-click close (<10 m) to an existing scenario.
- **Display the New Route:** Hold the mouse location still for 0.5 seconds.
- **Move the Map:** Press the mouse wheel and move the mouse.
- **Zoom In and Out:** Scroll the mouse wheel.

## CARLA Map Data
This tool uses pre-generated map data for improved efficiency. The files for the following towns are already provided: Town01, Town02, Town03, Town04, Town05, Town06, Town07, Town10, Town11, Town12, Town13, Town15.
If you need to use a different town, you can generate the map data using the provided script:
```Shell
cd path/to/CARLA
bash CarlaUE4.sh (-carla-rpc-port=<carla-port>)
```
```Another shell
cd path/to/carla_route_generator
python3 scripts/save_carla_map_data.py (--host=<carla-host> --port=<carla-port> --output-dir=<output-directory>)
```

## Contributing
Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.