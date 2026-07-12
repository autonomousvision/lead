import carla
import py_trees
from numpy import random
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorFlow, ScenarioTimeout, TrafficLightFreezer)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    CollisionTest, ScenarioTimeoutTest)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    DriveDistance, WaitEndIntersection)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import (ChangeOppositeBehavior,
                                              HandleJunctionScenario)
from srunner.tools.scenario_helper import (filter_junction_wp_direction,
                                           generate_target_waypoint,
                                           get_closest_traffic_light,
                                           get_junction_topology,
                                           get_same_dir_lanes)


class SequentialLaneChange(BasicScenario):

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=80, activate_scenario=True):
        """
        Setup all relevant parameters and create scenario
        """
        pass
        self._scenario_timeout = 240
        super().__init__("SequentialLaneChange",
                         ego_vehicles,
                         config,
                         world,
                         debug_mode,
                         criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Default initialization of other actors.
        Override this method in child class to provide custom initialization.
        """
        #super()._initialize_actors(config)
        from srunner.scenariomanager.carla_data_provider import ActiveScenario
        CarlaDataProvider.active_scenarios.append(ActiveScenario(type(self).__name__, scenario_id=id(self), trigger_location=config.trigger_points[0].location)) # added

    def _create_behavior(self):
        sequence = py_trees.composites.Sequence(name="SequentialLaneChange")
        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(DriveDistance(self.ego_vehicles[0], 200))
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        sequence.add_child(end_condition)

        from srunner.tools.background_manager import ClearScenarioType
        sequence.add_child(ClearScenarioType(id(self)))
        return sequence
        pass

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = [ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name)]
        if not self.route_mode:
            criteria.append(CollisionTest(self.ego_vehicles[0]))
        return criteria
        pass

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        super().__del__()
