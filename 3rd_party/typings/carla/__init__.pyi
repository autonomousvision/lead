from __future__ import annotations

from typing import Any
import numpy as np

import command

class Actor:
	"""
	CARLA defines actors as anything that plays a role in the simulation or can be moved around. That includes: pedestrians, vehicles, sensors and traffic signs (considering traffic lights as part of these). Actors are spawned in the simulation by carla.World and they need for a carla.ActorBlueprint to be created. These blueprints belong into a library provided by CARLA, find more about them [here](bp_library.md).
	"""

	attributes: dict
	"""A dictionary containing the attributes of the blueprint this actor was based on."""

	id: int
	"""Identifier for this actor. Unique during a given episode."""

	type_id: str
	"""The identifier of the blueprint this actor was based on, e.g. `vehicle.ford.mustang`."""

	is_alive: bool
	"""Returns whether this object was destroyed using this actor handle."""

	is_active: bool
	"""Returns whether this actor is active (True) or not (False)."""

	is_dormant: bool
	"""Returns whether this actor is dormant (True) or not (False) - the opposite of is_active."""

	parent: Actor
	"""Actors may be attached to a parent actor that they will follow around. This is said actor."""

	semantic_tags: list[int]
	"""A list of semantic tags provided by the blueprint listing components for this actor. E.g. a traffic light could be tagged with `Pole` and `TrafficLight`. These tags are used by the semantic segmentation sensor. Find more about this and other sensors [here](ref_sensors.md#semantic-segmentation-camera)."""

	actor_state: ActorState
	"""Returns the carla.ActorState, which can identify if the actor is Active, Dormant or Invalid."""

	bounding_box: BoundingBox
	"""Bounding box containing the geometry of the actor. Its location and rotation are relative to the actor it is attached to."""

	def add_angular_impulse(self, angular_impulse: Vector3D):
		"""
		Applies an angular impulse at the center of mass of the actor. This method should be used for instantaneous torques, usually applied once. Use add_torque() to apply rotation forces over a period of time.

		:param angular_impulse: (Vector3D) Angular impulse vector in global coordinates.
		"""
		...

	def add_force(self, force: Vector3D):
		"""
		Applies a force at the center of mass of the actor. This method should be used for forces that are applied over a certain period of time. Use add_impulse() to apply an impulse that only lasts an instant.

		:param force: (Vector3D) Force vector in global coordinates.
		"""
		...

	def add_impulse(self, impulse: Vector3D):
		"""
		Applies an impulse at the center of mass of the actor. This method should be used for instantaneous forces, usually applied once. Use add_force() to apply forces over a period of time.

		:param impulse: (Vector3D) Impulse vector in global coordinates.
		"""
		...

	def add_torque(self, torque: Vector3D):
		"""
		Applies a torque at the center of mass of the actor. This method should be used for torques that are applied over a certain period of time. Use add_angular_impulse() to apply a torque that only lasts an instant.

		:param torque: (Vector3D) Torque vector in global coordinates.
		"""
		...

	def destroy(self) -> bool:
		"""
		Tells the simulator to destroy this actor and returns True if it was successful. It has no effect if it was already destroyed.

		**warning**: This method blocks the script until the destruction is completed by the simulator.

		:return: bool
		"""
		...

	def disable_constant_velocity(self):
		"""
		Disables any constant velocity previously set for a carla.Vehicle actor.
		"""
		...

	def enable_constant_velocity(self, velocity: Vector3D):
		"""
		Sets a vehicle's velocity vector to a constant value over time. The resulting velocity will be approximately the `velocity` being set, as with set_target_velocity().

		*note*: Only carla.Vehicle actors can use this method.

		**warning**: Enabling a constant velocity for a vehicle managed by the [Traffic Manager](https://carla.readthedocs.io/en/latest/adv_traffic_manager/) may cause conflicts. This method overrides any changes in velocity by the TM.

		:param velocity: (Vector3D) Velocity vector in local space.
		"""
		...

	def get_acceleration(self) -> Vector3D:
		"""
		Returns the actor's 3D acceleration vector the client recieved during last tick. The method does not call the simulator.

		:return: carla.Vector3D
		"""
		...

	def get_angular_velocity(self) -> Vector3D:
		"""
		Returns the actor's angular velocity vector the client recieved during last tick. The method does not call the simulator.

		:return: carla.Vector3D
		"""
		...

	def get_location(self) -> Location:
		"""
		Returns the actor's location the client recieved during last tick. The method does not call the simulator.

		:return: carla.Location
		"""
		...

	def get_transform(self) -> Transform:
		"""
		Returns the actor's transform (location and rotation) the client recieved during last tick. The method does not call the simulator.

		:return: carla.Transform
		"""
		...

	def get_velocity(self) -> Vector3D:
		"""
		Returns the actor's velocity vector the client recieved during last tick. The method does not call the simulator.

		:return: carla.Vector3D
		"""
		...

	def get_world(self) -> World:
		"""
		Returns the world this actor belongs to.

		:return: carla.World
		"""
		...

	def set_target_angular_velocity(self, angular_velocity: Vector3D):
		"""
		Sets the actor's angular velocity vector. This is applied before the physics step so the resulting angular velocity will be affected by external forces such as friction.

		:param angular_velocity: (Vector3D) 
		"""
		...

	def set_location(self, location: Location):
		"""
		Teleports the actor to a given location.

		:param location: (Location) 
		"""
		...

	def set_simulate_physics(self, enabled: bool = True):
		"""
		Enables or disables the simulation of physics on this actor.

		:param enabled: (bool) 
		"""
		...

	def set_transform(self, transform: Transform):
		"""
		Teleports the actor to a given transform (location and rotation).

		:param transform: (Transform) 
		"""
		...

	def set_target_velocity(self, velocity: Vector3D):
		"""
		Sets the actor's velocity vector. This is applied before the physics step so the resulting angular velocity will be affected by external forces such as friction.

		:param velocity: (Vector3D) 
		"""
		...

	def __str__(self):
		"""

		"""
		...

	def set_enable_gravity(self, enabled: bool):
		"""
		Enables or disables gravity for the actor. __Default__ is True.

		:param enabled: (bool) 
		"""
		...



class ActorState:
	"""
	Class that defines the state of an actor.
	"""

	Invalid: Any
	"""An actor is Invalid if a problem has occurred."""

	Active: Any
	"""An actor is Active when it visualized and can affect other actors."""

	Dormant: Any
	"""An actor is Dormant when it is not visualized and will not affect other actors through physics. For example, actors are dormant if they are on an unloaded tile in a large map."""



class VehicleLightState:
	"""
	Class that recaps the state of the lights of a vehicle, these can be used as a flags. E.g: `VehicleLightState.HighBeam & VehicleLightState.Brake` will return `True` when both are active. Lights are off by default in any situation and should be managed by the user via script. The blinkers blink automatically. _Warning: Right now, not all vehicles have been prepared to work with this functionality, this will be added to all of them in later updates_
	"""

	NONE: Any
	"""All lights off"""

	Position: Any
	""""""

	LowBeam: Any
	""""""

	HighBeam: Any
	""""""

	Brake: Any
	""""""

	RightBlinker: Any
	""""""

	LeftBlinker: Any
	""""""

	Reverse: Any
	""""""

	Fog: Any
	""""""

	Interior: Any
	""""""

	Special1: Any
	"""This is reserved for certain vehicles that can have special lights, like a siren."""

	Special2: Any
	"""This is reserved for certain vehicles that can have special lights, like a siren."""

	All: Any
	"""All lights on"""



class Vehicle(Actor):
	"""
	One of the most important groups of actors in CARLA. These include any type of vehicle from cars to trucks, motorbikes, vans, bycicles and also official vehicles such as police cars. A wide set of these actors is provided in carla.BlueprintLibrary to facilitate differente requirements. Vehicles can be either manually controlled or set to an autopilot mode that will be conducted client-side by the traffic manager.
	"""

	bounding_box: BoundingBox
	"""Bounding box containing the geometry of the vehicle. Its location and rotation are relative to the vehicle it is attached to."""

	def apply_control(self, control: VehicleControl):
		"""
		Applies a control object on the next tick, containing driving parameters such as throttle, steering or gear shifting.

		:param control: (VehicleControl) 
		"""
		...

	def apply_ackermann_control(self, control: VehicleAckermannControl):
		"""
		Applies an Ackermann control object on the next tick.

		:param control: (VehicleAckermannControl) 
		"""
		...

	def apply_ackermann_controller_settings(self, settings: AckermannControllerSettings):
		"""
		Applies a new Ackermann control settings to this vehicle in the next tick.

		**warning**: This method does call the simulator.

		:param settings: (AckermannControllerSettings) 
		"""
		...

	def get_ackermann_controller_settings(self) -> AckermannControllerSettings:
		"""
		Returns the last Ackermann control settings applied to this vehicle.

		**warning**: This method does call the simulator to retrieve the value.

		:return: carla.AckermannControllerSettings
		"""
		...

	def apply_physics_control(self, physics_control: VehiclePhysicsControl):
		"""
		Applies a physics control object in the next tick containing the parameters that define the vehicle as a corporeal body. E.g.: moment of inertia, mass, drag coefficient and many more.

		:param physics_control: (VehiclePhysicsControl) 
		"""
		...

	def is_at_traffic_light(self) -> bool:
		"""
		Vehicles will be affected by a traffic light when the light is red and the vehicle is inside its bounding box. The client returns whether a traffic light is affecting this vehicle according to last tick (it does not call the simulator).

		:return: bool
		"""
		...

	def get_control(self) -> VehicleControl:
		"""
		The client returns the control applied in the last tick. The method does not call the simulator.

		:return: carla.VehicleControl
		"""
		...

	def get_light_state(self) -> VehicleLightState:
		"""
		Returns a flag representing the vehicle light state, this represents which lights are active or not.

		:return: carla.VehicleLightState
		"""
		...

	def get_physics_control(self) -> VehiclePhysicsControl:
		"""
		The simulator returns the last physics control applied to this vehicle.

		**warning**: This method does call the simulator to retrieve the value.

		:return: carla.VehiclePhysicsControl
		"""
		...

	def get_speed_limit(self) -> float:
		"""
		The client returns the speed limit affecting this vehicle according to last tick (it does not call the simulator). The speed limit is updated when passing by a speed limit signal, so a vehicle might have none right after spawning.

		:return: float
		"""
		...

	def get_traffic_light(self) -> TrafficLight:
		"""
		Retrieves the traffic light actor affecting this vehicle (if any) according to last tick. The method does not call the simulator.

		:return: carla.TrafficLight
		"""
		...

	def get_traffic_light_state(self) -> TrafficLightState:
		"""
		The client returns the state of the traffic light affecting this vehicle according to last tick. The method does not call the simulator. If no traffic light is currently affecting the vehicle, returns green.

		:return: carla.TrafficLightState
		"""
		...

	def enable_carsim(self, simfile_path: str):
		"""
		Enables the CarSim physics solver for this particular vehicle. In order for this function to work, there needs to be a valid license manager running on the server side. The control inputs are redirected to CarSim which will provide the position and orientation of the vehicle for every frame.

		:param simfile_path: (str) Path to the `.simfile` file with the parameters of the simulation.
		"""
		...

	def use_carsim_road(self, enabled: bool):
		"""
		Enables or disables the usage of CarSim vs terrain file specified in the `.simfile`. By default this option is disabled and CarSim uses unreal engine methods to process the geometry of the scene.

		:param enabled: (bool) 
		"""
		...

	def enable_chrono_physics(self, max_substeps: int, max_substep_delta_time: int, vehicle_json: str, powertrain_json: str, tire_json: str, base_json_path: str):
		"""
		Enables Chrono physics on a spawned vehicle.

		*note*: Ensure that you have started the CARLA server with the `ARGS="--chrono"` flag. You will not be able to use Chrono physics without this flag set.

		**warning**: Collisions are not supported. When a collision is detected, physics will revert to the default CARLA physics.

		:param max_substeps: (int) Max number of Chrono substeps

		:param max_substep_delta_time: (int) Max size of substep

		:param vehicle_json: (str) Path to vehicle json file relative to `base_json_path`

		:param powertrain_json: (str) Path to powertrain json file relative to `base_json_path`

		:param tire_json: (str) Path to tire json file relative to `base_json_path`

		:param base_json_path: (str) Path to `chrono/data/vehicle` folder. E.g., `/home/user/carla/Build/chrono-install/share/chrono/data/vehicle/` (the final `/` character is required).
		"""
		...

	def set_autopilot(self, enabled: bool = True, port: np.uint16 = 8000):
		"""
		Registers or deletes the vehicle from a Traffic Manager's list. When __True__, the Traffic Manager passed as parameter will move the vehicle around. The autopilot takes place client-side.

		:param enabled: (bool) 

		:param port: (np.uint16) The port of the TM-Server where the vehicle is to be registered or unlisted. If __None__ is passed, it will consider a TM at default port `8000`.
		"""
		...

	def set_light_state(self, light_state: VehicleLightState):
		"""
		Sets the light state of a vehicle using a flag that represents the lights that are on and off.

		:param light_state: (VehicleLightState) 
		"""
		...

	def __str__(self):
		"""

		"""
		...

	def set_wheel_steer_direction(self, wheel_location: VehicleWheelLocation, angle_in_deg: float):
		"""
		Sets the angle of a vehicle's wheel visually.

		**warning**: Does not affect the physics of the vehicle.

		:param wheel_location: (VehicleWheelLocation) 

		:param angle_in_deg: (float) 
		"""
		...

	def get_wheel_steer_angle(self, wheel_location: VehicleWheelLocation) -> float:
		"""
		Returns the physics angle in degrees of a vehicle's wheel.

		*note*: Returns the angle based on the physics of the wheel, not the visual angle.

		:param wheel_location: (VehicleWheelLocation) 

		:return: float
		"""
		...

	def get_failure_state(self) -> VehicleFailureState:
		"""
		Vehicle have failure states, to  indicate that it is incapable of continuing its route. This function returns the vehicle's specific failure state, or in other words, the cause that resulted in it.

		:return: carla.VehicleFailureState
		"""
		...

	def show_debug_telemetry(self, enabled: bool = True):
		"""
		Enables or disables the telemetry on this vehicle. This shows information about the vehicles current state and forces applied to it in the spectator window. Only information for one vehicle can be shown so that, if you enable a second one, the previous will be automatically disabled.

		:param enabled: (bool) 
		"""
		...

	def open_door(self, door_idx: VehicleDoor):
		"""
		Open the door `door_idx` if the vehicle has it. Use carla.VehicleDoor.All to open all available doors.

		:param door_idx: (VehicleDoor) door index
		"""
		...

	def close_door(self, door_idx: VehicleDoor):
		"""
		Close the door `door_idx` if the vehicle has it. Use carla.VehicleDoor.All to close all available doors.

		:param door_idx: (VehicleDoor) door index
		"""
		...



class Walker(Actor):
	"""
	This class inherits from the carla.Actor and defines pedestrians in the simulation. Walkers are a special type of actor that can be controlled either by an AI (carla.WalkerAIController) or manually via script, using a series of carla.WalkerControl to move these and their skeletons.
	"""

	def apply_control(self, control: WalkerControl):
		"""
		On the next tick, the control will move the walker in a certain direction with a certain speed. Jumps can be commanded too.

		:param control: (WalkerControl) 
		"""
		...

	def get_control(self) -> WalkerControl:
		"""
		The client returns the control applied to this walker during last tick. The method does not call the simulator.

		:return: carla.WalkerControl
		"""
		...

	def get_bones(self) -> WalkerBoneControlOut:
		"""
		Return the structure with all the bone transformations from the actor. For each bone, we get the name and its transform in three different spaces:
  - name: bone name
  - world: transform in world coordinates
  - component: transform based on the pivot of the actor
  - relative: transform based on the bone parent

		:return: carla.WalkerBoneControlOut
		"""
		...

	def set_bones(self, bones: WalkerBoneControlIn):
		"""
		Set the bones of the actor. For each bone we want to set we use a relative transform. Only the bones in this list will be set. For each bone you need to setup this info:
  - name: bone name
  - relative: transform based on the bone parent

		:param bones: (WalkerBoneControlIn) 
		"""
		...

	def blend_pose(self, blend_value: float):
		"""
		Set the blending value of the custom pose with the animation. The values can be:
  - 0: will show only the animation
  - 1: will show only the custom pose (set by the user with set_bones())
  - any other: will interpolate all the bone positions between animation and the custom pose

		:param blend_value: (float) 
		"""
		...

	def show_pose(self):
		"""
		Show the custom pose and hide the animation (same as calling blend_pose(1))
		"""
		...

	def hide_pose(self):
		"""
		Hide the custom pose and show the animation (same as calling blend_pose(0))
		"""
		...

	def get_pose_from_animation(self):
		"""
		Make a copy of the current animation frame as the custom pose. Initially the custom pose is the neutral pedestrian pose.
		"""
		...

	def __str__(self):
		"""

		"""
		...



class WalkerAIController(Actor):
	"""
	Class that conducts AI control for a walker. The controllers are defined as actors, but they are quite different from the rest. They need to be attached to a parent actor during their creation, which is the walker they will be controlling (take a look at carla.World if you are yet to learn on how to spawn actors). They also need for a special blueprint (already defined in carla.BlueprintLibrary as "controller.ai.walker"). This is an empty blueprint, as the AI controller will be invisible in the simulation but will follow its parent around to dictate every step of the way.
	"""

	def go_to_location(self, destination: Location):
		"""
		Sets the destination that the pedestrian will reach.

		:param destination: (Location) 
		"""
		...

	def start(self):
		"""
		Enables AI control for its parent walker.
		"""
		...

	def stop(self):
		"""
		Disables AI control for its parent walker.
		"""
		...

	def set_max_speed(self, speed: float = 1.4):
		"""
		Sets a speed for the walker in meters per second.

		:param speed: (float) An easy walking speed is set by default.
		"""
		...

	def __str__(self):
		"""

		"""
		...



class TrafficSign(Actor):
	"""
	Traffic signs appearing in the simulation except for traffic lights. These have their own class inherited from this in carla.TrafficLight. Right now, speed signs, stops and yields are mainly the ones implemented, but many others are borne in mind.
	"""

	trigger_volume: Any
	"""A carla.BoundingBox situated near a traffic sign where the carla.Actor who is inside can know about it."""



class TrafficLightState:
	"""
	All possible states for traffic lights. These can either change at a specific time step or be changed manually. The snipet in carla.TrafficLight.set_state changes the state of a traffic light on the fly.
	"""

	Red: Any
	""""""

	Yellow: Any
	""""""

	Green: Any
	""""""

	Off: Any
	""""""

	Unknown: Any
	""""""



class TrafficLight(TrafficSign):
	"""
	A traffic light actor, considered a specific type of traffic sign. As traffic lights will mostly appear at junctions, they belong to a group which contains the different traffic lights in it. Inside the group, traffic lights are differenciated by their pole index.
     
  Within a group the state of traffic lights is changed in a cyclic pattern: one index is chosen and it spends a few seconds in green, yellow and eventually red. The rest of the traffic lights remain frozen in red this whole time, meaning that there is a gap in the last seconds of the cycle where all the traffic lights are red. However, the state of a traffic light can be changed manually.
	"""

	state: TrafficLightState
	"""Current state of the traffic light."""

	def freeze(self, freeze: bool):
		"""
		Stops all the traffic lights in the scene at their current state.

		:param freeze: (bool) 
		"""
		...

	def is_frozen(self) -> bool:
		"""
		The client returns True if a traffic light is frozen according to last tick. The method does not call the simulator.

		:return: bool
		"""
		...

	def get_elapsed_time(self) -> float:
		"""
		The client returns the time in seconds since current light state started according to last tick. The method does not call the simulator.

		:return: float
		"""
		...

	def get_group_traffic_lights(self) -> list[TrafficLight]:
		"""
		Returns all traffic lights in the group this one belongs to.

		*note*: This method calls the simulator.

		:return: list(carla.TrafficLight)
		"""
		...

	def reset_group(self):
		"""
		Resets the state of the traffic lights of the group to the initial state at the start of the simulation.

		*note*: This method calls the simulator.
		"""
		...

	def get_pole_index(self) -> int:
		"""
		Returns the index of the pole that identifies it as part of the traffic light group of a junction.

		:return: int
		"""
		...

	def get_state(self) -> TrafficLightState:
		"""
		The client returns the state of the traffic light according to last tick. The method does not call the simulator.

		:return: carla.TrafficLightState
		"""
		...

	def get_green_time(self) -> float:
		"""
		The client returns the time set for the traffic light to be green, according to last tick. The method does not call the simulator.

		:return: float
		"""
		...

	def get_red_time(self) -> float:
		"""
		The client returns the time set for the traffic light to be red, according to last tick. The method does not call the simulator.

		:return: float
		"""
		...

	def get_yellow_time(self) -> float:
		"""
		The client returns the time set for the traffic light to be yellow, according to last tick. The method does not call the simulator.

		:return: float
		"""
		...

	def set_state(self, state: TrafficLightState):
		"""
		Sets a given state to a traffic light actor.

		:param state: (TrafficLightState) 
		"""
		...

	def set_green_time(self, green_time: float):
		"""


		:param green_time: (float) Sets a given time for the green light to be active.
		"""
		...

	def set_red_time(self, red_time: float):
		"""
		Sets a given time for the red state to be active.

		:param red_time: (float) 
		"""
		...

	def set_yellow_time(self, yellow_time: float):
		"""
		Sets a given time for the yellow light to be active.

		:param yellow_time: (float) 
		"""
		...

	def get_affected_lane_waypoints(self) -> list[Waypoint]:
		"""
		Returns a list of waypoints indicating the positions and lanes where the traffic light is having an effect.

		:return: list(carla.Waypoint)
		"""
		...

	def get_light_boxes(self) -> list[BoundingBox]:
		"""
		Returns a list of the bounding boxes encapsulating each light box of the traffic light.

		:return: list(carla.BoundingBox)
		"""
		...

	def get_opendrive_id(self) -> str:
		"""
		Returns the OpenDRIVE id of this traffic light.

		:return: str
		"""
		...

	def get_stop_waypoints(self) -> list[Waypoint]:
		"""
		Returns a list of waypoints indicating the stop position for the traffic light. These waypoints are computed from the trigger boxes of the traffic light that indicate where a vehicle should stop.

		:return: list(carla.Waypoint)
		"""
		...

	def __str__(self):
		"""

		"""
		...



class VehicleWheelLocation:
	"""
	`enum` representing the position of each wheel on a vehicle.  Used to identify the target wheel when setting an angle in carla.Vehicle.set_wheel_steer_direction or carla.Vehicle.get_wheel_steer_angle.
	"""

	FL_Wheel: Any
	"""Front left wheel of a 4 wheeled vehicle."""

	FR_Wheel: Any
	"""Front right wheel of a 4 wheeled vehicle."""

	BL_Wheel: Any
	"""Back left wheel of a 4 wheeled vehicle."""

	BR_Wheel: Any
	"""Back right wheel of a 4 wheeled vehicle."""

	Front_Wheel: Any
	"""Front wheel of a 2 wheeled vehicle."""

	Back_Wheel: Any
	"""Back wheel of a 2 wheeled vehicle."""



class VehicleDoor:
	"""
	Possible index representing the possible doors that can be open. Notice that not all possible doors are able to open in some vehicles.
	"""

	FL: Any
	"""Front left door."""

	FR: Any
	"""Front right door."""

	RL: Any
	"""Back left door."""

	RR: Any
	"""Back right door."""

	All: Any
	"""Represents all doors."""



class VehicleFailureState:
	"""
	Enum containing the different failure states of a vehicle, from which the it cannot recover. These are returned by get_failure_state() and only Rollover is currently implemented.
	"""

	NONE: Any
	""""""

	Rollover: Any
	""""""

	Engine: Any
	""""""

	TirePuncture: Any
	""""""



from __future__ import annotations

from typing import Any
import numpy as np

import command

class ActorAttributeType:
	"""
	CARLA provides a library of blueprints for actors in carla.BlueprintLibrary with different attributes each. This class defines the types those at carla.ActorAttribute can be as a series of enum. All this information is managed internally and listed here for a better comprehension of how CARLA works.
	"""

	Bool: Any
	""""""

	Int: Any
	""""""

	Float: Any
	""""""

	String: Any
	""""""

	RGBColor: Any
	""""""



class Color:
	"""
	Class that defines a 32-bit RGBA color.
	"""

	r: int
	"""Red color (0-255)."""

	g: int
	"""Green color (0-255)."""

	b: int
	"""Blue color (0-255)."""

	a: int
	"""Alpha channel (0-255)."""

	def __init__(self, r: int = 0, g: int = 0, b: int = 0, a: int = 255):
		"""
		Initializes a color, black by default.

		:param r: (int) 

		:param g: (int) 

		:param b: (int) 

		:param a: (int) 
		"""
		...

	def __eq__(self, other: Color):
		"""


		:param other: (Color) 
		"""
		...

	def __ne__(self, other: Color):
		"""


		:param other: (Color) 
		"""
		...

	def __str__(self):
		"""

		"""
		...



class FloatColor:
	"""
	Class that defines a float RGBA color.
	"""

	r: float
	"""Red color."""

	g: float
	"""Green color."""

	b: float
	"""Blue color."""

	a: float
	"""Alpha channel."""

	def __init__(self, r: float = 0, g: float = 0, b: float = 0, a: float = 1.0):
		"""
		Initializes a color, black by default.

		:param r: (float) 

		:param g: (float) 

		:param b: (float) 

		:param a: (float) 
		"""
		...

	def __eq__(self, other: FloatColor):
		"""


		:param other: (FloatColor) 
		"""
		...

	def __ne__(self, other: FloatColor):
		"""


		:param other: (FloatColor) 
		"""
		...



class OpticalFlowPixel:
	"""
	Class that defines a 2 dimensional vector representing an optical flow pixel.
	"""

	x: float
	"""Optical flow in the x component."""

	y: float
	"""Optical flow in the y component."""

	def __init__(self, x: float = 0, y: float = 0):
		"""
		Initializes the Optical Flow Pixel. Zero by default.

		:param x: (float) 

		:param y: (float) 
		"""
		...

	def __eq__(self, other: OpticalFlowPixel):
		"""


		:param other: (OpticalFlowPixel) 
		"""
		...

	def __ne__(self, other: OpticalFlowPixel):
		"""


		:param other: (OpticalFlowPixel) 
		"""
		...

	def __str__(self):
		"""

		"""
		...



class ActorAttribute:
	"""
	CARLA provides a library of blueprints for actors that can be accessed as carla.BlueprintLibrary. Each of these blueprints has a series of attributes defined internally. Some of these are modifiable, others are not. A list of recommended values is provided for those that can be set.
	"""

	id: str
	"""The attribute's name and identifier in the library."""

	is_modifiable: bool
	"""It is True if the attribute's value can be modified."""

	recommended_values: list[str]
	"""A list of values suggested by those who designed the blueprint."""

	type: ActorAttributeType
	"""The attribute's parameter type."""

	def as_bool(self):
		"""
		Reads the attribute as boolean value.
		"""
		...

	def as_color(self):
		"""
		Reads the attribute as carla.Color.
		"""
		...

	def as_float(self):
		"""
		Reads the attribute as float.
		"""
		...

	def as_int(self):
		"""
		Reads the attribute as int.
		"""
		...

	def as_str(self):
		"""
		Reads the attribute as string.
		"""
		...

	def __bool__(self):
		"""

		"""
		...

	def __float__(self):
		"""

		"""
		...

	def __int__(self):
		"""

		"""
		...

	def __str__(self):
		"""

		"""
		...

	def __eq__(self, other: str) -> bool:
		"""
		Returns true if this actor's attribute and `other` are the same.

		:param other: (str) 

		:return: bool
		"""
		...

	def __ne__(self, other: str) -> bool:
		"""
		Returns true if this actor's attribute and `other` are different.

		:param other: (str) 

		:return: bool
		"""
		...

	def __nonzero__(self) -> bool:
		"""
		Returns true if this actor's attribute is not zero or null.

		:return: bool
		"""
		...



class ActorBlueprint:
	"""
	CARLA provides a blueprint library for actors that can be consulted through carla.BlueprintLibrary. Each of these consists of an identifier for the blueprint and a series of attributes that may be modifiable or not. This class is the intermediate step between the library and the actor creation. Actors need an actor blueprint to be spawned. These store the information for said blueprint in an object with its attributes and some tags to categorize them. The user can then customize some attributes and eventually spawn the actors through carla.World.
	"""

	id: str
	"""The identifier of said blueprint inside the library. E.g. `walker.pedestrian.0001`."""

	tags: list[str]
	"""A list of tags each blueprint has that helps describing them. E.g. `['0001', 'pedestrian', 'walker']`."""

	def has_attribute(self, id: str) -> bool:
		"""
		Returns True if the blueprint contains the attribute `id`.

		:param id: (str) e.g. `gender` would return **True** for pedestrians' blueprints.

		:return: bool
		"""
		...

	def has_tag(self, tag: str) -> bool:
		"""
		Returns True if the blueprint has the specified `tag` listed.

		:param tag: (str) e.g. 'walker'

		:return: bool
		"""
		...

	def match_tags(self, wildcard_pattern: str) -> bool:
		"""
		Returns True if any of the tags listed for this blueprint matches `wildcard_pattern`. Matching follows [fnmatch](https://docs.python.org/2/library/fnmatch.html) standard.

		:param wildcard_pattern: (str) 

		:return: bool
		"""
		...

	def get_attribute(self, id: str) -> ActorAttribute:
		"""
		Returns the actor's attribute with `id` as identifier if existing.

		:param id: (str) 

		:return: carla.ActorAttribute
		"""
		...

	def set_attribute(self, id: str, value: str):
		"""
		If the `id` attribute is modifiable, changes its value to `value`.

		:param id: (str) The identifier for the attribute that is intended to be changed.

		:param value: (str) The new value for said attribute.
		"""
		...

	def __iter__(self):
		"""
		Iterate over the carla.ActorAttribute that this blueprint has.
		"""
		...

	def __len__(self):
		"""
		Returns the amount of attributes for this blueprint.
		"""
		...

	def __str__(self):
		"""

		"""
		...



class BlueprintLibrary:
	"""
	A class that contains the blueprints provided for actor spawning. Its main application is to return carla.ActorBlueprint objects needed to spawn actors. Each blueprint has an identifier and attributes that may or may not be modifiable. The library is automatically created by the server and can be accessed through carla.World.

  [Here](bp_library.md) is a reference containing every available blueprint and its specifics.
	"""

	def filter(self, wildcard_pattern: str) -> BlueprintLibrary:
		"""
		Filters a list of blueprints matching the `wildcard_pattern` against the id and tags of every blueprint contained in this library and returns the result as a new one. Matching follows [fnmatch](https://docs.python.org/2/library/fnmatch.html) standard.

		:param wildcard_pattern: (str) 

		:return: carla.BlueprintLibrary
		"""
		...

	def filter_by_attribute(self, name: str, value: str) -> BlueprintLibrary:
		"""
		Filters a list of blueprints with a given attribute matching the `value` against every blueprint contained in this library and returns the result as a new one. Matching follows [fnmatch](https://docs.python.org/2/library/fnmatch.html) standard.

		:param name: (str) 

		:param value: (str) 

		:return: carla.BlueprintLibrary
		"""
		...

	def find(self, id: str) -> ActorBlueprint:
		"""
		Returns the blueprint corresponding to that identifier.

		:param id: (str) 

		:return: carla.ActorBlueprint
		"""
		...

	def __getitem__(self, pos: int) -> ActorBlueprint:
		"""
		Returns the blueprint stored in `pos` position inside the data structure containing them.

		:param pos: (int) 

		:return: carla.ActorBlueprint
		"""
		...

	def __iter__(self):
		"""
		Iterate over the carla.ActorBlueprint stored in the library.
		"""
		...

	def __len__(self) -> int:
		"""
		Returns the amount of blueprints comprising the library.

		:return: int
		"""
		...

	def __str__(self) -> str:
		"""
		Parses the identifiers for every blueprint to string.

		:return: string
		"""
		...



from __future__ import annotations

from typing import Any
import numpy as np

import command

class Client:
	"""
	The Client connects CARLA to the server which runs the simulation. Both server and client contain a CARLA library (libcarla) with some differences that allow communication between them. Many clients can be created and each of these will connect to the RPC server inside the simulation to send commands. The simulation runs server-side. Once the connection is established, the client will only receive data retrieved from the simulation. Walkers are the exception. The client is in charge of managing pedestrians so, if you are running a simulation with multiple clients, some issues may arise. For example, if you spawn walkers through different clients, collisions may happen, as each client is only aware of the ones it is in charge of.

  The client also has a recording feature that saves all the information of a simulation while running it. This allows the server to replay it at will to obtain information and experiment with it. [Here](adv_recorder.md) is some information about how to use this recorder.
	"""

	def __init__(self, host: str = "127.0.0.1", port: int = 2000, worker_threads: int = 0):
		"""
		Client constructor

		:param host: (str) IP address where a CARLA Simulator instance is running. Default is localhost (127.0.0.1).

		:param port: (int) TCP port where the CARLA Simulator instance is running. Default are 2000 and the subsequent 2001.

		:param worker_threads: (int) Number of working threads used for background updates. If 0, use all available concurrency.
		"""
		...

	def apply_batch(self, commands: list):
		"""
		Executes a list of commands on a single simulation step and retrieves no information. If you need information about the response of each command, use the apply_batch_sync() method. [Here](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/generate_traffic.py) is an example on how to delete the actors that appear in carla.ActorList all at once.

		:param commands: (list) A list of commands to execute in batch. Each command is different and has its own parameters. They appear listed at the bottom of this page.
		"""
		...

	def apply_batch_sync(self, commands: list, due_tick_cue: bool = False) -> list[command.Response]:
		"""
		Executes a list of commands on a single simulation step, blocks until the commands are linked, and returns a list of command.Response that can be used to determine whether a single command succeeded or not. [Here](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/generate_traffic.py) is an example of it being used to spawn actors.

		:param commands: (list) A list of commands to execute in batch. The commands available are listed right above, in the method **apply_batch()**.

		:param due_tick_cue: (bool) A boolean parameter to specify whether or not to perform a carla.World.tick after applying the batch in _synchronous mode_. It is __False__ by default.

		:return: list(command.Response)
		"""
		...

	def generate_opendrive_world(self, opendrive: str, parameters: OpendriveGenerationParameters = (2.0, 50.0, 1.0, 0.6, true, true), reset_settings: bool = True):
		"""
		Loads a new world with a basic 3D topology generated from the content of an OpenDRIVE file. This content is passed as a `string` parameter. It is similar to `client.load_world(map_name)` but allows for custom OpenDRIVE maps in server side. Cars can drive around the map, but there are no graphics besides the road and sidewalks.

		:param opendrive: (str) Content of an OpenDRIVE file as `string`, __not the path to the `.xodr`__.

		:param parameters: (OpendriveGenerationParameters) Additional settings for the mesh generation. If none are provided, default values will be used.

		:param reset_settings: (bool) Option to reset the episode setting to default values, set to false to keep the current settings. This is useful to keep sync mode when changing map and to keep deterministic scenarios.
		"""
		...

	def load_world(self, map_name: str, reset_settings: bool = True, map_layers: MapLayer = MapLayer):
		"""
		Creates a new world with default settings using `map_name` map. All actors in the current world will be destroyed.

		**warning**: `map_layers` are only available for "Opt" maps

		:param map_name: (str) Name of the map to be used in this world. Accepts both full paths and map names, e.g. '/Game/Carla/Maps/Town01' or 'Town01'. Remember that these paths are dynamic.

		:param reset_settings: (bool) Option to reset the episode setting to default values, set to false to keep the current settings. This is useful to keep sync mode when changing map and to keep deterministic scenarios.

		:param map_layers: (MapLayer) Layers of the map that will be loaded. By default all layers are loaded. This parameter works like a flag mask.
		"""
		...

	def reload_world(self, reset_settings: bool = True):
		"""
		Reload the current world, note that a new world is created with default settings using the same map. All actors present in the world will be destroyed, __but__ traffic manager instances will stay alive.

		:param reset_settings: (bool) Option to reset the episode setting to default values, set to false to keep the current settings. This is useful to keep sync mode when changing map and to keep deterministic scenarios.
		"""
		...

	def replay_file(self, name: str, start: float, duration: float, follow_id: int, replay_sensors: bool):
		"""
		Load a new world with default settings using `map_name` map. All actors present in the current world will be destroyed, __but__ traffic manager instances will stay alive.

		:param name: (str) Name of the file containing the information of the simulation.

		:param start: (float) Time where to start playing the simulation. Negative is read as beginning from the end, being -10 just 10 seconds before the recording finished.

		:param duration: (float) Time that will be reenacted using the information `name` file. If the end is reached, the simulation will continue.

		:param follow_id: (int) ID of the actor to follow. If this is 0 then camera is disabled.

		:param replay_sensors: (bool) Flag to enable or disable the spawn of sensors during playback.
		"""
		...

	def stop_replayer(self, keep_actors: bool):
		"""
		Stop current replayer.

		:param keep_actors: (bool) True if you want autoremove all actors from the replayer, or False to keep them.
		"""
		...

	def show_recorder_actors_blocked(self, filename: str, min_time: float, min_distance: float) -> str:
		"""
		The terminal will show the information registered for actors considered blocked. An actor is considered blocked when it does not move a minimum distance in a period of time, being these `min_distance` and `min_time`.

		:param filename: (str) Name of the recorded file to load

		:param min_time: (float) Minimum time the actor has to move a minimum distance before being considered blocked. Default is 60 seconds.

		:param min_distance: (float) Minimum distance the actor has to move to not be considered blocked. Default is 100 centimeters.

		:return: string
		"""
		...

	def show_recorder_collisions(self, filename: str, category1: str, category2: str) -> str:
		"""
		The terminal will show the collisions registered by the recorder. These can be filtered by specifying the type of actor involved. The categories will be specified in `category1` and `category2` as follows:
  'h' = Hero, the one vehicle that can be controlled manually or managed by the user.
  'v' = Vehicle
  'w' = Walker
  't' = Traffic light
  'o' = Other
  'a' = Any
If you want to see only collisions between a vehicles and a walkers, use for `category1` as 'v' and `category2` as 'w' or vice versa. If you want to see all the collisions (filter off) you can use 'a' for both parameters.

		:param filename: (str) Name or absolute path of the file recorded, depending on your previous choice.

		:param category1: (str) Character variable specifying a first type of actor involved in the collision.

		:param category2: (str) Character variable specifying the second type of actor involved in the collision.

		:return: string
		"""
		...

	def show_recorder_file_info(self, filename: str, show_all: bool) -> str:
		"""
		The information saved by the recorder will be parsed and shown in your terminal as text (frames, times, events, state, positions...). The information shown can be specified by using the `show_all` parameter. [Here](ref_recorder_binary_file_format.md) is some more information about how to read the recorder file.

		:param filename: (str) Name or absolute path of the file recorded, depending on your previous choice.

		:param show_all: (bool) If __True__, returns all the information stored for every frame (traffic light states, positions of all actors, orientation and animation data...). If __False__, returns a summary of key events and frames.

		:return: string
		"""
		...

	def start_recorder(self, filename: str, additional_data: bool = False):
		"""
		Enables the recording feature, which will start saving every information possible needed by the server to replay the simulation.

		:param filename: (str) Name of the file to write the recorded data. A simple name will save the recording in 'CarlaUE4/Saved/recording.log'. Otherwise, if some folder appears in the name, it will be considered an absolute path.

		:param additional_data: (bool) Enables or disable recording non-essential data for reproducing the simulation (bounding box location, physics control parameters, etc)
		"""
		...

	def stop_recorder(self):
		"""
		Stops the recording in progress. If you specified a path in `filename`, the recording will be there. If not, look inside `CarlaUE4/Saved/`.
		"""
		...

	def get_available_maps(self) -> list[str]:
		"""
		Returns a list of strings containing the paths of the maps available on server. These paths are dynamic, they will be created during the simulation and so you will not find them when looking up in your files. One of the possible returns for this method would be:
  ['/Game/Carla/Maps/Town01',
  '/Game/Carla/Maps/Town02',
  '/Game/Carla/Maps/Town03',
  '/Game/Carla/Maps/Town04',
  '/Game/Carla/Maps/Town05',
  '/Game/Carla/Maps/Town06',
  '/Game/Carla/Maps/Town07']

		:return: list(str)
		"""
		...

	def get_client_version(self) -> str:
		"""
		Returns the client libcarla version by consulting it in the "Version.h" file. Both client and server can use different libcarla versions but some issues may arise regarding unexpected incompatibilities.

		:return: str
		"""
		...

	def get_server_version(self) -> str:
		"""
		Returns the server libcarla version by consulting it in the "Version.h" file. Both client and server should use the same libcarla version.

		:return: str
		"""
		...

	def get_trafficmanager(self, client_connection: int = 8000) -> TrafficManager:
		"""
		Returns an instance of the traffic manager related to the specified port. If it does not exist, this will be created.

		:param client_connection: (int) Port that will be used by the traffic manager. Default is `8000`.

		:return: carla.TrafficManager
		"""
		...

	def get_world(self) -> World:
		"""
		Returns the world object currently active in the simulation. This world will be later used for example to load maps.

		:return: carla.World
		"""
		...

	def set_replayer_time_factor(self, time_factor: float = 1.0):
		"""
		When used, the time speed of the reenacted simulation is modified at will. It can be used several times while a playback is in curse.

		:param time_factor: (float) 1.0 means normal time speed. Greater than 1.0 means fast motion (2.0 would be double speed) and lesser means slow motion (0.5 would be half speed).
		"""
		...

	def set_timeout(self, seconds: float):
		"""
		Sets the maximum time a network call is allowed before blocking it and raising a timeout exceeded error.

		:param seconds: (float) New timeout value. Default is 5 seconds.
		"""
		...

	def set_replayer_ignore_hero(self, ignore_hero: bool):
		"""


		:param ignore_hero: (bool) Enables or disables playback of the hero vehicle during a playback of a recorded simulation.
		"""
		...

	def set_replayer_ignore_spectator(self, ignore_spectator: bool):
		"""


		:param ignore_spectator: (bool) Determines whether the recorded spectator movements will be replicated by the replayer.
		"""
		...

	def set_files_base_folder(self, path: str):
		"""


		:param path: (str) Specifies the base folder where the local cache for required files will be placed.
		"""
		...

	def get_required_files(self, folder: str, download: bool = True):
		"""
		Asks the server which files are required by the client to use the current map. Option to download files automatically if they are not already in the cache.

		:param folder: (str) Folder where files required by the client will be downloaded to.

		:param download: (bool) If True, downloads files that are not already in cache.
		"""
		...

	def request_file(self, name: str):
		"""
		Requests one of the required files returned by carla.Client.get_required_files.

		:param name: (str) Name of the file you are requesting.
		"""
		...



class TrafficManager:
	"""
	The traffic manager is a module built on top of the CARLA API in C++. It handles any group of vehicles set to autopilot mode to populate the simulation with realistic urban traffic conditions and give the chance to user to customize some behaviours. The architecture of the traffic manager is divided in five different goal-oriented stages and a PID controller where the information flows until eventually, a carla.VehicleControl is applied to every vehicle registered in a traffic manager.
In order to learn more, visit the [documentation](adv_traffic_manager.md) regarding this module.
	"""

	def auto_lane_change(self, actor: Actor, enable: bool):
		"""
		Turns on or off lane changing behaviour for a vehicle.

		:param actor: (Actor) The vehicle whose settings are changed.

		:param enable: (bool) __True__ is default and enables lane changes. __False__ will disable them.
		"""
		...

	def collision_detection(self, reference_actor: Actor, other_actor: Actor, detect_collision: bool):
		"""
		Tunes on/off collisions between a vehicle and another specific actor. In order to ignore all other vehicles, traffic lights or walkers, use the specific __ignore__ methods described in this same section.

		:param reference_actor: (Actor) Vehicle that is going to ignore collisions.

		:param other_actor: (Actor) The actor that `reference_actor` is going to ignore collisions with.

		:param detect_collision: (bool) __True__ is default and enables collisions. __False__ will disable them.
		"""
		...

	def distance_to_leading_vehicle(self, actor: Actor, distance: float):
		"""
		Sets the minimum distance in meters that a vehicle has to keep with the others. The distance is in meters and will affect the minimum moving distance. It is computed from front to back of the vehicle objects.

		:param actor: (Actor) Vehicle whose minimum distance is being changed.

		:param distance: (float) Meters between both vehicles.
		"""
		...

	def force_lane_change(self, actor: Actor, direction: bool):
		"""
		Forces a vehicle to change either to the lane on its left or right, if existing, as indicated in `direction`. This method applies the lane change no matter what, disregarding possible collisions.

		:param actor: (Actor) Vehicle being forced to change lanes.

		:param direction: (bool) Destination lane. __True__ is the one on the right and __False__ is the left one.
		"""
		...

	def global_percentage_speed_difference(self, percentage: float):
		"""
		Sets the difference the vehicle's intended speed and its current speed limit. Speed limits can be exceeded by setting the `perc` to a negative value.
Default is 30. Exceeding a speed limit can be done using negative percentages.

		:param percentage: (float) Percentage difference between intended speed and the current limit.
		"""
		...

	def global_lane_offset(self, offset: float):
		"""
		Sets a global lane offset displacement from the center line. Positive values imply a right offset while negative ones mean a left one.
Default is 0. Numbers high enough to cause the vehicle to drive through other lanes might break the controller.

		:param offset: (float) Lane offset displacement from the center line.
		"""
		...

	def ignore_lights_percentage(self, actor: Actor, perc: float):
		"""
		During the traffic light stage, which runs every frame, this method sets the percent chance that traffic lights will be ignored for a vehicle.

		:param actor: (Actor) The actor that is going to ignore traffic lights.

		:param perc: (float) Between 0 and 100. Amount of times traffic lights will be ignored.
		"""
		...

	def ignore_signs_percentage(self, actor: Actor, perc: float):
		"""
		During the traffic light stage, which runs every frame, this method sets the percent chance that stop signs will be ignored for a vehicle.

		:param actor: (Actor) The actor that is going to ignore stop signs.

		:param perc: (float) Between 0 and 100. Amount of times stop signs will be ignored.
		"""
		...

	def ignore_vehicles_percentage(self, actor: Actor, perc: float):
		"""
		During the collision detection stage, which runs every frame, this method sets a percent chance that collisions with another vehicle will be ignored for a vehicle.

		:param actor: (Actor) The vehicle that is going to ignore other vehicles.

		:param perc: (float) Between 0 and 100. Amount of times collisions will be ignored.
		"""
		...

	def ignore_walkers_percentage(self, actor: Actor, perc: float):
		"""
		During the collision detection stage, which runs every frame, this method sets a percent chance that collisions with walkers will be ignored for a vehicle.

		:param actor: (Actor) The vehicle that is going to ignore walkers on scene.

		:param perc: (float) Between 0 and 100. Amount of times collisions will be ignored.
		"""
		...

	def vehicle_percentage_speed_difference(self, actor: Actor, percentage: float):
		"""
		Sets the difference the vehicle's intended speed and its current speed limit. Speed limits can be exceeded by setting the `perc` to a negative value.
Default is 30. Exceeding a speed limit can be done using negative percentages.

		:param actor: (Actor) Vehicle whose speed behaviour is being changed.

		:param percentage: (float) Percentage difference between intended speed and the current limit.
		"""
		...

	def vehicle_lane_offset(self, actor: Actor, offset: float):
		"""
		Sets a lane offset displacement from the center line. Positive values imply a right offset while negative ones mean a left one.
Default is 0. Numbers high enough to cause the vehicle to drive through other lanes might break the controller.

		:param actor: (Actor) Vehicle whose lane offset behaviour is being changed.

		:param offset: (float) Lane offset displacement from the center line.
		"""
		...

	def update_vehicle_lights(self, actor: Actor, do_update: bool):
		"""
		Sets if the Traffic Manager is responsible of updating the vehicle lights, or not.
Default is __False__. The traffic manager will not change the vehicle light status of a vehicle, unless its auto_update_status is st to __True__.

		:param actor: (Actor) Vehicle whose lights status is being changed.

		:param do_update: (bool) If __True__ the traffic manager will manage the vehicle lights for the specified vehicle.
		"""
		...

	def get_port(self) -> np.uint16:
		"""
		Returns the port where the Traffic Manager is connected. If the object is a TM-Client, it will return the port of its TM-Server. Read the [documentation](#adv_traffic_manager.md#multiclient-and-multitm-management) to learn the difference.

		:return: uint16
		"""
		...

	def set_global_distance_to_leading_vehicle(self, distance: float):
		"""
		Sets the minimum distance in meters that vehicles have to keep with the rest. The distance is in meters and will affect the minimum moving distance. It is computed from center to center of the vehicle objects.

		:param distance: (float) Meters between vehicles.
		"""
		...

	def set_desired_speed(self, actor: Actor, speed: float):
		"""
		Sets the speed of a vehicle to the specified value.

		:param actor: (Actor) Vehicle whose speed is being changed.

		:param speed: (float) Desired speed at which the vehicle will move.
		"""
		...

	def set_hybrid_physics_mode(self, enabled: bool = False):
		"""
		Enables or disables the hybrid physics mode. In this mode, vehicle's farther than a certain radius from the ego vehicle will have their physics disabled. Computation cost will be reduced by not calculating vehicle dynamics. Vehicles will be teleported.

		:param enabled: (bool) If __True__, enables the hybrid physics.
		"""
		...

	def set_hybrid_physics_radius(self, r: float = 50.0):
		"""
		With hybrid physics on, changes the radius of the area of influence where physics are enabled.

		:param r: (float) New radius where physics are enabled.
		"""
		...

	def set_osm_mode(self, mode_switch: bool = True):
		"""
		Enables or disables the OSM mode. This mode allows the user to run TM in a map created with the [OSM feature](tuto_G_openstreetmap.md). These maps allow having dead-end streets. Normally, if vehicles cannot find the next waypoint, TM crashes. If OSM mode is enabled, it will show a warning, and destroy vehicles when necessary.

		:param mode_switch: (bool) If __True__, the OSM mode is enabled.
		"""
		...

	def keep_right_rule_percentage(self, actor: Actor, perc: float):
		"""
		During the localization stage, this method sets a percent chance that vehicle will follow the *keep right* rule, and stay in the right lane.

		:param actor: (Actor) Vehicle whose behaviour is being changed.

		:param perc: (float) Between 0 and 100. Amount of times the vehicle will follow the keep right rule.
		"""
		...

	def set_random_device_seed(self, value: int):
		"""
		Sets a specific random seed for the Traffic Manager, thereby setting it to be deterministic.

		:param value: (int) Seed value for the random number generation of the Traffic Manager.
		"""
		...

	def set_synchronous_mode(self, mode_switch: bool = True):
		"""
		Sets the Traffic Manager to [synchronous mode](adv_traffic_manager.md#synchronous-mode). In a [multiclient situation](adv_traffic_manager.md#multiclient), only the TM-Server can tick. Similarly, in a [multiTM situation](adv_traffic_manager.md#multitm), only one TM-Server must tick. Use this method in the client that does the world tick, and right after setting the world to synchronous mode, to set which TM will be the master while in sync.

		**warning**: If the server is set to synchronous mode, the TM must be set to synchronous mode too in the same client that does the tick.

		:param mode_switch: (bool) If __True__, the TM synchronous mode is enabled.
		"""
		...

	def set_respawn_dormant_vehicles(self, mode_switch: bool = False):
		"""
		If __True__, vehicles in large maps will respawn near the hero vehicle when they become dormant. Otherwise, they will stay dormant until they are within `actor_active_distance` of the hero vehicle again.

		:param mode_switch: (bool) 
		"""
		...

	def set_boundaries_respawn_dormant_vehicles(self, lower_bound: float = 25.0, upper_bound: float = actor_active_distance):
		"""
		Sets the upper and lower boundaries for dormant actors to be respawned near the hero vehicle.

		**warning**: The `upper_bound` cannot be higher than the `actor_active_distance`. The `lower_bound` cannot be less than 25.

		:param lower_bound: (float) The minimum distance in meters from the hero vehicle that a dormant actor will be respawned.

		:param upper_bound: (float) The maximum distance in meters from the hero vehicle that a dormant actor will be respawned.
		"""
		...

	def set_path(self, actor: Actor, path: list):
		"""
		Sets a list of locations for a vehicle to follow while controlled by the Traffic Manager.

		**warning**: Ensure that the road topology doesn't impede the given path.

		:param actor: (Actor) The actor that must follow the given path

		:param path: (list) The list of carla.Locations for the actor to follow
		"""
		...

	def set_route(self, actor: Actor, path: list):
		"""
		Sets a list of route instructions for a vehicle to follow while controlled by the Traffic Manager. The possible route instructions are 'Left', 'Right', 'Straight'.

		**warning**: Ensure that the lane topology doesn't impede the given route.

		:param actor: (Actor) The actor that must follow the given route instructions

		:param path: (list) The list of route instructions (string) for the vehicle to follow.
		"""
		...

	def get_next_action(self, actor: Actor) -> str:
		"""
		Returns the next known road option and waypoint that an actor controlled by the Traffic Manager will follow.

		:param actor: (Actor) The actor that you wish to query.

		:return: list of two elements - [Road option (string e.g. 'Left', 'Right', 'Straight'), Next waypoint (carla.Waypoint)]
		"""
		...

	def get_all_actions(self, actor: Actor) -> str:
		"""
		Returns all known actions (i.e. road options and waypoints) that an actor controlled by the Traffic Manager will perform in its next steps.

		:param actor: (Actor) The actor that you wish to query.

		:return: list of lists with each element as follows - [Road option (string e.g. 'Left', 'Right', 'Straight'), Next waypoint (carla.Waypoint)]
		"""
		...

	def random_left_lanechange_percentage(self, actor: Actor, percentage: float):
		"""
		Adjust probability that in each timestep the actor will perform a left lane change, dependent on lane change availability.

		:param actor: (Actor) The actor that you wish to query.

		:param percentage: (float) The probability of lane change in percentage units (between 0 and 100)
		"""
		...

	def random_right_lanechange_percentage(self, actor: Actor, percentage: float):
		"""
		Adjust probability that in each timestep the actor will perform a right lane change, dependent on lane change availability.

		:param actor: (Actor) The actor that you wish to query.

		:param percentage: (float) The probability of lane change in percentage units (between 0 and 100)
		"""
		...

	def shut_down(self):
		"""
		Shuts down the traffic manager.
		"""
		...



class OpendriveGenerationParameters:
	"""
	This class defines the parameters used when generating a world using an OpenDRIVE file.
	"""

	vertex_distance: float
	"""Distance between vertices of the mesh generated. __Default is `2.0`__."""

	max_road_length: float
	"""Max road length for a single mesh portion. The mesh of the map is divided into portions, in order to avoid propagating issues. __Default is `50.0`__."""

	wall_height: float
	"""Height of walls created on the boundaries of the road. These prevent vehicles from falling off the road. __Default is `1.0`__."""

	additional_width: float
	"""Additional with applied junction lanes. Complex situations tend to occur at junctions, and a little increase can prevent vehicles from falling off the road.  __Default is `0.6`__."""

	smooth_junctions: bool
	"""If __True__, the mesh at junctions will be smoothed to prevent issues where roads blocked other roads. __Default is `True`__."""

	enable_mesh_visibility: bool
	"""If __True__, the road mesh will be rendered. Setting this to __False__ should reduce the rendering overhead.  __Default is `True`__."""

	enable_pedestrian_navigation: bool
	"""If __True__, Pedestrian navigation will be enabled using Recast tool. For very large maps it is recomended to disable this option. __Default is `True`__."""



from __future__ import annotations

from typing import Any
import numpy as np

import command

class VehicleControl:
	"""
	Manages the basic movement of a vehicle using typical driving controls.
	"""

	throttle: float
	"""A scalar value to control the vehicle throttle [0.0, 1.0]. Default is 0.0."""

	steer: float
	"""A scalar value to control the vehicle steering [-1.0, 1.0]. Default is 0.0."""

	brake: float
	"""A scalar value to control the vehicle brake [0.0, 1.0]. Default is 0.0."""

	hand_brake: bool
	"""Determines whether hand brake will be used. Default is False."""

	reverse: bool
	"""Determines whether the vehicle will move backwards. Default is False."""

	manual_gear_shift: bool
	"""Determines whether the vehicle will be controlled by changing gears manually. Default is False."""

	gear: int
	"""States which gear is the vehicle running on."""

	def __init__(self, throttle: float = 0.0, steer: float = 0.0, brake: float = 0.0, hand_brake: bool = False, reverse: bool = False, manual_gear_shift: bool = False, gear: int = 0):
		"""


		:param throttle: (float) Scalar value between [0.0,1.0]

		:param steer: (float) Scalar value between [0.0,1.0]

		:param brake: (float) Scalar value between [0.0,1.0]

		:param hand_brake: (bool) 

		:param reverse: (bool) 

		:param manual_gear_shift: (bool) 

		:param gear: (int) 
		"""
		...

	def __eq__(self, other: VehicleControl):
		"""


		:param other: (VehicleControl) 
		"""
		...

	def __ne__(self, other: VehicleControl):
		"""


		:param other: (VehicleControl) 
		"""
		...

	def __str__(self):
		"""

		"""
		...



class VehicleAckermannControl:
	"""
	Manages the basic movement of a vehicle using Ackermann driving controls.
	"""

	steer: float
	"""Desired steer (rad). Positive value is to the right. Default is 0.0."""

	steer_speed: float
	"""Steering velocity (rad/s). Zero steering angle velocity means change the steering angle as quickly as possible. Default is 0.0."""

	speed: float
	"""Desired speed (m/s). Default is 0.0."""

	acceleration: float
	"""Desired acceleration (m/s2) Default is 0.0."""

	jerk: float
	"""Desired jerk (m/s3). Default is 0.0."""

	def __init__(self, steer: float = 0.0, steer_speed: float = 0.0, speed: float = 0.0, acceleration: float = 0.0, jerk: float = 0.0):
		"""


		:param steer: (float) 

		:param steer_speed: (float) 

		:param speed: (float) 

		:param acceleration: (float) 

		:param jerk: (float) 
		"""
		...

	def __eq__(self, other: AckermannVehicleControl):
		"""


		:param other: (AckermannVehicleControl) 
		"""
		...

	def __ne__(self, other: AckermannVehicleControl):
		"""


		:param other: (AckermannVehicleControl) 
		"""
		...

	def __str__(self):
		"""

		"""
		...



class AckermannControllerSettings:
	"""
	Manages the settings of the Ackermann PID controller.
	"""

	speed_kp: float
	"""Proportional term of the speed PID controller."""

	speed_ki: float
	"""Integral term of the speed PID controller."""

	speed_kd: float
	"""Derivative term of the speed PID controller."""

	accel_kp: float
	"""Proportional term of the acceleration PID controller."""

	accel_ki: float
	"""Integral term of the acceleration PID controller."""

	accel_kd: float
	"""Derivative term of the acceleration PID controller."""

	def __init__(self, speed_kp: float = 0.15, speed_ki: float = 0.0, speed_kd: float = 0.25, accel_kp: float = 0.01, accel_ki: float = 0.0, accel_kd: float = 0.01):
		"""


		:param speed_kp: (float) 

		:param speed_ki: (float) 

		:param speed_kd: (float) 

		:param accel_kp: (float) 

		:param accel_ki: (float) 

		:param accel_kd: (float) 
		"""
		...

	def __eq__(self, other: AckermannControllerSettings):
		"""


		:param other: (AckermannControllerSettings) 
		"""
		...

	def __ne__(self, other: AckermannControllerSettings):
		"""


		:param other: (AckermannControllerSettings) 
		"""
		...

	def __str__(self):
		"""

		"""
		...



class WalkerControl:
	"""
	This class defines specific directions that can be commanded to a carla.Walker to control it via script.
  
  AI control can be settled for walkers, but the control used to do so is carla.WalkerAIController.
	"""

	direction: Vector3D
	"""Vector using global coordinates that will correspond to the direction of the walker."""

	speed: float
	"""A scalar value to control the walker's speed."""

	jump: bool
	"""If True, the walker will perform a jump."""

	def __init__(self, direction: Vector3D = [1.0, 0.0, 0.0], speed: float = 0.0, jump: bool = False):
		"""


		:param direction: (Vector3D) 

		:param speed: (float) 

		:param jump: (bool) 
		"""
		...

	def __eq__(self, other: WalkerControl):
		"""
		Compares every variable with `other` and returns True if these are all the same.

		:param other: (WalkerControl) 
		"""
		...

	def __ne__(self, other: WalkerControl):
		"""
		Compares every variable with `other` and returns True if any of these differ.

		:param other: (WalkerControl) 
		"""
		...

	def __str__(self):
		"""

		"""
		...



class WalkerBoneControlOut:
	"""
	This class is used to return all bone positions of a pedestrian. For each bone we get its _name_ and its transform in three different spaces (world, actor and relative).
	"""

	bone_transforms: str
	"""List of one entry per bone with this information:
  - name: bone name
  - world: transform in world coordinates
  - component: transform based on the pivot of the actor
  - relative: transform based on the bone parent"""

	def __str__(self):
		"""

		"""
		...



class WalkerBoneControlIn:
	"""
	This class grants bone specific manipulation for walker. The skeletons of walkers have been unified for clarity and the transform applied to each bone are always relative to its parent. Take a look [here](tuto_G_control_walker_skeletons.md) to learn more on how to create a walker and define its movement.
	"""

	bone_transforms: list[[name,transform]]
	"""List with the data for each bone we want to set:
  - name: bone name
  - relative: transform based on the bone parent"""

	def __init__(self, list: tuple):
		"""
		Initializes an object containing moves to be applied on tick. These are listed with the name of the bone and the transform that will be applied to it.

		:param list(name,transform): (tuple) 
		"""
		...

	def __str__(self):
		"""

		"""
		...



class GearPhysicsControl:
	"""
	Class that provides access to vehicle transmission details by defining a gear and when to run on it. This will be later used by carla.VehiclePhysicsControl to help simulate physics.
	"""

	ratio: float
	"""The transmission ratio of the gear."""

	down_ratio: float
	"""Quotient between current RPM and MaxRPM where the autonomous gear box should shift down."""

	up_ratio: float
	"""Quotient between current RPM and MaxRPM where the autonomous gear box should shift up."""

	def __init__(self, ratio: float = 1.0, down_ratio: float = 0.5, up_ratio: float = 0.65):
		"""


		:param ratio: (float) 

		:param down_ratio: (float) 

		:param up_ratio: (float) 
		"""
		...

	def __eq__(self, other: GearPhysicsControl):
		"""


		:param other: (GearPhysicsControl) 
		"""
		...

	def __ne__(self, other: GearPhysicsControl):
		"""


		:param other: (GearPhysicsControl) 
		"""
		...

	def __str__(self):
		"""

		"""
		...



class VehiclePhysicsControl:
	"""
	Summarizes the parameters that will be used to simulate a carla.Vehicle as a physical object. The specific settings for the wheels though are stipulated using carla.WheelPhysicsControl.
	"""

	torque_curve: list[Vector2D]
	"""Curve that indicates the torque measured in Nm for a specific RPM of the vehicle's engine."""

	max_rpm: float
	"""The maximum RPM of the vehicle's engine."""

	moi: float
	"""The moment of inertia of the vehicle's engine."""

	damping_rate_full_throttle: float
	"""Damping ratio when the throttle is maximum."""

	damping_rate_zero_throttle_clutch_engaged: float
	"""Damping ratio when the throttle is zero with clutch engaged."""

	damping_rate_zero_throttle_clutch_disengaged: float
	"""Damping ratio when the throttle is zero with clutch disengaged."""

	use_gear_autobox: bool
	"""If True, the vehicle will have an automatic transmission."""

	gear_switch_time: float
	"""Switching time between gears."""

	clutch_strength: float
	"""Clutch strength of the vehicle."""

	final_ratio: float
	"""Fixed ratio from transmission to wheels."""

	forward_gears: list[GearPhysicsControl]
	"""List of objects defining the vehicle's gears."""

	mass: float
	"""Mass of the vehicle."""

	drag_coefficient: float
	"""Drag coefficient of the vehicle's chassis."""

	center_of_mass: Vector3D
	"""Center of mass of the vehicle."""

	steering_curve: list[Vector2D]
	"""Curve that indicates the maximum steering for a specific forward speed."""

	use_sweep_wheel_collision: bool
	"""Enable the use of sweep for wheel collision. By default, it is disabled and it uses a simple raycast from the axis to the floor for each wheel. This option provides a better collision model in which the full volume of the wheel is checked against collisions."""

	wheels: list[WheelPhysicsControl]
	"""List of wheel physics objects. This list should have 4 elements, where index 0 corresponds to the front left wheel, index 1 corresponds to the front right wheel, index 2 corresponds to the back left wheel and index 3 corresponds to the back right wheel. For 2 wheeled vehicles, set the same values for both front and back wheels."""

	def __init__(self, torque_curve: list[Vector2D] = [[0.0, 500.0], [5000.0, 500.0]], max_rpm: float = 5000.0, moi: float = 1.0, damping_rate_full_throttle: float = 0.15, damping_rate_zero_throttle_clutch_engaged: float = 2.0, damping_rate_zero_throttle_clutch_disengaged: float = 0.35, use_gear_autobox: bool = True, gear_switch_time: float = 0.5, clutch_strength: float = 10.0, final_ratio: float = 4.0, forward_gears: list[GearPhysicsControl] = list(), drag_coefficient: float = 0.3, center_of_mass: Vector3D = [0.0, 0.0, 0.0], steering_curve: Vector2D = [[0.0, 1.0], [10.0, 0.5]], wheels: list[WheelPhysicsControl] = list(), use_sweep_wheel_collision: bool = False, mass: float = 1000.0):
		"""
		VehiclePhysicsControl constructor

		:param torque_curve: (list[Vector2D]) 

		:param max_rpm: (float) 

		:param moi: (float) 

		:param damping_rate_full_throttle: (float) 

		:param damping_rate_zero_throttle_clutch_engaged: (float) 

		:param damping_rate_zero_throttle_clutch_disengaged: (float) 

		:param use_gear_autobox: (bool) 

		:param gear_switch_time: (float) 

		:param clutch_strength: (float) 

		:param final_ratio: (float) 

		:param forward_gears: (list[GearPhysicsControl]) 

		:param drag_coefficient: (float) 

		:param center_of_mass: (Vector3D) 

		:param steering_curve: (Vector2D) 

		:param wheels: (list[WheelPhysicsControl]) 

		:param use_sweep_wheel_collision: (bool) 

		:param mass: (float) 
		"""
		...

	def __eq__(self, other: VehiclePhysicsControl):
		"""


		:param other: (VehiclePhysicsControl) 
		"""
		...

	def __ne__(self, other: VehiclePhysicsControl):
		"""


		:param other: (VehiclePhysicsControl) 
		"""
		...

	def __str__(self):
		"""

		"""
		...



class WheelPhysicsControl:
	"""
	Class that defines specific physical parameters for wheel objects that will be part of a carla.VehiclePhysicsControl to simulate vehicle it as a material object.
	"""

	tire_friction: float
	"""A scalar value that indicates the friction of the wheel."""

	damping_rate: float
	"""Damping rate of the wheel."""

	max_steer_angle: float
	"""Maximum angle that the wheel can steer."""

	radius: float
	"""Radius of the wheel."""

	max_brake_torque: float
	"""Maximum brake torque."""

	max_handbrake_torque: float
	"""Maximum handbrake torque."""

	position: Vector3D
	"""World position of the wheel. This is a read-only parameter."""

	long_stiff_value: float
	"""Tire longitudinal stiffness per unit gravitational acceleration. Each vehicle has a custom value."""

	lat_stiff_max_load: float
	"""Maximum normalized tire load at which the tire can deliver no more lateral stiffness no matter how much extra load is applied to the tire. Each vehicle has a custom value."""

	lat_stiff_value: float
	"""Maximum stiffness per unit of lateral slip. Each vehicle has a custom value."""

	def __init__(self, tire_friction: float = 2.0, damping_rate: float = 0.25, max_steer_angle: float = 70.0, radius: float = 30.0, max_brake_torque: float = 1500.0, max_handbrake_torque: float = 3000.0, position: Vector3D = (0.0,0.0,0.0)):
		"""


		:param tire_friction: (float) 

		:param damping_rate: (float) 

		:param max_steer_angle: (float) 

		:param radius: (float) 

		:param max_brake_torque: (float) 

		:param max_handbrake_torque: (float) 

		:param position: (Vector3D) 
		"""
		...

	def __eq__(self, other: WheelPhysicsControl):
		"""


		:param other: (WheelPhysicsControl) 
		"""
		...

	def __ne__(self, other: WheelPhysicsControl):
		"""


		:param other: (WheelPhysicsControl) 
		"""
		...

	def __str__(self):
		"""

		"""
		...



from __future__ import annotations

from typing import Any
import numpy as np

import command

class Vector2D:
	"""
	Helper class to perform 2D operations.
	"""

	x: float
	"""X-axis value."""

	y: float
	"""Y-axis value."""

	def __init__(self, x: float = 0.0, y: float = 0.0):
		"""


		:param x: (float) 

		:param y: (float) 
		"""
		...

	def length(self) -> float:
		"""
		Computes the length of the vector.

		:return: float
		"""
		...

	def squared_length(self) -> float:
		"""
		Computes the squared length of the vector.

		:return: float
		"""
		...

	def make_unit_vector(self) -> Vector3D:
		"""
		Returns a vector with the same direction and unitary length.

		:return: carla.Vector3D
		"""
		...

	def __add__(self, other: Vector2D):
		"""


		:param other: (Vector2D) 
		"""
		...

	def __sub__(self, other: Vector2D):
		"""


		:param other: (Vector2D) 
		"""
		...

	def __mul__(self, other: Vector2D):
		"""


		:param other: (Vector2D) 
		"""
		...

	def __truediv__(self, other: Vector2D):
		"""


		:param other: (Vector2D) 
		"""
		...

	def __eq__(self, other: Vector2D) -> bool:
		"""
		Returns __True__ if values for every axis are equal.

		:param other: (Vector2D) 

		:return: bool
		"""
		...

	def __ne__(self, bool: Vector2D) -> bool:
		"""
		Returns __True__ if the value for any axis is different.

		:param bool: (Vector2D) 

		:return: bool
		"""
		...

	def __str__(self) -> str:
		"""
		Returns the axis values for the vector parsed as string.

		:return: str
		"""
		...



class Vector3D:
	"""
	Helper class to perform 3D operations.
	"""

	x: float
	"""X-axis value."""

	y: float
	"""Y-axis value."""

	z: float
	"""Z-axis value."""

	def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
		"""


		:param x: (float) 

		:param y: (float) 

		:param z: (float) 
		"""
		...

	def length(self) -> float:
		"""
		Computes the length of the vector.

		:return: float
		"""
		...

	def squared_length(self) -> float:
		"""
		Computes the squared length of the vector.

		:return: float
		"""
		...

	def make_unit_vector(self) -> Vector3D:
		"""
		Returns a vector with the same direction and unitary length.

		:return: carla.Vector3D
		"""
		...

	def cross(self, vector: Vector3D) -> Vector3D:
		"""
		Computes the cross product between two vectors.

		:param vector: (Vector3D) 

		:return: carla.Vector3D
		"""
		...

	def dot(self, vector: Vector3D) -> float:
		"""
		Computes the dot product between two vectors.

		:param vector: (Vector3D) 

		:return: float
		"""
		...

	def distance(self, vector: Vector3D) -> float:
		"""
		Computes the distance between two vectors.

		:param vector: (Vector3D) 

		:return: float
		"""
		...

	def distance_squared(self, vector: Vector3D) -> float:
		"""
		Computes the squared distance between two vectors.

		:param vector: (Vector3D) 

		:return: float
		"""
		...

	def dot_2d(self, vector: Vector3D) -> float:
		"""
		Computes the 2-dimensional dot product between two vectors.

		:param vector: (Vector3D) 

		:return: float
		"""
		...

	def distance_2d(self, vector: Vector3D) -> float:
		"""
		Computes the 2-dimensional distance between two vectors.

		:param vector: (Vector3D) 

		:return: float
		"""
		...

	def distance_squared_2d(self, vector: Vector3D) -> float:
		"""
		Computes the 2-dimensional squared distance between two vectors.

		:param vector: (Vector3D) 

		:return: float
		"""
		...

	def get_vector_angle(self, vector: Vector3D) -> float:
		"""
		Computes the angle between a pair of 3D vectors in radians.

		:param vector: (Vector3D) 

		:return: float
		"""
		...

	def __add__(self, other: Vector3D):
		"""


		:param other: (Vector3D) 
		"""
		...

	def __sub__(self, other: Vector3D):
		"""


		:param other: (Vector3D) 
		"""
		...

	def __mul__(self, other: Vector3D):
		"""


		:param other: (Vector3D) 
		"""
		...

	def __truediv__(self, other: Vector3D):
		"""


		:param other: (Vector3D) 
		"""
		...

	def __eq__(self, other: Vector3D) -> bool:
		"""
		Returns __True__ if values for every axis are equal.

		:param other: (Vector3D) 

		:return: bool
		"""
		...

	def __ne__(self, other: Vector3D) -> bool:
		"""
		Returns __True__ if the value for any axis is different.

		:param other: (Vector3D) 

		:return: bool
		"""
		...

	def __str__(self) -> str:
		"""
		Returns the axis values for the vector parsed as string.

		:return: str
		"""
		...

	def __abs__(self) -> Vector3D:
		"""
		Returns a Vector3D with the absolute value of the components x, y and z.

		:return: carla.Vector3D
		"""
		...



class Location(Vector3D):
	"""
	Represents a spot in the world.
	"""

	x: float
	"""Distance from origin to spot on X axis."""

	y: float
	"""Distance from origin to spot on Y axis."""

	z: float
	"""Distance from origin to spot on Z axis."""

	def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
		"""


		:param x: (float) 

		:param y: (float) 

		:param z: (float) 
		"""
		...

	def distance(self, location: Location) -> float:
		"""
		Returns Euclidean distance from this location to another one.

		:param location: (Location) The other point to compute the distance with.

		:return: float
		"""
		...

	def __eq__(self, other: Location) -> bool:
		"""
		Returns __True__ if both locations are the same point in space.

		:param other: (Location) 

		:return: bool
		"""
		...

	def __ne__(self, other: Location) -> bool:
		"""
		Returns __True__ if both locations are different points in space.

		:param other: (Location) 

		:return: bool
		"""
		...

	def __str__(self) -> str:
		"""
		Parses the axis' values to string.

		:return: str
		"""
		...

	def __abs__(self) -> Location:
		"""
		Returns a Location with the absolute value of the components x, y and z.

		:return: carla.Location
		"""
		...



class Rotation:
	"""
	Class that represents a 3D rotation and therefore, an orientation in space. CARLA uses the Unreal Engine coordinates system. This is a Z-up left-handed system.  
The constructor method follows a specific order of declaration: `(pitch, yaw, roll)`, which corresponds to `(Y-rotation,Z-rotation,X-rotation)`.   ![UE4_Rotation](https://d26ilriwvtzlb.cloudfront.net/8/83/BRMC_9.jpg) *Unreal Engine's coordinates system*
	"""

	pitch: float
	"""Y-axis rotation angle."""

	yaw: float
	"""Z-axis rotation angle."""

	roll: float
	"""X-axis rotation angle."""

	def __init__(self, pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0):
		"""


		**warning**: The declaration order is different in CARLA (pitch,yaw,roll), and in the Unreal Engine Editor (roll,pitch,yaw). When working in a build from source, don't mix up the axes' rotations.

		:param pitch: (float) Y-axis rotation angle.

		:param yaw: (float) Z-axis rotation angle.

		:param roll: (float) X-axis rotation angle.
		"""
		...

	def get_forward_vector(self) -> Vector3D:
		"""
		Computes the vector pointing forward according to the rotation of the object.

		:return: carla.Vector3D
		"""
		...

	def get_right_vector(self) -> Vector3D:
		"""
		Computes the vector pointing to the right according to the rotation of the object.

		:return: carla.Vector3D
		"""
		...

	def get_up_vector(self) -> Vector3D:
		"""
		Computes the vector pointing upwards according to the rotation of the object.

		:return: carla.Vector3D
		"""
		...

	def __eq__(self, other: Rotation) -> bool:
		"""
		Returns __True__ if both rotations represent the same orientation for every axis.

		:param other: (Rotation) 

		:return: bool
		"""
		...

	def __ne__(self, other: Rotation) -> bool:
		"""
		Returns __True__ if both rotations represent the same orientation for every axis.

		:param other: (Rotation) 

		:return: bool
		"""
		...

	def __str__(self):
		"""
		Parses the axis' orientations to string.
		"""
		...



class Transform:
	"""
	Class that defines a transformation, a combination of location and rotation, without scaling.
	"""

	location: Location
	"""Describes a point in the coordinate system."""

	rotation: Rotation
	"""Describes a rotation for an object according to Unreal Engine's axis system."""

	def __init__(self, location: Location, rotation: Rotation):
		"""


		:param location: (Location) 

		:param rotation: (Rotation) 
		"""
		...

	def transform(self, in_point: Location):
		"""
		Translates a 3D point from local to global coordinates using the current transformation as frame of reference.

		:param in_point: (Location) Location in the space to which the transformation will be applied.
		"""
		...

	def transform_vector(self, in_vector: Vector3D):
		"""
		Rotates a vector using the current transformation as frame of reference, without applying translation. Use this to transform, for example, a velocity.

		:param in_vector: (Vector3D) Vector to which the transformation will be applied.
		"""
		...

	def get_forward_vector(self) -> Vector3D:
		"""
		Computes a forward vector using the rotation of the object.

		:return: carla.Vector3D
		"""
		...

	def get_right_vector(self) -> Vector3D:
		"""
		Computes a right vector using the rotation of the object.

		:return: carla.Vector3D
		"""
		...

	def get_up_vector(self) -> Vector3D:
		"""
		Computes an up vector using the rotation of the object.

		:return: carla.Vector3D
		"""
		...

	def get_matrix(self) -> list[list[float]]:
		"""
		Computes the 4-matrix representation of the transformation.

		:return: list(list(float))
		"""
		...

	def get_inverse_matrix(self) -> list[list[float]]:
		"""
		Computes the 4-matrix representation of the inverse transformation.

		:return: list(list(float))
		"""
		...

	def __eq__(self, other: Transform) -> bool:
		"""
		Returns __True__ if both location and rotation are equal for this and `other`.

		:param other: (Transform) 

		:return: bool
		"""
		...

	def __ne__(self, other: Transform) -> bool:
		"""
		Returns __True__ if any location and rotation are not equal for this and `other`.

		:param other: (Transform) 

		:return: bool
		"""
		...

	def __str__(self) -> str:
		"""
		Parses both location and rotation to string.

		:return: str
		"""
		...



class BoundingBox:
	"""
	Bounding boxes contain the geometry of an actor or an element in the scene. They can be used by carla.DebugHelper or a carla.Client to draw their shapes for debugging. Check out the snipet in carla.DebugHelper.draw_box where a snapshot of the world is used to draw bounding boxes for traffic lights.
	"""

	extent: Vector3D
	"""Vector from the center of the box to one vertex. The value in each axis equals half the size of the box for that axis.
`extent.x * 2` would return the size of the box in the X-axis."""

	location: Location
	"""The center of the bounding box."""

	rotation: Rotation
	"""The orientation of the bounding box."""

	def __init__(self, location: Location, extent: Vector3D):
		"""


		:param location: (Location) Center of the box, relative to its parent.

		:param extent: (Vector3D) Vector containing half the size of the box for every axis.
		"""
		...

	def contains(self, world_point: Location, transform: Transform) -> bool:
		"""
		Returns **True** if a point passed in world space is inside this bounding box.

		:param world_point: (Location) The point in world space to be checked.

		:param transform: (Transform) Contains location and rotation needed to convert this object's local space to world space.

		:return: bool
		"""
		...

	def get_local_vertices(self) -> list[Location]:
		"""
		Returns a list containing the locations of this object's vertices in local space.

		:return: list(carla.Location)
		"""
		...

	def get_world_vertices(self, transform: Transform) -> list[Location]:
		"""
		Returns a list containing the locations of this object's vertices in world space.

		:param transform: (Transform) Contains location and rotation needed to convert this object's local space to world space.

		:return: list(carla.Location)
		"""
		...

	def __eq__(self, other: BoundingBox) -> bool:
		"""
		Returns true if both location and extent are equal for this and `other`.

		:param other: (BoundingBox) 

		:return: bool
		"""
		...

	def __ne__(self, other: BoundingBox) -> bool:
		"""
		Returns true if either location or extent are different for this and `other`.

		:param other: (BoundingBox) 

		:return: bool
		"""
		...

	def __str__(self) -> str:
		"""
		Parses the location and extent of the bounding box to string.

		:return: str
		"""
		...



class GeoLocation:
	"""
	Class that contains geographical coordinates simulated data. The carla.Map can convert simulation locations by using the <georeference> tag in the OpenDRIVE file.
	"""

	latitude: float
	"""North/South value of a point on the map."""

	longitude: float
	"""West/East value of a point on the map."""

	altitude: float
	"""Height regarding ground level."""

	def __init__(self, latitude: float = 0.0, longitude: float = 0.0, altitude: float = 0.0):
		"""


		:param latitude: (float) 

		:param longitude: (float) 

		:param altitude: (float) 
		"""
		...

	def __eq__(self, other: GeoLocation):
		"""


		:param other: (GeoLocation) 
		"""
		...

	def __ne__(self, other: GeoLocation):
		"""


		:param other: (GeoLocation) 
		"""
		...

	def __str__(self):
		"""

		"""
		...



from __future__ import annotations

from typing import Any
import numpy as np

import command

class LightGroup:
	"""
	This class categorizes the lights on scene into different groups. These groups available are provided as a enum values that can be used as flags.  

__Note.__ So far, though there is a `vehicle` group, vehicle lights are not available as carla.Light objects. These have to be managed using carla.Vehicle and carla.VehicleLightState.
	"""

	None: Any
	"""All lights."""

	Vehicle: Any
	""""""

	Street: Any
	""""""

	Building: Any
	""""""

	Other: Any
	""""""



class LightState:
	"""
	This class represents all the light variables except the identifier and the location, which are should to be static. Using this class allows to manage all the parametrization of the light in one call.
	"""

	intensity: float
	"""Intensity of a light."""

	color: Color
	"""Color of a light."""

	group: LightGroup
	"""Group a light belongs to."""

	active: bool
	"""Switch of a light. It is __True__ when the light is on."""

	def __init__(self, intensity: float = 0.0, color: Color = Color(), group: LightGroup = LightGroup, active: bool = False):
		"""


		:param intensity: (float) Intensity of the light. Default is `0.0`.

		:param color: (Color) Color of the light. Default is black.

		:param group: (LightGroup) Group the light belongs to. Default is the generic group `None`.

		:param active: (bool) Swith of the light. Default is `False`, light is off.
		"""
		...



class Light:
	"""
	This class exposes the lights that exist in the scene, except for vehicle lights. The properties of a light can be queried and changed at will.
Lights are automatically turned on when the simulator enters night mode (sun altitude is below zero).
	"""

	color: Color
	"""Color of the light."""

	id: int
	"""Identifier of the light."""

	intensity: float
	"""Intensity of the light."""

	is_on: bool
	"""Switch of the light. It is __True__ when the light is on. When the night mode starts, this is set to __True__."""

	location: Location
	"""Position of the light."""

	light_group: LightGroup
	"""Group the light belongs to."""

	light_state: LightState
	"""State of the light. Summarizes its attributes, group, and if it is on/off."""

	def turn_off(self):
		"""
		Switches off the light.
		"""
		...

	def turn_on(self):
		"""
		Switches on the light.
		"""
		...

	def set_color(self, color: Color):
		"""
		Changes the color of the light to `color`.

		:param color: (Color) 
		"""
		...

	def set_intensity(self, intensity: float):
		"""
		Changes the intensity of the light to `intensity`.

		:param intensity: (float) 
		"""
		...

	def set_light_group(self, light_group: LightGroup):
		"""
		Changes the light to the group `light_group`.

		:param light_group: (LightGroup) 
		"""
		...

	def set_light_state(self, light_state: LightState):
		"""
		Changes the state of the light to `light_state`. This may change attributes, group and turn the light on/off all at once.

		:param light_state: (LightState) 
		"""
		...



class LightManager:
	"""
	This class handles the lights in the scene. Its main use is to get and set the state of groups or lists of lights in one call. An instance of this class can be retrieved by the carla.World.get_lightmanager().  

__Note.__ So far, though there is a `vehicle` group, vehicle lights are not available as carla.Light objects. These have to be managed using carla.Vehicle and carla.VehicleLightState.
	"""

	def is_active(self, lights: list[Light]) -> list[bool]:
		"""
		Returns a list with booleans stating if the elements in `lights` are switched on/off.

		:param lights: (list[Light]) List of lights to be queried.

		:return: list(bool)
		"""
		...

	def turn_off(self, lights: list[Light]):
		"""
		Switches off all the lights in `lights`.

		:param lights: (list[Light]) List of lights to be switched off.
		"""
		...

	def turn_on(self, lights: list[Light]):
		"""
		Switches on all the lights in `lights`.

		:param lights: (list[Light]) List of lights to be switched on.
		"""
		...

	def get_all_lights(self, light_group: LightGroup = LightGroup) -> list[Light]:
		"""
		Returns a list containing the lights in a certain group. By default, the group is `None`.

		:param light_group: (LightGroup) Group to filter the lights returned. Default is `None`.

		:return: list(carla.Light)
		"""
		...

	def get_color(self, lights: list[Light]) -> list[Color]:
		"""
		Returns a list with the colors of every element in `lights`.

		:param lights: (list[Light]) List of lights to be queried.

		:return: list(carla.Color)
		"""
		...

	def get_intensity(self, lights: list[Light]) -> list[float]:
		"""
		Returns a list with the intensity of every element in `lights`.

		:param lights: (list[Light]) List of lights to be queried.

		:return: list(float)
		"""
		...

	def get_light_group(self, lights: list[Light]) -> list[LightGroup]:
		"""
		Returns a list with the group of every element in `lights`.

		:param lights: (list[Light]) List of lights to be queried.

		:return: list(carla.LightGroup)
		"""
		...

	def get_light_state(self, lights: list[Light]) -> list[LightState]:
		"""
		Returns a list with the state of all the attributes of every element in `lights`.

		:param lights: (list[Light]) List of lights to be queried.

		:return: list(carla.LightState)
		"""
		...

	def get_turned_off_lights(self, light_group: LightGroup) -> list[Light]:
		"""
		Returns a list containing lights switched off in the scene, filtered by group.

		:param light_group: (LightGroup) List of lights to be queried.

		:return: list(carla.Light)
		"""
		...

	def get_turned_on_lights(self, light_group: LightGroup) -> list[Light]:
		"""
		Returns a list containing lights switched on in the scene, filtered by group.

		:param light_group: (LightGroup) List of lights to be queried.

		:return: list(carla.Light)
		"""
		...

	def set_active(self, lights: list[Light], active: list[bool]):
		"""
		Switches on/off the elements in `lights`.

		:param lights: (list[Light]) List of lights to be switched on/off.

		:param active: (list[bool]) List of booleans to be applied.
		"""
		...

	def set_color(self, lights: list[Light], color: Color):
		"""
		Changes the color of the elements in `lights` to `color`.

		:param lights: (list[Light]) List of lights to be changed.

		:param color: (Color) Color to be applied.
		"""
		...

	def set_colors(self, lights: list[Light], colors: list[Color]):
		"""
		Changes the color of each element in `lights` to the corresponding in `colors`.

		:param lights: (list[Light]) List of lights to be changed.

		:param colors: (list[Color]) List of colors to be applied.
		"""
		...

	def set_intensity(self, lights: list[Light], intensity: float):
		"""
		Changes the intensity of every element in `lights` to `intensity`.

		:param lights: (list[Light]) List of lights to be changed.

		:param intensity: (float) Intensity to be applied.
		"""
		...

	def set_intensities(self, lights: list[Light], intensities: list[float]):
		"""
		Changes the intensity of each element in `lights` to the corresponding in `intensities`.

		:param lights: (list[Light]) List of lights to be changed.

		:param intensities: (list[float]) List of intensities to be applied.
		"""
		...

	def set_light_group(self, lights: list[Light], light_group: LightGroup):
		"""
		Changes the group of every element in `lights` to `light_group`.

		:param lights: (list[Light]) List of lights to be changed.

		:param light_group: (LightGroup) Group to be applied.
		"""
		...

	def set_light_groups(self, lights: list[Light], light_groups: list[LightGroup]):
		"""
		Changes the group of each element in `lights` to the corresponding in `light_groups`.

		:param lights: (list[Light]) List of lights to be changed.

		:param light_groups: (list[LightGroup]) List of groups to be applied.
		"""
		...

	def set_light_state(self, lights: list[Light], light_state: LightState):
		"""
		Changes the state of the attributes of every element in `lights` to `light_state`.

		:param lights: (list[Light]) List of lights to be changed.

		:param light_state: (LightState) State of the attributes to be applied.
		"""
		...

	def set_light_states(self, lights: list[Light], light_states: list[LightState]):
		"""
		Changes the state of the attributes of each element in `lights` to the corresponding in `light_states`.

		:param lights: (list[Light]) List of lights to be changed.

		:param light_states: (list[LightState]) List of state of the attributes to be applied.
		"""
		...

	def set_day_night_cycle(self, active: bool):
		"""
		All scene lights have a day-night cycle, automatically turning on and off with the altitude of the sun. This interferes in cases where full control of the scene lights is required, so setting this to __False__ deactivates it. It can reactivated by setting it to __True__.

		:param active: (bool) (De)activation of the day-night cycle.
		"""
		...



from __future__ import annotations

from typing import Any
import numpy as np

import command

class LaneType:
	"""
	Class that defines the possible lane types accepted by OpenDRIVE 1.4. This standards define the road information. The snipet in carla.Map.get_waypoint makes use of a waypoint to get the current and adjacent lane types.
	"""

	NONE: Any
	""""""

	Driving: Any
	""""""

	Stop: Any
	""""""

	Shoulder: Any
	""""""

	Biking: Any
	""""""

	Sidewalk: Any
	""""""

	Border: Any
	""""""

	Restricted: Any
	""""""

	Parking: Any
	""""""

	Bidirectional: Any
	""""""

	Median: Any
	""""""

	Special1: Any
	""""""

	Special2: Any
	""""""

	Special3: Any
	""""""

	RoadWorks: Any
	""""""

	Tram: Any
	""""""

	Rail: Any
	""""""

	Entry: Any
	""""""

	Exit: Any
	""""""

	OffRamp: Any
	""""""

	OnRamp: Any
	""""""

	Any: Any
	"""Every type except for NONE."""



class LaneChange:
	"""
	Class that defines the permission to turn either left, right, both or none (meaning only going straight is allowed). This information is stored for every carla.Waypoint according to the OpenDRIVE file. The snipet in carla.Map.get_waypoint shows how a waypoint can be used to learn which turns are permitted.
	"""

	NONE: Any
	"""Traffic rules do not allow turning right or left, only going straight."""

	Right: Any
	"""Traffic rules allow turning right."""

	Left: Any
	"""Traffic rules allow turning left."""

	Both: Any
	"""Traffic rules allow turning either right or left."""



class LaneMarkingColor:
	"""
	Class that defines the lane marking colors according to OpenDRIVE 1.4.
	"""

	Standard: Any
	"""White by default."""

	Blue: Any
	""""""

	Green: Any
	""""""

	Red: Any
	""""""

	White: Any
	""""""

	Yellow: Any
	""""""

	Other: Any
	""""""



class LaneMarkingType:
	"""
	Class that defines the lane marking types accepted by OpenDRIVE 1.4. The snipet in carla.Map.get_waypoint shows how a waypoint can be used to retrieve the information about adjacent lane markings.    __Note on double types:__ Lane markings are defined under the OpenDRIVE standard that determines whereas a line will be considered "BrokenSolid" or "SolidBroken". For each road there is a center lane marking, defined from left to right regarding the lane's directions. The rest of the lane markings are defined in order from the center lane to the closest outside of the road.
	"""

	NONE: Any
	""""""

	Other: Any
	""""""

	Broken: Any
	""""""

	Solid: Any
	""""""

	SolidSolid: Any
	""""""

	SolidBroken: Any
	""""""

	BrokenSolid: Any
	""""""

	BrokenBroken: Any
	""""""

	BottsDots: Any
	""""""

	Grass: Any
	""""""

	Curb: Any
	""""""



class Map:
	"""
	Class containing the road information and waypoint managing. Data is retrieved from an OpenDRIVE file that describes the road. A query system is defined which works hand in hand with carla.Waypoint to translate geometrical information from the .xodr to natural world points. CARLA is currently working with [OpenDRIVE 1.4 standard](http://www.opendrive.org/docs/OpenDRIVEFormatSpecRev1.4H.pdf).
	"""

	name: str
	"""The name of the map. It corresponds to the .umap from Unreal Engine that is loaded from a CARLA server, which then references to the .xodr road description."""

	def __init__(self, name: str, xodr_content: str) -> list[Transform]:
		"""
		Constructor for this class. Though a map is automatically generated when initializing the world, using this method in no-rendering mode facilitates working with an .xodr without any CARLA server running.

		:param name: (str) Name of the current map.

		:param xodr_content: (str) .xodr content in string format.

		:return: list(carla.Transform)
		"""
		...

	def generate_waypoints(self, distance: float) -> list[Waypoint]:
		"""
		Returns a list of waypoints with a certain distance between them for every lane and centered inside of it. Waypoints are not listed in any particular order. Remember that waypoints closer than 2cm within the same road, section and lane will have the same identificator.

		:param distance: (float) Approximate distance between waypoints.

		:return: list(carla.Waypoint)
		"""
		...

	def save_to_disk(self, path: Any):
		"""
		Saves the .xodr OpenDRIVE file of the current map to disk.

		:param path: () Path where the file will be saved.
		"""
		...

	def to_opendrive(self) -> str:
		"""
		Returns the .xodr OpenDRIVe file of the current map as string.

		:return: str
		"""
		...

	def transform_to_geolocation(self, location: Location) -> GeoLocation:
		"""
		Converts a given `location`, a point in the simulation, to a carla.GeoLocation, which represents world coordinates. The geographical location of the map is defined inside OpenDRIVE within the tag <georeference>.

		:param location: (Location) 

		:return: carla.GeoLocation
		"""
		...

	def get_all_landmarks(self) -> list[Landmark]:
		"""
		Returns all the landmarks in the map. Landmarks retrieved using this method have a __null__ waypoint.

		:return: list(carla.Landmark)
		"""
		...

	def get_all_landmarks_from_id(self, opendrive_id: str) -> list[Landmark]:
		"""
		Returns the landmarks with a certain OpenDRIVE ID. Landmarks retrieved using this method have a __null__ waypoint.

		:param opendrive_id: (str) The OpenDRIVE ID of the landmarks.

		:return: list(carla.Landmark)
		"""
		...

	def get_all_landmarks_of_type(self, type: str) -> list[Landmark]:
		"""
		Returns the landmarks of a specific type. Landmarks retrieved using this method have a __null__ waypoint.

		:param type: (str) The type of the landmarks.

		:return: list(carla.Landmark)
		"""
		...

	def get_landmark_group(self, landmark: Landmark) -> list[Landmark]:
		"""
		Returns the landmarks in the same group as the specified landmark (including itself). Returns an empty list if the landmark does not belong to any group.

		:param landmark: (Landmark) A landmark that belongs to the group.

		:return: list(carla.Landmark)
		"""
		...

	def get_spawn_points(self) -> list[Transform]:
		"""
		Returns a list of recommendations made by the creators of the map to be used as spawning points for the vehicles. The list includes carla.Transform objects with certain location and orientation. Said locations are slightly on-air in order to avoid Z-collisions, so vehicles fall for a bit before starting their way.

		:return: list(carla.Transform)
		"""
		...

	def get_topology(self) -> str:
		"""
		Returns a list of tuples describing a minimal graph of the topology of the OpenDRIVE file. The tuples contain pairs of waypoints located either at the point a road begins or ends. The first one is the origin and the second one represents another road end that can be reached. This graph can be loaded into [NetworkX](https://networkx.github.io/) to work with. Output could look like this: [(w0, w1), (w0, w2), (w1, w3), (w2, w3), (w0, w4)].

		:return: list(tuple(carla.Waypoint, carla.Waypoint))
		"""
		...

	def get_waypoint(self, location: Location, project_to_road: bool = True, lane_type: LaneType = LaneType) -> Waypoint:
		"""
		Returns a waypoint that can be located in an exact location or translated to the center of the nearest lane. Said lane type can be defined using flags such as `LaneType.Driving & LaneType.Shoulder`.
 The method will return None if the waypoint is not found, which may happen only when trying to retrieve a waypoint for an exact location. That eases checking if a point is inside a certain road, as otherwise, it will return the corresponding waypoint.

		:param location: (Location) Location used as reference for the carla.Waypoint.

		:param project_to_road: (bool) If **True**, the waypoint will be at the center of the closest lane. This is the default setting. If **False**, the waypoint will be exactly in `location`. None means said location does not belong to a road.

		:param lane_type: (LaneType) Limits the search for nearest lane to one or various lane types that can be flagged.

		:return: carla.Waypoint
		"""
		...

	def get_waypoint_xodr(self, road_id: int, lane_id: int, s: float) -> Waypoint:
		"""
		Returns a waypoint if all the parameters passed are correct. Otherwise, returns __None__.

		:param road_id: (int) ID of the road to get the waypoint.

		:param lane_id: (int) ID of the lane to get the waypoint.

		:param s: (float) Specify the length from the road start.

		:return: carla.Waypoint
		"""
		...

	def get_crosswalks(self) -> list[Location]:
		"""
		Returns a list of locations with all crosswalk zones in the form of closed polygons. The first point is repeated, symbolizing where the polygon begins and ends.

		:return: list(carla.Location)
		"""
		...

	def cook_in_memory_map(self, path: str):
		"""
		Generates a binary file from the CARLA map containing information used by the Traffic Manager. This method is only used during the import process for maps.

		:param path: (str) Path to the intended location of the stored binary map file.
		"""
		...

	def __str__(self):
		"""

		"""
		...



class LaneMarking:
	"""
	Class that gathers all the information regarding a lane marking according to [OpenDRIVE 1.4 standard](http://www.opendrive.org/docs/OpenDRIVEFormatSpecRev1.4H.pdf) standard.
	"""

	color: LaneMarkingColor
	"""Actual color of the marking."""

	lane_change: LaneChange
	"""Permissions for said lane marking to be crossed."""

	type: LaneMarkingType
	"""Lane marking type."""

	width: float
	"""Horizontal lane marking thickness."""



class Waypoint:
	"""
	Waypoints in CARLA are described as 3D directed points. They have a carla.Transform which locates the waypoint in a road and orientates it according to the lane. They also store the road information belonging to said point regarding its lane and lane markings.    All the information regarding waypoints and the [waypoint API](../../core_map/#navigation-in-carla) is retrieved as provided by the OpenDRIVE file. Once the client asks for the map object to the server, no longer communication will be needed.
	"""

	id: int
	"""The identifier is generated using a hash combination of the road, section, lane and s values that correspond to said point in the OpenDRIVE geometry. The s precision is set to 2 centimeters, so 2 waypoints closer than 2 centimeters in the same road, section and lane, will have the same identificator."""

	transform: Transform
	"""Position and orientation of the waypoint according to the current lane information. This data is computed the first time it is accessed. It is not created right away in order to ease computing costs when lots of waypoints are created but their specific transform is not needed."""

	road_id: int
	"""OpenDRIVE road's id."""

	section_id: int
	"""OpenDRIVE section's id, based on the order that they are originally defined."""

	is_junction: bool
	"""True if the current Waypoint is on a junction as defined by OpenDRIVE."""

	junction_id: int
	"""OpenDRIVE junction's id. For more information refer to OpenDRIVE [documentation](http://www.opendrive.org/docs/OpenDRIVEFormatSpecRev1.4H.pdf#page=20)"""

	lane_id: int
	"""OpenDRIVE lane's id, this value can be positive or negative which represents the direction of the current lane with respect to the road. For more information refer to OpenDRIVE [documentation](http://www.opendrive.org/docs/OpenDRIVEFormatSpecRev1.4H.pdf#page=20)"""

	s: float
	"""OpenDRIVE s value of the current position."""

	lane_width: float
	"""Horizontal size of the road at current s."""

	lane_change: LaneChange
	"""Lane change definition of the current Waypoint's location, based on the traffic rules defined in the OpenDRIVE file. It states if a lane change can be done and in which direction."""

	lane_type: LaneType
	"""The lane type of the current Waypoint, based on OpenDRIVE 1.4 standard."""

	right_lane_marking: LaneMarking
	"""The right lane marking information based on the direction of the Waypoint."""

	left_lane_marking: LaneMarking
	"""The left lane marking information based on the direction of the Waypoint."""

	def next(self, distance: float) -> list[Waypoint]:
		"""
		Returns a list of waypoints at a certain approximate `distance` from the current one. It takes into account the road and its possible deviations without performing any lane change and returns one waypoint per option.
The list may be empty if the lane is not connected to any other at the specified distance.

		:param distance: (float) The approximate distance where to get the next waypoints.

		:return: list(carla.Waypoint)
		"""
		...

	def next_until_lane_end(self, distance: float) -> list[Waypoint]:
		"""
		Returns a list of waypoints from this to the end of the lane separated by a certain `distance`.

		:param distance: (float) The approximate distance between waypoints.

		:return: list(carla.Waypoint)
		"""
		...

	def previous(self, distance: float) -> list[Waypoint]:
		"""
		This method does not return the waypoint previously visited by an actor, but a list of waypoints at an approximate `distance` but in the opposite direction of the lane. Similarly to **next()**, it takes into account the road and its possible deviations without performing any lane change and returns one waypoint per option.
The list may be empty if the lane is not connected to any other at the specified distance.

		:param distance: (float) The approximate distance where to get the previous waypoints.

		:return: list(carla.Waypoint)
		"""
		...

	def previous_until_lane_start(self, distance: float) -> list[Waypoint]:
		"""
		Returns a list of waypoints from this to the start of the lane separated by a certain `distance`.

		:param distance: (float) The approximate distance between waypoints.

		:return: list(carla.Waypoint)
		"""
		...

	def get_junction(self) -> Junction:
		"""
		If the waypoint belongs to a junction this method returns the associated junction object. Otherwise returns null.

		:return: carla.Junction
		"""
		...

	def get_landmarks(self, distance: float, stop_at_junction: bool = False) -> list[Landmark]:
		"""
		Returns a list of landmarks in the road from the current waypoint until the specified distance.

		:param distance: (float) The maximum distance to search for landmarks from the current waypoint.

		:param stop_at_junction: (bool) Enables or disables the landmark search through junctions.

		:return: list(carla.Landmark)
		"""
		...

	def get_landmarks_of_type(self, distance: float, type: str, stop_at_junction: bool = False) -> list[Landmark]:
		"""
		Returns a list of landmarks in the road of a specified type from the current waypoint until the specified distance.

		:param distance: (float) The maximum distance to search for landmarks from the current waypoint.

		:param type: (str) The type of landmarks to search.

		:param stop_at_junction: (bool) Enables or disables the landmark search through junctions.

		:return: list(carla.Landmark)
		"""
		...

	def get_left_lane(self) -> Waypoint:
		"""
		Generates a Waypoint at the center of the left lane based on the direction of the current Waypoint, taking into account if the lane change is allowed in this location.
Will return None if the lane does not exist

		:return: carla.Waypoint
		"""
		...

	def get_right_lane(self) -> Waypoint:
		"""
		Generates a waypoint at the center of the right lane based on the direction of the current waypoint, taking into account if the lane change is allowed in this location.
Will return None if the lane does not exist.

		:return: carla.Waypoint
		"""
		...

	def __str__(self):
		"""

		"""
		...



class Junction:
	"""
	Class that embodies the intersections on the road described in the OpenDRIVE file according to OpenDRIVE 1.4 standards.
	"""

	id: int
	"""Identifier found in the OpenDRIVE file."""

	bounding_box: BoundingBox
	"""Bounding box encapsulating the junction lanes."""

	def get_waypoints(self, lane_type: LaneType) -> list[tuple[Waypoint]]:
		"""
		Returns a list of pairs of waypoints. Every tuple on the list contains first an initial and then a final waypoint within the intersection boundaries that describe the beginning and the end of said lane along the junction. Lanes follow their OpenDRIVE definitions so there may be many different tuples with the same starting waypoint due to possible deviations, as this are considered different lanes.

		:param lane_type: (LaneType) Type of lanes to get the waypoints.

		:return: list(tuple(carla.Waypoint))
		"""
		...



class LandmarkOrientation:
	"""
	Helper class to define the orientation of a landmark in the road. The definition is not directly translated from OpenDRIVE but converted for the sake of understanding.
	"""

	Positive: Any
	"""The landmark faces towards vehicles going on the same direction as the road's geometry definition (lanes 0 and negative in OpenDRIVE)."""

	Negative: Any
	"""The landmark faces towards vehicles going on the opposite direction to the road's geometry definition (positive lanes in OpenDRIVE)."""

	Both: Any
	"""Affects vehicles going in both directions of the road."""



class LandmarkType:
	"""
	Helper class containing a set of commonly used landmark types as defined by the default country code in the [OpenDRIVE standard](http://opendrive.org/docs/OpenDRIVEFormatSpecRev1.5M.pdf) (Germany 2017).
__carla.Landmark does not reference this class__. The landmark type is a string that varies greatly depending on the country code being used. This class only makes it easier to manage some of the most commonly used in the default set by describing them as an enum.
	"""

	Danger: Any
	"""Type 101."""

	LanesMerging: Any
	"""Type 121."""

	CautionPedestrian: Any
	"""Type 133."""

	CautionBicycle: Any
	"""Type 138."""

	LevelCrossing: Any
	"""Type 150."""

	StopSign: Any
	"""Type 206."""

	YieldSign: Any
	"""Type 205."""

	MandatoryTurnDirection: Any
	"""Type 209."""

	MandatoryLeftRightDirection: Any
	"""Type 211."""

	TwoChoiceTurnDirection: Any
	"""Type 214."""

	Roundabout: Any
	"""Type 215."""

	PassRightLeft: Any
	"""Type 222."""

	AccessForbidden: Any
	"""Type 250."""

	AccessForbiddenMotorvehicles: Any
	"""Type 251."""

	AccessForbiddenTrucks: Any
	"""Type 253."""

	AccessForbiddenBicycle: Any
	"""Type 254."""

	AccessForbiddenWeight: Any
	"""Type 263."""

	AccessForbiddenWidth: Any
	"""Type 264."""

	AccessForbiddenHeight: Any
	"""Type 265."""

	AccessForbiddenWrongDirection: Any
	"""Type 267."""

	ForbiddenUTurn: Any
	"""Type 272."""

	MaximumSpeed: Any
	"""Type 274."""

	ForbiddenOvertakingMotorvehicles: Any
	"""Type 276."""

	ForbiddenOvertakingTrucks: Any
	"""Type 277."""

	AbsoluteNoStop: Any
	"""Type 283."""

	RestrictedStop: Any
	"""Type 286."""

	HasWayNextIntersection: Any
	"""Type 301."""

	PriorityWay: Any
	"""Type 306."""

	PriorityWayEnd: Any
	"""Type 307."""

	CityBegin: Any
	"""Type 310."""

	CityEnd: Any
	"""Type 311."""

	Highway: Any
	"""Type 330."""

	DeadEnd: Any
	"""Type 357."""

	RecomendedSpeed: Any
	"""Type 380."""

	RecomendedSpeedEnd: Any
	"""Type 381."""



class Landmark:
	"""
	Class that defines any type of traffic landmark or sign affecting a road. These class mediates between the [OpenDRIVE 1.4 standard](http://www.opendrive.org/docs/OpenDRIVEFormatSpecRev1.4H.pdf) definition of the landmarks and their representation in the simulation. This class retrieves all the information defining a landmark in OpenDRIVE and facilitates information about which lanes does it affect and when.
Landmarks will be accessed by carla.Waypoint objects trying to retrieve the regulation of their lane. Therefore some attributes depend on the waypoint that is consulting the landmark and so, creating the object.
	"""

	road_id: int
	"""The OpenDRIVE ID of the road where this landmark is defined. Due to OpenDRIVE road definitions, this road may be different from the road the landmark is currently affecting. It is mostly the case in junctions where the road diverges in different routes.
Example: a traffic light is defined in one of the divergent roads in a junction, but it affects all the possible routes"""

	distance: float
	"""Distance between the landmark and the waypoint creating the object (querying `get_landmarks` or `get_landmarks_of_type`)."""

	s: float
	"""Distance where the landmark is positioned along the geometry of the road `road_id`."""

	t: float
	"""Lateral distance where the landmark is positioned from the edge of the road `road_id`."""

	id: str
	"""Unique ID of the landmark in the OpenDRIVE file."""

	name: str
	"""Name of the landmark in the in the OpenDRIVE file."""

	is_dynamic: bool
	"""Indicates if the landmark has state changes over time such as traffic lights."""

	orientation: LandmarkOrientation
	"""Indicates which lanes the landmark is facing towards to."""

	z_offset: float
	"""Height where the landmark is placed."""

	country: str
	"""Country code where the landmark is defined (default to OpenDRIVE is Germany 2017)."""

	type: str
	"""Type identifier of the landmark according to the country code."""

	sub_type: str
	"""Subtype identifier of the landmark according to the country code."""

	value: float
	"""Value printed in the signal (e.g. speed limit, maximum weight, etc)."""

	unit: str
	"""Units of measurement for the attribute `value`."""

	height: float
	"""Total height of the signal."""

	width: float
	"""Total width of the signal."""

	text: str
	"""Additional text in the signal."""

	h_offset: float
	"""Orientation offset of the signal relative to the the definition of `road_id` at `s` in OpenDRIVE."""

	pitch: float
	"""Pitch rotation of the signal (Y-axis in [UE coordinates system](https://carla.readthedocs.io/en/latest/python_api/#carlarotation))."""

	roll: float
	"""Roll rotation of the signal (X-axis in [UE coordinates system](https://carla.readthedocs.io/en/latest/python_api/#carlarotation))."""

	waypoint: Waypoint
	"""A waypoint placed in the lane of the one that made the query and at the `s` of the landmark. It is the first waypoint for which the landmark will be effective."""

	transform: Transform
	"""The location and orientation of the landmark in the simulation."""

	def get_lane_validities(self) -> list[tuple[int]]:
		"""
		Returns which lanes the landmark is affecting to. As there may be specific lanes where the landmark is not effective, the return is a list of pairs containing ranges of the __lane_id__ affected:
Example: In a road with 5 lanes, being 3 not affected: [(from_lane1,to_lane2),(from_lane4,to_lane5)]

		:return: list(tuple(int))
		"""
		...



from __future__ import annotations

from typing import Any
import numpy as np

import command

class Osm2Odr:
	"""
	Class that converts an OpenStreetMap map to OpenDRIVE format, so that it can be loaded in CARLA. Find out more about this feature in the [docs](tuto_G_openstreetmap.md).
	"""

	def convert(self, osm_file: str, settings: OSM2ODRSettings) -> str:
		"""
		Takes the content of an .osm file (OpenStreetMap format) and returns the content of the .xodr (OpenDRIVE format) describing said map. Some parameterization is passed to do the conversion.

		:param osm_file: (str) The content of the input OpenStreetMap file parsed as string.

		:param settings: (OSM2ODRSettings) Parameterization for the conversion.

		:return: str
		"""
		...



class Osm2OdrSettings:
	"""
	Helper class that contains the parameterization that will be used by carla.Osm2Odr to convert an OpenStreetMap map to OpenDRIVE format. Find out more about this feature in the [docs](tuto_G_openstreetmap.md).
	"""

	use_offsets: bool
	"""Enables the use of offset for the conversion. The offset will move the origin position of the map. Default value is __False__."""

	offset_x: float
	"""Offset in the X axis.  Default value is __0.0__."""

	offset_y: float
	"""Offset in the Y axis.  Default value is __0.0__."""

	default_lane_width: float
	"""Width of the lanes described in the resulting XODR map. Default value is __4.0__."""

	elevation_layer_height: float
	"""Defines the height separating two different [OpenStreetMap layers](https://wiki.openstreetmap.org/wiki/Key:layer). Default value is __0.0__."""

	center_map: bool
	"""When this option is enabled, the geometry of the map will be displaced so that the origin of coordinates matches the center of the bounding box of the entire road map."""

	proj_string: str
	"""Defines the [proj4](https://github.com/OSGeo/proj.4) string that will be used to compute the projection from geocoordinates to cartesian coordinates. This string will be written in the resulting OpenDRIVE unless the options `use_offsets` or `center_map` are enabled as these options override some of the definitions in the string."""

	generate_traffic_lights: bool
	"""Indicates wether to generate traffic light data in the OpenDRIVE. Road types defined by `set_traffic_light_excluded_way_types(way_types)` will not generate traffic lights."""

	all_junctions_with_traffic_lights: bool
	"""When disabled, the converter will generate traffic light data from the OpenStreetMaps data only. When enabled, all junctions will generate traffic lights."""

	def set_osm_way_types(self, way_types: list[str]):
		"""
		Defines the OpenStreetMaps road types that will be imported to OpenDRIVE. By default the road types imported are `motorway, motorway_link, trunk, trunk_link, primary, primary_link, secondary, secondary_link, tertiary, tertiary_link, unclassified, residential`. For a full list of road types check [here](https://wiki.openstreetmap.org/wiki/Main_Page).

		:param way_types: (list[str]) The list of road types.
		"""
		...

	def set_traffic_light_excluded_way_types(self, way_types: list[str]):
		"""
		Defines the OpenStreetMaps road types that will not generate traffic lights even if `generate_traffic_lights` is enabled. By default the road types excluded are `motorway_link, primary_link, secondary_link, tertiary_link`

		:param way_types: (list[str]) The list of road types.
		"""
		...



from __future__ import annotations

from typing import Any
import numpy as np

import command

class Sensor(Actor):
	"""
	Sensors compound a specific family of actors quite diverse and unique. They are normally spawned as attachment/sons of a vehicle (take a look at carla.World to learn about actor spawning). Sensors are thoroughly designed to retrieve different types of data that they are listening to. The data they receive is shaped as different subclasses inherited from carla.SensorData (depending on the sensor).

  Most sensors can be divided in two groups: those receiving data on every tick (cameras, point clouds and some specific sensors) and those who only receive under certain circumstances (trigger detectors). CARLA provides a specific set of sensors and their blueprint can be found in carla.BlueprintLibrary. All the information on their preferences and settlement can be found [here](ref_sensors.md), but the list of those available in CARLA so far goes as follow.
  Receive data on every tick.
  - [Depth camera](ref_sensors.md#depth-camera).
  - [Gnss sensor](ref_sensors.md#gnss-sensor).
  - [IMU sensor](ref_sensors.md#imu-sensor).
  - [Lidar raycast](ref_sensors.md#lidar-raycast-sensor).
  - [SemanticLidar raycast](ref_sensors.md#semanticlidar-raycast-sensor).
  - [Radar](ref_sensors.md#radar-sensor).
  - [RGB camera](ref_sensors.md#rgb-camera).
  - [RSS sensor](ref_sensors.md#rss-sensor).
  - [Semantic Segmentation camera](ref_sensors.md#semantic-segmentation-camera).
  Only receive data when triggered.
  - [Collision detector](ref_sensors.md#collision-detector).
  - [Lane invasion detector](ref_sensors.md#lane-invasion-detector).
  - [Obstacle detector](ref_sensors.md#obstacle-detector).
	"""

	is_listening: bool
	"""When True the sensor will be waiting for data."""

	def listen(self, callback: function):
		"""
		The function the sensor will be calling to every time a new measurement is received. This function needs for an argument containing an object type carla.SensorData to work with.

		:param callback: (function) The called function with one argument containing the sensor data.
		"""
		...

	def is_listening(self):
		"""
		Returns whether the sensor is in a listening state.
		"""
		...

	def stop(self):
		"""
		Commands the sensor to stop listening for data.
		"""
		...

	def enable_for_ros(self):
		"""
		Commands the sensor to be processed to be able to publish in ROS2 without any listen to it.
		"""
		...

	def disable_for_ros(self):
		"""
		Commands the sensor to not be processed for publishing in ROS2 if there is no any listen to it.
		"""
		...

	def is_enabled_for_ros(self):
		"""
		Returns if the sensor is enabled or not to publish in ROS2 if there is no any listen to it.
		"""
		...

	def listen_to_gbuffer(self, gbuffer_id: GBufferTextureID, callback: function):
		"""
		The function the sensor will be calling to every time the desired GBuffer texture is received. This function needs for an argument containing an object type carla.SensorData to work with.

		:param gbuffer_id: (GBufferTextureID) The ID of the target Unreal Engine GBuffer texture.

		:param callback: (function) The called function with one argument containing the received GBuffer texture.
		"""
		...

	def is_listening_gbuffer(self, gbuffer_id: GBufferTextureID):
		"""
		Returns whether the sensor is in a listening state for a specific GBuffer texture.

		:param gbuffer_id: (GBufferTextureID) The ID of the target Unreal Engine GBuffer texture.
		"""
		...

	def stop_gbuffer(self, gbuffer_id: GBufferTextureID):
		"""
		Commands the sensor to stop listening for the specified GBuffer texture.

		:param gbuffer_id: (GBufferTextureID) The ID of the Unreal Engine GBuffer texture.
		"""
		...

	def __str__(self):
		"""

		"""
		...



class RssSensor(Sensor):
	"""
	This sensor works a bit differently than the rest. Take look at the [specific documentation](adv_rss.md), and the [rss sensor reference](ref_sensors.md#rss-sensor) to gain full understanding of it.

The RSS sensor uses world information, and a [RSS library](https://github.com/intel/ad-rss-lib) to make safety checks on a vehicle. The output retrieved by the sensor is a carla.RssResponse. This will be used by a carla.RssRestrictor to modify a carla.VehicleControl before applying it to a vehicle.
	"""

	ego_vehicle_dynamics: Any
	"""States the [RSS parameters](https://intel.github.io/ad-rss-lib/ad_rss/Appendix-ParameterDiscussion/) that the sensor will consider for the ego vehicle if no actor constellation callback is registered."""

	other_vehicle_dynamics: Any
	"""States the [RSS parameters](https://intel.github.io/ad-rss-lib/ad_rss/Appendix-ParameterDiscussion/) that the sensor will consider for the rest of vehicles if no actor constellation callback is registered."""

	pedestrian_dynamics: Any
	"""States the [RSS parameters](https://intel.github.io/ad-rss-lib/ad_rss/Appendix-ParameterDiscussion/) that the sensor will consider for pedestrians if no actor constellation callback is registered."""

	road_boundaries_mode: RssRoadBoundariesMode
	"""Switches the [stay on road](https://intel.github.io/ad-rss-lib/ad_rss_map_integration/HandleRoadBoundaries/) feature. By default is __Off__."""

	routing_targets: Any
	"""The current list of targets considered to route the vehicle. If no routing targets are defined, a route is generated at random."""

	def append_routing_target(self, routing_target: Transform):
		"""
		Appends a new target position to the current route of the vehicle.

		:param routing_target: (Transform) New target point for the route. Choose these after the intersections to force the route to take the desired turn.
		"""
		...

	def reset_routing_targets(self):
		"""
		Erases the targets that have been appended to the route.
		"""
		...

	def drop_route(self):
		"""
		Discards the current route. If there are targets remaining in **routing_targets**, creates a new route using those. Otherwise, a new route is created at random.
		"""
		...

	def register_actor_constellation_callback(self, callback: Any):
		"""
		Register a callback to customize a carla.RssActorConstellationResult. By this callback the settings of RSS parameters are done per actor constellation and the settings (ego_vehicle_dynamics, other_vehicle_dynamics and pedestrian_dynamics) have no effect.

		:param callback: () The function to be called whenever a RSS situation is about to be calculated.
		"""
		...

	def set_log_level(self, log_level: RssLogLevel):
		"""
		Sets the log level.

		:param log_level: (RssLogLevel) New log level.
		"""
		...

	def set_map_log_level(self, log_level: RssLogLevel):
		"""
		Sets the map log level.

		:param log_level: (RssLogLevel) New map log level.
		"""
		...

	def __str__(self):
		"""

		"""
		...



class RssRestrictor:
	"""
	These objects apply restrictions to a carla.VehicleControl. It is part of the CARLA implementation of the [C++ Library for Responsibility Sensitive Safety](https://github.com/intel/ad-rss-lib). This class works hand in hand with a [rss sensor](ref_sensors.md#rss-sensor), which provides the data of the restrictions to be applied.
	"""

	def restrict_vehicle_control(self, vehicle_control: VehicleControl, proper_response: Any, ego_dynamics_on_route: RssEgoDynamicsOnRoute, vehicle_physics: VehiclePhysicsControl) -> VehicleControl:
		"""
		Applies the safety restrictions given by a carla.RssSensor to a carla.VehicleControl.

		:param vehicle_control: (VehicleControl) The input vehicle control to be restricted.

		:param proper_response: (Any) Part of the response generated by the sensor. Contains restrictions to be applied to the acceleration of the vehicle.

		:param ego_dynamics_on_route: (RssEgoDynamicsOnRoute) Part of the response generated by the sensor. Contains dynamics and heading of the vehicle regarding its route.

		:param vehicle_physics: (VehiclePhysicsControl) The current physics of the vehicle. Used to apply the restrictions properly.

		:return: carla.VehicleControl
		"""
		...

	def set_log_level(self, log_level: RssLogLevel):
		"""
		Sets the log level.

		:param log_level: (RssLogLevel) New log level.
		"""
		...



class RssRoadBoundariesMode:
	"""
	Enum declaration used in carla.RssSensor to enable or disable the [stay on road](https://intel.github.io/ad-rss-lib/ad_rss_map_integration/HandleRoadBoundaries/) feature. In summary, this feature considers the road boundaries as virtual objects. The minimum safety distance check is applied to these virtual walls, in order to make sure the vehicle does not drive off the road.
	"""

	On: Any
	"""Enables the _stay on road_ feature."""

	Off: Any
	"""Disables the _stay on road_ feature."""



class RssLogLevel:
	"""
	Enum declaration used in carla.RssSensor to set the log level.
	"""

	trace: Any
	""""""

	debug: Any
	""""""

	info: Any
	""""""

	warn: Any
	""""""

	err: Any
	""""""

	critical: Any
	""""""

	off: Any
	""""""



from __future__ import annotations

from typing import Any
import numpy as np

import command

class SensorData:
	"""
	Base class for all the objects containing data generated by a carla.Sensor. This objects should be the argument of the function said sensor is listening to, in order to work with them. Each of these sensors needs for a specific type of sensor data. Hereunder is a list of the sensors and their corresponding data.
  - Cameras (RGB, depth and semantic segmentation): carla.Image.
  - Collision detector: carla.CollisionEvent.
  - GNSS sensor: carla.GnssMeasurement.
  - IMU sensor: carla.IMUMeasurement.
  - Lane invasion detector: carla.LaneInvasionEvent.
  - LIDAR sensor: carla.LidarMeasurement.
  - Obstacle detector: carla.ObstacleDetectionEvent.
  - Radar sensor: carla.RadarMeasurement.
  - RSS sensor: carla.RssResponse.
  - Semantic LIDAR sensor: carla.SemanticLidarMeasurement.
	"""

	frame: int
	"""Frame count when the data was generated."""

	timestamp: float
	"""Simulation-time when the data was generated."""

	transform: Transform
	"""Sensor's transform when the data was generated."""



class ColorConverter:
	"""
	Class that defines conversion patterns that can be applied to a carla.Image in order to show information provided by carla.Sensor. Depth conversions cause a loss of accuracy, as sensors detect depth as float that is then converted to a grayscale value between 0 and 255. Take a look at the snipet in carla.Sensor.listen to see an example of how to create and save image data for sensor.camera.semantic_segmentation.
	"""

	CityScapesPalette: Any
	"""Converts the image to a segmented map using tags provided by the blueprint library. Used by the [semantic segmentation camera](ref_sensors.md#semantic-segmentation-camera)."""

	Depth: Any
	"""Converts the image to a linear depth map. Used by the [depth camera](ref_sensors.md#depth-camera)."""

	LogarithmicDepth: Any
	"""Converts the image to a depth map using a logarithmic scale, leading to better precision for small distances at the expense of losing it when further away."""

	Raw: Any
	"""No changes applied to the image. Used by the [RGB camera](ref_sensors.md#rgb-camera)."""



class CityObjectLabel:
	"""
	Enum declaration that contains the different tags available to filter the bounding boxes returned by carla.World.get_level_bbs(). These values correspond to the [semantic tag](ref_sensors.md#semantic-segmentation-camera) that the elements in the scene have.
	"""

	None: Any
	""""""

	Buildings: Any
	""""""

	Fences: Any
	""""""

	Other: Any
	""""""

	Pedestrians: Any
	""""""

	Poles: Any
	""""""

	RoadLines: Any
	""""""

	Roads: Any
	""""""

	Sidewalks: Any
	""""""

	TrafficSigns: Any
	""""""

	Vegetation: Any
	""""""

	Vehicles: Any
	""""""

	Walls: Any
	""""""

	Sky: Any
	""""""

	Ground: Any
	""""""

	Bridge: Any
	""""""

	RailTrack: Any
	""""""

	GuardRail: Any
	""""""

	TrafficLight: Any
	""""""

	Static: Any
	""""""

	Dynamic: Any
	""""""

	Water: Any
	""""""

	Terrain: Any
	""""""

	Any: Any
	""""""



class Image(SensorData):
	"""
	Class that defines an image of 32-bit BGRA colors that will be used as initial data retrieved by camera sensors. There are different camera sensors (currently three, RGB, depth and semantic segmentation) and each of these makes different use for the images. Learn more about them [here](ref_sensors.md).
	"""

	fov: float
	"""Horizontal field of view of the image."""

	height: int
	"""Image height in pixels."""

	width: int
	"""Image width in pixels."""

	raw_data: bytes
	"""Flattened array of pixel data, use reshape to create an image array."""

	def convert(self, color_converter: ColorConverter):
		"""
		Converts the image following the `color_converter` pattern.

		:param color_converter: (ColorConverter) 
		"""
		...

	def save_to_disk(self, path: str, color_converter: ColorConverter = Raw):
		"""
		Saves the image to disk using a converter pattern stated as `color_converter`. The default conversion pattern is Raw that will make no changes to the image.

		:param path: (str) Path that will contain the image.

		:param color_converter: (ColorConverter) Default Raw will make no changes.
		"""
		...

	def __getitem__(self, pos: int):
		"""


		:param pos: (int) 
		"""
		...

	def __iter__(self):
		"""
		Iterate over the carla.Color that form the image.
		"""
		...

	def __len__(self):
		"""

		"""
		...

	def __setitem__(self, pos: int, color: Color):
		"""


		:param pos: (int) 

		:param color: (Color) 
		"""
		...

	def __str__(self):
		"""

		"""
		...



class OpticalFlowImage(SensorData):
	"""
	Class that defines an optical flow image of 2-Dimension float (32-bit) vectors representing the optical flow detected in the field of view. The components of the vector represents the displacement of an object in the image plane. Each component outputs values in the normalized range [-2,2] which scales to [-2 size, 2 size] with size being the total resolution in the corresponding component.
	"""

	fov: float
	"""Horizontal field of view of the image."""

	height: int
	"""Image height in pixels."""

	width: int
	"""Image width in pixels."""

	raw_data: bytes
	"""Flattened array of pixel data, use reshape to create an image array."""

	def get_color_coded_flow(self) -> Image:
		"""
		Visualization helper. Converts the optical flow image to an RGB image.

		:return: carla.Image
		"""
		...

	def __getitem__(self, pos: int):
		"""


		:param pos: (int) 
		"""
		...

	def __iter__(self):
		"""
		Iterate over the carla.OpticalFlowPixel that form the image.
		"""
		...

	def __len__(self):
		"""

		"""
		...

	def __setitem__(self, pos: int, color: Color):
		"""


		:param pos: (int) 

		:param color: (Color) 
		"""
		...

	def __str__(self):
		"""

		"""
		...



class LidarMeasurement(SensorData):
	"""
	Class that defines the LIDAR data retrieved by a sensor.lidar.ray_cast. This essentially simulates a rotating LIDAR using ray-casting. Learn more about this [here](ref_sensors.md#lidar-raycast-sensor).
	"""

	channels: int
	"""Number of lasers shot."""

	horizontal_angle: float
	"""Horizontal angle the LIDAR is rotated at the time of the measurement."""

	raw_data: bytes
	"""Received list of 4D points. Each point consists of [x,y,z] coordinates plus the intensity computed for that point."""

	def save_to_disk(self, path: str):
		"""
		Saves the point cloud to disk as a .ply file describing data from 3D scanners. The files generated are ready to be used within [MeshLab](http://www.meshlab.net/), an open source system for processing said files. Just take into account that axis may differ from Unreal Engine and so, need to be reallocated.

		:param path: (str) 
		"""
		...

	def get_point_count(self, channel: int):
		"""
		Retrieves the number of points sorted by channel that are generated by this measure. Sorting by channel allows to identify the original channel for every point.

		:param channel: (int) 
		"""
		...

	def __getitem__(self, pos: int):
		"""


		:param pos: (int) 
		"""
		...

	def __iter__(self):
		"""
		Iterate over the carla.LidarDetection retrieved as data.
		"""
		...

	def __len__(self):
		"""

		"""
		...

	def __setitem__(self, pos: int, detection: LidarDetection):
		"""


		:param pos: (int) 

		:param detection: (LidarDetection) 
		"""
		...

	def __str__(self):
		"""

		"""
		...



class LidarDetection:
	"""
	Data contained inside a carla.LidarMeasurement. Each of these represents one of the points in the cloud with its location and its associated intensity.
	"""

	point: Location
	"""Point in xyz coordinates."""

	intensity: float
	"""Computed intensity for this point as a scalar value between [0.0 , 1.0]."""

	def __str__(self):
		"""

		"""
		...



class SemanticLidarMeasurement(SensorData):
	"""
	Class that defines the semantic LIDAR data retrieved by a sensor.lidar.ray_cast_semantic. This essentially simulates a rotating LIDAR using ray-casting. Learn more about this [here](ref_sensors.md#semanticlidar-raycast-sensor).
	"""

	channels: int
	"""Number of lasers shot."""

	horizontal_angle: float
	"""Horizontal angle the LIDAR is rotated at the time of the measurement."""

	raw_data: bytes
	"""Received list of raw detection points. Each point consists of [x,y,z] coordinates plus the cosine of the incident angle, the index of the hit actor, and its semantic tag."""

	def save_to_disk(self, path: str):
		"""
		Saves the point cloud to disk as a .ply file describing data from 3D scanners. The files generated are ready to be used within [MeshLab](http://www.meshlab.net/), an open-source system for processing said files. Just take into account that axis may differ from Unreal Engine and so, need to be reallocated.

		:param path: (str) 
		"""
		...

	def get_point_count(self, channel: int):
		"""
		Retrieves the number of points sorted by channel that are generated by this measure. Sorting by channel allows to identify the original channel for every point.

		:param channel: (int) 
		"""
		...

	def __getitem__(self, pos: int):
		"""


		:param pos: (int) 
		"""
		...

	def __iter__(self):
		"""
		Iterate over the carla.SemanticLidarDetection retrieved as data.
		"""
		...

	def __len__(self):
		"""

		"""
		...

	def __setitem__(self, pos: int, detection: SemanticLidarDetection):
		"""


		:param pos: (int) 

		:param detection: (SemanticLidarDetection) 
		"""
		...

	def __str__(self):
		"""

		"""
		...



class SemanticLidarDetection:
	"""
	Data contained inside a carla.SemanticLidarMeasurement. Each of these represents one of the points in the cloud with its location, the cosine of the incident angle, index of the object hit, and its semantic tag.
	"""

	point: Location
	"""[x,y,z] coordinates of the point."""

	cos_inc_angle: float
	"""Cosine of the incident angle between the ray, and the normal of the hit object."""

	object_idx: np.uint
	"""ID of the actor hit by the ray."""

	object_tag: np.uint
	"""[Semantic tag](https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera) of the component hit by the ray."""

	def __str__(self):
		"""

		"""
		...



class CollisionEvent(SensorData):
	"""
	Class that defines a collision data for sensor.other.collision. The sensor creates one of these for every collision detected. Each collision sensor produces one collision event per collision per frame. Multiple collision events may be produced in a single frame by collisions with multiple other actors. Learn more about this [here](ref_sensors.md#collision-detector).
	"""

	actor: Actor
	"""The actor the sensor is attached to, the one that measured the collision."""

	other_actor: Actor
	"""The second actor involved in the collision."""

	normal_impulse: Vector3D
	"""Normal impulse resulting of the collision."""



class ObstacleDetectionEvent(SensorData):
	"""
	Class that defines the obstacle data for sensor.other.obstacle. Learn more about this [here](ref_sensors.md#obstacle-detector).
	"""

	actor: Actor
	"""The actor the sensor is attached to."""

	other_actor: Actor
	"""The actor or object considered to be an obstacle."""

	distance: float
	"""Distance between `actor` and `other`."""

	def __str__(self):
		"""

		"""
		...



class LaneInvasionEvent(SensorData):
	"""
	Class that defines lanes invasion for sensor.other.lane_invasion. It works only client-side and is dependant on OpenDRIVE to provide reliable information. The sensor creates one of this every time there is a lane invasion, which may be more than once per simulation step. Learn more about this [here](ref_sensors.md#lane-invasion-detector).
	"""

	actor: Actor
	"""Gets the actor the sensor is attached to, the one that invaded another lane."""

	crossed_lane_markings: list[LaneMarking]
	"""List of lane markings that have been crossed and detected by the sensor."""

	def __str__(self):
		"""

		"""
		...



class GnssMeasurement(SensorData):
	"""
	Class that defines the Gnss data registered by a sensor.other.gnss. It essentially reports its position with the position of the sensor and an OpenDRIVE geo-reference.
	"""

	altitude: float
	"""Height regarding ground level."""

	latitude: float
	"""North/South value of a point on the map."""

	longitude: float
	"""West/East value of a point on the map."""

	def __str__(self):
		"""

		"""
		...



class IMUMeasurement(SensorData):
	"""
	Class that defines the data registered by a sensor.other.imu, regarding the sensor's transformation according to the current carla.World. It essentially acts as accelerometer, gyroscope and compass.
	"""

	accelerometer: Vector3D
	"""Linear acceleration."""

	compass: float
	"""Orientation with regard to the North ([0.0, -1.0, 0.0] in Unreal Engine)."""

	gyroscope: Vector3D
	"""Angular velocity."""

	def __str__(self):
		"""

		"""
		...



class RadarMeasurement(SensorData):
	"""
	Class that defines and gathers the measures registered by a sensor.other.radar, representing a wall of points in front of the sensor with a distance, angle and velocity in relation to it. The data consists of a carla.RadarDetection array. Learn more about this [here](ref_sensors.md#radar-sensor).
	"""

	raw_data: bytes
	"""The complete information of the carla.RadarDetection the radar has registered."""

	def get_detection_count(self):
		"""
		Retrieves the number of entries generated, same as **\__str__()**.
		"""
		...

	def __getitem__(self, pos: int):
		"""


		:param pos: (int) 
		"""
		...

	def __iter__(self):
		"""
		Iterate over the carla.RadarDetection retrieved as data.
		"""
		...

	def __len__(self):
		"""

		"""
		...

	def __setitem__(self, pos: int, detection: RadarDetection):
		"""


		:param pos: (int) 

		:param detection: (RadarDetection) 
		"""
		...

	def __str__(self):
		"""

		"""
		...



class RadarDetection:
	"""
	Data contained inside a carla.RadarMeasurement. Each of these represents one of the points in the cloud that a sensor.other.radar registers and contains the distance, angle and velocity in relation to the radar.
	"""

	altitude: float
	"""Altitude angle of the detection."""

	azimuth: float
	"""Azimuth angle of the detection."""

	depth: float
	"""Distance from the sensor to the detection position."""

	velocity: float
	"""The velocity of the detected object towards the sensor."""

	def __str__(self):
		"""

		"""
		...



class RssResponse(SensorData):
	"""
	Class that contains the output of a carla.RssSensor. This is the result of the RSS calculations performed for the parent vehicle of the sensor.

A carla.RssRestrictor will use the data to modify the carla.VehicleControl of the vehicle.
	"""

	response_valid: bool
	"""States if the response is valid. It is __False__ if calculations failed or an exception occured."""

	proper_response: Any
	"""The proper response that the RSS calculated for the vehicle."""

	rss_state_snapshot: Any
	"""Detailed RSS states at the current moment in time."""

	ego_dynamics_on_route: RssEgoDynamicsOnRoute
	"""Current ego vehicle dynamics regarding the route."""

	world_model: Any
	"""World model used for calculations."""

	situation_snapshot: Any
	"""Detailed RSS situations extracted from the world model."""

	def __str__(self):
		"""

		"""
		...



class RssEgoDynamicsOnRoute:
	"""
	Part of the data contained inside a carla.RssResponse describing the state of the vehicle. The parameters include its current dynamics, and how it is heading regarding the target route.
	"""

	ego_speed: Any
	"""The ego vehicle's speed."""

	min_stopping_distance: Any
	"""The current minimum stopping distance."""

	ego_center: Any
	"""The considered enu position of the ego vehicle."""

	ego_heading: Any
	"""The considered heading of the ego vehicle."""

	ego_center_within_route: bool
	"""States if the ego vehicle's center is within the route."""

	crossing_border: bool
	"""States if the vehicle is already crossing one of the lane borders."""

	route_heading: Any
	"""The considered heading of the route."""

	route_nominal_center: Any
	"""The considered nominal center of the current route."""

	heading_diff: Any
	"""The considered heading diff towards the route."""

	route_speed_lat: Any
	"""The ego vehicle's speed component _lat_ regarding the route."""

	route_speed_lon: Any
	"""The ego vehicle's speed component _lon_ regarding the route."""

	route_accel_lat: Any
	"""The ego vehicle's acceleration component _lat_ regarding the route."""

	route_accel_lon: Any
	"""The ego vehicle's acceleration component _lon_ regarding the route."""

	avg_route_accel_lat: Any
	"""The ego vehicle's acceleration component _lat_ regarding the route smoothened by an average filter."""

	avg_route_accel_lon: Any
	"""The ego acceleration component _lon_ regarding the route smoothened by an average filter."""

	def __str__(self):
		"""

		"""
		...



class RssActorConstellationData:
	"""
	Data structure that is provided within the callback registered by RssSensor.register_actor_constellation_callback().
	"""

	ego_match_object: Any
	"""The ego map matched information."""

	ego_route: Any
	"""The ego route."""

	ego_dynamics_on_route: RssEgoDynamicsOnRoute
	"""Current ego vehicle dynamics regarding the route."""

	other_match_object: Any
	"""The other object's map matched information. This is only valid if 'other_actor' is not 'None'."""

	other_actor: Actor
	"""The other actor. This is 'None' in case of query of default parameters or articial objects of kind ad.rss.world.ObjectType.ArtificialObject with no dedicated 'carla.Actor' (as e.g. for the [road boundaries](ref_sensors.md#rss-sensor) at the moment)"""

	def __str__(self):
		"""

		"""
		...



class RssActorConstellationResult:
	"""
	Data structure that should be returned by the callback registered by RssSensor.register_actor_constellation_callback().
	"""

	rss_calculation_mode: Any
	"""The calculation mode to be applied with the actor."""

	restrict_speed_limit_mode: Any
	"""The mode for restricting speed limit."""

	ego_vehicle_dynamics: Any
	"""The RSS dynamics to be applied for the ego vehicle."""

	actor_object_type: Any
	"""The RSS object type to be used for the actor."""

	actor_dynamics: Any
	"""The RSS dynamics to be applied for the actor."""

	def __str__(self):
		"""

		"""
		...



class DVSEvent:
	"""
	Class that defines a DVS event. An event is a quadruple, so a tuple of 4 elements, with `x`, `y` pixel coordinate location, timestamp `t` and polarity `pol` of the event. Learn more about them [here](ref_sensors.md).
	"""

	x: int
	"""X pixel coordinate."""

	y: int
	"""Y pixel coordinate."""

	t: int
	"""Timestamp of the moment the event happened."""

	pol: bool
	"""Polarity of the event. __True__ for positive and __False__ for negative."""

	def __str__(self):
		"""

		"""
		...



class DVSEventArray:
	"""
	Class that defines a stream of events in carla.DVSEvent. Such stream is an array of arbitrary size depending on the number of events. This class also stores the field of view, the height and width of the image and the timestamp from convenience. Learn more about them [here](ref_sensors.md).
	"""

	fov: float
	"""Horizontal field of view of the image."""

	height: int
	"""Image height in pixels."""

	width: int
	"""Image width in pixels."""

	raw_data: bytes
	""""""

	def to_image(self):
		"""
		Converts the image following this pattern: blue indicates positive events, red indicates negative events.
		"""
		...

	def to_array(self):
		"""
		Converts the stream of events to an array of int values in the following order [x, y, t, pol].
		"""
		...

	def to_array_x(self):
		"""
		Returns an array with X pixel coordinate of all the events in the stream.
		"""
		...

	def to_array_y(self):
		"""
		Returns an array with Y pixel coordinate of all the events in the stream.
		"""
		...

	def to_array_t(self):
		"""
		Returns an array with the timestamp of all the events in the stream.
		"""
		...

	def to_array_pol(self):
		"""
		Returns an array with the polarity of all the events in the stream.
		"""
		...

	def __getitem__(self, pos: int):
		"""


		:param pos: (int) 
		"""
		...

	def __iter__(self):
		"""
		Iterate over the carla.DVSEvent retrieved as data.
		"""
		...

	def __len__(self):
		"""

		"""
		...

	def __setitem__(self, pos: int, color: Color):
		"""


		:param pos: (int) 

		:param color: (Color) 
		"""
		...

	def __str__(self):
		"""

		"""
		...



class GBufferTextureID:
	"""
	Defines the identifiers of each GBuffer texture (See the method `carla.Sensor.listen_to_gbuffer`).
	"""

	SceneColor: Any
	"""The texture "SceneColor" contains the final color of the image."""

	SceneDepth: Any
	"""The texture "SceneDepth" contains the depth buffer - linear in world units."""

	SceneStencil: Any
	"""The texture "SceneStencil" contains the stencil buffer."""

	GBufferA: Any
	"""The texture "GBufferA" contains the world-space normal vectors in the RGB channels. The alpha channel contains "per-object data"."""

	GBufferB: Any
	"""The texture "GBufferB" contains the metallic, specular and roughness in the RGB channels, respectively. The alpha channel contains a mask where the lower 4 bits indicate the shading model and the upper 4 bits contain the selective output mask."""

	GBufferC: Any
	"""The texture "GBufferC" contains the diffuse color in the RGB channels, with the indirect irradiance in the alpha channel. If static lightning is not allowed, the alpha channel will contain the ambient occlusion instead."""

	GBufferD: Any
	"""The contents of the "GBufferD" varies depending on the rendered object's material shading model (GBufferB):
  - MSM_Subsurface (2), MSM_PreintegratedSkin (3), MSM_TwoSidedFoliage (6):
    RGB: Subsurface color.
    A: Opacity.
  - MSM_ClearCoat (4):
    R: Clear coat.
    G: Roughness.
  - MSM_SubsurfaceProfile (5):
    RGB: Subsurface profile.
  - MSM_Hair (7):
    RG: World normal.
    B: Backlit value.
  - MSM_Cloth (8):
    RGB: Subsurface color.
    A: Cloth value.
  - MSM_Eye (9):
    RG: Eye tangent.
    B: Iris mask.
    A: Iris distance."""

	GBufferE: Any
	"""The texture "GBufferE" contains the precomputed shadow factors in the RGBA channels. This texture is unavailable if the selective output mask (GBufferB) does not have its 4th bit set."""

	GBufferF: Any
	"""The texture "GBufferF" contains the world-space tangent in the RGB channels and the anisotropy in the alpha channel. This texture is unavailable if the selective output mask (GBufferB) does not have its 5th bit set."""

	Velocity: Any
	"""The texture "Velocity" contains the screen-space velocity of the scene objects."""

	SSAO: Any
	"""The texture "SSAO" contains the screen-space ambient occlusion texture."""

	CustomDepth: Any
	"""The texture "CustomDepth" contains the Unreal Engine custom depth data."""

	CustomStencil: Any
	"""The texture "CustomStencil" contains the Unreal Engine custom stencil data."""



from __future__ import annotations

from typing import Any
import numpy as np

import command

class WorldSnapshot:
	"""
	This snapshot comprises all the information for every actor on scene at a certain moment of time. It creates and gives acces to a data structure containing a series of carla.ActorSnapshot. The client recieves a new snapshot on every tick that cannot be stored.
	"""

	id: int
	"""A value unique for every snapshot to differentiate them."""

	frame: int
	"""Simulation frame in which the snapshot was taken."""

	timestamp: Timestamp
	"""Precise moment in time when snapshot was taken. This class works in seconds as given by the operative system."""

	def find(self, actor_id: int) -> ActorSnapshot:
		"""
		Given a certain actor ID, returns its corresponding snapshot or None if it is not found.

		:param actor_id: (int) 

		:return: carla.ActorSnapshot
		"""
		...

	def has_actor(self, actor_id: int) -> bool:
		"""
		Given a certain actor ID, checks if there is a snapshot corresponding it and so, if the actor was present at that moment.

		:param actor_id: (int) 

		:return: bool
		"""
		...

	def __iter__(self):
		"""
		Iterate over the carla.ActorSnapshot stored in the snapshot.
		"""
		...

	def __len__(self) -> int:
		"""
		Returns the amount of carla.ActorSnapshot present in this snapshot.

		:return: int
		"""
		...

	def __eq__(self, other: WorldSnapshot) -> bool:
		"""
		Returns __True__ if both **timestamp** are the same.

		:param other: (WorldSnapshot) 

		:return: bool
		"""
		...

	def __ne__(self, other: WorldSnapshot) -> bool:
		"""
		Returns True if both **timestamp** are different.

		:param other: (WorldSnapshot) 

		:return: bool
		"""
		...



class ActorSnapshot:
	"""
	A class that comprises all the information for an actor at a certain moment in time. These objects are contained in a carla.WorldSnapshot and sent to the client once every tick.
	"""

	id: int
	"""An identifier for the snapshot itself."""

	def get_acceleration(self) -> Vector3D:
		"""
		Returns the acceleration vector registered for an actor in that tick.

		:return: carla.Vector3D
		"""
		...

	def get_angular_velocity(self) -> Vector3D:
		"""
		Returns the angular velocity vector registered for an actor in that tick.

		:return: carla.Vector3D
		"""
		...

	def get_transform(self) -> Transform:
		"""
		Returns the actor's transform (location and rotation) for an actor in that tick.

		:return: carla.Transform
		"""
		...

	def get_velocity(self) -> Vector3D:
		"""
		Returns the velocity vector registered for an actor in that tick.

		:return: carla.Vector3D
		"""
		...



from __future__ import annotations

from typing import Any
import numpy as np

import command

class WeatherParameters:
	"""
	This class defines objects containing lighting and weather specifications that can later be applied in carla.World. So far, these conditions only intervene with [sensor.camera.rgb](ref_sensors.md#rgb-camera). They neither affect the actor's physics nor other sensors.        
  Each of these parameters acts indepently from the rest. Increasing the rainfall will not automatically create puddles nor change the road's humidity. That makes for a better customization but means that realistic conditions need to be scripted. However an example of dynamic weather conditions working realistically can be found [here](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/dynamic_weather.py).
	"""

	cloudiness: float
	"""Values range from 0 to 100, being 0 a clear sky and 100 one completely covered with clouds."""

	precipitation: float
	"""Rain intensity values range from 0 to 100, being 0 none at all and 100 a heavy rain."""

	precipitation_deposits: float
	"""Determines the creation of puddles. Values range from 0 to 100, being 0 none at all and 100 a road completely capped with water. Puddles are created with static noise, meaning that they will always appear at the same locations."""

	wind_intensity: float
	"""Controls the strenght of the wind with values from 0, no wind at all, to 100, a strong wind. The wind does affect rain direction and leaves from trees, so this value is restricted to avoid animation issues."""

	sun_azimuth_angle: float
	"""The azimuth angle of the sun. Values range from 0 to 360. Zero is an origin point in a sphere determined by Unreal Engine."""

	sun_altitude_angle: float
	"""Altitude angle of the sun. Values range from -90 to 90 corresponding to midnight and midday each."""

	fog_density: float
	"""Fog concentration or thickness. It only affects the RGB camera sensor. Values range from 0 to 100."""

	fog_distance: float
	"""Fog start distance. Values range from 0 to infinite."""

	wetness: float
	"""Wetness intensity. It only affects the RGB camera sensor. Values range from 0 to 100."""

	fog_falloff: float
	"""Density of the fog (as in specific mass) from 0 to infinity. The bigger the value, the more dense and heavy it will be, and the fog will reach smaller heights. Corresponds to Fog Height Falloff in the UE docs.  If the value is 0, the fog will be lighter than air, and will cover the whole scene.  A value of 1 is approximately as dense as the air, and reaches normal-sized buildings.  For values greater than 5, the air will be so dense that it will be compressed on ground level."""

	scattering_intensity: float
	"""Controls how much the light will contribute to volumetric fog. When set to 0, there is no contribution."""

	mie_scattering_scale: float
	"""Controls interaction of light with large particles like pollen or air pollution resulting in a hazy sky with halos around the light sources. When set to 0, there is no contribution."""

	rayleigh_scattering_scale: float
	"""Controls interaction of light with small particles like air molecules. Dependent on light wavelength, resulting in a blue sky in the day or red sky in the evening."""

	dust_storm: float
	"""Determines the strength of the dust storm weather. Values range from 0 to 100."""

	def __init__(self, cloudiness: float = 0.0, precipitation: float = 0.0, precipitation_deposits: float = 0.0, wind_intensity: float = 0.0, sun_azimuth_angle: float = 0.0, sun_altitude_angle: float = 0.0, fog_density: float = 0.0, fog_distance: float = 0.0, wetness: float = 0.0, fog_falloff: float = 0.0, scattering_intensity: float = 0.0, mie_scattering_scale: float = 0.0, rayleigh_scattering_scale: float = 0.0331):
		"""
		Method to initialize an object defining weather conditions. This class has some presets for different noon and sunset conditions listed in a note below.

		*note*: ClearNoon, CloudyNoon, WetNoon, WetCloudyNoon, SoftRainNoon, MidRainyNoon, HardRainNoon, ClearSunset, CloudySunset, WetSunset, WetCloudySunset, SoftRainSunset, MidRainSunset, HardRainSunset.

		:param cloudiness: (float) 0 is a clear sky, 100 complete overcast.

		:param precipitation: (float) 0 is no rain at all, 100 a heavy rain.

		:param precipitation_deposits: (float) 0 means no puddles on the road, 100 means roads completely capped by rain.

		:param wind_intensity: (float) 0 is calm, 100 a strong wind.

		:param sun_azimuth_angle: (float) 0 is an arbitrary North, 180 its corresponding South.

		:param sun_altitude_angle: (float) 90 is midday, -90 is midnight.

		:param fog_density: (float) Concentration or thickness of the fog, from 0 to 100.

		:param fog_distance: (float) Distance where the fog starts in meters.

		:param wetness: (float) Humidity percentages of the road, from 0 to 100.

		:param fog_falloff: (float) Density (specific mass) of the fog, from 0 to infinity.

		:param scattering_intensity: (float) Controls how much the light will contribute to volumetric fog. When set to 0, there is no contribution.

		:param mie_scattering_scale: (float) Controls interaction of light with large particles like pollen or air pollution resulting in a hazy sky with halos around the light sources. When set to 0, there is no contribution.

		:param rayleigh_scattering_scale: (float) Controls interaction of light with small particles like air molecules. Dependent on light wavelength, resulting in a blue sky in the day or red sky in the evening.
		"""
		...

	def __eq__(self, other: Any) -> bool:
		"""
		Returns True if both objects' variables are the same.

		:param other: () 

		:return: bool
		"""
		...

	def __ne__(self, other: Any) -> bool:
		"""
		Returns True if both objects' variables are different.

		:param other: () 

		:return: bool
		"""
		...

	def __str__(self):
		"""

		"""
		...



from __future__ import annotations

from typing import Any
import numpy as np

import command

class Timestamp:
	"""
	Class that contains time information for simulated data. This information is automatically retrieved as part of the carla.WorldSnapshot the client gets on every frame, but might also be used in many other situations such as a carla.Sensor retrieveing data.
	"""

	frame: int
	"""The number of frames elapsed since the simulator was launched."""

	elapsed_seconds: float
	"""Simulated seconds elapsed since the beginning of the current episode."""

	delta_seconds: float
	"""Simulated seconds elapsed since the previous frame."""

	platform_timestamp: float
	"""Time register of the frame at which this measurement was taken given by the OS in seconds."""

	def __init__(self, frame: int, elapsed_seconds: float, delta_seconds: float, platform_timestamp: float):
		"""


		:param frame: (int) 

		:param elapsed_seconds: (float) 

		:param delta_seconds: (float) 

		:param platform_timestamp: (float) 
		"""
		...

	def __eq__(self, other: Timestamp):
		"""


		:param other: (Timestamp) 
		"""
		...

	def __ne__(self, other: Timestamp):
		"""


		:param other: (Timestamp) 
		"""
		...

	def __str__(self):
		"""

		"""
		...



class ActorList:
	"""
	A class that contains every actor present on the scene and provides access to them. The list is automatically created and updated by the server and it can be returned using carla.World.
	"""

	def filter(self, wildcard_pattern: str) -> list:
		"""
		Filters a list of Actors matching `wildcard_pattern` against their variable type_id (which identifies the blueprint used to spawn them). Matching follows [fnmatch](https://docs.python.org/2/library/fnmatch.html) standard.

		:param wildcard_pattern: (str) 

		:return: list
		"""
		...

	def find(self, actor_id: int) -> Actor:
		"""
		Finds an actor using its identifier and returns it or None if it is not present.

		:param actor_id: (int) 

		:return: carla.Actor
		"""
		...

	def __getitem__(self, pos: int) -> Actor:
		"""
		Returns the actor corresponding to `pos` position in the list.

		:param pos: (int) 

		:return: carla.Actor
		"""
		...

	def __iter__(self):
		"""
		Iterate over the carla.Actor contained in the list.
		"""
		...

	def __len__(self) -> int:
		"""
		Returns the amount of actors listed.

		:return: int
		"""
		...

	def __str__(self) -> str:
		"""
		Parses to the ID for every actor listed.

		:return: str
		"""
		...



class WorldSettings:
	"""
	The simulation has some advanced configuration options that are contained in this class and can be managed using carla.World and its methods. These allow the user to choose between client-server synchrony/asynchrony, activation of "no rendering mode" and either if the simulation should run with a fixed or variable time-step. Check [this](adv_synchrony_timestep.md) out if you want to learn about it.
	"""

	synchronous_mode: bool
	"""States the synchrony between client and server. When set to true, the server will wait for a client tick in order to move forward. It is false by default."""

	no_rendering_mode: bool
	"""When enabled, the simulation will run no rendering at all. This is mainly used to avoid overhead during heavy traffic simulations. It is false by default."""

	fixed_delta_seconds: float
	"""Ensures that the time elapsed between two steps of the simulation is fixed. Set this to 0.0 to work with a variable time-step, as happens by default."""

	substepping: bool
	"""Enable the physics substepping. This option allows computing some physics substeps between two render frames. If synchronous mode is set, the number of substeps and its time interval are fixed and computed are so they fulfilled the requirements of carla.WorldSettings.max_substep and carla.WorldSettings.max_substep_delta_time. These last two parameters need to be compatible with carla.WorldSettings.fixed_delta_seconds. Enabled by default."""

	max_substep_delta_time: float
	"""Maximum delta time of the substeps. If the carla.WorldSettingsmax_substep is high enough, the substep delta time would be always below or equal to this value. By default, the value is set to 0.01."""

	max_substeps: int
	"""The maximum number of physics substepping that are allowed. By default, the value is set to 10."""

	max_culling_distance: float
	"""Configure the max draw distance for each mesh of the level."""

	deterministic_ragdolls: bool
	"""Defines wether to use deterministic physics for pedestrian death animations or physical ragdoll simulation.  When enabled, pedestrians have less realistic death animation but ensures determinism.  When disabled, pedestrians are simulated as ragdolls with more realistic simulation and collision but no determinsm can be ensured."""

	tile_stream_distance: float
	"""Used for large maps only. Configures the maximum distance from the hero vehicle to stream tiled maps. Regions of the map within this range will be visible (and capable of simulating physics). Regions outside this region will not be loaded."""

	actor_active_distance: float
	"""Used for large maps only. Configures the distance from the hero vehicle to convert actors to dormant. Actors within this range will be active, and actors outside will become dormant."""

	spectator_as_ego: bool
	"""Used for large maps only. Defines the influence of the spectator on tile loading in Large Maps. By default, the spectator will provoke loading of neighboring tiles in the absence of an ego actor. This might be inconvenient for applications that immediately spawn an ego actor."""

	def __init__(self, synchronous_mode: bool = False, no_rendering_mode: bool = False, fixed_delta_seconds: float = 0.0, max_culling_distance: float = 0.0, deterministic_ragdolls: bool = False, tile_stream_distance: float = 3000, actor_active_distance: float = 2000, spectator_as_ego: bool = True):
		"""
		Creates an object containing desired settings that could later be applied through carla.World and its method apply_settings().

		:param synchronous_mode: (bool) Set this to true to enable client-server synchrony.

		:param no_rendering_mode: (bool) Set this to true to completely disable rendering in the simulation.

		:param fixed_delta_seconds: (float) Set a fixed time-step in between frames. 0.0 means variable time-step and it is the default mode.

		:param max_culling_distance: (float) Configure the max draw distance for each mesh of the level.

		:param deterministic_ragdolls: (bool) Defines wether to use deterministic physics or ragdoll simulation for pedestrian deaths.

		:param tile_stream_distance: (float) Used for large maps only. Configures the maximum distance from the hero vehicle to stream tiled maps.

		:param actor_active_distance: (float) Used for large maps only. Configures the distance from the hero vehicle to convert actors to dormant.

		:param spectator_as_ego: (bool) Used for large maps only. Defines the influence of the spectator on tile loading in Large Maps.
		"""
		...

	def __eq__(self, other: WorldSettings) -> bool:
		"""
		Returns True if both objects' variables are the same.

		:param other: (WorldSettings) Settings to be compared with.

		:return: bool
		"""
		...

	def __ne__(self, other: WorldSettings) -> bool:
		"""
		Returns True if both objects' variables are different.

		:param other: (WorldSettings) Settings to be compared with.

		:return: bool
		"""
		...

	def __str__(self) -> str:
		"""
		Parses the established settings to a string and shows them in command line.

		:return: str
		"""
		...



class EnvironmentObject:
	"""
	Class that represents a geometry in the level, this geometry could be part of an actor formed with other EnvironmentObjects (ie: buildings).
	"""

	transform: Transform
	"""Contains the location and orientation of the EnvironmentObject in world space."""

	bounding_box: BoundingBox
	"""Object containing a location, rotation and the length of a box for every axis in world space."""

	id: int
	"""Unique ID to identify the object in the level."""

	name: str
	"""Name of the EnvironmentObject."""

	type: CityObjectLabel
	"""Semantic tag."""

	def __str__(self) -> str:
		"""
		Parses the EnvironmentObject to a string and shows them in command line.

		:return: str
		"""
		...



class AttachmentType:
	"""
	Class that defines attachment options between an actor and its parent. When spawning actors, these can be attached to another actor so their position changes accordingly. This is specially useful for sensors. The snipet in carla.World.spawn_actor shows some sensors being attached to a car when spawned. Note that the attachment type is declared as an enum within the class.
	"""

	Rigid: Any
	"""With this fixed attachment the object follow its parent position strictly. This is the recommended attachment to retrieve precise data from the simulation."""

	SpringArm: Any
	"""An attachment that expands or retracts the position of the actor, depending on its parent. This attachment is only recommended to record videos from the simulation where a smooth movement is needed. SpringArms are an Unreal Engine component so [check the UE docs](https://docs.unrealengine.com/en-US/Gameplay/HowTo/UsingCameras/SpringArmComponents/index.html) to learn more about them. Warning: The SpringArm attachment presents weird behaviors when an actor is spawned with a relative translation in the Z-axis (e.g. child_location = Location(0,0,2))."""

	SpringArmGhost: Any
	"""An attachment like the previous one but that does not make the collision test, and that means that it does not expands or retracts the position of the actor. The term **ghost** is because then the camera can cross walls and other geometries. This attachment is only recommended to record videos from the simulation where a smooth movement is needed. SpringArms are an Unreal Engine component so [check the UE docs](https://docs.unrealengine.com/en-US/Gameplay/HowTo/UsingCameras/SpringArmComponents/index.html) to learn more about them. Warning: The SpringArm attachment presents weird behaviors when an actor is spawned with a relative translation in the Z-axis (e.g. child_location = Location(0,0,2))."""



class LabelledPoint:
	"""
	Class that represent a position in space with a semantic label.
	"""

	location: Any
	"""Position in 3D space."""

	label: Any
	"""Semantic tag of the point."""



class MapLayer:
	"""
	Class that represents each manageable layer of the map. Can be used as flags. __WARNING: Only "Opt" maps are able to work with map layers.__
	"""

	NONE: Any
	"""No layers selected"""

	Buildings: Any
	""""""

	Decals: Any
	""""""

	Foliage: Any
	""""""

	Ground: Any
	""""""

	ParkedVehicles: Any
	""""""

	Particles: Any
	""""""

	Props: Any
	""""""

	StreetLights: Any
	""""""

	Walls: Any
	""""""

	All: Any
	"""All layers selected"""



class MaterialParameter:
	"""
	Class that represents material parameters. Not all objects in the scene contain all parameters.
	"""

	Normal: Any
	"""The Normal map of the object. Present in all objects."""

	Diffuse: Any
	"""The Diffuse texture of the object. Present in all objects."""

	AO_Roughness_Metallic_Emissive: Any
	"""A texture where each color channel represent a property of the material (R: Ambien oclusion, G: Roughness, B: Metallic, A: Emissive/Height map in some objects)"""

	Emissive: Any
	"""Emissive texture. Present in a few objects."""



class TextureColor:
	"""
	Class representing a texture object to be uploaded to the server. Pixel format is RGBA, uint8 per channel.
	"""

	width: int
	"""X-coordinate size of the texture."""

	height: int
	"""Y-coordinate size of the texture."""

	def __init__(self, width: int, height: int):
		"""
		Initializes a the texture with a (`width`, `height`) size.

		:param width: (int) 

		:param height: (int) 
		"""
		...

	def set_dimensions(self, width: int, height: int):
		"""
		Resizes the texture to te specified dimensions.

		:param width: (int) 

		:param height: (int) 
		"""
		...

	def get(self, x: int, y: int) -> Color:
		"""
		Get the (x,y) pixel data.

		:param x: (int) 

		:param y: (int) 

		:return: carla.Color
		"""
		...

	def set(self, x: int, y: int, value: Color):
		"""
		Sets the (x,y) pixel data with `value`.

		:param x: (int) 

		:param y: (int) 

		:param value: (Color) 
		"""
		...



class TextureFloatColor:
	"""
	Class representing a texture object to be uploaded to the server. Pixel format is RGBA, float per channel.
	"""

	width: int
	"""X-coordinate size of the texture."""

	height: int
	"""Y-coordinate size of the texture."""

	def __init__(self, width: int, height: int):
		"""
		Initializes a the texture with a (`width`, `height`) size.

		:param width: (int) 

		:param height: (int) 
		"""
		...

	def set_dimensions(self, width: int, height: int):
		"""
		Resizes the texture to te specified dimensions.

		:param width: (int) 

		:param height: (int) 
		"""
		...

	def get(self, x: int, y: int) -> FloatColor:
		"""
		Get the (x,y) pixel data.

		:param x: (int) 

		:param y: (int) 

		:return: carla.FloatColor
		"""
		...

	def set(self, x: int, y: int, value: FloatColor):
		"""
		Sets the (x,y) pixel data with `value`.

		:param x: (int) 

		:param y: (int) 

		:param value: (FloatColor) 
		"""
		...



class World:
	"""
	World objects are created by the client to have a place for the simulation to happen. The world contains the map we can see, meaning the asset, not the navigation map. Navigation maps are part of the carla.Map class. It also manages the weather and actors present in it. There can only be one world per simulation, but it can be changed anytime.
	"""

	id: int
	"""The ID of the episode associated with this world. Episodes are different sessions of a simulation. These change everytime a world is disabled or reloaded. Keeping track is useful to avoid possible issues."""

	debug: DebugHelper
	"""Responsible for creating different shapes for debugging. Take a look at its class to learn more about it."""

	def apply_settings(self, world_settings: WorldSettings) -> int:
		"""
		This method applies settings contained in an object to the simulation running and returns the ID of the frame they were implemented.

		**warning**: If synchronous mode is enabled, and there is a Traffic Manager running, this must be set to sync mode too. Read [this](adv_traffic_manager.md#synchronous-mode) to learn how to do it.

		:param world_settings: (WorldSettings) 

		:return: int
		"""
		...

	def on_tick(self, callback: WorldSnapshot) -> int:
		"""
		This method is used in [__asynchronous__ mode](https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/). It starts callbacks from the client for the function defined as `callback`, and returns the ID of the callback. The function will be called everytime the server ticks. It requires a carla.WorldSnapshot as argument, which can be retrieved from wait_for_tick(). Use remove_on_tick() to stop the callbacks.

		:param callback: (WorldSnapshot) Function with a snapshot as compulsory parameter that will be called when the client receives a tick.

		:return: int
		"""
		...

	def remove_on_tick(self, callback_id: callback):
		"""
		Stops the callback for `callback_id` started with on_tick().

		:param callback_id: (callback) The callback to be removed. The ID is returned when creating the callback.
		"""
		...

	def tick(self, seconds: float = 10.0) -> int:
		"""
		This method is used in [__synchronous__ mode](https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/), when the server waits for a client tick before computing the next frame. This method will send the tick, and give way to the server. It returns the ID of the new frame computed by the server.

		*note*: If no tick is received in synchronous mode, the simulation will freeze. Also, if many ticks are received from different clients, there may be synchronization issues. Please read the docs about [synchronous mode](https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/) to learn more.

		:param seconds: (float) Maximum time the server should wait for a tick. It is set to 10.0 by default.

		:return: int
		"""
		...

	def wait_for_tick(self, seconds: float = 10.0) -> WorldSnapshot:
		"""
		This method is used in [__asynchronous__ mode](https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/). It makes the client wait for a server tick. When the next frame is computed, the server will tick and return a snapshot describing the new state of the world.

		:param seconds: (float) Maximum time the server should wait for a tick. It is set to 10.0 by default.

		:return: carla.WorldSnapshot
		"""
		...

	def spawn_actor(self, blueprint: ActorBlueprint, transform: Transform, attach_to: Actor = None, attachment: AttachmentType = Rigid) -> Actor:
		"""
		The method will create, return and spawn an actor into the world. The actor will need an available blueprint to be created and a transform (location and rotation). It can also be attached to a parent with a certain attachment type.

		:param blueprint: (ActorBlueprint) The reference from which the actor will be created.

		:param transform: (Transform) Contains the location and orientation the actor will be spawned with.

		:param attach_to: (Actor) The parent object that the spawned actor will follow around.

		:param attachment: (AttachmentType) Determines how fixed and rigorous should be the changes in position according to its parent object.

		:return: carla.Actor
		"""
		...

	def try_spawn_actor(self, blueprint: ActorBlueprint, transform: Transform, attach_to: Actor = None, attachment: AttachmentType = Rigid) -> Actor:
		"""
		Same as spawn_actor() but returns None on failure instead of throwing an exception.

		:param blueprint: (ActorBlueprint) The reference from which the actor will be created.

		:param transform: (Transform) Contains the location and orientation the actor will be spawned with.

		:param attach_to: (Actor) The parent object that the spawned actor will follow around.

		:param attachment: (AttachmentType) Determines how fixed and rigorous should be the changes in position according to its parent object.

		:return: carla.Actor
		"""
		...

	def get_actor(self, actor_id: int) -> Actor:
		"""
		Looks up for an actor by ID and returns None if not found.

		:param actor_id: (int) 

		:return: carla.Actor
		"""
		...

	def get_actors(self, actor_ids: list = None) -> ActorList:
		"""
		Retrieves a list of carla.Actor elements, either using a list of IDs provided or just listing everyone on stage. If an ID does not correspond with any actor, it will be excluded from the list returned, meaning that both the list of IDs and the list of actors may have different lengths.

		:param actor_ids: (list) The IDs of the actors being searched. By default it is set to None and returns every actor on scene.

		:return: carla.ActorList
		"""
		...

	def get_blueprint_library(self) -> BlueprintLibrary:
		"""
		Returns a list of actor blueprints available to ease the spawn of these into the world.

		:return: carla.BlueprintLibrary
		"""
		...

	def get_vehicles_light_states(self) -> dict:
		"""
		Returns a dict where the keys are carla.Actor IDs and the values are carla.VehicleLightState of that vehicle.

		:return: dict
		"""
		...

	def get_level_bbs(self, actor_type: CityObjectLabel = Any) -> array[BoundingBox]:
		"""
		Returns an array of bounding boxes with location and rotation in world space. The method returns all the bounding boxes in the level by default, but the query can be filtered by semantic tags with the argument `actor_type`.

		:param actor_type: (CityObjectLabel) Semantic tag of the elements contained in the bounding boxes that are returned.

		:return: array(carla.BoundingBox)
		"""
		...

	def get_environment_objects(self, object_type: CityObjectLabel = Any) -> array[EnvironmentObject]:
		"""
		Returns a list of EnvironmentObject with the requested semantic tag.  The method returns all the EnvironmentObjects in the level by default, but the query can be filtered by semantic tags with the argument `object_type`.

		:param object_type: (CityObjectLabel) Semantic tag of the EnvironmentObjects that are returned.

		:return: array(carla.EnvironmentObject)
		"""
		...

	def enable_environment_objects(self, env_objects_ids: set[int], enable: bool):
		"""
		Enable or disable a set of EnvironmentObject identified by their id. These objects will appear or disappear from the level.

		:param env_objects_ids: (set[int]) Set of EnvironmentObject ids to change.

		:param enable: (bool) State to be applied to all the EnvironmentObject of the set.
		"""
		...

	def get_lightmanager(self) -> LightManager:
		"""
		Returns an instance of carla.LightManager that can be used to handle the lights in the scene.

		:return: carla.LightManager
		"""
		...

	def freeze_all_traffic_lights(self, frozen: bool):
		"""
		Freezes or unfreezes all traffic lights in the scene. Frozen traffic lights can be modified by the user but the time will not update them until unfrozen.

		:param frozen: (bool) 
		"""
		...

	def reset_all_traffic_lights(self):
		"""
		Resets the cycle of all traffic lights in the map to the initial state.
		"""
		...

	def get_map(self) -> Map:
		"""
		Asks the server for the XODR containing the map file, and returns this parsed as a carla.Map.

		**warning**: This method does call the simulation. It is expensive, and should only be called once.

		:return: carla.Map
		"""
		...

	def get_traffic_light(self, landmark: Landmark) -> TrafficLight:
		"""
		Provided a landmark, returns the traffic light object it describes.

		:param landmark: (Landmark) The landmark object describing a traffic light.

		:return: carla.TrafficLight
		"""
		...

	def get_traffic_light_from_opendrive_id(self, traffic_light_id: str) -> TrafficLight:
		"""
		Returns the traffic light actor corresponding to the indicated OpenDRIVE id.

		:param traffic_light_id: (str) The OpenDRIVE id.

		:return: carla.TrafficLight
		"""
		...

	def get_traffic_lights_from_waypoint(self, waypoint: Waypoint, distance: float) -> list[TrafficLight]:
		"""
		This function performs a search along the road in front of the specified waypoint and returns a list of traffic light actors found in the specified search distance.

		:param waypoint: (Waypoint) The input waypoint.

		:param distance: (float) Search distance.

		:return: list(carla.TrafficLight)
		"""
		...

	def get_traffic_lights_in_junction(self, junction_id: int) -> list[TrafficLight]:
		"""
		Returns the list of traffic light actors affecting the junction indicated in `junction_id`.

		:param junction_id: (int) The id of the junction.

		:return: list(carla.TrafficLight)
		"""
		...

	def get_traffic_sign(self, landmark: Landmark) -> TrafficSign:
		"""
		Provided a landmark, returns the traffic sign object it describes.

		:param landmark: (Landmark) The landmark object describing a traffic sign.

		:return: carla.TrafficSign
		"""
		...

	def get_random_location_from_navigation(self) -> Location:
		"""
		This can only be used with walkers. It retrieves a random location to be used as a destination using the go_to_location() method in carla.WalkerAIController. This location will be part of a sidewalk. Roads, crosswalks and grass zones are excluded. The method does not take into consideration locations of existing actors so if a collision happens when trying to spawn an actor, it will return an error. Take a look at [`generate_traffic.py`](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/generate_traffic.py) for an example.

		:return: carla.Location
		"""
		...

	def get_settings(self) -> WorldSettings:
		"""
		Returns an object containing some data about the simulation such as synchrony between client and server or rendering mode.

		:return: carla.WorldSettings
		"""
		...

	def get_snapshot(self) -> WorldSnapshot:
		"""
		Returns a snapshot of the world at a certain moment comprising all the information about the actors.

		:return: carla.WorldSnapshot
		"""
		...

	def get_spectator(self) -> Actor:
		"""
		Returns the spectator actor. The spectator is a special type of actor created by Unreal Engine, usually with ID=0, that acts as a camera and controls the view in the simulator window.

		:return: carla.Actor
		"""
		...

	def get_weather(self) -> WeatherParameters:
		"""
		Retrieves an object containing weather parameters currently active in the simulation, mainly cloudiness, precipitation, wind and sun position.

		:return: carla.WeatherParameters
		"""
		...

	def set_weather(self, weather: WeatherParameters):
		"""
		Changes the weather parameteres ruling the simulation to another ones defined in an object.

		:param weather: (WeatherParameters) New conditions to be applied.
		"""
		...

	def cast_ray(self, initial_location: Location, final_location: Location) -> list[LabelledPoint]:
		"""
		Casts a ray from the specified initial_location to final_location. The function then detects all geometries intersecting the ray and returns a list of carla.LabelledPoint in order.

		:param initial_location: (Location) The initial position of the ray.

		:param final_location: (Location) The final position of the ray.

		:return: list(carla.LabelledPoint)
		"""
		...

	def project_point(self, location: Location, direction: Vector3D, search_distance: float) -> LabelledPoint:
		"""
		Projects the specified point to the desired direction in the scene. The functions casts a ray from location in a direction and returns a carla.Labelled object with the first geometry this ray intersects. If no geometry is found in the search_distance range the function returns `None`.

		:param location: (Location) The point to be projected.

		:param direction: (Vector3D) The direction of projection.

		:param search_distance: (float) The maximum distance to perform the projection

		:return: carla.LabelledPoint
		"""
		...

	def ground_projection(self, location: Location, search_distance: float) -> LabelledPoint:
		"""
		Projects the specified point downwards in the scene. The functions casts a ray from location in the direction (0,0,-1) (downwards) and returns a carla.LabelledPoint object with the first geometry this ray intersects (usually the ground). If no geometry is found in the search_distance range the function returns `None`.

		:param location: (Location) The point to be projected.

		:param search_distance: (float) The maximum distance to perform the projection

		:return: carla.LabelledPoint
		"""
		...

	def load_map_layer(self, map_layers: MapLayer):
		"""
		Loads the selected layers to the level. If the layer is already loaded the call has no effect.

		**warning**: This only affects "Opt" maps. The minimum layout includes roads, sidewalks, traffic lights and traffic signs.

		:param map_layers: (MapLayer) Mask of level layers to be loaded.
		"""
		...

	def unload_map_layer(self, map_layers: MapLayer):
		"""
		Unloads the selected layers to the level. If the layer is already unloaded the call has no effect.

		**warning**: This only affects "Opt" maps. The minimum layout includes roads, sidewalks, traffic lights and traffic signs.

		:param map_layers: (MapLayer) Mask of level layers to be unloaded.
		"""
		...

	def set_pedestrians_cross_factor(self, percentage: float):
		"""


		*note*: Should be set before pedestrians are spawned.

		:param percentage: (float) Sets the percentage of pedestrians that can walk on the road or cross at any point on the road. Value should be between `0.0` and `1.0`. For example, a value of `0.1` would allow 10% of pedestrians to walk on the road. __Default is `0.0`__.
		"""
		...

	def set_pedestrians_seed(self, seed: int):
		"""


		*note*: Should be set before pedestrians are spawned. If you want to repeat the same exact bodies (blueprint) for each pedestrian, then use the same seed in the Python code (where the blueprint is choosen randomly) and here, otherwise the pedestrians will repeat the same paths but the bodies will be different.

		:param seed: (int) Sets the seed to use for any random number generated in relation to pedestrians.
		"""
		...

	def apply_color_texture_to_object(self, object_name: str, material_parameter: MaterialParameter, texture: TextureColor):
		"""
		Applies a `texture` object in the field corresponfing to `material_parameter` (normal, diffuse, etc) to the object in the scene corresponding to `object_name`.

		:param object_name: (str) 

		:param material_parameter: (MaterialParameter) 

		:param texture: (TextureColor) 
		"""
		...

	def apply_float_color_texture_to_object(self, object_name: str, material_parameter: MaterialParameter, texture: TextureFloatColor):
		"""
		Applies a `texture` object in the field corresponfing to `material_parameter` (normal, diffuse, etc) to the object in the scene corresponding to `object_name`.

		:param object_name: (str) 

		:param material_parameter: (MaterialParameter) 

		:param texture: (TextureFloatColor) 
		"""
		...

	def apply_textures_to_object(self, object_name: str, diffuse_texture: TextureColor, emissive_texture: TextureFloatColor, normal_texture: TextureFloatColor, ao_roughness_metallic_emissive_texture: TextureFloatColor):
		"""
		Applies all texture fields in carla.MaterialParameter to the object `object_name`. Empty textures here will not be applied.

		:param object_name: (str) 

		:param diffuse_texture: (TextureColor) 

		:param emissive_texture: (TextureFloatColor) 

		:param normal_texture: (TextureFloatColor) 

		:param ao_roughness_metallic_emissive_texture: (TextureFloatColor) 
		"""
		...

	def apply_color_texture_to_objects(self, objects_name_list: list[str], material_parameter: MaterialParameter, texture: TextureColor):
		"""
		Applies a `texture` object in the field corresponfing to `material_parameter` (normal, diffuse, etc) to the object in the scene corresponding to all objects in `objects_name_list`.

		:param objects_name_list: (list[str]) 

		:param material_parameter: (MaterialParameter) 

		:param texture: (TextureColor) 
		"""
		...

	def apply_float_color_texture_to_objects(self, objects_name_list: list[str], material_parameter: MaterialParameter, texture: TextureFloatColor):
		"""
		Applies a `texture` object in the field corresponfing to `material_parameter` (normal, diffuse, etc) to the object in the scene corresponding to all objects in `objects_name_list`.

		:param objects_name_list: (list[str]) 

		:param material_parameter: (MaterialParameter) 

		:param texture: (TextureFloatColor) 
		"""
		...

	def apply_textures_to_objects(self, objects_name_list: list[str], diffuse_texture: TextureColor, emissive_texture: TextureFloatColor, normal_texture: TextureFloatColor, ao_roughness_metallic_emissive_texture: TextureFloatColor):
		"""
		Applies all texture fields in carla.MaterialParameter to all objects in `objects_name_list`. Empty textures here will not be applied.

		:param objects_name_list: (list[str]) 

		:param diffuse_texture: (TextureColor) 

		:param emissive_texture: (TextureFloatColor) 

		:param normal_texture: (TextureFloatColor) 

		:param ao_roughness_metallic_emissive_texture: (TextureFloatColor) 
		"""
		...

	def get_names_of_all_objects(self) -> list[str]:
		"""
		Returns a list of the names of all objects in the scene that can be painted with the apply texture functions.

		:return: list(str)
		"""
		...

	def __str__(self) -> str:
		"""
		The content of the world is parsed and printed as a brief report of its current state.

		:return: string
		"""
		...



class DebugHelper:
	"""
	Helper class part of carla.World that defines methods for creating debug shapes. By default, shapes last one second. They can be permanent, but take into account the resources needed to do so. Take a look at the snipets available for this class to learn how to debug easily in CARLA.
	"""

	def draw_arrow(self, begin: Location, end: Location, thickness: float = 0.1, arrow_size: float = 0.1, color: Color = (255,0,0), life_time: float = -1.0):
		"""
		Draws an arrow from `begin` to `end` pointing in that direction.

		:param begin: (Location) Point in the coordinate system where the arrow starts.

		:param end: (Location) Point in the coordinate system where the arrow ends and points towards to.

		:param thickness: (float) Density of the line.

		:param arrow_size: (float) Size of the tip of the arrow.

		:param color: (Color) RGB code to color the object. Red by default.

		:param life_time: (float) Shape's lifespan. By default it only lasts one frame. Set this to 0 for permanent shapes.
		"""
		...

	def draw_box(self, box: BoundingBox, rotation: Rotation, thickness: float = 0.1, color: Color = (255,0,0), life_time: float = -1.0):
		"""
		Draws a box, ussually to act for object colliders.

		:param box: (BoundingBox) Object containing a location and the length of a box for every axis.

		:param rotation: (Rotation) Orientation of the box according to Unreal Engine's axis system.

		:param thickness: (float) Density of the lines that define the box.

		:param color: (Color) RGB code to color the object. Red by default.

		:param life_time: (float) Shape's lifespan. By default it only lasts one frame. Set this to 0 for permanent shapes.
		"""
		...

	def draw_line(self, begin: Location, end: Location, thickness: float = 0.1, color: Color = (255,0,0), life_time: float = -1.0):
		"""
		Draws a line in between `begin` and `end`.

		:param begin: (Location) Point in the coordinate system where the line starts.

		:param end: (Location) Spot in the coordinate system where the line ends.

		:param thickness: (float) Density of the line.

		:param color: (Color) RGB code to color the object. Red by default.

		:param life_time: (float) Shape's lifespan. By default it only lasts one frame. Set this to 0 for permanent shapes.
		"""
		...

	def draw_point(self, location: Location, size: float = 0.1, color: Color = (255,0,0), life_time: float = -1.0):
		"""
		Draws a point `location`.

		:param location: (Location) Spot in the coordinate system to center the object.

		:param size: (float) Density of the point.

		:param color: (Color) RGB code to color the object. Red by default.

		:param life_time: (float) Shape's lifespan. By default it only lasts one frame. Set this to 0 for permanent shapes.
		"""
		...

	def draw_string(self, location: Location, text: str, draw_shadow: bool = False, color: Color = (255,0,0), life_time: float = -1.0):
		"""
		Draws a string in a given location of the simulation which can only be seen server-side.

		:param location: (Location) Spot in the simulation where the text will be centered.

		:param text: (str) Text intended to be shown in the world.

		:param draw_shadow: (bool) Casts a shadow for the string that could help in visualization. It is disabled by default.

		:param color: (Color) RGB code to color the string. Red by default.

		:param life_time: (float) Shape's lifespan. By default it only lasts one frame. Set this to 0 for permanent shapes.
		"""
		...



