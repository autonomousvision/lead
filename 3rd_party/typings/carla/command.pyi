from __future__ import annotations

from typing import Any
import numpy as np

import carla

class Response:
	"""
	States the result of executing a command as either the ID of the actor to whom the command was applied to (when succeeded) or an error string (when failed).  actor ID, depending on whether or not the command succeeded. The method apply_batch_sync() in carla.Client returns a list of these to summarize the execution of a batch.
	"""

	actor_id: int
	"""Actor to whom the command was applied to. States that the command was successful."""

	error: str
	"""A string stating the command has failed."""

	def has_error(self) -> bool:
		"""
		Returns True if the command execution fails, and False if it was successful.

		:return: bool
		"""
		...



class SpawnActor:
	"""
	Command adaptation of spawn_actor() in carla.World. Spawns an actor into the world based on the blueprint provided and the transform. If a parent is provided, the actor is attached to it.
	"""

	transform: carla.Transform
	"""Transform to be applied."""

	parent_id: int
	"""Identificator of the parent actor."""

	def __init__(self):
		"""

		"""
		...

	def __init__(self, blueprint: carla.ActorBlueprint, transform: carla.Transform):
		"""


		:param blueprint: (carla.ActorBlueprint) 

		:param transform: (carla.Transform) 
		"""
		...

	def __init__(self, blueprint: carla.ActorBlueprint, transform: carla.Transform, parent: str):
		"""


		:param blueprint: (carla.ActorBlueprint) 

		:param transform: (carla.Transform) 

		:param parent: (str) 
		"""
		...

	def then(self, command: str):
		"""
		Links another command to be executed right after. It allows to ease very common flows such as spawning a set of vehicles by command and then using this method to set them to autopilot automatically.

		:param command: (str) a Carla command.
		"""
		...



class DestroyActor:
	"""
	Command adaptation of destroy() in carla.Actor that tells the simulator to destroy this actor. It has no effect if the actor was already destroyed. When executed with apply_batch_sync() in carla.Client there will be a command.Response that will return a boolean stating whether the actor was successfully destroyed.
	"""

	actor_id: int
	"""Actor affected by the command"""

	def __init__(self, actor: str):
		"""


		:param actor: (str) Actor or its ID to whom the command will be applied to.
		"""
		...



class ApplyVehiclePhysicsControl:
	"""
	Command adaptation of apply_physics_control() in carla.Vehicle. Applies a new physics control to a vehicle, modifying its physical parameters.
	"""

	actor_id: int
	"""Vehicle actor affected by the command."""

	physics_control: carla.VehiclePhysicsControl
	"""Physics control to be applied."""

	def __init__(self, actor: str, physics_control: carla.VehiclePhysicsControl):
		"""


		:param actor: (str) Actor or its ID to whom the command will be applied to.

		:param physics_control: (carla.VehiclePhysicsControl) 
		"""
		...



class ApplyVehicleControl:
	"""
	Command adaptation of apply_control() in carla.Vehicle. Applies a certain control to a vehicle.
	"""

	actor_id: int
	"""Vehicle actor affected by the command."""

	control: carla.VehicleControl
	"""Vehicle control to be applied."""

	def __init__(self, actor: str, control: carla.VehicleControl):
		"""


		:param actor: (str) Actor or its ID to whom the command will be applied to.

		:param control: (carla.VehicleControl) 
		"""
		...



class ApplyVehicleAckermannControl:
	"""
	Command adaptation of apply_ackermann_control() in carla.Vehicle. Applies a certain akermann control to a vehicle.
	"""

	actor_id: int
	"""Vehicle actor affected by the command."""

	control: carla.AckermannVehicleControl
	"""Vehicle ackermann control to be applied."""

	def __init__(self, actor: str, control: carla.AckermannVehicleControl):
		"""


		:param actor: (str) Actor or its ID to whom the command will be applied to.

		:param control: (carla.AckermannVehicleControl) 
		"""
		...



class ApplyWalkerControl:
	"""
	Command adaptation of apply_control() in carla.Walker. Applies a control to a walker.
	"""

	actor_id: int
	"""Walker actor affected by the command."""

	control: carla.WalkerControl
	"""Walker control to be applied."""

	def __init__(self, actor: str, control: carla.WalkerControl):
		"""


		:param actor: (str) Actor or its ID to whom the command will be applied to.

		:param control: (carla.WalkerControl) 
		"""
		...



class ApplyTransform:
	"""
	Command adaptation of set_transform() in carla.Actor. Sets a new transform to an actor.
	"""

	actor_id: int
	"""Actor affected by the command."""

	transform: carla.Transform
	"""Transformation to be applied."""

	def __init__(self, actor: str, transform: carla.Transform):
		"""


		:param actor: (str) Actor or its ID to whom the command will be applied to.

		:param transform: (carla.Transform) 
		"""
		...



class ApplyWalkerState:
	"""
	Apply a state to the walker actor. Specially useful to initialize an actor them with a specific location, orientation and speed.
	"""

	actor_id: int
	"""Walker actor affected by the command."""

	transform: carla.Transform
	"""Transform to be applied."""

	speed: float
	"""Speed to be applied."""

	def __init__(self, actor: str, transform: carla.Transform, speed: float):
		"""


		:param actor: (str) Actor or its ID to whom the command will be applied to.

		:param transform: (carla.Transform) 

		:param speed: (float) 
		"""
		...



class ApplyTargetVelocity:
	"""
	Command adaptation of set_target_velocity() in carla.Actor.
	"""

	actor_id: int
	"""Actor affected by the command."""

	velocity: carla.Vector3D
	"""The 3D velocity applied to the actor."""

	def __init__(self, actor: str, velocity: carla.Vector3D):
		"""


		:param actor: (str) Actor or its ID to whom the command will be applied to.

		:param velocity: (carla.Vector3D) Velocity vector applied to the actor.
		"""
		...



class ApplyTargetAngularVelocity:
	"""
	Command adaptation of set_target_angular_velocity() in carla.Actor. Sets the actor's angular velocity vector.
	"""

	actor_id: int
	"""Actor affected by the command."""

	angular_velocity: carla.Vector3D
	"""The 3D angular velocity that will be applied to the actor."""

	def __init__(self, actor: str, angular_velocity: carla.Vector3D):
		"""


		:param actor: (str) Actor or its ID to whom the command will be applied to.

		:param angular_velocity: (carla.Vector3D) Angular velocity vector applied to the actor.
		"""
		...



class ApplyImpulse:
	"""
	Command adaptation of add_impulse() in carla.Actor. Applies an impulse to an actor.
	"""

	actor_id: int
	"""Actor affected by the command."""

	impulse: carla.Vector3D
	"""Impulse applied to the actor."""

	def __init__(self, actor: str, impulse: carla.Vector3D):
		"""


		:param actor: (str) Actor or its ID to whom the command will be applied to.

		:param impulse: (carla.Vector3D) 
		"""
		...



class ApplyForce:
	"""
	Command adaptation of add_force() in carla.Actor. Applies a force to an actor.
	"""

	actor_id: int
	"""Actor affected by the command."""

	force: carla.Vector3D
	"""Force applied to the actor over time."""

	def __init__(self, actor: str, force: carla.Vector3D):
		"""


		:param actor: (str) Actor or its ID to whom the command will be applied to.

		:param force: (carla.Vector3D) 
		"""
		...



class ApplyAngularImpulse:
	"""
	Command adaptation of add_angular_impulse() in carla.Actor. Applies an angular impulse to an actor.
	"""

	actor_id: int
	"""Actor affected by the command."""

	impulse: carla.Vector3D
	"""Angular impulse applied to the actor."""

	def __init__(self, actor: str, impulse: carla.Vector3D):
		"""


		:param actor: (str) Actor or its ID to whom the command will be applied to.

		:param impulse: (carla.Vector3D) 
		"""
		...



class ApplyTorque:
	"""
	Command adaptation of add_torque() in carla.Actor. Applies a torque to an actor.
	"""

	actor_id: int
	"""Actor affected by the command."""

	torque: carla.Vector3D
	"""Torque applied to the actor over time."""

	def __init__(self, actor: str, torque: carla.Vector3D):
		"""


		:param actor: (str) Actor or its ID to whom the command will be applied to.

		:param torque: (carla.Vector3D) 
		"""
		...



class SetSimulatePhysics:
	"""
	Command adaptation of set_simulate_physics() in carla.Actor. Determines whether an actor will be affected by physics or not.
	"""

	actor_id: int
	"""Actor affected by the command."""

	enabled: bool
	"""If physics should be activated or not."""

	def __init__(self, actor: str, enabled: bool):
		"""


		:param actor: (str) Actor or its ID to whom the command will be applied to.

		:param enabled: (bool) 
		"""
		...



class SetAutopilot:
	"""
	Command adaptation of set_autopilot() in carla.Vehicle. Turns on/off the vehicle's autopilot mode.
	"""

	actor_id: int
	"""Actor that is affected by the command."""

	enabled: bool
	"""If autopilot should be activated or not."""

	port: np.uint16
	"""Port of the Traffic Manager where the vehicle is to be registered or unlisted."""

	def __init__(self, actor: str, enabled: bool, port: np.uint16 = 8000):
		"""


		:param actor: (str) Actor or its ID to whom the command will be applied to.

		:param enabled: (bool) 

		:param port: (np.uint16) The Traffic Manager port where the vehicle is to be registered or unlisted. If __None__ is passed, it will consider a TM at default port `8000`.
		"""
		...



class SetVehicleLightState:
	"""
	Command adaptation of set_light_state() in carla.Vehicle. Sets the light state of a vehicle.
	"""

	actor_id: int
	"""Actor that is affected by the command."""

	light_state: carla.VehicleLightState
	"""Defines the light state of a vehicle."""

	def __init__(self, actor: str, light_state: carla.VehicleLightState):
		"""


		:param actor: (str) Actor or its ID to whom the command will be applied to.

		:param light_state: (carla.VehicleLightState) Recaps the state of the lights of a vehicle, these can be used as a flags.
		"""
		...



class SetEnableGravity:
	"""
	Command adaptation of set_enable_gravity() in carla.Actor. Enables or disables gravity on an actor.
	"""

	actor_id: str
	"""Actor that is affected by the command."""

	enabled: bool
	""""""

	def __init__(self, actor: str, enabled: bool):
		"""


		:param actor: (str) Actor or Actor ID to which the command will be applied to.

		:param enabled: (bool) 
		"""
		...



class ShowDebugTelemetry:
	"""
	Command adaptation of show_debug_telemetry() in carla.Actor. Displays vehicle control telemetry data.
	"""

	actor_id: str
	"""Actor that is affected by the command."""

	enabled: bool
	""""""

	def __init__(self, actor: str, enabled: bool):
		"""


		:param actor: (str) Actor or Actor ID to which the command will be applied to.

		:param enabled: (bool) 
		"""
		...



