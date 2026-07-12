"""WorldStateMixin: lifecycle plus a cached per-step view of the CARLA world.

Split into category mixins: lifecycle, ego state, route state, surrounding
actors and scenario state.
"""

from leaderboard.autoagents import autonomous_agent_local

from lead.common.base_agent import BaseAgent
from lead.lead.world_state.actors_state import ActorsStateMixin
from lead.lead.world_state.ego_state import EgoStateMixin
from lead.lead.world_state.lifecycle import LifecycleMixin
from lead.lead.world_state.route_state import RouteStateMixin
from lead.lead.world_state.scenario_state import ScenarioStateMixin
from lead.lead.world_state.static_actors import StaticActorsMixin


class WorldStateMixin(
    LifecycleMixin,
    EgoStateMixin,
    RouteStateMixin,
    ActorsStateMixin,
    ScenarioStateMixin,
    StaticActorsMixin,
    BaseAgent,
    autonomous_agent_local.AutonomousAgent,
):
    """Lifecycle (setup/_init) + cached per-step view of the CARLA world."""


__all__ = [
    "ActorsStateMixin",
    "EgoStateMixin",
    "LifecycleMixin",
    "RouteStateMixin",
    "ScenarioStateMixin",
    "StaticActorsMixin",
    "WorldStateMixin",
]
