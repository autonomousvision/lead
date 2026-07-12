"""Agent section of the config tree: which learned driving policy to use."""

from lead.config.agent.transfuser_config import TransfuserConfig
from lead.config.node import ConfigNode, child_node


class AgentConfig(ConfigNode):
    """Configuration of the learned driving policy (model architecture).

    The policy implementation is swappable: ``target`` names the
    :class:`~lead.policy.abstract_policy.AbstractPolicy` subclass to
    instantiate for training and evaluation, and an agent config profile
    (``config_profiles/agent/``) can change it together with its
    hyperparameters.
    """

    # Name of the agent config profile (yaml in ``config_profiles/agent/``)
    # whose deltas are applied over these defaults.
    config_profile: str = "transfuser"

    # Dotted ``module:Class`` path of the AbstractPolicy implementation.
    target: str = "lead.policy.transfuser.transfuser:Transfuser"

    # TransFuser-specific architecture knobs.
    transfuser = child_node(TransfuserConfig)
