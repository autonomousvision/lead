"""Policy section of the config tree: which learned driving policy to use."""

from lead.config.node import ConfigNode, config_child_node
from lead.config.policy.ego_status.ego_status_config import EgoStatusConfig
from lead.config.policy.transfuser.transfuser_config import TransfuserConfig


class PolicyConfig(ConfigNode):
    """Configuration of the learned driving policy (model architecture).

    The policy implementation is swappable: ``target`` names the
    :class:`~lead.api.abstract_policy.AbstractPolicy` subclass to
    instantiate for training and evaluation. Each policy's knobs live in its
    own child section, an :class:`~lead.config.policy.abstract_policy_config.AbstractPolicyConfig`
    the policy publishes via ``get_policy_config``.
    """

    # Dotted ``module:Class`` path of the AbstractPolicy implementation.
    target: str = "lead.policy.transfuser.transfuser:Transfuser"

    # TransFuser-specific architecture knobs.
    transfuser = config_child_node(TransfuserConfig)

    # Ego-status baseline knobs.
    ego_status = config_child_node(EgoStatusConfig)
