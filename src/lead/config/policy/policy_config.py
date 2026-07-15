"""Policy section of the config tree: which learned driving policy to use."""

from lead.config.node import ConfigNode, child_node
from lead.config.policy.transfuser_config import TransfuserConfig


class PolicyConfig(ConfigNode):
    """Configuration of the learned driving policy (model architecture).

    The policy implementation is swappable: ``target`` names the
    :class:`~lead.policy.abstract_policy.AbstractPolicy` subclass to
    instantiate for training and evaluation, and a policy config profile
    (``config/profiles/policy/``) can change it together with its
    hyperparameters.
    """

    # Name of the policy config profile (yaml in ``config/profiles/policy/``)
    # whose deltas are applied over these defaults.
    config_profile: str = "transfuser"

    # Dotted ``module:Class`` path of the AbstractPolicy implementation.
    target: str = "lead.policy.transfuser.transfuser:Transfuser"

    # TransFuser-specific architecture knobs.
    transfuser = child_node(TransfuserConfig)
