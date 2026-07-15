"""Expert (data collection) section of the config tree."""

from lead.config.expert.bicycle_model_config import BicycleModelConfig
from lead.config.expert.data_collection_config import DataCollectionConfig
from lead.config.expert.driving_config import ExpertDrivingConfig
from lead.config.expert.occlusion_config import ExpertOcclusionConfig
from lead.config.expert.perturbation_config import PerturbationConfig
from lead.config.expert.pid_config import ExpertPidConfig
from lead.config.expert.sensor_rig_config import SensorRigConfig
from lead.config.expert.simulation_config import SimulationConfig
from lead.config.expert.storage_config import StorageConfig
from lead.config.expert.visualization_config import ExpertVisualizationConfig
from lead.config.node import ConfigNode, child_node


class ExpertConfig(ConfigNode):
    """Configuration of the privileged expert and the data it collects.

    This section fully describes a dataset — the sensor rig, the planning
    area, storage encoding and the expert's driving behavior; data collection
    stores the resolved section with the dataset and training loads it back so
    shared values always match the data.
    """

    # Name of the expert config profile (yaml in ``config/profiles/expert/``)
    # whose deltas are applied over these defaults.
    config_profile: str = "default"

    simulation = child_node(SimulationConfig)
    sensor_rig = child_node(SensorRigConfig)
    bicycle_model = child_node(BicycleModelConfig)
    perturbation = child_node(PerturbationConfig)
    storage = child_node(StorageConfig)
    driving = child_node(ExpertDrivingConfig)
    pid = child_node(ExpertPidConfig)
    occlusion = child_node(ExpertOcclusionConfig)
    data_collection = child_node(DataCollectionConfig)
    visualization = child_node(ExpertVisualizationConfig)
