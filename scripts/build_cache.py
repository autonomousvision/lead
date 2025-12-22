import torch
from lead.training.data_loader.carla_dataset import CARLAData
from lead.training.config_training import TrainingConfig
from tqdm import tqdm

config = TrainingConfig()
config.use_persistent_cache = True
config.use_training_session_cache = False
config.force_rebuild_data_cache = True

for k, v in config.training_dict().items():
    print(k, v)

data = CARLAData(
    root=config.carla_data,
    config=config,
    training_session_cache=None,
    build_cache=True
)
dataloader = torch.utils.data.DataLoader(
    data,
    batch_size=config.assigned_cpu_cores,
    shuffle=False,
    num_workers=config.assigned_cpu_cores,
    prefetch_factor=1
)

for i, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
    pass
