# Add your own policy

A walkthrough of `lead.policy.ego_status`, the smallest policy in the repo: an MLP
that plans waypoints from the target point and the ego speed, ~340 lines.

| file                                               | holds                     |
| :------------------------------------------------- | :------------------------ |
| `policy/ego_status/dataloader/sample.py`           | the fields of one example |
| `policy/ego_status/dataloader/dataset.py`          | scene → arrays            |
| `policy/ego_status/ego_status.py`                  | the model                 |
| `evaluation/agents/ego_status/ego_status_agent.py` | prediction → control      |
| `config/policy/ego_status/ego_status_config.py`    | the knobs                 |

## 1. Declare inputs and outputs

Only listing the fields is required.

```python
# src/lead/policy/ego_status/dataloader/sample.py
@dataclass
class EgoStatusTrainingSample(PolicyTrainingSample):
    target_point: Float32[np.ndarray, " 2"] | None = None
    speed: float | None = None
    future_waypoints: Float32[np.ndarray, "n_waypoints 2"] | None = None
    extras: dict[str, typing.Any] = field(default_factory=dict)


class EgoStatusForwardBatch(TypedDict, total=False):
    target_point: Float32[torch.Tensor, "b 2"]
    speed: Float[torch.Tensor, " b"]
    future_waypoints: Float32[torch.Tensor, "b n_waypoints 2"]


class EgoStatusPrediction(TypedDict):
    waypoints: Float[torch.Tensor, "b n_waypoints 2"]
```

## 2. Reading 123D logs and featurization

```python
# src/lead/policy/ego_status/dataloader/dataset.py
class EgoStatusDataset(AbstractPolicyDataset):
    sample_class = EgoStatusTrainingSample

    # Config values that change what a cached tensor holds:
    @property
    def cache_finger_print(self) -> dict[str, str]:
        return {
            "future_ego_pose_iterations": str(
                self.lead_config.policy.ego_status.future_ego_pose_iterations,
            ),
        }

    def get_sample_parts(self) -> dict[str, SamplePart]:
        config = self.lead_config.policy.ego_status
        return {
            "ego_features": SamplePart(
                reads=SceneLoadingSpec(),
                builds=self._build_ego_features,
            ),
            "planning_targets": SamplePart(
                reads=SceneLoadingSpec(
                    future_iterations=config.future_ego_pose_iterations,
                ),
                builds=self._build_planning_targets,
                caches={"future_waypoints": "raw"},
            ),
        }

    def _build_ego_features(self, scene_data: SceneData) -> dict[str, typing.Any]:
        return {
            "target_point": scene_data.target_point.astype(np.float32),
            "speed": carla_decoding.carla_forward_speed(scene_data.ego_state),
        }

    def _build_planning_targets(self, scene_data: SceneData) -> dict[str, typing.Any]:
        # future_ego_states is transformed into the ego frame of this tick.
        return {"future_waypoints": ...}
```

## 3. The policy

The model, and how its input is built. Two paths: `build_dataset` for training,
`build_features` plus `features_to_batch` for inference, where there is one live
scene and no labels.

```python
# src/lead/policy/ego_status/ego_status.py
class EgoStatus(AbstractPolicy[EgoStatusForwardBatch, EgoStatusPrediction]):
    def forward(self, batch: EgoStatusForwardBatch) -> EgoStatusPrediction:
        features = torch.cat(
            [batch["target_point"].float(), batch["speed"].float().reshape(-1, 1)],
            dim=1,
        )
        hidden = self.backbone(features)
        return {
            "waypoints": self.waypoints_head(hidden).reshape(hidden.shape[0], -1, 2),
        }

    def compute_loss(
        self,
        predictions: EgoStatusPrediction,
        batch: EgoStatusForwardBatch,
    ) -> tuple[TaskLosses, AuxiliaryLog]:
        losses = {
            "loss_waypoints": functional.l1_loss(
                predictions["waypoints"],
                batch["future_waypoints"].float(),
            ),
        }
        return losses, {}

    def per_task_loss_weights(self, epoch: int) -> dict[str, float]:
        return {"loss_waypoints": 1.0}

    def get_policy_config(self) -> EgoStatusConfig:
        return self.config

    # Training only: the policy owns the whole read side. The scene filter is
    # inherited from AbstractPolicy.build_scene_filter.
    def build_scene_loader(self) -> SceneLoader:
        return SceneLoader(
            self.lead_config.training.data.py123d_data_root,
            self.build_scene_filter(),
            perturbation_probability=0.5,
        )

    def build_dataset(self) -> SizedDataset:
        return EgoStatusDataset(
            lead_config=self.lead_config,
            scene_loader=self.build_scene_loader(),
        )

    # Inference only. Training never calls these two.
    def build_features(self, scene_data: SceneData) -> Mapping[str, typing.Any]:
        # Must produce the same names and arrays as _build_ego_features above.
        return {
            "target_point": scene_data.target_point.astype(np.float32),
            "speed": carla_decoding.carla_forward_speed(scene_data.ego_state),
        }

    def features_to_batch(
        self,
        features: Mapping[str, typing.Any],
        device: torch.device,
    ) -> EgoStatusForwardBatch:
        batch: EgoStatusForwardBatch = {}
        for key in ("target_point", "speed"):
            batch[key] = torch.as_tensor(
                np.asarray(features[key]),
                dtype=torch.float32,
                device=device,
            )[None]
        return batch
```

|               | training                       | inference             |
| :------------ | :----------------------------- | :-------------------- |
| entry point   | `build_dataset()`              | `build_features()`    |
| featurization | `_build_ego_features()`        | `build_features()`    |
| batching      | `PolicyTrainingSample.collate` | `features_to_batch()` |

The two featurizations must agree or the paths drift apart. At two lines the
baseline just repeats them; TransFuser keeps its featurization in free functions in
`policy/transfuser/dataloader/features.py` and calls those from both sides. Do that for
anything larger.

Labels have no such pair: there are no labels at inference, so these two carry inputs
only.

## 4. The driving agent

Prediction in, `carla.VehicleControl` out. The base class does sensor decoding and
localization.

```python
# src/lead/evaluation/agents/ego_status/ego_status_agent.py
def get_entry_point():
    return "EgoStatusAgent"


class EgoStatusAgent(AbstractDrivingAgent):
    def setup_policy(self, checkpoint_dir: str) -> None:
        # Runs once, after the weights are loaded.
        self.waypoint_tracker = WaypointTracker(self.lead_config)

    def compute_control(
        self,
        prediction: EgoStatusPrediction,
        features: dict[str, typing.Any],
    ) -> carla.VehicleControl:
        steer, throttle, brake = self.waypoint_tracker.step(
            prediction["waypoints"],
            features["speed"].unsqueeze(1),
        )
        return carla.VehicleControl(
            steer=float(steer),
            throttle=float(throttle),
            brake=float(brake),
        )
```

`WaypointTracker` is an existing PID controller; `PathSpeedTracker` is the other one,
for policies predicting a path and a target speed. The directory name must match the
policy package, since `python -m lead` derives the agent path from `policy.target`.

## 5. The config section

Your policy's knobs, one section subclassing `AbstractPolicyConfig` — the contract
`get_policy_config` publishes. It requires `cache_store_dir_name` and
`input_cameras`; a section missing either cannot be instantiated.

```python
# src/lead/config/policy/ego_status/ego_status_config.py
class EgoStatusConfig(AbstractPolicyConfig):
    # Where this policy's cache store lives, under the dataset root.
    @overridable_property
    def cache_store_dir_name(self) -> str:
        return "ego_status_training_cache"

    @property
    def input_cameras(self) -> list[CameraID]:
        return []

    hidden_dim: int = 256
    num_hidden_layers: int = 2
```

The temporal window comes with the contract: one `_length_s` (seconds from the anchor)
and one `_frequency` (Hz) per modality, for the past and the future alike.
`num_ego_pose_prediction` derives from the two, so the head's output width and the
scene filter can never disagree.

```python
class EgoStatusConfig(AbstractPolicyConfig):
    past_ego_pose_length_s: float = 0.0
    past_lidar_length_s: float = 0.0
    past_radar_length_s: float = 0.0
    past_rgb_length_s: float = 0.0
    # Inherited: future_ego_pose_length_s = 2.0, future_ego_pose_frequency = 4
```

Register it as a child section in `PolicyConfig` and return it from the policy's
`get_policy_config`:

```python
ego_status = config_child_node(EgoStatusConfig)
```

## Run it

Both commands take the policy on the command line, and the dataset root comes from `.env`:

```console
user@host:~/lead$ bash scripts/common/build_cache.sh policy.target=lead.policy.ego_status.ego_status:EgoStatus   # always first; training never builds it
user@host:~/lead$ python -m lead.training.train policy.target=lead.policy.ego_status.ego_status:EgoStatus training.data.read_from_cache_store=true
```

See [training](training.md) for the phases and config overrides, and
[architecture](architecture.md) for how a sample is assembled.
