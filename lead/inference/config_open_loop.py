from lead.common.config_base import BaseConfig


class OpenLoopConfig(BaseConfig):
    def __init__(self, raise_error_on_missing_key: bool = True):
        super().__init__()
        self.load_from_environment(
            loaded_config=None, env_key="OPEN_LOOP_CONFIG", raise_error_on_missing_key=raise_error_on_missing_key
        )

    # --- Speed and Control Settings ---
    # If true lower the target speed using a factor
    lower_target_speed = False
    # Factor to multiply the target speed with when lowering is enabled
    lower_target_speed_factor = 0.8
    # Confidence threshold for brake action (full brake applied if confidence exceeds this)
    brake_threshold = 0.9
    # If true be strict when load weight
    strict_weight_load = True

    def open_loop_dict(self):
        """Generate dictionary of all open loop configuration properties."""
        out = {}
        for k, v in OpenLoopConfig.__dict__.items():
            if isinstance(v, property):
                try:
                    out[k] = getattr(self, k)
                except Exception:
                    pass
        return out
