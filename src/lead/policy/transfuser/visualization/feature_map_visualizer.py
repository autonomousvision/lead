"""Debug visualizer plotting the raw label and prediction feature maps as a grid."""

import io

import jaxtyping as jt
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from PIL import Image

from lead.common.constants import RadarLabels
from lead.config import LeadConfig
from lead.policy.transfuser.dataloader.sample import TransfuserForwardBatch
from lead.policy.transfuser.transfuser import Prediction


class FeatureMapVisualizer:
    """Plots the CenterNet/BEV/radar label and prediction feature maps."""

    def __init__(
        self,
        lead_config: LeadConfig,
        data: TransfuserForwardBatch,
        prediction: Prediction,
    ) -> None:
        """Initialize the visualizer from one batched sample and its prediction.

        Args:
            lead_config: Root config tree.
            data: Dictionary containing batched input data tensors.
            prediction: Model outputs to visualize.
        """
        self.lead_config = lead_config
        self.config = lead_config.policy.transfuser
        self.data: TransfuserForwardBatch = data
        self.prediction: Prediction = prediction

    def visualize(self) -> jt.UInt8[npt.NDArray, "h w 3"]:
        """Render the feature-map grid.

        Returns:
            The rendered grid as an RGB image.
        """
        config = self.config
        data = self.data
        predictions = self.prediction

        # A head that is turned off contributes no label, so every label panel
        # tolerates a missing key and renders empty.
        def label(key: str, *index: int) -> jt.Shaped[npt.NDArray, "*shape"] | None:
            tensor = data.get(key)
            if tensor is None:
                return None
            for axis in index:
                tensor = tensor[axis]
            return tensor.detach().cpu().numpy()

        n_rows = 4
        n_cols = 4
        _, axs = plt.subplots(n_rows, n_cols, figsize=(24, 12))
        images = [
            (label("rasterized_lidar", 0, 0), "BEV LiDAR", "hot"),
            (label("center_net_heatmap", 0, 0), "Heatmap Label", "hot"),
            (label("center_net_wh", 0, 0), "WH Label", "hot"),
            (label("center_net_yaw_class", 0), "Yaw Class Label", "hot"),
            (label("center_net_yaw_res", 0, 0), "Yaw Res Label", "hot"),
            (label("center_net_offset", 0, 0), "Offset Label", "hot"),
            (label("center_net_velocity", 0, 0), "Velocity Label", "hot"),
            (label("bev_semantic", 0), "BEV Semantic Label", "hot"),
            (
                predictions.bounding_box.center_heatmap_pred[0]
                .detach()
                .argmax(0)
                .cpu()
                .numpy()
                if predictions.bounding_box is not None
                else None,
                "Heatmap Prediction",
                "hot",
            ),
            (
                predictions.bev_semantic[0].detach().argmax(0).cpu().numpy()
                if predictions.bev_semantic is not None
                else None,
                "BEV Semantic Prediction",
                "hot",
            ),
            (
                predictions.future_waypoints[0].detach().cpu().float().numpy()
                if predictions.future_waypoints is not None
                else None,
                "Waypoints",
                "hot",
            ),
            (
                predictions.route[0].detach().cpu().float().numpy()
                if predictions.route is not None
                else None,
                "Route",
                "hot",
            ),
            (label("radar", 0), "Radar Input", "hot"),
            (data.get("radar_detections"), "Radar Detection Label", "hot"),
            (predictions.radar_predictions, "Radar Detection Prediction", "hot"),
        ]

        for i, (img, title, cmap) in enumerate(images):
            ax = axs[i // n_cols, i % n_cols]
            ax.set_title(title)

            if img is None:
                continue

            if title == "Waypoints":
                ax.scatter(img[:, 0], img[:, 1], c="lime", s=20)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlim(0, 48)
                ax.set_ylim(-32, 32)
                ax.invert_yaxis()
            elif title == "Route":
                ax.plot(img[:, 0], img[:, 1], c="cyan", marker="o", markersize=3)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlim(0, 48)
                ax.set_ylim(-32, 32)
                ax.invert_yaxis()
            elif title == "Radar Input":
                x, y, vel = (img[:, 0], img[:, 1], img[:, 3])
                for xm, ym, vk in zip(x, y, vel, strict=True):
                    color = "red" if vk > 0 else "blue"
                    ax.scatter(xm, ym, c=color, s=abs(vk) * 3 + 2)
                ax.set_xlim(config.bev_min_x_meter, config.bev_max_x_meter)
                ax.set_ylim(config.bev_min_y_meter, config.bev_max_y_meter)
                ax.invert_yaxis()
            elif title == "Radar Detection Label":
                radar_labels = img[0].detach().cpu().float().numpy()
                x, y, v, valid = (
                    radar_labels[:, RadarLabels.X],
                    radar_labels[:, RadarLabels.Y],
                    radar_labels[:, RadarLabels.V],
                    radar_labels[:, RadarLabels.VALID],
                )
                for xm, ym, vk, validk in zip(x, y, v, valid, strict=True):
                    if validk > 0.5:
                        ax.scatter(xm, ym, c="blue", s=abs(vk) * 3 + 2)
                ax.set_xlim(config.bev_min_x_meter, config.bev_max_x_meter)
                ax.set_ylim(config.bev_min_y_meter, config.bev_max_y_meter)
                ax.invert_yaxis()
            elif title == "Radar Detection Prediction":
                radar_detection = img[0].detach().cpu().float().numpy()
                x, y, v, valid = (
                    radar_detection[:, RadarLabels.X],
                    radar_detection[:, RadarLabels.Y],
                    radar_detection[:, RadarLabels.V],
                    radar_detection[:, RadarLabels.VALID],
                )
                for xm, ym, vk, validk in zip(x, y, v, valid, strict=True):
                    if validk > 0.0:  # Implicit sigmoid threshold 0.5
                        ax.scatter(xm, ym, c="blue", s=abs(vk) * 3 + 2)
                ax.set_xlim(config.bev_min_x_meter, config.bev_max_x_meter)
                ax.set_ylim(config.bev_min_y_meter, config.bev_max_y_meter)
                ax.invert_yaxis()
            else:
                ax.imshow(img, cmap=cmap)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close()
        buf.seek(0)
        image = np.array(Image.open(buf).convert("RGB"))
        buf.close()
        return image
