"""Debug visualizer plotting the raw label and prediction feature maps as a grid."""

import io
import typing

import jaxtyping as jt
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from PIL import Image

from lead.common.constants import RadarLabels
from lead.config import LeadConfig
from lead.policy.transfuser.transfuser import Prediction


class FeatureMapVisualizer:
    """Plots the CenterNet/BEV/radar label and prediction feature maps."""

    def __init__(
        self,
        lead_config: LeadConfig,
        data: dict[str, typing.Any],
        prediction: Prediction,
    ) -> None:
        """Initialize the visualizer from one batched sample and its prediction.

        Args:
            lead_config: Root config tree.
            data: Dictionary containing batched input data tensors.
            prediction: Model outputs to visualize.
        """
        self.lead_config = lead_config
        self.data_config = lead_config.expert.data_collection
        self.data: dict[str, typing.Any] = data
        self.prediction: Prediction = prediction

    def visualize(self) -> jt.UInt8[npt.NDArray, "h w 3"]:
        """Render the feature-map grid.

        Returns:
            The rendered grid as an RGB image.
        """
        data_config = self.data_config
        data = self.data
        predictions = self.prediction

        n_rows = 4
        n_cols = 4
        _, axs = plt.subplots(n_rows, n_cols, figsize=(24, 12))
        images = [
            (
                data["rasterized_lidar"][0, 0].detach().cpu().numpy(),
                "BEV LiDAR",
                "hot",
            ),
            (
                data["center_net_heatmap"][0, 0].detach().cpu().numpy(),
                "Heatmap Label",
                "hot",
            ),
            (data["center_net_wh"][0, 0].detach().cpu().numpy(), "WH Label", "hot"),
            (
                data["center_net_yaw_class"][0].detach().cpu().numpy(),
                "Yaw Class Label",
                "hot",
            ),
            (
                data["center_net_yaw_res"][0, 0].detach().cpu().numpy(),
                "Yaw Res Label",
                "hot",
            ),
            (
                data["center_net_offset"][0, 0].detach().cpu().numpy(),
                "Offset Label",
                "hot",
            ),
            (
                data["center_net_velocity"][0, 0].detach().cpu().numpy(),
                "Velocity Label",
                "hot",
            ),
            (
                data["bev_semantic"][0].detach().cpu().numpy(),
                "BEV Semantic Label",
                "hot",
            ),
            (
                predictions.pred_bounding_box.center_heatmap_pred[0]
                .detach()
                .argmax(0)
                .cpu()
                .numpy()
                if predictions.pred_bounding_box is not None
                else None,
                "Heatmap Prediction",
                "hot",
            ),
            (
                predictions.pred_bev_semantic[0].detach().argmax(0).cpu().numpy()
                if predictions.pred_bev_semantic is not None
                else None,
                "BEV Semantic Prediction",
                "hot",
            ),
            (
                predictions.pred_future_waypoints[0].detach().cpu().numpy()
                if predictions.pred_future_waypoints is not None
                else None,
                "Waypoints",
                "hot",
            ),
            (
                predictions.pred_route[0].detach().cpu().numpy()
                if predictions.pred_route is not None
                else None,
                "Route",
                "hot",
            ),
            (data["radar"][0].cpu().numpy(), "Radar Input", "hot"),
            (data.get("radar_detections"), "Radar Detection Label", "hot"),
            (predictions.pred_radar_predictions, "Radar Detection Prediction", "hot"),
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
                ax.set_xlim(data_config.min_x_meter, data_config.max_x_meter)
                ax.set_ylim(data_config.min_y_meter, data_config.max_y_meter)
                ax.invert_yaxis()
            elif title == "Radar Detection Label":
                radar_labels = img[0]
                x, y, v, valid = (
                    radar_labels[:, RadarLabels.X],
                    radar_labels[:, RadarLabels.Y],
                    radar_labels[:, RadarLabels.V],
                    radar_labels[:, RadarLabels.VALID],
                )
                for xm, ym, vk, validk in zip(x, y, v, valid, strict=True):
                    if validk > 0.5:
                        ax.scatter(xm, ym, c="blue", s=abs(vk) * 3 + 2)
                ax.set_xlim(data_config.min_x_meter, data_config.max_x_meter)
                ax.set_ylim(data_config.min_y_meter, data_config.max_y_meter)
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
                ax.set_xlim(data_config.min_x_meter, data_config.max_x_meter)
                ax.set_ylim(data_config.min_y_meter, data_config.max_y_meter)
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
