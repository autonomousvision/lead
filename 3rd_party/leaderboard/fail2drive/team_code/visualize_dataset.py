"""
This script is used to visualize the dataset and merge RGB, LiDAR and BEV views as a sanity check.
"""

from copy import deepcopy
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
from data import CARLA_Data
import torch
from config import GlobalConfig
import transfuser_utils as t_u
from torch.utils.data import DataLoader


# this function is copied from model.py
def visualize_model(
    config,
    step,
    save_path,
    rgb,
    lidar_bev,
    target_point,
    target_point_next,
    pred_checkpoint=None,
    gt_wp=None,
    gt_bbs=None,
    gt_speed=None,
    gt_bev_semantic=None,
):
  # 0 Car, 1 Pedestrian, 2 Red light, 3 Stop sign, 4 emergency vehicle
  color_classes = [
      np.array([255, 165, 0]),
      np.array([0, 255, 0]),
      np.array([255, 0, 0]),
      np.array([250, 160, 160]),
      np.array([16, 133, 133])
  ]

  size_width = int((config.max_y - config.min_y) * config.pixels_per_meter)
  size_height = int((config.max_x - config.min_x) * config.pixels_per_meter)

  scale_factor = 4
  origin_x_ratio = config.max_x / (config.max_x -
                                   config.min_x) if config.crop_bev and config.crop_bev_height_only_from_behind else 1
  origin = ((size_width * scale_factor) // 2, (origin_x_ratio * size_height * scale_factor) // 2)
  loc_pixels_per_meter = config.pixels_per_meter * scale_factor

  ## add rgb image and lidar
  if config.use_ground_plane:
    images_lidar = np.concatenate(list(lidar_bev.detach().cpu().numpy()[0][:1]), axis=1)
  else:
    images_lidar = np.concatenate(list(lidar_bev.detach().cpu().numpy()[0][:1]), axis=1)

  if False:
    images_lidar = images_lidar > 0.001
    images_lidar_vel_y_pos = (lidar_bev.detach().cpu().numpy()[0][1] > 0.1)
    images_lidar_vel_x_pos = (lidar_bev.detach().cpu().numpy()[0][2] > 0.1)
    images_lidar_vel_y_neg = (lidar_bev.detach().cpu().numpy()[0][1] < -0.1)
    images_lidar_vel_x_neg = (lidar_bev.detach().cpu().numpy()[0][2] < -0.1)

    images_lidar_vel_y_pos = np.roll(images_lidar_vel_y_pos, 4, axis=0)
    images_lidar_vel_x_pos = np.roll(images_lidar_vel_x_pos, 4, axis=1)
    images_lidar_vel_y_neg = np.roll(images_lidar_vel_y_neg, -4, axis=0)
    images_lidar_vel_x_neg = np.roll(images_lidar_vel_x_neg, -4, axis=1)
    shift_or = np.clip(
        images_lidar_vel_y_pos + images_lidar_vel_x_pos + images_lidar_vel_y_neg + images_lidar_vel_x_neg, 0, 1)

    images_lidar = 255 - (images_lidar * 255).astype(np.uint8)
    speed_color_channel = 255 - (shift_or * 255).astype(np.uint8)
    images_lidar = np.stack([images_lidar, speed_color_channel, np.ones_like(images_lidar) * 255], axis=-1)
  else:
    images_lidar = 255 - (images_lidar * 255).astype(np.uint8)
    images_lidar = np.stack([images_lidar, images_lidar, images_lidar], axis=-1)

  images_lidar = cv2.resize(images_lidar,
                            dsize=(images_lidar.shape[1] * scale_factor, images_lidar.shape[0] * scale_factor),
                            interpolation=cv2.INTER_NEAREST)
  # # Render road over image
  # road = self.ss_bev_manager.get_road()
  # # Alpha blending the road over the LiDAR
  # images_lidar = road[:, :, 3:4] * road[:, :, :3] + (1 - road[:, :, 3:4]) * images_lidar

  if gt_bev_semantic is not None:
    bev_semantic_indices = gt_bev_semantic[0].detach().cpu().numpy()
    converter = np.array(config.bev_classes_list)
    converter[1][0:3] = 40
    bev_semantic_image = converter[bev_semantic_indices, ...].astype('uint8')
    alpha = np.ones_like(bev_semantic_indices) * 0.33
    alpha = alpha.astype(float)
    alpha[bev_semantic_indices == 0] = 0.0
    alpha[bev_semantic_indices == 1] = 0.1

    alpha = cv2.resize(alpha, dsize=(alpha.shape[1] * 4, alpha.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
    alpha = np.expand_dims(alpha, 2)
    bev_semantic_image = cv2.resize(bev_semantic_image,
                                    dsize=(bev_semantic_image.shape[1] * 4, bev_semantic_image.shape[0] * 4),
                                    interpolation=cv2.INTER_NEAREST)
    images_lidar = bev_semantic_image * alpha + (1 - alpha) * images_lidar

    images_lidar = np.ascontiguousarray(images_lidar, dtype=np.uint8)

  # Draw wps
  # Red ground truth
  if gt_wp is not None:
    gt_wp_color = (255, 255, 0)
    for wp in gt_wp.detach().cpu().numpy()[0]:
      wp_x = wp[0] * loc_pixels_per_meter + origin[0]
      wp_y = wp[1] * loc_pixels_per_meter + origin[1]
      cv2.circle(images_lidar, (int(wp_x), int(wp_y)), radius=10, color=gt_wp_color, thickness=-1)

  # Green predicted checkpoint
  if pred_checkpoint is not None:
    for wp in pred_checkpoint.detach().cpu().numpy()[0]:
      wp_x = wp[0] * loc_pixels_per_meter + origin[0]
      wp_y = wp[1] * loc_pixels_per_meter + origin[1]
      cv2.circle(images_lidar, (int(wp_x), int(wp_y)),
                 radius=8,
                 lineType=cv2.LINE_AA,
                 color=(0, 128, 255),
                 thickness=-1)

  # Draw target points
  if config.use_tp:
    x_tp = target_point[0][0] * loc_pixels_per_meter + origin[0]
    y_tp = target_point[0][1] * loc_pixels_per_meter + origin[1]
    cv2.circle(images_lidar, (int(x_tp), int(y_tp)), radius=12, lineType=cv2.LINE_AA, color=(255, 0, 0), thickness=-1)

    # draw next tp too
    x_tpn = target_point_next[0][0] * loc_pixels_per_meter + origin[0]
    y_tpn = target_point_next[0][1] * loc_pixels_per_meter + origin[1]
    cv2.circle(images_lidar, (int(x_tpn), int(y_tpn)), radius=12, lineType=cv2.LINE_AA, color=(255, 0, 0), thickness=-1)

  # Visualize Ego vehicle
  sample_box = np.array([
      int(images_lidar.shape[0] / 2),
      int(origin_x_ratio * images_lidar.shape[1] / 2), config.ego_extent_x * loc_pixels_per_meter,
      config.ego_extent_y * loc_pixels_per_meter,
      np.deg2rad(90.0), 0.0
  ])
  images_lidar = t_u.draw_box(images_lidar, sample_box, color=(0, 200, 0), pixel_per_meter=16, thickness=4)

  if gt_bbs is not None:
    gt_bbs = gt_bbs.detach().cpu().numpy()[0]
    real_boxes = gt_bbs.sum(axis=-1) != 0.
    gt_bbs = gt_bbs[real_boxes]
    for box in gt_bbs:
      inv_brake = 1.0 - box[6]
      # car: box[7] == 0, walker: box[7] == 1, traffic_light: box[7] == 2, stop_sign: box[7] == 3
      color_box = deepcopy(color_classes[int(box[7])])
      color_box[1] = color_box[1] * inv_brake
      box[:4] = box[:4] * scale_factor
      images_lidar = t_u.draw_box(images_lidar, box, color=color_box, pixel_per_meter=loc_pixels_per_meter, thickness=4)

  images_lidar = np.rot90(images_lidar, k=1)
  images_lidar = np.ascontiguousarray(images_lidar, dtype=np.uint8)
  rgb_image = rgb[0].permute(1, 2, 0).detach().cpu().numpy()

  if gt_speed is not None:
    gt_speed_float = gt_speed.detach().cpu().item()
    cv2.putText(images_lidar, f'Speed: {gt_speed_float:.2f}', (10, 690), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1,
                cv2.LINE_AA)

  all_images = np.concatenate((rgb_image, images_lidar), axis=0)
  all_images = Image.fromarray(all_images.astype(np.uint8))

  store_path = str(str(save_path) + (f'/{step:04}.png'))
  Path(store_path).parent.mkdir(parents=True, exist_ok=True)
  all_images.save(store_path)


if __name__ == '__main__':
  device = 'cpu'

  # the source path should be one level in the directory hierarchy above the folder(s) that contain the actual rgb, lidar, ... folders
  # aka the SAVE_PATH from start_autopilot.sh
  source_path = ['/home/zimjulian/code/leaderboard2_human_data/database/debug_v4']
  save_path = '/home/zimjulian/code/leaderboard2_human_data/data?visualization/debug_v4/Yield3998'

  config = GlobalConfig()
  print('Loading data')
  train_set = CARLA_Data(root=source_path, config=config)

  dataloader_train = DataLoader(train_set)

  for i, data in enumerate(dataloader_train):  #enumerate(tqdm(dataloader_train, disable=rank != 0)):
    #if i % 10 == 0:
    print('+++++++++ ' + str(i) + ' +++++++++')
    rgb = data['rgb'].to(device, dtype=torch.float32)
    bev_semantic_label = data['bev_semantic'].to(device, dtype=torch.long)
    depth_label = data['depth'].to(device, dtype=torch.float32)
    lidar = data['lidar'].to(device, dtype=torch.float32)
    target_point = data['target_point'].to(device, dtype=torch.float32)
    target_point_next = data['target_point_next'].to(device, dtype=torch.float32)
    bbs = data['bounding_boxes'].to(device, dtype=torch.float32)
    gt_speed = data['speed'].to(device, dtype=torch.float32)
    target_speed_twohot = data['target_speed_twohot'].to(device, dtype=torch.float32)
    checkpoint = data['route'][:, :20].to(device, dtype=torch.float32)

    visualize_model(config=config,
                    step=i,
                    save_path=save_path,
                    rgb=rgb,
                    lidar_bev=lidar,
                    target_point=target_point,
                    target_point_next=target_point_next,
                    gt_wp=None,
                    gt_bbs=bbs,
                    gt_speed=gt_speed,
                    pred_checkpoint=checkpoint,
                    gt_bev_semantic=bev_semantic_label)
