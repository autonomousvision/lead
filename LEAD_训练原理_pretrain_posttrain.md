# LEAD 训练原理:Pretrain & Posttrain 的数据与训练方式

> 适用:CARLA Leaderboard 2.0 训练(`carla_leaderboard_mode`)。
> 基于仓库代码梳理:`lead/data_loader/carla_dataset.py`、`lead/training/config_training.py`、`docs/source/carla_training.md`。
> 配套:环境/踩坑见 `踩坑总结_LEAD_H20.md`。

---

## 0. 一句话

LEAD 沿用 **TransFuser 系的两阶段训练**,本质是**模仿学习**——数据是规则专家(PDM-Lite)在 CARLA 里开车录下来的,模型学着模仿专家。
**先 Pretrain 学「看懂世界」(感知),再 Posttrain 学「怎么开」(规划)**。两阶段**用同一份数据**,区别在于「用哪些标签训 + 怎么采样」。

---

## 1. 数据从哪来:专家开车录的(Imitation Learning)

每条 route 的每一帧,都记录了「专家当时的观测」+「专家怎么开」。模型的训练目标就是模仿专家轨迹。

每帧在磁盘上的模态(`data/carla_leaderboard2/data/<场景>/<route>/`):

| 类别 | 目录/字段 | 说明 |
| :-- | :-- | :-- |
| **输入·传感器** | `rgb`(3 相机)、`lidar`、`radar` | 喂给模型的原始观测 |
| **输入·导航条件** | `metas` 里 `target_point`、`command` | 目标点 + 转向指令(直行/左转/右转) |
| **标签·感知** | `bboxes`、`semantics`、`hdmap`、`depth` | 监督感知头 |
| **标签·规划** | `metas` 里 `future_positions / future_speeds / future_yaws` | 自车未来轨迹/速度/朝向 = 专家怎么开 |
| **数据增强** | `rgb_perturbated` / `lidar_perturbated` / … | 传感器扰动版(见 §5) |
| 其它 | `results.json` | 路线结果摘要(过滤无效 route 用) |

> 注:约 997 个 route 的 zip 本身不含 `results.json`,加载器会优雅跳过,实际可训练 route ≈ 8718;详见踩坑文档 §4b。

---

## 2. 模型与两阶段总览

整个就**一个网络**(TransFuser v6:图像+LiDAR 融合 backbone + 多个输出头):

```
原始数据(rgb/lidar/...) 
   └─ build_cache(预处理压缩, ~83G) + build_buckets(分组均衡采样)
        │
   [Pretrain]  backbone + 感知头      → pretrain/model_0030.pth(感知)
        │  load_file + use_planning_decoder=true
   [Posttrain] 上面整个 + 规划头(端到端) → posttrain/model_0030.pth(最终策略)
        │
   闭环评测(CARLA) → Bench2Drive/Longest6/Town13 分数
```

为什么分两阶段:感知任务**信号密集、好优化**;规划信号稀疏、难。先把 backbone 的场景表征练好,再在好表征上学规划,比从零端到端学规划稳得多。

---

## 3. Pretrain(感知预训练):只用感知标签

**开关**:`use_planning_decoder=False`(即 `is_pretraining=True`)

```
rgb + lidar (+radar)  →  backbone  →  感知头:
                                       ├─ 2D 检测(CenterNet) ← bboxes 标签
                                       │     heatmap/offset/wh/velocity/yaw
                                       ├─ BEV 鸟瞰语义分割      ← hdmap
                                       ├─ 透视图语义分割        ← semantics
                                       ├─ 深度估计              ← depth
                                       └─ 雷达检测              ← radar
```

- **只用同一帧的感知标签**做监督,教 backbone 看懂场景(谁是车/人、在哪、多远、什么朝向)。
- **不用规划标签**:`is_pretraining=True` 时 `future_positions/speeds/yaws` 等会被删掉、planner 不建。所以 pretrain 的 wandb **没有驾驶 loss**。
- **采样**:`full_pretrain` bucket —— 全量**均匀采样**。
- **产出**:`pretrain/model_0030.pth`(只含 backbone + 感知头,无 planner)。

对应你 wandb 看到的 loss:`loss_center_net_*`(检测)、`loss_bev_semantic`、`loss_semantic`、`loss_depth`、`radar_loss`。

---

## 4. Posttrain(规划后训练):加规划标签,端到端

**命令**:`load_file=pretrain/model_0030.pth  use_planning_decoder=true`

```
rgb + lidar (+radar)  +  target_point + command(导航条件)
   →  backbone(从 pretrain 权重接着训)  →  感知头(继续微调)
                                        +  planning decoder(新增规划头):
                                            ├─ waypoints   ← 自车 future_positions
                                            ├─ path        ← 空间路径
                                            └─ target_speed ← 自车 future_speeds
```

- **额外喂导航条件**(目标点 + 转向指令),让模型输出**未来轨迹 / 路径 / 目标速度**。
- 监督 = **专家的未来轨迹**(`metas` 的 `future_*`)→ 模型模仿专家怎么开;同时感知头继续训。
- **端到端**:backbone **不冻结**(`freeze_backbone=False`)。
- **采样**:`full_posttrain` bucket —— 比 pretrain 多一步:**过滤掉每段序列的首尾帧**(初始化抖动伪影),再均匀采样。
- **重要**(文档明确):**epoch 计数重置为 0,optimizer 重新初始化**(pretrain ckpt 里没有 planner 状态)。
- **产出**:`posttrain/model_0030.pth` = **最终能开车的策略权重**。

---

## 5. 关键增强:传感器扰动(`*_perturbated`)—— LEAD 的核心

数据里每帧都存了一份**扰动版**传感器:把自车假想摆到**偏离正常的位姿**(随机平移/旋转),传感器画面随之改变,但**标签仍是「回到正确轨迹」的轨迹**。训练时以概率 `use_sensor_perturbation_prob=0.5` 使用扰动版。

- **作用**:教模型**从偏差中恢复**。只看专家的完美轨迹,模型一旦开偏就不知所措;喂「扰动观测 + 纠偏标签」,它学会把车开回正轨。
- 这正是 LEAD 论文主题 **"Minimizing Learner-Expert Asymmetry(减少学生-专家不对称)"** 在数据侧的体现,也是 carla_garage 系的经典 shift augmentation。

---

## 6. 两阶段对比

| | Pretrain | Posttrain |
| :-- | :-- | :-- |
| 输入 | rgb + lidar + radar | + target_point + command |
| 监督标签 | 感知(检测/分割/深度/雷达) | 感知 + **规划(专家未来轨迹/速度)** |
| 关键开关 | `use_planning_decoder=False` | `use_planning_decoder=True` + `load_file` |
| 采样 bucket | `full_pretrain`(均匀) | `full_posttrain`(去序列首尾) |
| 数据本体 | **同一份 routes / 同一 cache** | 同左 |
| backbone | 从头训 | 接 pretrain 权重,端到端继续训(不冻结) |
| epoch / LR | 31 / cosine 退火 | 31 / cosine(epoch 重置为 0) |
| 产出 | 感知权重 | 最终策略权重 |
| 学到啥 | 看懂世界 | 在看懂基础上学开车 |

---

## 7. 怎么判断训得好 & 后续

- **Pretrain 成功**:各感知 loss 收敛到低且稳、LR 退火到 ~0、step 跑满。
- **Posttrain 成功**:在上面基础上,`waypoints/path/target_speed` 的 loss 也收敛。
- **但最终「开得好不好」只能靠 CARLA 闭环评测**(Driving Score),光看 loss 不够。
  ⚠️ 评测在当前 H20 节点跑不了(GPU 图形栈缺失),需换图形卡或让平台开 `NVIDIA_DRIVER_CAPABILITIES=all`,详见 `踩坑总结_LEAD_H20.md` §3c。

---

## 8. 关键命令

```bash
# 准备(一次性)
python3 scripts/build_buckets_pretrain.py
python3 scripts/build_buckets_posttrain.py
python3 scripts/build_cache.py

# Pretrain(4 卡)
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup bash scripts/pretrain_ddp.sh > pretrain.log 2>&1 &

# Posttrain(4 卡,接 pretrain 权重)
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup bash scripts/posttrain_ddp.sh > posttrain.log 2>&1 &
```
