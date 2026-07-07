# P1 实现细节:训练 Visual-Intent 头

> 目标:在**感知预训练好的 backbone** 之上,训练一个**单通道 soft BEV 意图头**,验证「用可解释的视觉方式表达 intent」这一步成立。
> 本文档描述**当前代码里 P1 到底怎么跑的**(数据→标签→前向→loss→配置→启动)。
> 上位方案见 `方案_visual_intent_reactive_control.md`;对标见 `EvaDrive_梳理与定位.md`。

---

## 0. 一句话

```
rgb+lidar ─► TFv6 backbone ─► bev_feature_grid (B,64,H,W)
                                   │
                          IntentDecoder ─► pred_visual_intent (B,1,320,384) logits
                                   │  BCEWithLogits(pos_weight)
                          专家 route ──栅格化──► soft 高斯 intent 场 (B,1,320,384) 标签
```
P1 = 只加这一个头 + 一个 loss(`loss_visual_intent`),其余复用 LEAD/TFv6。

---

## 1. 数据流(逐张量)

| 环节 | 张量 / 形状 | 来源 |
| :-- | :-- | :-- |
| 输入 | rgb(3 相机)+ lidar | 数据集 `data` |
| backbone 输出 | `bev_features`(token 形)、`image_features` | `TransfuserBackbone` |
| **BEV 特征图** | `bev_feature_grid` **(B, 64, h, w)** | `self.backbone.top_down(bev_features)`（`tfv6.py:153`）|
| **intent 预测** | `pred_visual_intent` **(B, 1, 320, 384)** logits | `IntentDecoder(bev_feature_grid)` |
| **intent 标签** | `visual_intent_label` **(B, 1, 320, 384)** ∈[0,1] | dataloader 里栅格化 route |

- `64` = `config.bev_features_chanels`;`320×384` = `lidar_height_pixel × lidar_width_pixel`。
- intent 头接的是 **`bev_feature_grid`**(和 `bev_semantic`/检测头同一个输入),不是 planning 用的 token。

---

## 2. Intent 头结构(`lead/tfv6/intent_decoder.py`)

克隆自 `BEVDecoder`,输出通道改成 1:

```python
IntentDecoder.net = Sequential(
    Conv2d(64, 64, 3x3, pad=1), ReLU,
    Conv2d(64, 1, 1x1),                       # 单通道 = intent 占据/热力
    Upsample(size=(320, 384), bilinear),      # 升到 BEV 全分辨率
)
forward(bev_feature_grid) -> (B, 1, 320, 384)  # logits
```

轻量、可解释、和感知头同构,好调。

---

## 3. Intent 标签怎么造(核心)

**监督 = 专家未来路径投影到 BEV,撒高斯做成 soft 走廊**,不是硬线 —— 这样保住"模糊性/隐式多模态"。

函数 `rasterize_waypoints_to_bev(waypoints, config, sigma_m=1.5)`:

1. 取专家 `data["route"]`（10 个点,ego 米制,x=纵向 / y=横向）
2. ego 米 → BEV 像素:`col=(x-min_x)*ppm`,`row=(y-min_y)*ppm`（`ppm=4`,`min_x=-32, min_y=-40`）
3. 每个路径点撒 2D 高斯（`sigma=1.5m×4=6px`），**沿路径取 max** → 连续走廊
4. 输出 `(1, 320, 384)` ∈ [0,1]

**性能关键**:这步现在在 **dataloader 的 CPU worker 里预算**（`carla_dataset.py` 里 route 之后,产出 `data["visual_intent_label"]`),**不在 GPU 上每步现算**（否则 B=64 时会建 ~2GB 临时张量压在 GPU 关键路径上)。`IntentDecoder._intent_label` 优先用这个预算好的标签,没有才回退现算。

> BEV 朝向:`forward=+x` 映射到宽度轴(图上向右),和 `bev_semantic` 同帧,P0 已叠图确认对齐。人看着"朝右开"是正常的,不影响训练。

---

## 4. Loss(`IntentDecoder.compute_loss`)

```python
loss["loss_visual_intent"] = BCEWithLogitsLoss(pred_logits, soft_target, pos_weight)
```

- **BCEWithLogits**:把 intent 当每像素的 soft 占据回归,pred 与高斯标签逐像素比。
- **pos_weight**:intent 走廊只占 ~1.2% 像素,极稀疏 → 用 `pos_weight=(1-pos)/pos`（封顶 50）平衡正负,避免全预测 0。
- 日志:`visual_intent/pred_max`、`pred_mean`、`target_coverage`。

**总 loss 里的权重**:`config_training.py:detailed_loss_weights` 里
```python
weights["loss_visual_intent"] = visual_intent_loss_weight if use_intent_decoder else 0.0
```
默认权重 `1.0`,和其它任务一起在 `train.py` 归一化后加权求和。开关关时为 0,不影响原训练。

---

## 5. 训练配置(P1 当前用的）

由 `scripts/train_intent_ddp.sh` 设置:
```
use_intent_decoder=true
use_planning_decoder=false
load_file=outputs/local_training/pretrain/model_0030.pth
logdir=outputs/local_training/intent_p1
```

要点:
- **加载**:`strict=config.continue_failed_training`,P1 时为 `False` → **`strict=False`**:backbone + 感知头从 pretrain 加载,**intent 头随机初始化**（和 posttrain 加载 planning 头同机制）。
- **`use_planning_decoder=false` ⇒ `is_pretraining=True`**:
  - 用 **pretrain bucket**（`FullPretrainBucketCollection`）
  - 规划 loss 权重=0(不训 waypoints/target_speed）
  - **感知 loss 仍激活**（semantic/depth/bev_semantic/检测/radar 权重=1）→ 所以 P1 实际是**「感知 + intent」多任务联合微调**,backbone **不冻结**。
  - （想 intent-only 隔离,可 `freeze_backbone=true` 或把感知权重调 0;当前选的是联合。）
- **epoch=31**（`carla_leaderboard_mode` 默认），全局 batch 64,cosine 退火。
- 4 卡:每卡 batch 16、worker 24,~1h/epoch。

---

## 6. 代码改动点(git diff 可查,均开关门控、默认关)

| 文件 | 改动 |
| :-- | :-- |
| `lead/tfv6/intent_decoder.py` | **新增**:`IntentDecoder` + `rasterize_waypoints_to_bev` + `compute_loss` |
| `lead/tfv6/tfv6.py` | `__init__` 实例化;`forward` 出 `pred_visual_intent` 并进 `Prediction`;`compute_loss` 加 intent loss |
| `lead/training/config_training.py` | 开关 `use_intent_decoder`、`visual_intent_loss_weight`;权重表加 `loss_visual_intent` |
| `lead/data_loader/carla_dataset.py` | route 门控加 `use_intent_decoder`;dataloader 侧预算 `visual_intent_label` |
| `scripts/train_intent_ddp.sh` | **新增**:P1 的 4 卡 DDP 启动脚本 |
| `scripts/viz_visual_intent.py` | **新增**:P0 标签可视化 |

---

## 7. 启动 & 监控

```bash
cd $LEAD_PROJECT_ROOT
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup bash scripts/train_intent_ddp.sh > intent_p1_4gpu.log 2>&1 &

tail -f intent_p1_4gpu.log
tr '\r' '\n' < intent_p1_4gpu.log | grep -aoE "[0-9]+/[0-9]+ \[[^]]*\]" | tail -3
```
wandb 看 `unscaled_loss/loss_visual_intent`（应下降趋平)、`visual_intent/pred_*`。
产出:`outputs/local_training/intent_p1/model_00XX.pth`。

**已验证(单卡 1 epoch)**:`loss_visual_intent` 0.84→0.14,`pred_max 0.999 / pred_mean 0.041` → intent 头学出稀疏、聚焦的意图场,链路端到端通。

---

## 8. P1 验证了什么 / 没验证什么

- ✅ **验证**:能否用「backbone 特征 → 单通道 soft BEV 场」表达 intent,并从专家 route 监督学出来。
- ❌ **还没做**:control 头(把 intent 消解成轨迹/动作 + 避障)= **P2**;闭环安全/多目标 = P3(需图形 GPU,本 H20 不行)。

## 9. 待办
- [ ] **预测可视化**（P1 真正验收):`scripts/viz_intent_pred.py` —— 加载 `intent_p1` 权重,前向出 `pred_visual_intent`,sigmoid 后和 GT intent 并排叠 BEV,看学得像不像(会不会糊/偏/塌)。
- [ ] 决定是否 intent-only 隔离（`freeze_backbone`)重训一版做对照。
- [ ] P2:接 control 头(条件于 intent 场 + 可微碰撞代价)。
