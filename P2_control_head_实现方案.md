# P2 实现方案:反应式 Control 头(intent → 轨迹 + 避障)

> 目标:在 P1 的**模糊 intent 场**之上,接一个**学习式反应控制头**,把 intent 消解成具体 waypoints,并**受障碍约束(可微碰撞代价)**。
> 走**路线①(A)**:全程在 TFv6 内、复用 P1 的 backbone+intent 头,**第一版开环可训**(碰撞代价用录好数据里的占据),不依赖闭环 GPU。
> 上位方案 `方案_visual_intent_reactive_control.md`;P1 实现 `P1_visual_intent_实现细节.md`;对标 `EvaDrive_梳理与定位.md`。

---

## 0. 一句话

```
backbone ─► bev_feature_grid ─► IntentDecoder ─► pred_visual_intent(模糊场)
                                                      │(作为条件)
              bev_features + intent 场 ─► Control 头(PlanningDecoder 改造)─► waypoints
                                                      │  loss = 模仿 + 可微碰撞 + intent一致
```
P2 = 让 waypoints **在 intent 场的"活动空间"里、受障碍约束**地生成,而不是像 LEAD 那样直接从特征回归一条线。

---

## 1. 关键改造:让 Control 头「条件于 intent 场」

**Control 头 = 复用 `PlanningDecoder`**(DETR query transformer,现成出 waypoints/target_speed),**改动是把 intent 场喂进它的 context**。

### 1a. forward 顺序要调整(重要)
当前 `tfv6.py:forward` 里 **planning 在 line 125 就调用,而 intent 在 line 155 才算**。P2 要把 **intent 先算、再喂给 planning**:
```
① backbone → bev_features
② bev_feature_grid = top_down(bev_features)
③ pred_visual_intent = intent_decoder(bev_feature_grid)      # 提前到 planning 之前
④ planning_decoder(bev_features, ..., intent=pred_visual_intent)   # 条件化
```

### 1b. intent 场怎么进 context(改 `PlanningContextEncoder`)
`PlanningContextEncoder.forward` 现在把 `bev_features`(经 `dimension_adapter` Conv2d→token_dim)展平成 BEV token + 拼 status token(速度/命令/tp/radar)。**加一路 intent token**:
- 方案(轻):把 `pred_visual_intent`(B,1,320,384)下采样到 BEV token 网格,过一个 `Conv2d(1→token_dim)` + 位置编码,**flatten 成若干 token 追加进 context_tokens**(和 status token 同一套 concat 机制)。
- query 就能 cross-attend 到 intent 场 → waypoints 被意图"牵引"。

> 备选(更强):把 intent 场当**cost/attention 先验**,在 query 对 BEV token 做注意力时用 intent 加权。第一版先用"加 token"最省。

---

## 2. Loss:三项(决定它不是普通回归)

在 `PlanningDecoder.compute_loss` 里加:

### 2a. 模仿(现成)
`loss_spatio_temporal_waypoints` = L1(pred_waypoints, 专家 future_waypoints)。保留。

### 2b. 可微碰撞代价(新,核心)
让预测轨迹**别压到障碍**:
1. 从 `data["bev_semantic"]`(B,H,W 类别)取**障碍类**(车辆/行人等)→ 占据图 `occ`(B,1,H,W)∈{0,1}
2. 预测 waypoints(B,n,2,ego 米)→ BEV 归一化坐标 [-1,1]
3. `F.grid_sample(occ, wp_grid)` 采样每个 waypoint 处的占据值 → 落在障碍上就 >0
4. `loss_collision = mean(采样到的占据值)` —— **对 waypoints 可微**(grid_sample 对坐标可导),梯度把轨迹推离障碍

### 2c. intent 一致性(新,可选)
让轨迹落在 intent 高密度区:`loss_intent_consistency = -mean(grid_sample(sigmoid(intent), wp_grid))`(采样 intent 值,越高越好取负)。

---

## 3. 配置 & 开关

- 新开关 `use_control_conditioning`(默认 False):控制"是否把 intent 喂给 planning";开时才改 forward 顺序 + 加 intent token。
- 新权重:`collision_loss_weight`、`intent_consistency_loss_weight`,加进 `detailed_loss_weights`(仿 `loss_visual_intent` 那样,开关关时置 0)。
- P2 训练配置:
  ```
  use_intent_decoder=true
  use_planning_decoder=true          # planning 头 = control 头
  use_control_conditioning=true      # 让它条件于 intent
  load_file=outputs/local_training/intent_p1/model_0030.pth   # 接 P1 权重
  ```
  加载仍 `strict=False`:backbone+intent 头从 P1 加载,planning/control 头新初始化。

---

## 4. 代码改动点(预计)

| 文件 | 改动 |
| :-- | :-- |
| `lead/tfv6/tfv6.py` | forward 顺序:intent 提前;`planning_decoder(..., intent=pred_visual_intent)` |
| `lead/tfv6/planning_decoder.py` | `PlanningContextEncoder` 接收并编码 intent 场为 token;`compute_loss` 加碰撞 + intent 一致 |
| `lead/tfv6/collision_cost.py` | **新增**:`occupancy_from_bev_semantic()` + `differentiable_collision(waypoints, occ, config)`(grid_sample) |
| `lead/training/config_training.py` | `use_control_conditioning`、`collision_loss_weight`、`intent_consistency_loss_weight` + 权重表 |
| `scripts/train_control_ddp.sh` | **新增**:P2 的 4 卡启动脚本 |

均开关门控、默认关,不破坏现有训练。

---

## 5. 为什么第一版开环就行 + 诚实边界 ⚠️

- **开环可训**:碰撞代价用的是**录好数据每帧的 bev_semantic 占据**,不需要在线交互 → 绕开闭环 GPU 卡点。
- **但这是"软先验避障",不是真反应式安全**:专家几乎不碰撞、数据里负样本少,`loss_collision` 只是把轨迹**轻推离已知障碍**。**真正的反应式安全 / 分布外纠偏,仍需闭环 / 仿真造碰撞 / RL(= P3,需图形 GPU)**。
- **多模态仍未解**:P1 已发现 intent 会收敛到"专家单条路径的 soft 版"(单模)。P2 让 control 消解这个(单模)intent → 打通机制;**真多模是另一条线**(地图定义 intent support / latent),不在 P2 范围。P2 的价值 = **验证"intent 场 + 反应控制"这套两层结构能产出受约束的轨迹**。

---

## 6. P2 子阶段

| 子步 | 内容 | 闭环 |
| :-- | :-- | :-- |
| P2.1 | 改 forward 顺序 + intent token 进 planning,`use_control_conditioning`;先只加模仿 loss,验证条件化不掉点 | ❌ |
| P2.2 | 加**可微碰撞代价**(collision_cost.py + grid_sample),看轨迹是否绕障 | ❌ |
| P2.3 | 加 intent 一致性;调三项权重 | ❌ |
| P2.4(→P3) | 闭环/RL 精调真反应安全 | ✅ 需图形 GPU |

---

## 7. 验收 & 可视化

- **指标**:`waypoints_ade/fde`(模仿精度不塌)、`loss_collision`(下降)、闭环前用**开环碰撞率**(预测轨迹压障碍的比例)当代理指标。
- **可视化**(扩 `viz_intent_pred.py`):在 BEV 上同时叠 **intent 场(红)+ 预测 waypoints(点线)+ 障碍占据(另色)+ 专家 GT 轨迹**,看轨迹是否"在 intent 走廊内、绕开障碍、贴近专家"。

---

## 8. 待办 / 红线
- [ ] P2.1 接线(顺序 + intent token),先验证模仿不掉点
- [ ] P2.2 碰撞代价(`collision_cost.py`),确认障碍类 id、BEV 坐标↔像素约定(复用 P0/P1 那套 `pixels_per_meter` 映射)
- [ ] 扩可视化:轨迹 + 占据 + intent 同图
- **红线**:① 别把 intent 退化成硬 goal(保持 soft,control 才有"留白"可用);② 碰撞代价是软先验,别当成真安全宣称;③ 多模是独立线,P2 不解决,论文里要说清。
