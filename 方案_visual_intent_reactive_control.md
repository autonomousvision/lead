# 方案:隐式 Visual Intent + 反应式控制(路线①,在 TFv6 内实现)

> 目标:验证「先出模糊 intent、再由反应控制消解」这个第三种规划器范式。
> 原则:**复用 LEAD/TFv6 的 backbone 与感知 pretrain,只改规划头;第一版开环可训,不依赖闭环 GPU。**
> 背景/对标见 `EvaDrive_梳理与定位.md`;基座训练见 `LEAD_训练原理_pretrain_posttrain.md`。

---

## 1. 核心思路(一句话)

把 TFv6 现在的「特征 → 一次性回归一条 waypoints」改成两段:

```
backbone 特征 ──► Intent 头  ──► 模糊 intent 场(BEV soft heatmap)
                                   │
                 Control 头(条件于 intent 场 + BEV 障碍)──► waypoints/动作
```

- **Intent 头**:出"往哪走"的**分布式意图场**(不 commit 一条线)→ 保住模糊性/隐式多模态。
- **Control 头**:在 intent 场的"走廊"里,受障碍约束,**晚期消解**成具体轨迹。
- **多目标沿层次拆**:目标导向→intent 层;无碰撞/舒适→control 层。**不枚举候选、不 Pareto**(正面区别于 EvaDrive)。

---

## 2. 接在 TFv6 的哪里(已核对代码)

`tfv6.py:forward`:
```python
bev_features, image_features = self.backbone(data)          # token 形 BEV 特征
bev_feature_grid = self.backbone.top_down(bev_features)      # 2D BEV 特征图 (B,64,H,W)
self.bev_semantic_decoder(bev_feature_grid)                 # ← Intent 头接这里(同款输入)
self.planning_decoder(bev_features, ...)                    # ← Control 头由它改造
```
可复用张量:**`bev_feature_grid`(B,64,H,W)** 给 intent 头;**`bev_features`(BEV grid)** 给 control 头;**`bev_semantic`/`bboxes`** 做碰撞代价。

---

## 3. Intent 头 = 克隆 BEVDecoder,输出 1 通道 soft 场

- **模块**:仿 `lead/tfv6/bev_decoder.py`(conv→conv→Upsample 到 `lidar_height_pixel×lidar_width_pixel`),**输出通道 = 1**(intent 占据热力)。
- **输入**:`bev_feature_grid`(B,64,H,W)。
- **输出**:`(B,1,320,384)` intent heatmap logits。
- **标签**:把专家 `data["route"]`/`future_waypoints`(ego 米制)栅格化到同一 BEV 网格,沿路径撒**高斯** → soft target ∈[0,1]。
- **loss**:heatmap 损失(BCEWithLogits + pos_weight,或 focal / MSE)。分布形态 = 模糊性存活。
- 已实现于 `lead/tfv6/intent_decoder.py`(见 §6)。

BEV 网格约定(来自 config,**待 P0 可视化确认朝向**):`pixels_per_meter=4`,x(纵向,-32..64m)→ 宽 384,y(横向,-40..40m)→ 高 320。

---

## 4. Control 头 = 复用 PlanningDecoder,条件化在 intent 场上

- **模块**:复用 `PlanningDecoder`(DETR query transformer)。
- **改动**:`PlanningContextEncoder` 现在把 BEV grid 展平成 token + 拼 status token(速度/命令/target_point/radar)。**再拼上 intent 场 token**——把 intent heatmap 下采样/编码成若干 token 追加进 context(和 status token 同一套机制)。
- **输出**:仍是 waypoints / target_speed(下游仍接 PID)。
- **loss(决定它不是普通回归)**:
  1. 模仿:L1 到专家 waypoints(现成 `loss_spatio_temporal_waypoints`)
  2. **可微碰撞代价**:预测 waypoints → BEV 网格 → 与 `bev_semantic` 占据/`bboxes` 重叠处罚
  3. (可选)intent 一致性:waypoints 落在 intent 场高密度区

---

## 5. 分阶段(复用你已训好的成果)

| 阶段 | 内容 | 闭环 GPU |
| :-- | :-- | :-- |
| **P0** | 专家轨迹→BEV 高斯场 + 叠 `bev_semantic` 可视化,确认 intent 标签/网格朝向 | ❌ 现在就能做 |
| **P1** | 加载 LEAD backbone(冻结/微调),训 **Intent 头**(高斯场模仿),可视化预测 | ❌ 开环 |
| **P2** | 加 **Control 头**(条件于 intent),训 模仿 + 可微碰撞 + intent 一致 | ❌ 仍开环 |
| **P3** | 闭环/RL 精调反应式安全;长程 + 多目标内化;对打 EvaDrive/LEAD baseline | ✅ 需图形 GPU |

---

## 6. 已搭好的代码(scaffold)

- **`lead/tfv6/intent_decoder.py`**:
  - `rasterize_waypoints_to_bev(waypoints, config, sigma_m)` → `(B,1,H,W)` 高斯 intent 标签(P0/P1 共用)
  - `IntentDecoder(nn.Module)`:`bev_feature_grid → (B,1,H,W)` logits
  - `IntentDecoder.compute_loss(pred, data, loss, log)`:栅格化标签 + heatmap loss
  - `__main__` 冒烟测试:随机张量跑通 forward/loss(无需数据)

**未接线**(刻意留给你 review 后再动核心文件):
- `config_training.py` 加开关 `use_intent_decoder`
- `tfv6.py:__init__/forward` 实例化并调用 IntentDecoder、把 intent 场传给 planning_decoder
- `PlanningContextEncoder` 接收 intent 场 token
- P2 的可微碰撞代价函数

---

## 7. 两个必守的红线
1. **intent 必须 soft(分布/走廊),不能塌成一条线**——否则退回 LEAD 单模回归,novelty 归零。
2. **P2/P3 的避障**:专家几乎不碰撞,真反应式安全需闭环/仿真造碰撞/RL → 必须解决能跑 CARLA 闭环的图形 GPU(本 H20 不行,见 `踩坑总结_LEAD_H20.md` §3c)。

## 8. 下一步
- [ ] P0:写 `scripts/viz_visual_intent.py`(用 CARLAData 取样 → 栅格化 intent → 叠 bev_semantic → 存图),确认网格朝向
- [ ] P1:接线 `use_intent_decoder` + tfv6,训 intent 头
- [ ] 并行:deep-research 核 visual-intent / NMP / VLM-driving 防撞车
