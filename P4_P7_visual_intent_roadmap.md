# P4–P7 路线图:Visual Intent 从 VLM 化到生产

> **核心问题**:人类驾驶不显式枚举轨迹候选(contra EvaDrive 的 64 条 Pareto),而是先有个**模糊的"往哪走"意图**,再由反应式控制在意图走廊里消解成轨迹。这是第三种规划器范式(vs LEAD 单模回归 / EvaDrive 显式多模)。
>
> **原则**:每阶段独立可发(有闭环 Driving Score 对比);前一步成果是后一步地基,不走回头路。

---

## 背景:P1–P3(已完成/进行中,不在本文档重点)

| 阶段 | 内容 | 状态 |
| :-- | :-- | :-- |
| P1 | intent 头(专家 route→BEV 高斯),追平 LEAD | ✅ 开环 ADE 0.185 vs 0.191 |
| P2 | control 头条件化 intent + 可微碰撞代价 | ✅ 开环不掉点 |
| P3 | 闭环 Driving Score(LEAD vs P2,Bench2Drive 220) | 🔄 A800 跑中 |

**P1/P2 用"专家 route 投影"当 intent,只为验证两层机制(intent→control)端到端可训、不掉点。** 结论:机制通,但 intent = 单条专家 route → 单模、无世界知识、短视野。这些正是 P4+ 要解决的。

---

## 总览:P4–P7

| 阶段 | intent 生产者 | 关键升级 | 解决的病 | 闭环可比 |
| :-- | :-- | :-- | :-- | :-- |
| **P4** | **Qwen-VL(替代 TFv6)** | 世界知识 + 命令条件 | 无新信息(寄生同源特征)、无语义 | ✅ |
| **P5** | VLM + 地图 support | 真多模态(点亮所有可行臂) | 单模塌缩 | ✅ |
| **P6** | VLM 多相机融合 | 全视野(补盲区) | 前视盲区 | ✅ |
| **P7** | VGGT 几何 + VLM | 长线(64m+)、camera-only | 视野短(~10m)、依赖 LiDAR | ✅ |

---

## P4:Qwen-VL 替代 TFv6 当 intent 生产者(当前)

**一句话**:把 intent 从"TFv6 backbone 特征 → IntentDecoder"换成"原图 + 命令 → 冻结 Qwen-VL → 轻解码头 → 图像→BEV 投影",**control 头及一切下游不动**。

**为什么**(诊断③):P1/P2 的 intent 从和 control **同一份 TFv6 特征**预测,是同源信息的有损再编码 → 结构上不可能超过 LEAD,只能追平。想超过,唯一出路是注入**独立信息源** = VLM 的世界知识/语义。

**数据流**:
```
原图前视(384²) + 导航命令 ──► [Qwen-VL 冻结,命令在图像前] ──► 图像token LLM hidden states (12,12,2560)
                                                                    │
                                              [轻解码头 可训] ──► 图像空间意图 H_img
                                              [可微投影,GT depth+标定] ──► BEV intent 场 (B,1,320,384)
                                                                    │ 契约不变
                                              [P2 的 PlanningDecoder 加载] ──► waypoints ──► PID
```

**关键设计**:
- **命令放图像之前**:Qwen 是因果 LM,图像 token 只 attend 前文 → 命令必须在图像前,hidden states 才被命令浸染(A2 命门)。
- **冻结 VLM + 离线缓存**:每帧一次 prefill 前向,缓存图像 token hidden states(~72GB / 9.7万帧)。训练只训轻解码头,H20 可跑。
- **蒸馏目标**:仍是专家 route 的 BEV 高斯场(P1 的 `visual_intent_label`)。VLM 靠命令区分左/右/直,学出的意图带世界知识。

**两 env 分工**:
1. `qwenvl`:抽特征(冻结 Qwen3-VL-4B-Instruct),缓存 `<scenario>/<route>/<frame>.npy`。
2. `lead`:训练(轻解码头 + 投影 + 蒸馏)。

**当前进度**:
- ✅ Step 1:manifest(9.7万帧 + 命令,stride 10)
- 🔄 Step 2:8 卡抽 VLM 特征(~72GB)
- ⏭ Step 3:训练脚本(轻解码头 + 可微投影 + 蒸馏)

**go/no-go**:
- **P4a 蒸馏**:BEV intent 复现专家高斯场,可视化像样,loss 收敛。
- **P4b 接 control**:加载 P2 planning decoder(先冻结做干净 ablation),开环 ADE/FDE ≈ P2。
- **P4 闭环**:≳ LEAD/P2,尤其"需要世界知识/语义"的场景(路口、红灯、异常物)。
  - ≳ → VLM 世界知识有价值(证死诊断③)→ 进 P5。
  - ≈ → 查:图像→BEV 投影精度?命令条件失效?VLM 特征质量?

**P4 明确不解决**(留后):① 多模(标签仍单条 route→单模,留 P5);② 长程(GT depth 近距,~10m,留 P7);③ 盲区(单前视,留 P6)。

---

## P5:真多模态 intent(地图 support)

**问题**:P4 蒸馏专家单条 route → 即使有 VLM,intent 仍收敛成"命令选中的单臂"。真多模态(路口所有可行臂**同时**点亮、晚期消解)没出来 —— 这是相对 EvaDrive 的命根,也是 §9 一直没做的点。

**方案**:
- **监督换源**:不再只用专家单条 route。从 CARLA lane graph 取当前 horizon 内**所有可行驶后继车道臂**,栅格化成**多模 support 场**(multi-hot / K 通道)。VLM 学"哪些地方可以去",不是"专家去了哪"。
- **意图表征多模**:BEV intent 从 (B,1,H,W) → (B,K,H,W)。
- **control 条件化改 K 通道**:`intent_adapter` Conv2d(1→D) 改 (K→D)(**唯一动 control 的一处**),让 DETR decoder 看到整张多模场,由它 + 碰撞代价晚期挑/融合。
- **loss**:permutation-invariant(每个预测模式匹配一个 GT 臂,匈牙利式),防止塌回单模。

**go/no-go**:路口多臂点亮且不糊;开环不掉点;闭环命令切换/路口泛化 ≳ P4。
- ≳ P4 → 真多模成立 → 进 P6。
- 塌单模 → 查 loss(是否 permutation-invariant)/ support 标签质量。

---

## P6:多相机融合 intent

**问题**:P5 只用前视 → 盲区大(左/右转时侧方看不全)。LEAD 本是 3 相机(前左/前/前右 ±54.5°)。

**方案**:
- 3 路相机各自过 VLM,得 3 组图像 hidden states。
- 各按自己外参投影到同一 BEV 网格,**BEV 空间融合**(加权 / attention / learn-to-fuse),侧方补前视盲区。
- 蒸馏目标不变(专家 route 全视野)。

**go/no-go**:侧方/盲区场景(左转右侧来车、变道后视盲区)比 P5 安全;碰撞/off-road 子指标提升。
**复杂度**:缓存 ×3(~216GB)或侧方实时前向;投影融合要调(标定误差、重叠区 double-count)。

---

## P7:长线规划 + camera-only 探索

**问题**:P4–P6 的 intent 视野受专家 route ~10m 限制(诊断②),只能近距反应,不能提前规划"50m 外路口往左"。而**远处信息一直在相机像素里,是 LiDAR/BEV(64m 硬墙)把它扔了**。

**方案 A:VGGT 几何支撑**
- VGGT 从多帧图像恢复稠密长程几何(depth/occupancy),投影到 64m BEV → 给 VLM 远距几何支撑(看到 50m 外路口)。
- VLM 在 VGGT 几何上解码 64m intent 走廊(不再受 10m 限制);蒸馏目标换扩展 route(加距离上限)或地图可行驶区。
- **远粗近细**:远处给方向性模糊意图(正合"模糊意图"叙事),近处精确。

**方案 B(可选):camera-only**
- P4–P6 全程保留 LiDAR。P7 尝试 VGGT 几何 + VLM 替代 LiDAR 分支 → 纯相机。
- **先验 VGGT 在 CARLA 渲染图的度量精度**(它在真实图上训的,可能 OOD)。差就退回"VGGT 增强 + LiDAR 保留"。

**go/no-go**:长线场景(Longest6/Town13)明显优于 P6(提前 5–10s 准备路口);camera-only ≈ LEAD 则可去 LiDAR。

---

## 决策树(每阶段闭环 go/no-go)

```
P4 闭环 ≳ LEAD ── 是 ─► 进 P5(多模)
              └─ 否 ─► 查:投影精度/命令条件/VLM特征 → 修完再闭环
P5 闭环 ≳ P4(路口泛化) ── 是 ─► 进 P6(多相机)
                        └─ 否 ─► 查:support标签/permutation loss/塌单模
P6 盲区场景 ≳ P5 ── 是 ─► 进 P7(长线)
                └─ 否 ─► 调融合;无增益则 P7 只用前视
P7 长线场景 ≳ P6 ── 是 ─► camera-only≈LEAD? 可去LiDAR : 保留LiDAR+VGGT增强
                └─ 否 ─► 回退 P6 为终态,或换几何方法
```

---

## 并行深挖(不阻塞主线)

1. **NMP 文献**(Waymo planning Transformer / nuPlan PDM):隐式多模规划怎么评。
2. **VLM-driving 防撞**(GPT-Driver/DriveVLM):prompt 命令条件 + 安全约束避免幻觉撞车。
3. **VGGT/几何重建**在 CARLA 的度量误差(决定 P7 camera-only 上界)。
4. **地图 support**:OpenDRIVE/lane graph → BEV multi-hot;地图缺失时的 vision-only support。

---

## 交付物(P7 完成后)

1. 五版模型闭环 DS 对比表(LEAD / P4 / P5 / P6 / P7,Bench2Drive + Longest6)。
2. 开源代码(P4–P7 训练/推理 + VLM 抽特征工具)。
3. 论文:*Visual Intent for Autonomous Driving: A Third Paradigm Beyond Explicit Multi-modality* —— 核心 claim:VLM hidden states(命令条件)+ 轻解码头实现"模糊意图 + 反应控制",隐式保有多模、晚期消解,比 EvaDrive 显式枚举更高效更像人。
4. 可选:P7 camera-only 成立则单独发纯视觉端到端方案。

---

## 为什么这么拆

- **P4 = 核心突破**:VLM 世界知识注入 = 唯一能真正超过 LEAD 的信息增量(P1/P2 追平但超不过,根因在此)。
- **P5 = novelty 本体**:地图 support → 真多模,这是相对 EvaDrive 的本质差异(隐式 vs 显式)。
- **P6 = 工程完备**:多相机补盲区(生产必需,对 novelty 贡献小,故排 P5 后)。
- **P7 = 上界验证**:VGGT 长线 + camera-only,证明 visual intent 能走多远。

每步独立闭环可比,前一步成果是后一步输入(P4 VLM intent 头 → P5 加 support → P6 扩相机 → P7 换几何),不走回头路。

---

**当前行动**:P4 Step 2(8 卡抽 VLM 特征)跑中 → 完成后写 Step 3 训练脚本 → P4 闭环对比 LEAD。

