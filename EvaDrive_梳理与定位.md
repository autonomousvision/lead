# EvaDrive 梳理 & 对我们 visual-intent 方向的定位

> 论文:**EvaDrive: Evolutionary Adversarial Policy Optimization for End-to-End Autonomous Driving**
> arXiv:2508.09158(2025/08)· [abs](https://arxiv.org/abs/2508.09158) · [HTML v2](https://arxiv.org/html/2508.09158v2)
> 注:用户提到的「APO」= 本文方法 **Adversarial Policy Optimization**,与 EvaDrive 是同一篇,不存在独立的 APO 论文。

---

## 0. 一句话

把端到端轨迹规划做成一个 **「生成器 ↔ 多目标评判器」的对抗 + 进化闭环**:生成器一轮轮提候选轨迹,多目标 critic 用 **Pareto 前沿**(而非单一标量奖励)来评和选,反复精修——对应"人开车时不断想象、评测、优化多个备选方案"。

我们引用的那句人类驾驶 motivation 就出自本文开篇。

---

## 1. 想解决的两个痛点

1. **生成与评测脱节**:主流"先生成一堆轨迹、再打分选"的框架,生成和评估割裂,无法迭代精修。
2. **多目标被标量化**:RL 把安全/舒适/进度等多维偏好塌成一个标量奖励,丢掉关键 trade-off(scalarization bias)。

---

## 2. 架构与输入

- **感知**:ResNet34 backbone,**3 相机(左/前/右)**,输入 2048×512 → 冻结视觉特征 `F_img` + ego 状态 `s_ego`。
- （和 LEAD 几乎同一套底子:ResNet34 / 3 相机 / NAVSIM+Bench2Drive 生态)
- **整体流程**:Actor 迭代出 K 轮候选 → 多目标 Critic 评 → Pareto 选 → 引导下一轮 → K 轮后定最终轨迹。

---

## 3. 核心一:层次化生成器(两阶段)

**Stage I — 自回归意图建模(管时间因果)**
- 初始 anchors `A₀ ∈ ℝ^(64×P×3)`(64 候选,每个 P 个位姿 (x,y,θ))
- 交叉注意力 `MHCA_T`:Q=anchors,K=V=concat(anchors, 历史轨迹特征 `H_hist`, `F_img`)
- **矩形因果 mask**:每个时刻只看当前+过去 → 时序连贯意图特征 `A_AR`

**Stage II — 单步 diffusion 精修(管空间灵活)**
- DDIM 加噪 `Ã = α·A_AR + σ·ε`,**单步去噪**(强调不做多步,省推理)
- 空间交叉注意力 `MHCA_S(Ã, F_img)` → 解码最终 64 条轨迹 `T_pred ∈ ℝ^(64×P×3)`
- 消融:两阶段**顺序反过来掉点**(85.9 vs 88.1 PDMS)→ 先时序后空间是关键

> 顺带:EvaDrive 是**自回归 + diffusion 都用**(不是二选一)。

---

## 4. 核心二:对抗 / 进化循环(Algorithm 1)

**K=2 轮**(消融最优;≥3 轮反而掉点),每轮:
1. 生成 **N=64** 候选
2. 每条算**多目标奖励向量** `r(a)`
3. **快速非支配排序**取 Pareto 前沿 `P_t`(~8–12 非支配解)
4. 从前沿**均匀采 M=8** 条引导轨迹
5. 用它们**条件化下一轮生成**

支配定义:`a_j ≻ a_i` ⟺ 所有指标 ≥、至少一个 >。Pareto 选择保多样性、防 mode collapse。**Gumbel-Softmax + 可微非支配排序**保证全程可微。

---

## 5. 核心三:多目标 Critic(不标量化)

- **向量奖励** `R = [r₁,...,r_K]`;NAVSIM 5 维:**NC(无碰撞)/ DAC(可行驶区)/ EP(进度)/ TTC(碰撞时间)/ C(舒适)**
- critic:对每条轨迹 temporal max-pool → MLP 出 N 个标量分量,**评测阶段不聚合**(留给 Pareto 探索)
- **权重只在「优化」时用**,评测不用:`V_w = Σ w_i·V^(i)`,Σw=1
  - 权重 = 驾驶风格:保守 (0.4:0.2:0.1:0.2:0.1)→93.5;激进 (0.1:0.2:0.4:0.1:0.2)→94.9
  - ⚠️ **让模型自学权重会崩**(87.2)——钻 loss 地形空子导致 reward collapse

---

## 6. 训练(交替对抗)

- **Actor loss** = 模仿(对专家 BCE)+ 多样性(最小化轨迹对互信息,防塌缩)+ 对抗偏好(最大化加权多目标奖励 + KL 锚参考策略)
- **Critic loss** = 奖励监督(对仿真真值 BCE)+ 对抗对齐(区分专家 vs 生成)
- 博弈:`min_φ max_θ V_w(φ,θ)`
- 协议:generator/critic **各 5 epoch 交替,共 30 epoch,4×H20**,Adam lr 7.5e-5,bs 8/卡

---

## 7. 结果

- **NAVSIM v1:94.9 PDMS**(超 DiffusionDrive +6.8、DriveSuprim +5.0、TrajHF +0.9)
- **Bench2Drive:64.96 Driving Score**
- 关键消融:两阶段顺序、K=2 最优、闭环多轮 +2.0 PDMS、自学权重失败(87.2)

---

## 8. 对我们 visual-intent 路线的定位(重点)

EvaDrive 是我们 claim 的**完美靶子**——它把"人会想象多个备选"隐喻**字面化**到极致(显式 64 候选 × 2 轮 + Pareto 打分选)。我们的反命题:**人不物化多个备选,而是基于一个模糊的、几何接地的 visual intent 行动,多模态从不被显式枚举,在与环境交互中被隐式消解**。

**要立住,必须扛住的差异化(逐条对标):**

| 维度 | EvaDrive(显式) | 我们(隐式 visual intent) | 风险/必答 |
| :-- | :-- | :-- | :-- |
| 意图 | 自回归 latent,喂给候选生成 | 建在 **VGGT 3D 重建**上的**空间/场式、本质模糊**的意图 | 必须和它的"autoregressive intent"划清界限 |
| 多模态 | 64 候选 × 2 轮显式枚举 | **单次前向出一个模糊意图场**,不采多条 | 卖点=single-pass、便宜、可长程 |
| 多目标 | **Pareto over 候选(最强项)** | 无候选 → trade-off 需**内化进表征/训练目标** | ⚠️ **最致命的必答题** |
| 效率 | 多轮 + 64 候选,推理重 | 单次,轻,**适配长程实时闭环** | 我们最硬的可测优势 |
| 战场 | NAVSIM 94.9(强),**Bench2Drive 闭环仅 64.96** | 主打**长程闭环** | 闭环空间大 = 我们的机会窗口 |

**我们能立住的核心 contribution(不靠认知论)**:
1. **效率/可扩展**:显式 imagine→evaluate→refine 昂贵、难扩长程闭环;隐式 visual intent 单次产出。
2. **几何接地**:VGGT 把意图落在度量 3D 空间(前向=意图走廊,后向=历史上下文)。
3. **模糊性机制**(防回归塌缩,必须有):latent / 分布 / energy / occupancy-flow 之一,让多模态隐式存活。
4. **多目标内化**:回答"没有候选怎么权衡"——这是相对 EvaDrive 必须打的点。

---

## 8.5 关键澄清:「两阶段」「intent」不是一回事(三种规划器范式)

一个常见混淆:LEAD 的「两阶段」和 EvaDrive 的「intent」**不在同一个轴上**,不能直接比。

| | 是什么轴 | 讲的是 |
| :-- | :-- | :-- |
| **LEAD 两阶段** | **训练课程轴** | 先训感知、再训规划——**什么时候训什么** |
| **EvaDrive intent** | **规划器架构轴** | 规划器内部**怎么产出轨迹** |

### LEAD/TFv6 的「两阶段」= 训练课程(非推理两阶段)
继承 carla_garage,目的是**让优化更稳**:
- **Pretrain**:`use_planning_decoder=False`,只训 backbone + 感知头(检测/分割/深度/雷达),先"看懂场景"。
- **Posttrain**:加载 pretrain 权重,`use_planning_decoder=True`,接规划头端到端训(backbone 不冻结)。
- 规划头本身很简单:**DETR 风格 query-transformer,一次前向直接回归 waypoints**(cumsum 增量)+ 分类 target_speed;推理时 **waypoints → PID 控制器**。
- 即 LEAD 的"intent"很弱——只有 `target_point + command` 作**输入条件**,规划器不建模显式 intent,**规划 = 一次性回归一条线**。

### EvaDrive 的「intent」= 规划器内部模块
- autoregressive intent(时序 latent)→ diffusion 精修出 64 候选 → 多目标 Pareto critic 评+选 → 多轮。
- 它的 intent 是**喂给候选生成的时序种子**,**规划 = 多轮生成 + 打分选**。

### 在「规划器怎么出轨迹」这个正确的轴上三方对比

| | 规划器范式 | intent 是什么 | 多模态 | 控制 |
| :-- | :-- | :-- | :-- | :-- |
| **LEAD/TFv6** | **一次性回归**一条线 | 弱:target_point+command 作输入条件 | 无(单一确定轨迹) | 固定 PID 跟踪 |
| **EvaDrive** | **多轮生成 + Pareto 选** | 时序 latent,喂给候选生成 | **显式**:64 候选 × 2 轮 | 轨迹交给下游 |
| **我们(visual-intent)** | **单次出模糊意图场** | 强:VGGT 几何接地的**分布/场式**意图 | **隐式**:不物化候选 | **学习式反应控制**晚期消解 + 避障 |

> ⚠️「intent」一词三义,别混:LEAD=导航条件(往哪);EvaDrive=生成候选的时序种子;我们=模糊留白的空间意图,专给反应控制留余量。

### 对我们的意义
- 不是在"LEAD 两阶段 vs EvaDrive intent"里二选一——我们在**规划器范式**这个轴上开**第三条路**:既不像 LEAD 一次性 commit 一条线,也不像 EvaDrive 显式枚举打分,而是**单次出模糊意图场 + 反应控制晚期消解**。
- **LEAD 的两阶段(感知课程)与我们的规划器创新正交,可直接沿用**:训练课程仍是"感知 pretrain → intent/control posttrain",我们只是把 posttrain 那个"一次性回归 planner"**换成"模糊 intent 场 + 学习式反应控制"**。
- 一句话:**LEAD 给训练框架 + 感知底座;EvaDrive 是规划器范式上要对打的靶子;我们是第三种规划器范式。**

### Planning / Control 两层 = 隐式 intent 的「另一半」
- **上层 = 隐式 visual intent(planning)**:全局、粗、模糊,不承诺具体轨迹。
- **下层 = 学习式反应控制(control)**:局部、细、实时,在 intent 的"活动空间"内贴几何避障、保动力学可行,把模糊 intent **晚期消解**成动作。
- **核心 thesis**:intent 的**模糊性不是缺陷,而是给控制层留的余量**。committed 轨迹把控制锁死(只能 track);模糊 intent 场欠约束到刚好,让反应控制在其 support 内自由避障 + 保持目标导向。→ **explicit trajectory + PID 过约束;implicit intent + reactive control 用"留白"换安全**。
- **多目标沿层次拆解**(不靠枚举/Pareto,正面绕开 EvaDrive):progress/目标导向 → intent 层;无碰撞/TTC/舒适/可行驶区 → control 层。
- **诚实提醒**:planning+control 两层是 AD 标准做法(LEAD 已是 waypoints+PID)。novelty 必须落在:①上层从"承诺 waypoints"变"模糊场";②下层从"固定 PID"变"学习式带安全目标的反应控制";③两层后训练耦合。否则会被"this is just waypoints + controller"毙掉。
- **要划清边界的近邻工作**:Neural Motion Planner(出 cost volume 但仍显式采样打分)、Safety filter / CBF、residual policy、分层/goal-conditioned RL、MPC-with-learned-cost、occupancy-flow 规划。我们的差异 = **隐式模糊意图场(不枚举)+ 学习式反应控制(晚期消解、内化安全)+ 端到端后训练耦合**。
- **现实约束**:control 层要学"避障"但专家数据几乎不碰撞(负样本少),大概率需**闭环交互 / 仿真造碰撞 / RL** → 又回到"必须有能跑 CARLA 闭环的图形 GPU"。

---

## 9. 待办 / 下一步

- [ ] 查 **visual intent / intention field / implicit multimodality / late-commitment** 已有工作(防撞车)
- [ ] 查 **VGGT 在驾驶里被用过没**、闭环实时性
- [ ] 定两个设定:**CARLA 特权 vs vision-only/真实迁移**;**"前后取点" = 确定 reference vs 意图分布采样**
- [ ] 解决**能跑 CARLA 闭环的图形 GPU**(本 H20 节点跑不了,见 `踩坑总结_LEAD_H20.md` §3c)——闭环是我们卖点的验证场
- [ ] 据以上画方法草图:VGGT → 模糊意图场 → 长程层次解码 → 多目标内化训练目标

---

*相关文档:`LEAD_训练原理_pretrain_posttrain.md`(基座方法)、`踩坑总结_LEAD_H20.md`(环境/评测限制)。*
