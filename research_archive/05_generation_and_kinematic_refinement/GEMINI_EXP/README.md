# GEMINI_EXP: 自主多轮学术优化实验体系

本目录是严格遵循项目学术规范（`RULE[user_global]`, `GEMINI.md`）的独立实验体系。
所有新增代码、数据、模型、配置与分析报告均完全隔离于此目录及其子目录中，绝不改动任何既有实验目录与已有成果。

---

## 一、学术规范与质量合同

所有在此开展的实验均遵循以下最高准则：

1. **唯一好候选标准**：
   - 同一条候选轨迹必须同时满足：$\text{ADE} \le 0.20\,\text{m}$ 且 $\text{FDE} \le 0.40\,\text{m}$。
   - 连续联合指标：$J = \min_k \max(\text{ADE}_k / 0.20, \text{FDE}_k / 0.40)$。$J \le 1.0$ 代表至少存在一条完整好候选。
   - 严禁拼接不同候选的 minADE 与 minFDE。

2. **主评价人群与优先级**：
   - Primary 目标：`outer change`（253 窗口，68 个目标 tracks）。
   - 泛化四划分全部报告：`FIT`（18,909 训练窗口）、`outer`（1,035 外层测试窗口）、`inner`（1,856 内层校准窗口）、`original_CAL`（2,879 独立窗口）。
   - 四组别细分：`overall`、`stable`、`transition`、`change`。
   - 指标优先级：
     1. 完整好候选覆盖率（Coverage）
     2. 救回与丢失（Rescue / Lost）
     3. 去重后的有效好候选聚类（Effective good clusters）
     4. 连续指标（minADE / minFDE / J）
     5. 运动学合规性（speed p95, accel mean/p95, jerk mean/p95 均必须 $\le 1.01 \times \text{native}$）
     6. 轨迹选择与排序（Scorer / Top-1）

3. **科学对照与证伪**：
   - 每次实验严格改变一个主要科学因素（或严格正交析因）。
   - 必须包含：CONTROL、TREATMENT、HYPOTHESIS、FALSIFICATION。
   - 3 随机种子（42, 43, 44）逐一报告，禁止用 best seed 代替整体。
   - 必须主动寻找反例：负向 seed、丢失率超过救回率、运动学破坏、仅训练集过拟合等。

4. **审计与独立性**：
   - 每个子实验具备独立目录及完备生命周期文件：
     `DESIGN.md` -> `protocol.json` -> `PREFLIGHT.json` -> `runtime_state.json` -> `history.json` -> `RESULT.json` -> `ARITHMETIC_AUDIT.json` -> `RESULT_ANALYSIS.md`。
   - 保存输出由 CPU 独立代码重算验证，确认无浮点泄露或评价分母偏差。

---

## 二、已完成实验档案与科学结论 (AUDITED EXPERIMENTS)

| 编号 | 实验代号 | 核心科学问题 / 假设 | 对照基线 | 状态与结论 |
|---|---|---|---|---|
| **EXP01** | `EXP01_NONOWNER_ANCHOR_FACTORIAL` | 非负责候选软锚定能否减少好模式丢失？ | 标准多任务细化器 vs $\lambda_{\text{anchor}}=1.0$ | **SUCCESS_SCIENTIFIC_NEGATIVE**: 软锚定产生19对1的反向梯度抵消，将共享MLP彻底冻结在恒等映射（平均位移5微米）。 |
| **D1** | `D1_TIMED_MULTI_HORIZON_PLAN_20261020` | 多时步计划显式监督 ($t=4,8,12,20$) 能否提升短期轨迹与整体覆盖？ | 终点计划监督 (Arm A) vs 4/8/12/20全时步计划 (Arm B) | **SUCCESS_SCIENTIFIC_NEGATIVE**: 计划误差降低20%~40%，但 outer change 覆盖率反而下降 (-2.11 pp)，中间时步过度刚性约束破坏了自然曲率。 |
| **D2** | `D2_MOTION_PRETRAINED_REPRESENTATION_20261020` | 历史运动自监督预训练表示能否打破跨域泛化瓶颈？ | 遮挡重建编码器 (Arm A) vs 因果自回归编码器 (Arm B) | **SUCCESS_SCIENTIFIC_NEGATIVE**: 预训练下游强依赖历史表示，但历史信息无法消除未来多模态分歧，两臂 outer change 均未超过原生 (38%~39% vs 43.5%)。 |
| **EXP03** | `EXP03_KINEMATIC_DAMPING_FACTORIAL` | 控制能量阻尼与二阶平滑惩罚能否消除高频抖动并减少 Lost？ | 标准基线 (Arm A) vs $\lambda_{\text{ctrl}}=0.05, \lambda_{\text{smooth}}=0.10$ (Arm B) | **SUCCESS_SCIENTIFIC_NEGATIVE**: 阻尼显著减少好模式丢失 (Lost 减少 48%，$p=0.0132$)，额外加速度削减 67% (1.040×)，但因救回衰减使得净增益跨零 ($p=0.4765$)，且加速度未完全达标 1.01×。 |
| **D3** | `D3_FUTURE_INTERACTION_CONDITIONED_20261020` | 两轮未来时空交互消息更新能否打破单智能体泛化瓶颈？ | 过去邻居编码 (Arm A) vs 两轮未来交互消息 (Arm B) | **SUCCESS_SCIENTIFIC_NEGATIVE**: 未来时空冲突消息更新未能带来显著增益 (+0.13 pp, CI [-2.08, +2.22] pp, $p=0.8899$)，基于等速外推的局部空间碰撞场无法解释大尺度意图转向。 |

---

## 三、下一阶段规划 (NEXT PHASES)

- **EXP04: 模式解耦与意图路由条件化 (Modal Decoupled Routing & FiLM Modulation)**
  - 核心动机：单网络权重共享导致 20 个候选梯度互相冲突；按模式几何簇进行子空间解耦或 FiLM 调制，避免单胜者梯度破坏其余候选。
