# 行人轨迹预测研究与工程规范 (Trajectory Forecasting Research Protocol)

本规范适用于本项目（行人轨迹预测与生成式多模态建模），旨在**在确保科学严密性、反例审计与结果可复现的前提下，最大化释放研究灵活性与系统创新能力**。

---

## 一、双轨工作哲学 (Dual-Track Workflow)

为平衡“快速探索”与“严谨基准”，项目实行明确的**双轨制**：

### 1. 敏捷探针与探索轨道 (Exploration / Agile Track)
* **适用场景**：新灵感验证、快速探针脚本（Probe）、EDA 分析、Sanity Check、轻量代码调试。
* **工作方式**：
  * 鼓励在 `scratch/` 目录下快速编写探针脚本（如 `probe_*.py`），快速输出指标分布、梯度分析或数据切片。
  * **免除繁重审计**：无需在探索阶段机械生成全套 10 个审计与状态文件（`PRECHECK.json`, `SOURCE_AUDIT.json` 等），以控制台清晰输出和轻量日志为主。
* **支持协同创新 (Co-adaptation)**：
  * 深度学习模块往往紧密耦合。允许研究者和模型**协同重构**相互依赖的模块（例如：轨迹表征转换与对应的损失函数缩放必须联合修改）。
  * 约束：方案提出时需说明协同设计的内在逻辑，并在后续消融规划中说明如何隔离因果贡献。

### 2. 正式基准与验证轨道 (Formal / Benchmark Track)
* **适用场景**：正式对比实验、消融实验定论、基准打榜、模型归档与外部报告。
* **工作方式**：
  * 严格锁定配置、记录数据与代码 SHA。
  * 严格执行多 seed 独立评估，保留完整的输入输出审计记录与运行时状态。

---

## 二、不可动摇的科研底线 (Scientific Golden Rules)

不论处于哪个轨道，以下学术诚信与客观性原则**绝不妥协**：

1. **绝对禁止信息泄露 (No Leakage)**：
   * 严禁未来 GT、future labels、future events 或未来分组信息作为推理输入或渗入 forward 计算图。
2. **绝对禁止挑 Seed 与数据粉饰**：
   * 禁止用“最佳 seed”代替多 seed 综合表现；
   * 禁止因结果不理想而事后改变 checkpoint 选择规则或重新筛选随机种子；
   * 正式结果必须逐 seed 完整披露并报告均值与方差。
3. **严禁删除失败实验**：
   * 科学上的阴性结果（Hypothesis falsified）是极有价值的科研资产，必须作为 `SUCCESS_SCIENTIFIC_NEGATIVE` 完整归档，严禁为了数字好看而销毁实验痕迹。
4. **严格的因果与统计推理**：
   * 严禁“均值改善即模型更好”、“训练 loss 下降即泛化改善”、“minADE 单独改善即整体达标”等推断谬误。
   * 结论汇报中，必须严格区分：
     * **OBSERVATION（实际观察到的数据现象）**
     * **INTERPRETATION（与现象一致的可能解释）**
     * **NOT PROVEN（当前数据尚无法证明的部分）**
5. **必须主动寻找并报告反例与代价**：
   * 汇报改善时，必须同时检查：是否存在 seed 方向相反？Coverage 是否与 minADE 倒挂？Rescue 是否伴随大量 Lost？运动学生理指标（加速度/急动度）是否恶化？增益是否仅局限在特定简单子集？

---

## 三、评价指标与核心价值导向

当前生成端的核心目标是**生成物理可行、多样且完整的有效候选轨迹**，而非盲目刷低单一的 minADE：

### 1. 完整好候选定义 (Good Candidate Standard)
同一条 candidate 轨迹必须同时满足两项空间约束：
* $ADE \le 0.2\text{ m}$
* $FDE \le 0.4\text{ m}$

### 2. 连续联合指标 $J$ (Joint Metric)
$$J = \min_k \max\left(\frac{ADE_k}{0.2}, \frac{FDE_k}{0.4}\right)$$
* $J \le 1.0$ 表示该样本至少存在一条完整达标的候选轨迹。

*(注：上述 0.2m / 0.4m 为当前行人基准阈值。如在车辆数据集或像素空间实验中，需依据相应任务配置文件动态适配基准。)*

### 3. 指标优先级梯队
1. **完整好候选覆盖率 (Complete Good Candidate Coverage)**
2. **挽救率与丢失率净收益 (Net Rescue: Rescue vs Lost)**
3. **有效好簇数量 (Effective Good Clusters, GT无关去重后)**
4. **联合指标与误差 ($J$, minADE, minFDE)**
5. **时序与局部误差 (Early / Middle / Late 同路径误差)**
6. **运动学合理性 (Speed p95, Acceleration mean/p95, Jerk mean/p95)**
7. **多环境泛化度 (FIT / Inner / Outer / CAL)**

---

## 四、人机协作与沟通方式

* **开放式讨论与方案构思时**：
  * 保持自然、深入、客观的同行（Research Peer）交流风格；
  * 鼓励发散思维、头脑风暴，共同剖析问题数学本质与物理机制。
* **正式实验节点汇报时**：
  * 保持结构清晰，核心结论直击要害，建议采用以下精炼架构：
    * **STATUS**：当前进展与状态
    * **WHAT WAS TESTED**：测试的具体假设与配置
    * **RESULT**：核心定量结果与反例排查
    * **WHAT IT RULES OUT & NOT PROVES**：已排除的假设与尚待验证的盲区
    * **NEXT DECISION**：下一步建议行动与决策点
