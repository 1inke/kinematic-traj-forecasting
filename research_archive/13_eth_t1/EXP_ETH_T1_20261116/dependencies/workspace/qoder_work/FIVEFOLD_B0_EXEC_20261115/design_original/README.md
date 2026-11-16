# FIVEFOLD_B0_20261115：当前 B0 的标准 ETH/UCY 五折基准

2026-11-15｜**DESIGN_AND_CODE**：设计和代码已完成，尚未在真实模型上运行（见文末“验证情况”）。

## 1. 这一步要回答什么

| 编号 | 问题 | 为什么重要 |
|---|---|---|
| Q1 | 当前 E2E 版 B0（295,474 参数），在统一协议、经过预算选择、3 个 seed 下，标准五折的等权 minADE20/minFDE20 是多少？ | 决定接下来的“条件量化”重构以谁为底座 |
| Q2 | 它与 EXP-156 M2（355k，单 seed，约 0.220/0.364）、EXP-133（284k，单 seed，0.2325/0.4428）相比如何？ | M2 多了 V2 宽修正和模式模板。如果 B0 明显更差，说明这些组件有价值，重构不能直接丢掉 |
| Q3 | seed 之间的波动有多大？特别是只有 364 个测试窗的 ETH 折 | 判断以往单 seed 五折结论的可信度；论文表格必须报告 seed 波动 |
| Q4 | 为后续工作准备数据 | 打分校准分析、V3 运动判据、ETH 折诊断都要用逐窗口的完整输出 |

**不回答**：任何新方法的好坏；不做因素组合；不改 baseline。

## 2. 协议

| 项 | 本研究 | EXP-156（对照） |
|---|---|---|
| 数据 | `processed_data_diverse/{fold}_{data,num}_{train,test}.npy`，EXP-156A 已审核 | 同 |
| 折 | eth / hotel / univ / zara01 / zara02，留一场景 | 同 |
| 观测/预测 | 8 / 12 帧，2.5 Hz（3.2 s / 4.8 s） | 同 |
| 模型 | 当前 B0：F16 编码器 + h3 解码器，295,474 参数 | H3→V2→M2 三阶段，355,048 参数 |
| 训练 | 与 NEXT_PHASE B0 完全一致：ModeQueryLoss + 0.20 软覆盖，AdamW 2.5e-4/wd 5e-4，1 epoch warmup + cosine，EMA 0.995，batch 64，反射 + 0.9–1.1 速度增强 | 类似配方，30/20/30 epoch |
| 训练轮数 | E\* ∈ {64, 128}，由 fold 内验证集决定（§3） | 固定 |
| checkpoint | **固定取最终 EMA**，不在任何集合上挑 checkpoint | 按验证集 oracle_combo 选择 |
| 验证集 | scene_tail 5%（与 EXP-156 的规则逐行一致，已测试） | 同 |
| 指标 | 只评目标行人；best-of-20；minADE 与 minFDE 各自独立取最小；对测试窗取均值；五折等权平均 | 同 |
| seed | 42 / 43 / 44 | 1 |

训练代码不是新写的：`run_fivefold.py` 直接调用 NEXT_PHASE 执行目录里的 `train_core.Trainer` 和 `infer`，没有任何改动。新写的只有三部分：数据适配、轮数选择/重跑/续跑的调度，以及指标。

## 3. 流程

```
F0 审核 ─► F1 预算运行（10 次）─► F2 决定 E*（封存）─► F3 最终运行（15 次，测试集只在这里打开）─► F4 汇总
```

**F0 审核**（`audit_folds.py`，只用 numpy）。在 EXP-156A 的检查之外，新增：
- 训练集与测试集的场景 id 不相交；
- 目标行人 20 帧全部有限；报告填充 agent 的取值；
- 坐标尺度检查：观测段速度中位数应在合理的 m/s 范围；
- 记录 scene_tail 切分后的规模，以及四个文件的 SHA256，写入每次运行的合同。

任何一项失败，后续步骤都拒绝运行。

**F1 预算运行**：5 折 × E ∈ {64, 128}，seed 42。在 fold 训练集的 95% 上训练，在 5% 的 scene_tail 验证集上评价。

**F2 决定 E\***（规则在任何运行前固定，写在 `fivefold_common.py`）：
- 若 64→128 时，五折等权的验证集 minADE 相对改善 ≥1%，且等权 minFDE 不变差，则 E\* = 128；否则 E\* = 64。
- 若改善 ≥3%，额外标记“128 可能仍未饱和”，但本轮不跑 256。
- 决定写入 `BUDGET_DECISION.json`（只写一次），其中 `test_accessed=false`。

**F3 最终运行**：5 折 × seed 42/43/44，用 E\* 在 100% fold 训练集上训练，在测试集上评价一次。没有 `BUDGET_DECISION.json` 时，程序拒绝运行，测试集保持未打开。

**F4 汇总**（`summarize_fivefold.py`）：
- 先在每个 seed 内算五折等权平均，再报告 3 个 seed 的均值 ± 标准差；
- 逐折表；去掉 ETH 后的平均；
- 与 EXP-156 M2、EXP-133 及文献量级对照。

### 运行命令
```bash
# 0) numpy 自检与调度自检（无需 GPU）
python -I code/test_fivefold_numpy.py
python code/test_orchestration_fake.py
# 1) 审核
python -I code/audit_folds.py --data-dir <ASTRA_ACCURACY_Q20>/datasets/eth_ucy/processed_data_diverse --out F0/AUDIT.json
# 2) 真实 torch 冒烟：1 个 epoch、512 窗，几分钟；输出目录名必须含 SMOKE，与正式结果隔离
python code/run_fivefold.py budget --smoke --folds hotel --out FIVEFOLD_SMOKE \
  --exec-dir <qoder_work>/NEXT_PHASE_EXEC_20261114 --data-dir <...>/processed_data_diverse --audit F0/AUDIT.json
# 3) 全流程（可中断，重跑同一命令即从最近的 epoch 断点续跑）
python code/run_fivefold.py all --out FIVEFOLD_RUNS --exec-dir <...> --data-dir <...> --audit F0/AUDIT.json --device cuda:0
# 4) 汇总
python -I code/summarize_fivefold.py --out FIVEFOLD_RUNS \
  --exp156 research_archive/02_cross_dataset_and_efficiency/OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/EXP_156B_FOLD_LOCAL_CURRENT_DECODER/summary.json
```

## 4. 运行量与耗时

按 NEXT_PHASE E1C 实测：B0 每 epoch 约 19.8 s / 340 次更新，约 1,100 窗/s。五折训练集合计 124,510 窗。

| 阶段 | 运行数 | 估计训练时间 |
|---|---:|---:|
| F1 预算（95% 数据，64 + 128 epoch） | 10 | 约 5.7 h |
| F3 最终（E\* = 64） | 15 | 约 6.0 h |
| F3 最终（E\* = 128） | 15 | 约 12.1 h |
| **合计** | 25 | **约 12 h（E\*=64）/ 18 h（E\*=128）** |

推理时间可以忽略（测试集合计 34,161 窗）。磁盘方面：每次运行保留最近 2 个 epoch 断点，可在 DONE 后删除；univ 折的测试预测每个 seed 约 47 MB（float32）。

注意：轮数按 epoch 计，univ 折训练集只有 9,874 窗，所以同样 epoch 下更新次数约为其他折的 1/3。这与 EXP-156 的惯例一致，汇总时会注明。

## 5. 指标

**主指标**：五折等权 minADE20 / minFDE20，3 个 seed 的均值 ± 标准差。

**次指标**（逐折、逐 seed）：
- same-path FDE（ADE 最优那条候选的 FDE）；
- Top1 ADE/FDE；
- 按 logit 排序的 top-3/5/10 minADE/minFDE；
- 项目覆盖率（同一候选 ADE≤0.2 且 FDE≤0.4），以及 change 组覆盖率；
- 匀速外推参照。

**打分校准（仅描述，为后续研究做准备，不作任何判定）**：
- 以 J-winner 所在区域为目标的 NLL 与 Brier；
- “Top1 是否为 J-winner”的 10 箱 ECE。

**成本**：参数量由每次运行断言（必须等于 295,474，否则报错）。延迟不重测：结构与 NEXT_PHASE 完全相同，沿用 `cost/E1C_E64_s42.json` 的实测值（CPU 单样本 p50 10.54 ms；GPU 单样本 22.42 ms；GPU batch64 27.27 ms，后者是整个 batch 的延迟）。

## 6. 结果怎么读（预先写下，只作解读指引，不是门）

| 观察 | 解读与下一步 |
|---|---|
| B0 等权 ADE 与 M2 相差 ≤0.005，FDE 相差 ≤0.01 | B0 就是重构底座：参数更少、结构更简单，V2/M2 组件在新框架下不必保留 |
| B0 比 M2 差 >0.01 ADE 或 >0.02 FDE | V2 宽修正或模式模板确有贡献；重构必须以 M2 为底座，或在新框架里证明它们可被替代 |
| B0 比 M2 更好 | 说明训练预算和固定最终 EMA 的协议本身就是改进；以 B0 为底座，并在论文中说明 |
| ETH 折的 seed 标准差 ≥ 方法间差距 | 以往单 seed 的 ETH 结论不可靠；后续所有五折比较都至少报告 3 个 seed |
| 与 TUTR/LED 的差距在 0.01–0.02 ADE 以内 | 论文可以按“同量级精度、显著更轻”来定位；重构目标是缩小 FDE 差距 |
| ETH 测试集的观测速度分布明显高于训练集（F0 的速度检查） | ETH 的落后主要来自跨场景速度偏移；之后再考虑速度归一化的表示（本轮不做） |

## 7. 与公开结果对比的边界

- `processed_data_diverse` 是项目自己的以目标为中心的预处理，测试窗数量可能与公开加载器（例如 Social-GAN 系）不同。
- 文献数字目前只是 EXP-133 文档里整理的量级（STAR、CAGN、AgentFormer、TUTR、LED）。写入论文表格前，必须回到原论文核对协议：best-of-20、逐人取最小、五折等权、是否使用场景图像。
- EXP-156/133 是单 seed，且使用验证集选 checkpoint。与本研究不同协议，只作量级参照，不作配对比较。

## 8. 输出

```
F0/AUDIT.json
FIVEFOLD_RUNS/ENVIRONMENT.json                环境、torch/cuda、全部源码 SHA
FIVEFOLD_RUNS/BUDGET_DECISION.json            E*、逐折增益、test_accessed=false
FIVEFOLD_RUNS/runs/<phase>_<fold>_E<E>_s<seed>/
    metrics.json        合同（数据/源码/切分 SHA）+ 训练信息（更新数、batch 顺序哈希链、参数量）+ 分组指标
    {val|test}_predictions.npz     paths[N,20,12,2] float32、logits、ids、scene、group   ← 校准和 V3 判据用
    {val|test}_window_metrics.npz  逐窗口全部指标
    DONE.json
FIVEFOLD_RUNS/SUMMARY.json / SUMMARY.md
```
所有结果文件只写一次，已存在就拒绝覆盖。

## 9. 验证情况

| 部分 | 状态 |
|---|---|
| `test_fivefold_numpy.py`（6 项） | **通过**：scene_tail 切分与 EXP-156 原代码逐元素一致；数据布局；分组公式；指标算术；E\* 规则三种情形；审核 CLI 与汇总端到端（含与 EXP-156 的对照计算） |
| `test_orchestration_fake.py` | **通过**（用假的 torch/train_core）：未决定 E\* 时最终运行被拒；预算阶段从不打开测试集；中途模拟崩溃后续跑，batch 顺序哈希链和结果与不中断运行完全一致；DONE 跳过；重复执行不触发覆盖；冒烟模式隔离；汇总端到端 |
| 真实 torch 路径 | **未运行**（本环境没有 torch 和数据）。正式运行前先做 §3 第 2 步的冒烟；它会同时检查参数量是否等于 295,474 |
