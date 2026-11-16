# EXPLORE_FAST_20261116：方法探索期的快速实验协议

**定位**：这一阶段的实验只用于**选方法**，不写进论文。目标是在结论仍然可信的前提下，让每次实验尽量快。等方法选定后，再按论文规范完整重跑（3 个 seed、完整预算、标准五折）。

## 保留什么，去掉什么

| 保留（保证结论可信的最低要求） | 去掉（只服务于论文规范的部分） |
|---|---|
| 与 B0 同预算、同 seed 配对比较 | 每 epoch 断点、续跑合同、来源哈希 |
| 固定取最终 EMA，不挑 checkpoint | 聚簇、bootstrap、运动学面板、多端口伤害判定 |
| 运行前写下主指标（一行 `--note` 即可） | 审计、封存、锁文件、多页审查报告 |
| 用已知噪声尺度（σ）来判定，而不是看到正数就算有效 | 默认 3 个 seed（改为先跑 1 个，必要时再加） |
| 中期停止只用 CALIB（从训练中留出），不用 dev | 每个阶段单独写设计文档 |
| 方法选定后，必须在标准五折上确认 | — |

一次运行只产出一个 `result.json`，并在 `LEDGER.csv` 里追加一行。跑崩了就直接重跑。

## 三个工具

### 1. `eval_budget_on_test.py`：五折结果，今天就能出（零训练）
五折预算阶段已经训练好了 10 个模型（5 折 × 64/128 epoch，seed 42，用 95% 训练集）。这个脚本直接把它们在测试集上评一次，不需要任何训练。

它的训练数据协议和 EXP-156 相同（同样是 95% 训练集 + scene_tail 验证集），所以可以直接和 M2 的单 seed 结果并排比较。这就足以回答当初做五折的方向问题：B0 与 M2、与文献相比处在什么位置。

```bash
python code/eval_budget_on_test.py --fivefold-code <FIVEFOLD_B0_20261115/code> \
  --exec-dir <NEXT_PHASE_EXEC_20261114> --data-dir <.../processed_data_diverse> --out FIVEFOLD_RUNS \
  --exp156 research_archive/02_cross_dataset_and_efficiency/OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/EXP_156B_FOLD_LOCAL_CURRENT_DECODER/summary.json
```

- 需要已经存在 `BUDGET_DECISION.json`。
- 只新增 `test_explore_*` 文件，不碰原有的 `metrics.json`/`DONE.json`，也不改五折代码目录。正在进行的 `final` 运行之后仍可续跑。
- 3 个 seed、128 epoch 的 `final` 运行可以先停下来，推迟到"确认档"再做。

### 2. `fast_run.py`：短预算、单 seed 优先的探索运行
在 C0 的 fit 上训练、dev 上评价，只算轻量指标：minADE/FDE、覆盖率、change 覆盖率、Top1、全 K20 接续加速度。凡是 `train_core.Trainer` 认识的臂都能跑（B0、T1–T3、E2_\*、E3_\*）。以后加新方法，只需要在 `Trainer` 里加一个臂名。

```bash
python code/fast_run.py --exec-dir <NEXT_PHASE_EXEC_20261114> --out EXPLORE \
  --arms B0 <新方法> --epochs <E_explore> --seeds 42 --note "主指标: change 覆盖率"
```

**可选：中期无望停止。** 加上 `--calib-every 2 --stop-if-behind coverage` 后：
- 从 50% 预算开始，若最近 3 次 CALIB 观察相对 B0 同期曲线的平均差 ≤0，就停止运行。
- 这类运行改为在 FIT85 上训练（把 CALIB 留出），目录带 `_fit85` 后缀，自动与全 fit 的运行分开比较；对应的 B0 也必须用同样的参数跑一次。

### 3. `fast_compare.py`：配对比较和判定

| 判定 | 条件 | 下一步 |
|---|---|---|
| promising（有希望） | 配对差均值 ≥ 2σ/√n，且每个 seed 同号 | 进入候选名单 |
| dead（无效） | 均值 ≤0，或运行被无望停止 | 放下 |
| ambiguous（不确定） | 介于两者之间 | 同一预算再加一个 seed |

σ 默认取 E1 在 dev 上测得的值：overall 1.1pp、change 1.8pp。如果某个预算下 B0 跑了两个 seed，会顺便打印两者的差，作为该预算噪声的粗略检查；需要时用 `--sigma-overall` 或 `--sigma-change` 覆盖默认值。

## 第一步：校准探索预算（约 1 小时，只做一次）

```bash
python code/fast_run.py --exec-dir <...> --out EXPLORE --arms B0 T1 E3_self --epochs 16 32 --seeds 42
python code/fast_run.py --exec-dir <...> --out EXPLORE --arms B0 --epochs 16 32 --seeds 43
python -I code/fast_compare.py --out EXPLORE --calibrate
```

- 共 8 次运行：16 epoch 约 5 分钟，32 epoch 约 11 分钟。
- 判断标准：在 16 和 32 中找最短的预算，使 T1 和 E3_self 都能复现它们在 64 epoch、seed 42 下的效果。已知效果为：T1 overall +1.66 / change +3.85；E3_self +2.61 / +4.58。"复现"指 overall 和 change 都同号，且幅度至少达到原来的一半。
- 这个预算就是之后的探索预算。如果 16 和 32 都没通过，说明短预算会扭曲方法排序，探索仍需用 64 epoch。

## 之后的日常流程

1. 新方法用探索预算、seed 42 跑一次（可选加中期停止）。
2. 用 `fast_compare` 判定：dead 直接放下；ambiguous 加 seed 43；promising 进候选名单。
3. 候选名单里积累了 1–3 个方案后，再按论文规范进入确认档：3 个 seed、五折选出的预算、标准五折。

按每 epoch 约 20 s 估算：一个无效方案在 16 epoch、带中期停止时，只需约 3 分钟；在 32 epoch 时约 6 分钟。

## 需要知道的代价

- **dev 会被反复使用**，到探索后期 dev 上的数字会偏乐观。这对"选哪个方法"影响不大，但这些数字不能当作结论引用。真正的结论只来自确认档的五折。
- **σ 是在 64 epoch 下测的**，短预算下的噪声可能更大。校准时 B0 两个 seed 的差可以提供一个粗略检查。
- **单 seed 判定会有误判**：一个真实效果恰好等于阈值的方法，大约有 12% 的概率被误判为 dead；效果像 T1、E3 这样大的方法，误判概率不到 1%。

## 验证情况

- `test_explore_fake.py` 用假的 torch 和 `train_core` 跑通了以下内容：B0 优先运行、账本写入、已有结果跳过；中期停止会正确停掉更差的臂，并把运行标为 `fit85`；判定和预算校准；`eval_budget_on_test` 在未决定预算时被拒、不覆盖原有文件，且之后 `final` 仍可正常续跑。
- 真实 torch 路径在本环境中**未运行**。第一次使用时，先做上面的校准步骤，它本身就是一次真实环境的冒烟测试。
