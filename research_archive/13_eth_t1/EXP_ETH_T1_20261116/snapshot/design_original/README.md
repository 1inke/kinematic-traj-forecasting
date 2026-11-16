# 【待执行】实验一 EXP-ETH-T1：改训练方式，能不能补上 ETH 折？

**探索档**：只用于方法选择，不进论文。

## 为什么做

- ETH 是五折里我们**最弱**的一折：B0 为 0.444/0.664（95% 训练、128 epoch），TUTR 为 0.40/0.61。
- ETH 也是唯一**干净**的一折：我们做实验一直用的 C0 数据，和另外四折的测试集有重叠，但和 ETH 没有。
- ETH 测试集里 52% 是转弯、变速窗口，正是 T1（退火软 WTA）最擅长的部分。

这个实验回答的问题：**只改训练方式（T1），ETH 能不能明显变好？** 结果决定下一步主线怎么走。

## 做什么

| 项 | 设置 |
|---|---|
| 数据 | ETH 折；训练用 95% scene_tail 训练集，与五折预算阶段**完全相同的索引**；测试用 ETH 测试集（364 窗） |
| 对比 | **B0**：直接复用五折预算阶段已训好的 `budget_eth_E64_s42`，**不重训**。**T1**：新训练 1 次 |
| 训练 | 64 epoch，seed 42，固定取最终 EMA；配方与 B0 完全相同，只有损失换成 T1 |
| 评价 | 两个模型各在 ETH 测试集上评一次：标准 minADE20/minFDE20，以及覆盖率、分组、Top1、接续加速度 |

**新训练只有 1 次，单独运行约 30 分钟**（ETH 95% 训练集约 2.9 万窗，每 epoch 约 450 步）。

## 判定规则（运行前已固定）

主指标：ETH 测试集上 T1 相对 B0 的 minADE20 和 minFDE20 的改善比例。

| 判定 | 条件 | 下一步 |
|---|---|---|
| **有帮助** | minADE 和 minFDE 都改善 ≥2% | 进入训练目标重构，作为主线 |
| **无帮助** | minADE 和 minFDE 改善都 <0.5%（或变差） | 先做零训练诊断，查清 ETH 差在哪里（例如速度分布），再定重构重点 |
| **不确定** | 介于两者之间 | B0 和 T1 各补跑一个 seed 43（2 次，约 1 小时）再判断 |

噪声依据：ETH 折上已有两个 seed 的 B0（100% 训练、128 epoch），minADE 相差 0.0027（0.6%），minFDE 相差 0.0088（1.4%）。2% 大约是这个差距的 3 倍和 1.4 倍。

**次要（只作描述）**：
- 各组覆盖率和救回/丢失。change 组有 190 窗，1 窗约等于 0.53pp，所以小差异不可靠。
- Top1、接续加速度。
- 按观测速度分三档的误差，以及训练集和测试集的速度对比。这一项用于"无帮助"时判断原因。
- 参照行：B0 128 epoch、M2、TUTR。

## 运行

```bash
# 0) 自检（无需 GPU，几秒）
python code/test_eth_t1.py --fivefold-code <qoder_work/FIVEFOLD_B0_EXEC_20261115>/code
# 1) 预检（几秒）：B0 断点是否为 ETH/64 epoch/seed 42、训练索引是否一致、T1 参数量 295,474、一步训练有限
python code/run_eth_t1.py preflight --explore-dir <qoder_work/EXPLORE_FAST_20261116> --out <qoder_work/EXP_ETH_T1_20261116>
# 2) 训练 T1（约 30 分钟；中断后重跑同一命令即可续跑）
python code/run_eth_t1.py train --explore-dir <...> --out <...>
# 3) 在 ETH 测试集上评价 B0 和 T1（几十秒）
python code/run_eth_t1.py evaluate --explore-dir <...> --out <...>
# 4) 分析（CPU，几秒）
python -I code/analyze_eth_t1.py --out <qoder_work/EXP_ETH_T1_20261116> \
    --fivefold-code <qoder_work/FIVEFOLD_B0_EXEC_20261115>/code --data-dir <.../processed_data_diverse>
```

结果写入 `ETH_T1_RESULT.md` 和 `ETH_T1_RESULT.json`。

注意：B0 的 64 epoch 断点必须还在本机，即 `budget_eth_E64_s42` 目录里最后一个 `ckpt_epoch*.pt`。如果已被删掉，预检会报错。这时需要先按同样设置重训一次 B0，再跑本实验。

## 局限

- **单 seed、95% 训练集、64 epoch**：结论只用于决定主线方向，不能当作 ETH 上的正式结果。
- **ETH 测试只有 364 窗**，覆盖率这类按窗口计数的指标波动较大，所以主判定用的是连续误差 minADE/FDE。
- **T1 只是新训练目标的雏形。** "有帮助"说明这个方向对 ETH 有效；具体重构方案的效果，还要在重构后再测。

## 文件与验证

| 文件 | 内容 | 验证 |
|---|---|---|
| `code/run_eth_t1.py` | 预检、训练、评价 | 用桩模块测完整流程：未预检时拒绝训练；训练后重跑不重复；B0 断点的训练索引不一致时预检失败。真实 torch 未在本环境运行 |
| `code/analyze_eth_t1.py` | 判定与报告 | 判定规则覆盖五种情形；速度分档在合成五折数据上跑通 |
| `code/test_eth_t1.py` | 以上测试 | 全部通过 |
