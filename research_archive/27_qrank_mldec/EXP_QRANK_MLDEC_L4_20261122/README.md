# EXP-QRANK-ML

2026 年 11 月 23 日完成。归档包含 Z0 先验诊断、33 组 C0 筛选训练和 51 组正式五折结果。两条正式对照共用置信度排序、固定 0.5 m 匹配阈值和 EMTA 分配。QFL 使用质量感知置信度监督，E0 使用原置信度监督。每次训练 64 轮。ETH 使用五个种子，其余折各三个。[实验设计](code/sealed/DESIGN_QRANK.md) · [执行补充](code/sealed/AMENDMENT.md)

## 结果

| 方法 | 五折 minFDE 改善 | 五折 Top-1 FDE 改善 | 五折 brier-minFDE 改善 | ETH minFDE 改善 | F-PASS |
|---|---:|---:|---:|---:|---|
| O-conf+QFL | +6.36% | +1.31% | −3.26% | −2.63% | 未通过 |
| O-conf+E0 | +5.70% | +1.31% | −3.07% | −3.75% | 未通过 |

两臂均有四折的 minFDE 改善，ETH 退步超过预设的 1% 容差，保留 B0 为主基线。QFL 相对 E0 的五折 minFDE 改善 0.69%，Top-1 基本持平，brier-minFDE 退步 0.19%。QFL 在 C0 上的 brier-minFDE 收益为 3.54%。[完整分析](ANALYSIS_ZH.md) · [C0](results/STAGE1_RESULT.json) · [五折](results/STAGE2_RESULT.json) · [直接配对比较](results/QFL_VS_E0.json)

正式五折包含 41 组本轮 L4 训练、2 组本机 RTX 5070 Laptop 训练和 8 组校验后复用的 L4 基线。ZARA02 seed 44 的 QFL、E0 在本机，B0 在 L4。按用户在正式五折完成后先审阅结果的要求，P2 在模型初始化前暂缓。原调度器记录的 `failed` 是启动拦截返回码，正式 C0 和五折的完成回执为 `PASS`。[状态](STATUS.json) · [正式完成回执](results/DONE.json) · [P2 控制记录](results/P2_REVIEW_CONTROL.json)

## 代码与复核

- [训练](code/sealed/qrank_train.py) · [调度](code/sealed/qrank_study.py) · [度量](code/sealed/qrank_eval.py) · [预设判据](code/sealed/qrank_decisions.py)
- [L4 预检](results/PREFLIGHT.json) · [Z0 结果](results/Z0.json) · [文件校验](ARTIFACT_MANIFEST.json)

归档保留 84 份结果回执和对应的无坐标逐窗指标。用安装了 NumPy 和 CPU PyTorch 的环境运行 `python -I -B tools/reanalyze.py`，可复算 C0、晋级判定、五折和 QFL/E0 配对分析。重跑训练时，按清单准备封存输入，并从 Phase 26 将两份共用源文件补入 `code/sealed` 的对应位置。原始轨迹、完整坐标预测、模型权重和断点保存在原实验环境。[共用源文件](code/SHARED_SOURCES.json) · [省略资产清单](OMITTED_ARTIFACTS.json)
