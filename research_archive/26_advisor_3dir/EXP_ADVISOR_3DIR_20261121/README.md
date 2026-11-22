# EXP-ADVISOR-3DIR

2026-11-22 完成。归档包含 Z2 候选池诊断、C0 的 33 次筛选训练，以及 B0 与 R2 在 ETH、ZARA01、HOTEL、ZARA02、UNIV 上的 30 次正式五折训练。每次正式训练使用 64 轮，配对种子为 42、43、44。

## 结果

- **Z2 候选池诊断。** 合并五个 B0 模型的 100 条候选后，C0 dev 的 minFDE 改善 33.63%。加权 k-medoids 压回 20 条后，minFDE 点估计改善 2.40%，95% 配对区间为 −0.34% 至 +4.85%，转向难组的 minFDE 退步 10.89%。[Z2 原报告](Z2/REPORT.md) · [分组结果](Z2/RESULT.json)
- **C0 筛选。** 应用深监督的两遍修正 R2 比 B0 的 minFDE 改善 3.68%，比接入同样信息的单遍对照 R3 改善 3.48%。R2 通过 R-PASS 和 R-DS，排序与候选分组压缩方向的准入判据未通过。[筛选结果](results/stage1/RESULT.json) · [第一遍和修正统计](results/stage1/MECH_R.json) · [晋级记录](results/stage2/ADMISSION.json)
- **五折验证。** R2 的五折等权 minFDE 改善 3.33%，minADE 改善 1.83%，共有 4 折的 minFDE 改善。ETH 三个种子的 minFDE 均退步，折均值退步 2.85%，因此 `F-PASS=false`。五折 Top-1 FDE 折均值均退步，等权平均退步 4.89%。按预设判据保留 B0 为主基线。[五折结果](results/stage2/RESULT.json) · [执行状态](STATUS.json)

在 ETH 上，两种方法都在本机 RTX 5070 Laptop GPU 训练。B0、R2 第一遍及 R2 最终输出的三种子平均 minFDE 依次为 0.609、0.617、0.626 m。ZARA01 的 B0 在本机训练，R2 在 Colab L4 训练。其余三折两臂均在 L4 训练。每组的 `RESULT.json` 带运行合约。可选敏感性分析的状态为 `NOT_STARTED`。

## 代码和复核

- [设计与预设判据](code/sealed/DESIGN.md) · [正式训练和评估](code/sealed/advisor_train.py) · [原评估代码](code/sealed/advisor_eval.py) · [五折续跑](code/fivefold/continue_fivefold.py)
- [L4 预检](results/cloud/PREFLIGHT.json) · [完成凭据](results/DONE.json) · [文件清单](ARTIFACT_MANIFEST.json) · [源文件校验](COPY_PROVENANCE.json)

本目录保留 63 份正式运行的结果回执和逐窗指标、Z2 的 fit 和 dev 派生数组，以及模型、评分、筛选、训练和恢复代码。原始轨迹、完整预测、模型权重与训练断点保存在原实验环境。[省略文件散列](OMITTED_ARTIFACTS.json)

用安装了 NumPy 和 CPU PyTorch 的 Python 运行 `python -I -B tools/reanalyze.py`。复核器按逐窗指标重算 C0、五折与 Z2 的分组和判据。训练入口需要与 [输入清单](code/sealed/COLAB_INPUTS.json) 一致的数据。Z2 的本机入口另依赖 [Phase 23 的数据读取代码](../../23_adapt_es_precheck/EXP_ADAPT_ES_PRECHECK_20261120/snapshot/)。
