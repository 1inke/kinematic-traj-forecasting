# DPP 双重可行性评分：脚本准备包

当前状态：**脚本已编写，仅合成测试；按2026-11-04最新人类要求，不启动。**

未运行真实数据预检、FIT阈值校准或离线实验；未训练任何模型；未创建后台进程、计划任务或自动跟进。之前基础代码保留，旧实验产物不改不删。

## 已写好的流程

| 文件 | 功能 | 本轮执行情况 |
|---|---|---|
| scoring.py | 米制速度/加速度/历史接续评分、SDF查询、原半径质量、DPP核与EC、有效ASD/FSD | 仅合成测试 |
| independent.py | 逐候选独立算术、谱分解EC与边际概率、同槽ADE/FDE复核 | 仅合成测试 |
| prepare.py | 未来获准后核输入SHA/ID、仅FIT校准参数、封存真实来源 | **未运行** |
| preflight.py | 未来获准后的真实128窗保存/读回/双实现预检 | **未运行** |
| evaluate.py | 未来获准后的5固定候选源×fit/dev离线评分；不修改候选 | **未运行** |
| review.py | 全保存分数/原始惩罚/DPP EC与边际概率第二实现及最终报告 | **未运行** |
| run_stage_a.py | 手动串行入口，无自动任务或训练调用 | 仅测试默认不运行/禁止启动 |
| stage_b_admission.py | 正式DSF质量消融缺失条件，不构造替代模型 | 静态脚本 |
| test_synthetic.py | 合成轨迹/地图、数值、FD、异常与启动防护测试 | 已运行合成测试 |

## 实验边界

1. 原文首先要求阶段A离线诊断。现成可执行流程为固定K20上的运动学质量和DPP核分析，**不是原DSF端到端训练已实现或完成**。
2. 固定候选源为fresh-B0及DGeo-V2四臂，均seed42。它们不是本文B0-B4五训练组或5独立随机种子。
3. 原论文B0需要真实Gaussian latent及DSF生成接口。H3 embedding不能伪装成该latent。**阶段B训练器/模型适配尚未实现**，准入脚本明确拒绝，而非用新结构代替。
4. 当前无合格同recording可通行图。真实q_scene/q_joint保持未知；场景函数只有合成地图测试，不代表真实效果。不能将缺图样本默认评分1或0。
5. 文档未定数值已在protocol.draft.json披露：仅FIT99.5%阈值、IQR尺度、FIT固定hash128核尺度、epsilon_q=1e-6、固定转向分箱。均未在真实数据计算或选择，没有验证集扫参。
6. epsilon_q只用于DPP质量下限；旧同槽ADE<=.2且FDE<=.4、原运动学验收、基线与历史门不变。raw/logq与下溢单独报告。
7. 离线只评分不会改善K20 minADE/FDE或覆盖率。有效多样性、EC与权重重分布不能冒称生成器改善，不进行候选删选或DPP采样。

## 防止误启动

execution_control.json的allow_experiment_start=false。所有真实数据入口默认DRY_RUN；即使传--execute，也会在读取真实数据前拒绝。需要你之后明确允许启动，再由助手更新控制状态。

检查入口（不会启动实验）：
    .venv-governance/Scripts/python.exe -B qoder_work/DPP_DUAL_FEASIBILITY_OFFLINE_EXEC_20261104/run_stage_a.py

本轮只执行test_synthetic.py。最新收据见synthetic_tests目录，绑定当前源码SHA；早先13测试收据SYNTHETIC_TESTS.json原样保留。真实输入预检、资源峰值与运行ETA尚未测量，不给出伪生产估计。

## 研究审查

沿用ARS反方审查，同助手角色非外审。缺图和DSF接口使完整双重质量学习问题尚不可识别。阈值来源相同使“分数低对应定义内违规高”具有同义性，不独立证明行人安全。无真实结果、显著性结论或自动训练。

原文：<LOCAL_DOWNLOADS>/DPP_双重可行性质量评分_改进实验设计报告.md

方法核对来源：[Yuan & Kitani, ICLR2020, section4.2](https://arxiv.org/html/1907.04967)。EC按solve(I+L,I)与谱分解交叉核对；核为轨迹向量欧氏距离的Gaussian核。原DSF网络及潜变量不属于当前包的已验证实现。

