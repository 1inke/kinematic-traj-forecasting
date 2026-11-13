# 过夜总摘要 OVERNIGHT_DIGEST_20261019

- 写成时间：2026-10-19T05:36:53+09:00（Asia/Tokyo）
- **BREAKTHROUGH=false**
- 状态：**standby**（不再派新实验方向）
- 用户授权语境：睡觉期间低占用排队自跑；有大突破或用户回来再喊

---

## 一句话结论

**native＋KIN 池在 outer（及 original_CAL 点估计）上存在达 W 的 K20 见证（K0 / K0-C）；学习到的选择器（V1 / K1-I / K1-S）与去门控几何排序都没稳定跨过联合门。卡在「可校准 / 可学的 GT-free 停手」，不是再扫距离键或继续盲加结构。**

---

## 1. K0：联合可达性有见证 → GO_K1

目录：<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018

| seed | native change_good | V1 | K0-C 策略 oracle | 池/策略见证 |
|---:|---:|---:|---:|---|
| 42 | 110 | 116（W 未过） | **126** | 是 |
| 43 | 110 | 119 | **133** | 是 |
| 44 | 110 | 110 | **127** | 是 |

- 判定：池非空；卡在选择。登记 GO_K1=true（GO_K1_REGISTER.json）。
- 生成器全程冻结。

正式分析：RESULT_ANALYSIS.md / RESULT_ANALYSIS.json

---

## 2. K1：I vs S → 分支 C；GO_K2=false

执行产物：...\k1_exec\

| seed | native cg | I | S | K0-C | S 的 W |
|---:|---:|---:|---:|---:|---|
| 42 | 110 | 107 | 113 | 126 | False |
| 43 | 110 | 120 | 113 | 133 | False |
| 44 | 110 | 107 | 111 | 127 | False |

要点：
- S 相对 I：change 误差 CI 有利，但相对 native 精度保护不够；三 seed 均未过 W。
- S 平均约 2.5 次 swap（~30% 零 swap）；I 约 9 次。
- **GO_K2=false**；合同动作：结束 I/S 对照，不进 K2、不扫 HGB、不重训生成器。
- 分支决定：**C**（策略 oracle 成功，学习器失败）。见 BRANCH_DECISION.md。

---

## 3. 诊断：对 K0-C 有利动作，S pred_gate 误拒约 82%

目录：<WORKSPACE_ROOT>\K0C_LEARNER_GAP_DIAGNOSTIC_20261019

- 三 seed 合计 oracle 接受动作 1228；S pred_gate 误拒 **1007（0.8200）**。
- 同动作上 GTF 几何 top1 命中约 **0.65%**（misrank 0.9935）。
- 结论：瓶颈在门控/排序与停手，不是动作空间为空。BREAKTHROUGH=false。

---

## 4. 已否方向（勿再开弱变体）

| 方向 | 结果 |
|---|---|
| STRATEGY GTF / GTF_REL（几何 min_D） | BREAKTHROUGH=false；过换或仍远低于 K0-C |
| DEPLOY_S_RANK_MOTION_ONLY（去 pred_gate，保留运动学） | B_all_W=false；去门后覆盖/误差更差，swap~18 |
| HGB 扫参 / K2 / 重训 generator | 合同禁止或已否 |

路径：
- <WORKSPACE_ROOT>\STRATEGY_ONE_FACTOR_GTFREE_RANK_20261019
- <WORKSPACE_ROOT>\DEPLOY_S_RANK_MOTION_ONLY_20261019

---

## 5. original_CAL 迁移补评（本回合收尾）

目录：<WORKSPACE_ROOT>\ORIGINAL_CAL_TRANSFER_EVAL_20261019

- KIN cache：(2879,20,12,2) × seed 42/43/44 已齐（先前诊断轮生成）。
- **身份 / track CI：BLOCKED_IDENTITY**（全部 track_key=?|-1；**未伪造** hierarchical CI；仅点估计）。
- V1：astra2（sklearn 1.4）加载冻结 HGB 选索引；venv-k0 评测。
- K0-C：ORACLE_DIAGNOSTIC_ONLY（用 GT）。

| seed | native cg | V1 | K1-S | K0-C | K0-C W |
|---:|---:|---:|---:|---:|---|
| 42 | 243 | 245 | 248 | 299 | **True** |
| 43 | 243 | 247 | 252 | 297 | **True** |
| 44 | 243 | 252 | 245 | 302 | **True** |

读法：与 outer **同型**——池＋oracle 可达 W；V1 / K1-S 点估计未过 W。不能把 CAL 点估计说成带 track 稳健区间的迁移闭环。

---

## 6. 下一科学问题（standby 后的正题）

**不要再扫距离键 / ON-OFF 同一 pred_gate / 几何弱变体。**

值得设计的下一问：

> 在运动学可行（motion-PASS）的 native→KIN 动作上，**何种 GT-free、可校准的停手 / 接受规则**，能学到接近 K0-C 的稀疏手术式替换，同时保住难例精度？

候选方向（仅设计层，本夜不再开跑）：
- *calibrated stop*：用 holdout 校准「再换一次是否改善集合边际」的阈值；
- 或对 motion-PASS 动作的成对偏好，而非再调 L/ΔA/ΔF 硬门开关。

---

## 7. 产物绝对路径清单

### 主线联合可行性
- <WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_DESIGN_20261018
- <WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018
- <WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\RESULT_ANALYSIS.md
- <WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\GO_K1_REGISTER.json
- <WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\BRANCH_DECISION.md
- <WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\k1_exec\RESULT_ANALYSIS.md
- <WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\k1_exec\GO_K2_REGISTER.json
- <WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\OVERNIGHT_LOG.md
- <WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\OVERNIGHT_DIGEST_20261019.md（本文件）

### 诊断与已否对照
- <WORKSPACE_ROOT>\K0C_LEARNER_GAP_DIAGNOSTIC_20261019
- <WORKSPACE_ROOT>\STRATEGY_ONE_FACTOR_GTFREE_RANK_20261019
- <WORKSPACE_ROOT>\DEPLOY_S_RANK_MOTION_ONLY_20261019

### original_CAL
- <WORKSPACE_ROOT>\ORIGINAL_CAL_TRANSFER_EVAL_20261019
- <WORKSPACE_ROOT>\ORIGINAL_CAL_TRANSFER_EVAL_20261019\RESULT_ANALYSIS.md
- <WORKSPACE_ROOT>\ORIGINAL_CAL_TRANSFER_EVAL_20261019\STATUS.json
- KIN：<WORKSPACE_ROOT>\UNION_SELECTOR_K40_TO_K20_20261018\results\cache_pred\kin_original_CAL_seed{42,43,44}.npy

---

## 8. Standby 声明

- **BREAKTHROUGH=false**
- **不再派新实验方向**
- 资源：无重任务；THREADS=2 Idle 约定保留给用户回来后的显式任务
- 用户醒来可读本文件 + OVERNIGHT_LOG.md
