# 过夜实验报告（联合可行性系列 · 2026-10-19）

- 报告对象：默城 陈
- 系列目录：`<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018`
- 过夜截止：约 2026-10-19 05:37（Asia/Tokyo）
- 摘要来源：`OVERNIGHT_DIGEST_20261019.md`（写于 2026-10-19T05:36:53+09:00）；本报告为其正式展开，不与之矛盾
- **BREAKTHROUGH=false**；状态：**standby**

---

## 1. 摘要与结论

**结论（先说）：BREAKTHROUGH=false；过夜工作已进入 standby，不再自动开新实验方向。**

一句话机制判断：

> native∪KIN 候选在 outer（及 original_CAL 点估计）上，**oracle 可稳定达 W 的 K20 见证（K0 / K0-C）**；学习型/去闸门选取（V1 / K1-I / K1-S / GTF / motion-only）**均未稳定复现该增益**。瓶颈在 **GT-free 停机/接受准则（含 pred_gate 过度拒识）**，不是候选空间为空，也不是再扫超参或重训 generator。

关键事实压缩：

| 判定项 | 结果 |
|---|---|
| K0 outer 联合可达 + K0-C 达 W | 是（3 seed）；GO_K1=true |
| K1 I vs S 联合门 | 全 seed 未过 W；GO_K2=false；分支 **C** |
| 学习器相对 K0-C 的缺口 | S pred_gate 对 oracle 可接受动作误拒约 **82%**；GTF 几何 top1 命中约 **0.65%** |
| 已否方向 | GTF/GTF_REL；motion-only 去 pred_gate；HGB 扫 / K2 / 重训 generator |
| original_CAL | KIN cache 就绪；track CI = **BLOCKED_IDENTITY**（仅点估计）；K0-C W=True×3，V1/K1-S 未达 W |
| 当前姿态 | **standby**；有突破或用户回来再喊 |

---

## 2. 授权与约束

睡觉期间授权边界（按合同与过夜链执行）：

1. **低占用自跑**：THREADS=2、Idle 优先级、亲和约 2 核；不抢占用户白天算力。
2. **不重训 generator**；K0 oracle 仅作可达性诊断（`ORACLE_DIAGNOSTIC_ONLY`），不作 K1 teacher、不授权训练。
3. **有突破或用户回来再喊**：未达突破则写 digest / status 后 **standby**，不开新重负载方向。
4. 合同禁止或已否：HGB 超参扫、K2、重训 generator、把 K0 oracle 当部署策略。

过夜编排：补完 K0 seed44 FIT_cal → 正式 K1 I vs S → 分支 C 决策 → 轻量诊断与一因子策略 → original_CAL 迁移评测 → digest → standby。

---

## 3. 做了什么（按时间线 / 实验块）

时区均为 **Asia/Tokyo**。细节见 `OVERNIGHT_LOG.md`。

| 时段（约） | 实验块 | 动作与结果 |
|---|---|---|
| 00:04–00:15 | K0 收尾 | seed44 FIT_cal 补跑完成；合并 secondary；**GO_K1 保持 true**；n_results=9 |
| 00:15–02:58 | K1 正式 I vs S | `run_k1.py` → `k1_exec\`；中途偶发双进程/中断后以 Idle+亲和重挂；**GO_K2=false** |
| 03:09–03:10 | 分支决策 | 写入 `BRANCH_DECISION.md`：**C**（策略 oracle 成功，学习器失败）；index-overlap 诊断；不启 K2 |
| 03:13–03:22 | 一因子 GT-free 排序 | `STRATEGY_ONE_FACTOR_GTFREE_RANK_20261019`：GTF 绝对 min_D → GTF_REL；**BREAKTHROUGH=false** |
| 03:26 | 学习器缺口诊断 | `K0C_LEARNER_GAP_DIAGNOSTIC_20261019`：pred_gate 误拒 0.8200；GTF misrank 0.9935 |
| ~03:30 | CAL KIN cache | original_CAL KIN cache 状态 OK（形状 (2879,20,12,2)，seed 42/43/44） |
| ~04:45 | 部署变体 | `DEPLOY_S_RANK_MOTION_ONLY_20261019`：去 pred_gate、仅运动学 → **B_all_W=false** |
| 05:36 | original_CAL 迁移 | `ORIGINAL_CAL_TRANSFER_EVAL_20261019`：track CI **BLOCKED_IDENTITY**；点估计表完成 |
| 05:36:53 | 过夜收口 | 写 `OVERNIGHT_DIGEST_20261019.md`；`OVERNIGHT_STATUS.txt`：**standby** |

---

## 4. 关键数字表

以下数字均已核实；勿外推未列出的量。

### 4.1 K0 outer（native∪KIN 池；GO_K1=true；generator 冻结）

| seed | native change_good | V1 | K0-C |
|---:|---:|---:|---:|
| 42 | 110 | 116（W 未过） | **126** |
| 43 | 110 | 119 | **133** |
| 44 | 110 | 110 | **127** |

- 池非空；瓶颈是**选择**，不是生成。
- 形式登记：`GO_K1_REGISTER.json`；分析：`RESULT_ANALYSIS.md` / `.json`。

### 4.2 K1 I vs S（`k1_exec`；GO_K2=false；分支 C）

| seed | native cg | I | S | K0-C | S 过 W |
|---:|---:|---:|---:|---:|---|
| 42 | 110 | 107 | 113 | 126 | False |
| 43 | 110 | 120 | 113 | 133 | False |
| 44 | 110 | 107 | 111 | 127 | False |

- S 平均约 **2.5** 次 swap（约 **30%** 零 swap）；I 约 **9** 次 swap。
- 结束 I/S 对照；**不开 K2**；不扫 HGB；不重训 generator。

### 4.3 学习器缺口（`K0C_LEARNER_GAP_DIAGNOSTIC_20261019`）

| 指标 | 值 |
|---|---|
| 三 seed 合计 oracle 可接受动作 | 1228 |
| 其中被 S pred_gate 误拒 | **1007**（**0.8200**，约 82%） |
| GTF 几何 top1 命中率 | 约 **0.65%**（misrank **0.9935**） |

### 4.4 original_CAL 迁移点估计（`ORIGINAL_CAL_TRANSFER_EVAL_20261019`）

- KIN cache：(2879, 20, 12, 2)，seed 42/43/44 就绪。
- track / 分层 CI：**BLOCKED_IDENTITY**（全部 `track_key=?|-1`）→ **仅点估计**，不可作分层 CI 推断。

| seed | native cg | V1 | K1-S | K0-C | K0-C W |
|---:|---:|---:|---:|---:|---|
| 42 | 243 | 245 | 248 | 299 | **True** |
| 43 | 243 | 247 | 252 | 297 | **True** |
| 44 | 243 | 252 | 245 | 302 | **True** |

机制与 outer **同构**：oracle 可达 W；V1 / K1-S 学习臂未达 W。不能把 CAL 点估计误读成可达 track 比较或闭环迁移成功。

### 4.5 Outer 绝对误差（米；minADE / minFDE）

**Native（三 seed 相同）：**

| 范围 | minADE | minFDE |
|---|---:|---:|
| overall | 0.175478 | 0.296633 |
| change | 0.249147 | 0.429378 |

**K0-C：**

| seed | overall ADE/FDE | change ADE/FDE | change_good |
|---:|---|---|---:|
| 42 | 0.159474 / 0.256509 | 0.236645 / 0.390795 | 126 |
| 43 | 0.158402 / 0.255217 | 0.232437 / 0.384533 | 133 |
| 44 | 0.159187 / 0.259212 | 0.231093 / 0.391509 | 127 |

**对照臂（节选）：**

| 臂 | seed | overall ADE/FDE | change ADE/FDE |
|---|---:|---|---|
| union40 | 42 | 0.155632 / 0.241003 | 0.229487 / 0.367888 |
| K1-S | 42 | 0.175988 / 0.298449 | 0.252874 / 0.451054 |
| K1-S | 43 | 0.174984 / 0.300666 | 0.258521 / 0.466126 |
| K1-S | 44 | 0.176493 / 0.301587 | 0.256030 / 0.457093 |

读法：K0-C 相对 native 同时抬 coverage（change_good）并压 ADE/FDE；K1-S 在 change 段误差相对 native **变差**，且未过 W。

---

## 5. 机制诊断

1. **可达性已证、可部署未证**  
   K0 / K0-C（oracle）在 outer 三 seed 上池非空且策略达 W；说明 native∪KIN 集合里存在能过联合门的 K20。标签 `ORACLE_DIAGNOSTIC_ONLY`：证明上界可达，**不等于**给出可离线部署的选择器。

2. **瓶颈在选择器，不在生成器**  
   Generator 全程冻结。池非空；学习臂（V1、I、S）与几何排序（GTF）均未复现 K0-C 式「稀疏、近 native、同时抬覆盖并降误差」的 swap 集合。

3. **S 过保守，不是过交换**  
   S ≈ 2.5 swaps、约 30% 零 swap；对 oracle 可接受动作 pred_gate 误拒约 82%。I 交换更多（≈9）但仍未过联合门。外层拒识计数（BRANCH 证据）：S 的 pred_gate 拒识量远高于 I，而 motion 拒识反而更少——主病是 **停机/接受闸门假阴性**。

4. **集合结构：K0-C 近 native 的外科式替换；S 稀疏但集合不同**  
   Index overlap（outer）：K0-C 与 native 平均交集 ≈19.6/20，S∩K0-C 完全相同比例仅约 18%。S 与 oracle 选的不是同一批槽位。

5. **几何弱排序几乎无效**  
   GTF top1 命中 ~0.65%（misrank 0.9935）→ 纯距离/相对距离排序无法逼近 oracle 排序。

6. **去闸门会过交换、伤覆盖**  
   motion-only（去 pred_gate）→ swap~18、`B_all_W=false`：闸门不能简单拆掉；需要的是 **校准后的 GT-free 接受**，不是 ON/OFF 同一 pred_gate。

7. **original_CAL 同构、CI 受阻**  
   点估计再现「oracle 达 W、学习臂未达」；因 `track_key` 全为 `?|-1`，分层/track CI 为 **BLOCKED_IDENTITY**，不可据此宣称 CAL 闭环迁移成功。

---

## 6. 已否方向

| 方向 | 结果 | 为何否 |
|---|---|---|
| `STRATEGY_ONE_FACTOR_GTFREE_RANK_20261019`（GTF / GTF_REL，含相对 min_D） | BREAKTHROUGH=false | 更差或远低于 K0-C；过交换或选错槽 |
| `DEPLOY_S_RANK_MOTION_ONLY_20261019`（去 pred_gate，仅运动学） | B_all_W=false；swap~18 | 覆盖/误差变差；证明「拆闸」不可行 |
| HGB 超参扫 / K2 / 重训 generator | 未开或合同禁止 | 合同约束 + 分支 C 已否「追树深/状态/权重」；过夜授权不重训 generator |

---

## 7. 相对项目基线（native F16 K20）优劣

基线：**native F16 → K20**（三 seed outer change_good=110；overall ADE/FDE = 0.175478 / 0.296633）。

| 方法 | 相对 native 的利 | 相对 native 的弊 / 未解决 |
|---|---|---|
| **K0-C（oracle）** | change_good → 126/133/127；overall 与 change ADE/FDE 均下降（例 seed42 change 0.249→0.237） | **不可部署**（需 GT）；仅诊断上界 |
| **V1（条件提升）** | 早期 COMPLEMENTARY_POOL：coverage ≈0.648（native ≈0.604，+4.4pp） | ADE/FDE 约 **+11% / +19%**；本系列 outer 上 W 仍未过（cg 116/119/110） |
| **K1-I** | 集合边际监督减轻 V1 式误差崩（相对） | cg 常低于 native（107/120/107）；未过 W；交换偏多 |
| **K1-S** | 对 I：change 误差 CI 有局部优势；swap 更稀疏 | 相对 native 仍伤精度（change ADE 升）；三 seed 未过 W；距 K0-C 大缺口 |
| **GTF / motion-only** | 验证了「弱几何 / 拆闸」假设 | 明确更差或不过 W |
| **union40（seed42）** | overall/change 误差可更低 | seed43 等 split 上曾出现 motion 失败、非质量上界否定；不能当作可部署默认臂 |

**综合：** 相对 native K20，**唯一同时抬覆盖并压误差的见证是 K0-C oracle**；现有学习/GT-free 臂要么伤误差、要么不过联合门。项目状态是「上界存在、选择器未学会」，不是「池不够大」。

---

## 8. 下一科学问题（未开跑）

**不要再扫：** 距离键 / ON-OFF 同一 pred_gate / 几何弱变体。

**值得排的下一问（standby，过夜未启动）：**

> 在运动学已通过（motion-PASS）的 native→KIN 动作上，设计 **GT-free、可校准的停机 / 接受准则**，使部署选择在无 GT 条件下科学逼近 K0-C 集合，同时守住覆盖与精度。

可选子方向（仅假说层）：

- *calibrated stop*：holdout 校准后，回收一部分被误拒的边际动作（阈值/阈值阈值）。
- 把 motion-PASS 后的排序改成对偏置敏感的形式，避免回到 L/ΔA/ΔF 硬闸门堆叠。

---

## 9. 产物路径清单

### 联合可行性主链

- `<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_DESIGN_20261018`
- `<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018`
- `...\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\RESULT_ANALYSIS.md`（及 `.json`）
- `...\GO_K1_REGISTER.json`
- `...\BRANCH_DECISION.md`
- `...\k1_exec\RESULT_ANALYSIS.md`
- `...\k1_exec\GO_K2_REGISTER.json`
- `...\OVERNIGHT_LOG.md`
- `...\OVERNIGHT_DIGEST_20261019.md`
- `...\OVERNIGHT_REPORT_20261019.md`（本报告）
- `...\OVERNIGHT_STATUS.txt`

### 诊断与已否方向

- `<WORKSPACE_ROOT>\K0C_LEARNER_GAP_DIAGNOSTIC_20261019`
- `<WORKSPACE_ROOT>\STRATEGY_ONE_FACTOR_GTFREE_RANK_20261019`
- `<WORKSPACE_ROOT>\DEPLOY_S_RANK_MOTION_ONLY_20261019`

### original_CAL

- `<WORKSPACE_ROOT>\ORIGINAL_CAL_TRANSFER_EVAL_20261019`
- `...\ORIGINAL_CAL_TRANSFER_EVAL_20261019\RESULT_ANALYSIS.md`
- `...\ORIGINAL_CAL_TRANSFER_EVAL_20261019\STATUS.json`
- KIN：`<WORKSPACE_ROOT>\UNION_SELECTOR_K40_TO_K20_20261018\results\cache_pred\kin_original_CAL_seed{42,43,44}.npy`

### 本报告副本（box）

- `/workspace/CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018/OVERNIGHT_REPORT_20261019.md`

---

## 10. Standby 声明

- **BREAKTHROUGH=false**
- **状态：standby**——不自动开启新实验方向
- 资源约束仍有效（低占用 / Idle）；等待用户回来或明确新授权
- 用户回来请先读：本报告 + `OVERNIGHT_DIGEST_20261019.md` + `OVERNIGHT_LOG.md` + `BRANCH_DECISION.md`

---

*写于过夜收口之后；数字与 digest / 各 STATUS 对齐；展开叙述不改变任何判定。*
