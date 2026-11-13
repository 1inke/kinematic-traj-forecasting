# K0 联合可达性 — 完整结果分析

- **时间**：2026-10-18 23:33（Asia/Tokyo）
- **执行目录**：`<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\`
- **标签**：`ORACLE_DIAGNOSTIC_ONLY`（见证/策略 oracle 仅诊断，不可部署）
- **分流结论**：**GO_K1 = 是**（理由：`outer_3seed_pool_and_K0C_W`）
- **生成器**：冻结，本轮不训练

配套：`RESULT_ANALYSIS.json`、`GO_K1_REGISTER.json`

---

## 1. 一句话结论

**现有 native＋KIN 候选池，在 outer 三 seed 上都能组成同时满足覆盖、精度保护与运动学的 K20**；策略级 oracle（K0-C）也达 W。  
因此按设计 §5：**进入 K1（改集合边际质量选择）**，而不是先否定池或重训生成器。

V1 条件提升选择器 **未** 达联合 W：覆盖有升，但变化组 ADE/FDE 恶化，说明「单候选 good 概率 + 固定配额」不足以保住稀有精度支撑。

---

## 2. Outer 三 seed 门禁字段（齐）

| 字段 | seed42 | seed43 | seed44 |
|------|--------|--------|--------|
| pool_joint_witness | true | true | true |
| policy_oracle_witness | true | true | true |
| relaxation_certificate_status | R_INCUMBENT_VERIFIED_W | 同左 | 同左 |
| quality_upper_bound_negation | false | false | false |
| k0a / k0b_R / k0c · W_pass | true | true | true |
| native20 / kin20 / v1 · W_pass | false | false | false |

设计门：**outer 三 seed 池见证 ∧ K0-C 达 W → GO_K1=是**。本轮满足。

---

## 3. Outer 组合对照（N=1035）

### 3.1 变化组好窗数 / ADE / FDE

| 组合 | seed42 ch_good | ADE / FDE | seed43 | ADE / FDE | seed44 | ADE / FDE | W |
|------|----------------|-----------|--------|-----------|--------|-----------|---|
| native20 | 110 | 0.2491 / 0.4294 | 110 | 0.2491 / 0.4294 | 110 | 0.2491 / 0.4294 | ✗ |
| kin20 | 85 | 0.3063 / 0.5646 | 100 | 0.2867 / 0.5255 | 92 | 0.2998 / 0.5544 | ✗ |
| v1_cond_boost_11_9 | 116 | 0.3209 / 0.6343 | 119 | 0.3179 / 0.6195 | 110 | 0.3166 / 0.6239 | ✗ |
| union40 | 128 | 0.2295 / 0.3679 | 135 | 0.2266 / 0.3669 | 129 | 0.2258 / 0.3643 | ✓/✗/✓* |
| k0a | 128 | 0.2295 / 0.3679 | 135 | 0.2266 / 0.3669 | 129 | 0.2258 / 0.3643 | ✓ |
| k0b_R | 128 | 0.2353 / 0.3915 | 135 | 0.2349 / 0.3945 | 129 | 0.2339 / 0.3910 | ✓ |
| **k0c** | **126** | **0.2366 / 0.3908** | **133** | **0.2324 / 0.3845** | **127** | **0.2311 / 0.3915** | **✓** |

\* seed43 的 union40：`W_pass=false`（`change_motion=FAIL`），见 §5。

### 3.2 总体好窗数（overall_good）

| 组合 | 42 | 43 | 44 |
|------|----|----|-----|
| native20 | 625 | 625 | 625 |
| v1 | 667 | 674 | 671 |
| k0c | 716 | 717 | 720 |
| k0a | 721 | 723 | 724 |

相对 native：K0-C 总体好窗约 +91～+95；变化组好窗约 +16～+23。  
相对 V1：K0-C 变化组好窗更高，且 ADE/FDE **优于** native，而 V1 变化组 ADE/FDE 相对 native 恶化约 27%～45%（与既有新结果分析一致）。

### 3.3 解读

1. **池有互补价值且可压缩到 K20**：k0a≈union 的覆盖/误差上界，k0c 在略降一点覆盖后仍过 W，并保住精度。
2. **V1 未过 W**：不是「完全没进展」，而是联合约束下不够——覆盖上升不能抵消变化组精度损失。
3. **kin20 单独不够**：变化组覆盖与运动学都弱于联合选型。

---

## 4. 三字段分列（K0 证书）

对 outer 每一 seed：

| 字段 | 含义 | 结果 |
|------|------|------|
| **pool_joint_witness** | 池内是否存在达 W 的 K20 | 三 seed = true |
| **policy_oracle_witness** | 策略 oracle（K0-C）是否找到达 W 的 K20 | 三 seed = true |
| **relaxation_certificate_status** | 松弛/在位解证书 | 均为 `R_INCUMBENT_VERIFIED_W` |
| **quality_upper_bound_negation** | 是否严格否证质量上界 | 均为 false |

→ 分流：**有合格 K20 见证 → 生成器先不动，进 K1 改集合选择。**

---

## 5. seed43 union40 运动学 FAIL 的说明（重要）

在 seed43 的 **outer / inner / FIT_cal** 上：

- `union40.W_pass = false`，主因 **`change_motion = FAIL`**（全池 40 条合并评估）；
- 但同一 split 上 **`k0a` / `k0b_R` / `k0c` 的 `W_pass = true`**，且 `change_motion = PASS`。

**含义**：

- 「把 native20∪kin20 共 40 条当作一个大集合做运动学」≠「从池中选出的某一个 K20 的联合可行性」。
- 全池合并可能引入运动学更差的轨迹，拖垮整体运动学统计；**子集选择可以丢掉这些轨迹**。
- 因此：**不能**把 union40 的 FAIL 读成「池不够 / 质量上界被否定」。  
  本轮 `quality_upper_bound_negation` 仍为 **false**。

---

## 6. 副 split 状态（不挡 GO_K1）

| Split | seed42 | seed43 | seed44 | 挡 GO_K1？ |
|-------|--------|--------|--------|------------|
| outer | DONE（见证+W） | DONE | DONE | —（主门） |
| inner | DONE（k0c W） | DONE（k0c W；union motion FAIL） | **IN_PROGRESS / 待齐** | 否 |
| FIT_cal | DONE（k0c W） | DONE（k0c W；union motion FAIL） | **NOT_DONE** | 否 |
| **original_CAL** | — | — | — | **NOT_DONE**；否 |

已完成副 split 的 k0c 摘录（诊断用）：

| | overall_good | change_good | change ADE/FDE | W |
|--|--------------|-------------|----------------|---|
| 42 inner | 1150 | 207 | 0.2431 / 0.4101 | ✓ |
| 42 FIT_cal | 1944 | 301 | 0.2664 / 0.4398 | ✓ |
| 43 inner | 1142 | 211 | 0.2413 / 0.4071 | ✓ |
| 43 FIT_cal | 1940 | 296 | 0.2660 / 0.4376 | ✓ |

`run_k0.py` 在分析撰写时仍可能在跑 seed44/inner；**勿覆盖已有 solver 日志**；缺 `result_fields` 时 resume 跳过已有即可。原 CAL 迁移验证本轮未做，记入后续。

---

## 7. 与既有「新结果分析 / W0」的衔接

- W0 表明：安全且目标有利的修复存在，但学习器未稳定学到；问题在参数化/选择，不是可行域为空。
- 本轮 K0 进一步表明：**在固定池上，联合 W 的 K20 可达**（oracle 见证）。
- 因此下一刀应打在 **集合边际质量选择（K1）**，而不是盲加生成结构或重训生成器。

---

## 8. 分支与下一步

| 项 | 决定 |
|----|------|
| GO_K1 | **是** |
| 训练生成器 | **否（冻结）** |
| K1 内容 | 实现可学习的集合级选择（对照 K0-C oracle）；保留稀有原生支撑 + 真正互补路径 |
| 本回合是否已启动 K1 训练 | **否**（优先交付本分析；用户已授权系列实验，K1 实现可在产物落地后另开） |
| 副 split / original_CAL | 续跑补齐；不挡 GO_K1 |

---

## 9. 产物路径

- 本分析：`...\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\RESULT_ANALYSIS.md`
- 结构化：`...\RESULT_ANALYSIS.json`
- 门禁登记：`...\GO_K1_REGISTER.json`
- 分 seed 明细：`...\results\seed{42,43,44}\{outer,inner,FIT_cal}\result_fields.json`

