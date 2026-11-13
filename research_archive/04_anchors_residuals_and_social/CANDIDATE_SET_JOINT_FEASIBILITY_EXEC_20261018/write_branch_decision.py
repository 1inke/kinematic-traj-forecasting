# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime
import json

base = Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018")
k1 = base / "k1_exec"
ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S+09:00")

md = f"""# BRANCH_DECISION (after K1)

- time: {ts} (Asia/Tokyo)
- exec: `{base}`
- K1 artifacts: `{k1}`

## Gates

| item | value |
|---|---|
| GO_K1 | true (K0 outer 3-seed pool + K0-C reached W) |
| GO_K2 | **false** |
| start K2 | **no** |
| BREAKTHROUGH | **false** |

Main fail clauses (from `k1_exec/GO_K2_REGISTER.json`): all seeds miss change ADE/FDE >=5% improve; transition protect fails; bootstrap S-native change coverage lo<=0 and error hi>=0; secondary inner/FIT_cal fail; original_CAL still BLOCKED_NO_KIN_CACHE.

## Design branch: C

> Strategy oracle succeeds; learner fails (`04` section 5C)

Evidence:
1. Pool+policy reachable: K0-C outer change_good ~126/133/127, W=true.
2. Learned arms miss joint gate: I and S both W_pass=false.
3. Local mechanism: S beats I on change error (Bonferroni CI upper <0) but coverage unstable; vs native still harms precision -> not deployable.
4. Behavior: S ~2.5 swaps (~30% zero); I ~9 swaps. S pred_gate rejects >> I (over-conservative, not over-swapping).

Contract action: **end this I/S contrast**; no tree-depth/state/label-weight chase; no K2; no generator retrain; no K0 oracle teacher.

## Gap to K0-C

| seed | native cg | S | I | K0-C | S-native ADE | K0-C-native ADE |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 110 | 113 | 107 | 126 | +0.0037 | -0.0125 |
| 43 | 110 | 113 | 120 | 133 | +0.0094 | -0.0167 |
| 44 | 110 | 111 | 107 | 127 | +0.0069 | -0.0181 |

Set-marginal supervision reduced I/V1-style error collapse, but did **not** learn K0-C style swaps that raise coverage and cut error. Bottleneck is GT-free pred gates / ranking / deploy-state, not empty pool.

## Outer reject totals (3 seeds)

| arm | motion | pred_gate | no_candidate |
|---|---:|---:|---:|
| I | 67708 | 382677 | 5 |
| S | 7254 | 903463 | 175 |

## Overnight follow-up

- Done: K0 secondary, formal K1, RESULT_ANALYSIS, GO_K2=false, this branch note.
- Will **not** start K2 / new run_k1 / hyperparam scan.
- Optional read-only FP/FN later; default standby until user returns.

See: `k1_exec/RESULT_ANALYSIS.md`, `BRANCH_C_DIAGNOSTIC.json`.
"""

(base / "BRANCH_DECISION.md").write_text(md, encoding="utf-8")
(k1 / "BRANCH_DECISION.md").write_text(md, encoding="utf-8")

ptr = f"""# K1 result pointer

- analysis: `k1_exec/RESULT_ANALYSIS.md`
- GO register: `k1_exec/GO_K2_REGISTER.json` -> GO_K2=false
- branch: `BRANCH_DECISION.md` -> Branch C, BREAKTHROUGH=false
- diagnostic: `BRANCH_C_DIAGNOSTIC.json`
- written: {ts} (Asia/Tokyo)
"""
(base / "K1_RESULT_POINTER.md").write_text(ptr, encoding="utf-8")

# ensure Chinese summary also exists
zh = f"""# 分支决策（K1 后）

- 时间：{ts}（Asia/Tokyo）
- **GO_K2 = false**；**不启动 K2**；**BREAKTHROUGH = false**
- 归类：**分支 C**（策略 oracle 成功，学习器失败）

## 要点

1. K0 已证明池+K0-C 可达 W；本轮 I/S 学习选择器都未过联合门。
2. S 相对 I：变化组误差显著更好，但是相对 native 仍伤 ADE/FDE，覆盖提升不稳。
3. S 偏保守（约 2.5 次替换，约 30% 零交换），pred_gate 拒绝远多于 I。
4. 与 K0-C 差距仍大（change_good 111–113 vs 126–133）。
5. 按设计结束 I/S 对照：不扫超参、不启 K2、不重训生成器。

详情见同目录 `BRANCH_DECISION.md` 与 `k1_exec/RESULT_ANALYSIS.md`。
"""
(base / "分支决策_K1后.md").write_text(zh, encoding="utf-8")

log = base / "OVERNIGHT_LOG.md"
with log.open("a", encoding="utf-8") as f:
    f.write(f"\n- [{ts}] K1 DONE GO_K2=False; wrote BRANCH_DECISION (C); no K2; Idle standby\n")
    f.write(f"- [{ts}] BREAKTHROUGH=false; S>I on change-error CI but harms native; large gap to K0-C\n")
(base / "OVERNIGHT_STATUS.txt").write_text(
    "overnight_policy: threads=2 Idle affinity=2cores one_heavy_at_a_time\n"
    f"last: K1 done GO_K2=false Branch=C BREAKTHROUGH=false standby\nupdated: {ts}\n",
    encoding="utf-8",
)
print("OK", ts)
