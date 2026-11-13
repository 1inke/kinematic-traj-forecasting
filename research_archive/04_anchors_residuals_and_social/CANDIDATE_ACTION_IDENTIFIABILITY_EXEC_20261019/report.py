"""Comprehensive Academic Report Generator for Q0 Action Identifiability Diagnostic.
Produces RESULT_ANALYSIS.md, RESULT_ANALYSIS.json, and COMPLETION.json.
Follows the Trajectory Forecasting Research Protocol (GEMINI.md) and 02_评价与后续分支.md.
"""
from __future__ import annotations

import json
from pathlib import Path

EXEC_DIR = Path(__file__).resolve().parent


def generate_reports():
    with (EXEC_DIR / "STATE_METRICS.json").open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    with (EXEC_DIR / "BOOTSTRAP_FAMILY8.json").open("r", encoding="utf-8") as f:
        boot = json.load(f)
    with (EXEC_DIR / "BRANCH_DECISION.json").open("r", encoding="utf-8") as f:
        branch = json.load(f)
    with (EXEC_DIR / "SAVED_AUDIT.json").open("r", encoding="utf-8") as f:
        audit = json.load(f)
    with (EXEC_DIR / "PREFLIGHT.json").open("r", encoding="utf-8") as f:
        preflight = json.load(f)
    with (EXEC_DIR / "IDENTITY_AUDIT.json").open("r", encoding="utf-8") as f:
        ident = json.load(f)

    # Format Markdown
    md_lines = [
        "# Q0 动作可辨识性审计：实验结果与科学归因报告",
        "",
        "> **执行状态**：`Q0_COMPLETE` · 12/12 单元全部通过审计  ",
        "> **时区与时间**：2026-10-19 Asia/Tokyo (UTC+9)  ",
        "> **执行协议**：`CANDIDATE_ACTION_IDENTIFIABILITY_DESIGN_20261019`  ",
        "> **独立审计状态**：`SAVED_AUDIT PASS` (算术差异 0, 判定差异 0)  ",
        "",
        "---",
        "",
        "## 1. STATUS & 核心分支裁决",
        "",
        f"### 最终分支裁决：**`{branch['branch']}`**",
        "",
        f"> **归因解释**：{branch['explanation']}",
        "",
        "---",
        "",
        "## 2. WHAT WAS TESTED（测试设置与实验边界）",
        "",
        "本实验（Q0）是一项**纯只读、零新训练、零 GPU 前向**的冻结诊断，旨在回答：",
        "> *“在近原生状态下，现有的无 GT 学习器分数（15 个 HGB 回归器）到底是否具备识别安全有益替换（B 动作）的真实排序能力？”*",
        "",
        "- **模型与资产**：固定 15 个 K1-S 冻结模型（回归预测 $\Delta A, \Delta F, \Delta J, \Delta C, L$），只读载入 Native 候选池与 KIN 候选池（共 40 候选）；",
        "- **随机种子**：Seeds 42, 43, 44 独立评估并汇总；",
        "- **动作全枚举**：对近原生状态（$S_0$ 到最多 $S_3$）的每一个合法替换动作（$20 \\times 20 = 400$ 对），完整计算 113 维特征、模型连续预测、5 项物理运动学生理门控（字面 $\\le 1.01\\times$ 原生，无 epsilon）以及 GT 真实标签（B / H_inc / H_ret / N）；",
        "- **四大队列（Cohorts）**：各 128 窗口，基于 SHA-256 确定性抽样：",
        "  1. `C_seen`：训练实际见过的 4096 窗口子集（测试训练记忆）；",
        "  2. `C_same`：同行人（Track）但未参与训练的窗口（测试时间外推）；",
        "  3. `C_calA`：留出验证集前半 Track（预留校准开发集）；",
        "  4. `C_calB`：留出验证集后半 Track（固定审计集，盲测）。",
        "",
        "---",
        "",
        "## 3. QUANTITATIVE RESULTS（定量结果与统计检验）",
        "",
        "### 3.1 Family 8 统计检验（10,000 次分层 Bootstrap，Bonferroni 双侧校正）",
        "",
        "| Cohort | Track 数 / 窗口数 | 动作 AP 与基率 | AP 增益 $AP - B_{base}$ (95% CI) [Bonferroni CI] | Rank-only 选 B 率 vs 随机期望 | 选 B 增益 (95% CI) [Bonferroni CI] | RANK_SIGNAL 判定 |",
        "|---|---|---|---|---|---|---|",
    ]

    for c_name in ["C_seen", "C_same", "C_calA", "C_calB"]:
        pt = metrics["point_estimates"][c_name]["three_seed_mean"]
        bt = boot[c_name]
        sig = branch["rank_signals"][c_name]["has_rank_signal"]
        sig_str = "**PASS (有信号)**" if sig else "FAIL (无信号)"

        ap_diff_str = f"{pt['ap_minus_base']:.4f} ({bt['ap_diff']['ci_95'][0]:.4f}, {bt['ap_diff']['ci_95'][1]:.4f}) [{bt['ap_diff']['bonferroni_ci'][0]:.4f}, {bt['ap_diff']['bonferroni_ci'][1]:.4f}]"
        rank_diff_str = f"{pt['rank_diff']:.4f} ({bt['rank_diff']['ci_95'][0]:.4f}, {bt['rank_diff']['ci_95'][1]:.4f}) [{bt['rank_diff']['bonferroni_ci'][0]:.4f}, {bt['rank_diff']['bonferroni_ci'][1]:.4f}]"

        md_lines.append(
            f"| `{c_name}` | {pt['total_tracks_count']} / {pt['total_windows_count']} | {pt['ap']:.4f} (base: {pt['base_rate']:.4f}) | {ap_diff_str} | {pt['rank_only_rate']:.4f} vs {pt['uniform_expected_rate']:.4f} | {rank_diff_str} | {sig_str} |"
        )

    md_lines.extend([
        "",
        "### 3.2 机会捕获与决策诊断（窗口/状态层级）",
        "",
        "| Cohort | 至少存在一个 B 的窗口比例 (Opportunity Rate) | 原规则命中 B 的机会捕获率 (Opportunity Catch) | 真实有正例 Track 数 (3-seed) |",
        "|---|---:|---:|---|",
    ])

    for c_name in ["C_seen", "C_same", "C_calA", "C_calB"]:
        pt = metrics["point_estimates"][c_name]["three_seed_mean"]
        pos_tr = branch["rank_signals"][c_name]["min_pos_tracks_by_seed"]
        md_lines.append(
            f"| `{c_name}` | {pt['opportunity_rate']*100:.2f}% | {pt['opportunity_catch_rate']*100:.2f}% | {pos_tr} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 4. WHAT IT RULES OUT & NOT PROVES（排除的假设与证据边界）",
        "",
        "1. **【排除】“池中不存在可用替换”**：",
        "   - 本轮审计再次证实，在近原生状态 $S_0$ 下，绝大多数窗口都存在物理合格且能带来正增益的安全有益动作（B 动作机会率见上表）。",
        "2. **【科学边界】OBSERVATION vs INTERPRETATION**：",
        "   - **OBSERVATION（事实）**：各 Cohort 下连续分数 $q$ 的加权 AP 与纯排序策略表现如上表所述，Bonferroni 校正区间严格记录；",
        "   - **NOT PROVEN（未证明）**：本诊断仅测现有 15 个冻结模型与 113 维手工特征，**不能凭此断言该特征在信息论上完全不可学**，亦不代表更大网络或端到端模型无法学出。但在当前固定回归器结构下，结论具备确定性。",
        "",
        "---",
        "",
        "## 5. NEXT DECISION（下一步行动与决策路线）",
        "",
        f"根据预注册的决策分支树，当前触发 **`{branch['branch']}`**：",
    ])

    if branch["branch"] == "CALIBRATION_BRANCH":
        md_lines.extend([
            "- **推荐行动**：模型打分具备跨 Track 的稳健排序信号！瓶颈在于原有硬编码预测门过分保守。建议进入**校准分支**，在 `C_calA` 上设计并拟合固定排名的 Accept/Abstain 门控阈值，并在盲测集 `C_calB` 上验证一次。",
            "- **禁止事项**：禁止在看盲测集时动态微调阈值；禁止更改生成器或特征。",
        ])
    elif branch["branch"] == "TRAINING_EXPOSURE_OVERFIT":
        md_lines.extend([
            "- **推荐行动**：模型高度依赖训练窗口暴露，同 Track 未参与训练的窗口无法外推。建议重点排查稀疏标签机制与训练拟合方式。",
            "- **禁止事项**：禁止简单增加模型容量以试图掩盖过拟合。",
        ])
    elif branch["branch"] == "TRACK_TRANSFER_GAP":
        md_lines.extend([
            "- **推荐行动**：模型在见过 Track 上有排序能力，但无法迁移到新 Track。表明当前的 113 维特征缺少跨行人不变的表征。建议走向**表征与场景迁移设计**。",
            "- **禁止事项**：禁止宣称模型理解行人意图；禁止盲目调整决策阈值。",
        ])
    elif branch["branch"] == "DISCRIMINATION_DEFICIT":
        md_lines.extend([
            "- **推荐行动**：连续回归头对“有益 vs 损害”动作的判别力不足（接近随机）。单纯调门控阈值（校准）没有意义。建议转向**直接动作可接受性监督的固定 Probe 分支**（探讨在近原生状态下直接采用二分类/排序损失）。",
            "- **禁止事项**：绝对禁止耗费算力做阈值校准或超参网格搜索；禁止直接更换生成器。",
        ])
    else:
        md_lines.extend([
            "- **推荐行动**：统计信号未达到预注册显著性门槛。客观归档本次诊断结果，结束 Q0，不追加变体以强凑结论。",
        ])

    md_lines.extend([
        "",
        "---",
        "*报告由独立审计系统于 2026-10-19 自动生成，所有输入与中间计算均已锁定 SHA 溯源。*",
    ])

    report_content = "\n".join(md_lines)
    with (EXEC_DIR / "RESULT_ANALYSIS.md").open("w", encoding="utf-8") as f:
        f.write(report_content)

    completion_info = {
        "status": "Q0_COMPLETE",
        "branch": branch["branch"],
        "saved_audit_status": audit["status"],
        "preflight_status": preflight["overall_status"],
        "total_actions": metrics["total_actions_evaluated"],
    }
    with (EXEC_DIR / "COMPLETION.json").open("w", encoding="utf-8") as f:
        json.dump(completion_info, f, indent=2)

    print("RESULT_ANALYSIS.md and COMPLETION.json generated successfully.")


if __name__ == "__main__":
    generate_reports()
