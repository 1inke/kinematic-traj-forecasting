"""从 V1/improve/V2 结果写中文 RESULT_ANALYSIS.md 与 SUMMARY。"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve().parent
V1 = HERE / "results" / "v1" / "SUMMARY.json"
IMP = HERE / "results" / "improve" / "IMPROVE_SUMMARY.json"
V2 = HERE / "results" / "v2" / "V2_ABLATION.json"
V0 = HERE / "results" / "v0" / "MANIFEST.json"
OUT = HERE / "RESULT_ANALYSIS.md"
SUM = HERE / "SUMMARY.md"


def pct(x):
    return f"{100*x:.2f}%"


def main():
    lines = []
    lines.append("# 互补候选池验证：执行分析报告")
    lines.append("")
    lines.append(f"日期：{datetime.now().strftime('%Y-%m-%d')}。目录：`COMPLEMENTARY_POOL_VALIDATION_20261018`。")
    lines.append("对照合同：`GROK_KINEMATIC_REVIEW_20261018/NEXT_VALIDATION_DESIGN.md`。")
    lines.append("**不改写**旧 UNION_SELECTOR / KIN / P1 数字。")
    lines.append("")

    if V0.exists():
        v0 = json.loads(V0.read_text(encoding="utf-8"))
        lines += [
            "## V0 证据冻结",
            f"- 状态：{v0.get('status')}",
            f"- 原选择器重放：`{v0.get('replay_status')}`",
            f"- 无 generator 训练：{v0.get('no_generator_training')}",
            "- 结论：缺少原 HGB / selected_indices / paths → 转入 **V1 冻结重现**。",
            "",
        ]
    else:
        lines += ["## V0", "- 未找到 MANIFEST。", ""]

    if not V1.exists():
        lines += ["## V1", "- **尚未完成**（SUMMARY.json 缺失）。", ""]
        OUT.write_text("\n".join(lines), encoding="utf-8")
        SUM.write_text("V1 未完成\n", encoding="utf-8")
        print("partial report written")
        return

    agg = json.loads(V1.read_text(encoding="utf-8"))
    dec = agg["decision"]
    outer = agg["tables"]["outer"]
    prim = outer["cond_boost_11_9"]["overall"]
    nat = outer["native20"]["overall"]
    ch_p = outer["cond_boost_11_9"].get("change", {})
    ch_n = outer["native20"].get("change", {})

    lines += [
        "## V1 锁定重现（cond_boost_11_9）",
        "",
        "固定：HGB 超参、二元 good 标签、11/9 + boost0.35、稳定并列（分数↓ / slot↑）。",
        "FIT 内按 track 划出校准集；**未**用 outer 重选主方法。",
        "",
        "### outer 主表（三 seed 均值）",
        "",
        "| 组合 | 覆盖 | minADE | minFDE | motion_pass |",
        "|---|---:|---:|---:|---:|",
    ]
    for port in ("native20", "kin20", "union40", "cond_top_11_9", "cond_boost_11_9"):
        o = outer[port]["overall"]
        lines.append(
            f"| {port} | {o['coverage_mean']:.4f} | {o['minADE_mean']:.5f} | {o['minFDE_mean']:.5f} | {o['motion_pass']}/3 |"
        )

    lines += [
        "",
        f"- 覆盖相对 native：**{dec['coverage_lift_pp']:+.2f} pp**",
        f"- ADE 是否恶化：{dec['ade_worsened']}；FDE 是否恶化：{dec['fde_worsened']}",
        f"- 判定标签：**{dec['label']}**",
        f"- 全部 seed×group 五运动学门：{dec['motion_all_seed_group_pass']}",
        "",
    ]
    if dec.get("motion_fails"):
        lines.append("运动学失败明细：")
        for f in dec["motion_fails"]:
            lines.append(f"- seed{f['seed']} {f['group']}: {f['fails']} ratios={f['ratios']}")
        lines.append("")

    if ch_p:
        lines += [
            "### 变化组（change）",
            f"- native 覆盖均值：{ch_n.get('coverage_mean')}",
            f"- 选择器覆盖均值：{ch_p.get('coverage_mean')}",
            f"- rescue/lost per seed：{ch_p.get('rescue')} / {ch_p.get('lost')}",
            f"- motion：{ch_p.get('motion_status')}",
            "",
        ]

    ci = agg.get("ci_exploratory", {})
    lines += ["### 探索性 CI（family8 Bonferroni；outer 曾参与历史选法）", ""]
    for k, v in ci.items():
        if not isinstance(v, dict):
            continue
        lines.append(f"- **{k}**: mean={v.get('mean')} ci_family8={v.get('ci_family8')} valid={v.get('valid')}")
    lines.append("")

    # improve
    if IMP.exists():
        imp = json.loads(IMP.read_text(encoding="utf-8"))
        lines += [
            "## 自改进（FIT_cal 选臂，outer 仅确认）",
            f"- 选中：`{imp.get('chosen_on_FIT_cal')}`",
            f"- 规则：{imp.get('selection_rule')}",
            "",
            "### FIT_cal 排名",
        ]
        for r in imp.get("ranking_FIT_cal", []):
            lines.append(
                f"- {r['method']}: cov={r['FIT_cal_cov']:.4f} ade={r['FIT_cal_ade']:.4f} lost={r['FIT_cal_lost']:.1f} ch_pass={r['FIT_cal_change_motion_pass']} score={r['select_score']:.2f}"
            )
        lines.append("")
        lines.append("### outer 确认")
        for m, c in imp.get("outer_confirmation", {}).items():
            lines.append(
                f"- {m}: cov={c['coverage_mean']:.4f} ade={c['minADE_mean']:.4f} fde={c['minFDE_mean']:.4f} lost={c['lost_mean']:.1f} change_cov={c['change_coverage_mean']:.4f} ch_motion={c['change_motion_pass']}/3"
            )
        lines.append("")

    if V2.exists():
        v2 = json.loads(V2.read_text(encoding="utf-8"))
        lines += ["## V2 条件特征消融", ""]
        for split in ("FIT_cal", "outer"):
            if split not in v2:
                continue
            lines.append(f"### {split}")
            for arm, d in v2[split].items():
                lines.append(
                    f"- {arm}: cov={d['coverage_mean']:.4f} ade={d['minADE_mean']:.4f} lost={d['lost_mean']:.1f} change_cov={d['change_coverage_mean']:.4f}"
                )
            lines.append("")

    lines += [
        "## 结论与边界",
        "",
        "1. 互补池证据保留；K20 选择器若覆盖↑但 ADE/FDE↑ → 覆盖—误差折中，非全面优于 native。",
        "2. 必须看 seed×group 运动学；整体 PASS 不能替代 change。",
        "3. 本轮锁定方法、补齐产物；outer CI 为探索性。",
        "4. 未自动扫 quota/boost/epoch；未扩 generator。",
        "5. W0 协同归因仍为可选机制诊断，非本分支前置门。",
        "",
        "## 产物入口",
        "- `results/v0/MANIFEST.json`",
        "- `results/v1/SUMMARY.json` / `artifacts/v1/*`",
        "- `results/improve/IMPROVE_SUMMARY.json`",
        "- `results/v2/V2_ABLATION.json`",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # short summary
    s = [
        f"# SUMMARY",
        f"- V0: {json.loads(V0.read_text(encoding='utf-8')).get('replay_status') if V0.exists() else 'missing'}",
        f"- V1 label: {dec['label']}",
        f"- outer lift: {dec['coverage_lift_pp']:+.2f}pp",
        f"- ADE worsen: {dec['ade_worsened']}, FDE worsen: {dec['fde_worsened']}",
        f"- motion all seed×group: {dec['motion_all_seed_group_pass']}",
    ]
    if IMP.exists():
        s.append(f"- improve chosen: {json.loads(IMP.read_text(encoding='utf-8')).get('chosen_on_FIT_cal')}")
    SUM.write_text("\n".join(s) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
