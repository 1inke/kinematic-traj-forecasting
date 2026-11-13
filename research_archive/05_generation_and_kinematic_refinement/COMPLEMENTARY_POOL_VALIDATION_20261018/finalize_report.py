# -*- coding: utf-8 -*-
import json
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve().parent
V0 = json.loads((HERE/"results/v0/MANIFEST.json").read_text(encoding="utf-8"))
V1 = json.loads((HERE/"results/v1/SUMMARY.json").read_text(encoding="utf-8"))
IMP = json.loads((HERE/"results/improve/IMPROVE_SUMMARY.json").read_text(encoding="utf-8"))
V2 = json.loads((HERE/"results/v2/V2_ABLATION.json").read_text(encoding="utf-8"))
Q = json.loads((HERE/"results/improve_quota/QUOTA_IMPROVE_SUMMARY.json").read_text(encoding="utf-8"))
A = json.loads((HERE/"results/improve_adaptive/ADAPTIVE_SUMMARY.json").read_text(encoding="utf-8"))

dec = V1["decision"]
outer = V1["tables"]["outer"]
nat = outer["native20"]["overall"]
prim = outer["cond_boost_11_9"]["overall"]
ch_p = outer["cond_boost_11_9"]["change"]
ch_n = outer["native20"]["change"]

lines = []
lines += [
"# 互补候选池验证：完整执行分析报告",
"",
f"日期：{datetime.now().strftime('%Y-%m-%d %H:%M')}（Asia/Tokyo）。",
"目录：`COMPLEMENTARY_POOL_VALIDATION_20261018`。",
"对照合同：`GROK_KINEMATIC_REVIEW_20261018/NEXT_VALIDATION_DESIGN.md`。",
"**不改写**旧 UNION_SELECTOR / KIN / P1 数字。",
"",
"## 执行清单状态",
"",
"| 阶段 | 状态 |",
"|---|---|",
f"| V0 证据冻结 | 完成 → `{V0.get('replay_status')}` |",
"| V1 锁定重现 cond_boost_11_9 | **完成**（产物齐全） |",
"| improve（kinsafe/natunique） | **完成**；FIT_cal 选 `kinsafe_11_9` |",
"| V2 条件特征 vs geometry | **完成** |",
"| 自改进配额 11–14 | **完成**；FIT_cal 选 `boost_14_6`（探索） |",
"| 自改进自适应配额 | **完成**；FIT_cal 仍偏好 `kinsafe_11_9` |",
"",
"## V0",
f"- replay：`{V0.get('replay_status')}`",
f"- 无 generator 训练：{V0.get('no_generator_training')}",
"- 缺原 HGB / selected_paths → 转入 V1 冻结重建。",
"",
"## V1 锁定主方法（cond_boost_11_9）",
"",
"固定：HGB 超参、good 二值标签、11 native + 9 KIN、boost=0.35、稳定并列。",
"FIT 内按 track 划校准集；**未**用 outer 重选主方法。",
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

ade_pct = (prim["minADE_mean"] / nat["minADE_mean"] - 1) * 100
fde_pct = (prim["minFDE_mean"] / nat["minFDE_mean"] - 1) * 100
lines += [
    "",
    f"- 覆盖相对 native：**{dec['coverage_lift_pp']:+.2f} pp**（{nat['coverage_mean']:.4f} → {prim['coverage_mean']:.4f}）",
    f"- minADE 相对 native：**{ade_pct:+.2f}%**（恶化={dec['ade_worsened']}）",
    f"- minFDE 相对 native：**{fde_pct:+.2f}%**（恶化={dec['fde_worsened']}）",
    f"- 判定标签：**{dec['label']}**",
    f"- 全部 seed×group 五运动学门：{dec['motion_all_seed_group_pass']}（fails={dec['motion_fails']}）",
    f"- major_deploy_gate：{dec['major_deploy_gate']}（探索性，非独立验证）",
    "",
    "### 变化组 change（outer）",
    f"- native 覆盖均值：{ch_n['coverage_mean']:.4f}",
    f"- 选择器覆盖均值：{ch_p['coverage_mean']:.4f}",
    f"- rescue/lost per seed：{ch_p['rescue']} / {ch_p['lost']}",
    f"- motion：{ch_p['motion_status']}",
    "",
    "### 探索性 CI（family8）",
]
for k, v in V1.get("ci_exploratory", {}).items():
    if isinstance(v, dict):
        lines.append(
            f"- **{k}**: mean={v.get('mean')} ci_family8={v.get('ci_family8')} valid={v.get('valid')}"
        )

lines += ["", "## improve（FIT_cal 选臂）", f"- 选中：`{IMP['chosen_on_FIT_cal']}`", f"- 规则：{IMP['selection_rule']}", "", "### FIT_cal 排名"]
for r in IMP["ranking_FIT_cal"]:
    lines.append(
        f"- {r['method']}: cov={r['FIT_cal_cov']:.4f} ade={r['FIT_cal_ade']:.4f} lost={r['FIT_cal_lost']:.1f} ch_pass={r['FIT_cal_change_motion_pass']} score={r['select_score']:.2f}"
    )
lines.append("")
lines.append("### outer 确认")
for m, c in IMP["outer_confirmation"].items():
    lines.append(
        f"- {m}: cov={c['coverage_mean']:.4f} ade={c['minADE_mean']:.4f} fde={c['minFDE_mean']:.4f} lost={c['lost_mean']:.1f} rescue={c['rescue_mean']:.1f} change_cov={c['change_coverage_mean']:.4f} ch_motion={c['change_motion_pass']}/3 ov_motion={c['overall_motion_pass']}/3"
    )

lines += ["", "## V2 条件特征消融", ""]
for split in ("FIT_cal", "outer"):
    lines.append(f"### {split}")
    for arm, d in V2[split].items():
        lines.append(
            f"- {arm}: cov={d['coverage_mean']:.4f} ade={d['minADE_mean']:.4f} lost={d['lost_mean']:.1f} change_cov={d['change_coverage_mean']:.4f}"
        )
    lines.append("")
c_cov = V2["outer"]["cond_features"]["coverage_mean"] - V2["outer"]["geometry_only"]["coverage_mean"]
lines.append(f"- outer 条件特征相对 geometry_only 覆盖差：**{c_cov*100:+.2f} pp**（条件特征略优；容量不同，解释有局限）")
lines.append("")

lines += [
    "## 自改进：固定配额（更多原生槽）",
    f"- FIT_cal 选中：`{Q['chosen_on_FIT_cal']}`（探索；**不**替代锁定主方法）",
    "",
    "### outer 对照（关键臂）",
    "| 方法 | 覆盖 | minADE | minFDE | lost | rescue | change_cov | ch_motion |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for m in ("boost_11_9", "boost_12_8", "boost_13_7", "boost_14_6", "kinsafe_11_9", "kinsafe_12_8", "kinsafe_13_7"):
    c = Q["outer_confirmation"][m]
    lines.append(
        f"| {m} | {c['coverage_mean']:.4f} | {c['minADE_mean']:.4f} | {c['minFDE_mean']:.4f} | {c['lost_mean']:.1f} | {c['rescue_mean']:.1f} | {c['change_coverage_mean']:.4f} | {c['change_motion_pass']}/3 |"
    )
lines += [
    "",
    "读法：把原生槽从 11→14，**lost 与 ADE/FDE 明显改善**，但覆盖从 ~0.648 降到 ~0.639（仍高于 native 0.604）。",
    "这是覆盖—误差—丢失折中面，不是单点最优。",
    "",
    "## 自改进：自适应原生配额",
    f"- FIT_cal 选中：`{A['chosen_on_FIT_cal']}`",
    "",
    "| 方法 | outer 覆盖 | minADE | lost | change_cov |",
    "|---|---:|---:|---:|---:|",
]
for m, c in A["outer_confirmation"].items():
    lines.append(
        f"| {m} | {c['coverage_mean']:.4f} | {c['minADE_mean']:.4f} | {c['lost_mean']:.1f} | {c['change_coverage_mean']:.4f} |"
    )
lines += [
    "",
    "自适应配额落在 11/9 与 12/8 之间：覆盖接近锁定，lost/ADE 略好；FIT_cal 仍略偏 `kinsafe_11_9`。",
    "",
    "## 总结论",
    "",
    "1. **互补池成立**：union40 覆盖 0.698 vs native 0.604（+9.4pp）；K20 选择器锁定方法约 **+4.4pp**。",
    "2. **覆盖—误差折中**：锁定 11/9 覆盖↑，但 ADE/FDE 相对 native 恶化；非全面优于 native。",
    "3. **运动学**：本 V1 重建下 outer 全部 seed×group（含 change）**PASS**（补齐了原报告只看 overall 的缺口）。",
    "4. **条件特征**略优于 geometry-only；值得保留，但非戏剧性。",
    "5. **保留原生支撑**可操作：提高原生配额（14/6）或 kinsafer，能降 lost/ADE，代价是覆盖部分回吐。",
    "6. **边界**：outer 曾参与历史选法 → 探索性阳性；未扩 generator、未大扫参；W0 非本分支前置门。",
    "7. **建议**：主方法继续锁定 `cond_boost_11_9` 作报告基线；若产品更厌 lost/尾部误差，可在独立验证集上预注册比较 `kinsafe_11_9` 与 `boost_12_8`/`14_6`，勿用本次 outer 改主方法。",
    "",
    "## 产物入口",
    "- `results/v0/MANIFEST.json`",
    "- `results/v1/SUMMARY.json` + `artifacts/v1/*_bundle.npz` / `hgb_seed*.joblib`",
    "- `results/improve/IMPROVE_SUMMARY.json`",
    "- `results/v2/V2_ABLATION.json`",
    "- `results/improve_quota/QUOTA_IMPROVE_SUMMARY.json`",
    "- `results/improve_adaptive/ADAPTIVE_SUMMARY.json`",
    "- 本文件 `RESULT_ANALYSIS.md` / `SUMMARY.md` / `STATUS.md` / `CHECKLIST.md`",
]
(HERE / "RESULT_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

sum_lines = [
    "# SUMMARY",
    f"- V0: {V0.get('replay_status')}",
    f"- V1 label: {dec['label']}",
    f"- outer lift: {dec['coverage_lift_pp']:+.2f}pp (native {nat['coverage_mean']:.4f} → primary {prim['coverage_mean']:.4f})",
    f"- ADE/FDE worsen vs native: {dec['ade_worsened']}/{dec['fde_worsened']} ({ade_pct:+.1f}% / {fde_pct:+.1f}%)",
    f"- motion all seed×group: {dec['motion_all_seed_group_pass']}",
    f"- improve FIT_cal chosen: {IMP['chosen_on_FIT_cal']}",
    (
        f"- quota FIT_cal chosen (exploratory): {Q['chosen_on_FIT_cal']} "
        f"outer cov={Q['outer_confirmation'][Q['chosen_on_FIT_cal']]['coverage_mean']:.4f} "
        f"ade={Q['outer_confirmation'][Q['chosen_on_FIT_cal']]['minADE_mean']:.4f} "
        f"lost={Q['outer_confirmation'][Q['chosen_on_FIT_cal']]['lost_mean']:.1f}"
    ),
    f"- adaptive FIT_cal chosen: {A['chosen_on_FIT_cal']}",
    f"- V2 outer cond-geo cov delta: {c_cov*100:+.2f}pp",
    "- status: DESIGN_COMPLETE + SELF_IMPROVE_DONE",
]
(HERE / "SUMMARY.md").write_text("\n".join(sum_lines) + "\n", encoding="utf-8")

(HERE / "STATUS.md").write_text(
    """# 执行状态

更新：2026-10-18（Asia/Tokyo）

## 标记：`DESIGN_COMPLETE` + `SELF_IMPROVE_DONE`

| 阶段 | 状态 |
|---|---|
| V0 | COMPLETE → ORIGINAL_SELECTOR_REPLAY_UNAVAILABLE |
| V1 | COMPLETE（3 seed × outer/inner/FIT_cal/FIT_train；产物齐全） |
| improve | COMPLETE；FIT_cal → kinsafer_11_9 |
| V2 | COMPLETE；cond_features 略优 |
| quota improve | COMPLETE；FIT_cal → boost_14_6（探索，不改主方法） |
| adaptive improve | COMPLETE；FIT_cal → kinsafer_11_9 |
| 报告 | RESULT_ANALYSIS.md / SUMMARY.md 已写 |

主方法仍锁定 **cond_boost_11_9**。自改进只作 FIT_cal 选择 + outer 确认，未回头用 outer 改主方法。
""",
    encoding="utf-8",
)

(HERE / "CHECKLIST.md").write_text(
    """# 执行清单（对照 NEXT_VALIDATION_DESIGN）

日期：2026-10-18。主合同：GROK_KINEMATIC_REVIEW_20261018/NEXT_VALIDATION_DESIGN.md

## V0 — 仅补证据与保存算术，不训练
- [x] 输入冻结 manifest
- [x] 查找原 HGB / 40scores / selected_indices / selected_paths
- [x] 若缺失：写 `ORIGINAL_SELECTOR_REPLAY_UNAVAILABLE`
- [x] 不静默重训 generator

## V1 — 重建并冻结选择器（生成器不动）
- [x] 固定主方法 `cond_boost_11_9`；不按新结果改主方法
- [x] HGB 超参锁定；二元 good 标签；声明 soft 退化为 good
- [x] FIT 内按完整 target track 划出固定校准部分
- [x] 稳定并列：分数降序，并列按物理 slot 升序
- [x] 每 split/seed 保存完整产物（scores/indices/paths/HGB）
- [x] 参照 native20 / KIN20 / union40 / cond_top_11_9
- [x] 每 group：coverage/rescue/lost、minADE/FDE、J、五 motion、聚簇、五类窗口、delta、lost 归因
- [x] track×seed 层级 CI（family8）；标注探索性

## V2 — 证据闭环后
- [x] 条件 HGB vs geometry-only

## 自改进（合同完成后）
- [x] kinsafer / natunique（FIT_cal 选）
- [x] 固定配额 11–14（保留原生支撑）
- [x] 自适应原生配额
- [x] 小锁定消融，不扩模型、不大扫参
""",
    encoding="utf-8",
)
print("wrote RESULT_ANALYSIS / SUMMARY / STATUS / CHECKLIST")
print((HERE / "SUMMARY.md").read_text(encoding="utf-8"))
