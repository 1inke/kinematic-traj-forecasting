# -*- coding: utf-8 -*-
"""Aggregate short seeds 42/43/44 into SUMMARY.json and RESULTS.md."""
from __future__ import annotations
import json
import time
from pathlib import Path
import statistics as stats

ROOT = Path(r"<WORKSPACE_ROOT>\KINEMATIC_DECODER_EXPLICIT_INTEGRATION_20261018")
SHORT = ROOT / "results" / "short"
SEEDS = (42, 43, 44)


def mean_std(xs):
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(stats.mean(xs)), float(stats.stdev(xs))


def extract(seed: int) -> dict:
    p = SHORT / f"COMPARE_seed{seed}.json"
    if not p.exists() and seed == 42:
        p = SHORT / "COMPARE.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    b, k = d["BASE_outer"], d["KIN_outer"]
    dec = d.get("decision", {})
    return dict(
        seed=seed,
        path=str(p),
        BASE=dict(
            coverage=b["overall"]["coverage"],
            minADE=b["overall"]["minADE"],
            minFDE=b["overall"]["minFDE"],
            rescue=b["overall"]["rescue"],
            lost=b["overall"]["lost"],
            motion=b["_motion"]["gate"]["status"],
            motion_fails=list(b["_motion"]["gate"]["fails"]),
            accel_ratio=b["_motion"]["gate"]["ratios"]["accel_mean"],
            speed_p95_ratio=b["_motion"]["gate"]["ratios"]["speed_p95"],
            jerk_mean_ratio=b["_motion"]["gate"]["ratios"]["jerk_mean"],
        ),
        KIN=dict(
            coverage=k["overall"]["coverage"],
            minADE=k["overall"]["minADE"],
            minFDE=k["overall"]["minFDE"],
            rescue=k["overall"]["rescue"],
            lost=k["overall"]["lost"],
            motion=k["_motion"]["gate"]["status"],
            motion_fails=list(k["_motion"]["gate"]["fails"]),
            accel_ratio=k["_motion"]["gate"]["ratios"]["accel_mean"],
            speed_p95_ratio=k["_motion"]["gate"]["ratios"]["speed_p95"],
            jerk_mean_ratio=k["_motion"]["gate"]["ratios"]["jerk_mean"],
        ),
        native_coverage=b["overall"]["native_coverage"],
        weak_positive=bool(dec.get("weak_positive", False)),
        cov_ok=bool(dec.get("cov_ok", False)),
        ade_ok=bool(dec.get("ade_ok", False)),
        motion_better=bool(dec.get("motion_better", False)),
        artifacts=dict(
            BASE_pt=str(SHORT / f"BASE_seed{seed}.pt"),
            KIN_pt=str(SHORT / f"KIN_seed{seed}.pt"),
            BASE_json=str(SHORT / f"BASE_seed{seed}_RESULT.json"),
            KIN_json=str(SHORT / f"KIN_seed{seed}_RESULT.json"),
            COMPARE=str(p),
        ),
    )


rows = [extract(s) for s in SEEDS]

def arm_agg(arm: str):
    cov = [r[arm]["coverage"] for r in rows]
    ade = [r[arm]["minADE"] for r in rows]
    fde = [r[arm]["minFDE"] for r in rows]
    acc = [r[arm]["accel_ratio"] for r in rows]
    spd = [r[arm]["speed_p95_ratio"] for r in rows]
    cm, cs = mean_std(cov)
    am, as_ = mean_std(ade)
    fm, fs = mean_std(fde)
    acm, acs = mean_std(acc)
    sm, ss = mean_std(spd)
    return dict(
        coverage_mean=cm,
        coverage_std=cs,
        minADE_mean=am,
        minADE_std=as_,
        minFDE_mean=fm,
        minFDE_std=fs,
        accel_ratio_mean=acm,
        accel_ratio_std=acs,
        speed_p95_ratio_mean=sm,
        speed_p95_ratio_std=ss,
        motion_PASS_count=sum(1 for r in rows if r[arm]["motion"] == "PASS"),
        per_seed_motion=[r[arm]["motion"] for r in rows],
        per_seed_fails=[r[arm]["motion_fails"] for r in rows],
    )


delta_cov = [r["KIN"]["coverage"] - r["BASE"]["coverage"] for r in rows]
delta_ade_rel = [r["KIN"]["minADE"] / r["BASE"]["minADE"] for r in rows]
dcm, dcs = mean_std(delta_cov)
arm, ars = mean_std(delta_ade_rel)

summary = dict(
    status="EXECUTED_3SEED",
    phase="short",
    seeds=list(SEEDS),
    protocol=dict(epochs=8, batch=256, subset=None, K=20, H=12, dt=0.4),
    FACTS=dict(
        per_seed=[{
            "seed": r["seed"],
            "native_coverage": r["native_coverage"],
            "BASE": r["BASE"],
            "KIN": r["KIN"],
            "weak_positive": r["weak_positive"],
            "cov_ok": r["cov_ok"],
            "ade_ok": r["ade_ok"],
            "motion_better": r["motion_better"],
            "artifacts": r["artifacts"],
        } for r in rows],
        aggregate=dict(
            BASE=arm_agg("BASE"),
            KIN=arm_agg("KIN"),
            KIN_minus_BASE_coverage_pp_mean=dcm * 100,
            KIN_minus_BASE_coverage_pp_std=dcs * 100,
            KIN_over_BASE_minADE_mean=arm,
            KIN_over_BASE_minADE_std=ars,
            weak_positive_count=sum(1 for r in rows if r["weak_positive"]),
            all_seeds_cov_ok=all(r["cov_ok"] for r in rows),
            all_seeds_ade_ok=all(r["ade_ok"] for r in rows),
        ),
    ),
    INFERENCE=dict(
        pre_registered_weak_positive_any=any(r["weak_positive"] for r in rows),
        note_zh=(
            "三 seed 上 KIN 相对 BASE 的 coverage 与 minADE 改善方向一致；"
            "严格运动学门仍未过（各 seed KIN 均 FAIL，失败项在 speed_p95 或 accel_mean 上略超 1.01）。"
            "accel/jerk 相对 BASE 明显更接近或低于 native。这支持结构偏置有助于平滑与误差，"
            "但不构成预注册弱阳性通过，也不能外推模式保持或 P1。"
        ),
        recommended_next=(
            "可选：新设计号速度软限幅专治 speed_p95；或停止本结构版本。"
            "不扫一致性权重/隐层；不接 P1/P2。"
        ),
    ),
    at=time.strftime("%Y-%m-%dT%H:%M:%S+0900"),
)

(ROOT / "results" / "SUMMARY.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
(SHORT / "AGGREGATE_3SEED.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)

fa = summary["FACTS"]["aggregate"]
lines = []
lines.append("# 结果报告（short 三 seed：42/43/44）")
lines.append("")
lines.append(f"时间：{summary['at']}（Asia/Tokyo）")
lines.append("")
lines.append("## FACTS（事实）")
lines.append("")
lines.append("协议：8 epoch，全 FIT，batch 256，K=20；评价 outer overall；质量 ADE<=0.2 且 FDE<=0.4；motion_factor=1.01。")
lines.append("")
lines.append("### 每 seed outer overall")
lines.append("")
lines.append("| seed | arm | coverage | minADE | minFDE | motion | fails | accel_ratio | speed_p95_ratio | weak_positive |")
lines.append("|---:|---|---:|---:|---:|---|---|---:|---:|---|")
for r in rows:
    for arm in ("BASE", "KIN"):
        a = r[arm]
        fails = ",".join(a["motion_fails"]) if a["motion_fails"] else "-"
        wp = r["weak_positive"] if arm == "KIN" else "-"
        lines.append(
            f"| {r['seed']} | {arm} | {a['coverage']:.4f} | {a['minADE']:.4f} | {a['minFDE']:.4f} | "
            f"{a['motion']} | {fails} | {a['accel_ratio']:.3f} | {a['speed_p95_ratio']:.3f} | {wp} |"
        )
lines.append("")
lines.append(f"native coverage（三 seed 相同分母）= {rows[0]['native_coverage']:.4f}")
lines.append("")
lines.append("### 三 seed 均值 ± 样本标准差")
lines.append("")
lines.append("| arm | coverage | minADE | minFDE | accel_ratio | speed_p95_ratio | motion PASS 数 |")
lines.append("|---|---|---|---|---|---|---:|")
for arm in ("BASE", "KIN"):
    a = fa[arm]
    lines.append(
        f"| {arm} | {a['coverage_mean']:.4f}±{a['coverage_std']:.4f} | "
        f"{a['minADE_mean']:.4f}±{a['minADE_std']:.4f} | "
        f"{a['minFDE_mean']:.4f}±{a['minFDE_std']:.4f} | "
        f"{a['accel_ratio_mean']:.3f}±{a['accel_ratio_std']:.3f} | "
        f"{a['speed_p95_ratio_mean']:.3f}±{a['speed_p95_ratio_std']:.3f} | "
        f"{a['motion_PASS_count']}/3 |"
    )
lines.append("")
lines.append(
    f"KIN−BASE coverage：均值 **{fa['KIN_minus_BASE_coverage_pp_mean']:.2f} pp** "
    f"（std {fa['KIN_minus_BASE_coverage_pp_std']:.2f} pp）"
)
lines.append(
    f"KIN/BASE minADE：均值 **{fa['KIN_over_BASE_minADE_mean']:.3f}** "
    f"（std {fa['KIN_over_BASE_minADE_std']:.3f}）"
)
lines.append(f"预注册 weak_positive 通过数：**{fa['weak_positive_count']}/3**")
lines.append(f"全部 seed cov_ok={fa['all_seeds_cov_ok']}；全部 seed ade_ok={fa['all_seeds_ade_ok']}")
lines.append("")
lines.append("### 产物路径")
for r in rows:
    lines.append(f"- seed {r['seed']}: {r['artifacts']['COMPARE']}")
    lines.append(f"  - {r['artifacts']['BASE_json']}")
    lines.append(f"  - {r['artifacts']['KIN_json']}")
lines.append(f"- 汇总：{ROOT / 'results' / 'SUMMARY.json'}")
lines.append("")
lines.append("## INFERENCE（解读，非门控改写）")
lines.append("")
lines.append(summary["INFERENCE"]["note_zh"])
lines.append("")
lines.append("后续：" + summary["INFERENCE"]["recommended_next"])
lines.append("")
lines.append("未改 MODE_PRESERVING/P1；未做 λ/隐层扫参；未跑速度软限幅消融（留待新设计号）。")

md = "\n".join(lines) + "\n"
(ROOT / "RESULTS.md").write_text(md, encoding="utf-8")
(ROOT / "results" / "RESULTS.md").write_text(md, encoding="utf-8")

# refresh NEXT
(ROOT / "NEXT.md").write_text(
    """# 后续（未自动启动）

## 当前状态
short 三 seed（42/43/44）已完成。预注册弱阳性门 **0/3 通过**。

## 可选下一步（需再授权）
1. 新设计号：速度软限幅专治 speed_p95 / 略超的 accel_mean，再单 seed 复测门。
2. 停止本结构版本，把三 seed 事实写入总进度。
3. 不扫一致性权重/隐层；不接 P1/P2。
""",
    encoding="utf-8",
)

print(json.dumps({
    "weak_positive_count": fa["weak_positive_count"],
    "cov_gain_pp_mean": fa["KIN_minus_BASE_coverage_pp_mean"],
    "ade_ratio_mean": fa["KIN_over_BASE_minADE_mean"],
    "KIN_cov": [r["KIN"]["coverage"] for r in rows],
    "BASE_cov": [r["BASE"]["coverage"] for r in rows],
}, indent=2))
print("WROTE SUMMARY+RESULTS")
