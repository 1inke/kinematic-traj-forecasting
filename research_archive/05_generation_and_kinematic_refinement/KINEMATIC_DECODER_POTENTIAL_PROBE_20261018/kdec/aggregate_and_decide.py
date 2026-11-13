# -*- coding: utf-8 -*-
"""汇总 LONG / LONG_RELAX 三 seed，写潜力门结论。"""
from __future__ import annotations
import json, time, statistics as stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LONG = ROOT / "results" / "long"
SEEDS = (42, 43, 44)
VARIANTS = ("LONG", "LONG_RELAX")


def mean_std(xs):
    if not xs:
        return None, None
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(stats.mean(xs)), float(stats.stdev(xs))


def load_variant(tag: str):
    rows = []
    for s in SEEDS:
        p = LONG / f"COMPARE_{tag}_seed{s}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        kin = d["splits"]["KIN"]
        rows.append(dict(
            seed=s,
            path=str(p),
            decision=d["decision"],
            outer=kin["outer"],
            FIT=kin.get("FIT"),
            inner=kin.get("inner"),
            original_CAL=kin.get("original_CAL"),
            migration=kin.get("migration"),
            clamp=d.get("clamp"),
            mid_evals=d.get("KIN_mid_evals"),
        ))
    return rows


def gate_variant(rows):
    if len(rows) < 3:
        return dict(incomplete=True, n=len(rows))
    A = sum(1 for r in rows if r["decision"].get("seed_gate_A"))
    covs = [r["outer"]["coverage"] for r in rows]
    native = rows[0]["decision"]["native_coverage"]
    mean_cov, std_cov = mean_std(covs)
    B = bool(mean_cov is not None and mean_cov >= native)
    motion_pass = sum(1 for r in rows if r["outer"]["motion"] == "PASS")
    ades = [r["outer"]["minADE"] for r in rows]
    gaps = [r["decision"]["gap_vs_native_pp"] for r in rows]
    mig = []
    for r in rows:
        if r.get("migration"):
            mig.append(r["migration"]["FIT_minus_outer_cov_pp"])
    mig_mean, mig_std = mean_std(mig) if mig else (None, None)
    migration_ok = bool(mig) and all(abs(x) <= 3.0 for x in mig)
    hit = (A >= 2) or B
    ceiling = (not hit) and (mean_cov is not None) and (mean_cov < native - 0.02) and (motion_pass >= 2)
    return dict(
        n=len(rows),
        gate_A_count=A,
        gate_A=A >= 2,
        gate_B=B,
        gate_C_migration_ok=migration_ok,
        potential_gate_HIT=hit,
        ceiling_under_recipe=ceiling,
        motion_PASS=motion_pass,
        coverage_mean=mean_cov,
        coverage_std=std_cov,
        minADE_mean=mean_std(ades)[0],
        minADE_std=mean_std(ades)[1],
        native_coverage=native,
        gap_vs_native_pp_mean=mean_std(gaps)[0],
        gap_vs_native_pp_std=mean_std(gaps)[1],
        FIT_minus_outer_cov_pp_mean=mig_mean,
        FIT_minus_outer_cov_pp_std=mig_std,
    )


def main():
    by_var = {}
    for tag in VARIANTS:
        rows = load_variant(tag)
        by_var[tag] = dict(per_seed=rows, aggregate=gate_variant(rows))

    # optional BASE ref
    base_ref = None
    bp = LONG / "COMPARE_BASE_REF_seed42.json"
    if bp.exists():
        base_ref = json.loads(bp.read_text(encoding="utf-8"))

    any_hit = any(by_var[t]["aggregate"].get("potential_gate_HIT") for t in VARIANTS if by_var[t]["aggregate"].get("n", 0) >= 3)
    all_ceiling = all(by_var[t]["aggregate"].get("ceiling_under_recipe") for t in VARIANTS if by_var[t]["aggregate"].get("n", 0) >= 3)

    best = None
    best_cov = -1
    for t in VARIANTS:
        ag = by_var[t]["aggregate"]
        if ag.get("coverage_mean") is not None and ag["coverage_mean"] > best_cov:
            best_cov = ag["coverage_mean"]
            best = t

    summary = dict(
        status="EXECUTED",
        seeds=list(SEEDS),
        variants=by_var,
        BASE_REF=base_ref["splits"] if base_ref else None,
        STOP=dict(
            potential_gate_HIT=bool(any_hit),
            ceiling_under_recipe=bool(all_ceiling and not any_hit),
            best_variant=best,
            best_coverage_mean=best_cov if best else None,
        ),
        INFERENCE={},
        at=time.strftime("%Y-%m-%dT%H:%M:%S+0900"),
    )

    # Chinese one-liner
    if any_hit:
        note = f"潜力门命中（best={best}）：更长协议下 KIN+clamp 可达到/超过 native 覆盖。"
    elif all_ceiling:
        note = (
            f"潜力门未命中；两变体均平台低于 native≥2pp（best={best} mean_cov={best_cov:.4f} "
            f"vs native≈{by_var[best]['aggregate']['native_coverage']:.4f}）。本配方下冲 native 的上行空间有限。"
        )
    else:
        note = f"已跑完汇总；best={best} mean_cov={best_cov:.4f}。见 RESULTS。"
    summary["INFERENCE"]["note_zh"] = note

    (ROOT / "results" / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# 结果报告 POTENTIAL_PROBE",
        "",
        f"时间：{summary['at']}（Asia/Tokyo）",
        "",
        "## 预注册门",
        "- (A) seed：cov≥native 且 motion PASS ≥2/3",
        "- (B) 均值：mean cov ≥ native",
        "- (C) 迁移：|FIT−outer|≤3pp",
        "",
    ]
    for tag in VARIANTS:
        ag = by_var[tag]["aggregate"]
        rows = by_var[tag]["per_seed"]
        lines += [
            f"## 变体 {tag}",
            "",
            f"- potential_gate_HIT: **{ag.get('potential_gate_HIT')}** (A={ag.get('gate_A')} count={ag.get('gate_A_count')}/3; B={ag.get('gate_B')})",
            f"- ceiling_under_recipe: {ag.get('ceiling_under_recipe')}",
            f"- coverage mean±std: {ag.get('coverage_mean'):.4f}±{ag.get('coverage_std'):.4f}  (native {ag.get('native_coverage'):.4f})",
            f"- gap vs native: {ag.get('gap_vs_native_pp_mean'):+.2f}±{ag.get('gap_vs_native_pp_std'):.2f} pp",
            f"- minADE mean: {ag.get('minADE_mean'):.4f}",
            f"- motion PASS: {ag.get('motion_PASS')}/3",
            f"- FIT−outer cov pp: {ag.get('FIT_minus_outer_cov_pp_mean')} (migration_ok={ag.get('gate_C_migration_ok')})",
            "",
            "| seed | outer_cov | native | gap_pp | minADE | motion | FIT_cov | CAL_cov |",
            "|---:|---:|---:|---:|---:|---|---:|---:|",
        ]
        for r in rows:
            cal = r["original_CAL"]["coverage"] if r.get("original_CAL") else float("nan")
            fitc = r["FIT"]["coverage"] if r.get("FIT") else float("nan")
            lines.append(
                f"| {r['seed']} | {r['outer']['coverage']:.4f} | {r['decision']['native_coverage']:.4f} | "
                f"{r['decision']['gap_vs_native_pp']:+.2f} | {r['outer']['minADE']:.4f} | {r['outer']['motion']} | "
                f"{fitc:.4f} | {cal:.4f} |"
            )
        lines.append("")

    lines += [
        f"## STOP",
        f"- potential_gate_HIT: **{summary['STOP']['potential_gate_HIT']}**",
        f"- ceiling_under_recipe: **{summary['STOP']['ceiling_under_recipe']}**",
        f"- best_variant: {summary['STOP']['best_variant']}",
        "",
        "## INFERENCE",
        summary["INFERENCE"]["note_zh"],
        "",
    ]
    md = "\n".join(lines)
    (ROOT / "RESULTS.md").write_text(md, encoding="utf-8")
    (ROOT / "results" / "RESULTS.md").write_text(md, encoding="utf-8")
    print(json.dumps(summary["STOP"], ensure_ascii=False, indent=2))
    print(summary["INFERENCE"]["note_zh"])


if __name__ == "__main__":
    main()
