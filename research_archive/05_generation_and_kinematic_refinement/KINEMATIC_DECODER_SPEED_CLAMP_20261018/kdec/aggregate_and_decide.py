# -*- coding: utf-8 -*-
"""汇总某 round 的 42/43/44，写 SUMMARY/RESULTS，返回是否达到强结果。"""
from __future__ import annotations
import json, time, statistics as stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHORT = ROOT / "results" / "short"
SEEDS = (42, 43, 44)


def mean_std(xs):
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(stats.mean(xs)), float(stats.stdev(xs))


def main(round_tag="r1"):
    rows = []
    for s in SEEDS:
        p = SHORT / f"COMPARE_{round_tag}_seed{s}.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        b, k, dec = d["BASE_outer"], d["KIN_outer"], d["decision"]
        rows.append(dict(
            seed=s, path=str(p), decision=dec,
            BASE=dict(
                coverage=b["overall"]["coverage"], minADE=b["overall"]["minADE"], minFDE=b["overall"]["minFDE"],
                motion=b["_motion"]["gate"]["status"], fails=b["_motion"]["gate"]["fails"],
                accel_ratio=b["_motion"]["gate"]["ratios"]["accel_mean"],
                speed_ratio=b["_motion"]["gate"]["ratios"]["speed_p95"],
            ),
            KIN=dict(
                coverage=k["overall"]["coverage"], minADE=k["overall"]["minADE"], minFDE=k["overall"]["minFDE"],
                motion=k["_motion"]["gate"]["status"], fails=k["_motion"]["gate"]["fails"],
                accel_ratio=k["_motion"]["gate"]["ratios"]["accel_mean"],
                speed_ratio=k["_motion"]["gate"]["ratios"]["speed_p95"],
            ),
            native_coverage=b["overall"]["native_coverage"],
            clamp=d.get("clamp"),
        ))

    wp = sum(1 for r in rows if r["decision"]["weak_positive"])
    stronger = sum(1 for r in rows if r["decision"].get("stronger"))
    kin_pass = sum(1 for r in rows if r["KIN"]["motion"] == "PASS")

    def agg(arm):
        cov = [r[arm]["coverage"] for r in rows]
        ade = [r[arm]["minADE"] for r in rows]
        acc = [r[arm]["accel_ratio"] for r in rows]
        spd = [r[arm]["speed_ratio"] for r in rows]
        cm, cs = mean_std(cov); am, as_ = mean_std(ade); acm, acs = mean_std(acc); sm, ss = mean_std(spd)
        return dict(coverage_mean=cm, coverage_std=cs, minADE_mean=am, minADE_std=as_,
                    accel_ratio_mean=acm, accel_ratio_std=acs, speed_ratio_mean=sm, speed_ratio_std=ss,
                    motion_PASS=sum(1 for r in rows if r[arm]["motion"]=="PASS"))

    dcov = [r["KIN"]["coverage"]-r["BASE"]["coverage"] for r in rows]
    dade = [r["KIN"]["minADE"]/r["BASE"]["minADE"] for r in rows]
    stop_strong = (wp >= 2) or (stronger >= 2) or (kin_pass >= 2 and all(r["KIN"]["coverage"] >= r["BASE"]["coverage"]-0.01 for r in rows) and all(r["KIN"]["minADE"] <= r["BASE"]["minADE"]*1.10 for r in rows) and kin_pass >= 2)

    summary = dict(
        status="EXECUTED_3SEED",
        round=round_tag,
        seeds=list(SEEDS),
        FACTS=dict(
            per_seed=rows,
            aggregate=dict(
                BASE=agg("BASE"), KIN=agg("KIN"),
                weak_positive_count=wp, stronger_count=stronger, KIN_motion_PASS=kin_pass,
                coverage_gain_pp_mean=mean_std(dcov)[0]*100,
                coverage_gain_pp_std=mean_std(dcov)[1]*100,
                minADE_ratio_mean=mean_std(dade)[0],
                minADE_ratio_std=mean_std(dade)[1],
            ),
        ),
        STOP=dict(
            strong_good_result=bool(stop_strong),
            criterion="weak_positive>=2/3 OR stronger>=2/3 OR (KIN motion PASS>=2/3 with cov/ade advantage)",
            weak_positive_count=wp,
            stronger_count=stronger,
            KIN_motion_PASS=kin_pass,
        ),
        INFERENCE=dict(
            note_zh=(
                f"Round {round_tag}: weak_positive {wp}/3, stronger {stronger}/3, KIN motion PASS {kin_pass}/3。"
                + ("已达比较强的好结果，可停。" if stop_strong else "尚未达强结果门，可进入下一轮干预。")
            )
        ),
        at=time.strftime("%Y-%m-%dT%H:%M:%S+0900"),
    )
    (ROOT / "results" / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "results" / "rounds" / f"{round_tag}_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fa = summary["FACTS"]["aggregate"]
    lines = [
        f"# 结果报告 SPEED_CLAMP round={round_tag}",
        "",
        f"时间：{summary['at']}（Asia/Tokyo）",
        "",
        "## FACTS",
        "",
        "| seed | arm | coverage | minADE | motion | fails | accel_r | speed_r | weak_pos | stronger |",
        "|---:|---|---:|---:|---|---|---:|---:|---|---|",
    ]
    for r in rows:
        for arm in ("BASE", "KIN"):
            a = r[arm]
            fails = ",".join(a["fails"]) if a["fails"] else "-"
            wp_s = r["decision"]["weak_positive"] if arm == "KIN" else "-"
            st_s = r["decision"].get("stronger") if arm == "KIN" else "-"
            lines.append(
                f"| {r['seed']} | {arm} | {a['coverage']:.4f} | {a['minADE']:.4f} | {a['motion']} | {fails} | "
                f"{a['accel_ratio']:.3f} | {a['speed_ratio']:.3f} | {wp_s} | {st_s} |"
            )
    lines += [
        "",
        f"weak_positive **{wp}/3**；stronger **{stronger}/3**；KIN motion PASS **{kin_pass}/3**",
        f"coverage 增益均值 {fa['coverage_gain_pp_mean']:.2f}±{fa['coverage_gain_pp_std']:.2f} pp；"
        f"minADE 比 {fa['minADE_ratio_mean']:.3f}±{fa['minADE_ratio_std']:.3f}",
        "",
        f"## STOP: strong_good_result={stop_strong}",
        "",
        "## INFERENCE",
        summary["INFERENCE"]["note_zh"],
        "",
    ]
    md = "\n".join(lines)
    (ROOT / "RESULTS.md").write_text(md, encoding="utf-8")
    (ROOT / "results" / "RESULTS.md").write_text(md, encoding="utf-8")
    print(json.dumps(summary["STOP"], ensure_ascii=False, indent=2))
    return stop_strong

if __name__ == "__main__":
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else "r1"
    ok = main(tag)
    raise SystemExit(0 if True else 1)  # always exit 0; stop flag in JSON
