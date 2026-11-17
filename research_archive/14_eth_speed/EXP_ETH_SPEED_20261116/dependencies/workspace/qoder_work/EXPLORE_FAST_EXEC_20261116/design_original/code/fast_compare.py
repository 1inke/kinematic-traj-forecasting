"""Compare exploration runs with B0 (same budget, seed, train set) and give a verdict.  numpy only.

Verdict per arm and budget, on the primary metric (default overall coverage, pp):
  promising  : mean paired diff >= 2*sigma/sqrt(n) and every seed has the same sign
  dead       : mean paired diff <= 0  (or the run was STOPPED_FUTILE)
  ambiguous  : anything else -> run one more seed (same budget) before deciding
sigma defaults to the E1 dev noise at 64 epochs (overall 1.1pp, change 1.8pp, B0-vs-B0
coverage-difference SD, E2E_V2_CONTRACT); override with --sigma-overall/--sigma-change once a
budget-specific estimate exists (B0 with two seeds at that budget is printed as a crude check).

--calibrate: checks, per budget, whether T1 and E3_self (seed 42, full-fit runs) reproduce their
64-epoch effects (dev, seed 42: T1 overall +1.66 / change +3.85; E3_self +2.61 / +4.58):
reproduced = same sign and >= 50% of the 64-epoch magnitude on BOTH metrics.
The shortest budget where both reproduce is the recommended exploration budget.

usage: python -I fast_compare.py --out EXPLORE [--primary change_coverage] [--calibrate]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REF64_SEED42 = {"T1": {"coverage": 1.66, "change_coverage": 3.85},
                "E3_self": {"coverage": 2.61, "change_coverage": 4.58}}


def load(out):
    rows = {}
    for p in sorted((Path(out) / "runs").glob("*/result.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        tag = r.get("train_set", "fit")
        val = None
        if r["status"] == "DONE":
            o, c = r["dev"]["overall"], r["dev"].get("change")
            val = dict(coverage=100 * o["coverage"], change_coverage=100 * c["coverage"] if c else None,
                       minADE=o["minADE"], minFDE=o["minFDE"], top1=100 * o["top1_good"])
        rows[(r["arm"], r["epochs"], r["seed"], tag)] = dict(status=r["status"], val=val,
                                                             wall=r["wall_seconds"], stopped=r.get("stopped_at_epoch"))
    return rows


def compare(rows, primary, sig):
    groups = defaultdict(lambda: defaultdict(dict))
    for (arm, E, seed, tag), r in rows.items():
        groups[(E, tag)][arm][seed] = r
    report = []
    for (E, tag), arms in sorted(groups.items()):
        b0 = arms.get("B0", {})
        noise = None
        good_b0 = [s for s, r in b0.items() if r["val"]]
        if len(good_b0) >= 2:
            a, b = good_b0[:2]
            noise = {k: b0[a]["val"][k] - b0[b]["val"][k] for k in ("coverage", "change_coverage")}
        for arm, seeds in sorted(arms.items()):
            if arm == "B0":
                continue
            diffs, stopped = defaultdict(list), False
            for seed, r in seeds.items():
                if r["status"] != "DONE":
                    stopped = True; continue
                if seed not in b0 or not b0[seed]["val"]:
                    continue
                for k in ("coverage", "change_coverage", "minADE", "top1"):
                    if r["val"][k] is not None and b0[seed]["val"][k] is not None:
                        d = r["val"][k] - b0[seed]["val"][k]
                        diffs[k].append(100 * d / b0[seed]["val"][k] if k == "minADE" else d)
            d = diffs.get(primary, [])
            if stopped and not d:
                verdict = "dead (stopped futile)"
            elif not d:
                verdict = "no paired B0 run"
            else:
                m, n = float(np.mean(d)), len(d)
                thr = 2 * sig[primary] / np.sqrt(n)
                same = all(np.sign(x) == np.sign(m) for x in d)
                verdict = "promising" if (m >= thr and same) else ("dead" if m <= 0 else f"ambiguous (need >= {thr:.2f}pp; add a seed)")
            report.append(dict(E=E, train_set=tag, arm=arm, n=len(d), verdict=verdict,
                               **{f"d_{k}": (float(np.mean(v)) if v else None) for k, v in
                                  [(k, diffs.get(k, [])) for k in ("coverage", "change_coverage", "minADE", "top1")]},
                               per_seed_primary=d, b0_seed_noise=noise))
    return report


def calibrate(rows):
    out = {}
    budgets = sorted({E for (_, E, _, tag) in rows if tag == "fit"})
    for E in budgets:
        b0 = rows.get(("B0", E, 42, "fit"))
        res = {}
        for arm, ref in REF64_SEED42.items():
            r = rows.get((arm, E, 42, "fit"))
            if not (b0 and b0["val"] and r and r["val"]):
                res[arm] = None; continue
            d = {k: r["val"][k] - b0["val"][k] for k in ref}
            res[arm] = dict(diff=d, reproduced=all(np.sign(d[k]) == np.sign(ref[k]) and abs(d[k]) >= .5 * abs(ref[k]) for k in ref))
        out[E] = res
    ok = [E for E, r in out.items() if all(v and v["reproduced"] for v in r.values())]
    return dict(per_budget=out, recommended_budget=min(ok) if ok else None,
                rule="same sign and >=50% of the 64-epoch seed-42 magnitude on overall AND change coverage, both T1 and E3_self")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--primary", default="coverage", choices=("coverage", "change_coverage"))
    ap.add_argument("--sigma-overall", type=float, default=1.1)
    ap.add_argument("--sigma-change", type=float, default=1.8)
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    rows = load(a.out)
    rep = compare(rows, a.primary, {"coverage": a.sigma_overall, "change_coverage": a.sigma_change})
    lines = [f"# 探索对比（主指标：{a.primary}，σ overall {a.sigma_overall} / change {a.sigma_change}）", "",
             "| E | 训练集 | 方法 | seed数 | Δ覆盖 pp | Δchange pp | ΔminADE % | ΔTop1 pp | 判定 |", "|---|---|---|---:|---:|---:|---:|---:|---|"]
    f = lambda x: "—" if x is None else f"{x:+.2f}"
    for r in rep:
        lines.append(f"| {r['E']} | {r['train_set']} | {r['arm']} | {r['n']} | {f(r['d_coverage'])} | {f(r['d_change_coverage'])} | "
                     f"{f(r['d_minADE'])} | {f(r['d_top1'])} | {r['verdict']} |")
    if a.calibrate:
        c = calibrate(rows)
        lines += ["", "## 预算校准（对照 64 epoch、seed 42 的已知效果）", "",
                  "| E | T1 Δ覆盖 / Δchange | E3_self Δ覆盖 / Δchange | 复现 |", "|---|---|---|---|"]
        for E, r in c["per_budget"].items():
            cell = lambda v: "缺" if v is None else f"{v['diff']['coverage']:+.2f} / {v['diff']['change_coverage']:+.2f}"
            rep_ok = all(v and v["reproduced"] for v in r.values())
            lines.append(f"| {E} | {cell(r['T1'])} | {cell(r['E3_self'])} | {'是' if rep_ok else '否'} |")
        lines += ["", f"**建议的探索预算：{c['recommended_budget'] or '没有预算通过，探索仍需 64 epoch'}**"]
    text = "\n".join(lines) + "\n"
    (Path(a.out) / "COMPARE.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
