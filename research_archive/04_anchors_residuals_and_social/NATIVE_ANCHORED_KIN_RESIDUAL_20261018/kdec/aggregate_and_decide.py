"""汇总 3-seed RES 结果。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[1]
KIN_CEILING = 0.5646


def main():
    rows = []
    for seed in (42, 43, 44):
        p = HERE / "results" / "short" / f"COMPARE_r1_seed{seed}.json"
        if not p.exists():
            print("missing", p)
            continue
        c = json.loads(p.read_text(encoding="utf-8"))
        d = c["decision"]
        rows.append(dict(seed=seed, **d))
    if not rows:
        raise SystemExit("no rows")
    covs = [r["coverage"] for r in rows]
    gates = [r["motion"] for r in rows]
    summary = dict(
        n_seeds=len(rows),
        coverage_mean=float(np.mean(covs)),
        coverage_std=float(np.std(covs)),
        native_coverage=rows[0]["native_coverage"],
        gap_vs_native_pp_mean=float(np.mean([r["gap_vs_native_pp"] for r in rows])),
        motion_pass=sum(1 for g in gates if g == "PASS"),
        beat_native_count=sum(1 for r in rows if r["beat_native"]),
        beat_kin_ceiling_count=sum(1 for r in rows if r["beat_kin_ceiling"]),
        weak_positive_count=sum(1 for r in rows if r["weak_positive"]),
        rescue_mean=float(np.mean([r["rescue"] for r in rows])),
        lost_mean=float(np.mean([r["lost"] for r in rows])),
        minADE_mean=float(np.mean([r["minADE"] for r in rows])),
        KIN_CEILING=KIN_CEILING,
        rows=rows,
        gate_HIT=bool(
            sum(1 for r in rows if r["beat_native"]) >= 2
            or (float(np.mean(covs)) >= rows[0]["native_coverage"] and sum(1 for g in gates if g == "PASS") == len(rows))
        ),
        beats_kin_ceiling_mean=bool(float(np.mean(covs)) >= KIN_CEILING and sum(1 for g in gates if g == "PASS") == len(rows)),
    )
    out = HERE / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 结果 NATIVE_ANCHORED_KIN_RESIDUAL",
        "",
        f"- coverage mean±std: {summary['coverage_mean']:.4f}±{summary['coverage_std']:.4f}",
        f"- native: {summary['native_coverage']:.4f}  gap: {summary['gap_vs_native_pp_mean']:.2f} pp",
        f"- motion PASS: {summary['motion_pass']}/{summary['n_seeds']}",
        f"- beat_native: {summary['beat_native_count']}/{summary['n_seeds']}",
        f"- beat_kin_ceiling: {summary['beat_kin_ceiling_count']}/{summary['n_seeds']}",
        f"- rescue/lost mean: {summary['rescue_mean']:.1f} / {summary['lost_mean']:.1f}",
        f"- minADE mean: {summary['minADE_mean']:.4f}",
        f"- gate_HIT: {summary['gate_HIT']}",
        "",
        "| seed | cov | native | gap_pp | motion | rescue | lost | minADE |",
        "|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['coverage']:.4f} | {r['native_coverage']:.4f} | {r['gap_vs_native_pp']:.2f} | {r['motion']} | {r['rescue']} | {r['lost']} | {r['minADE']:.4f} |"
        )
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
