from __future__ import annotations
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]

def main():
    rows = []
    for seed in (42, 43, 44):
        p = HERE / "results" / "short" / f"COMPARE_r1_seed{seed}.json"
        if p.exists():
            c = json.loads(p.read_text(encoding="utf-8"))
            rows.append(dict(seed=seed, **c["decision"]))
    if not rows:
        # allow single-seed
        p = HERE / "results" / "short" / "COMPARE_r1_seed42.json"
        if p.exists():
            c = json.loads(p.read_text(encoding="utf-8"))
            rows.append(dict(seed=42, **c["decision"]))
    summary = dict(
        n_seeds=len(rows),
        coverage_mean=float(np.mean([r["coverage"] for r in rows])),
        native_coverage=rows[0]["native_coverage"],
        gap_pp_mean=float(np.mean([r["gap_pp"] for r in rows])),
        motion_pass=sum(1 for r in rows if r["motion"]=="PASS"),
        beat_kin_ceiling_count=sum(1 for r in rows if r["beat_kin_ceiling"]),
        beat_native_count=sum(1 for r in rows if r["beat_native"]),
        minADE_mean=float(np.mean([r["minADE"] for r in rows])),
        rows=rows,
        gate_HIT=bool(sum(1 for r in rows if r["beat_kin_ceiling"]) >= max(1, len(rows)//2 + len(rows)%2)),
    )
    (HERE / "results" / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 结果 MODE_DIVERSE_KINEMATIC", "",
             f"- cov mean: {summary['coverage_mean']:.4f} (native {summary['native_coverage']:.4f})",
             f"- gap_pp: {summary['gap_pp_mean']:.2f}",
             f"- motion PASS: {summary['motion_pass']}/{summary['n_seeds']}",
             f"- beat_kin_ceiling: {summary['beat_kin_ceiling_count']}",
             f"- gate_HIT: {summary['gate_HIT']}", ""]
    (HERE / "results" / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (HERE / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
