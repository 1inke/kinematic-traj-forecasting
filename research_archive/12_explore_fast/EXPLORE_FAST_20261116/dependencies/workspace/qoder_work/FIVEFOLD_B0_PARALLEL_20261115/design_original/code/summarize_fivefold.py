"""Five-fold summary tables (numpy only): per fold x seed, seed mean/SD, equal-weight average,
comparison with EXP-156 M2 / EXP-133 and the literature magnitudes listed in EXP-133.

usage: python -I summarize_fivefold.py --out FIVEFOLD_RUNS [--exp156 <EXP_156B.../summary.json>]
Writes FIVEFOLD_RUNS/SUMMARY.json and SUMMARY.md (refuses to overwrite).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fivefold_common import FINAL_SEEDS, FOLDS, read_json, write_json_once, now  # noqa: E402

KEYS = ("minADE", "minFDE", "same_path_FDE", "top1_ADE", "top1_FDE", "top3_minADE", "top3_minFDE",
        "top5_minADE", "top5_minFDE", "coverage", "top1_good", "nll_J", "brier_J", "ece_top1_is_J",
        "cv_ADE", "cv_FDE")
# Magnitudes as listed in EXP-133 RESULT_ANALYSIS (protocol may differ; verify against the
# original papers before any thesis table).
LITERATURE_EXP133 = {"STAR": (0.26, 0.53), "CAGN": (0.25, 0.43), "AgentFormer": (0.23, 0.40),
                     "TUTR": (0.21, 0.35), "LED": (0.21, 0.33)}
EXP133 = {"eth": (0.5011, 1.0323), "hotel": (0.1407, 0.2187), "univ": (0.2162, 0.3885),
          "zara01": (0.1738, 0.3244), "zara02": (0.1305, 0.2502)}


def collect(out):
    dec = read_json(Path(out) / "BUDGET_DECISION.json")
    E = int(dec["E_star"])
    rows = {}
    for fold in FOLDS:
        for seed in FINAL_SEEDS:
            p = Path(out) / "runs" / f"final_{fold}_E{E}_s{seed}" / "metrics.json"
            if not p.exists():
                raise FileNotFoundError(p)
            m = read_json(p)
            rows[(fold, seed)] = dict(overall=m["test"]["overall"], change=m["test"].get("change"),
                                      parameters=m["train"]["parameters"])
    return dec, rows


def summarize(dec, rows, exp156=None):
    per_fold = {}
    for fold in FOLDS:
        per_fold[fold] = {}
        for k in KEYS:
            v = np.array([rows[(fold, s)]["overall"][k] for s in FINAL_SEEDS], float)
            per_fold[fold][k] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), seeds=v.tolist())
        ch = [rows[(fold, s)]["change"] for s in FINAL_SEEDS]
        if all(x is not None for x in ch):
            c = np.array([x["coverage"] for x in ch], float)
            per_fold[fold]["change_coverage"] = dict(mean=float(c.mean()), sd=float(c.std(ddof=1)),
                                                     seeds=c.tolist(), n=ch[0]["n"])
        else:
            per_fold[fold]["change_coverage"] = None      # fold has no change-group test windows
        per_fold[fold]["n_test"] = rows[(fold, FINAL_SEEDS[0])]["overall"]["n"]
    # equal-weight five-fold average, computed per seed then summarised across seeds
    avg = {}
    for k in KEYS:
        per_seed = [np.mean([rows[(f, s)]["overall"][k] for f in FOLDS]) for s in FINAL_SEEDS]
        avg[k] = dict(mean=float(np.mean(per_seed)), sd=float(np.std(per_seed, ddof=1)), seeds=per_seed)
    no_eth = {k: float(np.mean([per_fold[f][k]["mean"] for f in FOLDS if f != "eth"])) for k in ("minADE", "minFDE")}
    cmp = {"EXP133_single_seed_equal_weight": tuple(float(np.mean([EXP133[f][i] for f in FOLDS])) for i in (0, 1))}
    if exp156:
        r = {x["fold"]: x for x in exp156["rows"]}
        cmp["EXP156_M2_single_seed_equal_weight"] = (float(np.mean([r[f]["min_ade_20"] for f in FOLDS])),
                                                    float(np.mean([r[f]["min_fde_20"] for f in FOLDS])))
        cmp["EXP156_M2_per_fold"] = {f: (r[f]["min_ade_20"], r[f]["min_fde_20"]) for f in FOLDS}
        cmp["EXP156_M2_parameters"] = r["eth"]["parameters"]
    return dict(study="FIVEFOLD_B0_20261115", at=now(), E_star=dec["E_star"],
                parameters=rows[(FOLDS[0], FINAL_SEEDS[0])]["parameters"],
                per_fold=per_fold, equal_weight_average=avg, equal_weight_without_eth=no_eth,
                comparisons=cmp, literature_as_listed_in_EXP133=LITERATURE_EXP133)


def markdown(s):
    L = [f"# 五折结果（B0，E*={s['E_star']}，seed 42/43/44，参数 {s['parameters']:,}）", "",
         "| 折 | n | minADE20 | minFDE20 | Top1 ADE/FDE | 覆盖率 | change覆盖率 | CV ADE/FDE |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for f in FOLDS:
        p = s["per_fold"][f]
        L.append(f"| {f} | {p['n_test']} | {p['minADE']['mean']:.4f}±{p['minADE']['sd']:.4f} | "
                 f"{p['minFDE']['mean']:.4f}±{p['minFDE']['sd']:.4f} | {p['top1_ADE']['mean']:.3f}/{p['top1_FDE']['mean']:.3f} | "
                 f"{100*p['coverage']['mean']:.1f}% | {('%.1f%%' % (100*p['change_coverage']['mean'])) if p['change_coverage'] else '—'} | "
                 f"{p['cv_ADE']['mean']:.3f}/{p['cv_FDE']['mean']:.3f} |")
    a = s["equal_weight_average"]
    L += ["", f"**五折等权**：minADE20 {a['minADE']['mean']:.4f}±{a['minADE']['sd']:.4f}，"
              f"minFDE20 {a['minFDE']['mean']:.4f}±{a['minFDE']['sd']:.4f}（±为 3 个 seed 的标准差）。"
              f"去掉 ETH：{s['equal_weight_without_eth']['minADE']:.4f}/{s['equal_weight_without_eth']['minFDE']:.4f}。", "",
          "| 对照 | ADE/FDE |", "|---|---:|"]
    for k, v in s["comparisons"].items():
        if isinstance(v, tuple) or (isinstance(v, list) and len(v) == 2 and not isinstance(v[0], (list, tuple))):
            L.append(f"| {k} | {v[0]:.4f}/{v[1]:.4f} |")
    for k, v in s["literature_as_listed_in_EXP133"].items():
        L.append(f"| {k}（EXP-133 所列，协议待核） | {v[0]:.2f}/{v[1]:.2f} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--exp156")
    a = ap.parse_args()
    dec, rows = collect(a.out)
    s = summarize(dec, rows, read_json(a.exp156) if a.exp156 else None)
    write_json_once(Path(a.out) / "SUMMARY.json", s)
    with (Path(a.out) / "SUMMARY.md").open("x", encoding="utf-8") as f:
        f.write(markdown(s))
    print(markdown(s))


if __name__ == "__main__":
    main()
