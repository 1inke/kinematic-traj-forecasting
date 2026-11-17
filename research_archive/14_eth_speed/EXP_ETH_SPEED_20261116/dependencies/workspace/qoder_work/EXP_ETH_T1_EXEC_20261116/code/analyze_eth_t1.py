"""【待执行】实验一 EXP-ETH-T1 analysis (numpy only, CPU, seconds).

Reads runs/B0_eth_E64_s42 and runs/T1_eth_E64_s42 (result.json + test_window_metrics.npz).

PRIMARY (fixed before running) - ETH test, standard best-of-20, relative change T1 vs B0:
  helps      : minADE improves >= 2% AND minFDE improves >= 2%
  no help    : minADE improves < 0.5% (or gets worse) AND minFDE improves < 0.5% (or gets worse)
  unclear    : anything else -> add seed 43 for both arms (2 runs, ~1 h) before deciding
Noise reference: two seeds of B0 (100% train, 128 epochs) on ETH differ by 0.0027 ADE (0.6%) and
0.0088 FDE (1.4%); 2% is about 3x / 1.4x that.
SECONDARY (descriptive): coverage overall / change (190 change windows: 1 window = 0.53 pp),
rescued/lost vs B0, Top1, junction acceleration; context rows (B0-E128, M2, TUTR); optional
breakdown by observed speed (needs --fivefold-code and --data-dir) and train-vs-test speed.

usage: python -I analyze_eth_t1.py --out <EXP_ETH_T1_20261116> [--fivefold-code <...> --data-dir <...>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CONTEXT = {"B0 E128 95% seed42 (budget reference)": (0.4445, 0.6638),
           "B0 E128 100% seed42 (final)": (0.4203, 0.6186), "B0 E128 100% seed43 (final)": (0.4229, 0.6274),
           "EXP-156 M2 (single seed)": (0.4085, 0.5996), "TUTR (ICCV 2023, Table 1)": (0.40, 0.61)}
HELP, NOHELP = 0.02, 0.005


def load(run):
    r = json.loads((run / "result.json").read_text(encoding="utf-8-sig"))
    with np.load(run / "test_window_metrics.npz", allow_pickle=False) as z:
        w = {k: z[k] for k in z.files}
    return r, w


def verdict(b, t):
    ga = (b["minADE"] - t["minADE"]) / b["minADE"]
    gf = (b["minFDE"] - t["minFDE"]) / b["minFDE"]
    if ga >= HELP and gf >= HELP:
        v = "helps"
    elif ga < NOHELP and gf < NOHELP:
        v = "no_help"
    else:
        v = "unclear"
    return v, ga, gf


def speed_breakdown(fivefold_code, data_dir, wb, wt):
    sys.path.insert(0, str(Path(fivefold_code).resolve()))
    from fivefold_data import load_split
    test = load_split(data_dir, "eth", "test"); train = load_split(data_dir, "eth", "train")
    sp = lambda d: np.linalg.norm(np.diff(d["past"][:, 0], axis=1), axis=-1).mean(1) / .4
    s_te, s_tr = sp(test), sp(train)
    if not np.array_equal(test["ids"], wb["ids"]):
        raise ValueError("test ids differ from window metrics")
    edges = np.quantile(s_te, [0, 1 / 3, 2 / 3, 1])
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s_te >= lo) & (s_te <= hi)
        rows.append(dict(speed_mps=f"{lo:.2f}-{hi:.2f}", n=int(m.sum()), B0_minADE=float(wb["minADE"][m].mean()),
                         T1_minADE=float(wt["minADE"][m].mean()), B0_cov=float(wb["coverage"][m].mean()),
                         T1_cov=float(wt["coverage"][m].mean())))
    q = lambda s: dict(median=float(np.median(s)), p90=float(np.quantile(s, .9)))
    return dict(train_speed=q(s_tr), test_speed=q(s_te), by_test_speed_tertile=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--fivefold-code"); ap.add_argument("--data-dir")
    a = ap.parse_args()
    out = Path(a.out)
    (rb, wb), (rt, wt) = load(out / "runs/B0_eth_E64_s42"), load(out / "runs/T1_eth_E64_s42")
    if not (np.array_equal(wb["ids"], wt["ids"]) and np.array_equal(wb["group"], wt["group"])):
        raise ValueError("B0 and T1 were evaluated on different windows")
    b, t = rb["test"]["overall"], rt["test"]["overall"]
    v, ga, gf = verdict(b, t)
    groups = {}
    for name, mask in [("overall", np.ones(len(wb["ids"]), bool)), ("stable", wb["group"] == 0),
                       ("transition", wb["group"] == 1), ("change", wb["group"] == 2)]:
        old, new = wb["coverage"][mask].astype(bool), wt["coverage"][mask].astype(bool)
        groups[name] = dict(n=int(mask.sum()), B0_cov=100 * float(old.mean()), T1_cov=100 * float(new.mean()),
                            rescued=int((~old & new).sum()), lost=int((old & ~new).sum()),
                            B0_minADE=float(wb["minADE"][mask].mean()), T1_minADE=float(wt["minADE"][mask].mean()))
    speed = speed_breakdown(a.fivefold_code, a.data_dir, wb, wt) if a.fivefold_code and a.data_dir else None
    res = dict(verdict=v, relative_gain_minADE=ga, relative_gain_minFDE=gf, B0=b, T1=t, groups=groups, speed=speed)
    (out / "ETH_T1_RESULT.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    names = {"helps": "有帮助：T1 在 ETH 上明显更好 → 进入训练目标重构",
             "no_help": "无帮助：T1 在 ETH 上没有改善 → 先诊断 ETH 差在哪里，再定重构重点",
             "unclear": "不确定：介于两者之间 → B0 和 T1 各补一个 seed 43 再判断"}
    L = ["# 【已完成】实验一 EXP-ETH-T1 结果（ETH 测试 364 窗，64 epoch，seed 42，95% 训练）", "",
         f"**判定：{names[v]}**", "",
         "| | minADE20 | minFDE20 | 覆盖率 | Top1 好轨迹 | 接续加速度 |", "|---|---:|---:|---:|---:|---:|",
         f"| B0 | {b['minADE']:.4f} | {b['minFDE']:.4f} | {100*b['coverage']:.1f}% | {100*b['top1_good']:.1f}% | {b.get('junction_accel_allK20', float('nan')):.3f} |",
         f"| T1 | {t['minADE']:.4f} | {t['minFDE']:.4f} | {100*t['coverage']:.1f}% | {100*t['top1_good']:.1f}% | {t.get('junction_accel_allK20', float('nan')):.3f} |",
         f"| T1 相对改善 | {100*ga:+.1f}% | {100*gf:+.1f}% | | | |", "",
         "判定线：两项都改善 ≥2% 为有帮助；两项都 <0.5% 为无帮助（B0 两个 seed 在 ETH 上相差约 0.6% / 1.4%）。", "",
         "## 分组", "", "| 组 | 窗口数 | B0 覆盖 | T1 覆盖 | 救回 | 丢失 | B0 minADE | T1 minADE |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, g in groups.items():
        L.append(f"| {k} | {g['n']} | {g['B0_cov']:.1f}% | {g['T1_cov']:.1f}% | {g['rescued']} | {g['lost']} | {g['B0_minADE']:.4f} | {g['T1_minADE']:.4f} |")
    L += ["", "## 参照（ETH 折，不同协议，仅看量级）", "", "| | minADE/minFDE |", "|---|---:|"]
    L += [f"| {k} | {x:.4f}/{y:.4f} |" for k, (x, y) in CONTEXT.items()]
    if speed:
        L += ["", "## 速度", "",
              f"训练集观测速度中位数 {speed['train_speed']['median']:.2f} m/s（p90 {speed['train_speed']['p90']:.2f}），"
              f"测试集 {speed['test_speed']['median']:.2f} m/s（p90 {speed['test_speed']['p90']:.2f}）。", "",
              "| 测试速度三分位 (m/s) | n | B0 minADE | T1 minADE | B0 覆盖 | T1 覆盖 |", "|---|---:|---:|---:|---:|---:|"]
        for r in speed["by_test_speed_tertile"]:
            L.append(f"| {r['speed_mps']} | {r['n']} | {r['B0_minADE']:.4f} | {r['T1_minADE']:.4f} | {100*r['B0_cov']:.1f}% | {100*r['T1_cov']:.1f}% |")
    text = "\n".join(L) + "\n"
    (out / "ETH_T1_RESULT.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
