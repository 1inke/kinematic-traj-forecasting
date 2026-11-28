"""Dual-T4 batch scheduler, V2 scaling, and formal bootstrap summary for Encoder & Decoder studies."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "qoder_work/NEXT_PHASE_EXEC_20261114"
for _p in (ROOT, BASE, ROOT / "ASTRA_ACCURACY_Q20", ROOT / "FRONT_ENCODER_SERIES", ROOT / "HIERARCHICAL_DECODER_SERIES"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import common  # noqa: E402
common.ASTRA = ROOT / "ASTRA_ACCURACY_Q20"

from prepare_inputs import digest  # noqa: E402
from run_b0 import write_json  # noqa: E402
from encoder_c1c2 import ENCODER_ARMS, run_v2_scaling  # noqa: E402
from decoder_cascade_cover import DECODER_ARMS  # noqa: E402

SEEDS = (42, 43, 44)
RUNNER = Path(__file__).with_name("unified_runner.py")
BOOT_DRAWS = 4000
BOOT_SEED = 20261127


def run_worker(args, gpu: str, command: str, extra: list[str]) -> None:
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=gpu,
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    cmd = [sys.executable, "-u", str(RUNNER), command, "--study", args.study, "--data", str(args.data), *extra]
    if args.local_gpu:
        cmd.append("--local-gpu")
    if args.smoke:
        cmd.append("--smoke")
    if gpu == "cpu":
        cmd.append("--cpu")
    subprocess.run(cmd, env=env, check=True)


def cluster_draws(tracks: np.ndarray, n_boot: int = BOOT_DRAWS, seed: int = BOOT_SEED):
    units, inv = np.unique(tracks, return_inverse=True)
    rng = np.random.default_rng(seed)
    sel = rng.integers(len(units), size=(n_boot, len(units)), dtype=np.int32)
    counts = np.zeros((n_boot, len(units)), dtype=np.int32)
    np.add.at(counts, (np.arange(n_boot)[:, None], sel), 1)
    return units, inv, counts


def paired_gain_stats(
    arm_vecs: list[np.ndarray],
    base_vecs: list[np.ndarray],
    tracks: np.ndarray,
    recordings: np.ndarray,
    no_zara_mask: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict:
    """Compute relative error reduction (positive = improvement) with 4000-draw track-cluster CI."""
    if mask is None:
        mask = np.ones(len(tracks), dtype=bool)
    if not mask.any():
        return dict(
            windows=0, tracks=0, arm_mean=0.0, base_mean=0.0, delta_m=0.0,
            gain_pct=0.0, ci95_pct=[0.0, 0.0], by_seed_gain_pct=[0.0] * len(arm_vecs),
            without_zara02_gain_pct=0.0, without_zara02_ci95_pct=[0.0, 0.0],
            leave_one_recording_pct={}, same_sign_recordings=0, total_recordings=0,
            top10_cluster_share_abs=0.0, top10_cluster_share_signed=0.0,
        )
    a_m = np.stack([v[mask].astype(np.float64) for v in arm_vecs])
    b_m = np.stack([v[mask].astype(np.float64) for v in base_vecs])
    a_mean = a_m.mean(axis=0)
    b_mean = b_m.mean(axis=0)
    diff = b_mean - a_mean  # positive = arm has lower error than base

    tr = tracks[mask]
    rec = recordings[mask]
    nz = no_zara_mask[mask]

    units, inv, counts = cluster_draws(tr)
    d_sum = np.bincount(inv, weights=diff)
    b_sum = np.bincount(inv, weights=b_mean)
    denom_mean = float(b_sum.sum())
    gain_pct = float(100.0 * d_sum.sum() / denom_mean) if denom_mean > 1e-12 else 0.0

    num_boot = counts @ d_sum
    den_boot = np.maximum(counts @ b_sum, 1e-12)
    boot_pct = 100.0 * (num_boot / den_boot)
    ci95 = np.quantile(boot_pct, [0.025, 0.975]).tolist()

    by_seed_pct = [
        float(100.0 * (bv.mean() - av.mean()) / max(bv.mean(), 1e-12))
        for av, bv in zip(a_m, b_m)
    ]

    # Top-10 cluster share
    abs_top = np.argsort(-np.abs(d_sum), kind="stable")[:10]
    top10_abs_share = float(np.abs(d_sum[abs_top]).sum() / np.abs(d_sum).sum()) if np.abs(d_sum).sum() > 1e-12 else 0.0
    top10_signed_share = float(d_sum[abs_top].sum() / d_sum.sum()) if abs(d_sum.sum()) > 1e-12 else 0.0

    # Without ZARA02
    if nz.any():
        _, inv_nz, cnt_nz = cluster_draws(tr[nz])
        dn = np.bincount(inv_nz, weights=diff[nz])
        bn = np.bincount(inv_nz, weights=b_mean[nz])
        nz_pct = float(100.0 * dn.sum() / max(bn.sum(), 1e-12))
        nz_boot = 100.0 * (cnt_nz @ dn) / np.maximum(cnt_nz @ bn, 1e-12)
        nz_ci = np.quantile(nz_boot, [0.025, 0.975]).tolist()
    else:
        nz_pct, nz_ci = gain_pct, ci95

    # Leave-one-recording-out
    loro = {}
    for r_name in sorted(set(rec.tolist())):
        keep = rec != r_name
        if keep.any() and b_mean[keep].sum() > 1e-12:
            loro[r_name] = float(100.0 * diff[keep].sum() / b_mean[keep].sum())
    same_sign = int(sum(np.sign(val) == np.sign(gain_pct) for val in loro.values()))

    return dict(
        windows=int(mask.sum()),
        tracks=int(len(units)),
        arm_mean=float(a_mean.mean()),
        base_mean=float(b_mean.mean()),
        delta_m=float((a_mean - b_mean).mean()),
        gain_pct=gain_pct,
        ci95_pct=[float(ci95[0]), float(ci95[1])],
        by_seed_gain_pct=by_seed_pct,
        without_zara02_gain_pct=nz_pct,
        without_zara02_ci95_pct=[float(nz_ci[0]), float(nz_ci[1])],
        leave_one_recording_pct=loro,
        same_sign_recordings=same_sign,
        total_recordings=len(loro),
        top10_cluster_share_abs=top10_abs_share,
        top10_cluster_share_signed=top10_signed_share,
    )


def summarize_study(study: str, out_dir: Path) -> dict:
    arms = ENCODER_ARMS if study == "encoder" else DECODER_ARMS
    runs_dir = out_dir / "runs"

    results = {}
    arrays = {}
    ref_ids = ref_tracks = ref_recs = ref_no_zara = None

    for arm in arms:
        results[arm] = []
        arrays[arm] = []
        for seed in SEEDS:
            rdir = runs_dir / f"{arm}_s{seed}"
            res = json.loads((rdir / "RESULT.json").read_bytes())
            mpath = rdir / "METRICS.npz"
            assert res["status"] == "PASS" and digest(mpath) == res["metrics_sha256"], rdir
            with np.load(mpath, allow_pickle=False) as z:
                data = {k: z[k] for k in z.files}
            if ref_ids is None:
                ref_ids = data["ids"]
                ref_tracks = data["tracks"]
                ref_recs = data["recordings"]
                ref_no_zara = data["without_zara02"]
            else:
                np.testing.assert_array_equal(ref_ids, data["ids"])
                np.testing.assert_array_equal(ref_tracks, data["tracks"])
            results[arm].append(res)
            arrays[arm].append(data)

    def vecs(arm: str, metric: str) -> list[np.ndarray]:
        return [d[metric] for d in arrays[arm]]

    def compare(arm: str, base: str, metric: str, mask: np.ndarray | None = None) -> dict:
        return paired_gain_stats(
            vecs(arm, metric), vecs(base, metric), ref_tracks, ref_recs, ref_no_zara, mask=mask
        )

    arm_means = {}
    base_arm = "E0" if study == "encoder" else "M0"
    base_jcov = np.mean([d["J_cov"] for d in arrays[base_arm]], axis=0) > 0.5
    for arm in arms:
        keys = [
            "J_cov", "J_cont", "eff_good_clusters", "minADE", "minFDE",
            "Top1_ADE", "Top1_FDE", "brier_minFDE", "ES",
            "early_ADE", "mid_ADE", "late_ADE", "endpoint_spread_m",
            "speed_p95", "acc_mean", "acc_p95", "jerk_mean", "jerk_p95",
        ]
        if study == "decoder":
            for oname in ("conf", "riskp", "cov_sel", "cov_t05", "cov_t10", "cov_t20"):
                for k in (1, 3, 5, 20):
                    keys.extend([f"{oname}_minADE{k}", f"{oname}_minFDE{k}"])

        row = {
            "parameters": results[arm][0]["parameters"],
            "trainable_parameters": results[arm][0]["trainable_parameters"],
        }
        for k in keys:
            if k in arrays[arm][0]:
                vals = [float(d[k].mean()) for d in arrays[arm]]
                row[k] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)), by_seed=vals)

        # Net rescue vs base_arm on J_cov (per-seed averaged)
        res_rates, lost_rates = [], []
        for d_arm, d_base in zip(arrays[arm], arrays[base_arm]):
            g_a = d_arm["J_cov"] > 0.5
            g_b = d_base["J_cov"] > 0.5
            res_rates.append(float((~g_b & g_a).mean()))
            lost_rates.append(float((g_b & ~g_a).mean()))
        row["net_rescue_jcov_vs_base"] = dict(
            rescue_mean=float(np.mean(res_rates)),
            lost_mean=float(np.mean(lost_rates)),
            net_pp=float(100.0 * (np.mean(res_rates) - np.mean(lost_rates))),
            by_seed_net_pp=[float(100.0 * (r - l)) for r, l in zip(res_rates, lost_rates)],
        )
        arm_means[arm] = row

    comparisons = {}
    gates = {}
    manipulation = {}

    if study == "encoder":
        c_e0_e1 = {m: compare("E0", "E1", m) for m in ("minFDE", "minADE", "Top1_FDE", "J_cont")}
        c_e3_e0 = {m: compare("E3", "E0", m) for m in ("minFDE", "minADE", "Top1_FDE", "J_cont", "brier_minFDE")}
        c_e4_e0 = {m: compare("E4", "E0", m) for m in ("minFDE", "Top1_FDE")}
        c_e4_e3 = {m: compare("E4", "E3", m) for m in ("minFDE", "Top1_FDE")}
        c_e2_e0 = {m: compare("E2", "E0", m) for m in ("minFDE", "minADE", "Top1_FDE", "J_cont")}
        c_e2_e3 = {m: compare("E2", "E3", m) for m in ("minFDE", "minADE", "Top1_FDE", "J_cont")}
        c_e5_e0 = {m: compare("E5", "E0", m) for m in ("minFDE", "minADE", "Top1_FDE", "J_cont")}
        c_e6_e0 = {m: compare("E6", "E0", m) for m in ("minFDE", "minADE", "Top1_FDE", "J_cont")}

        comparisons = dict(
            E0_vs_E1=c_e0_e1,
            E3_vs_E0=c_e3_e0,
            E4_vs_E0=c_e4_e0,
            E4_vs_E3=c_e4_e3,
            E2_vs_E0=c_e2_e0,
            E2_vs_E3=c_e2_e3,
            E5_vs_E0=c_e5_e0,
            E6_vs_E0=c_e6_e0,
        )

        g1 = c_e0_e1["minFDE"]["gain_pct"] >= 1.5 and c_e0_e1["minFDE"]["ci95_pct"][0] > 0.0
        g2 = (
            (c_e3_e0["minFDE"]["gain_pct"] >= 0.75 and c_e3_e0["minFDE"]["ci95_pct"][0] > 0.0)
            or (c_e3_e0["Top1_FDE"]["gain_pct"] >= 2.0 and c_e3_e0["minFDE"]["gain_pct"] >= -0.3)
        )
        g3_trigger = (
            c_e2_e0["minFDE"]["gain_pct"] > 0.5 and c_e2_e0["minFDE"]["ci95_pct"][0] > 0.0
            and c_e2_e3["minFDE"]["gain_pct"] > 0.5 and c_e2_e3["minFDE"]["ci95_pct"][0] > 0.0
        )
        g6 = c_e4_e3["minFDE"]["gain_pct"] >= 0.5
        gates = dict(
            G1_description_effective=bool(g1),
            G2_c2_effective=bool(g2),
            G3_e2_beats_both_trigger=bool(g3_trigger),
            G3_keep_temporal_cross=bool(not g3_trigger),
            G4_e5_vs_e0_gain_pct=c_e5_e0["minFDE"]["gain_pct"],
            G5_e6_vs_e0_gain_pct=c_e6_e0["minFDE"]["gain_pct"],
            G6_use_c2_lag=bool(g6),
        )

        manipulation["MC1_E1"] = [r["diagnostics"] for r in results["E1"]]
        manipulation["MC2_E3"] = [r["diagnostics"] for r in results["E3"]]
        manipulation["MC2_E4"] = [r["diagnostics"] for r in results["E4"]]
        manipulation["MC3_E5"] = [r["diagnostics"] for r in results["E5"]]
        manipulation["MC3_E6"] = [r["diagnostics"] for r in results["E6"]]
        manipulation["MC1_all_pass"] = all(d["mc1_pass"] for d in manipulation["MC1_E1"])
        manipulation["MC2_E3_all_pass"] = all(d.get("mc2_pass", False) for d in manipulation["MC2_E3"])
        manipulation["MC3_all_pass"] = all(d["mc3_pass"] for d in manipulation["MC3_E5"] + manipulation["MC3_E6"])
    else:
        fast_m = arrays["M0"][0]["fast"]
        c_m2_m1 = {
            "overall": {m: compare("M2", "M1", m) for m in ("minFDE", "minADE", "Top1_FDE", "J_cont", "endpoint_spread_m")},
            "fast": {m: compare("M2", "M1", m, mask=fast_m) for m in ("minFDE", "minADE", "Top1_FDE")},
        }
        c_m3_m2 = {m: compare("M3", "M2", m) for m in ("minFDE", "Top1_FDE", "brier_minFDE", "cov_sel_minFDE3", "cov_sel_minFDE5")}
        c_m4_m3 = {
            m: compare("M4", "M3", m)
            for m in (
                "Top1_FDE", "brier_minFDE", "minFDE",
                "cov_sel_minFDE3", "cov_sel_minFDE5", "cov_sel_minADE3", "cov_sel_minADE5",
                "conf_minFDE3", "conf_minFDE5",
            )
        }
        c_m4_m5 = {m: compare("M4", "M5", m) for m in ("cov_sel_minFDE3", "cov_sel_minFDE5", "brier_minFDE", "Top1_FDE", "minFDE")}
        c_m4_m6 = {m: compare("M4", "M6", m) for m in ("minFDE", "Top1_FDE", "cov_sel_minFDE3", "cov_sel_minFDE5", "brier_minFDE")}
        c_m4_m0 = {m: compare("M4", "M0", m) for m in ("minFDE", "Top1_FDE", "cov_sel_minFDE3", "cov_sel_minFDE5", "brier_minFDE", "J_cont")}
        c_jm4_m4 = {m: compare("JM4", "M4", m) for m in ("minFDE", "Top1_FDE", "cov_sel_minFDE3", "cov_sel_minFDE5", "brier_minFDE", "J_cont")}
        c_jm4_m0 = {m: compare("JM4", "M0", m) for m in ("minFDE", "Top1_FDE", "cov_sel_minFDE3", "cov_sel_minFDE5", "brier_minFDE", "J_cont")}

        comparisons = dict(
            M2_vs_M1=c_m2_m1,
            M3_vs_M2=c_m3_m2,
            M4_vs_M3=c_m4_m3,
            M4_vs_M5=c_m4_m5,
            M4_vs_M6=c_m4_m6,
            M4_vs_M0=c_m4_m0,
            JM4_vs_M4=c_jm4_m4,
            JM4_vs_M0=c_jm4_m0,
        )

        g1 = (
            c_m2_m1["overall"]["minFDE"]["gain_pct"] >= 1.0
            and c_m2_m1["overall"]["minFDE"]["ci95_pct"][0] > 0.0
            and c_m2_m1["fast"]["minFDE"]["gain_pct"] >= 0.0
        )
        g2 = (
            c_m3_m2["Top1_FDE"]["gain_pct"] >= 3.0
            and abs(c_m3_m2["minFDE"]["gain_pct"]) <= 1.0
        )
        g3 = (
            c_m4_m3["Top1_FDE"]["gain_pct"] >= -1.0
            and (
                (c_m4_m3["cov_sel_minFDE3"]["gain_pct"] >= 1.5 and c_m4_m3["cov_sel_minFDE3"]["ci95_pct"][0] > 0.0)
                or (c_m4_m3["cov_sel_minFDE5"]["gain_pct"] >= 1.5 and c_m4_m3["cov_sel_minFDE5"]["ci95_pct"][0] > 0.0)
                or (c_m4_m3["brier_minFDE"]["gain_pct"] >= 2.0)
            )
        )
        g4 = c_m4_m5["cov_sel_minFDE3"]["gain_pct"] >= 1.0
        g5 = c_m4_m6["minFDE"]["gain_pct"] >= 1.0
        g6_m4 = results["M4"][0]["inference_parameters"] <= 400000
        g6_jm4 = results["JM4"][0]["inference_parameters"] <= 400000
        gates = dict(
            G1_cascade_effective=bool(g1),
            G2_disc_layer_effective=bool(g2),
            G3_cover_order_effective=bool(g3),
            G4_redundancy_features_helpful=bool(g4),
            G5_beats_capacity_control=bool(g5),
            G6_m4_within_400k=bool(g6_m4),
            G6_jm4_within_400k=bool(g6_jm4),
            DECODER_STAGE1_PASS=bool(g1 and g2 and g3 and g6_m4),
        )

        # MC-a, MC-b, MC-c, MC-d
        mc_a = {}
        for arm in ("M1", "M2", "M4", "JM4"):
            mc_a[arm] = [r["diagnostics"]["MC_a"] for r in results[arm]]
        m2_pos = float(np.mean([x["train_final_l0_pos_mean"] for x in mc_a["M2"]]))
        m2_c0 = float(np.mean([x["c0_dev"]["overall_match_rate"] for x in mc_a["M2"]]))
        m2_eth_fast = float(np.mean([x["eth_test_l0"]["fast_match_rate"] or 0.0 for x in mc_a["M2"]]))
        mc_b_all = all(r["diagnostics"].get("MC_b_pass", False) for arm in ("M3", "M4", "M5", "JM4") for r in results[arm])

        # MC-c: compare M4 vs M3 internal disc_order and output order
        mc_c_rows = []
        for s_idx, seed in enumerate(SEEDS):
            o4 = arrays["M4"][s_idx]["disc_order"]
            o3 = arrays["M3"][s_idx]["disc_order"]
            pos2_5_diff = float(np.any(o4[:, 1:5] != o3[:, 1:5], axis=1).mean())
            top5_ov = float((o4[:, :5, None] == o3[:, None, :5]).any(axis=-1).mean())
            mc_c_rows.append(dict(seed=seed, pos2to5_any_diff=pos2_5_diff, top5_overlap=top5_ov))
        mc_c_pass = (
            np.mean([x["pos2to5_any_diff"] for x in mc_c_rows]) >= 0.15
            and np.mean([x["top5_overlap"] for x in mc_c_rows]) <= 0.80
        )

        m6_pdiff = abs(results["M6"][0]["parameters"] - results["M4"][0]["parameters"]) / results["M4"][0]["parameters"]
        mc_d_pass = (
            m6_pdiff <= 0.02
            and all(r["updates"] == results["M4"][0]["updates"] for r in results["M6"])
        )
        manipulation = dict(
            MC_a=dict(
                m2_l0_pos_mean=m2_pos,
                m2_c0_dev_match=m2_c0,
                m2_eth_fast_match=m2_eth_fast,
                pass_flag=bool(m2_pos >= 1.5 and m2_c0 >= 0.90 and m2_eth_fast >= 0.20),
                details=mc_a,
            ),
            MC_b_pass=bool(mc_b_all),
            MC_c=dict(per_seed=mc_c_rows, pass_flag=bool(mc_c_pass)),
            MC_d=dict(m6_vs_m4_param_diff=m6_pdiff, pass_flag=bool(mc_d_pass)),
        )

    summary = dict(
        study=study,
        arms=list(arms),
        seeds=list(SEEDS),
        windows=len(ref_ids),
        tracks=len(np.unique(ref_tracks)),
        arm_means=arm_means,
        comparisons=comparisons,
        gates=gates,
        manipulation=manipulation,
    )
    write_json(out_dir / "STUDY_SUMMARY.json", summary)

    # Generate human-readable REPORT.md
    lines = [f"# {study.upper()} Study Summary (ETH/UCY C0, 3 seeds 42-44)\n"]
    lines.append(f"- Windows: `{summary['windows']}`, Tracks: `{summary['tracks']}`\n")
    lines.append("## Arm Means\n")
    if study == "encoder":
        lines.append("| Arm | Params | Trainable | J_cov | J_cont | minFDE | minADE | Top1_FDE | brier_minFDE | NetRescue(pp) |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for arm in arms:
            r = arm_means[arm]
            lines.append(
                f"| {arm} | {r['parameters']} | {r['trainable_parameters']} | "
                f"{r['J_cov']['mean']:.4f} | {r['J_cont']['mean']:.4f} | "
                f"{r['minFDE']['mean']:.6f} | {r['minADE']['mean']:.6f} | "
                f"{r['Top1_FDE']['mean']:.6f} | {r['brier_minFDE']['mean']:.6f} | "
                f"{r['net_rescue_jcov_vs_base']['net_pp']:+.2f} |"
            )
    else:
        lines.append("| Arm | Params | J_cov | J_cont | minFDE@20 | Top1_FDE | cov_sel@3 | cov_sel@5 | conf@3 | brier | NetRescue(pp) |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for arm in arms:
            r = arm_means[arm]
            lines.append(
                f"| {arm} | {r['parameters']} | {r['J_cov']['mean']:.4f} | {r['J_cont']['mean']:.4f} | "
                f"{r['minFDE']['mean']:.6f} | {r['Top1_FDE']['mean']:.6f} | "
                f"{r['cov_sel_minFDE3']['mean']:.6f} | {r['cov_sel_minFDE5']['mean']:.6f} | "
                f"{r['conf_minFDE3']['mean']:.6f} | {r['brier_minFDE']['mean']:.6f} | "
                f"{r['net_rescue_jcov_vs_base']['net_pp']:+.2f} |"
            )
    lines.append("\n## Gates\n```json\n" + json.dumps(gates, indent=2) + "\n```\n")
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", choices=("encoder", "decoder"), required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--protocol", default="local")
    p.add_argument("--budget-hours", type=float, default=9.5)
    p.add_argument("--local-gpu", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        devices = [str(i) for i in range(torch.cuda.device_count())]
        names = [torch.cuda.get_device_name(i) for i in range(len(devices))]
        if not args.local_gpu and not args.smoke:
            if len(devices) != 2 or any("T4" not in n for n in names):
                raise RuntimeError(f"Expected 2x Tesla T4 on Kaggle, got {names}")
    elif args.smoke:
        devices = ["cpu"]
        names = ["CPU"]
    else:
        raise RuntimeError("CUDA GPU required")

    print(f"STUDY_START study={args.study} devices={names} protocol={args.protocol}", flush=True)
    write_json(args.out / "RUNTIME.json", dict(study=args.study, devices=names, protocol=args.protocol,
                                               torch=str(torch.__version__), cuda=torch.version.cuda))

    # 1. Parallel per-GPU preflight
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        jobs = [
            pool.submit(run_worker, args, gpu, "preflight", ["--out", str(args.out / f"preflight_{gpu}")])
            for gpu in devices
        ]
        for j in jobs:
            j.result()
    print(f"DUAL_T4_PREFLIGHT_PASS {args.study} {names}", flush=True)

    arms = ENCODER_ARMS if args.study == "encoder" else DECODER_ARMS
    # Put E0 / M0 seeds first so paired reference runs finish in Wave 1
    first_arm = arms[0]
    tasks = [(first_arm, seed) for seed in SEEDS] + [
        (arm, seed) for seed in SEEDS for arm in arms[1:]
    ]
    write_json(args.out / "PLAN.json", dict(study=args.study, protocol=args.protocol,
                                            tasks=[f"{a}_s{s}" for a, s in tasks]))

    deadline = time.time() + args.budget_hours * 3600.0
    lanes = devices if args.smoke else (devices * 2 if len(devices) == 2 else [devices[0], devices[0]])

    # 2. Dispatch training across concurrent GPU lanes
    queued = deque(tasks)
    active: dict[int, tuple[subprocess.Popen, tuple[str, int]]] = {}
    while queued or active:
        for lane_id, (proc, task) in list(active.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del active[lane_id]
            if rc != 0:
                for p_other, _ in active.values():
                    p_other.terminate()
                for p_other, _ in active.values():
                    p_other.wait(timeout=30)
                raise RuntimeError(f"Training failed for {task} with exit code {rc}")
        for lane_id, gpu in enumerate(lanes):
            if not queued or lane_id in active or time.time() >= deadline:
                continue
            arm, seed = queued.popleft()
            rdir = args.out / "runs" / f"{arm}_s{seed}"
            if (rdir / "TRAINED.json").is_file():
                continue
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=gpu,
                PYTHONDONTWRITEBYTECODE="1",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
            )
            cmd = [
                sys.executable, "-u", str(RUNNER), "train",
                "--study", args.study, "--arm", arm, "--seed", str(seed),
                "--data", str(args.data), "--out", str(rdir),
                "--protocol", args.protocol, "--deadline", str(deadline),
            ]
            if args.local_gpu:
                cmd.append("--local-gpu")
            if args.smoke:
                cmd.append("--smoke")
            if gpu == "cpu":
                cmd.append("--cpu")
            print(f"START train {rdir.name} physical_gpu={gpu}", flush=True)
            active[lane_id] = (subprocess.Popen(cmd, env=env), (arm, seed))
        if time.time() >= deadline and not active:
            break
        if active:
            time.sleep(2)

    all_trained = all((args.out / "runs" / f"{a}_s{s}" / "TRAINED.json").is_file() for a, s in tasks)
    if not all_trained:
        write_json(args.out / "STATUS.json", dict(status="TIME_LIMIT_RESUMABLE", study=args.study, protocol=args.protocol))
        print(f"STUDY_STATUS {args.study} TIME_LIMIT_RESUMABLE", flush=True)
        return

    # 3. Evaluate E0/M0 first, then remaining arms across physical GPUs (1 per GPU)
    def eval_task(gpu: str, arm: str, seed: int) -> None:
        rdir = args.out / "runs" / f"{arm}_s{seed}"
        if (rdir / "RESULT.json").is_file():
            return
        extra = ["--arm", arm, "--seed", str(seed), "--out", str(rdir), "--protocol", args.protocol]
        if args.study == "encoder" and arm != "E0":
            extra.extend(["--e0-out", str(args.out / "runs" / f"E0_s{seed}")])
        run_worker(args, gpu, "evaluate", extra)

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        first_jobs = [pool.submit(eval_task, devices[i % len(devices)], first_arm, s) for i, s in enumerate(SEEDS)]
        for j in first_jobs:
            j.result()

    rest = [(a, s) for a in arms[1:] for s in SEEDS]
    def eval_lane(gpu: str, subset: list[tuple[str, int]]) -> None:
        for arm, seed in subset:
            eval_task(gpu, arm, seed)

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        rest_jobs = [pool.submit(eval_lane, devices[i], rest[i :: len(devices)]) for i in range(len(devices))]
        for j in rest_jobs:
            j.result()

    # 4. Run V2 efficiency scaling in encoder study
    if args.study == "encoder":
        n_vals = (8, 16, 32) if args.smoke else (8, 16, 32, 64, 128, 256)
        b_vals = (1, 16) if args.smoke else (1, 64)
        v2_rows = run_v2_scaling(
            device="cpu" if devices[0] == "cpu" else "cuda:0",
            n_values=n_vals,
            batch_sizes=b_vals,
            warmup=2 if args.smoke else 5,
            iterations=5 if args.smoke else 20,
            cycles=2 if args.smoke else 4,
        )
        write_json(args.out / "V2_SCALING.json", v2_rows)
        with (args.out / "V2_SCALING.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(v2_rows[0]))
            writer.writeheader()
            writer.writerows(v2_rows)
        print(f"V2_SCALING_COMPLETE rows={len(v2_rows)}", flush=True)

    # 5. Summarize and evaluate formal gates
    summary = summarize_study(args.study, args.out)
    write_json(args.out / "STATUS.json", dict(status="COMPLETE", study=args.study, protocol=args.protocol,
                                              gates=summary["gates"]))
    print(f"STUDY_STATUS {args.study} COMPLETE gates={json.dumps(summary['gates'])}", flush=True)


if __name__ == "__main__":
    main()
