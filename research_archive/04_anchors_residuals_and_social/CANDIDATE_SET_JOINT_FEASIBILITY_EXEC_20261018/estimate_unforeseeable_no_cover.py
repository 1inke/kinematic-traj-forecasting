# -*- coding: utf-8 -*-
"""Estimate |{no good cover} ∩ {unforeseeable}| on ETH outer.
Design: UNFORESEEABLE_NO_COVER_ESTIMATE_DESIGN.md
Read-only over caches; writes JSON/MD under this EXEC only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(r"<WORKSPACE_ROOT>")
EXEC = ROOT / "CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018"
VIS = EXEC / "visual_review_overnight" / "data" / "arrays.npz"
KIN_TMPL = ROOT / "UNION_SELECTOR_K40_TO_K20_20261018" / "results" / "cache_pred" / "kin_outer_seed{seed}.npy"

DT = 0.4
EPS = 1e-6
V0_LOW = 0.15  # m/s
QUALITY_MAIN = (0.2, 0.4)
QUALITY_SOFT = (0.3, 0.5)

# (name, L_m, Theta_deg, H_hist_deg)
THRESHOLDS = [
    ("T1_main", 0.40, 45.0, 12.0),
    ("T2_loose", 0.30, 35.0, 8.0),
    ("T3_tight", 0.60, 60.0, 18.0),
]


def ang_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed angle from a->b in degrees. a,b (...,2)."""
    ang = np.arctan2(b[..., 1], b[..., 0]) - np.arctan2(a[..., 1], a[..., 0])
    ang = (ang + np.pi) % (2 * np.pi) - np.pi
    return np.degrees(ang)


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, EPS)


def coverage_mask(paths: np.ndarray, gt: np.ndarray, ade_t: float, fde_t: float) -> np.ndarray:
    """paths (N,K,12,2), gt (N,12,2) -> covered (N,) bool."""
    e = np.linalg.norm(paths.astype(np.float64) - gt[:, None].astype(np.float64), axis=-1)
    ade = e.mean(-1)
    fde = e[:, :, -1]
    good = (ade <= ade_t) & (fde <= fde_t)
    return good.any(1)


def kinematics(past: np.ndarray, gt: np.ndarray):
    """Per-window sudden / cue features. past (N,8,2), gt (N,12,2)."""
    N = len(past)
    past = past.astype(np.float64)
    gt = gt.astype(np.float64)

    v0 = (past[:, -1] - past[:, -2]) / DT  # (N,2)
    speed0 = np.linalg.norm(v0, axis=-1)
    low = speed0 < V0_LOW

    # fallback direction: first->last past displacement
    disp_h = past[:, -1] - past[:, 0]
    u = unit(v0)
    u_fb = unit(disp_h)
    u = np.where(low[:, None], u_fb, u)

    # CV extrapolation
    t = (np.arange(12, dtype=np.float64) + 1.0) * DT  # (12,)
    cv = v0[:, None, :] * t[None, :, None]  # (N,12,2)

    # lateral of gt from heading u (through origin)
    cross = gt[..., 0] * u[:, None, 1] - gt[..., 1] * u[:, None, 0]  # (N,12)
    max_lat = np.max(np.abs(cross), axis=1)
    max_dev = np.max(np.linalg.norm(gt - cv, axis=-1), axis=1)

    # GT velocities
    # prepend origin as step -1 position for first gt vel from 0
    gt_ext = np.concatenate([np.zeros((N, 1, 2)), gt], axis=1)
    v_gt = np.diff(gt_ext, axis=1) / DT  # (N,12,2)
    v_gt_early = v_gt[:, :4].mean(1)
    v_gt_late = v_gt[:, -4:].mean(1)
    h_early = ang_diff_deg(v0, v_gt_early)
    h_late = ang_diff_deg(v0, v_gt_late)
    heading_delta = np.maximum(np.abs(h_early), np.abs(h_late))

    # future turn sign: sign of mean lateral of gt (or late heading)
    mean_lat = cross.mean(1)
    fut_sign = np.sign(mean_lat)
    fut_sign = np.where(fut_sign == 0, np.sign(h_late), fut_sign)

    # history turns from past velocities
    v_h = np.diff(past, axis=1) / DT  # (N,7,2)
    # angle changes between consecutive hist vels
    turns = []
    for k in range(v_h.shape[1] - 1):
        turns.append(ang_diff_deg(v_h[:, k], v_h[:, k + 1]))
    turns = np.stack(turns, axis=1)  # (N,6)
    hist_turn = turns[:, -3:].sum(1)  # last 3 segments
    hist_turn_abs = np.abs(hist_turn)
    hist_sign = np.sign(hist_turn)
    hist_sign = np.where(hist_sign == 0, 0.0, hist_sign)

    speed_gt = np.linalg.norm(v_gt, axis=-1).mean(1)
    speed_ratio = speed_gt / np.maximum(speed0, EPS)

    return dict(
        v0=v0,
        speed0=speed0,
        low_speed=low,
        max_lat=max_lat,
        max_dev=max_dev,
        heading_delta_deg=heading_delta,
        speed_ratio=speed_ratio,
        hist_turn_deg=hist_turn,
        hist_turn_abs=hist_turn_abs,
        hist_sign=hist_sign,
        fut_sign=fut_sign,
        h_early=h_early,
        h_late=h_late,
    )


def sudden_and_cue(feat: dict, L: float, Theta: float, H: float):
    low = feat["low_speed"]
    # normal: lat or heading
    sud = (feat["max_lat"] >= L) | (feat["heading_delta_deg"] >= Theta)
    # low-speed: use max_dev
    sud = np.where(low, feat["max_dev"] >= L, sud)
    same = (feat["hist_sign"] != 0) & (feat["hist_sign"] == feat["fut_sign"])
    cue = (feat["hist_turn_abs"] >= H) & same
    unfore = sud & (~cue)
    return sud.astype(bool), cue.astype(bool), unfore.astype(bool)


def summarize_block(no_cover: np.ndarray, sud: np.ndarray, cue: np.ndarray, unfore: np.ndarray, N: int):
    nc = no_cover
    n_nc = int(nc.sum())
    n_uf = int((nc & unfore).sum())
    n_sud_cue = int((nc & sud & cue).sum())
    n_sud_nocue = int((nc & sud & (~cue)).sum())
    n_not_sud = int((nc & (~sud)).sum())
    return {
        "N": N,
        "n_no_cover": n_nc,
        "frac_no_cover": float(n_nc / N) if N else None,
        "n_unfore_in_no_cover": n_uf,
        "frac_unfore_among_no_cover": float(n_uf / n_nc) if n_nc else None,
        "frac_unfore_of_all": float(n_uf / N) if N else None,
        "breakdown_no_cover": {
            "not_sudden": n_not_sud,
            "sudden_and_cued": n_sud_cue,
            "sudden_uncued_unforeseeable": n_sud_nocue,
        },
        "n_sudden_all": int(sud.sum()),
        "n_unfore_all": int(unfore.sum()),
        "n_cued_all": int(cue.sum()),
    }


def run_seed(seed: int = 42) -> dict:
    z = np.load(VIS)
    # arrays.npz is built for seed42; for other seeds rebuild pool from caches
    if seed == 42:
        past = z["past"]
        gt = z["future"]
        native = z["native"]
        groups = z["change_group"]
        sample_id = z["sample_id"]
        kin = np.load(str(KIN_TMPL).format(seed=seed))
        assert kin.shape[0] == len(past)
    else:
        # load full caches and outer indices
        from importlib.util import spec_from_file_location, module_from_spec
        import sys
        k0_dir = EXEC / "k0"
        sys.path.insert(0, str(EXEC))
        from k0 import data_io
        data = data_io.load_bundle()
        ix = data["split"]["outer"]
        past = data["past"][ix]
        gt = data["gt"][ix]
        native = data["native"][ix]
        groups = data["groups"][ix]
        sample_id = data["sample_id"][ix]
        kin, _ = data_io.load_kin("outer", seed, data)
        if kin is None:
            return {"seed": seed, "status": "BLOCKED_NO_KIN"}

    N = len(past)
    pool40 = np.concatenate([native.astype(np.float64), kin.astype(np.float64)], axis=1)
    feat = kinematics(past, gt)

    out = {
        "seed": seed,
        "N": N,
        "split": "outer",
        "dt": DT,
        "quality_main": list(QUALITY_MAIN),
        "quality_soft": list(QUALITY_SOFT),
        "thresholds": {
            name: {"L_m": L, "Theta_deg": Th, "H_hist_deg": H}
            for name, L, Th, H in THRESHOLDS
        },
        "n_low_speed": int(feat["low_speed"].sum()),
        "feature_quantiles": {
            "max_lat_p50_p90": [float(np.quantile(feat["max_lat"], q)) for q in (0.5, 0.9)],
            "heading_delta_p50_p90": [float(np.quantile(feat["heading_delta_deg"], q)) for q in (0.5, 0.9)],
            "hist_turn_abs_p50_p90": [float(np.quantile(feat["hist_turn_abs"], q)) for q in (0.5, 0.9)],
        },
        "results": {},
    }

    sets = {
        "native_K20": native.astype(np.float64),
        "K40_union": pool40,
    }
    # optional selected-set contrast from visual (seed42 only)
    if seed == 42:
        for key in ("k1_S", "k0c", "deploy_B"):
            if key in z.files:
                sets[f"selected_{key}"] = z[key].astype(np.float64)

    for qname, (ade_t, fde_t) in (("main_0.2_0.4", QUALITY_MAIN), ("soft_0.3_0.5", QUALITY_SOFT)):
        out["results"][qname] = {}
        cov = {sname: coverage_mask(paths, gt, ade_t, fde_t) for sname, paths in sets.items()}
        for tname, L, Th, H in THRESHOLDS:
            sud, cue, unfore = sudden_and_cue(feat, L, Th, H)
            block = {"params": {"L_m": L, "Theta_deg": Th, "H_hist_deg": H}}
            for sname, covered in cov.items():
                no_cover = ~covered
                block[sname] = summarize_block(no_cover, sud, cue, unfore, N)
                # change-group slice for native/K40 only
                if sname in ("native_K20", "K40_union") and groups is not None:
                    ch = groups == 2  # change; verify encoding
                    # change_group: typically 0 stable, 1 transition, 2 change — check
                    block[sname]["by_change_group"] = {}
                    for gval, gname in ((0, "stable"), (1, "transition"), (2, "change")):
                        m = groups == gval
                        if not m.any():
                            continue
                        sub = summarize_block(no_cover[m], sud[m], cue[m], unfore[m], int(m.sum()))
                        block[sname]["by_change_group"][gname] = sub
            out["results"][qname][tname] = block

    # group encoding sanity
    out["change_group_counts"] = {
        str(int(g)): int((groups == g).sum()) for g in np.unique(groups)
    }
    out["sample_id_sha16"] = __import__("hashlib").sha256(
        np.ascontiguousarray(sample_id).tobytes()
    ).hexdigest()[:16]
    return out


def to_md(rec: dict) -> str:
    lines = []
    lines.append(f"# 无好覆盖 ∩ 不可预见 估算结果（seed{rec['seed']} outer）")
    lines.append("")
    lines.append(f"- N={rec['N']}，低速窗(‖v0‖<{V0_LOW})={rec['n_low_speed']}")
    lines.append(f"- 主质量门 ADE/FDE={rec['quality_main']}；软门={rec['quality_soft']}")
    lines.append(f"- change_group 计数: {rec['change_group_counts']}")
    lines.append(f"- 特征分位 max_lat p50/p90={rec['feature_quantiles']['max_lat_p50_p90']}；"
                 f"headingΔ={rec['feature_quantiles']['heading_delta_p50_p90']}；"
                 f"|hist_turn|={rec['feature_quantiles']['hist_turn_abs_p50_p90']}")
    lines.append("")
    for qname, qblock in rec["results"].items():
        lines.append(f"## 质量门 `{qname}`")
        lines.append("")
        # main table for T1
        t1 = qblock["T1_main"]
        lines.append("### T1 主阈值（L=0.40m，Θ=45°，H=12°）")
        lines.append("")
        lines.append("| 集合 | 无覆盖 | 其中不可预见 | 占无覆盖 | 占全测 | 无覆盖内:非急变 / 急变有预兆 / 急变无预兆 |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for sname in t1:
            if sname == "params":
                continue
            r = t1[sname]
            if "n_no_cover" not in r:
                continue
            bd = r["breakdown_no_cover"]
            lines.append(
                f"| {sname} | {r['n_no_cover']} | {r['n_unfore_in_no_cover']} | "
                f"{r['frac_unfore_among_no_cover']:.3f} | {r['frac_unfore_of_all']:.3f} | "
                f"{bd['not_sudden']} / {bd['sudden_and_cued']} / {bd['sudden_uncued_unforeseeable']} |"
            )
        lines.append("")
        # sensitivity
        lines.append("### 阈值敏感性（native_K20 / K40_union）")
        lines.append("")
        lines.append("| 阈值组 | 集合 | 无覆盖 | 不可预见∩无覆盖 | 占无覆盖 |")
        lines.append("|---|---|---:|---:|---:|")
        for tname in ("T1_main", "T2_loose", "T3_tight"):
            blk = qblock[tname]
            for sname in ("native_K20", "K40_union"):
                r = blk[sname]
                lines.append(
                    f"| {tname} | {sname} | {r['n_no_cover']} | {r['n_unfore_in_no_cover']} | "
                    f"{r['frac_unfore_among_no_cover']:.3f} |"
                )
        lines.append("")
        # change group for T1 K40
        if "by_change_group" in t1.get("K40_union", {}):
            lines.append("### T1 × K40_union × change_group")
            lines.append("")
            lines.append("| 组 | N | 无覆盖 | 不可预见∩无覆盖 | 占无覆盖 |")
            lines.append("|---|---:|---:|---:|---:|")
            for gname, r in t1["K40_union"]["by_change_group"].items():
                frac = r["frac_unfore_among_no_cover"]
                frac_s = f"{frac:.3f}" if frac is not None else "NA"
                lines.append(
                    f"| {gname} | {r['N']} | {r['n_no_cover']} | {r['n_unfore_in_no_cover']} | {frac_s} |"
                )
            lines.append("")
    lines.append("## 解读要点")
    lines.append("")
    lines.append("- **不可预见∩无覆盖**：开环设定下，历史无同号转弯预兆的急变，难要求单条贴 GT；这部分不宜单独归因选择器。")
    lines.append("- **急变∧有预兆仍无覆盖**：更偏向池/生成器未给出对应模态。")
    lines.append("- **非急变仍无覆盖**：门限误差、其它多模态等原因。")
    lines.append("- ETH 无地图/邻居 → 可能高估不可预见；见设计文档局限节。")
    lines.append("")
    return "\n".join(lines)


def main():
    rec = run_seed(42)
    jpath = EXEC / "UNFORESEEABLE_NO_COVER_ESTIMATE_seed42.json"
    mpath = EXEC / "UNFORESEEABLE_NO_COVER_ESTIMATE_seed42.md"
    jpath.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    mpath.write_text(to_md(rec), encoding="utf-8")
    print("WROTE", jpath)
    print("WROTE", mpath)
    # print T1 main headline
    t1 = rec["results"]["main_0.2_0.4"]["T1_main"]
    for s in ("native_K20", "K40_union"):
        r = t1[s]
        print(
            f"{s}: no_cover={r['n_no_cover']} unfore_in_nc={r['n_unfore_in_no_cover']} "
            f"frac_nc={r['frac_unfore_among_no_cover']:.3f}"
        )


if __name__ == "__main__":
    main()
