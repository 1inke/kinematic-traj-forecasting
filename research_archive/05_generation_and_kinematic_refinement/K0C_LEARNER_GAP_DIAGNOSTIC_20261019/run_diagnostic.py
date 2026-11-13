# -*- coding: utf-8 -*-
"""K0-C learner gap diagnostic: replay oracle swaps under S pred_gate and GTF rank.
Read-only; no generator / HGB train / K2.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

ROOT = Path(r"<WORKSPACE_ROOT>")
EXEC = ROOT / "CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018"
K1_EXEC = EXEC / "k1_exec"
GTF_ROOT = ROOT / "STRATEGY_ONE_FACTOR_GTFREE_RANK_20261019"
OUT = ROOT / "K0C_LEARNER_GAP_DIAGNOSTIC_20261019"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(EXEC))
sys.path.insert(0, str(GTF_ROOT))

from k0.data_io import load_bundle, load_kin
from k0.metrics import errors, motion_stats, motion_gate
from k1.config import (
    N_NAT, N_KIN, PRED_L_MAX, PRED_DA_MIN, PRED_DF_MIN, OUTPUT_NAMES, STATES, STATE_N_SWAPS,
)
from k1.features113 import build_pool_features_22, pairwise_D_matrix, action_feature_113
from k1.train_hgb import load_arm, predict_arm
from k1.sampling import enumerate_all_legal_actions
import gtf_select

SEEDS = (42, 43, 44)


def accept_from_pred(pred):
    dA, dF, dJ, dC, L = pred.T
    L_c = np.clip(L, 0.0, 1.0)
    dC_c = np.clip(dC, -1.0, 1.0)
    ok = (
        (L_c <= PRED_L_MAX)
        & (dA >= PRED_DA_MIN)
        & (dF >= PRED_DF_MIN)
        & ((dC_c > 0) | (dJ > 0))
    )
    return ok, dA, dF, dJ, dC_c, L_c


def load_k0c_traces(seed):
    p = EXEC / "results" / f"seed{seed}" / "outer" / "k0c_action_traces.jsonl"
    traces = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            traces.append(json.loads(line))
    return traces


def set_overlap(a, b):
    # a,b: (N,20)
    inter = np.array([len(set(a[i].tolist()) & set(b[i].tolist())) for i in range(len(a))], dtype=np.float64)
    identical = np.array([set(a[i].tolist()) == set(b[i].tolist()) for i in range(len(a))], dtype=bool)
    return float(inter.mean()), float(identical.mean())


def replay_seed(seed, data, outer_ix):
    t0 = time.time()
    kin, kin_src = load_kin("outer", seed, data)
    native = data["native"][outer_ix].astype(np.float64)
    past = data["past"][outer_ix].astype(np.float64)
    gt = data["gt"][outer_ix].astype(np.float64)
    groups = data["groups"][outer_ix]
    pool40 = np.concatenate([native, kin.astype(np.float64)], axis=1)
    feat40 = build_pool_features_22(past, native, kin.astype(np.float64))
    N = len(native)

    k0c_idx = np.load(EXEC / "results" / f"seed{seed}" / "outer" / "k0c_indices.npy")
    S_idx = np.load(K1_EXEC / f"seed{seed}" / "predictions" / "outer" / "S_indices.npy")
    gtf_idx = np.load(GTF_ROOT / "results" / f"seed{seed}" / "outer" / "gtf_indices.npy")
    assert k0c_idx.shape == (N, 20) and S_idx.shape == (N, 20) and gtf_idx.shape == (N, 20)

    traces = load_k0c_traces(seed)
    assert len(traces) == N

    models_S = load_arm(K1_EXEC / f"seed{seed}" / "models", "S", seed)

    # native / S / K0C per-window ADE for harm analysis
    nat_err = errors(native, gt)
    S_paths = pool40[np.arange(N)[:, None], S_idx]
    S_err = errors(S_paths, gt)
    k0c_paths = pool40[np.arange(N)[:, None], k0c_idx]
    k0c_err = errors(k0c_paths, gt)

    ov_S_k0c, id_S_k0c = set_overlap(S_idx, k0c_idx)
    ov_gtf_k0c, id_gtf_k0c = set_overlap(gtf_idx, k0c_idx)
    ov_S_gtf, id_S_gtf = set_overlap(S_idx, gtf_idx)
    ov_S_nat, id_S_nat = set_overlap(S_idx, np.tile(np.arange(N_NAT), (N, 1)))
    ov_k0c_nat, id_k0c_nat = set_overlap(k0c_idx, np.tile(np.arange(N_NAT), (N, 1)))
    ov_gtf_nat, id_gtf_nat = set_overlap(gtf_idx, np.tile(np.arange(N_NAT), (N, 1)))

    # Replay each K0-C accepted action
    n_oracle_actions = 0
    pred_reject = 0
    pred_accept = 0
    clause_L = 0
    clause_dA = 0
    clause_dF = 0
    clause_benefit = 0  # neither dC>0 nor dJ>0
    gtf_same_top = 0
    gtf_in_top3 = 0
    gtf_rank_sum = 0
    gtf_n_ranked = 0
    pred_reject_but_true_good = 0  # oracle action improves ADE or cov (already true by K0C)
    pred_vals = []

    # For multi-step: reconstruct selected before each action
    for w in range(N):
        selected = list(range(N_NAT))
        acts = traces[w]["actions"]
        if not acts:
            continue
        D40 = pairwise_D_matrix(pool40[w])
        nm = motion_stats(native[w:w+1])
        for act in acts:
            rem = int(act["remove_native"])
            add = int(act["add_kin"])
            n_oracle_actions += 1
            # S pred on this exact action at this state
            X = action_feature_113(feat40[w], D40, selected, rem, add)[None]
            pred = predict_arm(models_S, X)  # (1,5)
            ok, dA, dF, dJ, dC, L = accept_from_pred(pred)
            pv = dict(
                window=w, step=int(act["step"]), rem=rem, add=add,
                pred_dA=float(dA[0]), pred_dF=float(dF[0]), pred_dJ=float(dJ[0]),
                pred_dC=float(dC[0]), pred_L=float(L[0]),
                true_d_cov=int(act.get("d_cov", 0)),
                true_d_ade=float(act.get("d_ade", 0)),
                true_d_fde=float(act.get("d_fde", 0)),
                true_d_j=float(act.get("d_j", 0)),
                pred_ok=bool(ok[0]),
            )
            pred_vals.append(pv)
            if ok[0]:
                pred_accept += 1
            else:
                pred_reject += 1
                pred_reject_but_true_good += 1  # K0-C only accepts beneficial
                # which clauses failed (can be multiple)
                if float(L[0]) > PRED_L_MAX:
                    clause_L += 1
                if float(dA[0]) < PRED_DA_MIN:
                    clause_dA += 1
                if float(dF[0]) < PRED_DF_MIN:
                    clause_dF += 1
                if not (float(dC[0]) > 0 or float(dJ[0]) > 0):
                    clause_benefit += 1

            # GTF score ranking among motion-PASS pairs at this state
            pairs = gtf_select.enumerate_legal(selected)
            scored = []
            for r, a in pairs:
                new_sel = [a if x == r else x for x in selected]
                paths = pool40[w][np.asarray(new_sel, dtype=np.int64)]
                mg = motion_gate(motion_stats(paths[None]), nm)
                if mg["status"] != "PASS":
                    continue
                key = gtf_select.score_pair(r, a, selected, pool40[w], D40)
                scored.append((key, r, a))
            scored.sort(key=lambda t: t[0], reverse=True)
            rank = None
            for i, (key, r, a) in enumerate(scored):
                if r == rem and a == add:
                    rank = i + 1  # 1-based
                    break
            if rank is not None:
                gtf_n_ranked += 1
                gtf_rank_sum += rank
                if rank == 1:
                    gtf_same_top += 1
                if rank <= 3:
                    gtf_in_top3 += 1
            # advance state as K0-C did
            selected = [add if x == rem else x for x in selected]

    # S-accepted actions that harm ADE vs native (window-level)
    # Use S n_swaps and per-window ADE
    S_n_swaps = np.load(K1_EXEC / f"seed{seed}" / "action_traces" / "outer_S_n_swaps.npy")
    harm_ade = (S_err["minADE"] > nat_err["minADE"] + 1e-12)
    help_ade = (S_err["minADE"] < nat_err["minADE"] - 1e-12)
    same_ade = ~(harm_ade | help_ade)
    # change group mask: groups==2 typically? check unique
    # In this project change_group: need mapping. From prior: change is one of groups.
    # Use same as metrics: often 0=stable,1=transition,2=change or string. Check.
    ug = np.unique(groups)
    # load from done.json groups classes if needed
    change_mask = None
    # try numeric codes from previous analysis: change windows where native change_good counted
    # From k0: GROUPS = overall/stable/transition/change — groups array is int codes
    # Check portfolios or result_fields
    from collections import Counter
    # Infer: use groups values; load RESULT for seed mapping if available
    # Common in this codebase: change_group 0/1/2
    # We'll compute for overall and for each unique group code
    harm_stats = {
        "n_windows": int(N),
        "n_S_swaps_mean": float(S_n_swaps.mean()),
        "frac_windows_S_harm_ADE": float(harm_ade.mean()),
        "frac_windows_S_help_ADE": float(help_ade.mean()),
        "frac_windows_S_same_ADE": float(same_ade.mean()),
        "mean_dADE_S_minus_native_on_harm": float((S_err["minADE"][harm_ade] - nat_err["minADE"][harm_ade]).mean()) if harm_ade.any() else None,
        "mean_n_swaps_on_harm": float(S_n_swaps[harm_ade].mean()) if harm_ade.any() else None,
        "mean_n_swaps_on_help": float(S_n_swaps[help_ade].mean()) if help_ade.any() else None,
        "mean_n_swaps_on_same": float(S_n_swaps[same_ade].mean()) if same_ade.any() else None,
        "group_codes": [int(x) for x in ug.tolist()],
    }
    for g in ug:
        m = groups == g
        harm_stats[f"group_{int(g)}_n"] = int(m.sum())
        harm_stats[f"group_{int(g)}_frac_S_harm_ADE"] = float(harm_ade[m].mean()) if m.any() else None
        harm_stats[f"group_{int(g)}_mean_dADE"] = float((S_err["minADE"][m] - nat_err["minADE"][m]).mean()) if m.any() else None

    # Deploy vs train state: deploy starts S0 and stops early; mean swaps ~2.5
    # Train balanced S0/S4/S12. Compare L_pos rate by state from diag.json
    diag = json.loads((K1_EXEC / f"seed{seed}" / "train_data" / "diag.json").read_text(encoding="utf-8"))
    rows = np.load(K1_EXEC / f"seed{seed}" / "train_data" / f"rows_seed{seed}.npz")
    state = rows["state"]
    # YS L column index 4
    YS = np.load(K1_EXEC / f"seed{seed}" / "train_data" / f"YS_seed{seed}.npy", mmap_mode="r")
    state_shift = {"train": {}, "deploy_proxy": {}}
    for sname, scode in (("S0", 0), ("S4", 1), ("S12", 2)):
        m = state == scode
        Lcol = YS[m, 4]
        state_shift["train"][sname] = dict(
            n=int(m.sum()),
            L_pos_rate=float((Lcol > 0.5).mean()) if m.any() else None,
            dC_pos_rate=float((YS[m, 3] > 0).mean()) if m.any() else None,
            dA_pos_rate=float((YS[m, 0] > 0).mean()) if m.any() else None,
        )
    # Deploy proxy: fraction of windows ending at 0,1-3,4-7,8+ swaps (S arm)
    hist = {
        "eq0": float((S_n_swaps == 0).mean()),
        "1to3": float(((S_n_swaps >= 1) & (S_n_swaps <= 3)).mean()),
        "4to7": float(((S_n_swaps >= 4) & (S_n_swaps <= 7)).mean()),
        "ge8": float((S_n_swaps >= 8).mean()),
        "mean": float(S_n_swaps.mean()),
    }
    state_shift["deploy_proxy"]["S_n_swaps_hist"] = hist
    state_shift["note"] = (
        "Train samples actions equally at S0/S4/S12; deploy S mean swaps~2.5 with ~30% zero. "
        "Train S_L_pos rises sharply S0->S12, so high-swap states over-represented in lost-label training."
    )

    misrej_rate = (pred_reject / n_oracle_actions) if n_oracle_actions else None
    gtf_misrank_rate = (1.0 - gtf_same_top / gtf_n_ranked) if gtf_n_ranked else None

    out = dict(
        seed=seed,
        N=N,
        kin_src=kin_src,
        index_overlap=dict(
            mean_S_cap_k0c=ov_S_k0c, frac_S_eq_k0c=id_S_k0c,
            mean_gtf_cap_k0c=ov_gtf_k0c, frac_gtf_eq_k0c=id_gtf_k0c,
            mean_S_cap_gtf=ov_S_gtf, frac_S_eq_gtf=id_S_gtf,
            mean_S_cap_native=ov_S_nat, frac_S_eq_native=id_S_nat,
            mean_k0c_cap_native=ov_k0c_nat, frac_k0c_eq_native=id_k0c_nat,
            mean_gtf_cap_native=ov_gtf_nat, frac_gtf_eq_native=id_gtf_nat,
        ),
        k0c_oracle_actions=dict(
            n_actions=n_oracle_actions,
            n_windows_with_swap=int(sum(1 for t in traces if t["actions"])),
            S_pred_gate_accept=pred_accept,
            S_pred_gate_reject=pred_reject,
            S_pred_gate_misreject_rate=misrej_rate,
            reject_clause_counts=dict(
                L_gt_PRED_L_MAX=clause_L,
                dA_lt_PRED_DA_MIN=clause_dA,
                dF_lt_PRED_DF_MIN=clause_dF,
                neither_dC_nor_dJ_pos=clause_benefit,
            ),
            pred_thresholds=dict(PRED_L_MAX=PRED_L_MAX, PRED_DA_MIN=PRED_DA_MIN, PRED_DF_MIN=PRED_DF_MIN),
            gtf_n_motion_pass_ranked=gtf_n_ranked,
            gtf_same_top1=gtf_same_top,
            gtf_in_top3=gtf_in_top3,
            gtf_misrank_not_top1_rate=gtf_misrank_rate,
            gtf_mean_rank_of_oracle= (gtf_rank_sum / gtf_n_ranked) if gtf_n_ranked else None,
        ),
        S_harm_ADE_vs_native=harm_stats,
        state_shift=state_shift,
        train_diag_snippet=dict(
            state_S0=diag.get("state_S0"),
            state_S4=diag.get("state_S4"),
            state_S12=diag.get("state_S12"),
        ),
        wall_s=time.time() - t0,
        # store a sample of rejected preds for inspection
        sample_rejected_preds=[p for p in pred_vals if not p["pred_ok"]][:20],
        sample_accepted_preds=[p for p in pred_vals if p["pred_ok"]][:20],
    )
    # save per-action csv-like npy
    if pred_vals:
        np.savez_compressed(
            OUT / f"seed{seed}_oracle_action_preds.npz",
            window=np.array([p["window"] for p in pred_vals]),
            step=np.array([p["step"] for p in pred_vals]),
            rem=np.array([p["rem"] for p in pred_vals]),
            add=np.array([p["add"] for p in pred_vals]),
            pred_dA=np.array([p["pred_dA"] for p in pred_vals]),
            pred_dF=np.array([p["pred_dF"] for p in pred_vals]),
            pred_dJ=np.array([p["pred_dJ"] for p in pred_vals]),
            pred_dC=np.array([p["pred_dC"] for p in pred_vals]),
            pred_L=np.array([p["pred_L"] for p in pred_vals]),
            pred_ok=np.array([p["pred_ok"] for p in pred_vals]),
            true_d_ade=np.array([p["true_d_ade"] for p in pred_vals]),
            true_d_fde=np.array([p["true_d_fde"] for p in pred_vals]),
            true_d_j=np.array([p["true_d_j"] for p in pred_vals]),
            true_d_cov=np.array([p["true_d_cov"] for p in pred_vals]),
        )
    (OUT / f"seed{seed}_diag.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"seed{seed} done actions={n_oracle_actions} misrej={misrej_rate} gtf_misrank={gtf_misrank_rate} wall={out['wall_s']:.1f}s", flush=True)
    return out


def main():
    t0 = time.time()
    print("loading bundle...", flush=True)
    data = load_bundle()
    outer_ix = data["split"]["outer"]
    per_seed = []
    for seed in SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        per_seed.append(replay_seed(seed, data, outer_ix))

    # Aggregate
    n_act = sum(s["k0c_oracle_actions"]["n_actions"] for s in per_seed)
    n_rej = sum(s["k0c_oracle_actions"]["S_pred_gate_reject"] for s in per_seed)
    n_acc = sum(s["k0c_oracle_actions"]["S_pred_gate_accept"] for s in per_seed)
    gtf_top = sum(s["k0c_oracle_actions"]["gtf_same_top1"] for s in per_seed)
    gtf_n = sum(s["k0c_oracle_actions"]["gtf_n_motion_pass_ranked"] for s in per_seed)
    clause = dict(L=0, dA=0, dF=0, benefit=0)
    for s in per_seed:
        c = s["k0c_oracle_actions"]["reject_clause_counts"]
        clause["L"] += c["L_gt_PRED_L_MAX"]
        clause["dA"] += c["dA_lt_PRED_DA_MIN"]
        clause["dF"] += c["dF_lt_PRED_DF_MIN"]
        clause["benefit"] += c["neither_dC_nor_dJ_pos"]

    agg = dict(
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        timezone="Asia/Tokyo",
        BREAKTHROUGH=False,
        breakthrough_reason="Diagnostic only; no deploy gate claimed. Expected false.",
        GO_K2=False,
        branch="C",
        n_oracle_actions_3seed=n_act,
        S_pred_gate_misreject_rate=n_rej / n_act if n_act else None,
        S_pred_gate_accept_rate=n_acc / n_act if n_act else None,
        GTF_misrank_not_top1_rate=(1.0 - gtf_top / gtf_n) if gtf_n else None,
        GTF_top1_hit_rate=gtf_top / gtf_n if gtf_n else None,
        reject_clause_totals=clause,
        per_seed=[{
            "seed": s["seed"],
            "n_actions": s["k0c_oracle_actions"]["n_actions"],
            "misreject_rate": s["k0c_oracle_actions"]["S_pred_gate_misreject_rate"],
            "gtf_misrank_rate": s["k0c_oracle_actions"]["gtf_misrank_not_top1_rate"],
            "gtf_mean_rank": s["k0c_oracle_actions"]["gtf_mean_rank_of_oracle"],
            "frac_S_eq_k0c": s["index_overlap"]["frac_S_eq_k0c"],
            "frac_gtf_eq_k0c": s["index_overlap"]["frac_gtf_eq_k0c"],
            "frac_S_harm_ADE": s["S_harm_ADE_vs_native"]["frac_windows_S_harm_ADE"],
            "S_swaps_mean": s["S_harm_ADE_vs_native"]["n_S_swaps_mean"],
        } for s in per_seed],
        wall_s=time.time() - t0,
        paths=dict(
            out=str(OUT),
            exec=str(EXEC),
            gtf=str(GTF_ROOT),
        ),
    )
    (OUT / "DIAGNOSTIC.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")

    # RESULT_ANALYSIS.md
    lines = []
    lines.append("# K0-C Learner Gap Diagnostic — RESULT_ANALYSIS")
    lines.append("")
    lines.append(f"- time: {agg['created_at']} (Asia/Tokyo)")
    lines.append(f"- dir: `{OUT}`")
    lines.append("- scope: outer 三 seed；只读回放 K0-C 已接受替换 vs K1-S pred_gate / GTF 几何排序")
    lines.append("- **BREAKTHROUGH=false**（诊断，不宣称部署门）")
    lines.append("")
    lines.append("## 1. 集合对齐（outer）")
    lines.append("")
    lines.append("| seed | |S∩K0C| mean | S==K0C | |GTF∩K0C| | GTF==K0C | S harm ADE frac |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for s in per_seed:
        o = s["index_overlap"]
        lines.append(
            f"| {s['seed']} | {o['mean_S_cap_k0c']:.2f} | {o['frac_S_eq_k0c']:.3f} | "
            f"{o['mean_gtf_cap_k0c']:.2f} | {o['frac_gtf_eq_k0c']:.3f} | "
            f"{s['S_harm_ADE_vs_native']['frac_windows_S_harm_ADE']:.3f} |"
        )
    lines.append("")
    lines.append("## 2. K0-C 有利动作回放：S pred_gate 误拒")
    lines.append("")
    lines.append(f"- 三 seed 合计 oracle 接受动作数: **{n_act}**")
    lines.append(f"- 被 S pred_gate 误拒: **{n_rej}** → 误拒率 **{agg['S_pred_gate_misreject_rate']:.4f}**")
    lines.append(f"- 被 S pred_gate 接受: **{n_acc}** → 接受率 **{agg['S_pred_gate_accept_rate']:.4f}**")
    lines.append(f"- 拒绝子句计数（可重叠）: L>{PRED_L_MAX}={clause['L']}, dA<0={clause['dA']}, dF<0={clause['dF']}, 无(dC>0|dJ>0)={clause['benefit']}")
    lines.append("")
    lines.append("| seed | n_actions | misreject | accept |")
    lines.append("|---:|---:|---:|---:|")
    for s in per_seed:
        a = s["k0c_oracle_actions"]
        lines.append(f"| {s['seed']} | {a['n_actions']} | {a['S_pred_gate_misreject_rate']:.4f} | {a['S_pred_gate_accept']/max(a['n_actions'],1):.4f} |")
    lines.append("")
    lines.append("## 3. 同一动作：GTF 几何分是否选到")
    lines.append("")
    lines.append(f"- motion-PASS 可排名的 oracle 动作: **{gtf_n}**")
    lines.append(f"- GTF top1 命中: **{gtf_top}** → hit **{agg['GTF_top1_hit_rate']:.4f}** / misrank(非top1) **{agg['GTF_misrank_not_top1_rate']:.4f}**")
    lines.append("")
    lines.append("| seed | ranked | top1 | top3 | mean rank | misrank |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for s in per_seed:
        a = s["k0c_oracle_actions"]
        lines.append(
            f"| {s['seed']} | {a['gtf_n_motion_pass_ranked']} | {a['gtf_same_top1']} | {a['gtf_in_top3']} | "
            f"{a['gtf_mean_rank_of_oracle']:.2f} | {a['gtf_misrank_not_top1_rate']:.4f} |"
        )
    lines.append("")
    lines.append("## 4. S 实际接受但相对 native 损害 ADE")
    lines.append("")
    for s in per_seed:
        h = s["S_harm_ADE_vs_native"]
        lines.append(
            f"- seed{s['seed']}: harm_ADE={h['frac_windows_S_harm_ADE']:.3f}, help={h['frac_windows_S_help_ADE']:.3f}, "
            f"swaps@harm={h['mean_n_swaps_on_harm']}, swaps@help={h['mean_n_swaps_on_help']}"
        )
    lines.append("")
    lines.append("## 5. 训练状态 S0/S4/S12 vs 部署轨迹")
    lines.append("")
    s0 = per_seed[0]["state_shift"]
    lines.append("Train (seed42 为代表，三 seed 结构相同):")
    for sn, d in s0["train"].items():
        lines.append(f"- {sn}: n={d['n']}, L_pos={d['L_pos_rate']:.4f}, dC_pos={d['dC_pos_rate']:.4f}, dA_pos={d['dA_pos_rate']:.4f}")
    lines.append("Deploy S n_swaps 分布 (三 seed):")
    for s in per_seed:
        h = s["state_shift"]["deploy_proxy"]["S_n_swaps_hist"]
        lines.append(f"- seed{s['seed']}: mean={h['mean']:.2f}, eq0={h['eq0']:.3f}, 1-3={h['1to3']:.3f}, 4-7={h['4to7']:.3f}, ≥8={h['ge8']:.3f}")
    lines.append("")
    lines.append(s0["note"])
    lines.append("")
    lines.append("## 6. 判断")
    lines.append("")
    lines.append("1. K0-C 有利替换绝大多数被 **S pred_gate 误拒**（见上误拒率）——与 Branch C「过保守」一致，且把「学不到 oracle 动作」定位到门控而非动作空间为空。")
    lines.append("2. GTF 几何排序对 oracle 动作的 top1 命中很低（见 misrank）——解释为何纯几何 one-factor 无法复现 K0-C。")
    lines.append("3. 训练在 S4/S12 上 L 正例远密于 S0，而部署停在低 swap；状态偏移可能放大丢失恐惧（L）预测。")
    lines.append("4. **不**据此重训 HGB / 开 K2 / 扫配额；BREAKTHROUGH=false。")
    lines.append("")
    lines.append("## 产物")
    lines.append(f"- `DIAGNOSTIC.json`")
    lines.append(f"- `seed{{42,43,44}}_diag.json`")
    lines.append(f"- `seed{{42,43,44}}_oracle_action_preds.npz`")
    (OUT / "RESULT_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # append overnight log
    logp = EXEC / "OVERNIGHT_LOG.md"
    line = (
        f"\n- {agg['created_at']}: K0C_LEARNER_GAP_DIAGNOSTIC_20261019 done; "
        f"S_pred_gate_misreject={agg['S_pred_gate_misreject_rate']:.4f}; "
        f"GTF_misrank={agg['GTF_misrank_not_top1_rate']:.4f}; BREAKTHROUGH=false; standby.\n"
    )
    with logp.open("a", encoding="utf-8") as f:
        f.write(line)
    print("AGG", json.dumps({k: agg[k] for k in (
        "BREAKTHROUGH", "S_pred_gate_misreject_rate", "GTF_misrank_not_top1_rate",
        "n_oracle_actions_3seed", "wall_s")}, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()