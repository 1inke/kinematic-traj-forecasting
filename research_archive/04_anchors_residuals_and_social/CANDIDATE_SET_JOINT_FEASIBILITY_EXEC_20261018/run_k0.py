#!/usr/bin/env python3
"""K0 joint reachability orchestrator. No generator/K1 training."""
from __future__ import annotations
import os
for _k, _v in (("OMP_NUM_THREADS", "2"), ("MKL_NUM_THREADS", "2"), ("OPENBLAS_NUM_THREADS", "2"), ("NUMEXPR_NUM_THREADS", "2"), ("VECLIB_MAXIMUM_THREADS", "2")):
    os.environ.setdefault(_k, _v)
import argparse
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from k0.config import EXEC, SEEDS, ORACLE_TAG, SIZES
from k0.data_io import load_kin, load_v1_selected_indices, array_sha, file_sha
from k0.freeze_inputs import run_freeze
from k0.metrics import evaluate_set, check_W, gather, errors
from k0.k0a import run_k0a
from k0.k0b_ilp import build_instance, solve_instance, verify_solution, preflight_64
from k0.k0c_greedy import run_k0c
from k0.serialize import dump, eval_public, jsonable


def slim_W(w):
    return jsonable(w)


def portfolio_block(name, paths, gt, native, groups, indices=None, extra=None, do_cluster=False):
    ev = evaluate_set(paths, gt, native, groups, do_cluster=do_cluster)
    w = check_W(ev, groups)
    blk = dict(
        name=name,
        eval=eval_public(ev),
        W=slim_W(w),
        W_pass=bool(w["pass_"]),
    )
    if indices is not None:
        blk["indices_sha"] = array_sha(indices.astype(np.int64))
    if extra:
        blk.update(extra)
    return blk


def run_split(seed, split_name, pool40, gt, native, groups, ids, budgets, out_split: Path,
              v1_indices=None, do_k0b=True, do_cluster_outer=False):
    out_split.mkdir(parents=True, exist_ok=True)
    do_cl = do_cluster_outer and split_name == "outer"
    result = dict(
        seed=seed, split=split_name, N=int(len(ids)),
        tag=ORACLE_TAG,
        pool_joint_witness=None,
        policy_oracle_witness=None,
        relaxation_certificate_status=None,
    )
    # baselines
    ports = {}
    ports["native20"] = portfolio_block("native20", native, gt, native, groups, do_cluster=do_cl)
    ports["kin20"] = portfolio_block("kin20", pool40[:, 20:40], gt, native, groups)
    ports["union40"] = portfolio_block("union40", pool40, gt, native, groups)
    if v1_indices is not None:
        v1_paths = gather(pool40, v1_indices)
        ports["v1_cond_boost_11_9"] = portfolio_block(
            "v1_cond_boost_11_9", v1_paths, gt, native, groups, indices=v1_indices
        )

    # K0-A
    print(f"[{seed}/{split_name}] K0-A", flush=True)
    t0 = time.time()
    k0a = run_k0a(pool40, gt, native, groups, do_cluster=do_cl)
    np.save(out_split / "k0a_indices.npy", k0a["indices"])
    dump(out_split / "k0a_match.json", k0a["match_union_quality"])
    dump(out_split / "k0a_W.json", k0a["W"])
    dump(out_split / "k0a_union_W.json", k0a["union_W"])
    ports["k0a"] = portfolio_block("k0a", k0a["paths"], gt, native, groups, indices=k0a["indices"],
                                   extra=dict(
                                       quality_upper_bound_negation=k0a["quality_upper_bound_negation"],
                                       union_quality_fail_clauses=k0a["union_quality_fail_clauses"],
                                       match=k0a["match_union_quality"],
                                       wall_s=time.time() - t0,
                                   ), do_cluster=do_cl)
    result["quality_upper_bound_negation"] = k0a["quality_upper_bound_negation"]
    result["union_W_pass"] = bool(k0a["union_W"]["pass_"])

    # K0-B
    k0b_summary = dict(ran=False)
    if do_k0b and k0a["quality_upper_bound_negation"]:
        do_k0b = False
        result["relaxation_certificate_status"] = "QUALITY_UPPER_BOUND_NEGATION"
        result["pool_joint_witness"] = False
        dump(out_split / "k0b_skipped_quality_ub.json", dict(
            reason="union40 quality clauses already fail W; K20 cannot improve quality beyond union",
            fail_clauses=k0a["union_quality_fail_clauses"],
        ))
    if do_k0b:
        print(f"[{seed}/{split_name}] K0-B R", flush=True)
        tl = budgets["time_limit_s"]
        nl = budgets.get("node_limit")
        k0b_summary = dict(ran=True, mode_tried=[], budgets=budgets)
        witness = None
        cert = "UNKNOWN_WITHIN_BUDGET"
        try:
            t0 = time.time()
            prob, x, c, meta = build_instance(pool40, gt, native, groups, mode="R")
            dump(out_split / "INSTANCE_SPEC_R.json", dict(meta=meta, build_s=time.time() - t0, mode="R"))
            res_r = solve_instance(prob, x, c, meta, time_limit_s=tl, node_limit=nl,
                                   log_path=out_split / "solve_R_log.json")
            k0b_summary["mode_tried"].append("R")
            k0b_summary["R"] = {k: v for k, v in res_r.items() if k != "indices"}
            if res_r.get("has_incumbent"):
                ver = verify_solution(pool40, gt, native, groups, res_r["indices"])
                dump(out_split / "k0b_R_W.json", ver["W"])
                np.save(out_split / "k0b_R_indices.npy", res_r["indices"])
                ports["k0b_R"] = portfolio_block("k0b_R", ver["paths"], gt, native, groups,
                                                 indices=res_r["indices"],
                                                 extra=dict(verified_W_pass=ver["W"]["pass_"]))
                if ver["W"]["pass_"]:
                    witness = "R"
                    cert = "R_INCUMBENT_VERIFIED_W"
                    result["pool_joint_witness"] = True
                else:
                    cert = "R_INCUMBENT_FAILED_TRUE_GATE"
            elif res_r.get("status") == "NUMERIC_INFEASIBILITY_UNVERIFIED":
                cert = "NUMERIC_INFEASIBILITY_UNVERIFIED"
            # try S if R did not give verified W
            if witness is None and res_r.get("status") != "BLOCKED_SOLVER_BACKEND":
                print(f"[{seed}/{split_name}] K0-B S", flush=True)
                t0 = time.time()
                prob_s, x_s, c_s, meta_s = build_instance(pool40, gt, native, groups, mode="S")
                dump(out_split / "INSTANCE_SPEC_S.json", dict(meta=meta_s, build_s=time.time() - t0, mode="S"))
                res_s = solve_instance(prob_s, x_s, c_s, meta_s, time_limit_s=tl, node_limit=nl,
                                       log_path=out_split / "solve_S_log.json")
                k0b_summary["mode_tried"].append("S")
                k0b_summary["S"] = {k: v for k, v in res_s.items() if k != "indices"}
                if res_s.get("has_incumbent"):
                    ver = verify_solution(pool40, gt, native, groups, res_s["indices"])
                    dump(out_split / "k0b_S_W.json", ver["W"])
                    np.save(out_split / "k0b_S_indices.npy", res_s["indices"])
                    ports["k0b_S"] = portfolio_block("k0b_S", ver["paths"], gt, native, groups,
                                                     indices=res_s["indices"],
                                                     extra=dict(verified_W_pass=ver["W"]["pass_"]))
                    if ver["W"]["pass_"]:
                        witness = "S"
                        cert = "S_INCUMBENT_VERIFIED_W"
                        result["pool_joint_witness"] = True
                    elif cert != "R_INCUMBENT_FAILED_TRUE_GATE":
                        cert = "S_INCUMBENT_FAILED_TRUE_GATE"
                elif res_s.get("status") == "NUMERIC_INFEASIBILITY_UNVERIFIED" and witness is None:
                    if cert != "R_INCUMBENT_FAILED_TRUE_GATE":
                        cert = "NUMERIC_INFEASIBILITY_UNVERIFIED"
                elif witness is None and cert == "UNKNOWN_WITHIN_BUDGET":
                    cert = "UNKNOWN_WITHIN_BUDGET"
            if res_r.get("status") == "BLOCKED_SOLVER_BACKEND":
                cert = "BLOCKED_SOLVER_BACKEND"
                result["pool_joint_witness"] = None
        except Exception as e:
            k0b_summary["error"] = f"{type(e).__name__}: {e}"
            k0b_summary["traceback"] = traceback.format_exc()
            cert = "BLOCKED_SOLVER_BACKEND" if "pulp" in str(e).lower() or "solver" in str(e).lower() else "ERROR"
            dump(out_split / "k0b_exception.json", k0b_summary)
        result["relaxation_certificate_status"] = cert
        if result["pool_joint_witness"] is None and witness is None and cert == "UNKNOWN_WITHIN_BUDGET":
            result["pool_joint_witness"] = False  # no witness found (not proof of impossibility)
        k0b_summary["witness"] = witness
        k0b_summary["relaxation_certificate_status"] = cert
        dump(out_split / "k0b_summary.json", k0b_summary)
    else:
        if result.get("relaxation_certificate_status") is None:
            result["relaxation_certificate_status"] = "SKIPPED"
        if result.get("pool_joint_witness") is None and not k0a["quality_upper_bound_negation"]:
            result["pool_joint_witness"] = None

    # If quality UB negation on union, note pool cannot meet those quality clauses
    if k0a["quality_upper_bound_negation"]:
        result["pool_joint_witness"] = False
        result["relaxation_certificate_status"] = (
            result.get("relaxation_certificate_status") or "QUALITY_UPPER_BOUND_NEGATION"
        )
        if result["relaxation_certificate_status"] not in (
            "R_INCUMBENT_VERIFIED_W", "S_INCUMBENT_VERIFIED_W"
        ):
            result["relaxation_certificate_status"] = "QUALITY_UPPER_BOUND_NEGATION"

    # K0-C
    print(f"[{seed}/{split_name}] K0-C", flush=True)
    t0 = time.time()
    k0c = run_k0c(pool40, gt, native, groups, do_cluster=do_cl)
    np.save(out_split / "k0c_indices.npy", k0c["indices"])
    dump(out_split / "k0c_W.json", k0c["W"])
    dump(out_split / "k0c_rejects.json", k0c["reject_totals"])
    # save compact action traces (counts + first/last few)
    dump(out_split / "k0c_traces_meta.json", dict(
        n_windows=len(k0c["action_traces"]),
        n_swaps_mean=k0c["n_swaps_mean"],
        n_swaps_hist={str(i): int(sum(1 for t in k0c["action_traces"] if len(t) == i)) for i in range(21)},
        reject_totals=k0c["reject_totals"],
        wall_s=time.time() - t0,
    ))
    # full traces as npz of variable lists via jsonl
    with (out_split / "k0c_action_traces.jsonl").open("w", encoding="utf-8") as f:
        for b, tr in enumerate(k0c["action_traces"]):
            f.write(json.dumps(dict(window=b, sample_id=str(ids[b]), actions=tr), ensure_ascii=False) + "\n")
    ports["k0c"] = portfolio_block("k0c", k0c["paths"], gt, native, groups, indices=k0c["indices"],
                                   extra=dict(n_swaps_mean=k0c["n_swaps_mean"], rejects=k0c["reject_totals"]))
    result["policy_oracle_witness"] = bool(k0c["W"]["pass_"])

    # If K0-C reaches W, it is also a pool joint witness
    if result["policy_oracle_witness"]:
        result["pool_joint_witness"] = True

    dump(out_split / "portfolios.json", ports)
    result["portfolios"] = {k: dict(W_pass=v["W_pass"], overall_good=v["eval"].get("overall", {}).get("good_count"),
                                    change_good=v["eval"].get("change", {}).get("good_count"),
                                    change_minADE=v["eval"].get("change", {}).get("minADE"),
                                    change_minFDE=v["eval"].get("change", {}).get("minFDE"),
                                    overall_motion=v["eval"].get("overall", {}).get("motion_gate", {}).get("status"),
                                    change_motion=v["eval"].get("change", {}).get("motion_gate", {}).get("status"),
                                    W=v.get("W"))
                            for k, v in ports.items()}
    dump(out_split / "result_fields.json", result)
    return result


def decide_go_k1(all_results):
    """01 §5 trigger table. Register only; do not train."""
    outer = [r for r in all_results if r["split"] == "outer"]
    if len(outer) < 3:
        return dict(GO_K1=False, reason="outer_incomplete", detail="need 3 seeds")
    pool_ok = all(r.get("pool_joint_witness") is True for r in outer)
    policy_ok = all(r.get("policy_oracle_witness") is True for r in outer)
    any_pool = any(r.get("pool_joint_witness") is True for r in outer)
    if pool_ok and policy_ok:
        return dict(GO_K1=True, reason="outer_3seed_pool_and_K0C_W",
                    note="副split完整报告; 记录迁移限制. 本阶段不自动开训.")
    if policy_ok and not pool_ok:
        # K0-C itself is pool positive witness
        return dict(GO_K1=True, reason="K0C_W_implies_pool_witness_solver_maybe_unresolved",
                    note="K0-C达到W即池阳性见证, 可进K1.")
    if any_pool and not policy_ok:
        return dict(GO_K1=False, reason="pool_witness_but_K0C_miss_W",
                    branch="strategy_limit", note="停在策略限制分支; 不训当前K1.")
    # check UB negation
    if all(r.get("quality_upper_bound_negation") for r in outer):
        return dict(GO_K1=False, reason="union_quality_upper_bound_negation",
                    branch="pool_insufficient_quality")
    certs = [r.get("relaxation_certificate_status") for r in outer]
    if all(c == "NUMERIC_INFEASIBILITY_UNVERIFIED" for c in certs) and not any_pool:
        return dict(GO_K1=False, reason="numeric_infeasibility_unverified",
                    note="不得据此要求换generator")
    if all(c == "UNKNOWN_WITHIN_BUDGET" for c in certs) and not any_pool and not policy_ok:
        return dict(GO_K1=False, reason="UNKNOWN_WITHIN_BUDGET",
                    note="结束搜索, 不增加轮次")
    if any(c == "BLOCKED_SOLVER_BACKEND" for c in certs) and not policy_ok:
        return dict(GO_K1=False, reason="BLOCKED_SOLVER_BACKEND")
    return dict(GO_K1=False, reason="mixed_or_incomplete", certs=certs,
                pool=[r.get("pool_joint_witness") for r in outer],
                policy=[r.get("policy_oracle_witness") for r in outer])


def write_analysis(all_results, go, freeze_status, budgets, out_dir: Path, cal_blocked):
    lines = []
    lines.append("# K0 联合可达性 RESULT_ANALYSIS")
    lines.append("")
    lines.append(f"日期：{datetime.now().astimezone().isoformat(timespec='seconds')}（Asia/Tokyo）")
    lines.append("标签：`ORACLE_DIAGNOSTIC_ONLY`。本阶段未训练 generator / K1 / K2。")
    lines.append("")
    lines.append("## 1. 执行状态")
    lines.append("")
    lines.append(f"- freeze_inputs: **{freeze_status}**")
    lines.append(f"- original_CAL: **{cal_blocked}**")
    lines.append(f"- solver budgets (outer): `{json.dumps(budgets, ensure_ascii=False)}`")
    lines.append(f"- GO_K1 登记: **{go.get('GO_K1')}** — {go.get('reason')} — {go.get('note','')}")
    lines.append("")
    lines.append("## 2. 三字段（每 seed × split）")
    lines.append("")
    lines.append("| seed | split | pool_joint_witness | policy_oracle_witness | relaxation_certificate_status |")
    lines.append("|---:|---|---|---|---|")
    for r in sorted(all_results, key=lambda x: (x['split'], x['seed'])):
        lines.append(
            f"| {r['seed']} | {r['split']} | {r.get('pool_joint_witness')} | "
            f"{r.get('policy_oracle_witness')} | {r.get('relaxation_certificate_status')} |"
        )
    lines.append("")
    lines.append("## 3. Outer 对照（好窗口数 / change ADE / change FDE / W）")
    lines.append("")
    for r in [x for x in all_results if x["split"] == "outer"]:
        lines.append(f"### seed {r['seed']}")
        lines.append("")
        lines.append("| method | overall_good | change_good | change_minADE | change_minFDE | W_pass |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for name, blk in r.get("portfolios", {}).items():
            # need W from stored — portfolios slim may lack W_pass per method; use fields
            lines.append(
                f"| {name} | {blk.get('overall_good')} | {blk.get('change_good')} | "
                f"{blk.get('change_minADE')} | {blk.get('change_minFDE')} | {blk.get('W_pass')} |"
            )
        lines.append("")
    lines.append("## 4. 分支解释")
    lines.append("")
    lines.append(json.dumps(go, ensure_ascii=False, indent=2))
    lines.append("")
    lines.append("## 5. 未做之事")
    lines.append("")
    lines.append("- 未训练 K1 / K2 / generator")
    lines.append("- 未扫配额 / boost / λ / 网络 / epoch")
    lines.append("- 未改写旧 P1 / MODE_PRESERVING / COMPLEMENTARY_POOL_VALIDATION 结果")
    lines.append("- K0 oracle 产物不进入 K1 teacher 白名单")
    lines.append("")
    md = "\n".join(lines)
    (out_dir / "RESULT_ANALYSIS.md").write_text(md, encoding="utf-8")
    dump(out_dir / "RESULT_ANALYSIS.json", dict(
        go=go, freeze_status=freeze_status, budgets=budgets,
        cal_blocked=cal_blocked, results=all_results, tag=ORACLE_TAG,
    ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-freeze", action="store_true")
    ap.add_argument("--skip-k0b", action="store_true")
    ap.add_argument("--splits", default="outer,inner,FIT_cal",
                    help="comma list; original_CAL optional")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--time-limit-outer", type=int, default=0,
                    help="0=set after preflight")
    args = ap.parse_args()

    EXEC.mkdir(parents=True, exist_ok=True)
    print("== freeze_inputs ==", flush=True)
    fr = run_freeze(EXEC / "freeze")
    if fr["status"] == "BLOCKED":
        dump(EXEC / "BLOCKED.json", dict(stage="freeze_inputs", errors=fr["errors"]))
        write_analysis([], dict(GO_K1=False, reason="freeze_BLOCKED"), "BLOCKED", {}, EXEC,
                       cal_blocked="NOT_RUN")
        print("BLOCKED at freeze", fr["errors"], flush=True)
        return 2
    data = fr["data"]
    fit_train, fit_cal = fr["fit_train"], fr["fit_cal"]

    # original_CAL status
    cal_blocked = fr["manifest"]["binding"].get("original_CAL_kin", {})
    cal_label = "BLOCKED_NO_KIN_CACHE" if all(
        v.get("status") == "BLOCKED_NO_KIN_CACHE" for v in cal_blocked.values()
    ) else "PARTIAL_OR_OK"

    # preflight on 64 FIT_train
    print("== preflight_64 ==", flush=True)
    seed0 = 42
    kin_ft, _ = load_kin("FIT_train", seed0, data, fit_train, fit_cal)
    nat_ft = data["native"][fit_train]
    gt_ft = data["gt"][fit_train]
    grp_ft = data["groups"][fit_train]
    pool_ft = np.concatenate([nat_ft, kin_ft], axis=1)
    pf = preflight_64(pool_ft, gt_ft, nat_ft, grp_ft, EXEC / "preflight", time_limit_s=180)
    dump(EXEC / "preflight" / "RESOURCE_preflight.json", pf)

    # register budgets BEFORE formal outer
    if args.time_limit_outer > 0:
        outer_tl = args.time_limit_outer
    else:
        # safety: based on preflight wall, scale, cap
        hint = float(pf.get("est_outer_wall_hint_s") or 3600)
        # register conservative wall: min(max(2*preflight_wall*scale, 600), 7200)
        pw = float(pf.get("wall_s") or 60)
        scale = float(pf.get("outer_scale_factor") or (1035 / 64))
        outer_tl = int(min(7200, max(600, 2.0 * pw * scale)))
    budgets = dict(
        time_limit_s=outer_tl,
        node_limit=None,  # CBC time-limited; node not set → unlimited within time
        registered_before_outer=True,
        preflight_wall_s=pf.get("wall_s"),
        preflight_status=pf.get("status") or pf.get("pulp_status"),
        note="Time limit is stop mechanism, not infeasibility evidence.",
    )
    dump(EXEC / "SOLVER_BUDGETS.json", budgets)
    print(f"Registered outer time_limit_s={outer_tl}", flush=True)

    if args.preflight_only or args.only_freeze:
        print("stop after freeze/preflight", flush=True)
        return 0

    seeds = [int(x) for x in args.seeds.split(",")]
    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]
    all_results = []

    for seed in seeds:
        for split_name in split_names:
            print(f"==== seed={seed} split={split_name} ====", flush=True)
            if split_name == "outer":
                ix = data["split"]["outer"]
                kin, src = load_kin("outer", seed, data)
            elif split_name == "inner":
                ix = data["split"]["inner"]
                kin, src = load_kin("inner", seed, data)
            elif split_name == "FIT_cal":
                ix = fit_cal
                kin, src = load_kin("FIT_cal", seed, data, fit_train, fit_cal)
            elif split_name == "FIT_train":
                ix = fit_train
                kin, src = load_kin("FIT_train", seed, data, fit_train, fit_cal)
            elif split_name == "original_CAL":
                kin, src = load_kin("original_CAL", seed, data)
                if kin is None:
                    dump(EXEC / f"original_CAL_seed{seed}_BLOCKED.json", dict(status=src))
                    all_results.append(dict(
                        seed=seed, split="original_CAL",
                        pool_joint_witness=None, policy_oracle_witness=None,
                        relaxation_certificate_status="BLOCKED_NO_KIN_CACHE",
                    ))
                    continue
                ix = data["split"]["original_CAL"]
            else:
                raise ValueError(split_name)

            native = data["native"][ix]
            gt = data["gt"][ix]
            groups = data["groups"][ix]
            ids = data["sample_id"][ix]
            assert kin.shape[0] == len(ix), f"kin {kin.shape} vs {len(ix)}"
            pool40 = np.concatenate([native, kin], axis=1)

            v1_idx, v1_src = load_v1_selected_indices(split_name, seed)
            if v1_idx is not None:
                # align by sample_id
                with np.load(v1_src.replace("artifacts", "artifacts"), allow_pickle=False) as z:
                    pass
                # bundle already keyed; check id equality
                import numpy as _np
                from pathlib import Path as _P
                with _np.load(v1_src, allow_pickle=False) as z:
                    bid = z["sample_id"]
                if not _np.array_equal(bid, ids):
                    print(f"WARN v1 bundle id mismatch {split_name} seed{seed}; skip v1 compare", flush=True)
                    v1_idx = None

            out_split = EXEC / "results" / f"seed{seed}" / split_name
            # skip-resume: if result_fields exists and complete, continue
            rf = out_split / "result_fields.json"
            if rf.exists():
                try:
                    prev = json.loads(rf.read_text(encoding="utf-8"))
                    if prev.get("policy_oracle_witness") is not None and prev.get("seed") == seed:
                        print(f"resume-skip existing {out_split}", flush=True)
                        all_results.append(prev)
                        continue
                except Exception:
                    pass

            do_k0b = (not args.skip_k0b) and split_name in ("outer", "inner", "FIT_cal")
            # FIT_cal/inner get same budget as registered (design: each seed/split max R+S once)
            bgt = dict(budgets)
            if split_name != "outer":
                # keep same registered wall; do not inflate after seeing outer
                pass
            res = run_split(
                seed, split_name, pool40, gt, native, groups, ids, bgt, out_split,
                v1_indices=v1_idx, do_k0b=do_k0b, do_cluster_outer=True,
            )
            res["kin_source"] = src
            all_results.append(res)
            dump(EXEC / "runtime_state.json", dict(
                updated=datetime.now().astimezone().isoformat(timespec="seconds"),
                n_done=len(all_results), last=dict(seed=seed, split=split_name),
            ))

    go = decide_go_k1(all_results)
    dump(EXEC / "GO_K1_REGISTER.json", go)
    write_analysis(all_results, go, fr["status"], budgets, EXEC, cal_label)

    # arithmetic audit stub: recompute one outer seed native vs k0a coverage equality checks already in match
    dump(EXEC / "ARITHMETIC_AUDIT.json", dict(
        note="K0-A match_union_quality and W clause recomputation in metrics.check_W",
        tag=ORACLE_TAG,
    ))
    dump(EXEC / "RESOURCE.json", dict(budgets=budgets, preflight=pf))
    print("DONE", go, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
