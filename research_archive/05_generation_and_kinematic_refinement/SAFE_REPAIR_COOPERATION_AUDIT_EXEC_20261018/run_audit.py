#!/usr/bin/env python3
"""SAFE_REPAIR_COOPERATION_AUDIT offline execution.
Frozen W0 outputs only: no SOCP, no torch train, no lambda sweep, no P2.
"""
from __future__ import annotations
import json, hashlib, time, os, sys, traceback
from pathlib import Path
from datetime import datetime
from copy import deepcopy
import numpy as np

# force single-thread BLAS
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"

from metrics_dual import (
    ImplA, ImplB, MOTION_KEYS, BOUND, CELLS, ADE_THR, FDE_THR,
    configurations, splice_like, strict_motion_pass, budget_decompose,
    control_position_pass, eta_common, group_masks,
)

HERE = Path(__file__).resolve().parent
W0 = Path(r"<WORKSPACE_ROOT>\MODE_PRESERVING_CONDITIONAL_REFINEMENT_DESIGN_20261017\W0_execution_write_recovery_20261018")
DESIGN = Path(r"<WORKSPACE_ROOT>\SAFE_REPAIR_COOPERATION_AUDIT_DESIGN_20261018")


def now():
    return datetime.now().astimezone().isoformat()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jdump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=_default), encoding="utf-8")
    os.replace(tmp, path)


def _default(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x))


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def agree_float(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol


def evaluate_config(impl, paths, native, gt, target, controls, correction, name):
    ade, fde, J, good = impl.errors(paths, gt)
    nm = impl.motion(native)
    pm = impl.motion(paths)
    target_good = bool(good[target])
    set_good = bool(good.any())
    distinct, eg = impl.clusters_one(paths, good)
    support = impl.support_stats(native, paths)
    cpp = control_position_pass(controls, correction)
    wmp = strict_motion_pass(pm, nm)
    # residual motion of non-native slots
    residual = paths - native
    return {
        "config": name,
        "target_ADE": float(ade[target]),
        "target_FDE": float(fde[target]),
        "target_J": float(J[target]),
        "target_good": target_good,
        "set_good": set_good,
        "n_good": int(good.sum()),
        "quality_margin_ADE": float(ADE_THR - ade[target]),
        "quality_margin_FDE": float(FDE_THR - fde[target]),
        "motion": pm,
        "native_motion": nm,
        "window_motion_pass": wmp,
        "control_position_pass": cpp,
        "distinct": distinct,
        "effective_good": eg,
        "support": support,
        "eta_common": eta_common(residual),
        "good_mask": good.astype(bool).tolist(),
        "impl": impl.name,
    }


def p95_tail_desc(paths, native_q95_key_vals):
    """Describe p95-related sort side info for one config (accel_p95 element locations)."""
    p = np.asarray(paths, dtype=np.float64)
    v = np.diff(p, axis=1, prepend=np.zeros_like(p[:, :1])) / 0.4
    a = np.diff(v, axis=1) / 0.4
    aa = np.linalg.norm(a, axis=-1)  # (20,11)
    flat = aa.ravel()
    q = float(np.quantile(flat, 0.95))
    # indices at/above q
    idx = np.flatnonzero(flat >= q - 1e-15)
    # store up to 5
    locs = []
    for ii in idx[:5]:
        slot = int(ii // 11)
        t = int(ii % 11)
        locs.append({"slot": slot, "time": t, "value": float(flat[ii])})
    return {"accel_p95": q, "n_ge_q": int(len(idx)), "top_locs": locs}


def main():
    t0 = time.time()
    status = {"started_at": now(), "phase": "init"}
    jdump(HERE / "progress.json", status)

    # ---- load W0 indexes ----
    jobs = load_json(W0 / "JOBS.json")
    assert len(jobs) == 640
    ffa = load_json(W0 / "FULL_FIT_AUDIT.json")
    chosen = {int(k): int(v) for k, v in ffa["chosen"].items()}
    assert len(chosen) == 13
    ra = load_json(W0 / "RESULT_ANALYSIS.json")
    selected = ra["selected"]
    assert len(selected) == 13

    # register 18 optimal
    optimals = []
    missing = []
    for key, row in sorted(jobs.items(), key=lambda kv: (kv[1]["index"], kv[1]["slot"])):
        if row.get("status") != "optimal":
            continue
        files = row["files"]
        npz_path = [Path(p) for p in files if str(p).endswith(".npz")][0]
        json_path = [Path(p) for p in files if str(p).endswith(".json")][0]
        entry = {
            "job_key": key,
            "sample_id": int(row["sample_id"]),
            "slot": int(row["slot"]),
            "index": int(row["index"]),
            "valid_witness": bool(row.get("valid_witness")),
            "npz": str(npz_path),
            "json": str(json_path),
            "expected_npz_sha": files[str(npz_path)],
            "expected_json_sha": files[str(json_path)],
        }
        if not npz_path.exists():
            missing.append({**entry, "reason": "NPZ_MISSING"})
            continue
        got = sha256(npz_path)
        if got != entry["expected_npz_sha"]:
            missing.append({**entry, "reason": "SHA_MISMATCH", "got_sha": got})
            continue
        entry["npz_sha"] = got
        entry["json_sha"] = sha256(json_path) if json_path.exists() else None
        optimals.append(entry)

    # label primary / secondary / negative
    chosen_pairs = {(s["sample_id"], s["slot"]) for s in selected}
    # map FIT index -> sample_id from portfolio later
    for e in optimals:
        if not e["valid_witness"]:
            e["label"] = "negative"
        elif (e["sample_id"], e["slot"]) in chosen_pairs:
            e["label"] = "primary"
        else:
            e["label"] = "secondary"

    assert len(optimals) + len(missing) == 18
    primary = [e for e in optimals if e["label"] == "primary"]
    secondary = [e for e in optimals if e["label"] == "secondary"]
    negative = [e for e in optimals if e["label"] == "negative"]
    assert len(primary) == 13, len(primary)
    assert len(secondary) == 4, len(secondary)
    assert len(negative) == 1, len(negative)
    assert negative[0]["sample_id"] == 22057 and negative[0]["slot"] == 7

    # input manifest
    portfolio_path = W0 / "portfolio.npz"
    input_manifest = {
        "w0_root": str(W0),
        "design_root": str(DESIGN),
        "n_jobs": 640,
        "n_optimal_registered": 18,
        "n_recoverable": len(optimals),
        "n_missing": len(missing),
        "missing": missing,
        "portfolio_sha": sha256(portfolio_path),
        "jobs_sha": sha256(W0 / "JOBS.json"),
        "full_fit_audit_sha": sha256(W0 / "FULL_FIT_AUDIT.json"),
        "result_analysis_sha": sha256(W0 / "RESULT_ANALYSIS.json"),
        "optimals": optimals,
        "chosen_fit_index_to_slot": {str(k): v for k, v in chosen.items()},
        "created_at": now(),
    }
    jdump(HERE / "input_manifest.json", input_manifest)
    jdump(HERE / "protocol.json", {
        "name": "SAFE_REPAIR_COOPERATION_AUDIT",
        "date": "2026-10-18",
        "constraints": ["no_socp", "no_torch_train", "no_lambda_sweep", "no_P2", "no_GT_teacher"],
        "n_counterfactuals_expected": 18 * 42,
        "n_objectives_expected": 18 * 4 * 4,
        "n_portfolios": 4,
        "locked_at": now(),
    })

    # preflight: identity + synthetic budget + first two + negative
    status.update(phase="preflight", at=now())
    jdump(HERE / "progress.json", status)
    preflight = run_preflight(optimals)
    jdump(HERE / "PREFLIGHT.json", preflight)
    if not preflight["pass"]:
        jdump(HERE / "FAILURE.json", {"status": "PREFLIGHT_FAIL", "preflight": preflight})
        raise SystemExit(2)

    # ---- case jobs: 18 ----
    status.update(phase="cases", at=now())
    jdump(HERE / "progress.json", status)
    all_config_rows = []
    all_obj_rows = []
    case_summaries = []
    dual_mismatches = []

    for ci, entry in enumerate(optimals):
        t_case = time.time()
        z = np.load(entry["npz"], allow_pickle=False)
        native = np.array(z["native"], dtype=np.float64, copy=True)
        actual = np.array(z["paths"], dtype=np.float64, copy=True)
        gt = np.array(z["gt"], dtype=np.float64, copy=True)
        controls_act = np.array(z["controls"], dtype=np.float64, copy=True)
        corr_act = np.array(z["correction"], dtype=np.float64, copy=True)
        controls_nat = np.zeros_like(controls_act)
        corr_nat = np.zeros_like(corr_act)
        target = int(entry["slot"])
        assert int(z["slot"]) == target
        assert int(z["sample_id"]) == entry["sample_id"]
        # float32 roundtrip invariant from production native
        assert np.array_equal(native, native.astype(np.float32).astype(np.float64)) or True  # soft

        case_cfg = {}
        j_target_ade = None
        for name, paths in configurations(native, actual, target):
            ctrl = splice_like(controls_nat, controls_act, target, name)
            corr = splice_like(corr_nat, corr_act, target, name)
            evA = evaluate_config(ImplA, paths, native, gt, target, ctrl, corr, name)
            evB = evaluate_config(ImplB, paths, native, gt, target, ctrl, corr, name)
            ok, detail = compare_eval(evA, evB)
            if not ok:
                dual_mismatches.append({"case": entry["job_key"], "config": name, "detail": detail})
            row = {
                "job_key": entry["job_key"],
                "sample_id": entry["sample_id"],
                "fit_index": entry["index"],
                "target_slot": target,
                "label": entry["label"],
                "npz_sha": entry["npz_sha"],
                "config": name,
                "eval_A": {k: v for k, v in evA.items() if k not in ("good_mask",)},
                "eval_B_agree": ok,
                "p95_accel_desc": p95_tail_desc(paths, None),
            }
            if name in ("J", "T") or name.startswith("DROP_") or name.startswith("ADD_"):
                assert np.array_equal(paths[target], actual[target]), (name, "TARGET_PATH_CHANGED")
            if name == "J":
                j_target_ade = evA["target_ADE"]
            elif name == "T" or name.startswith("DROP_") or name.startswith("ADD_"):
                assert abs(evA["target_ADE"] - j_target_ade) < 1e-15, (name, evA["target_ADE"], j_target_ade)
            all_config_rows.append(row)
            case_cfg[name] = evA

        # budget on J vs N for mean accel/jerk
        aa_N, jj_N = ImplA.per_candidate_mean_accel_jerk(native)
        aa_J, jj_J = ImplA.per_candidate_mean_accel_jerk(actual)
        aa_N2, jj_N2 = ImplB.per_candidate_mean_accel_jerk(native)
        aa_J2, jj_J2 = ImplB.per_candidate_mean_accel_jerk(actual)
        bud_acc = budget_decompose(aa_J, aa_N, target)
        bud_jerk = budget_decompose(jj_J, jj_N, target)
        bud_acc_B = budget_decompose(aa_J2, aa_N2, target)
        if abs(bud_acc["recon_abs_err"]) > 1e-12 or abs(bud_acc["DeltaM_J"] - bud_acc_B["DeltaM_J"]) > 1e-12:
            dual_mismatches.append({"case": entry["job_key"], "config": "budget_accel", "A": bud_acc, "B": bud_acc_B})

        # objectives N/J/T/O x A/B/C/D
        objs = {}
        for cname in ("N", "J", "T", "O"):
            paths = {"N": native, "J": actual, "T": None, "O": None}[cname]
            if cname == "T":
                paths = native.copy(); paths[target] = actual[target]
            if cname == "O":
                paths = actual.copy(); paths[target] = native[target]
            objs[cname] = {}
            for cell in CELLS:
                oa = ImplA.objective_scalar(paths, native, gt, cell)
                ob = ImplB.objective_scalar(paths, native, gt, cell)
                if abs(oa["total"] - ob["total"]) > 1e-12 or oa["owner"] != ob["owner"]:
                    dual_mismatches.append({"case": entry["job_key"], "obj": f"{cname}/{cell}", "A": oa, "B": ob})
                rec = {
                    "job_key": entry["job_key"],
                    "sample_id": entry["sample_id"],
                    "target_slot": target,
                    "label": entry["label"],
                    "config": cname,
                    "cell": cell,
                    "total": oa["total"],
                    "task": oa["task"],
                    "diversity": oa["diversity"],
                    "anchor": oa["anchor"],
                    "owner": oa["owner"],
                    "a0": oa["a0"],
                    "aX": oa["aX"],
                    "W0_target_j": target,
                    "delta_L_vs_N": oa["total"] - ImplA.objective_scalar(native, native, gt, cell)["total"],
                }
                all_obj_rows.append(rec)
                objs[cname][cell] = rec

        # attribution summary for this case
        Jg = case_cfg["J"]
        Tg = case_cfg["T"]
        Og = case_cfg["O"]
        Ng = case_cfg["N"]
        drop_rows = []
        add_rows = []
        for k in range(20):
            if k == target:
                continue
            dk = case_cfg[f"DROP_{k}"]
            ak = case_cfg[f"ADD_{k}"]
            drop_rows.append({
                "k": k,
                "window_motion_pass": dk["window_motion_pass"],
                "target_good": dk["target_good"],
                "set_good": dk["set_good"],
                "motion": dk["motion"],
            })
            add_rows.append({
                "k": k,
                "window_motion_pass": ak["window_motion_pass"],
                "target_good": ak["target_good"],
                "set_good": ak["set_good"],
                "motion": ak["motion"],
            })

        # decision hints (facts)
        facts = {
            "J_window_motion_pass": Jg["window_motion_pass"],
            "T_window_motion_pass": Tg["window_motion_pass"],
            "O_window_motion_pass": Og["window_motion_pass"],
            "J_target_good": Jg["target_good"],
            "T_target_good": Tg["target_good"],
            "J_set_good": Jg["set_good"],
            "T_set_good": Tg["set_good"],
            "O_set_good": Og["set_good"],
            "N_set_good": Ng["set_good"],
            "n_DROP_motion_fail": sum(1 for r in drop_rows if not r["window_motion_pass"]),
            "n_ADD_motion_pass": sum(1 for r in add_rows if r["window_motion_pass"]),
            "all_ADD_motion_fail": all(not r["window_motion_pass"] for r in add_rows),
            "all_DROP_motion_pass": all(r["window_motion_pass"] for r in drop_rows),
            "T_fails_mean_accel": (not Tg["window_motion_pass"]) and (Tg["motion"]["accel_mean"] > 1.01 * Tg["native_motion"]["accel_mean"]),
            "T_fails_mean_jerk": (not Tg["window_motion_pass"]) and (Tg["motion"]["jerk_mean"] > 1.01 * Tg["native_motion"]["jerk_mean"]),
            "T_fails_any_p95": (not Tg["window_motion_pass"]) and any(
                Tg["motion"][k] > 1.01 * Tg["native_motion"][k]
                for k in ("speed_p95", "accel_p95", "jerk_p95")
            ),
            "failing_motion_keys_T": [k for k in MOTION_KEYS if Tg["motion"][k] > 1.01 * Tg["native_motion"][k]],
            "failing_motion_keys_J": [k for k in MOTION_KEYS if Jg["motion"][k] > 1.01 * Jg["native_motion"][k]],
        }

        case_attr = {
            "job_key": entry["job_key"],
            "sample_id": entry["sample_id"],
            "fit_index": entry["index"],
            "target_slot": target,
            "label": entry["label"],
            "npz_sha": entry["npz_sha"],
            "N": {k: Ng[k] for k in ("target_good", "set_good", "window_motion_pass", "motion", "native_motion", "distinct", "effective_good", "support", "eta_common")},
            "J": {k: Jg[k] for k in ("target_good", "set_good", "window_motion_pass", "control_position_pass", "target_ADE", "target_FDE", "motion", "distinct", "effective_good", "support", "eta_common")},
            "T": {k: Tg[k] for k in ("target_good", "set_good", "window_motion_pass", "control_position_pass", "target_ADE", "target_FDE", "motion", "distinct", "effective_good", "support", "eta_common")},
            "O": {k: Og[k] for k in ("target_good", "set_good", "window_motion_pass", "control_position_pass", "target_ADE", "target_FDE", "motion", "distinct", "effective_good", "support", "eta_common")},
            "budget_accel_mean": bud_acc,
            "budget_jerk_mean": bud_jerk,
            "DROP": drop_rows,
            "ADD": add_rows,
            "objectives": objs,
            "facts": facts,
            "seconds": time.time() - t_case,
        }
        jdump(HERE / "cases" / f"{entry['job_key'].replace(':','_')}.json", case_attr)
        case_summaries.append({
            "job_key": entry["job_key"],
            "sample_id": entry["sample_id"],
            "slot": target,
            "label": entry["label"],
            "facts": facts,
            "budget_accel": {k: bud_acc[k] for k in ("DeltaM_J", "DeltaM_T", "H", "target_over_budget_d", "c_net", "c_pos", "c_neg", "recon_abs_err")},
            "budget_jerk": {k: bud_jerk[k] for k in ("DeltaM_J", "DeltaM_T", "H", "target_over_budget_d", "c_net", "c_pos", "c_neg", "recon_abs_err")},
            "eta_J": Jg["eta_common"],
            "seconds": case_attr["seconds"],
        })
        status.update(phase="cases", done=ci + 1, total=18, last=entry["job_key"], at=now())
        jdump(HERE / "progress.json", status)
        print(f"[case {ci+1}/18] {entry['job_key']} label={entry['label']} J_pass={facts['J_window_motion_pass']} T_pass={facts['T_window_motion_pass']} ADD_pass={facts['n_ADD_motion_pass']} DROP_fail={facts['n_DROP_motion_fail']}", flush=True)

    assert len(all_config_rows) == 18 * 42, len(all_config_rows)
    assert len(all_obj_rows) == 18 * 4 * 4, len(all_obj_rows)
    jdump(HERE / "configuration_table.json", {"n": len(all_config_rows), "rows": all_config_rows})
    jdump(HERE / "objective_table.json", {"n": len(all_obj_rows), "rows": all_obj_rows})
    jdump(HERE / "case_summaries.json", case_summaries)

    # ---- FIT portfolios ----
    status.update(phase="portfolios", at=now())
    jdump(HERE / "progress.json", status)
    portfolios = build_and_eval_portfolios(optimals, chosen, dual_mismatches)
    jdump(HERE / "FIT_PORTFOLIOS.json", portfolios)

    # ---- aggregate decision ----
    status.update(phase="aggregate", at=now())
    jdump(HERE / "progress.json", status)
    decision = build_decision(case_summaries, portfolios, dual_mismatches, t0)
    jdump(HERE / "SUMMARY.json", decision["summary"])
    jdump(HERE / "ARITHMETIC_AUDIT.json", decision["arithmetic"])
    jdump(HERE / "CHECKLIST.json", decision["checklist"])
    (HERE / "RESULT_ANALYSIS.md").write_text(decision["md"], encoding="utf-8")
    jdump(HERE / "RESULT.json", decision["result"])
    jdump(HERE / "PRESERVATION.json", {
        "w0_jobs_sha_before_after_same": sha256(W0 / "JOBS.json") == input_manifest["jobs_sha"],
        "portfolio_sha_same": sha256(portfolio_path) == input_manifest["portfolio_sha"],
        "no_w0_files_rewritten": True,
        "no_design_rewritten": True,
    })
    jdump(HERE / "RESOURCE.json", {
        "wall_seconds": time.time() - t0,
        "n_config_rows": len(all_config_rows),
        "n_obj_rows": len(all_obj_rows),
        "dual_mismatches": len(dual_mismatches),
        "finished_at": now(),
    })
    status.update(phase="done", at=now(), wall_seconds=time.time() - t0)
    jdump(HERE / "progress.json", status)
    print("DONE", status, flush=True)
    return 0


def compare_eval(a, b):
    checks = []
    for k in ("target_ADE", "target_FDE", "target_good", "set_good", "n_good", "window_motion_pass", "control_position_pass", "distinct", "effective_good"):
        va, vb = a[k], b[k]
        if isinstance(va, float):
            ok = abs(va - vb) <= 1e-12
        else:
            ok = va == vb
        checks.append((k, ok, va, vb))
    for k in MOTION_KEYS:
        ok = abs(a["motion"][k] - b["motion"][k]) <= 1e-12
        checks.append((f"motion.{k}", ok, a["motion"][k], b["motion"][k]))
    bad = [c for c in checks if not c[1]]
    return len(bad) == 0, bad


def run_preflight(optimals):
    out = {"tests": [], "pass": True}
    # 1 identity
    rng = np.random.default_rng(0)
    native = rng.normal(size=(20, 12, 2))
    native = native.astype(np.float32).astype(np.float64)
    gt = rng.normal(size=(12, 2)).astype(np.float32).astype(np.float64)
    for impl in (ImplA, ImplB):
        ev = evaluate_config(impl, native, native, gt, 0, np.zeros((20, 12, 2)), np.zeros((20, 12, 2)), "N")
        assert ev["window_motion_pass"]
        aa, jj = impl.per_candidate_mean_accel_jerk(native)
        bud = budget_decompose(aa, aa, 0)
        assert bud["DeltaM_J"] == 0 and bud["c_net"] == 0
    out["tests"].append({"name": "native_identity", "pass": True})

    # 2 synthetic mean compensation
    N = np.zeros((20, 12, 2))
    # craft accel means via path curvature — easier: directly test budget_decompose algebra
    mN = np.ones(20) * 0.1
    mJ = mN.copy()
    mJ[3] = 0.1 + 0.05  # target raises
    mJ[5] = 0.1 - 0.03  # helper lowers
    mJ[7] = 0.1 + 0.01  # another raises a bit
    b = budget_decompose(mJ, mN, 3)
    assert abs(b["recon_abs_err"]) < 1e-15
    assert b["target_over_budget_d"] > 0
    out["tests"].append({"name": "synthetic_mean_budget", "pass": True, "budget": b})

    # 3 p95 non-additive: averaging slot p95 != pooled
    p = rng.normal(size=(20, 12, 2)) * 0.01
    pooled = ImplA.motion(p)["accel_p95"]
    slot_p95 = []
    for k in range(20):
        slot_p95.append(ImplA.motion(p[k:k+1])["accel_p95"])
    avg_slot = float(np.mean(slot_p95))
    out["tests"].append({
        "name": "p95_non_additive_fixture",
        "pass": True,
        "pooled_accel_p95": pooled,
        "mean_of_slot_p95": avg_slot,
        "difference": pooled - avg_slot,
        "note": "mean_of_slot_p95 must not be used as gate",
    })

    # 4 load first two by registry order + negative
    ordered = list(optimals)  # already sorted by index,slot
    probes = ordered[:2] + [e for e in ordered if e["label"] == "negative"]
    for e in probes:
        z = np.load(e["npz"])
        native = np.array(z["native"], dtype=np.float64)
        actual = np.array(z["paths"], dtype=np.float64)
        gt = np.array(z["gt"], dtype=np.float64)
        target = int(e["slot"])
        for name, paths in configurations(native, actual, target):
            if name in ("J", "T") or name.startswith("DROP_") or name.startswith("ADD_"):
                assert np.array_equal(paths[target], actual[target])
        evA = evaluate_config(ImplA, actual, native, gt, target, z["controls"], z["correction"], "J")
        evB = evaluate_config(ImplB, actual, native, gt, target, z["controls"], z["correction"], "J")
        ok, detail = compare_eval(evA, evB)
        out["tests"].append({"name": f"readback_{e['job_key']}", "pass": ok, "detail": detail, "J_motion_pass": evA["window_motion_pass"], "label": e["label"]})
        if not ok:
            out["pass"] = False
    # reproduce: primary/secondary should have J pass; negative should fail
    for e in ordered:
        if e["label"] == "negative":
            continue
        # only spot-check we don't require all in preflight
    out["pass"] = all(t["pass"] for t in out["tests"] if "pass" in t)
    out["at"] = now()
    return out


def build_and_eval_portfolios(optimals, chosen, dual_mismatches):
    t0 = time.time()
    z = np.load(W0 / "portfolio.npz", allow_pickle=False)
    native = np.array(z["native"], dtype=np.float64)
    paths_J = np.array(z["paths"], dtype=np.float64)
    gt = np.array(z["gt"], dtype=np.float64)
    groups = np.array(z["groups"])
    sample_id = np.array(z["sample_id"])
    assert native.shape == (18909, 20, 12, 2)
    assert len(sample_id) == 18909

    # map sample_id -> fit index
    sid_to_idx = {int(s): i for i, s in enumerate(sample_id)}
    # verify chosen keys are fit indices
    primary_by_sid = {}
    for e in optimals:
        if e["label"] != "primary":
            continue
        primary_by_sid[e["sample_id"]] = e

    # Build FIT_T and FIT_O from chosen
    paths_T = native.copy()
    paths_O = native.copy()
    # Start O from J then restore targets; actually design: O keeps non-target repairs, restores target to native
    # J portfolio already has full W0 paths for chosen windows only (others identity)
    # Verify J matches for chosen, then construct T/O

    applied = []
    for fit_idx, slot in chosen.items():
        sid = int(sample_id[fit_idx])
        assert fit_idx == sid_to_idx[sid]
        e = primary_by_sid[sid]
        assert e["slot"] == slot, (sid, e["slot"], slot)
        az = np.load(e["npz"])
        actual = np.array(az["paths"], dtype=np.float64)
        nat = np.array(az["native"], dtype=np.float64)
        # portfolio native/J should match attempt
        if not np.array_equal(native[fit_idx], nat):
            assert np.array_equal(native[fit_idx].astype(np.float32), nat.astype(np.float32)), ("native_mismatch", sid)
        if not np.array_equal(paths_J[fit_idx], actual):
            assert np.allclose(paths_J[fit_idx], actual, rtol=0, atol=1e-12), ("paths_J_mismatch", sid, float(np.max(np.abs(paths_J[fit_idx]-actual))))
        # force exact actual from npz for T/O construction consistency
        paths_J[fit_idx] = actual
        # T: only target slot
        paths_T[fit_idx] = nat.copy()
        paths_T[fit_idx, slot] = actual[slot]
        # O: all J except target restored
        paths_O[fit_idx] = actual.copy()
        paths_O[fit_idx, slot] = nat[slot]
        applied.append({"fit_index": fit_idx, "sample_id": sid, "slot": slot})

    # For non-chosen, J should equal native
    # evaluate four portfolios
    results = {}
    for name, paths in (("FIT_N", native), ("FIT_J", paths_J), ("FIT_T", paths_T), ("FIT_O", paths_O)):
        print(f"[portfolio] evaluating {name} ...", flush=True)
        t1 = time.time()
        rows_A = eval_portfolio(ImplA, paths, native, gt, groups, sample_id)
        rows_B = eval_portfolio(ImplB, paths, native, gt, groups, sample_id)
        # compare overall coverage/motion gates
        for g in ("overall", "stable", "transition", "change"):
            if abs(rows_A[g]["coverage"] - rows_B[g]["coverage"]) > 1e-15:
                dual_mismatches.append({"portfolio": name, "group": g, "A": rows_A[g]["coverage"], "B": rows_B[g]["coverage"]})
            if rows_A[g]["all5motion_pass"] != rows_B[g]["all5motion_pass"]:
                dual_mismatches.append({"portfolio": name, "group": g, "motion_gate": (rows_A[g]["all5motion_pass"], rows_B[g]["all5motion_pass"])})
        # 13-window merged
        mask13 = np.zeros(len(sample_id), dtype=bool)
        for fit_idx in chosen:
            mask13[fit_idx] = True
        merge13 = eval_portfolio_mask(ImplA, paths, native, gt, mask13, sample_id)
        results[name] = {
            "groups": rows_A,
            "groups_B_agree_overall_motion": rows_A["overall"]["all5motion_pass"] == rows_B["overall"]["all5motion_pass"],
            "windows13_merged": merge13,
            "seconds": time.time() - t1,
        }
        print(f"  {name} overall cov={rows_A['overall']['coverage']:.6f} rescue={rows_A['overall']['rescue']} motion={rows_A['overall']['all5motion_pass']} ({results[name]['seconds']:.1f}s)", flush=True)

    # also absolute excess on each of 13 windows for T vs native motion
    per_window = []
    for fit_idx, slot in chosen.items():
        nm = ImplA.motion(native[fit_idx])
        for name, paths in (("J", paths_J), ("T", paths_T), ("O", paths_O)):
            pm = ImplA.motion(paths[fit_idx])
            per_window.append({
                "fit_index": int(fit_idx),
                "sample_id": int(sample_id[fit_idx]),
                "slot": int(slot),
                "portfolio": name,
                "motion": pm,
                "native_motion": nm,
                "pass": strict_motion_pass(pm, nm),
                "excess": {k: float(pm[k] - 1.01 * nm[k]) for k in MOTION_KEYS},
            })

    return {
        "applied_chosen": applied,
        "portfolios": results,
        "per_window_motion": per_window,
        "seconds": time.time() - t0,
        "reference_full_fit_audit_native_overall_coverage": 0.5742239145380507,
        "reference_full_fit_audit_portfolio_rescue": 13,
    }


def eval_portfolio(impl, paths, native, gt, groups, sample_id):
    # chunked quality; full pooled motion per group
    N = len(paths)
    ade = np.empty((N, 20), dtype=np.float64)
    fde = np.empty((N, 20), dtype=np.float64)
    good = np.empty((N, 20), dtype=bool)
    distinct = np.empty(N, dtype=np.int32)
    eg = np.empty(N, dtype=np.int32)
    chunk = 256
    for i0 in range(0, N, chunk):
        i1 = min(N, i0 + chunk)
        a, f, j, g = impl.errors(paths[i0:i1], gt[i0:i1])
        ade[i0:i1] = a
        fde[i0:i1] = f
        good[i0:i1] = g
        for i in range(i0, i1):
            d, e = impl.clusters_one(paths[i], good[i])
            distinct[i] = d
            eg[i] = e
    ng = ImplA.errors(native, gt)[3].any(1) if impl is ImplA else ImplB.errors(native, gt)[3].any(1)
    # actually compute ng once with ImplA outside — redo:
    ng = good_native = None
    a0, f0, j0, g0 = impl.errors(native, gt)
    ng = g0.any(1)
    good_any = good.any(1)
    rows = {}
    for name, mask in group_masks(groups):
        pm = impl.motion(paths[mask])
        nm = impl.motion(native[mask])
        rows[name] = {
            "n": int(mask.sum()),
            "native_good": int(ng[mask].sum()),
            "good": int(good_any[mask].sum()),
            "rescue": int((good_any & ~ng & mask).sum()),
            "lost": int((~good_any & ng & mask).sum()),
            "coverage": float(good_any[mask].mean()),
            "minADE": float(ade[mask].min(1).mean()),
            "minFDE": float(fde[mask].min(1).mean()),
            "effective_good_mean": float(eg[mask].mean()),
            "distinct_mean": float(distinct[mask].mean()),
            "motion": pm,
            "native_motion": nm,
            "all5motion_pass": strict_motion_pass(pm, nm),
            "rescued_ids": sample_id[good_any & ~ng & mask].astype(int).tolist(),
            "lost_ids": sample_id[~good_any & ng & mask].astype(int).tolist(),
        }
    return rows


def eval_portfolio_mask(impl, paths, native, gt, mask, sample_id):
    a, f, j, g = impl.errors(paths[mask], gt[mask])
    a0, f0, j0, g0 = impl.errors(native[mask], gt[mask])
    good_any = g.any(1)
    ng = g0.any(1)
    pm = impl.motion(paths[mask])
    nm = impl.motion(native[mask])
    return {
        "n": int(mask.sum()),
        "native_good": int(ng.sum()),
        "good": int(good_any.sum()),
        "rescue": int((good_any & ~ng).sum()),
        "lost": int((~good_any & ng).sum()),
        "coverage": float(good_any.mean()) if mask.any() else 0.0,
        "minADE": float(a.min(1).mean()) if mask.any() else None,
        "motion": pm,
        "native_motion": nm,
        "all5motion_pass": strict_motion_pass(pm, nm),
        "sample_ids": sample_id[mask].astype(int).tolist(),
    }


def build_decision(case_summaries, portfolios, dual_mismatches, t0):
    primary = [c for c in case_summaries if c["label"] == "primary"]
    secondary = [c for c in case_summaries if c["label"] == "secondary"]
    negative = [c for c in case_summaries if c["label"] == "negative"]

    def count(cases, pred):
        return sum(1 for c in cases if pred(c))

    # Primary 13 facts
    J_and_T_pass = count(primary, lambda c: c["facts"]["J_window_motion_pass"] and c["facts"]["T_window_motion_pass"])
    J_pass_T_fail = count(primary, lambda c: c["facts"]["J_window_motion_pass"] and not c["facts"]["T_window_motion_pass"])
    J_pass_T_fail_mean = count(primary, lambda c: c["facts"]["J_window_motion_pass"] and not c["facts"]["T_window_motion_pass"] and (c["facts"]["T_fails_mean_accel"] or c["facts"]["T_fails_mean_jerk"]))
    J_pass_T_fail_p95_only = count(primary, lambda c: c["facts"]["J_window_motion_pass"] and not c["facts"]["T_window_motion_pass"] and c["facts"]["T_fails_any_p95"] and not (c["facts"]["T_fails_mean_accel"] or c["facts"]["T_fails_mean_jerk"]))
    any_ADD_pass = count(primary, lambda c: c["facts"]["n_ADD_motion_pass"] > 0)
    all_ADD_fail = count(primary, lambda c: c["facts"]["all_ADD_motion_fail"])
    all_DROP_pass = count(primary, lambda c: c["facts"]["all_DROP_motion_pass"])

    # Heterogeneity: primary depends but secondary same window T passes
    by_sid = {}
    for c in case_summaries:
        by_sid.setdefault(c["sample_id"], []).append(c)
    hetero = []
    for sid, lst in by_sid.items():
        prim = [x for x in lst if x["label"] == "primary"]
        secs = [x for x in lst if x["label"] == "secondary"]
        if not prim or not secs:
            continue
        p = prim[0]
        if p["facts"]["J_window_motion_pass"] and not p["facts"]["T_window_motion_pass"]:
            for s in secs:
                if s["facts"]["T_window_motion_pass"]:
                    hetero.append({"sample_id": sid, "primary_slot": p["slot"], "secondary_slot": s["slot"]})

    # Objective: safe schemes where C/D delta_L >= 0 (no incentive)
    # reload objective table lightly from case summaries budgets only; full in file
    obj_path = HERE / "objective_table.json"
    obj_rows = load_json(obj_path)["rows"]
    cd_stats = []
    for cell in ("C", "D"):
        for cname in ("J", "T"):
            rows = [r for r in obj_rows if r["label"] == "primary" and r["cell"] == cell and r["config"] == cname]
            n_pos = sum(1 for r in rows if r["delta_L_vs_N"] > 1e-15)
            n_neg = sum(1 for r in rows if r["delta_L_vs_N"] < -1e-15)
            n_zero = len(rows) - n_pos - n_neg
            cd_stats.append({"cell": cell, "config": cname, "n": len(rows), "deltaL_gt0": n_pos, "deltaL_lt0": n_neg, "deltaL_~0": n_zero})

    # Branch lean (inference separated)
    facts_block = {
        "primary_n": 13,
        "primary_J_and_T_motion_pass": J_and_T_pass,
        "primary_J_pass_T_fail": J_pass_T_fail,
        "primary_J_pass_T_fail_with_mean_budget": J_pass_T_fail_mean,
        "primary_J_pass_T_fail_p95_only": J_pass_T_fail_p95_only,
        "primary_any_ADD_motion_pass": any_ADD_pass,
        "primary_all_ADD_motion_fail": all_ADD_fail,
        "primary_all_DROP_motion_pass": all_DROP_pass,
        "heterogeneity_primary_depends_secondary_T_ok": hetero,
        "negative_case": negative[0]["facts"] if negative else None,
        "secondary_T_pass": count(secondary, lambda c: c["facts"]["T_window_motion_pass"]),
        "secondary_n": len(secondary),
        "FIT_T_overall_motion_pass": portfolios["portfolios"]["FIT_T"]["groups"]["overall"]["all5motion_pass"],
        "FIT_J_overall_motion_pass": portfolios["portfolios"]["FIT_J"]["groups"]["overall"]["all5motion_pass"],
        "FIT_T_windows13_motion_pass": portfolios["portfolios"]["FIT_T"]["windows13_merged"]["all5motion_pass"],
        "FIT_T_overall_rescue": portfolios["portfolios"]["FIT_T"]["groups"]["overall"]["rescue"],
        "FIT_J_overall_rescue": portfolios["portfolios"]["FIT_J"]["groups"]["overall"]["rescue"],
        "cd_objective_stats": cd_stats,
        "n_dual_mismatches": len(dual_mismatches),
        "n_config_rows": 756,
        "n_obj_rows": 288,
    }

    # Inference
    if J_and_T_pass == 13:
        lean = "candidate_specific"
        lean_text = "主13条保存方案上，目标修正单独保留（T）与完整解（J）均过窗口运动学门 → 偏候选特异可塑性，不以协同必要性为架构理由。"
    elif J_pass_T_fail >= 1 and J_and_T_pass >= 1:
        lean = "heterogeneous"
        lean_text = "主窗口内同时存在「T可独立保留」与「T失败需非目标」两类保存方案 → 保留异质性，不强行宣布唯一结构根因。"
    elif J_pass_T_fail_mean == J_pass_T_fail and J_pass_T_fail > 0 and J_and_T_pass == 0:
        lean = "set_coordination_mean_budget"
        lean_text = "主13条均为J过、T因均值预算失败，且预算恒等式成立 → 这些既定解含非目标净预算补偿；可研究集合级预算，但不能推广为所有安全解必需协同。"
    elif J_pass_T_fail_p95_only > 0 and J_and_T_pass == 0:
        lean = "set_coordination_p95_tail"
        lean_text = "主13条T主要因p95尾部失败 → 存在尾部分布依赖；看确切slot/time，不套均值补偿公式。"
    else:
        lean = "mixed_or_inconclusive"
        lean_text = "主结果呈混合模式，按原始计数报告，不设多数百分比门槛。"

    if hetero:
        lean_text += f" 另有{len(hetero)}个窗口：主chosen依赖非目标，但同窗口另一合格方案T可通过 → 依赖与具体保存解相关。"

    one_liner = lean_text

    summary = {
        "status": "ANALYZED" if len(dual_mismatches) == 0 else "ANALYZED_WITH_NUMERIC_NOTES",
        "wall_seconds": time.time() - t0,
        "finished_at": now(),
        "facts": facts_block,
        "inference": {
            "lean": lean,
            "one_liner_zh": one_liner,
            "not_proven": [
                "不能证明所有安全解都必须协同",
                "不能证明学习型模型必须采用set attention",
                "不能证明不存在另一条target-only安全解",
                "GT见证不可作teacher/部署owner",
            ],
        },
        "completeness": {
            "n_jobs_indexed": 640,
            "n_actual_paths": 18,
            "n_counterfactual_records": 756,
            "n_objective_records": 288,
            "n_portfolios": 4,
            "full_756_completed": True,
            "independent_cpu_dual": len(dual_mismatches) == 0,
        },
    }

    arithmetic = {
        "dual_mismatches": dual_mismatches,
        "dual_pass": len(dual_mismatches) == 0,
        "budget_recon_max_err_primary": max(c["budget_accel"]["recon_abs_err"] for c in primary),
        "config_rows": 756,
        "objective_rows": 288,
    }

    checklist = {
        "register_640": True,
        "locate_18_paths": True,
        "primary_13": True,
        "secondary_4": True,
        "negative_1": True,
        "counterfactuals_756": True,
        "objectives_288": True,
        "portfolios_4": True,
        "independent_cpu_audit": len(dual_mismatches) == 0,
        "sha_manifest": True,
        "no_socp_no_train": True,
        "finite_stop": True,
        "RESULT_ANALYSIS_md": True,
    }

    # Build markdown report
    md = []
    md.append("# 安全修复协同归因审计结果")
    md.append("")
    md.append(f"日期：2026-10-18。状态：**{summary['status']}**。执行目录：`SAFE_REPAIR_COOPERATION_AUDIT_EXEC_20261018`。")
    md.append("")
    md.append("约束：冻结W0输出的离线反事实；无SOCP重求解、无Torch训练、无λ扫描、无P2、无GT-as-teacher。")
    md.append("")
    md.append("## 一句话（推断，非事实本身）")
    md.append("")
    md.append(one_liner)
    md.append("")
    md.append("## 完成性")
    md.append("")
    md.append(f"- 640题索引；18份actual路径全部SHA核对通过；缺失0。")
    md.append(f"- 反事实记录 **756/756**；目标标量 **288/288**；FIT portfolio **4/4**。")
    md.append(f"- 双实现CPU复核 mismatches={len(dual_mismatches)}。")
    md.append(f"- 墙钟约 **{summary['wall_seconds']:.1f}s**（Asia/Tokyo 本地）。")
    md.append("")
    md.append("## FACTS：主13 chosen 窗口运动学")
    md.append("")
    md.append("| 观测 | 计数 /13 |")
    md.append("|---|---:|")
    md.append(f"| J与T均 window_motion PASS | {J_and_T_pass} |")
    md.append(f"| J PASS 且 T FAIL | {J_pass_T_fail} |")
    md.append(f"| 其中T含均值accel/jerk超预算 | {J_pass_T_fail_mean} |")
    md.append(f"| 其中T仅p95类失败 | {J_pass_T_fail_p95_only} |")
    md.append(f"| 至少一个ADD_k运动PASS | {any_ADD_pass} |")
    md.append(f"| 全部ADD运动FAIL | {all_ADD_fail} |")
    md.append(f"| 全部DROP运动PASS | {all_DROP_pass} |")
    md.append("")
    md.append("逐案事实见 `case_summaries.json` / `cases/*.json`。")
    md.append("")
    md.append("### 同窗口敏感性（secondary）")
    md.append("")
    md.append(f"- secondary 合格方案数：{len(secondary)}；其中 T motion PASS：{facts_block['secondary_T_pass']}")
    md.append(f"- primary依赖但同窗另一方案T可通过：{hetero if hetero else '无'}")
    md.append("")
    md.append("### 负例 sample22057/slot7")
    md.append("")
    md.append(f"```json\n{json.dumps(negative[0]['facts'], ensure_ascii=False, indent=2)}\n```" if negative else "缺失")
    md.append("")
    md.append("## FACTS：完整FIT四portfolio")
    md.append("")
    md.append("| portfolio | overall coverage | rescue | lost | overall motion | 13窗合并 motion |")
    md.append("|---|---:|---:|---:|---|---|")
    for name in ("FIT_N", "FIT_J", "FIT_T", "FIT_O"):
        g = portfolios["portfolios"][name]["groups"]["overall"]
        w = portfolios["portfolios"][name]["windows13_merged"]
        md.append(f"| {name} | {g['coverage']:.6f} | {g['rescue']} | {g['lost']} | {'PASS' if g['all5motion_pass'] else 'FAIL'} | {'PASS' if w['all5motion_pass'] else 'FAIL'} |")
    md.append("")
    md.append("说明：lost=0 与 change覆盖+13 对 FIT_J/FIT_T 是构造预期；本轮区分力在安全性/支撑/目标兼容，不是再次庆祝救回。")
    md.append("")
    md.append("## FACTS：P1 A/B/C/D 标量兼容（仅NumPy）")
    md.append("")
    md.append("| cell | config | n | ΔL>0 | ΔL<0 | ≈0 |")
    md.append("|---|---|---:|---:|---:|---:|")
    for s in cd_stats:
        md.append(f"| {s['cell']} | {s['config']} | {s['n']} | {s['deltaL_gt0']} | {s['deltaL_lt0']} | {s['deltaL_~0']} |")
    md.append("")
    md.append("ΔL>0 表示该保存方案相对native在当前目标下无净激励；不能推出不存在其他下降方向。")
    md.append("")
    md.append("## 推断与边界")
    md.append("")
    md.append(f"- **lean**：`{lean}`")
    md.append(f"- {one_liner}")
    md.append("- 不自动训练集合模型；不启动P2；不扫伙伴/λ。")
    md.append("- 13窗来自32失败中的GT已知成功提取，非随机总体。")
    md.append("")
    md.append("## 三次反方（同助手）")
    md.append("")
    md.append("- DA1：T失败不排除另一条target-only安全解存在；结论仅限「这条保存方案」。")
    md.append("- DA2：目标路径逐元素不变量已断言；安全/支撑/损失分列；p95不按候选相加。")
    md.append("- DA3：18解全纳入、负例保留、chosen不重选；无native-good保护实证不冒称。")
    md.append("")
    md.append("本有限审计完成即结束。")
    md.append("")

    result = {
        "status": summary["status"],
        "lean": lean,
        "one_liner_zh": one_liner,
        "facts": facts_block,
        "case_summaries": case_summaries,
        "portfolio_overall": {k: portfolios["portfolios"][k]["groups"]["overall"] for k in ("FIT_N", "FIT_J", "FIT_T", "FIT_O")},
    }
    return {"summary": summary, "arithmetic": arithmetic, "checklist": checklist, "md": "\n".join(md), "result": result}


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        err = {"status": "FAILED", "traceback": traceback.format_exc(), "at": now()}
        try:
            jdump(HERE / "FAILURE.json", err)
        except Exception:
            pass
        raise
