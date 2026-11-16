"""E1 analysis: run-to-run noise floor, method-vs-B0 excess exchange, GT-referenced motion.

Numpy only, reads saved outputs, never trains or runs a model.

Inputs
  --data      C0 dev (or fit-calib) npz with keys past[N,9,8,2], future[N,12,2], ids[N], group[N]
              (group 0 stable / 1 transition / 2 change, same as C0).
  --ref       prediction npz files of the reference config (B0), one per seed, each with
              paths[N,20,12,2] (world, metres), logits[N,20], ids[N].
  --method    NAME file_seed42 file_seed43 ...   (repeatable; files in the SAME seed order as --ref)
  --motion-ref C0/MOTION_REFERENCE.json  (FIT-GT pooled reference, sealed before B0)
  --out       output json (refuses to overwrite)

Definitions (identical to the current E2E evaluation)
  good       same slot ADE<=.20 and FDE<=.40; coverage = any good slot
  rescue/lost of B relative to A: windows good in B but not A / good in A but not B
  motion     12 v, 11 a, 10 j with the last observed point prepended; junction a/j as causal_dataset.motion
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

DT = 0.4
GROUPS = ("overall", "stable", "transition", "change")
MOTION_KEYS = ("speed_p95", "accel_mean", "accel_p95", "jerk_mean", "jerk_p95",
               "junction_accel_mean", "junction_accel_p95", "junction_jerk_mean", "junction_jerk_p95")


# ---------------------------------------------------------------- quality
def quality(paths, future):
    e = np.linalg.norm(paths - future[:, None], axis=-1)
    ade, fde = e.mean(-1), e[..., -1]
    good = (ade <= 0.2) & (fde <= 0.4)
    j = np.maximum(ade / 0.2, fde / 0.4)
    return dict(ADE=ade, FDE=fde, J=j, good=good, coverage=good.any(1), J_winner=j.argmin(1))


def group_masks(group):
    group = np.asarray(group)
    return {"overall": np.ones(len(group), bool), "stable": group == 0,
            "transition": group == 1, "change": group == 2}


def pair_exchange(cov_ref, cov_new, group):
    out = {}
    for name, m in group_masks(group).items():
        ref_good = int((cov_ref & m).sum())
        rescue = int((cov_new & ~cov_ref & m).sum())
        lost = int((~cov_new & cov_ref & m).sum())
        n = int(m.sum())
        out[name] = dict(n=n, ref_good=ref_good, new_good=int((cov_new & m).sum()), rescue=rescue, lost=lost,
                         lost_rate=lost / ref_good if ref_good else None,
                         coverage_diff_pp=100.0 * (int((cov_new & m).sum()) - ref_good) / n if n else None)
    return out


# ---------------------------------------------------------------- motion
def motion_arrays(paths, past_focal):
    """paths [N,K,12,2]; past_focal [N,8,2] -> dict of arrays (pooled later)."""
    p = np.broadcast_to(past_focal[:, None], paths.shape[:2] + past_focal.shape[1:])
    v = np.diff(np.concatenate([p[..., -1:, :], paths], axis=-2), axis=-2) / DT
    a = np.diff(v, axis=-2) / DT
    j = np.diff(a, axis=-2) / DT
    pv = (p[..., -1, :] - p[..., -2, :]) / DT
    pa = (p[..., -1, :] - 2 * p[..., -2, :] + p[..., -3, :]) / DT ** 2
    ja = (v[..., 0, :] - pv) / DT
    jj = (ja - pa) / DT
    n = np.linalg.norm
    return dict(speed=n(v, axis=-1), acceleration=n(a, axis=-1), jerk=n(j, axis=-1),
                junction_acceleration=n(ja, axis=-1), junction_jerk=n(jj, axis=-1))


def pooled(m):
    q = lambda x: float(np.quantile(x, 0.95))
    return dict(speed_p95=q(m["speed"]), accel_mean=float(m["acceleration"].mean()),
                accel_p95=q(m["acceleration"]), jerk_mean=float(m["jerk"].mean()), jerk_p95=q(m["jerk"]),
                junction_accel_mean=float(m["junction_acceleration"].mean()),
                junction_accel_p95=q(m["junction_acceleration"]),
                junction_jerk_mean=float(m["junction_jerk"].mean()),
                junction_jerk_p95=q(m["junction_jerk"]))


def motion_stats(paths, past_focal):
    return pooled(motion_arrays(paths, past_focal))


def gt_ratio(model_stats, gt_stats):
    """ratio = model / GT and |log ratio| (0 = matches GT level; symmetric in over/under)."""
    out = {}
    for k, g in gt_stats.items():
        if k not in model_stats:
            continue
        r = model_stats[k] / g if g > 0 else None
        out[k] = dict(model=model_stats[k], gt=g, ratio=r,
                      abs_log_ratio=abs(float(np.log(r))) if r else None)
    return out


def fit_reference(motion_ref_json):
    raw = json.loads(Path(motion_ref_json).read_text(encoding="utf-8"))["raw"]
    return dict(speed_p95=raw["speed"]["p95"], accel_mean=raw["acceleration"]["mean"],
                accel_p95=raw["acceleration"]["p95"], jerk_mean=raw["jerk"]["mean"], jerk_p95=raw["jerk"]["p95"],
                junction_accel_mean=raw["junction_acceleration"]["mean"],
                junction_accel_p95=raw["junction_acceleration"]["p95"],
                junction_jerk_mean=raw["junction_jerk"]["mean"],
                junction_jerk_p95=raw["junction_jerk"]["p95"])


# ---------------------------------------------------------------- population summary
def summarize(pred, data, fit_ref=None):
    q = quality(pred["paths"], data["future"])
    top = pred["logits"].argmax(1)
    row = np.arange(len(top))
    past = data["past"][:, 0]
    out = {}
    for name, m in group_masks(data["group"]).items():
        cov = q["coverage"][m]
        top_good = q["good"][row, top][m]
        allk = motion_stats(pred["paths"][m], past[m])
        oracle = motion_stats(pred["paths"][m][np.arange(m.sum()), q["J_winner"][m]][:, None], past[m])
        top1 = motion_stats(pred["paths"][m][np.arange(m.sum()), top[m]][:, None], past[m])
        gt = motion_stats(data["future"][m][:, None], past[m])
        out[name] = dict(
            n=int(m.sum()), coverage=float(cov.mean()), good=int(cov.sum()),
            minADE=float(q["ADE"][m].min(1).mean()), minFDE=float(q["FDE"][m].min(1).mean()),
            top1_coverage=float(top_good.mean()),
            top1_conditional_recall=float(top_good[cov].mean()) if cov.any() else None,
            motion_allK20=allk, motion_top1=top1, motion_J_oracle=oracle, motion_same_population_GT=gt,
            # J-oracle vs its own GT windows: the best candidate should move like the truth.
            oracle_vs_GT=gt_ratio(oracle, gt),
            allK20_vs_fitGT=gt_ratio(allk, fit_ref) if fit_ref else None)
    return out, q


def _stats(values):
    v = np.asarray([x for x in values if x is not None], float)
    if not len(v):
        return None
    return dict(n=len(v), mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else None,
                min=float(v.min()), max=float(v.max()))


def null_distribution(ref_q):
    """All ordered pairs (A as reference, B as new) of reference-config runs."""
    pairs = {}
    for (ia, qa), (ib, qb) in itertools.permutations(enumerate(ref_q), 2):
        pairs[f"{ia}->{ib}"] = pair_exchange(qa["coverage"], qb["coverage"], ref_q.group)
    summary = {}
    for g in GROUPS:
        lost = [p[g]["lost_rate"] for p in pairs.values()]
        diff = [p[g]["coverage_diff_pp"] for p in pairs.values()]
        summary[g] = dict(lost_rate=_stats(lost), coverage_diff_pp=_stats(diff),
                          legacy_lost_le_1pct_pass_fraction=float(np.mean([x is not None and x <= .01 for x in lost])))
    return pairs, summary


class QList(list):
    group = None


def load(npz, ids):
    d = dict(np.load(npz, allow_pickle=False))
    order = {k: i for i, k in enumerate(d["ids"].tolist())}
    idx = np.array([order[k] for k in ids.tolist()])
    return dict(paths=d["paths"][idx].astype(np.float64), logits=d["logits"][idx].astype(np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--method", nargs="+", action="append", default=[])
    ap.add_argument("--motion-ref")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out_path = Path(a.out)
    if out_path.exists():
        raise FileExistsError(out_path)
    data = dict(np.load(a.data, allow_pickle=False))
    data = dict(past=data["past"].astype(np.float64), future=data["future"].astype(np.float64),
                group=data["group"], ids=data["ids"])
    fit_ref = fit_reference(a.motion_ref) if a.motion_ref else None
    result = dict(reference=a.ref, populations={}, null={}, methods={})
    ref_q = QList(); ref_q.group = data["group"]
    for f in a.ref:
        s, q = summarize(load(f, data["ids"]), data, fit_ref)
        result["populations"][f] = s; ref_q.append(q)
    if len(ref_q) >= 2:
        result["null"]["pairs"], result["null"]["summary"] = null_distribution(ref_q)
    for spec in a.method:
        name, files = spec[0], spec[1:]
        if len(files) > len(ref_q):
            raise ValueError("method has more seeds than reference")
        rows, mq = {}, []
        for i, f in enumerate(files):
            s, q = summarize(load(f, data["ids"]), data, fit_ref)
            result["populations"][f] = s; mq.append(q)
            rows[f"seed_index{i}"] = pair_exchange(ref_q[i]["coverage"], q["coverage"], data["group"])
        cmp = {}
        null = result["null"].get("summary", {})
        for g in GROUPS:
            diffs = [rows[k][g]["coverage_diff_pp"] for k in rows]
            lost = [rows[k][g]["lost_rate"] for k in rows]
            nsd = (null.get(g, {}).get("coverage_diff_pp") or {}).get("sd")
            nlost = (null.get(g, {}).get("lost_rate") or {}).get("mean")
            mean = float(np.mean(diffs))
            cmp[g] = dict(
                coverage_diff_pp=_stats(diffs), lost_rate=_stats(lost),
                excess_lost_rate=(float(np.mean(lost)) - nlost) if nlost is not None else None,
                null_sd_coverage_diff_pp=nsd,
                # mean of n paired seeds: noise SD ~ null_sd/sqrt(n)
                signal_threshold_pp=(2 * nsd / np.sqrt(len(diffs))) if nsd else None,
                all_seeds_same_sign=bool(np.all(np.sign(diffs) == np.sign(mean))) if mean else False,
                signal=bool(nsd and abs(mean) > 2 * nsd / np.sqrt(len(diffs))
                            and np.all(np.sign(diffs) == np.sign(mean))))
        result["methods"][name] = dict(files=files, paired_exchange=rows, comparison=cmp)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1, allow_nan=False)
    print("written", out_path)


if __name__ == "__main__":
    main()
