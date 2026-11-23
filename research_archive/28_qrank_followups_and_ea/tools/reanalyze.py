"""Verify and reanalyze the three published follow-up studies with NumPy."""

from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
QRANK = ROOT.parent / "27_qrank_mldec/EXP_QRANK_MLDEC_L4_20261122/code/sealed"
sys.path[:0] = [str(QRANK), str(ROOT / "decel_followup")]
from analysis import summarize
from qrank_decisions import paired, short_compare


def read(path):
    return json.loads(path.read_bytes())


def compare(left, right):
    if isinstance(right, dict):
        if left.keys() != right.keys():
            raise ValueError("Summary fields differ")
        for k in right:
            compare(left[k], right[k])
    elif isinstance(right, (tuple, list)):
        if len(left) != len(right):
            raise ValueError("Summary lengths differ")
        for a, b in zip(left, right):
            compare(a, b)
    elif isinstance(right, float):
        if not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"Recomputed {left} differs from archived {right}")
    elif left != right:
        raise ValueError(f"Recomputed {left} differs from archived {right}")


def array(path):
    receipt = read(path.with_name("RESULT.json"))
    if receipt["status"] != "PASS" or sha256(path.read_bytes()).hexdigest() != receipt["metrics_sha256"]:
        raise ValueError(f"Invalid metrics receipt: {path}")
    with np.load(path, allow_pickle=False) as z:
        values = {key: z[key] for key in z.files}
    if "windows" in receipt and len(values["ids"]) != receipt["windows"]:
        raise ValueError(f"Unexpected window count: {path}")
    return values


def replay_studies():
    frozen = ROOT / "frozen_replay/results"
    followup = ROOT / "decel_followup/results"
    baseline = read(frozen / "SUMMARY.json")
    geometry = read(followup / "SUMMARY.json")
    windows = 0
    for fold, seeds in baseline["folds"].items():
        for arm, groups in baseline["report"][fold].items():
            names = [f"{arm}_{fold}_s{seed}" for seed in seeds]
            base_arrays = [array(frozen / "runs" / name / "metrics.npz") for name in names]
            geometry_arrays = [array(followup / "runs" / name / "metrics.npz") for name in names]
            for earlier, later in zip(base_arrays, geometry_arrays):
                for key in ("ids", "fast", "fast_deceleration"):
                    np.testing.assert_array_equal(earlier[key], later[key])
                for layer in ("first", "final"):
                    for metric, selector in (("minFDE", "best"), ("Top1_FDE", "top1")):
                        np.testing.assert_allclose(earlier[f"{layer}_conf/{metric}"],
                                                   later[f"{layer}/{selector}_fde_t12"], atol=5e-4, rtol=0)
            windows += sum(len(row["ids"]) for row in base_arrays)
            for group, facts in groups.items():
                compare(int(base_arrays[0][group].sum()), facts["windows"])
                for condition, metrics in facts["conditions"].items():
                    for metric, value in metrics.items():
                        observed = np.mean([row[f"{condition}/{metric}"][row[group]].mean()
                                            for row in base_arrays])
                        compare(float(observed), value)
                for trial, control in baseline["conditions"]:
                    for metric, expected in facts["paired"][f"{trial}_vs_{control}"].items():
                        select = lambda condition: [{metric: row[f"{condition}/{metric}"],
                                                     "ids": row["ids"], group: row[group]} for row in base_arrays]
                        compare(short_compare(select(trial), select(control), metric, group), expected)
            compare(summarize(geometry_arrays), geometry["report"][fold][arm])
    return windows


def ea():
    root = ROOT / "e_a_diag/results"
    archives = read(root / "SUMMARY.json")
    count = 0
    for fold, result in archives["inference"].items():
        path = root / fold
        compare(read(path / "RESULT.json"), result)
        audit = read(path / "AUDIT.json")
        compare(audit, archives["audits"][fold])
        if sha256((path / "AUDIT.json").read_bytes()).hexdigest() != result["audit_sha256"]:
            raise ValueError(f"Audit differs from the result receipt: {fold}")
        values = array(path / "metrics.npz")
        count += len(values["ids"])
        for label, reported in (("all", audit["overall"]), *audit["by_recording"].items()):
            mask = np.ones(len(values["ids"]), dtype=bool) if label == "all" else values["recording"] == label
            for key, value in reported.items():
                if key in values:
                    compare(float(values[key][mask].mean()), value)
            conflicts = int(values["proxy_6m_conflicts"][mask].sum())
            omitted = int(values["proxy_6m_omitted"][mask].sum())
            compare(conflicts, reported["proxy_6m_conflicts_total"])
            compare(omitted, reported["proxy_6m_omitted_total"])
            compare(omitted / max(conflicts, 1), reported["proxy_6m_omitted_share"])
            compare(conflicts / max(int(values["proxy_6m_neighbors"][mask].sum()), 1),
                    reported["proxy_6m_conflict_share"])
            compare(int(mask.sum()), reported["windows"])
        masks = dict(overall=np.ones(len(values["ids"]), dtype=bool),
                     archived_omitted=values["causal_any_archived_omission"],
                     risk_exclusive_restore=values["risk_exclusive_restored"] > 0,
                     no_causal_conflict=values["causal_no_conflict"], changed=values["changed"])
        for group, reported in result["groups"].items():
            mask = masks[group]
            compare(int(mask.sum()), reported["windows"])
            compare(len(np.unique(values["tracks"][mask])), reported["tracks"])
            for arm, means in reported["mean"].items():
                for metric, expected in means.items():
                    compare(float(values[f"{arm}_{metric}"][mask].mean()), expected)
            for metric, facts in reported["conflict_vs_distance"].items():
                compare(paired(values[f"conflict_{metric}"][mask], values[f"distance_{metric}"][mask],
                               values["tracks"][mask]), facts)
    return count


def main():
    manifest = (ROOT / "FILES.sha256").read_text(encoding="utf-8").splitlines()
    files = set()
    for line in manifest:
        expected, name = line.split("  ", 1)
        if sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Archive checksum differs: {name}")
        files.add(name)
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()
              and p.name != "FILES.sha256" and "__pycache__" not in p.parts}
    compare(sorted(files), sorted(actual))
    replay_windows = replay_studies()
    ea_windows = ea()
    print(f"PASS: {replay_windows} model-window records in each QRANK study, {ea_windows} E-A windows")


if __name__ == "__main__":
    main()
