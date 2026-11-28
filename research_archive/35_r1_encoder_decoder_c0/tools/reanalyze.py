"""Independent standard-library verifier for Phase 35 (R1 Encoder C1/C2 + Decoder Cascade/Cover C0)."""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PROTOCOL = "c1eb3a6cca5e68b2050060158c44a47cbd57fcc63187de12a3a7958a29726b2c"
SEEDS = (42, 43, 44)


def strict_loads(path: Path):
    def bad_const(val: str):
        raise ValueError(f"Disallowed JSON constant {val!r} in {path}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=bad_const)


def verify_files() -> dict[str, str]:
    manifest = strict_loads(ROOT / "FILES.json")
    actual = {}
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        if rel == "FILES.json" or "__pycache__" in p.parts:
            continue
        actual[rel] = sha256(p.read_bytes()).hexdigest()
    if actual != manifest:
        missing = sorted(set(manifest) - set(actual))
        extra = sorted(set(actual) - set(manifest))
        changed = sorted(k for k in set(actual) & set(manifest) if actual[k] != manifest[k])
        raise AssertionError(f"FILES.json mismatch: missing={missing}, extra={extra}, changed={changed}")
    return manifest


def check_study(name: str, expected_arms: tuple[str, ...]) -> dict:
    sdir = ROOT / "results" / name
    status = strict_loads(sdir / "STATUS.json")
    runtime = strict_loads(sdir / "RUNTIME.json")
    summary = strict_loads(sdir / "STUDY_SUMMARY.json")
    assert status["status"] == "COMPLETE" and status["study"] == name
    assert status["protocol"] == EXPECTED_PROTOCOL and runtime["protocol"] == EXPECTED_PROTOCOL
    assert tuple(summary["arms"]) == expected_arms
    assert tuple(summary["seeds"]) == SEEDS

    for arm in expected_arms:
        seed_metrics = []
        for s in SEEDS:
            res = strict_loads(sdir / "runs" / f"{arm}_s{s}" / "RESULT.json")
            assert res["status"] == "PASS"
            assert res["contract"]["protocol"] == EXPECTED_PROTOCOL
            for v in res["metrics"].values():
                if isinstance(v, (int, float)):
                    assert math.isfinite(float(v)), f"Non-finite {arm}_s{s}"
            seed_metrics.append(res["metrics"])
        row = summary["arm_means"][arm]
        for k, entry in row.items():
            if not isinstance(entry, dict) or "mean" not in entry or "by_seed" not in entry:
                continue
            recomputed_by_seed = [float(m[k]) for m in seed_metrics]
            recomputed_mean = sum(recomputed_by_seed) / len(recomputed_by_seed)
            assert abs(recomputed_mean - float(entry["mean"])) <= 1e-12, (name, arm, k)
            for a, b in zip(recomputed_by_seed, entry["by_seed"]):
                assert abs(a - float(b)) <= 1e-12, (name, arm, k)
    return summary


def main() -> None:
    manifest = verify_files()
    enc = check_study("encoder", ("E0", "E1", "E2", "E3", "E4", "E5", "E6"))
    dec = check_study("decoder", ("M0", "M1", "M2", "M3", "M4", "M5", "M6", "JM4"))

    assert enc["manipulation"]["MC1_all_pass"] is True
    assert enc["manipulation"]["MC2_E3_all_pass"] is True
    assert enc["manipulation"]["MC3_all_pass"] is True
    assert dec["manipulation"]["MC_b_pass"] is True
    assert dec["manipulation"]["MC_c"]["pass_flag"] is True
    assert dec["manipulation"]["MC_d"]["pass_flag"] is True

    v2 = strict_loads(ROOT / "results/encoder/V2_SCALING.json")
    n128 = {r["impl"]: r for r in v2 if r["batch_size"] == 64 and r["n"] == 128}
    n256 = {r["impl"]: r for r in v2 if r["batch_size"] == 64 and r["n"] == 256}
    assert n128["fixed_k_sparse"]["status"] == "OK" and n128["masked_dense_E0"]["status"] == "OK"
    assert n128["fixed_k_sparse"]["mean_ms"] < n128["masked_dense_E0"]["mean_ms"]
    assert n256["masked_dense_E0"]["status"] == "OOM" and n256["fixed_k_sparse"]["status"] == "OK"

    v0 = strict_loads(ROOT / "v0/V0_REPORT.json")
    assert v0["V0_c_c2_zero_init_equivalence"]["pass"] is True
    assert v0["c0_dev_full_pass"] is True
    assert v0["strict_design_6_2_all_folds_k3_k5_pass"] is False

    print(
        f"REANALYZE_PASS files={len(manifest)} "
        f"encoder_arms={len(enc['arms'])} decoder_arms={len(dec['arms'])} "
        f"protocol={EXPECTED_PROTOCOL}"
    )


if __name__ == "__main__":
    main()
