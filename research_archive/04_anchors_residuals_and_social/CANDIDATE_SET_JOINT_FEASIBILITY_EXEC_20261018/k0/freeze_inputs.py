"""freeze_inputs: manifest + SHA + ID/split/pool binding. BLOCKED on failure."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .config import (
    EXEC, DESIGN, CACHE, LABELS, SPLITS, ENCODER, SEL, CPV, CLAMP, SEEDS, SIZES, ROOT,
)
from .data_io import file_sha, array_sha, load_bundle, make_fit_cal_indices, load_kin


def run_freeze(out_dir: Path | None = None) -> dict:
    out_dir = out_dir or (EXEC / "freeze")
    out_dir.mkdir(parents=True, exist_ok=True)
    status = "OK"
    errors = []
    files = {}
    required = {
        "eth_train.npz": CACHE,
        "labels_train.npz": LABELS,
        "splits.npz": SPLITS,
        "train_encoder.npz": ENCODER,
        "FIT_CAL_SPLIT.json": CPV / "results" / "v1" / "FIT_CAL_SPLIT.json",
    }
    for seed in SEEDS:
        required[f"KIN_seed{seed}.pt"] = CLAMP / "results" / "short" / f"KIN_seed{seed}.pt"
        required[f"kin_FIT_seed{seed}.npy"] = SEL / "results" / "cache_pred" / f"kin_FIT_seed{seed}.npy"
        required[f"kin_outer_seed{seed}.npy"] = SEL / "results" / "cache_pred" / f"kin_outer_seed{seed}.npy"
        required[f"kin_inner_seed{seed}.npy"] = CPV / "artifacts" / "v1" / f"kin_inner_seed{seed}.npy"
        for split in ("outer", "inner", "FIT_cal", "FIT_train"):
            required[f"bundle_{split}_seed{seed}"] = CPV / "artifacts" / "v1" / f"{split}_seed{seed}_bundle.npz"
            required[f"json_{split}_seed{seed}"] = CPV / "results" / "v1" / f"seed{seed}_{split}.json"

    for name, path in required.items():
        if not path.exists():
            errors.append(f"MISSING {name}: {path}")
            files[name] = dict(path=str(path), exists=False)
            status = "BLOCKED"
        else:
            files[name] = dict(path=str(path), exists=True, sha256=file_sha(path), size=path.stat().st_size)

    # design docs SHA
    design_files = {}
    if DESIGN.exists():
        for p in sorted(DESIGN.glob("*.md")):
            design_files[p.name] = dict(sha256=file_sha(p), size=p.stat().st_size)

    binding = {}
    try:
        data = load_bundle()
        binding["sample_id_sha"] = array_sha(data["sample_id"])
        binding["groups_sha"] = array_sha(data["groups"])
        binding["native_sha"] = array_sha(data["native"])
        binding["gt_sha"] = array_sha(data["gt"])
        binding["past_sha"] = array_sha(data["past"])
        binding["split_sizes"] = {k: int(len(v)) for k, v in data["split"].items()}
        for k, n in SIZES.items():
            if binding["split_sizes"][k] != n:
                errors.append(f"SIZE {k}: {binding['split_sizes'][k]} != {n}")
                status = "BLOCKED"
        fit_train, fit_cal, rec = make_fit_cal_indices(data)
        binding["FIT_train_n"] = int(len(fit_train))
        binding["FIT_cal_n"] = int(len(fit_cal))
        binding["FIT_train_id_sha"] = array_sha(data["sample_id"][fit_train])
        binding["FIT_cal_id_sha"] = array_sha(data["sample_id"][fit_cal])
        binding["FIT_CAL_SPLIT"] = rec
        # pool binding: kin shapes + first/last sample align via length
        for seed in SEEDS:
            kin_o, src_o = load_kin("outer", seed, data)
            kin_i, src_i = load_kin("inner", seed, data)
            kin_f, src_f = load_kin("FIT", seed, data)
            binding[f"kin_outer_seed{seed}"] = dict(
                shape=list(kin_o.shape), sha=array_sha(kin_o), src=src_o,
            )
            binding[f"kin_inner_seed{seed}"] = dict(
                shape=list(kin_i.shape), sha=array_sha(kin_i), src=src_i,
            )
            binding[f"kin_FIT_seed{seed}"] = dict(
                shape=list(kin_f.shape), sha=array_sha(kin_f), src=src_f,
            )
            if kin_o.shape != (SIZES["outer"], 20, 12, 2):
                errors.append(f"kin_outer seed{seed} shape {kin_o.shape}")
                status = "BLOCKED"
        # original_CAL kin presence
        cal_status = {}
        for seed in SEEDS:
            kin_c, src_c = load_kin("original_CAL", seed, data)
            if kin_c is None:
                cal_status[str(seed)] = dict(status="BLOCKED_NO_KIN_CACHE", src=src_c)
            else:
                cal_status[str(seed)] = dict(status="OK", shape=list(kin_c.shape), src=src_c, sha=array_sha(kin_c))
        binding["original_CAL_kin"] = cal_status
    except Exception as e:
        errors.append(f"LOAD_FAIL: {type(e).__name__}: {e}")
        status = "BLOCKED"
        data = None
        fit_train = fit_cal = None

    manifest = dict(
        tag="freeze_inputs",
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        timezone_label="Asia/Tokyo",
        status=status,
        errors=errors,
        files=files,
        design_files=design_files,
        binding=binding,
        note="Current取证 SHA; does not claim historical training-time lock.",
    )
    (out_dir / "input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "SOURCE_PRESERVATION.json").write_text(
        json.dumps(dict(
            old_results_readonly=True,
            paths_not_modified=[
                str(CPV), str(SEL), str(CLAMP), str(CACHE), str(LABELS), str(SPLITS), str(ENCODER),
            ],
            oracle_tag="ORACLE_DIAGNOSTIC_ONLY",
            k1_teacher_whitelist=[],
        ), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return dict(status=status, manifest=manifest, data=data, fit_train=fit_train, fit_cal=fit_cal, errors=errors)
