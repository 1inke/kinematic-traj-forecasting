"""V0：输入冻结 + 原选择器产物探查；不训练。"""
from __future__ import annotations
import json
from pathlib import Path
from kdec.io_data import ROOT, CLAMP, SEL, CACHE, LABELS, SPLITS, ENCODER, file_sha, load_bundle

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "v0"
OUT.mkdir(parents=True, exist_ok=True)


def find_selector_artifacts():
    patterns = [
        "**/*hgb*", "**/*joblib*", "**/*ranker*",
        "**/selected*", "**/scores*", "**/pool_idx*",
        "**/*indices*", "**/*paths*.npy",
    ]
    found = []
    for base in (SEL, CLAMP):
        if not base.exists():
            continue
        for pat in patterns:
            for p in base.glob(pat):
                if p.is_file():
                    found.append(str(p))
    # also list results tree
    result_files = []
    r = SEL / "results"
    if r.exists():
        for p in r.rglob("*"):
            if p.is_file():
                result_files.append(str(p.relative_to(SEL)))
    return found, result_files


def main():
    data = load_bundle()
    sources = {
        "CACHE": str(CACHE),
        "LABELS": str(LABELS),
        "SPLITS": str(SPLITS),
        "ENCODER": str(ENCODER),
    }
    shas = {k: file_sha(Path(v)) for k, v in sources.items()}
    ckpts = {}
    caches = {}
    for seed in (42, 43, 44):
        ck = CLAMP / "results" / "short" / f"KIN_seed{seed}.pt"
        ckpts[seed] = dict(path=str(ck), exists=ck.exists(), sha=file_sha(ck) if ck.exists() else None)
        for split in ("FIT", "outer"):
            c = SEL / "results" / "cache_pred" / f"kin_{split}_seed{seed}.npy"
            caches[f"{split}_{seed}"] = dict(
                path=str(c), exists=c.exists(),
                sha=file_sha(c) if c.exists() else None,
                shape=list(__import__("numpy").load(c).shape) if c.exists() else None,
            )
    found, result_files = find_selector_artifacts()
    # decide replay
    has_hgb = any("hgb" in f.lower() or f.endswith(".joblib") for f in found)
    has_idx = any("selected" in f.lower() or "indices" in f.lower() for f in found)
    replay = "AVAILABLE" if (has_hgb and has_idx) else "ORIGINAL_SELECTOR_REPLAY_UNAVAILABLE"

    manifest = dict(
        status="V0_COMPLETE",
        replay_status=replay,
        note="当前取证；不伪称历史训练时锁定。无 HGB/selected_indices 时转 V1 重建。",
        splits={k: int(len(v)) for k, v in data["split"].items()},
        source_sha=shas,
        checkpoints=ckpts,
        kin_caches=caches,
        selector_artifact_hits=found,
        selector_result_files=result_files,
        locked_primary_method="cond_boost_11_9",
        historical_strategy_family_archived=True,
        no_generator_training=True,
    )
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "ORIGINAL_SELECTOR_REPLAY_UNAVAILABLE.txt" if replay != "AVAILABLE"
     else OUT / "ORIGINAL_SELECTOR_REPLAY_AVAILABLE.txt").write_text(
        replay + "\n" + manifest["note"] + "\n", encoding="utf-8"
    )
    print(json.dumps({k: manifest[k] for k in ("status", "replay_status", "no_generator_training")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
