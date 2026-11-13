"""加载冻结数据与 track 身份。"""
from __future__ import annotations
from pathlib import Path
import hashlib
import numpy as np

ROOT = Path(r"<WORKSPACE_ROOT>")
CACHE = ROOT / "REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz"
LABELS = ROOT / "F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz"
SPLITS = ROOT / "GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz"
ENCODER = ROOT / "GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz"
CLAMP = ROOT / "KINEMATIC_DECODER_SPEED_CLAMP_20261018"
SEL = ROOT / "UNION_SELECTOR_K40_TO_K20_20261018"
SIZES = dict(FIT=18909, outer=1035, inner=1856, original_CAL=2879)


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def array_sha(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr).tobytes())
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    return h.hexdigest()


def load_bundle():
    with np.load(CACHE, allow_pickle=False) as z:
        sample_id = z["sample_id"].copy()
        native = z["candidate"][..., :24].reshape(-1, 20, 12, 2).astype(np.float64)
        groups = z["change_group"].copy()
    with np.load(LABELS, allow_pickle=False) as z:
        assert np.array_equal(z["sample_id"], sample_id)
        gt = z["future"].astype(np.float64)
    with np.load(ENCODER, allow_pickle=False) as z:
        assert np.array_equal(z["sample_id"], sample_id)
        past = z["past"].astype(np.float64)
    assert np.allclose(past[:, -1], 0.0)
    sp = np.load(SPLITS, allow_pickle=False)
    assert np.array_equal(sp["sample_id"], sample_id[:25912])
    split = dict(
        FIT=sp["grouped_train_index"].copy(),
        outer=sp["outer_eval_index"].copy(),
        inner=sp["inner_cal_index"].copy(),
        original_CAL=np.arange(25912, 28791),
    )
    for k, n in SIZES.items():
        assert len(split[k]) == n
    # track identity for first 25912
    n_id = 25912
    seq = np.array(["?"] * len(sample_id), dtype=object)
    ped = np.full(len(sample_id), -1, dtype=np.int64)
    scene = np.array(["?"] * len(sample_id), dtype=object)
    seq[:n_id] = sp["sequence"]
    ped[:n_id] = sp["pedestrian_id"].astype(np.int64)
    scene[:n_id] = sp["scene_family"]
    track_key = np.array([f"{s}|{p}" for s, p in zip(seq, ped)], dtype=object)
    return dict(
        sample_id=sample_id,
        past=past,
        gt=gt,
        native=native,
        groups=groups,
        split=split,
        sequence=seq,
        pedestrian_id=ped,
        scene_family=scene,
        track_key=track_key,
    )
