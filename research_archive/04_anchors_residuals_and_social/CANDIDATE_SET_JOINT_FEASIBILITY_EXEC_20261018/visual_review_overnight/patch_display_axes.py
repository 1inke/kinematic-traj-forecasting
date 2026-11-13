# -*- coding: utf-8 -*-
"""Patch overnight viz: per-scene display axes (hotel/eth: plot y = -world_y)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\visual_review_overnight")
RAW = Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA_ACCURACY_Q20\datasets\eth_ucy\processed_data_diverse\eth_data_train.npy")

SCENE_BY_ID = {0: "eth", 1: "hotel", 2: "students003", 3: "zara01", 4: "zara02"}
# Same H polarity family as process_eth (eth+hotel treated together): world +y ~ video down.
FLIP_Y = {"eth", "hotel"}


def load_js_assignment(path: Path, prefix: str):
    text = path.read_text(encoding="utf-8")
    if not text.startswith(prefix):
        raise ValueError(f"unexpected prefix in {path.name}: {text[:80]!r}")
    body = text[len(prefix):]
    if body.endswith(";\n"):
        body = body[:-2]
    elif body.endswith(";"):
        body = body[:-1]
    return json.loads(body)


def write_js(path: Path, assignment: str, value) -> None:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    path.write_text(assignment + "=" + text + ";\n", encoding="utf-8")


def flip_y_points(obj, sign: np.ndarray):
    """obj: nested list of [x,y]; sign: (N,) with +1 or -1."""
    arr = np.asarray(obj, dtype=np.float64)
    # broadcast sign over leading axis
    while sign.ndim < arr.ndim:
        sign = sign[..., None]
    out = arr.copy()
    out[..., 1] = arr[..., 1] * sign[..., 0]
    return np.round(out, 5)


def main():
    common = load_js_assignment(HERE / "data" / "common.js", "window.AUDIT_COMMON=")
    ids = np.asarray(common["ids"], dtype=np.int64)
    n = len(ids)
    raw = np.load(RAW, mmap_mode="r")
    scene_ids = np.asarray(raw[ids, 0, 0, 0], dtype=np.int64)
    scenes = [SCENE_BY_ID.get(int(s), f"unknown_{int(s)}") for s in scene_ids]
    y_sign = np.array([-1.0 if sc in FLIP_Y else 1.0 for sc in scenes], dtype=np.float64)

    # Idempotency: if already patched, skip re-flip of coords but refresh metadata.
    already = common.get("protocol", {}).get("coord_frame") == "world_display_remapped"
    if already and common.get("scenes") == scenes:
        print(json.dumps({"status": "ALREADY_PATCHED", "n": n, "flip_counts": {k: scenes.count(k) for k in sorted(set(scenes))}}, ensure_ascii=False))
        return

    if already:
        # Re-deriving from world would need unflipping; refuse silent double-flip.
        raise SystemExit("common.js already remapped but scene list differs; restore from rebuild first")

    # Flip spatial fields in common
    common["past"] = flip_y_points(common["past"], y_sign).tolist()
    common["gt"] = flip_y_points(common["gt"], y_sign).tolist()
    # neighbors: list of (A-1,T,2) per sample — variable length
    new_nb = []
    for i, nb in enumerate(common["neighbors"]):
        if not nb:
            new_nb.append(nb)
            continue
        arr = np.asarray(nb, dtype=np.float64)
        arr = arr.copy()
        arr[..., 1] *= y_sign[i]
        new_nb.append(np.round(arr, 5).tolist())
    common["neighbors"] = new_nb
    if "origin" in common:
        common["origin"] = flip_y_points(common["origin"], y_sign).tolist()
    if "heading" in common:
        # heading is a direction vector; flip y component to match display frame
        common["heading"] = flip_y_points(common["heading"], y_sign).tolist()

    common["scenes"] = scenes
    common["scene_ids"] = scene_ids.astype(int).tolist()
    common["display_y_sign"] = y_sign.astype(int).tolist()
    proto = dict(common.get("protocol") or {})
    proto["coord_frame"] = "world_display_remapped"
    proto["coord_note"] = (
        "展示坐标：世界米制，按场景显示重映射。"
        "students003/zara01/zara02 画 (x,y)；hotel/eth 画 (x,-y)，"
        "因 H 极性使世界 +y 朝视频下方。不是全场景通用「视频同向上」。"
    )
    proto["display_flip_y_scenes"] = sorted(FLIP_Y)
    proto["scene_id_map"] = SCENE_BY_ID
    common["protocol"] = proto
    write_js(HERE / "data" / "common.js", "window.AUDIT_COMMON", common)

    # Flip each model paths p: (N,K,T,2)
    models = [m["key"] for m in common["models"]]
    for key in models:
        path = HERE / "data" / f"{key}.js"
        payload = load_js_assignment(path, f'window.AUDIT_MODELS["{key}"]=')
        p = np.asarray(payload["p"], dtype=np.float64)
        assert p.shape[0] == n and p.shape[-1] == 2
        sign = y_sign.reshape(n, *([1] * (p.ndim - 1)))
        p = p.copy()
        p[..., 1] *= sign[..., 0]
        payload["p"] = np.round(p, 5).tolist()
        write_js(path, f'window.AUDIT_MODELS["{key}"]', payload)
        print("flipped", key, flush=True)

    # arrays.npz if present
    npz_path = HERE / "data" / "arrays.npz"
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as z:
            data = {k: z[k].copy() for k in z.files}
        for key in ("past", "future", "origin", "heading", "neighbor_past"):
            if key in data and data[key].ndim >= 1 and data[key].shape[-1] == 2:
                arr = data[key].astype(np.float64)
                # neighbor_past: (N,8,8,2) — agents without target
                s = y_sign
                while s.ndim < arr.ndim:
                    s = s.reshape(s.shape + (1,))
                arr = arr.copy()
                arr[..., 1] *= s[..., 0]
                data[key] = arr.astype(np.float32) if data[key].dtype == np.float32 else arr
        for key in models:
            if key in data and data[key].ndim >= 2 and data[key].shape[-1] == 2:
                arr = data[key].astype(np.float64).copy()
                s = y_sign
                while s.ndim < arr.ndim:
                    s = s.reshape(s.shape + (1,))
                arr[..., 1] *= s[..., 0]
                data[key] = arr.astype(np.float32)
        data["scene_id"] = scene_ids.astype(np.int64)
        data["display_y_sign"] = y_sign.astype(np.int8)
        np.savez_compressed(npz_path, **data)

    counts = {k: scenes.count(k) for k in sorted(set(scenes))}
    flip_n = int((y_sign < 0).sum())
    # sanity 740
    i740 = int(np.where(ids == 740)[0][0])
    past740 = common["past"][i740]
    report = {
        "status": "PATCHED_DISPLAY_AXES",
        "n": n,
        "scene_counts": counts,
        "flip_y_n": flip_n,
        "sample_740": {
            "scene": scenes[i740],
            "y_sign": int(y_sign[i740]),
            "past0": past740[0],
            "past_last": past740[-1],
            "dy_past": round(past740[-1][1] - past740[0][1], 5),
        },
    }
    (HERE / "audit_display_axes.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
