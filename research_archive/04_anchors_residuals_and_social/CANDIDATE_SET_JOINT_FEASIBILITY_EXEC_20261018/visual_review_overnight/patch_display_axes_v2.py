# -*- coding: utf-8 -*-
"""Evidence-based overnight viz display transform.

FACT (from BIWI/UCY H + ASTRA world_to_pixel + video overlays):
  eth/hotel:     display = (world_y, -world_x)   # +x down, +y right on video
  students/zara: display = (world_x,  world_y)   # +x right, +y up on video

Previous wrong patch only did (x, -y) for hotel/eth. This script undoes that
then applies the H-validated transform. Does not change training data.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\visual_review_overnight")
RAW = Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy\processed_data_diverse\eth_data_train.npy")

SCENE_BY_ID = {0: "eth", 1: "hotel", 2: "students003", 3: "zara01", 4: "zara02"}
# Scenes where video-aligned display is (y, -x) not (x, y)
SWAP_NEGX = {"eth", "hotel"}
TARGET_FRAME = "world_display_H_aligned_v2"


def load_js(path: Path, prefix: str):
    text = path.read_text(encoding="utf-8")
    if not text.startswith(prefix):
        raise ValueError(f"bad prefix {path.name}: {text[:60]!r}")
    body = text[len(prefix):]
    if body.endswith(";\n"):
        body = body[:-2]
    elif body.endswith(";"):
        body = body[:-1]
    return json.loads(body)


def write_js(path: Path, assignment: str, value) -> None:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    path.write_text(assignment + "=" + text + ";\n", encoding="utf-8")


def to_world_from_current(arr: np.ndarray, scenes, old_frame: str, old_y_sign: np.ndarray | None):
    """Recover world meters from currently stored display coords."""
    out = np.asarray(arr, dtype=np.float64).copy()
    if old_frame == TARGET_FRAME:
        # already final; convert back to world for re-apply idempotency
        for i, sc in enumerate(scenes):
            if sc in SWAP_NEGX:
                # display=(y,-x) => x=-display_y, y=display_x
                x = -out[i, ..., 1]
                y = out[i, ..., 0]
                out[i, ..., 0] = x
                out[i, ..., 1] = y
        return out
    if old_frame == "world_display_remapped" and old_y_sign is not None:
        # previous wrong: hotel/eth stored (x, -y)
        for i, sc in enumerate(scenes):
            if old_y_sign[i] < 0 or sc in SWAP_NEGX:
                out[i, ..., 1] *= -1.0
        return out
    # plain world
    return out


def world_to_display(world: np.ndarray, scenes):
    out = np.asarray(world, dtype=np.float64).copy()
    for i, sc in enumerate(scenes):
        if sc in SWAP_NEGX:
            x = out[i, ..., 0].copy()
            y = out[i, ..., 1].copy()
            out[i, ..., 0] = y
            out[i, ..., 1] = -x
    return out


def transform_nested_list(obj, scenes, old_frame, old_y_sign):
    arr = np.asarray(obj, dtype=np.float64)
    # arr shape (N, ..., 2)
    w = to_world_from_current(arr, scenes, old_frame, old_y_sign)
    d = world_to_display(w, scenes)
    return np.round(d, 5)


def transform_neighbors(neighbors, scenes, old_frame, old_y_sign):
    new = []
    for i, nb in enumerate(neighbors):
        if not nb:
            new.append(nb)
            continue
        arr = np.asarray(nb, dtype=np.float64)
        # fake batch of 1
        w = to_world_from_current(arr[None], [scenes[i]], old_frame,
                                  None if old_y_sign is None else old_y_sign[i:i+1])[0]
        d = world_to_display(w[None], [scenes[i]])[0]
        new.append(np.round(d, 5).tolist())
    return new


def main():
    common = load_js(HERE / "data" / "common.js", "window.AUDIT_COMMON=")
    ids = np.asarray(common["ids"], dtype=np.int64)
    n = len(ids)
    raw = np.load(RAW, mmap_mode="r")
    scene_ids = np.asarray(raw[ids, 0, 0, 0], dtype=np.int64)
    scenes = [SCENE_BY_ID.get(int(s), f"unknown_{int(s)}") for s in scene_ids]

    proto = dict(common.get("protocol") or {})
    old_frame = proto.get("coord_frame", "world")
    old_y_sign = None
    if "display_y_sign" in common:
        old_y_sign = np.asarray(common["display_y_sign"], dtype=np.float64)
    elif old_frame == "world_display_remapped":
        old_y_sign = np.array([-1.0 if sc in SWAP_NEGX else 1.0 for sc in scenes])

    print(json.dumps({"old_frame": old_frame, "n": n,
                      "scene_counts": {k: scenes.count(k) for k in sorted(set(scenes))}},
                     ensure_ascii=False))

    # Prefer RAW world for past/gt when available (authoritative)
    past_w = np.asarray(raw[ids, 0, :8, 4:6], dtype=np.float64)
    gt_w = np.asarray(raw[ids, 0, 8:, 4:6], dtype=np.float64)
    common["past"] = np.round(world_to_display(past_w, scenes), 5).tolist()
    common["gt"] = np.round(world_to_display(gt_w, scenes), 5).tolist()

    # neighbors: from common after undo, OR rebuild from raw channels if present
    # raw layout: agents 1..8 neighbors, past 8 frames
    nb_w = []
    num = None
    num_path = RAW.parent / "eth_num_train.npy"
    if num_path.exists():
        num = np.load(num_path, mmap_mode="r")
    for i, sid in enumerate(ids):
        # agents 1..8 in raw; valid count from num if available
        n_valid = int(num[sid]) if num is not None else 9
        n_nb = max(0, n_valid - 1)
        if n_nb <= 0:
            nb_w.append([])
            continue
        traj = np.asarray(raw[sid, 1:1+n_nb, :8, 4:6], dtype=np.float64)
        nb_w.append(traj)
    # If common neighbors length differs, fall back to undo path for that sample
    new_nb = []
    for i, traj in enumerate(nb_w):
        if traj is None or (isinstance(traj, list) and not traj):
            # fallback
            nb = common["neighbors"][i]
            if not nb:
                new_nb.append(nb)
                continue
            arr = np.asarray(nb, dtype=np.float64)
            w = to_world_from_current(arr[None], [scenes[i]], old_frame,
                                      None if old_y_sign is None else old_y_sign[i:i+1])[0]
            d = world_to_display(w[None], [scenes[i]])[0]
            new_nb.append(np.round(d, 5).tolist())
        else:
            d = world_to_display(np.asarray(traj)[None], [scenes[i]])[0]
            new_nb.append(np.round(d, 5).tolist())
    common["neighbors"] = new_nb

    if "origin" in common:
        common["origin"] = transform_nested_list(
            common["origin"], scenes, old_frame, old_y_sign).tolist()
    if "heading" in common:
        # heading is a direction: same linear map (from world)
        h = np.asarray(common["heading"], dtype=np.float64)
        hw = to_world_from_current(h, scenes, old_frame, old_y_sign)
        common["heading"] = np.round(world_to_display(hw, scenes), 5).tolist()

    common["scenes"] = scenes
    common["scene_ids"] = scene_ids.astype(int).tolist()
    # display_code: 0 identity, 1 = (y,-x)
    common["display_code"] = [1 if sc in SWAP_NEGX else 0 for sc in scenes]
    # keep display_y_sign for old viewer compat but set to +1 always (no longer used for geometry)
    common["display_y_sign"] = [1] * n

    proto["coord_frame"] = TARGET_FRAME
    proto["coord_note"] = (
        "Display meters aligned to video via H evidence. "
        "students003/zara01/zara02: (x,y) with +x right +y up. "
        "hotel/eth: (y,-x) because BIWI H maps +x→image-down, +y→image-right. "
        "Citations: ASTRA utils/visualizer.world_to_pixel; GUIDE-CoT eth5 H_*.txt; "
        "overlays hotel ped259 / students ped102."
    )
    proto["display_transform"] = {
        "students003": "(x, y)",
        "zara01": "(x, y)",
        "zara02": "(x, y)",
        "hotel": "(y, -x)",
        "eth": "(y, -x)",
    }
    proto["display_flip_y_scenes"] = []  # obsolete
    proto["scene_id_map"] = SCENE_BY_ID
    proto["h_sources"] = {
        "eth": "GUIDE-CoT data/eth5/eth/H.txt (EWAP BIWI)",
        "hotel": "GUIDE-CoT data/eth5/hotel/H.txt (EWAP BIWI)",
        "students003": "GUIDE-CoT llm_module/datasets/homography/students003_H.txt",
        "zara01": "GUIDE-CoT llm_module/datasets/homography/crowds_zara01_H.txt",
        "zara02": "GUIDE-CoT llm_module/datasets/homography/crowds_zara02_H.txt",
    }
    common["protocol"] = proto
    write_js(HERE / "data" / "common.js", "window.AUDIT_COMMON", common)

    # model paths
    models = [m["key"] for m in common["models"]]
    for key in models:
        path = HERE / "data" / f"{key}.js"
        payload = load_js(path, f'window.AUDIT_MODELS["{key}"]=')
        p = np.asarray(payload["p"], dtype=np.float64)
        assert p.shape[0] == n and p.shape[-1] == 2
        pw = to_world_from_current(p, scenes, old_frame, old_y_sign)
        payload["p"] = np.round(world_to_display(pw, scenes), 5).tolist()
        write_js(path, f'window.AUDIT_MODELS["{key}"]', payload)
        print("patched model", key, flush=True)

    npz_path = HERE / "data" / "arrays.npz"
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as z:
            data = {k: z[k].copy() for k in z.files}
        for key in ("past", "future", "origin", "heading", "neighbor_past"):
            if key not in data:
                continue
            arr = data[key].astype(np.float64)
            if arr.ndim < 1 or arr.shape[-1] != 2:
                continue
            if key == "neighbor_past":
                # (N, A, T, 2) — transform per sample
                out = arr.copy()
                for i, sc in enumerate(scenes):
                    w = to_world_from_current(arr[i:i+1], [sc], old_frame,
                                              None if old_y_sign is None else old_y_sign[i:i+1])
                    out[i] = world_to_display(w, [sc])[0]
                # Prefer raw rebuild for neighbor_past past-only if shape matches
                data[key] = out
            else:
                w = to_world_from_current(arr, scenes, old_frame, old_y_sign)
                data[key] = world_to_display(w, scenes)
        # overwrite past/future from raw for consistency
        data["past"] = world_to_display(past_w, scenes)
        data["future"] = world_to_display(gt_w, scenes)
        data["scene_id"] = scene_ids
        data["display_code"] = np.asarray(common["display_code"], dtype=np.int32)
        data["display_y_sign"] = np.ones(n, dtype=np.int32)
        # model arrays in npz
        for key in models:
            if key in data and data[key].ndim >= 2 and data[key].shape[-1] == 2:
                w = to_world_from_current(data[key].astype(np.float64), scenes, old_frame, old_y_sign)
                data[key] = world_to_display(w, scenes)
        np.savez_compressed(npz_path, **data)
        print("patched arrays.npz", flush=True)

    # Sanity: sample 740
    i = list(ids).index(740)
    print("SANITY 740 scene", scenes[i])
    print("  display past0", common["past"][i][0], "expected ~ (2.39, -3.46)")
    print("  display past-1", common["past"][i][-1], "expected ~ (-1.53, -3.45)")
    print("  display gt-1", common["gt"][i][-1], "expected ~ (-9.10, -2.85)")
    # students unchanged check
    for i2, sc in enumerate(scenes):
        if sc == "students003":
            sid = int(ids[i2])
            print("SANITY stu", sid, "disp0", common["past"][i2][0],
                  "raw", past_w[i2, 0].tolist())
            break
    print("DONE", TARGET_FRAME)


if __name__ == "__main__":
    main()
