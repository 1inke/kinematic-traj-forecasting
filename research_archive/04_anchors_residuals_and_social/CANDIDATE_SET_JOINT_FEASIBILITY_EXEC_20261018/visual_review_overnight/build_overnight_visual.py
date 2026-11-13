# -*- coding: utf-8 -*-
"""Build overnight joint-feasibility results into the candidate visual audit viewer.
Read-only over old MODE_PRESERVING / P1; writes only under visual_review_overnight.

Coordinate frame (2026-10-19): world meters + per-scene display remap.
  - Absolute world (x, y) from eth_data_*; NOT ego-centric heading frame.
  - Display: students003/zara keep (x,y); hotel/eth plot (x,-y) (H polarity: world +y ~ video down).
  - Not a universal "video-up" claim across all scenes. See COORD_FRAME.md.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, json, time
import numpy as np
from classification import diagnose, LABELS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # kinematic-traj-forecasting
EXEC = HERE.parent
SEED = 42
SPLIT = "outer"
RAW_DIR = Path(r"<LOCAL_DESKTOP>/DRSE/ASTRA_ACCURACY_Q20/datasets/eth_ucy/processed_data_diverse")

def heading_and_origin(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample history heading (unit) and last-obs world origin for agent 0.
    coords: (N, A, T, 2) world meters."""
    focal = coords[:, 0, :8]
    displacement = focal[:, -1] - focal[:, -4]
    speed = np.linalg.norm(displacement, axis=-1)
    heading = np.zeros_like(displacement)
    heading[:, 0] = 1
    active = speed > 0.03
    heading[active] = displacement[active] / speed[active, None]
    origin = focal[:, -1].astype(np.float32)
    return heading.astype(np.float32), origin


def local_coordinates(coords: np.ndarray) -> np.ndarray:
    """Ego-centric frame at last observed step (legacy training frame).
    Kept only to validate encoder past / invert candidates back to world."""
    heading, origin = heading_and_origin(coords)
    relative = coords - origin[:, None, None, :]
    cosine, sine = heading[:, 0, None, None], heading[:, 1, None, None]
    return np.stack(
        (relative[..., 0] * cosine + relative[..., 1] * sine,
         -relative[..., 0] * sine + relative[..., 1] * cosine),
        -1,
    ).astype(np.float32)


def local_to_world(local: np.ndarray, origin: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """Invert ego-centric local frame -> absolute world meters.
    local[...,0] along heading, local[...,1] left of heading.
    origin: (N, 2), heading: (N, 2) unit."""
    cos, sin = heading[:, 0], heading[:, 1]
    # Broadcast (N,) over trailing spatial dims of local
    shape = (local.shape[0],) + (1,) * (local.ndim - 2)
    cos = cos.reshape(shape)
    sin = sin.reshape(shape)
    ox = origin[:, 0].reshape(shape)
    oy = origin[:, 1].reshape(shape)
    xl, yl = local[..., 0], local[..., 1]
    dx = xl * cos - yl * sin
    dy = xl * sin + yl * cos
    return np.stack([dx + ox, dy + oy], axis=-1).astype(np.float32)


SCENE_BY_ID = {0: "eth", 1: "hotel", 2: "students003", 3: "zara01", 4: "zara02"}
# BIWI eth/hotel share H polarity (world +y ~ video down); UCY plaza scenes do not.
FLIP_Y_SCENES = {"eth", "hotel"}


def display_remap_y(coords: np.ndarray, y_sign: np.ndarray) -> np.ndarray:
    """Apply per-sample display y sign. coords (..., 2); y_sign (N,) with +/-1."""
    out = np.asarray(coords, dtype=np.float32).copy()
    sign = np.asarray(y_sign, dtype=np.float32)
    while sign.ndim < out.ndim:
        sign = sign.reshape(sign.shape + (1,))
    out[..., 1] = out[..., 1] * sign[..., 0]
    return out


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_js(path: Path, assignment: str, value) -> None:
    text = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item(),
    )
    path.write_text(assignment + "=" + text + ";\n", encoding="utf-8")

def gather_paths(pool: np.ndarray, indices: np.ndarray) -> np.ndarray:
    # pool (N,40,12,2), indices (N,20) -> (N,20,12,2)
    idx = indices.astype(np.int64)
    return np.take_along_axis(pool, idx[:, :, None, None], axis=1)

def gather_scores(scores40: np.ndarray, indices: np.ndarray) -> np.ndarray:
    idx = indices.astype(np.int64)
    return np.take_along_axis(scores40, idx, axis=1)

def main() -> None:
    started = time.time()
    (HERE / "data").mkdir(exist_ok=True)

    train_cache = ROOT / "REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz"
    labels_path = ROOT / "F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz"
    encoder_path = ROOT / "GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz"
    kin_path = ROOT / f"UNION_SELECTOR_K40_TO_K20_20261018/results/cache_pred/kin_{SPLIT}_seed{SEED}.npy"
    bundle_path = ROOT / f"COMPLEMENTARY_POOL_VALIDATION_20261018/artifacts/v1/{SPLIT}_seed{SEED}_bundle.npz"
    k1_dir = EXEC / f"k1_exec/seed{SEED}/predictions/{SPLIT}"
    k0c_idx_path = EXEC / f"results/seed{SEED}/{SPLIT}/k0c_indices.npy"
    deploy_dir = ROOT / "DEPLOY_S_RANK_MOTION_ONLY_20261019/indices"

    with np.load(train_cache, allow_pickle=False) as z:
        train_ids = z["sample_id"].copy()
        train_cand = z["candidate"].copy()
        train_logits = z["logits"].copy()
        train_group = z["change_group"].copy()
    with np.load(labels_path, allow_pickle=False) as z:
        lab_ids = z["sample_id"].copy()
        lab_future = z["future"].copy()
    with np.load(encoder_path, allow_pickle=False) as z:
        enc_ids = z["sample_id"].copy()
        enc_past = z["past"].copy()

    bundle = np.load(bundle_path, allow_pickle=False)
    sid = np.load(k1_dir / "sample_id.npy")
    groups = np.load(k1_dir / "groups.npy")
    I_idx = np.load(k1_dir / "I_indices.npy")
    S_idx = np.load(k1_dir / "S_indices.npy")
    k0c_idx = np.load(k0c_idx_path)
    kin = np.load(kin_path)
    assert np.array_equal(bundle["sample_id"], sid)
    n = len(sid)
    assert I_idx.shape == (n, 20) and S_idx.shape == (n, 20) and k0c_idx.shape == (n, 20)
    assert kin.shape == (n, 20, 12, 2)

    id_to_train = {int(i): r for r, i in enumerate(train_ids)}
    id_to_lab = {int(i): r for r, i in enumerate(lab_ids)}
    id_to_enc = {int(i): r for r, i in enumerate(enc_ids)}
    trows = np.array([id_to_train[int(i)] for i in sid])
    lrows = np.array([id_to_lab[int(i)] for i in sid])
    erows = np.array([id_to_enc[int(i)] for i in sid])

    native_paths = train_cand[trows, :, :24].reshape(n, 20, 12, 2).astype(np.float32)
    pool = np.concatenate([native_paths, kin.astype(np.float32)], axis=1)  # (N,40,12,2)
    scores40 = bundle["scores40"].astype(np.float64)
    past_local = enc_past[erows].astype(np.float32)
    future_local = lab_future[lrows].astype(np.float32)
    # Prefer train change_group; fall back to k1 groups
    change_group = train_group[trows].astype(np.int64)
    # Sanity: past vs encoder already aligned by construction
    past_chk = float(np.abs(past_local.reshape(n, 16) - np.load(train_cache)["motion"][trows, :16]).max())
    assert past_chk < 5e-5, past_chk

    # --- World frame: invert ego-local training coords to absolute meters ---
    raw_data = RAW_DIR / "eth_data_train.npy"
    raw_num = RAW_DIR / "eth_num_train.npy"
    raw = np.load(raw_data, mmap_mode="r")
    num = np.load(raw_num, mmap_mode="r")
    assert int(sid.min()) >= 0 and int(sid.max()) < len(raw)
    coordinates = np.asarray(raw[sid.astype(np.int64), :9, :20, -2:], dtype=np.float32)
    heading, origin = heading_and_origin(coordinates)
    local = local_coordinates(coordinates)
    past_from_raw = local[:, 0, :8]
    raw_past_chk = float(np.abs(past_from_raw - past_local).max())
    assert raw_past_chk < 5e-5, raw_past_chk
    # Round-trip check
    recon = local_to_world(local, origin, heading)
    assert float(np.abs(recon - coordinates).max()) < 1e-5

    # Absolute world trajectories for viz (ADE/FDE unchanged under rigid transform)
    past = local_to_world(past_local, origin, heading)
    future = local_to_world(future_local, origin, heading)
    native_paths_w = local_to_world(native_paths, origin, heading)
    pool_w = local_to_world(pool, origin, heading)
    # Neighbors: take world past directly from raw (agents 1..)
    valid = np.minimum(np.asarray(num[sid.astype(np.int64)]).reshape(n), 9).astype(np.int64)
    assert valid.min() >= 1 and valid.max() <= 9
    neighbors = [
        np.round(coordinates[i, 1:int(valid[i]), :8].astype(float), 5).tolist()
        for i in range(n)
    ]
    # Target past from raw must match inverted encoder past
    world_past_chk = float(np.abs(past - coordinates[:, 0, :8]).max())
    assert world_past_chk < 5e-5, world_past_chk

    # Per-sample scene tag (eth_data ch0) + display remap for viz only
    scene_ids = np.asarray(raw[sid.astype(np.int64), 0, 0, 0], dtype=np.int64)
    scenes = [SCENE_BY_ID.get(int(s), f"unknown_{int(s)}") for s in scene_ids]
    y_sign = np.array([-1.0 if sc in FLIP_Y_SCENES else 1.0 for sc in scenes], dtype=np.float32)
    past = display_remap_y(past, y_sign)
    future = display_remap_y(future, y_sign)
    native_paths_w = display_remap_y(native_paths_w, y_sign)
    pool_w = display_remap_y(pool_w, y_sign)
    origin = display_remap_y(origin, y_sign)
    heading = display_remap_y(heading, y_sign)
    neighbors = [
        np.round(display_remap_y(coordinates[i, 1:int(valid[i]), :8], y_sign[i:i+1]).astype(float), 5).tolist()
        for i in range(n)
    ]

    fut_chk_ade = float(np.abs(
        np.linalg.norm(native_paths.astype(np.float64) - future_local[:, None], axis=-1).mean(-1)
        - np.load(train_cache)["ade"][trows]
    ).max())
    assert fut_chk_ade < 2e-5, fut_chk_ade

    # Deploy arms (same outer N=1035)
    deploy_A = np.load(deploy_dir / f"seed{SEED}_A_S_indices.npy")
    deploy_B = np.load(deploy_dir / f"seed{SEED}_B_S_motion_indices.npy")
    assert deploy_A.shape == (n, 20) and deploy_B.shape == (n, 20)

    models = [
        ("native", "原生 K20 · logits argmax", native_paths_w,
         train_logits[trows].astype(np.float64),
         "native_logits_argmax"),
        ("v1", "V1 并集选组 · 继承 scores40", gather_paths(pool_w, bundle["selected_pool_indices"]),
         gather_scores(scores40, bundle["selected_pool_indices"]), "portfolio_scores40_argmax"),
        ("k1_I", "K1-I 学习选择器 · outer", gather_paths(pool_w, I_idx), gather_scores(scores40, I_idx),
         "portfolio_scores40_argmax"),
        ("k1_S", "K1-S 学习选择器 · outer（过夜主线）", gather_paths(pool_w, S_idx), gather_scores(scores40, S_idx),
         "portfolio_scores40_argmax"),
        ("k0c", "K0-C 池可达 oracle · 事后参考", gather_paths(pool_w, k0c_idx), gather_scores(scores40, k0c_idx),
         "portfolio_scores40_argmax_NOT_oracle_pick"),
        ("deploy_A", "DEPLOY A · K1-S 锁门", gather_paths(pool_w, deploy_A), gather_scores(scores40, deploy_A),
         "portfolio_scores40_argmax"),
        ("deploy_B", "DEPLOY B · 去 pred_gate + S 运动学", gather_paths(pool_w, deploy_B), gather_scores(scores40, deploy_B),
         "portfolio_scores40_argmax"),
    ]


    protocol = {
        "purpose": "Overnight joint-feasibility selector visual review (K1 I/S, K0-C, DEPLOY)",
        "seed": SEED,
        "split": SPLIT,
        "n": int(n),
        "K": 20,
        "pool": "native20+kin20=K40",
        "diagnostic_ade_threshold_m": 0.3,
        "diagnostic_fde_threshold_m": 0.5,
        "timezone_label": "Asia/Tokyo",
        "coord_frame": "world_display_remapped",
        "coord_note": "展示坐标：世界米制按场景重映射。students/zara 画(x,y)；hotel/eth 画(x,-y)。非全场景通用「视频同向上」。",
        "display_flip_y_scenes": sorted(FLIP_Y_SCENES),
        "scene_id_map": SCENE_BY_ID,
        "note": "橙线=组内 scores40/logits 最高分槽；K0-C 为池可达参考不是部署选择。不改写旧 P1/MODE_PRESERVING。",
    }

    metadata = []
    precise = {}
    for key, label, paths, scores, sem in models:
        print("Building", key, flush=True)
        assert paths.shape == (n, 20, 12, 2) and scores.shape == (n, 20)
        assert np.isfinite(paths).all() and np.isfinite(scores).all()
        # diagnose in world frame (distances identical to local)
        d, ade, fde = diagnose(paths, future, scores)
        precise[key] = paths
        model_rows = []
        for i in range(n):
            item = dict(
                model=key, sample_id=int(sid[i]), row_index=i,
                group=["stable", "transition", "change"][int(change_group[i])],
                class_label=LABELS[int(d["category"][i])],
                selection_semantics=sem,
            )
            item.update({field: value[i].item() for field, value in d.items()})
            item.update(ade_limit_m=0.3, fde_limit_m=0.5)
            model_rows.append(item)
        metadata.append(dict(
            key=key, label=label,
            selection="组内最高分（非 GT）" if "oracle" not in sem.lower() or "NOT_oracle" in sem else "组内 scores40；集合本身为 oracle 可达",
            selected_ade=float(d["ade_selected"].mean()),
            selected_fde=float(d["fde_selected"].mean()),
            minade=float(d["minade"].mean()),
            minfde=float(d["minfde"].mean()),
            counts=[int((d["category"] == j).sum()) for j in range(6)],
            exact_ade_hits=int(d["selected_exact_ade_slot"].sum()),
            exact_fde_hits=int(d["selected_exact_fde_slot"].sum()),
            oracle_disagreement=int((~d["oracle_same_slot"]).sum()),
        ))
        write_js(HERE / "data" / f"{key}.js", f'window.AUDIT_MODELS["{key}"]', dict(
            p=np.round(paths.astype(np.float64), 5), a=ade, f=fde, d=model_rows,
        ))

    sources = [train_cache, labels_path, encoder_path, kin_path, bundle_path, raw_data, raw_num,
               k1_dir / "I_indices.npy", k1_dir / "S_indices.npy", k0c_idx_path,
               deploy_dir / f"seed{SEED}_A_S_indices.npy",
               deploy_dir / f"seed{SEED}_B_S_motion_indices.npy"]
    input_hashes = {str(p): sha(p) for p in sources}
    common = dict(
        ids=sid, past=np.round(past.astype(float), 5), gt=np.round(future.astype(float), 5),
        neighbors=neighbors,
        groups=change_group, scores=train_logits[trows].astype(np.float64),
        models=metadata, labels=LABELS,
        data_sha=input_hashes[str(k1_dir / "S_indices.npy")],
        protocol=protocol,
        origin=np.round(origin.astype(float), 5),
        heading=np.round(heading.astype(float), 5),
        scenes=scenes,
        scene_ids=scene_ids.astype(int).tolist(),
        display_y_sign=y_sign.astype(int).tolist(),
    )
    write_js(HERE / "data" / "common.js", "window.AUDIT_COMMON", common)
    np.savez_compressed(
        HERE / "data" / "arrays.npz",
        sample_id=sid, past=past, future=future, logits=train_logits[trows],
        change_group=change_group,
        neighbor_past=display_remap_y(coordinates[:, 1:, :8], y_sign),
        num_valid=valid,
        origin=origin, heading=heading,
        scene_id=scene_ids, display_y_sign=y_sign.astype(np.int8),
        **precise,
    )

    # Update index.html script tags + title
    index = (HERE / "index.html").read_text(encoding="utf-8")
    scripts = ['<script>window.AUDIT_MODELS={};</script>',
               '<script src="data/common.js"></script>']
    for key, *_ in models:
        scripts.append(f'<script src="data/{key}.js"></script>')
    scripts.append('<script src="viewer.js"></script>')
    import re
    index2 = re.sub(
        r'<script>window\.AUDIT_MODELS=\{\};</script>.*?</script>\s*</body>',
        "\n".join(scripts) + "\n</body>",
        index, count=1, flags=re.S,
    )
    index2 = index2.replace(
        "候选 → 评分 → 选择：人工可视化诊断",
        "过夜联合可行性 · K1/K0-C/DEPLOY 可视化",
    )
    # Soften footer claim about 1516
    index2 = index2.replace("全1516", f"全{n}")
    (HERE / "index.html").write_text(index2, encoding="utf-8")

    audit = dict(
        status="PASS_OVERNIGHT_VISUAL_ADAPTER",
        seed=SEED, split=SPLIT, n=n, K=20, models=[m[0] for m in models],
        past_alignment_max_error=past_chk, raw_past_alignment_max_error=raw_past_chk,
        world_past_alignment_max_error=world_past_chk,
        native_ade_recompute_max_error=fut_chk_ade,
        coord_frame="world_video_aligned",
        neighbor_nonempty=int(sum(1 for x in neighbors if x)),
        neighbor_mean_count=float(np.mean([len(x) for x in neighbors])),
        sources=input_hashes, seconds=time.time() - started,
        timezone_label="Asia/Tokyo",
        built_at=time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
    )
    (HERE / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "START_HERE.md").write_text(
        "\n".join([
            "# 过夜结果可视化（接入候选可视化诊断工具）",
            "",
            f"- 样本：ETH outer seed{SEED}，N={n}，每例 K20（从 native+KIN 的 K40 池选出）",
            "- 模型切换：native / v1 / k1_I / k1_S / k0c / deploy_A / deploy_B",
            "- 橙线=组内最高分槽（scores40 或原生 logits），不是 GT 选择",
            "- k0c 集合本身是池可达 oracle 参考；deploy_B 是去 pred_gate 的调整变体",
            "- **坐标系**：世界米制 + 按场景显示重映射（students/zara (x,y)；hotel/eth (x,-y)）。详见 COORD_FRAME.md",
            "",
            "## 打开方式",
            "",
            "1. 双击运行 `START_SERVER.bat`，或在本目录执行：",
            "   `python -m http.server 8765`",
            "2. 浏览器打开：http://127.0.0.1:8765/index.html",
            "3. 建议先看 model=`k1_S`，再对比 `k0c` 与 `deploy_B`",
            "",
            "数据由 `build_overnight_visual.py` 从 k1_exec / results / DEPLOY 索引生成；不改写旧 P1。",
            "",
        ]),
        encoding="utf-8",
    )
    (HERE / "START_SERVER.bat").write_text(
        "@echo off\r\ncd /d \"%~dp0\"\r\necho Open http://127.0.0.1:8765/index.html\r\npython -m http.server 8765\r\n",
        encoding="ascii",
    )
    print(json.dumps({"status": audit["status"], "n": n, "models": audit["models"],
                      "coord_frame": "world_display_remapped",
                      "seconds": round(audit["seconds"], 2)}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
