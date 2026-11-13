import json, numpy as np
from pathlib import Path
ROOT = Path(r"<WORKSPACE_ROOT>")
EXEC = ROOT / "CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018"
meta = np.load(ROOT/"F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz", allow_pickle=True)
ids = meta["sample_id"].astype(np.int64)
row = int(np.where(ids==24102)[0][0])
info = {
  "sample_id": 24102,
  "scene_family": str(meta["scene_family"][row]),
  "sequence": str(meta["sequence"][row]),
  "pedestrian_id": int(meta["pedestrian_id"][row]),
  "start_frame": int(meta["start_frame"][row]),
  "ambiguous_track_time": bool(meta["ambiguous_track_time"][row]),
  "matching_source_windows": int(meta["matching_source_windows"][row]),
  # F16/ETH-UCY processed pipeline uses dt=0.4s (2.5 Hz). Raw UCY frame indices are often already at that rate in traj txt; video itself is typically 25fps — if start_frame is raw video index, seconds = frame/25; if subsampled index, seconds = frame*0.4.
  "dt_protocol_s": 0.4,
  "time_if_subsampled_index_s": int(meta["start_frame"][row]) * 0.4,
  "time_if_raw_25fps_video_s": int(meta["start_frame"][row]) / 25.0,
  "visual_tool": {
    "url_hint": "http://127.0.0.1:8765/index.html?model=k1_S",
    "ids_index": 972,
    "dir": str(EXEC/"visual_review_overnight"),
  },
  "note": "Processed ETH/UCY train cache; sequence students003 (UCY university/students). No map raster in this pipeline. Overnight visual neighbors[] empty for all samples, but eth_train social[96] is nonzero (neighbors used as fused features, not exported as polylines)."
}
(EXEC/"_sample_24102_scene_quick.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
print(json.dumps(info, indent=2))

# Try find students003 txt quickly in common places only
cands = list((ROOT/"kinematic-traj-forecasting").glob("**/students003*"))
cands += list((ROOT/".cache").glob("**/students003*")) if (ROOT/".cache").exists() else []
cands += list(Path(r"<USER_HOME>").glob("**/students003_train.txt"))[:20] if False else []
# limited walk of offline-memory-bundles
omb = ROOT/"offline-memory-bundles"
if omb.exists():
  for p in omb.rglob("students003*"):
    cands.append(p)
    if len(cands)>30: break
print("found_files", [str(p) for p in cands[:20]])
