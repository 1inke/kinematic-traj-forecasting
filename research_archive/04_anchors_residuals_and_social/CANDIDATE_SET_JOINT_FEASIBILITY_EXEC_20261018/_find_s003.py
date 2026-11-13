import json, numpy as np
from pathlib import Path
ROOT = Path(r"<WORKSPACE_ROOT>")
meta = np.load(ROOT/"F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz", allow_pickle=True)
i = int(np.where(meta["sample_id"]==24102)[0][0])
print("scene", meta["scene_family"][i], "seq", meta["sequence"][i], "ped", meta["pedestrian_id"][i], "start_frame", meta["start_frame"][i])
# find students003 files
cands = list(ROOT.rglob("*students003*"))
print("n_cands", len(cands))
for p in cands[:40]:
    print(p)
