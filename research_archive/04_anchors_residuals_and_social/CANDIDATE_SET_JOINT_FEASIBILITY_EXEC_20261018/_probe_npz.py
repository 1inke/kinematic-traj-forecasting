import numpy as np
from pathlib import Path
ROOT = Path(r"<WORKSPACE_ROOT>")
paths = {
  "eth_train": ROOT/"REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz",
  "labels": ROOT/"F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz",
  "encoder": ROOT/"GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz",
}
for name,p in paths.items():
    print(f"\n=== {name} exists={p.exists()} ===")
    if not p.exists():
        continue
    z = np.load(p, allow_pickle=True)
    print("keys:", z.files)
    # find 24102
    sid = None
    for k in z.files:
        if "id" in k.lower() or k in ("sample_id","ids"):
            arr = z[k]
            print(f"  {k}: shape={arr.shape} dtype={arr.dtype}")
            if arr.dtype.kind in "iu" or (arr.dtype.kind=="f"):
                hit = np.where(arr.astype(np.int64)==24102)[0]
                print(f"    hits 24102: {hit[:5]} count={len(hit)}")
                if len(hit):
                    sid = int(hit[0])
    if sid is not None:
        for k in z.files:
            a = z[k]
            if hasattr(a, "shape") and len(a.shape)>=1 and a.shape[0] > sid:
                item = a[sid]
                if hasattr(item, "shape"):
                    print(f"  row[{sid}].{k}: shape={item.shape} dtype={item.dtype}")
                    if item.size <= 20:
                        print(f"    values={item}")
                    elif item.dtype.kind in "iufc":
                        flat = item.reshape(-1)[:8]
                        print(f"    head={flat}")
                else:
                    print(f"  row[{sid}].{k}: {item}")
            elif k.lower() in ("scene","scenes","meta","frame","frames","ped","ped_id","file","files","H","homography"):
                print(f"  {k}: {a}")
