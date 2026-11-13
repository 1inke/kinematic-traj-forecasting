import numpy as np, json
from pathlib import Path
ROOT = Path(r"<WORKSPACE_ROOT>")
meta = np.load(ROOT/"F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz", allow_pickle=True)
print("metadata keys:", meta.files)
for k in meta.files:
    a = meta[k]
    print(f"  {k}: shape={getattr(a,'shape',None)} dtype={getattr(a,'dtype',None)}")
    if getattr(a, 'shape', None) == () or (hasattr(a,'size') and a.size<=5):
        print("   val", a)
ids = meta["sample_id"] if "sample_id" in meta.files else None
# find 24102
for k in meta.files:
    a = meta[k]
    if hasattr(a, 'dtype') and a.dtype.kind in 'iu' and a.ndim==1:
        hits = np.where(a.astype(np.int64)==24102)[0]
        if len(hits):
            print(f"hits in {k}: {hits}")
row = None
if ids is not None:
    hits = np.where(ids.astype(np.int64)==24102)[0]
    print("sample_id hits", hits)
    row = int(hits[0]) if len(hits) else None
elif "sample_id" not in meta.files:
    # maybe aligned by order with train
    print("no sample_id key; trying index from eth_train")
    train = np.load(ROOT/"REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz")
    row = int(np.where(train["sample_id"]==24102)[0][0])
    print("train row", row, "meta len", len(meta[meta.files[0]]))

if row is not None:
    print("\n=== row for 24102 ===")
    out = {}
    for k in meta.files:
        a = meta[k]
        if hasattr(a, 'shape') and len(a.shape)>=1 and a.shape[0] > row:
            v = a[row]
            if isinstance(v, np.ndarray):
                if v.dtype.kind in ('U','S','O') or v.size<=16:
                    out[k] = v.tolist() if hasattr(v,'tolist') else str(v)
                else:
                    out[k] = f"array{v.shape}"
            else:
                out[k] = v.item() if hasattr(v,'item') else (v.decode() if isinstance(v,bytes) else v)
            print(f"  {k} = {out[k]}")
    Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\_sample_24102_meta_scene.json").write_text(json.dumps(out, indent=2, default=str), encoding='utf-8')

audit = json.loads((ROOT/"F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata_audit.json").read_text(encoding='utf-8'))
print("\nmetadata_audit keys", list(audit.keys())[:30])
print(json.dumps({k:audit[k] for k in list(audit)[:15]}, indent=2, default=str)[:2000])
