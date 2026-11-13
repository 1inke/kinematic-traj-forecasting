import numpy as np, json, re
from pathlib import Path
ROOT = Path(r"<WORKSPACE_ROOT>")

# 1) decode sample_id distribution
z = np.load(ROOT/"REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz")
ids = z["sample_id"]
print("sample_id min/max/unique", ids.min(), ids.max(), len(np.unique(ids)))
print("24102 // 1000", 24102//1000, "mod", 24102%1000)
print("24102 // 10000", 24102//10000, "mod", 24102%10000)
# check if id encodes frame/ped like frame*100+ped or similar - look for docs
for p in [
  ROOT/"REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES",
  ROOT/"OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES",
  ROOT/"ETH_UCY_FULL_CROSSVAL_SERIES",
  ROOT/"GENERATION_NATIVE_DECODER_SERIES",
  ROOT/"F16_OBSERVABLE_CUE_DIAGNOSIS_20261003",
]:
  if not p.exists():
    continue
  for f in list(p.glob("**/*protocol*")) + list(p.glob("**/*README*")) + list(p.glob("**/*sample_id*")) + list(p.glob("**/*dataset*")):
    if f.is_file() and f.suffix.lower() in {".md",".json",".txt",".py"} and f.stat().st_size < 500_000:
      t = f.read_text(encoding="utf-8", errors="ignore")
      if "sample_id" in t or "scene" in t.lower() or "eth" in t.lower():
        if "24102" in t or "sample_id" in t[:2000] or "frame" in t.lower():
          print("DOC", f, "size", f.stat().st_size)

print("\n=== social reshape guesses for 24102 ===")
soc = z["social"][np.where(ids==24102)[0][0]]
print("social shape", soc.shape, "nonzero", np.count_nonzero(soc))
for n_nb, feat in [(4,24),(6,16),(8,12),(3,32),(12,8),(16,6)]:
  if soc.size == n_nb*feat:
    arr = soc.reshape(n_nb, feat)
    print(f"reshape ({n_nb},{feat}) row_norms", np.linalg.norm(arr,axis=1))
    print("  row0", arr[0,:8])
