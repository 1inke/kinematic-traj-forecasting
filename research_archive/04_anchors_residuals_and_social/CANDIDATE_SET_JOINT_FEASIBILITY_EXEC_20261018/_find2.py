from pathlib import Path
import numpy as np

# classic train file used by F16 metadata
cands = list(Path(r"<LOCAL_DESKTOP>\DRSE").rglob("*students003*"))
print("cands", len(cands))
for p in cands:
    print(p, p.stat().st_size)

# also eth_num / eth_data for sample 24102
ASTRA = Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA_ACCURACY_Q20")
ALT = Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA")
for base in [ASTRA, ALT]:
    raw = base/"datasets/eth_ucy/processed_data_diverse/eth_data_train.npy"
    num = base/"datasets/eth_ucy/processed_data_diverse/eth_num_train.npy"
    print(base.name, "raw", raw.exists(), "num", num.exists())
