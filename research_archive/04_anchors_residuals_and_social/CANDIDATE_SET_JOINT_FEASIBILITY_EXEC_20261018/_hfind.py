import numpy as np
from pathlib import Path
astra = Path(r"<LOCAL_DESKTOP>/DRSE/ASTRA_ACCURACY_Q20")
for rel in ["datasets/eth_ucy/labels/univ/students003.txt","datasets/eth_ucy/labels/eth/students003_train.txt"]:
    p = astra/rel
    print("FILE", rel, "exists", p.exists())
    if not p.exists():
        continue
    # peek first line ncols
    with p.open() as f:
        line = f.readline().strip().split()
    print(" ncols", len(line), "first", line[:6], "...", line[12:20] if len(line)>18 else line[12:])
    table = np.loadtxt(p, dtype=str)
    frame = table[:,0].astype(float).astype(int)
    ped = table[:,1].astype(float).astype(int)
    m = (ped==102)&(frame==144)
    print(" hits", m.sum())
    if m.any():
        row = table[m][0]
        print(" full row", row)
        # try last columns as pixels
        nums = row.astype(float)
        print(" as float", nums)
