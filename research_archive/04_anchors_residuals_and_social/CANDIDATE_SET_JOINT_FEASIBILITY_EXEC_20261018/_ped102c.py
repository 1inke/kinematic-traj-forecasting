from pathlib import Path
import numpy as np, re

# Parse TrajNet-style: find lines where ped==102
p = Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy\labels\univ\students003.txt")
lines=p.read_text(encoding="utf-8",errors="ignore").splitlines()
# find ped 102
hits=[]
for i,line in enumerate(lines):
    a=line.split()
    if len(a)<4: continue
    if abs(float(a[1])-102)<1e-9:
        hits.append((i,line[:200], [float(x) for x in a if re.match(r'^-?\d',x)]))
print("hits", len(hits))
if hits:
    i,s,nums=hits[0]
    print("line0 nums len", len(nums), "first20", nums[:20])
    print("line0", s)
    # non -1 coords
    vals=[(j,v) for j,v in enumerate(nums) if abs(v+1)>1e-6]
    print("nonzero-ish", vals[:40])

# eth_num / eth_data for id 24102
num=np.load(r"<LOCAL_DESKTOP>\DRSE\ASTRA_ACCURACY_Q20\datasets\eth_ucy\processed_data_diverse\eth_num_train.npy")
print("eth_num shape", num.shape, num.dtype)
print("eth_num[24102]", num[24102] if num.ndim==1 and len(num)>24102 else "n/a")
# maybe sample_id indexing
raw=np.load(r"<LOCAL_DESKTOP>\DRSE\ASTRA_ACCURACY_Q20\datasets\eth_ucy\processed_data_diverse\eth_data_train.npy", mmap_mode='r')
print("eth_data shape", raw.shape, raw.dtype)
print("raw[24102,0] shape-ish", np.array(raw[24102]).shape)
# print last dims which are often xy
arr=np.array(raw[24102])
print(arr.shape)
# guess layout
print("slice sample", arr.reshape(-1)[:40])
