from pathlib import Path
import numpy as np, cv2, json

raw=np.load(r"<LOCAL_DESKTOP>\DRSE\ASTRA_ACCURACY_Q20\datasets\eth_ucy\processed_data_diverse\eth_data_train.npy", mmap_mode='r')
arr=np.array(raw[24102])  # (9,20,6)
ego=arr[0]  # (20,6)
print("ego frames", ego[:20,1])
print("ego xy")
for t in range(20):
    print(t, int(ego[t,1]), ego[t,4], ego[t,5])
# neighbors with valid xy
print("neighbors valid counts")
for i in range(1,9):
    xy=arr[i,:,4:6]
    valid=np.isfinite(xy).all(1) & (np.abs(xy+1).sum(1)>1e-3)
    print(i, int(valid.sum()), "first", xy[valid][0] if valid.any() else None)

# find H
roots=[Path(r"<LOCAL_DESKTOP>\DRSE")]
hs=[]
for r in roots:
  for p in r.rglob("*"):
    name=p.name.lower()
    if p.is_file() and (name in ("h.txt","homography.txt") or "homograph" in name):
      hs.append(p)
print("H files", len(hs))
for p in hs[:30]:
  print(p)
