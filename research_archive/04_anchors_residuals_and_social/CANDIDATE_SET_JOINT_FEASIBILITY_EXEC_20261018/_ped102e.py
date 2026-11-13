from pathlib import Path
import numpy as np

raw=np.load(r"<LOCAL_DESKTOP>\DRSE\ASTRA_ACCURACY_Q20\datasets\eth_ucy\processed_data_diverse\eth_data_train.npy", mmap_mode='r')
arr=np.array(raw[24102])
ego=arr[0]
print("ego t frame x y")
for t in range(20):
    print(t, int(ego[t,1]), float(ego[t,4]), float(ego[t,5]))
print("neighbors")
for i in range(1,9):
    xy=arr[i,:,4:6]
    valid=(np.abs(xy[:,0]+1)>1e-6) | (np.abs(xy[:,1]+1)>1e-6)
    print(i, "nvalid", int(valid.sum()), "t0", xy[0].tolist(), "t7", xy[7].tolist())

# H search limited dirs
dirs=[
 Path(r"<LOCAL_DESKTOP>\DRSE\crowd-data"),
 Path(r"<LOCAL_DESKTOP>\DRSE\eth_dataset"),
 Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy"),
]
for d in dirs:
  if not d.exists():
    continue
  for p in d.rglob("H.txt"):
    print("H", p)
    print(p.read_text()[:250])
  for p in d.rglob("*homograph*"):
    print("homo", p)
# check imgs count / size
img=Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy\imgs\students003\frame_0144.jpg")
print("frame144 exists", img.exists())
img0=Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy\imgs\students003\frame_0000.jpg")
import struct
# just list how many frames
n=len(list(Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy\imgs\students003").glob("frame_*.jpg")))
print("n_frames", n)
