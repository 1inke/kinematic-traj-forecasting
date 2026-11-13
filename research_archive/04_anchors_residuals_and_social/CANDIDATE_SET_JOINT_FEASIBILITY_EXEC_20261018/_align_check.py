import json, numpy as np
from pathlib import Path

# visual common.js for 24063
raw=(Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\visual_review_overnight\data\common.js").read_text(encoding="utf-8"))
i=raw.find("{"); data=json.loads(raw[i:].rstrip().rstrip(";"))
idx=data["ids"].index(24063)
past=np.array(data["past"][idx])
gt=np.array(data["gt"][idx])
print("past shape", past.shape, "first", past[0], "last", past[-1])
print("gt first", gt[0], "last", gt[-1])
print("past delta", past[-1]-past[0], "gt delta", gt[-1]-gt[0])

# eth_data world for 24063
rawn=np.load(r"<LOCAL_DESKTOP>\DRSE\ASTRA_ACCURACY_Q20\datasets\eth_ucy\processed_data_diverse\eth_data_train.npy", mmap_mode="r")
# find index by sample_id in eth_num? sample_id IS the row index in eth_data often
# metadata sample_id 24063 -> check
meta=np.load(r"<WORKSPACE_ROOT>\F16_OBSERVABLE_CUE_DIAGNOSIS_20261003\metadata.npz")
mi=int(np.where(meta["sample_id"]==24063)[0][0])
print("meta ped", int(meta["pedestrian_id"][mi]), "start", int(meta["start_frame"][mi]), "seq", meta["sequence"][mi])

arr=np.array(rawn[24063,0])  # ego (20,6)
print("ego frames", arr[:,1])
print("ego xy start", arr[0,4:6], "t0(8th)", arr[7,4:6], "end", arr[19,4:6])
print("world delta obs", arr[7,4:6]-arr[0,4:6], "fut", arr[19,4:6]-arr[7,4:6])

# vsp ped 51 pixel motion t0->end
import re
vsp=Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\_univ_rar\students003.vsp").read_text(encoding="utf-8",errors="replace").splitlines()
splines=[]; i=1
while i<len(vsp):
    m=re.match(r"(\d+)\s+-\s+Num of control points", vsp[i])
    if not m: i+=1; continue
    n=int(m.group(1)); i+=1; pts=[]
    for _ in range(n):
        nums=re.findall(r"-?\d+\.?\d*", vsp[i])
        pts.append((float(nums[2]), float(nums[0]), float(nums[1]))); i+=1
    splines.append(pts)
pts=splines[50]
frs=np.array([p[0] for p in pts]); xs=np.array([p[1] for p in pts]); ys=np.array([p[2] for p in pts])
for t in [149,155,161]:
    vf=t*10; x=float(np.interp(vf,frs,xs)); y=float(np.interp(vf,frs,ys))
    print(f"vsp ped51 t{t} centered=({x:.1f},{y:.1f}) pixel=({x+360:.1f},{y+288:.1f})")
print("vsp delta t149->161", 
      float(np.interp(1610,frs,xs)-np.interp(1490,frs,xs)),
      float(np.interp(1610,frs,ys)-np.interp(1490,frs,ys)))

# Does viz "沿历史朝向 x" mean +x is forward along history? past ends at 0,0 typically
# If past goes left-to-right on plot, video should move along that heading after rotation to local frame
