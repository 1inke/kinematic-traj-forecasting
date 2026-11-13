from pathlib import Path
import re, numpy as np
from PIL import Image, ImageDraw

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

def pix(ped, txt_f):
    pts=splines[ped-1]
    frs=np.array([p[0] for p in pts]); xs=np.array([p[1] for p in pts]); ys=np.array([p[2] for p in pts])
    vf=txt_f*10
    x=float(np.interp(vf,frs,xs)); y=float(np.interp(vf,frs,ys))
    return x+360, y+288, frs.min()<=vf<=frs.max()

# at ~64s = txt frame ~160
for name,ped,tf in [("24102 ped102",102,151),("24102 ped102 end",102,163),("24063 ped51 t0",51,149),("24063 ped51 end",51,161)]:
    u,v,ok=pix(ped,tf)
    print(name, "f", tf, "pixel", round(u,1), round(v,1), "ok", ok)

# side-by-side mark both on frame 161
im=Image.open(r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy\imgs\students003\frame_0161.jpg").convert("RGB")
d=ImageDraw.Draw(im)
for ped,col,lab,tf in [(102,(255,0,0),"24102/102",163),(51,(0,255,0),"24063/51",161)]:
    u,v,_=pix(ped,tf if ped==51 else 161)  # both at f161 for same frame
    u,v,_=pix(ped,161)
    r=24
    d.ellipse([u-r,v-r,u+r,v+r], outline=col, width=5)
    d.text((u+20,v-20), lab, fill=col)
out=r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\compare_24102_24063_f161.png"
im.save(out); print("wrote", out)
# positions at same frame 161
print("same f161:", "102", pix(102,161), "51", pix(51,161))
