from pathlib import Path
import re, numpy as np
from PIL import Image, ImageDraw

# ped 51 at pred end txt frame 161 -> vf 1610
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
W,H=720,576
vf=1610
x=float(np.interp(vf,frs,xs)); y=float(np.interp(vf,frs,ys)); u,v=x+W/2,y+H/2
print("pixel",u,v)

# mark on dataset frame_0161
img_dir=Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy\imgs\students003")
im=Image.open(img_dir/"frame_0161.jpg").convert("RGB")
d=ImageDraw.Draw(im); r=26
d.ellipse([u-r,v-r,u+r,v+r], outline=(255,0,0), width=5)
d.line([u-32,v,u+32,v], fill=(255,255,0), width=4); d.line([u,v-32,u,v+32], fill=(255,255,0), width=4)
d.rectangle([5,5,710,60], fill=(0,0,0))
d.text((12,12), f"24063 = ped 51 @ pred_end ~64.4s  pixel≈({u:.0f},{v:.0f})", fill=(255,255,0))
# trail last few
for t in range(149,162):
    vf_t=t*10
    xx=float(np.interp(vf_t,frs,xs)); yy=float(np.interp(vf_t,frs,ys))
    uu,vv=xx+W/2,yy+H/2
    d.ellipse([uu-3,vv-3,uu+3,vv+3], fill=(0,255,255))
out=Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\sample_24063_ped51_end.png")
im.save(out); print("wrote", out)
