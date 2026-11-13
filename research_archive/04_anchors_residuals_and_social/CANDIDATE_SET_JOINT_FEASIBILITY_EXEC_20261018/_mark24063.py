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
pts=splines[50]  # ped 51
frs=np.array([p[0] for p in pts]); xs=np.array([p[1] for p in pts]); ys=np.array([p[2] for p in pts])
W,H=720,576
img_dir=Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy\imgs\students003")
out=Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\sample_24063_ped51_t0.png")
t0_txt=149
vf=t0_txt*10
x=float(np.interp(vf,frs,xs)); y=float(np.interp(vf,frs,ys)); u,v=x+W/2,y+H/2
print("pixel",u,v,"frange",frs.min(),frs.max())
im=Image.open(img_dir/f"frame_{t0_txt:04d}.jpg").convert("RGB")
d=ImageDraw.Draw(im); r=22
d.ellipse([u-r,v-r,u+r,v+r], outline=(255,0,0), width=5)
d.line([u-28,v,u+28,v], fill=(255,255,0), width=3); d.line([u,v-28,u,v+28], fill=(255,255,0), width=3)
d.rectangle([5,5,710,55], fill=(0,0,0))
d.text((12,12), f"24063 ped51 | predict start ~59.6s (f149)  end ~64.4s", fill=(255,255,0))
im.save(out); print("wrote", out)
