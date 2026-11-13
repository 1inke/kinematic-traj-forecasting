from pathlib import Path
import re
import numpy as np
from PIL import Image, ImageDraw

vsp = Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\_univ_rar\students003.vsp").read_text(encoding="utf-8", errors="replace").splitlines()
splines=[]
i=1
while i < len(vsp):
    m=re.match(r"(\d+)\s+-\s+Num of control points", vsp[i])
    if not m:
        i+=1; continue
    n=int(m.group(1)); i+=1
    pts=[]
    for _ in range(n):
        nums=re.findall(r"-?\d+\.?\d*", vsp[i])
        x,y,fr,ang=float(nums[0]),float(nums[1]),float(nums[2]),float(nums[3])
        pts.append((fr,x,y,ang)); i+=1
    splines.append(pts)

# which splines cover frame 144?
cands=[]
for si,pts in enumerate(splines):
    frs=np.array([p[0] for p in pts])
    if frs.min()<=144<=frs.max():
        xs=np.array([p[1] for p in pts]); ys=np.array([p[2] for p in pts])
        x=float(np.interp(144, frs, xs)); y=float(np.interp(144, frs, ys))
        cands.append((si+1,x,y,frs.min(),frs.max()))
print("splines covering f144:", len(cands))
for c in cands[:30]:
    print(c)

# Also: match ped102 world (0.393,4.415) via H if we invent from many points
# Read all ped102 points from txt
rows=[]
for line in Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy\labels\univ\students003.txt").read_text(encoding="utf-8",errors="ignore").splitlines():
    a=line.split()
    if len(a)<16: continue
    if abs(float(a[1])-102)>1e-9: continue
    fr=float(a[0]); x=float(a[13]); y=float(a[15])
    rows.append((fr,x,y))
print("ped102 txt rows", len(rows), "first", rows[:3], "last", rows[-3:])

# Hypothesis: txt frame == vsp frame. Find spline whose interp at 144 is close in CENTERED pixel to... need H
# Try: maybe world coords equal somehow? Unlikely.
# Fit H from all people at frame 144: for each ped in txt at 144, find nearest vsp spline at 144

# Build map ped->world at 144
from collections import defaultdict
at={}
for line in Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy\labels\univ\students003.txt").read_text(encoding="utf-8",errors="ignore").splitlines():
    a=line.split()
    if len(a)<16: continue
    fr=float(a[0])
    if abs(fr-144)>1e-6: continue
    ped=int(float(a[1])); x=float(a[13]); y=float(a[15])
    at[ped]=(x,y)
print("n peds at f144 in txt", len(at), "has102", 102 in at, at.get(102))

# For each txt ped, find best matching vsp among those covering 144 by... we only have world, vsp has pixel-centered.
# Collect vsp positions at 144
vpos=[]
for si,pts in enumerate(splines):
    frs=np.array([p[0] for p in pts])
    if frs.min()<=144<=frs.max():
        xs=np.array([p[1] for p in pts]); ys=np.array([p[2] for p in pts])
        vpos.append((si+1, float(np.interp(144,frs,xs)), float(np.interp(144,frs,ys))))
print("vpos sample", vpos[:5], "n", len(vpos))

# Use DLT to estimate H: world->centered_pixel using ALL matched pairs if we match by order? 
# Better approach from literature: UCY H for students003
# Known matrix often used (pixel to world):
# Try fetching from raw.githubusercontent.com alternatives

# Match by ranking: sort both lists? No.
# Check if ped id == spline id: for ped 102, spline 102 position at frames that overlap
pts=splines[101]
print("spline102 frame range", pts[0][0], pts[-1][0])
# Maybe frame in txt is NOT video frame - eth_num said 7 for scene. 
# processed frames might be 0-indexed subsampled: video_frame = txt_frame * 10 or *20?
for scale in [1,10,20,25]:
    vf=144*scale
    pts=splines[101]
    frs=np.array([p[0] for p in pts])
    print("scale", scale, "vf", vf, "in range", frs.min()<=vf<=frs.max())
