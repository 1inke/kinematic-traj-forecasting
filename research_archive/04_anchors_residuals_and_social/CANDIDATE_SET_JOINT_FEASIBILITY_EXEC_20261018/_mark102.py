from pathlib import Path
import re
from PIL import Image, ImageDraw, ImageFont
import numpy as np

vsp = Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\_univ_rar\students003.vsp").read_text(encoding="utf-8", errors="replace").splitlines()
# parse splines
splines=[]
i=0
assert "number of splines" in vsp[0]
i=1
while i < len(vsp):
    m=re.match(r"(\d+)\s+-\s+Num of control points", vsp[i])
    if not m:
        i+=1; continue
    n=int(m.group(1)); i+=1
    pts=[]
    for _ in range(n):
        # x y frame angle
        nums=re.findall(r"-?\d+\.?\d*", vsp[i])
        x,y,fr,ang=float(nums[0]),float(nums[1]),float(nums[2]),float(nums[3])
        pts.append((fr,x,y,ang)); i+=1
    splines.append(pts)
print("n_splines", len(splines))
# ped id 102 -> 0-based index 101 if 1-based ids
for ped_idx in [101, 102, 100]:
    if ped_idx >= len(splines):
        continue
    pts=splines[ped_idx]
    frs=[p[0] for p in pts]
    print("spline", ped_idx+1, "frames", frs[0], "..", frs[-1], "n", len(pts))
    # interpolate at frame 144 and 125 (5s*25)
    for target in [125, 144, 151]:
        # linear interp in frame
        frs=np.array([p[0] for p in pts], float)
        xs=np.array([p[1] for p in pts], float)
        ys=np.array([p[2] for p in pts], float)
        if target < frs.min() or target > frs.max():
            print("  t", target, "OOR")
            continue
        x=float(np.interp(target, frs, xs)); y=float(np.interp(target, frs, ys))
        print("  t", target, "xy_centered", x, y)

# Convert centered coords to pixel: image 720x576, origin center -> u=x+360, v=y+288
W,H=720,576
def to_pix(x,y):
    return x+W/2, y+H/2

# Use spline 102 (1-based) = index 101
pts=splines[101]
frs=np.array([p[0] for p in pts], float)
xs=np.array([p[1] for p in pts], float)
ys=np.array([p[2] for p in pts], float)
# load frame 144 and also user cares about ~5s = frame 125
out_dir=Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018")
img_dir=Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy\imgs\students003")
for target in [125, 144]:
    x=float(np.interp(target, frs, xs)); y=float(np.interp(target, frs, ys))
    u,v=to_pix(x,y)
    print("mark frame", target, "pixel", u, v)
    impath=img_dir/f"frame_{target:04d}.jpg"
    im=Image.open(impath).convert("RGB")
    draw=ImageDraw.Draw(im)
    r=18
    draw.ellipse([u-r,v-r,u+r,v+r], outline=(255,0,0), width=4)
    draw.line([u-25,v,u+25,v], fill=(255,0,0), width=3)
    draw.line([u,v-25,u,v+25], fill=(255,0,0), width=3)
    draw.text((u+22,v-22), f"ped102 / id24102 @f{target}", fill=(255,255,0))
    # also draw neighbors from eth_data at this frame if available
    out=out_dir/f"sample_24102_mark_f{target}.png"
    im.save(out)
    print("wrote", out)

# sanity: check world motion direction vs pixel motion for ped102
print("traj centered first/last", (xs[0],ys[0]), (xs[-1],ys[-1]))
