from pathlib import Path
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

W,H=720,576
pts=splines[101]  # ped 102
frs=np.array([p[0] for p in pts], float)
xs=np.array([p[1] for p in pts], float)
ys=np.array([p[2] for p in pts], float)

out_dir=Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018")
img_dir=Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy\imgs\students003")

# txt frame 144 -> video frame 1440; but extracted jpgs might be 1-per-txt-frame indexed as frame_0144 = video?
# Check: n_frames=541. Video length?
# If jpgs are named by txt frame index, use frame_0144.jpg with coords from vsp at vf=1440
# If jpgs are every video frame, use frame_1440 - but only 541 frames exist so jpgs are subsampled

# List some frame names max
frames=sorted(img_dir.glob("frame_*.jpg"))
print("n jpg", len(frames), "first", frames[0].name, "last", frames[-1].name)

# Time: user at 5s. Video 25fps -> video frame 125. txt frame = 125/10 = 12.5?
# start_frame 144 -> video 1440 -> time 1440/25 = 57.6s !!! 
# That contradicts earlier "5 second" advice.

# Re-read: maybe students003 processed frames ARE video frames already (not *10)
# But spline 102 starts at 1439 - so ped 102 appears late in video (~57s)
# User screenshot at 5s is WRONG time for this sample!

print("IMPORTANT: ped102 appears video frames", frs.min(), frs.max(), "time_s", frs.min()/25, frs.max()/25)

# For marking: use jpg indexing. What is frame_0144?
# Compare: at txt frame 144, world moves. At vsp 1440 for spline 102.
# Are imgs indexed by txt frame (0..540) or video frame?
# 541 frames - if video ~540*something. students003.avi length?

# If imgs are every 10th video frame: frame_0144.jpg = video frame 1440. Perfect.
# Confirm by checking frame_0000 corresponds to start.

target_txt=144
vf=target_txt*10  # 1440
x=float(np.interp(vf, frs, xs)); y=float(np.interp(vf, frs, ys))
u,v = x+W/2, y+H/2
print("at txt144/vf1440 centered", x,y, "pixel", u,v)

im=Image.open(img_dir/f"frame_{target_txt:04d}.jpg").convert("RGB")
draw=ImageDraw.Draw(im)
r=22
draw.ellipse([u-r,v-r,u+r,v+r], outline=(255,0,0), width=5)
draw.line([u-30,v,u+30,v], fill=(255,255,0), width=3)
draw.line([u,v-30,u,v+30], fill=(255,255,0), width=3)
draw.rectangle([5,5,710,70], fill=(0,0,0))
draw.text((12,12), f"ped 102 / sample 24102  |  students003 frame_0144 (~{vf/25:.1f}s)", fill=(255,255,0))
draw.text((12,40), f"pixel≈({u:.0f},{v:.0f})  obs start; prediction ~frame_0151", fill=(255,200,0))
# draw short history trail in image
for t_txt in range(144,152):
    vf_t=t_txt*10
    if vf_t<frs.min() or vf_t>frs.max():
        continue
    xx=float(np.interp(vf_t, frs, xs)); yy=float(np.interp(vf_t, frs, ys))
    uu,vv=xx+W/2, yy+H/2
    draw.ellipse([uu-3,vv-3,uu+3,vv+3], fill=(0,255,255))
out=out_dir/"sample_24102_ped102_marked.png"
im.save(out)
print("wrote", out, "size", out.stat().st_size)

# Also mark at 5s for user clarity that it's NOT this person
print("at 5s video frame 125: ped102 present?", frs.min()<=125<=frs.max())
