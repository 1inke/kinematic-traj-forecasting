from pathlib import Path
import re, numpy as np
from PIL import Image, ImageDraw

# verify motion of spline 102 in pixel vs world ped102
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
        pts.append((fr,x,y)); i+=1
    splines.append(pts)

pts=splines[101]
frs=np.array([p[0] for p in pts],float)
xs=np.array([p[1] for p in pts],float)
ys=np.array([p[2] for p in pts],float)
print("spline102 control points:")
for p in pts:
    print(p)

# world ped102
world=[]
for line in Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy\labels\univ\students003.txt").read_text(encoding="utf-8",errors="ignore").splitlines():
    a=line.split()
    if len(a)>=16 and abs(float(a[1])-102)<1e-9:
        world.append((float(a[0]), float(a[13]), float(a[15])))
print("world first/last", world[0], world[-1])

# Compare deltas over overlapping scaled frames
for t0,t1 in [(144,151),(144,163),(150,160)]:
    vf0,vf1=t0*10,t1*10
    if vf0<frs.min() or vf1>frs.max():
        print("skip", t0,t1); continue
    dxp=float(np.interp(vf1,frs,xs)-np.interp(vf0,frs,xs))
    dyp=float(np.interp(vf1,frs,ys)-np.interp(vf0,frs,ys))
    # world
    w0=[w for w in world if abs(w[0]-t0)<1e-6][0]
    w1=[w for w in world if abs(w[0]-t1)<1e-6][0]
    print(f"txt {t0}->{t1}: world d=({w1[1]-w0[1]:.2f},{w1[2]-w0[2]:.2f}) pixel_c d=({dxp:.1f},{dyp:.1f})")

# Video duration via file / ffprobe if any
import subprocess, shutil
avi=r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy\videos\students003.avi"
ff=shutil.which("ffprobe") or r"C:\ffmpeg\bin\ffprobe.exe"
if Path(ff).exists() if ff else False:
    r=subprocess.run([ff,"-v","error","-show_entries","format=duration","-of","default=nw=1:nk=1",avi],capture_output=True,text=True)
    print("duration", r.stdout.strip())
else:
    print("no ffprobe, avi size", Path(avi).stat().st_size)
