from pathlib import Path
import re, numpy as np
from PIL import Image, ImageDraw

# world tracks for ped 51 and 102
def load_ped(ped):
    rows=[]
    for line in Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy\labels\univ\students003.txt").read_text(encoding="utf-8",errors="ignore").splitlines():
        a=line.split()
        if len(a)<16: continue
        if int(float(a[1]))!=ped: continue
        rows.append((float(a[0]), float(a[13]), float(a[15])))
    return np.array(rows)

# vsp
vsp=Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\_univ_rar\students003.vsp").read_text(encoding="utf-8",errors="replace").splitlines()
splines=[]; i=1
while i<len(vsp):
    m=re.match(r"(\d+)\s+-\s+Num of control points", vsp[i])
    if not m: i+=1; continue
    n=int(m.group(1)); i+=1; pts=[]
    for _ in range(n):
        nums=re.findall(r"-?\d+\.?\d*", vsp[i])
        pts.append((float(nums[2]), float(nums[0]), float(nums[1]))); i+=1
    splines.append(np.array(pts))

def sample_spline(si, frames):
    pts=splines[si]
    frs,xs,ys=pts[:,0],pts[:,1],pts[:,2]
    out=[]
    for f in frames:
        vf=f*10
        if vf<frs.min() or vf>frs.max():
            return None
        out.append([float(np.interp(vf,frs,xs)), float(np.interp(vf,frs,ys))])
    return np.array(out)

def match_ped(ped):
    W=load_ped(ped)
    # use frames overlapping prediction window area
    frames=np.arange(max(140,int(W[0,0])), min(170,int(W[-1,0]))+1)
    Ww=np.array([[r[1],r[2]] for r in W if r[0] in frames])
    ff=np.array([r[0] for r in W if r[0] in frames])
    best=None
    for si in range(len(splines)):
        P=sample_spline(si, ff)
        if P is None or len(P)!=len(Ww): continue
        # affine P ≈ Ww @ A + b  or Ww ≈ P @ A + b (pixel from world)
        # fit Ww = P @ A + b via least squares on centered
        Pc=P-P.mean(0); Wc=Ww-Ww.mean(0)
        # 2x2: Wc ≈ Pc @ M.T
        M,_,_,_=np.linalg.lstsq(Pc, Wc, rcond=None)
        pred=Pc@M + Ww.mean(0)
        err=np.linalg.norm(pred-Ww, axis=1).mean()
        # also direction corr of deltas
        if len(P)<3: continue
        dP=np.diff(P,axis=0); dW=np.diff(Ww,axis=0)
        c=np.corrcoef(dP[:,0], dW[:,0])[0,1] + np.corrcoef(dP[:,1], dW[:,1])[0,1]
        score=err - 0.5*c  # lower better
        if best is None or score<best[0]:
            best=(score, err, c, si+1, P[-1], Ww[-1])
    return best

for ped in [102,51]:
    b=match_ped(ped)
    print("ped", ped, "best_spline", b[3], "err", round(b[1],3), "corr", round(b[2],3), "vsp_last", b[4], "world_last", b[5])

# Mark with CORRECT splines on frame 161
m102=match_ped(102); m51=match_ped(51)
si102, si51=m102[3]-1, m51[3]-1

def pix(si, tf):
    P=sample_spline(si, [tf])[0]
    return P[0]+360, P[1]+288

im=Image.open(r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy\imgs\students003\frame_0161.jpg").convert("RGB")
d=ImageDraw.Draw(im)
for si,col,lab in [(si102,(255,0,0),"24102 ped102"), (si51,(0,255,0),"24063 ped51")]:
    u,v=pix(si,161)
    print(lab, "pixel", u,v)
    r=26
    d.ellipse([u-r,v-r,u+r,v+r], outline=col, width=5)
    d.line([u-30,v,u+30,v], fill=col, width=3); d.line([u,v-30,u,v+30], fill=col, width=3)
    d.text((u+18,v-28), lab, fill=col)
    # short trail
    for t in range(149,162):
        try:
            uu,vv=pix(si,t)
            d.ellipse([uu-2,vv-2,uu+2,vv+2], fill=col)
        except: pass
out=r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\compare_REMAPPED_f161.png"
im.save(out); print("wrote", out)
