from pathlib import Path
import re, numpy as np
from PIL import Image, ImageDraw

# Load all world positions at frame 161 from txt
at={}
for line in Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA\datasets\eth_ucy\labels\univ\students003.txt").read_text(encoding="utf-8",errors="ignore").splitlines():
    a=line.split()
    if len(a)<16: continue
    if abs(float(a[0])-161)>1e-6: continue
    at[int(float(a[1]))]=(float(a[13]), float(a[15]))
print("n at 161", len(at), "102", at.get(102), "51", at.get(51))

# Parse all vsp positions at vf=1610
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

def vsp_at(si, vf):
    pts=splines[si]
    frs=np.array([p[0] for p in pts]); xs=np.array([p[1] for p in pts]); ys=np.array([p[2] for p in pts])
    if vf<frs.min() or vf>frs.max(): return None
    return float(np.interp(vf,frs,xs)), float(np.interp(vf,frs,ys))

vpos=[]
for si in range(len(splines)):
    p=vsp_at(si,1610)
    if p: vpos.append((si+1,)+p)
print("vsp n at 1610", len(vpos))

# Try known H matrices from public repos (pixel<-world or world<-pixel)
# Form1: world = H @ [u,v,1]  (common ETH)
candidates=[]
# SGAN-style / various
candidates.append(("guess1", np.array([
 [0.0213266929223662, -0.00159145950240012, -8.41420215320282],
 [0.000918344490753411, -0.0159700264972413, 14.1405451628211],
 [0,0,1.0]])))
# From issue discussions sometimes inverted
# Fit H from Hungarian matching: for each world ped find nearest vsp after trying linear transform
# Simple RANSAC-free: assume u = a*wx+b*wy+c, v=d*wx+e*wy+f, solve with ALL if we match by sort order? bad.

# Match by: optimize assignment between world peds and vsp points with affine
from itertools import permutations
# Use cv2-free affine fit via least squares on matched pairs if ped id == spline id for SOME and check residual for 102,51

def world_to_pix_H(xy, H):
    # ASTRA: pixel_h = world_h @ inv(H.T); then swap for UCY
    w=np.array([xy[0],xy[1],1.0])
    ph=w @ np.linalg.inv(H.T)
    p=ph[:2]/ph[2]
    # UCY swap to (row,col) -> our display uses (u,v)=(col,row)? 
    # ASTRA says after swap pixels_coord is (row,col). Image draw uses (x=col, y=row)
    row,col=p[1],p[0]  # if p was (col,row) before swap... follow ASTRA: pixels_coord = p[:,::-1] so (row,col)
    # Actually code: pixels_coord = pixel_homogeneous[:,:2]/... then if UCY: pixels_coord = pixels_coord[:,::-1]
    # so before swap is (x,y) from formula, after swap (y,x). Image coords typically x=col=first after? 
    # Let's try both conventions
    return p  # raw before swap

for name,H in candidates:
    for ped in [102,51]:
        p=world_to_pix_H(at[ped], H)
        pswap=p[::-1]
        print(name, "ped", ped, "raw", p, "swap", pswap, "as_uv_raw", (p[0]+0,p[1]), "centered_add", (p[0],p[1]))

# Better: fit affine world->centered_pixel using bipartite greedy after scale normalize
W=np.array([at[k] for k in sorted(at)])
P=np.array([[x,y] for _,x,y in vpos])
print("world range", W.min(0), W.max(0), "vsp range", P.min(0), P.max(0))

# Procrustes-like: for each world point, we need correspondence. 
# Use Hungarian on pairwise after normalizing both to zero mean unit std - still need same count
# n_world=27, n_vsp=? may differ

# Greedy: normalize, for each world find nearest vsp in normalized space under many random rotations? 
# Simpler approach from UCY: annotations in vsp ARE in image coords with origin center.
# World coords come from H @ pixel. So pixel_h = inv(H) @ world_h

def try_invH(H, xy, swap=False):
    w=np.array([xy[0],xy[1],1.0])
    # world = H @ pixel => pixel = inv(H) @ world
    ph = np.linalg.inv(H) @ w
    p = ph[:2]/ph[2]
    if swap: p=p[::-1]
    return p

for name,H in candidates:
    for swap in [False, True]:
        errs=[]
        # if spline id == ped id for all at, compute error
        for ped,xy in at.items():
            if ped-1>=len(splines): continue
            vp=vsp_at(ped-1,1610)
            if vp is None: continue
            pred=try_invH(H, xy, swap)
            # vsp is centered; pred might be absolute pixel
            # try pred as absolute and as centered
            for mode,pp in [("abs",pred),("cen",pred),("abs-c",(pred[0]-360,pred[1]-288))]:
                if mode=="cen":
                    continue
            # compare pred absolute to vsp+360
            vabs=(vp[0]+360, vp[1]+288)
            # and pred centered to vp
            e1=np.hypot(pred[0]-vabs[0], pred[1]-vabs[1])
            e2=np.hypot(pred[0]-vp[0], pred[1]-vp[1])
            e3=np.hypot((pred[0]-360)-vp[0], (pred[1]-288)-vp[1])
            errs.append((min(e1,e2,e3), ped))
        errs.sort()
        print(name, "swap", swap, "median minerr", np.median([e for e,_ in errs]), "p102", [e for e,p in errs if p==102], "p51", [e for e,p in errs if p==51])
