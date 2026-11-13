import cv2, numpy as np, json
from pathlib import Path

ASTRA = Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA")
OUT = Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\visual_review_overnight")
HDIR = Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\visual_review_overnight\homography_cache")
# Will copy H from box via paths written next to script - load from ASTRA after we copy

def load_H(path):
    return np.loadtxt(path)

def world_to_pixel_astra(world_xy, H, scene):
    world = np.asarray(world_xy, dtype=np.float64)
    ones = np.ones((world.shape[0], 1))
    wh = np.concatenate([world, ones], axis=-1)
    ph = wh @ np.linalg.inv(H.T)
    pix = ph[:, :2] / ph[:, 2:3]
    pix = np.round(pix).astype(int)
    if scene not in ("eth", "hotel"):
        pix = pix[:, ::-1]
    return pix

def axis_probe(H, scene, origin):
    out = {}
    for name, dw in [("+dx",[1.,0.]), ("+dy",[0.,1.]), ("-dy",[0.,-1.])]:
        p0 = world_to_pixel_astra(origin[None], H, scene)[0]
        p1 = world_to_pixel_astra((origin+np.array(dw))[None], H, scene)[0]
        out[name] = {"drow": int(p1[0]-p0[0]), "dcol": int(p1[1]-p0[1])}
    return out

def recommend_display(axis):
    # Want plot_x ~ image col, plot_y ~ -image row
    # Local linear: [dcol; -drow] = M [dx; dy]
    # From unit steps:
    dx = axis["+dx"]; dy = axis["+dy"]
    # M columns
    # col0 = [dcol_dx, -drow_dx]; col1 = [dcol_dy, -drow_dy]
    M = np.array([
        [dx["dcol"], dy["dcol"]],
        [-dx["drow"], -dy["drow"]],
    ], dtype=float)
    # Which simple transform among identity, flip_y, swap, swap_flip matches M best (up to scale)?
    cands = {
        "identity (x,y)": np.array([[1.,0.],[0.,1.]]),
        "flip_y (x,-y)": np.array([[1.,0.],[0.,-1.]]),
        "flip_x (-x,y)": np.array([[-1.,0.],[0.,1.]]),
        "swap (y,x)": np.array([[0.,1.],[1.,0.]]),
        "swap_negx (y,-x)": np.array([[0.,1.],[-1.,0.]]),
        "swap_negy (-y,x)": np.array([[0.,-1.],[1.,0.]]),
        "neg_swap (-y,-x)": np.array([[0.,-1.],[-1.,0.]]),
        "rot90_ccw (-y,x) same": None,
    }
    # normalize M columns to unit for comparison
    def score(A, M):
        # find best positive scale s minimizing ||s A - M||_F over s, allowing per-axis scales
        # use least squares: vec(M) ≈ s * vec(A) is too strict; allow diag scale sx,sy on world before A
        # Simpler: correlation of each output axis
        # For each candidate A, compute predicted display delta for +dx and +dy, compare direction to M cols
        errs = []
        for col in range(2):
            m = M[:, col]
            a = A[:, col]
            if np.linalg.norm(m) < 1e-6 or np.linalg.norm(a) < 1e-6:
                # if M col near 0, skip
                continue
            # cosine similarity after optimal positive scale
            cos = np.dot(m, a) / (np.linalg.norm(m) * np.linalg.norm(a))
            errs.append(cos)
        return float(np.mean(errs)) if errs else -1.0

    ranked = []
    for name, A in cands.items():
        if A is None:
            continue
        ranked.append((score(A, M), name, A.tolist()))
    ranked.sort(reverse=True)
    return {"M_dcol_negdrow_per_dx_dy": M.tolist(), "ranked": ranked[:5], "best": ranked[0][1] if ranked else None}

# Load Hs - paths next to OUT
Hpaths = {
    "eth": OUT / "homography_cache" / "eth5_eth_H.txt",
    "hotel": OUT / "homography_cache" / "eth5_hotel_H.txt",
    "students003": OUT / "homography_cache" / "students003_H.txt",
    "zara01": OUT / "homography_cache" / "crowds_zara01_H.txt",
    "zara02": OUT / "homography_cache" / "crowds_zara02_H.txt",
}

report = {}

# HOTEL already known; recompute recommendation
H_hotel = np.loadtxt(Hpaths["hotel"])
origin_h = np.array([3.46, 2.39])
ax_h = axis_probe(H_hotel, "hotel", origin_h)
report["hotel"] = {"axis": ax_h, "display": recommend_display(ax_h)}

# ETH: pick a ped with clear motion from labels
lab_eth = ASTRA / "datasets/eth_ucy/labels/eth/biwi_eth.txt"
peds = {}
with open(lab_eth) as f:
    for line in f:
        p = line.split()
        if len(p) < 16: continue
        fr, ped = float(p[0]), int(round(float(p[1])))
        x, y = float(p[13]), float(p[15])
        peds.setdefault(ped, []).append((fr, x, y))
# choose longest nearly-straight large displacement
cands = []
for ped, pts in peds.items():
    a = np.array(pts)
    if len(a) < 15: continue
    d = a[-1,1:]-a[0,1:]
    cands.append((np.linalg.norm(d), ped, d.tolist(), int(a[0,0]), int(a[-1,0]), a[len(a)//2,1:].tolist()))
cands.sort(reverse=True)
H_eth = np.loadtxt(Hpaths["eth"])
ped, origin = cands[0][1], np.array(cands[0][5])
ax_e = axis_probe(H_eth, "eth", origin)
# project a segment and overlay
a = np.array(peds[ped])
seg = a[::max(1,len(a)//20)][:20]
pix = world_to_pixel_astra(seg[:,1:], H_eth, "eth")
report["eth"] = {
    "example_ped": ped, "delta_world": cands[0][2],
    "axis": ax_e, "display": recommend_display(ax_e),
    "delta_pixel_drow_dcol": (pix[-1]-pix[0]).tolist(),
}
vid = cv2.VideoCapture(str(ASTRA/"datasets/eth_ucy/videos/eth.avi"))
fr = int(seg[len(seg)//2,0]); vid.set(cv2.CAP_PROP_POS_FRAMES, fr)
ok, img = vid.read()
if ok:
    for (r,c) in pix:
        cv2.circle(img,(int(c),int(r)),4,(0,255,0),-1)
    r,c = pix[len(pix)//2]
    cv2.circle(img,(int(c),int(r)),12,(0,0,255),2)
    o = world_to_pixel_astra(origin[None], H_eth, "eth")[0]
    pdx = world_to_pixel_astra((origin+np.array([1.,0]))[None], H_eth, "eth")[0]
    pdy = world_to_pixel_astra((origin+np.array([0.,1.]))[None], H_eth, "eth")[0]
    cv2.arrowedLine(img,(int(o[1]),int(o[0])),(int(pdx[1]),int(pdx[0])),(255,0,0),2)
    cv2.arrowedLine(img,(int(o[1]),int(o[0])),(int(pdy[1]),int(pdy[0])),(0,165,255),2)
    cv2.putText(img,f"eth ped{ped} f{fr}",(15,35),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    cv2.imwrite(str(OUT/"eth_Hproj_mid.jpg"), img)
vid.release()

# STUDENTS003: ped 102 around frame 161 (sample 24102 history)
lab_s = ASTRA / "datasets/eth_ucy/labels/univ/students003.txt"
# load ped 102 frames near 145-170
rows=[]
with open(lab_s) as f:
    for line in f:
        p=line.split()
        if len(p)<16: continue
        if int(round(float(p[1])))!=102: continue
        fr=float(p[0]); x=float(p[13]); y=float(p[15])
        if 150<=fr<=170: rows.append((fr,x,y))
rows=np.array(rows)
print("students ped102 n", len(rows), rows[:3] if len(rows) else None)
H_s = np.loadtxt(Hpaths["students003"])
if len(rows):
    origin = rows[0,1:]
    ax_s = axis_probe(H_s, "students003", origin)
    pix = world_to_pixel_astra(rows[:,1:], H_s, "students003")
    report["students003"] = {
        "example_ped": 102,
        "world_start": rows[0,1:].tolist(), "world_end": rows[-1,1:].tolist(),
        "delta_world": (rows[-1,1:]-rows[0,1:]).tolist(),
        "delta_pixel_drow_dcol": (pix[-1]-pix[0]).tolist(),
        "axis": ax_s, "display": recommend_display(ax_s),
        "note": "ASTRA swaps UCY H output (col,row)->(row,col)",
    }
    vid = cv2.VideoCapture(str(ASTRA/"datasets/eth_ucy/videos/students003.avi"))
    print("stu video", vid.isOpened(), "frames", vid.get(cv2.CAP_PROP_FRAME_COUNT), "size", int(vid.get(3)), int(vid.get(4)))
    # UCY annotation frame indexing: often frame*1 or need check. Try annotation frame directly.
    for fr_try in [int(rows[len(rows)//2,0]), int(rows[0,0]), int(rows[-1,0])]:
        for off,tag in [(0,"f")]:
            vid.set(cv2.CAP_PROP_POS_FRAMES, fr_try+off)
            ok,img=vid.read()
            if not ok: continue
            img2=img.copy()
            for (r,c) in pix:
                cv2.circle(img2,(int(c),int(r)),4,(0,255,0),-1)
            # pick closest
            idx=int(np.argmin(np.abs(rows[:,0]-fr_try)))
            r,c=pix[idx]
            cv2.circle(img2,(int(c),int(r)),14,(0,0,255),2)
            o=world_to_pixel_astra(origin[None],H_s,"students003")[0]
            pdx=world_to_pixel_astra((origin+np.array([1.,0]))[None],H_s,"students003")[0]
            pdy=world_to_pixel_astra((origin+np.array([0.,1.]))[None],H_s,"students003")[0]
            cv2.arrowedLine(img2,(int(o[1]),int(o[0])),(int(pdx[1]),int(pdx[0])),(255,0,0),2)
            cv2.arrowedLine(img2,(int(o[1]),int(o[0])),(int(pdy[1]),int(pdy[0])),(0,165,255),2)
            cv2.putText(img2,f"stu ped102 ann{fr_try} w=({rows[idx,1]:.2f},{rows[idx,2]:.2f}) rc=({r},{c})",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
            cv2.imwrite(str(OUT/f"stu_Hproj_{tag}{fr_try}.jpg"), img2)
            print("wrote stu", fr_try, "rc", r,c, "img", img.shape)
    vid.release()

# ZARA01 / ZARA02 axis only at a typical origin from labels
for scene, labrel, Hkey, vidname in [
    ("zara01", "labels/zara01/crowds_zara01.txt", "zara01", "zara01.avi"),
    ("zara02", "labels/zara02/crowds_zara02.txt", "zara02", "zara02.avi"),
]:
    lab = ASTRA / "datasets/eth_ucy" / labrel
    # first ped first point as origin
    with open(lab) as f:
        for line in f:
            p=line.split()
            if len(p)<16: continue
            origin=np.array([float(p[13]), float(p[15])])
            break
    H=np.loadtxt(Hpaths[Hkey])
    ax=axis_probe(H, scene, origin)
    report[scene]={"origin_example": origin.tolist(), "axis": ax, "display": recommend_display(ax)}

with open(OUT/"coord_validation_report2.json","w",encoding="utf-8") as f:
    json.dump(report,f,indent=2)
print(json.dumps(report, indent=2))
