import os, cv2, numpy as np, json
from pathlib import Path

ASTRA = Path(r"<LOCAL_DESKTOP>\DRSE\ASTRA")
HDIR = ASTRA / "datasets" / "eth_ucy" / "homography"
OUT = Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\visual_review_overnight")
OUT.mkdir(parents=True, exist_ok=True)
HDIR.mkdir(parents=True, exist_ok=True)

# BIWI H: pixel -> world. Citations: GUIDE-CoT data/eth5/{eth,hotel}/H.txt (same as EWAP H.txt)
H_hotel = np.array([
    [1.1048200e-02, 6.6958900e-04, -3.3295300e+00],
    [-1.5966000e-03, 1.1632400e-02, -5.3951400e+00],
    [1.1190700e-04, 1.3617400e-05, 5.4276600e-01],
], dtype=np.float64)
H_eth = np.array([
    [2.8128700e-02, 2.0091900e-03, -4.6693600e+00],
    [8.0625700e-04, 2.5195500e-02, -5.0608800e+00],
    [3.4555400e-04, 9.2512200e-05, 4.6255300e-01],
], dtype=np.float64)
np.savetxt(HDIR / "H_hotel.txt", H_hotel)
np.savetxt(HDIR / "H_eth.txt", H_eth)

def world_to_pixel_astra(world_xy, H, scene):
    """Match ASTRA utils.visualizer.world_to_pixel for (T,2). Returns (row,col)."""
    world = np.asarray(world_xy, dtype=np.float64)
    ones = np.ones((world.shape[0], 1))
    wh = np.concatenate([world, ones], axis=-1)
    ph = wh @ np.linalg.inv(H.T)
    pix = ph[:, :2] / ph[:, 2:3]
    pix = np.round(pix).astype(int)
    if scene not in ("eth", "hotel"):
        pix = pix[:, ::-1]
    return pix

def world_to_pixel_cv(world_xy, H):
    world = np.asarray(world_xy, dtype=np.float64).reshape(-1, 1, 2)
    _, Hinv = cv2.invert(H.astype(np.float64))
    pix = cv2.perspectiveTransform(world, Hinv).reshape(-1, 2)
    return pix  # (col, row)

def load_label_track(path, ped_id, f0, f1):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 16:
                continue
            fr = float(parts[0]); ped = float(parts[1])
            if int(round(ped)) != int(ped_id):
                continue
            x = float(parts[13]); y = float(parts[15])
            if f0 <= fr <= f1:
                rows.append((fr, x, y))
    return np.array(rows)

report = {"scenes": {}}

# ---- HOTEL ped 259 / sample 740 window ----
lab_hotel = ASTRA / "datasets/eth_ucy/labels/hotel/biwi_hotel.txt"
tr = load_label_track(lab_hotel, 259, 1086, 1105)
assert len(tr) > 0, "no hotel track"
pix_astra = world_to_pixel_astra(tr[:, 1:], H_hotel, "hotel")
pix_cv = world_to_pixel_cv(tr[:, 1:], H_hotel)

origin = tr[0, 1:]
axis = {}
for name, dw in [("+dx", [1.0, 0.0]), ("+dy", [0.0, 1.0]), ("-dy", [0.0, -1.0]), ("+dx+dy", [1.0, 1.0])]:
    p0 = world_to_pixel_astra(origin[None], H_hotel, "hotel")[0]
    p1 = world_to_pixel_astra((origin + np.array(dw))[None], H_hotel, "hotel")[0]
    axis[name] = {"drow": int(p1[0]-p0[0]), "dcol": int(p1[1]-p0[1])}

d_world = (tr[-1, 1:] - tr[0, 1:]).tolist()
d_pix = (pix_astra[-1] - pix_astra[0]).tolist()

# Also raw CV: (dcol, drow)
d_cv = (np.round(pix_cv[-1] - pix_cv[0]).astype(int)).tolist()

hotel_info = {
    "ped": 259,
    "frames": tr[:, 0].astype(int).tolist(),
    "world_start": tr[0, 1:].tolist(),
    "world_end": tr[-1, 1:].tolist(),
    "delta_world_xy": d_world,
    "astra_pixels_rowcol_start": pix_astra[0].tolist(),
    "astra_pixels_rowcol_end": pix_astra[-1].tolist(),
    "delta_astra_drow_dcol": d_pix,
    "delta_cv_dcol_drow": d_cv,
    "unit_step_pixel_delta": axis,
    "interpretation": None,
}
# Infer: which world axis aligns with image horizontal (col)
# If |dcol| >> |drow| for track, motion is horizontal on screen
# Track is mostly -dy in world
if abs(d_pix[1]) > abs(d_pix[0]) * 2:
    motion_screen = "horizontal (mostly dcol)"
elif abs(d_pix[0]) > abs(d_pix[1]) * 2:
    motion_screen = "vertical (mostly drow)"
else:
    motion_screen = "diagonal"
hotel_info["track_motion_on_screen"] = motion_screen

# From unit steps: does +dy increase row (down) or col?
dy = axis["+dy"]
dx = axis["+dx"]
hotel_info["mapping_from_H"] = {
    "world_+x_increases": f"row by {dx['drow']}, col by {dx['dcol']}",
    "world_+y_increases": f"row by {dy['drow']}, col by {dy['dcol']}",
}
# Display transform for "video-aligned" plot (x right = image right, y up = image up):
# image right = +col, image up = -row
# We want plot_x ~ something proportional to col, plot_y ~ something proportional to -row
# Approximate locally with linear map from world to (col, -row)
# From unit steps at origin:
# dcol ≈ a*dx + b*dy;  -drow ≈ c*dx + d*dy
# For hotel: if +dy mainly changes col, then plot_x should follow world y (or -y)
report["scenes"]["hotel"] = hotel_info

vid = str(ASTRA / "datasets/eth_ucy/videos/hotel.avi")
cap = cv2.VideoCapture(vid)
W = int(cap.get(3)); Hgt = int(cap.get(4)); nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
hotel_info["video"] = {"w": W, "h": Hgt, "nframes": nfr, "fps": cap.get(cv2.CAP_PROP_FPS)}
# Draw overlays at start/mid/end
for idx in [0, len(tr)//2, len(tr)-1]:
    fr = int(tr[idx, 0])
    # try frame index = annotation frame (common for BIWI)
    for offset, tag in [(0, "f"), (-1, "fm1")]:
        pos = fr + offset
        if pos < 0 or pos >= nfr:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, img = cap.read()
        if not ok:
            continue
        img2 = img.copy()
        for (r, c) in pix_astra:
            cv2.circle(img2, (int(c), int(r)), 4, (0, 255, 0), -1)
        r, c = pix_astra[idx]
        cv2.circle(img2, (int(c), int(r)), 14, (0, 0, 255), 2)
        cv2.putText(img2, f"ann{fr} seek{pos} w=({tr[idx,1]:.2f},{tr[idx,2]:.2f}) rc=({r},{c})",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        # axis legend at origin projection
        o = world_to_pixel_astra(origin[None], H_hotel, "hotel")[0]
        ox, oy = int(o[1]), int(o[0])
        pdx = world_to_pixel_astra((origin + np.array([1.0, 0]))[None], H_hotel, "hotel")[0]
        pdy = world_to_pixel_astra((origin + np.array([0.0, 1.0]))[None], H_hotel, "hotel")[0]
        cv2.arrowedLine(img2, (ox, oy), (int(pdx[1]), int(pdx[0])), (255, 0, 0), 2)  # +x blue
        cv2.arrowedLine(img2, (ox, oy), (int(pdy[1]), int(pdy[0])), (0, 165, 255), 2)  # +y orange
        cv2.putText(img2, "+x", (int(pdx[1]), int(pdx[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        cv2.putText(img2, "+y", (int(pdy[1]), int(pdy[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
        outp = OUT / f"hotel_Hproj_{tag}{fr}.jpg"
        cv2.imwrite(str(outp), img2)
        print("wrote", outp.name)
cap.release()

# ---- Also check a few other hotel peds with large |dx| to see "left-right" crowd ----
# Find ped with large world-x motion in hotel labels
peds = {}
with open(lab_hotel) as f:
    for line in f:
        parts = line.split()
        if len(parts) < 16:
            continue
        fr = float(parts[0]); ped = int(round(float(parts[1])))
        x = float(parts[13]); y = float(parts[15])
        peds.setdefault(ped, []).append((fr, x, y))
best = []
for ped, pts in peds.items():
    a = np.array(pts)
    if len(a) < 10:
        continue
    dx = a[-1,1] - a[0,1]; dy = a[-1,2] - a[0,2]
    best.append((abs(dx), abs(dy), ped, float(dx), float(dy), int(a[0,0]), int(a[-1,0])))
best.sort(reverse=True)
hotel_info["top_abs_dx_peds"] = best[:8]
hotel_info["top_abs_dy_peds"] = sorted(best, key=lambda t: t[1], reverse=True)[:8]

# Project one high-|dx| ped if any
if best:
    ped = best[0][2]
    a = np.array(peds[ped])
    # take middle 20 frames
    mid = len(a)//2
    seg = a[max(0, mid-10): mid+10]
    pix = world_to_pixel_astra(seg[:,1:], H_hotel, "hotel")
    d = pix[-1] - pix[0]
    hotel_info["high_dx_example"] = {
        "ped": ped, "delta_world": [float(seg[-1,1]-seg[0,1]), float(seg[-1,2]-seg[0,2])],
        "delta_pixel_drow_dcol": d.tolist(),
        "frames": [int(seg[0,0]), int(seg[-1,0])],
    }

# Save report so far
with open(OUT / "coord_validation_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print(json.dumps(hotel_info, indent=2))
print("OK hotel section")
