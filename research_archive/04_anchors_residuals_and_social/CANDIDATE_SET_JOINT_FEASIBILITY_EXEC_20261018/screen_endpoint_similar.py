import json, os
import numpy as np

base = r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018"
vis = os.path.join(base, "visual_review_overnight", "data")
arr = np.load(os.path.join(vis, "arrays.npz"), allow_pickle=True)
ids = arr["sample_id"].astype(np.int64)
past = arr["past"].astype(np.float64)
future = arr["future"].astype(np.float64)
logits = arr["logits"].astype(np.float64)
num_valid = arr["num_valid"].astype(np.int64)
native = arr["native"].astype(np.float64)
k1_S = arr["k1_S"].astype(np.float64)
N = len(ids)
dt = 0.4

meta = np.load(r"<WORKSPACE_ROOT>\F16_OBSERVABLE_CUE_DIAGNOSIS_20261003\metadata.npz", allow_pickle=True)
mid = meta["sample_id"].astype(np.int64)
seq = meta["sequence"]
sfam = meta["scene_family"]
sframe = meta["start_frame"].astype(np.int64)
ped = meta["pedestrian_id"].astype(np.int64)
meta_map = {int(mid[i]): i for i in range(len(mid))}

def ade_fde(pred, gt):
    d = np.linalg.norm(pred - gt[None], axis=-1)
    return d.mean(axis=1), d[:, -1]

def const_vel_endpoint(p):
    v = (p[-1] - p[-2]) / dt
    return p[-1] + v * dt * 12

rows = []
for i in range(N):
    sid = int(ids[i])
    gt = future[i]
    p = past[i]
    gt_end = gt[-1]
    cv_end = const_vel_endpoint(p)
    v = p[-1] - p[-2]
    speed = float(np.linalg.norm(v) / dt)
    if speed < 1e-6:
        lat = float(np.linalg.norm(gt_end - cv_end))
        along = 0.0
    else:
        h = v / (np.linalg.norm(v) + 1e-9)
        nvec = np.array([-h[1], h[0]])
        delta = gt_end - cv_end
        along = float(np.dot(delta, h))
        lat = float(abs(np.dot(delta, nvec)))

    ade_n, fde_n = ade_fde(native[i], gt)
    ade_s, fde_s = ade_fde(k1_S[i], gt)
    min_ade_n, min_fde_n = float(ade_n.min()), float(fde_n.min())
    min_ade_s, min_fde_s = float(ade_s.min()), float(fde_s.min())
    best_n = int(ade_n.argmin())
    end_gap_n = float(np.linalg.norm(native[i, best_n, -1] - gt_end))
    top = int(logits[i].argmax())
    top_end_gap_n = float(np.linalg.norm(native[i, top, -1] - gt_end))
    top_fde_n = float(fde_n[top])
    n_neigh = int(num_valid[i])

    mi = meta_map.get(sid)
    if mi is None:
        sequence, family, start_frame, ped_id = "?", "?", -1, -1
    else:
        sequence = str(seq[mi])
        family = str(sfam[mi])
        start_frame = int(sframe[mi])
        ped_id = int(ped[mi])

    t0_frame = start_frame + 7 if start_frame >= 0 else -1
    tend_frame = start_frame + 19 if start_frame >= 0 else -1
    t0_sec = t0_frame * 0.4 if t0_frame >= 0 else -1
    tend_sec = tend_frame * 0.4 if tend_frame >= 0 else -1
    obs_start_sec = start_frame * 0.4 if start_frame >= 0 else -1

    score = max(min_fde_n, min_fde_s, top_end_gap_n, lat, end_gap_n)
    rows.append({
        "sample_id": sid,
        "sequence": sequence,
        "scene_family": family,
        "pedestrian_id": ped_id,
        "start_frame": start_frame,
        "obs_start_sec": round(obs_start_sec, 1) if obs_start_sec >= 0 else None,
        "t0_sec": round(t0_sec, 1) if t0_sec >= 0 else None,
        "pred_end_sec": round(tend_sec, 1) if tend_sec >= 0 else None,
        "n_neighbors": n_neigh,
        "minFDE_native": round(min_fde_n, 3),
        "minFDE_k1_S": round(min_fde_s, 3),
        "minADE_native": round(min_ade_n, 3),
        "top_end_gap_native": round(top_end_gap_n, 3),
        "top_fde_native": round(top_fde_n, 3),
        "bestADE_end_gap_native": round(end_gap_n, 3),
        "gt_vs_cv_lat": round(lat, 3),
        "gt_vs_cv_along": round(along, 3),
        "speed": round(speed, 3),
        "score": score,
    })

ref = [r for r in rows if r["sample_id"] == 24102]
print("REF 24102:", json.dumps(ref[0] if ref else None, ensure_ascii=False, indent=2))

cands = []
for r in rows:
    if r["sample_id"] == 24102:
        continue
    flags = []
    if r["minFDE_native"] >= 0.8:
        flags.append("minFDE_n")
    if r["minFDE_k1_S"] >= 0.8:
        flags.append("minFDE_s")
    if r["top_end_gap_native"] >= 1.0:
        flags.append("top_end")
    if r["gt_vs_cv_lat"] >= 0.8:
        flags.append("lat")
    if r["bestADE_end_gap_native"] >= 0.8:
        flags.append("best_end")
    if len(flags) < 2 and r["score"] < 1.2:
        continue
    if not flags:
        continue
    r = dict(r)
    r["flags"] = flags
    pref = 0
    seq_l = r["sequence"].lower()
    if "univ" in seq_l or "student" in seq_l or r["scene_family"] == "university":
        pref += 2
    if r["n_neighbors"] >= 1:
        pref += 1
    if r["n_neighbors"] >= 3:
        pref += 1
    if r["gt_vs_cv_lat"] >= 1.0 and r["top_end_gap_native"] >= 1.0:
        pref += 2
    r["pref"] = pref
    cands.append(r)

cands.sort(key=lambda x: (x["pref"], x["score"], x["top_end_gap_native"]), reverse=True)

picked = []
seq_count = {}
for r in cands:
    sc = r["sequence"]
    if seq_count.get(sc, 0) >= 5:
        continue
    picked.append(r)
    seq_count[sc] = seq_count.get(sc, 0) + 1
    if len(picked) >= 18:
        break

print("N_cands", len(cands), "picked", len(picked))
print("scene counts", seq_count)
for r in picked:
    print(
        f"{r['sample_id']}\t{r['sequence']}\tstart={r['start_frame']}\tt0={r['t0_sec']}s\tend={r['pred_end_sec']}s\tnN={r['n_neighbors']}\tminFDE_n={r['minFDE_native']}\ttop_end={r['top_end_gap_native']}\tlat={r['gt_vs_cv_lat']}\tflags={','.join(r['flags'])}"
    )

out = {
    "idea": "Look beyond current 12-step prediction horizon so endpoint intent can differentiate go-around vs stop-in-front; inspired by 24102.",
    "definition": {
        "minFDE_native_ge": 0.8,
        "minFDE_k1_S_ge": 0.8,
        "top_end_gap_native_ge": 1.0,
        "gt_vs_cv_lat_ge": 0.8,
        "bestADE_end_gap_ge": 0.8,
        "need_at_least_flags": 2,
        "or_score_ge": 1.2,
        "time_mapping": "annot_frame*10/25fps = annot*0.4s; t0=start+7; pred_end=start+19",
        "exclude": 24102,
    },
    "reference_24102": ref[0] if ref else None,
    "n_candidates_raw": len(cands),
    "samples": picked,
}
json_path = os.path.join(base, "ENDPOINT_HORIZON_IDEA_SIMILAR_SAMPLES_24102.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

lines = []
lines.append("# Endpoint-horizon idea: similar samples to 24102")
lines.append("")
lines.append("**Idea (user, 2026-10-19 Asia/Tokyo):** For 24102, models put the 12-step endpoint in front of a person-wall while GT goes around downward. Suggestion: consider a farther future endpoint (beyond the current prediction window) so the generator/selector can prefer more flexible detours.")
lines.append("")
lines.append("**This note:** operational screen for endpoint-large cases on ETH outer seed42 visual set (N=1035). Does **not** change any P1 protocol.")
lines.append("")
lines.append("## Screen definition")
lines.append("")
lines.append("- Flags (need >=2, or composite score >=1.2):")
lines.append("  - minFDE_native >= 0.8 m")
lines.append("  - minFDE_k1_S >= 0.8 m")
lines.append("  - top-logit native endpoint vs GT endpoint >= 1.0 m")
lines.append("  - GT endpoint vs constant-velocity endpoint |lateral| >= 0.8 m")
lines.append("  - best-minADE native endpoint vs GT endpoint >= 0.8 m")
lines.append("- Prefer: university/students + neighbors; exclude 24102; at most 5 per sequence.")
lines.append("- Time: video_sec ≈ annot_frame * 0.4 (frame×10/25). Observation start = start_frame; **t0** = start+7; **pred_end** = start+19.")
lines.append("")
lines.append("## Reference: 24102")
lines.append("")
if ref:
    r = ref[0]
    lines.append(f"- sequence `{r['sequence']}`, ped `{r['pedestrian_id']}`, start_frame `{r['start_frame']}`")
    lines.append(f"- obs≈{r['obs_start_sec']}s, t0≈{r['t0_sec']}s, pred_end≈{r['pred_end_sec']}s, neighbors={r['n_neighbors']}")
    lines.append(f"- minFDE_native={r['minFDE_native']}, minFDE_k1_S={r['minFDE_k1_S']}, top_end_gap={r['top_end_gap_native']}, gt_vs_cv_lat={r['gt_vs_cv_lat']}")
lines.append("")
lines.append(f"## Selected similar samples (n={len(picked)}, raw flagged={len(cands)})")
lines.append("")
lines.append("| sample_id | sequence | start_frame | t0≈s | pred_end≈s | n_nei | minFDE_n | top_end_gap | lat_vs_CV | why-like |")
lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
for r in picked:
    why = []
    if r["top_end_gap_native"] >= 1.0:
        why.append("top终点远离GT")
    if r["minFDE_native"] >= 0.8:
        why.append("K20也盖不住末点")
    if r["gt_vs_cv_lat"] >= 0.8:
        why.append("相对匀速大横偏")
    if r["n_neighbors"] >= 3:
        why.append("多人邻域")
    if "student" in r["sequence"].lower() or r["scene_family"] == "university":
        why.append("校园广场")
    why_s = "；".join(why) if why else ",".join(r["flags"])
    lines.append(
        f"| {r['sample_id']} | {r['sequence']} | {r['start_frame']} | {r['t0_sec']} | {r['pred_end_sec']} | {r['n_neighbors']} | {r['minFDE_native']} | {r['top_end_gap_native']} | {r['gt_vs_cv_lat']} | {why_s} |"
    )
lines.append("")
lines.append("## How to watch")
lines.append("")
lines.append("1. Open the matching ETH/UCY video (students003 / students001 / uni_examples / etc.).")
lines.append("2. Seek to **t0≈** for prediction start; **pred_end≈** for current 12-step horizon end.")
lines.append("3. Ask: does a farther horizon (e.g. +2–4 s past pred_end) make the go-around / intent clearer than the 12-step endpoint alone?")
lines.append("")
lines.append("Companion JSON: `ENDPOINT_HORIZON_IDEA_SIMILAR_SAMPLES_24102.json`")
lines.append("")

md_path = os.path.join(base, "ENDPOINT_HORIZON_IDEA_SIMILAR_SAMPLES_24102.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("WROTE", md_path)
print("WROTE", json_path)
