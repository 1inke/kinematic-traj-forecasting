import json, re, os, sys
from pathlib import Path
base = Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\visual_review_overnight")
common_path = base / "data" / "common.js"
raw = common_path.read_text(encoding="utf-8")
# strip window.AUDIT_COMMON=
m = re.match(r"window\.AUDIT_COMMON\s*=\s*(\{.*\})\s*;?\s*$", raw, re.S)
if not m:
    # try find first {
    i = raw.find("{")
    rawj = raw[i:]
    if rawj.endswith(";"):
        rawj = rawj[:-1]
else:
    rawj = m.group(1)
print("loading json...", flush=True)
data = json.loads(rawj)
print("keys:", sorted(data.keys()))
ids = data["ids"]
print("n_ids", len(ids), "has 24102", 24102 in ids)
idx = ids.index(24102)
print("index", idx)
# print per-key info for this index
for k,v in data.items():
    if k == "ids":
        continue
    if isinstance(v, list) and len(v) == len(ids):
        item = v[idx]
        if isinstance(item, list):
            # maybe nested traj
            shape_hint = type(item).__name__
            if item and isinstance(item[0], list):
                shape_hint = f"list[{len(item)}]x{len(item[0]) if item[0] else 0}"
                if item[0] and isinstance(item[0][0], (int,float)):
                    print(f"{k}: {shape_hint} first={item[0][:2]} last={item[-1][:2] if item else None}")
                else:
                    print(f"{k}: nested {shape_hint}")
            else:
                print(f"{k}: list len={len(item)} sample={item[:5] if len(item)>5 else item}")
        else:
            print(f"{k}: {item}")
    elif isinstance(v, dict):
        print(f"{k}: dict keys={list(v.keys())[:10]}")
    elif isinstance(v, list):
        print(f"{k}: list len={len(v)} (not per-sample)")
    else:
        print(f"{k}: {type(v).__name__}={v}")

# dump sample meta to file
out = {
    "sample_id": 24102,
    "index": idx,
}
for k,v in data.items():
    if isinstance(v, list) and len(v) == len(ids):
        out[k] = v[idx]
meta_path = Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\_sample_24102_meta.json")
meta_path.write_text(json.dumps(out, indent=2)[:200000], encoding="utf-8")
print("wrote", meta_path, "bytes", meta_path.stat().st_size)
