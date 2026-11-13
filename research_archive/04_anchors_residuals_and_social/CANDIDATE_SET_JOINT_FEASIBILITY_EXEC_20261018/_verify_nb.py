import json
from pathlib import Path
vr = Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\visual_review_overnight")
raw = (vr/"data/common.js").read_text(encoding="utf-8")
data = json.loads(raw[raw.find("{"):].rstrip().rstrip(";"))
idx = data["ids"].index(24102)
nbs = data["neighbors"][idx]
print("24102 index", idx)
print("n_neighbors", len(nbs))
for i,p in enumerate(nbs):
    print(f"  nb{i}: steps={len(p)} first={p[0]} last={p[-1]}")
lens = [len(x) for x in data["neighbors"]]
print("overall nonempty", sum(1 for L in lens if L>0), "/", len(lens))
print("mean", round(sum(lens)/len(lens),3), "max", max(lens), "min", min(lens))
print("past last", data["past"][idx][-1])
audit = json.loads((vr/"audit.json").read_text(encoding="utf-8"))
print("audit", {k:audit[k] for k in audit if "neighbor" in k or "raw_past" in k or k in ("n","status")})
# viewer still has show-neighbors
vj = (vr/"viewer.js").read_text(encoding="utf-8")
print("viewer show-neighbors", "show-neighbors" in vj)
ih = (vr/"index.html").read_text(encoding="utf-8")
print("index checkbox", "show-neighbors" in ih)
