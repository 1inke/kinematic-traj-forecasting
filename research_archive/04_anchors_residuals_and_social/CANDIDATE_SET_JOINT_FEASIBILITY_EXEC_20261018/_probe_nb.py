import json, re
from pathlib import Path
from collections import Counter
base = Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\visual_review_overnight")
raw = (base/"data"/"common.js").read_text(encoding="utf-8")
i = raw.find("{"); rawj = raw[i:].rstrip().rstrip(";")
data = json.loads(rawj)
nlen = [len(x) if isinstance(x,list) else -1 for x in data["neighbors"]]
print("neighbor_count_hist", Counter(nlen).most_common(20))
print("n_with_neighbors", sum(1 for x in nlen if x>0), "/", len(nlen))
print("groups_for_24102", data["groups"][data["ids"].index(24102)])
print("protocol", json.dumps(data["protocol"], indent=2)[:1500])
# sample a few with neighbors
for i,nb in enumerate(data["neighbors"]):
    if nb:
        print("example with neighbors id", data["ids"][i], "n_nb", len(nb), "nb0_keys_or_type", type(nb[0]).__name__, "len0", len(nb[0]) if isinstance(nb[0],list) else nb[0])
        break
