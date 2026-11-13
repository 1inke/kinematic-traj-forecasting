import sys, json
from pathlib import Path
EXEC = Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018")
sys.path.insert(0, str(EXEC))
import numpy as np
from k0.freeze_inputs import run_freeze
from k0.data_io import load_kin
from k0.k0a import run_k0a
from k0.metrics import evaluate_set

fr = run_freeze()
data = fr["data"]
ix = data["split"]["outer"]
kin, _ = load_kin("outer", 42, data)
nat = data["native"][ix]; gt = data["gt"][ix]; grp = data["groups"][ix]
pool = np.concatenate([nat, kin], 1)
ua = run_k0a(pool, gt, nat, grp, do_cluster=False)
summary = dict(
    union_W_pass=ua["union_W"]["pass_"],
    union_fail=[k for k,v in ua["union_W"]["clauses"].items() if isinstance(v,dict) and not v.get("pass_",True)],
    k0a_W_pass=ua["W"]["pass_"],
    k0a_fail=[k for k,v in ua["W"]["clauses"].items() if isinstance(v,dict) and not v.get("pass_",True)],
    quality_ub_neg=ua["quality_upper_bound_negation"],
    union_quality_fail=ua["union_quality_fail_clauses"],
    change=ua["union_eval"]["change"],
)
# strip motion dicts size
ch = summary["change"]
summary["change_brief"] = {k: ch[k] for k in ("good_count","native_good_count","minADE","native_minADE","minFDE","native_minFDE","mean_minJ","native_mean_minJ","coverage","lost","motion_gate")}
(EXEC/"logs"/"quick_outer42_k0a.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(json.dumps(summary["change_brief"], ensure_ascii=False, indent=2, default=str))
print("union_fail", summary["union_fail"])
print("quality_ub", summary["quality_ub_neg"], summary["union_quality_fail"])
