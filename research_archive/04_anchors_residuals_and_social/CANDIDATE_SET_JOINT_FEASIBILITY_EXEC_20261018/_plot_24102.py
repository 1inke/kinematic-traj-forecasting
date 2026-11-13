import json, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"<WORKSPACE_ROOT>")
EXEC = ROOT / "CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018"
raw = (EXEC/"visual_review_overnight/data/common.js").read_text(encoding="utf-8")
data = json.loads(raw[raw.find("{"):].rstrip().rstrip(";"))
idx = data["ids"].index(24102)
past = np.array(data["past"][idx], float)
gt = np.array(data["gt"][idx], float)
# k1_S paths if available
k1s_path = EXEC/"visual_review_overnight/data/k1_S.js"
preds = None
if k1s_path.exists():
    raw2 = k1s_path.read_text(encoding="utf-8")
    j = json.loads(raw2[raw2.find("{"):].rstrip().rstrip(";"))
    # structure may be paths or trajectories
    for key in ("paths","traj","predictions","cand"):
        if key in j:
            preds = np.array(j[key][idx], float)
            break
    if preds is None and isinstance(j, dict):
        # maybe list at top under one key
        for k,v in j.items():
            if isinstance(v, list) and len(v)==len(data["ids"]):
                try:
                    preds = np.array(v[idx], float)
                    print("using key", k, "shape", preds.shape)
                    break
                except Exception as e:
                    pass

z = np.load(ROOT/"REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz")
r = int(np.where(z["sample_id"]==24102)[0][0])
soc = z["social"][r].reshape(4,8,3)

fig, ax = plt.subplots(figsize=(7.2,6.2))
ax.plot(past[:,0], past[:,1], "k-o", ms=4, lw=2, label="past")
ax.plot(gt[:,0], gt[:,1], "g-o", ms=4, lw=2, label="GT")
ax.plot([0],[0], "k*", ms=16, label="t0")
if preds is not None and preds.ndim==3:
    for i in range(min(20, preds.shape[0])):
        ax.plot(preds[i,:,0], preds[i,:,1], color="orange", alpha=0.25, lw=1)
    ax.plot([],[], color="orange", alpha=0.6, label="k1_S candidates")
for i in range(4):
    nb = soc[i,:,:2]
    if np.allclose(nb,0):
        continue
    ax.plot(nb[:,0], nb[:,1], "--", alpha=0.75, label=f"nb{i} social-rel (layout guess)")
ax.set_aspect("equal", adjustable="datalim")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
ax.set_title("ID 24102 | UCY students003 | ped 102 | start_frame 144\n(~57.6s @2.5Hz index; or ~5.76s if frame=25fps video index)")
out = EXEC / "sample_24102_bev_scene.png"
fig.tight_layout(); fig.savefig(out, dpi=150)
print("WROTE", out, "bytes", out.stat().st_size)
# also save machine-local copy with clean name at visual dir
out2 = EXEC/"visual_review_overnight"/"sample_24102_scene.png"
fig.savefig(out2, dpi=150)
print("WROTE", out2)
