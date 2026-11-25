"""NumPy reference of the prefix-winner rule + eval; run with python -I. Mirrors prefix_loss.py logic."""
import numpy as np, sys
import os; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from order_numpy import risk_order as ref_risk, conf_order as ref_conf, softmax as ref_soft   # archive 27 sealed
from prefix_eval import risk_order, conf_order, softmax, prefix_metrics

rng = np.random.default_rng(0)
N, K, T = 3000, 20, 12
fut = np.cumsum(rng.normal(.4, .2, (N, T, 2)), 1)
paths = fut[:, None] + np.cumsum(rng.normal(0, .15, (N, K, T, 2)), 2) + rng.normal(0, .8, (N, K, 1, 2))
logits = rng.normal(size=(N, K))
# 1) orders identical to archive 27 reference
assert (risk_order(paths[:, :, -1], softmax(logits)) == ref_risk(paths[:, :, -1], ref_soft(logits))).all()
assert (conf_order(logits) == ref_conf(logits)).all()
# 2) prefix-winner rule: k=K reproduces global WTA winner; nested minima are monotone
err = np.linalg.norm(paths - fut[:, None], axis=-1); q = err.mean(2) + .25 * err[:, :, -1]
o = risk_order(paths[:, :, -1], softmax(logits)); qo = np.take_along_axis(q, o, 1)
win = {k: np.take_along_axis(o, qo[:, :k].argmin(1)[:, None], 1)[:, 0] for k in (1, 3, 5, 20)}
assert (win[20] == q.argmin(1)).all()
assert (win[1] == o[:, 0]).all()
qk = {k: q[np.arange(N), win[k]] for k in win}
assert (qk[1] >= qk[3] - 1e-12).all() and (qk[3] >= qk[5] - 1e-12).all() and (qk[5] >= qk[20] - 1e-12).all()
# 3) eval: @20 equal across orders, nested monotone
m = prefix_metrics(paths, logits, fut)
assert np.allclose(m["conf_minFDE20"], m["riskp_minFDE20"]) and np.allclose(m["riskp_minFDE20"], m["minFDE"])
for o_ in ("conf", "riskp"):
    assert (m[f"{o_}_minFDE1"] >= m[f"{o_}_minFDE3"]).all() and (m[f"{o_}_minFDE3"] >= m[f"{o_}_minFDE5"]).all()
print("distinct share (j*_k != global winner) on synthetic:", {k: float((win[k] != win[20]).mean()) for k in (1, 3, 5)})
print("ALL NUMPY CHECKS PASSED")
