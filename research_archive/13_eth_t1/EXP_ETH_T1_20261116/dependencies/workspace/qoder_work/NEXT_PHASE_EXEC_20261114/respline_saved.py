"""E2-0 zero-training diagnostic: re-join SAVED B0 anchors with a cubic spline.

h3 paths pass exactly through their anchors at steps 4/8/12 (linear interpolation),
so the anchors can be read back from saved paths.  This script keeps those anchors,
re-interpolates them with the natural / clamped spline of spline_basis.py, and
writes new prediction npz files with identical logits and ids.  Compare them with
exchange_null.py (--ref original --method natural ... --method clamped ...).

It answers only: "with anchors fixed, what does the joining rule alone change?"
Anchors were trained for straight segments, so this is a lower bound on what a
trained spline decoder could do, not a test of E2.

usage: python -I respline_saved.py --data C0/dev.npz --pred B0_s42_dev.npz --outdir E2_0/
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spline_basis import apply_basis  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=False)
    data = dict(np.load(a.data, allow_pickle=False))
    pred = dict(np.load(a.pred, allow_pickle=False))
    order = {k: i for i, k in enumerate(pred["ids"].tolist())}
    idx = np.array([order[k] for k in data["ids"].tolist()])
    paths = pred["paths"][idx].astype(np.float64)
    logits = pred["logits"][idx]
    past = data["past"][:, 0].astype(np.float64)          # focal [N,8,2] world
    p0 = past[:, -1]
    v0 = past[:, -1] - past[:, -2]                         # metres per step
    anchors = paths[:, :, [3, 7, 11]] - p0[:, None, None]  # [N,20,3,2]
    # Integrity: saved h3 paths must be reproduced by the linear basis from these anchors.
    lin = apply_basis(anchors, np.broadcast_to(v0[:, None], anchors.shape[:2] + (2,)), "linear") + p0[:, None, None]
    err = float(np.abs(lin - paths).max())
    if err > 1e-4:
        raise ValueError(f"saved paths are not piecewise-linear through steps 4/8/12 (max err {err:.3g}); "
                         "not an h3 population")
    for variant in ("natural", "clamped"):
        new = apply_basis(anchors, np.broadcast_to(v0[:, None], anchors.shape[:2] + (2,)), variant) + p0[:, None, None]
        np.savez_compressed(out / f"respline_{variant}.npz", paths=new.astype(np.float64),
                            logits=logits, ids=data["ids"])
    np.savez_compressed(out / "respline_linear_check.npz", paths=lin, logits=logits, ids=data["ids"])
    print("linear reconstruction max error", err, "->", out)


if __name__ == "__main__":
    main()
