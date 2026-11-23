"""Z0 (zero-training, CPU): post-hoc risk ordering on saved archive-26 predictions.
Usage: python -I z0_posthoc.py --dev <dev.npz> --runs <run_dir> [<run_dir> ...] --out Z0.json
Each run_dir must hold dev_predictions.npz (paths (N,20,12,2) world metres, logits (N,20), ids).
For five-fold runs pass that fold's test.npz as --dev."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from order_numpy import risk_order, conf_order, softmax, kinematic_excess

def units(ids):
    if ids.dtype.kind in "US":            # C0: track = scene|track
        key = np.array(["|".join(s.split("|")[:2]) for s in ids]); return key, "track"
    return np.arange(len(ids)) // 15, "block15"   # five-fold: row ids only -> contiguous blocks

def boot(delta, base, unit, B=2000, seed=0):
    u, inv = np.unique(unit, return_inverse=True)
    s, c = np.bincount(inv, delta), np.bincount(inv, base)
    idx = np.random.default_rng(seed).integers(0, len(u), (B, len(u)))
    return (np.percentile(s[idx].sum(1) / c[idx].sum(1), [2.5, 97.5]) * 100).round(2).tolist()

def run(dev, rd):
    p = np.load(rd / "dev_predictions.npz", allow_pickle=True)
    paths, z, ids = p["paths"].astype(np.float64), p["logits"].astype(np.float64), p["ids"]
    if paths.ndim == 5: paths = paths[:, 0]
    assert (ids == dev["ids"]).all(), "window order differs from dev file"
    fut, past = dev["future"].astype(np.float64), dev["past"][:, 0].astype(np.float64)
    fde = np.linalg.norm(paths[:, :, -1] - fut[:, None, -1], axis=-1); n = np.arange(len(z))
    unit, kind = units(ids)
    w = softmax(z); orders = {"conf": conf_order(z), "risk_p": risk_order(paths[:, :, -1], w)}
    res = dict(windows=len(z), units=int(len(np.unique(unit))), unit=kind,
               top10_unit_share=float(np.sort(np.unique(unit, return_counts=True)[1])[::-1][:10].sum() / len(z)))
    for k in (1, 3, 5, 20):
        m = {o: fde[n[:, None], orders[o][:, :k]].min(1) for o in orders}
        res[f"min{k}"] = {o: float(v.mean()) for o, v in m.items()}
        if k < 20:
            res[f"min{k}"]["gain_pct"] = float((m["conf"].mean() - m["risk_p"].mean()) / m["conf"].mean() * 100)
            res[f"min{k}"]["ci95_pct"] = boot(m["conf"] - m["risk_p"], m["conf"], unit)
    kx = kinematic_excess(paths, past)
    res["kinematic"] = dict(share_candidates_positive=float((kx > 0).mean()),
                            share_windows_any_positive=float((kx > 0).any(1).mean()), mean=float(kx.mean()))
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dev", required=True)
    ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); dev = np.load(a.dev, allow_pickle=True)
    out = {Path(r).name: run(dev, Path(r)) for r in a.runs}
    Path(a.out).write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1)[:3000])
