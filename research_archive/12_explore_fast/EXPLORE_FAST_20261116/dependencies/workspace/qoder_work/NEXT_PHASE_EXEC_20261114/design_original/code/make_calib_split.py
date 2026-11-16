"""E1-B: carve a FIT-internal calibration split by whole target track (numpy only).

The training-budget decision must not use dev.  15% of FIT target tracks, chosen by
SHA256 of "<recording>|<track>|20261114" (sorted, lowest first), become CALIB; the rest
FIT85.  Window ids are expected as "<recording>|<track>|<start>" (C0 format).
Writes index arrays only; never touches labels.

usage: python -I make_calib_split.py --fit C0/fit.npz --out E1/calib_split.npz
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

FRACTION, SALT = 0.15, "20261114"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fit", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    ids = np.load(a.fit, allow_pickle=False)["ids"]
    tracks = np.array(["|".join(str(x).split("|")[:2]) for x in ids])
    uniq = sorted(set(tracks.tolist()), key=lambda t: hashlib.sha256(f"{t}|{SALT}".encode()).hexdigest())
    calib_tracks = set(uniq[: int(np.ceil(FRACTION * len(uniq)))])
    calib = np.array([t in calib_tracks for t in tracks])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, fit85_index=np.flatnonzero(~calib), calib_index=np.flatnonzero(calib))
    print(json.dumps(dict(tracks=len(uniq), calib_tracks=len(calib_tracks),
                          fit85_windows=int((~calib).sum()), calib_windows=int(calib.sum()))))


if __name__ == "__main__":
    main()
