"""Common paths, artifact receipts and read-only experimental inputs."""

import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
OLD = ROOT.parent / "NEXT_PHASE_EXEC_20261114"
PREC = ROOT.parent / "EXP_PREC_DIAG_EXEC_20261119"
HISTORY = ROOT.parent / "EXP_PREC_HISTORY_DIAG_EXEC_20261119"
DEC = ROOT.parent / "EXP_DEC_INFO_PROBE_EXEC_20261117"
PROBE = ROOT.parent / "EXP_SET_INTERACT_PROBE_EXEC_20261117"
sys.path[:0] = [str(OLD), str(PREC), str(HISTORY), str(DEC / "code"), str(PROBE / "code")]

from probe_io import arrays, now, npz, read, save_torch, seal, sha, verify, write


def status(phase, **fields):
    write(ROOT / "CURRENT.json", dict(at=now(), pid=os.getpid(), phase=phase, **fields))


def anchored(path, manifest):
    digest = manifest["files"][str(path)]
    assert sha(path) == digest, path
    return digest


def finish(folder, files, **info):
    write(folder / "DONE.json", dict(status="SUCCESS", at=now(), files=seal(files), **info))


def finished(folder):
    receipt = folder / "DONE.json"
    if not receipt.exists():
        return False
    result = read(receipt)
    assert result["status"] == "SUCCESS"
    verify(result["files"])
    return True


def observation(past):
    from run_history_diag import observation_masks
    masks = observation_masks(past[:, 0].astype(np.float64))
    return np.select([masks["stationary"], masks["steady"]], ["stationary", "steady"], default="other")


def populations(data):
    from run_prec_diag import groups, maneuver_labels
    behavior = maneuver_labels(data["past"], data["future"])
    history = observation(data["past"])
    masks = groups(data, behavior)
    masks.update({f"obs_{g}": history == g for g in ("stationary", "steady", "other")})
    assert [int(masks[f"obs_{g}"].sum()) for g in ("stationary", "steady", "other")] == [1109, 1991, 1653]
    return {k: v for k, v in masks.items() if v.any()}, behavior, history
