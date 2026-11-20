"""Extract sealed local labels before the NumPy-only CPU job."""
from pathlib import Path
import sys

import torch
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"EXP_PREC_DIAG_EXEC_20261119"))
from run_prec_diag import read, sha, write


def main():
    folder=ROOT.parent/"EXP_SCORE_COUPLING_PROBE_EXEC_20261120/cache"
    source=folder/"dev.pt"
    digest=read(folder/"DONE.json")["files"][source.name]
    assert sha(source)==digest
    saved=torch.load(source,map_location="cpu",weights_only=True)["target"][:,0]
    assert saved.shape==(4753,12,2) and saved.dtype==torch.float32
    out=ROOT/"PROBE_CACHED_TARGET.npz"
    np.savez_compressed(out,target=saved.numpy())
    write(ROOT/"TARGET_EXTRACT.json",dict(source=str(source),source_sha256=digest,
                                         output_sha256=sha(out),shape=list(saved.shape)))
    print("Extracted 4,753 sealed local evaluation targets on CPU.",flush=True)


if __name__=="__main__":
    main()
