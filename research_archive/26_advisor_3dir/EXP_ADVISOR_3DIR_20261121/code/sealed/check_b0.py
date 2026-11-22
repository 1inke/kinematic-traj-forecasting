"""Verify mounted Colab inputs and run two frozen-configuration B0 forward windows."""
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np


def check(root):
    for bundle in ("c0_inputs","baseline_sources"):
        manifest = json.loads((root/f"{bundle}.manifest.json").read_bytes())
        for name,digest in manifest["members"].items():
            assert sha256((root/name).read_bytes()).hexdigest()==digest, name
    base = root/"qoder_work/NEXT_PHASE_EXEC_20261114"
    sys.path.insert(0,str(base))
    import common
    common.ASTRA = root/"ASTRA_ACCURACY_Q20"
    from train_core import infer, make_model, torch

    torch.set_num_threads(1)
    for split,n in (("fit",21719),("dev",4753)):
        with np.load(base/f"C0/{split}.npz",allow_pickle=False) as raw:
            assert len(np.unique(raw["ids"]))==n
            assert raw["past"].shape==(n,9,8,2) and raw["future"].shape==(n,12,2)
            if split=="dev":
                data = {k:raw[k][:2] for k in ("past","num_valid","ids")}
    model, _ = make_model("cpu", 42)
    count = sum(p.numel() for p in model.parameters())
    assert count == 295474, count
    paths,logits = infer(model,data,batch=2)
    assert paths.shape==(2,20,12,2) and logits.shape==(2,20)
    assert np.isfinite(paths).all() and np.isfinite(logits).all()
    print(json.dumps(dict(status="PASS", device="cpu", parameters=count, forward_windows=len(paths),
                          torch=torch.__version__, numpy=np.__version__)))


if __name__=="__main__":
    check(Path(__file__).resolve().parent)

