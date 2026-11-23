"""Complete the zero-training comparison from the historical Drive forecasts."""
import argparse
import json
from pathlib import Path

import numpy as np

from advisor_train import inputs
from prepare_inputs import digest
from run_b0 import write_json
from z0_posthoc import run


def execute(old,output):
    local = Path(__file__).resolve().parent/"data/Z0_LOCAL.json"
    done = json.loads(local.read_bytes())
    stage2 = old/"ADVISOR_FIVEFOLD_20261122_031129/stage2/runs"
    for fold in ("eth","zara01","hotel","zara02","univ"):
        (_, _),(path,_) = inputs(fold)
        with np.load(path,allow_pickle=False) as dev:
            for arm in ("B0","R2"):
                for seed in (42,43,44):
                    name = f"{arm}_{fold}_s{seed}"
                    if name in done:
                        continue
                    folder = stage2/name
                    receipt = json.loads((folder/"RESULT.json").read_bytes())
                    p = folder/"dev_predictions.npz"
                    sha = digest(p)
                    assert sha==receipt["artifacts"][p.name], name
                    row = run(dev,folder)
                    row["source_sha256"] = sha
                    done[name] = row
                    print("Z0",name,flush=True)
    assert len(done)==45
    share = float(np.mean([done[f"B0_s{s}"]["kinematic"]["share_candidates_positive"] for s in (42,43,44)]))
    write_json(output,dict(status="PASS",runs=done,B0_C0_kinematic_share=share,MC5=share>=.05))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--old",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    execute(args.old,args.out)
