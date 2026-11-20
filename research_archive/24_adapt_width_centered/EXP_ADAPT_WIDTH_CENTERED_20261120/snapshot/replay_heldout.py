"""Replay the two sealed fit85 models and retain their dev forecasts locally."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
WIDTH=ROOT.parent/"EXP_WIDTH_CAUSE_DIAG_EXEC_20261120"
sys.path.insert(0,str(WIDTH))

import numpy as np
import torch

from run_width import OLD, arrays, as_numpy, check, read, save_npz, sealed, sha, write
from width_model import model_for, prediction, tensors, world_paths


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device="cuda:0"
    inputs={}
    old=read(WIDTH/"PREFLIGHT.json")["files"]
    data={}
    for phase in ("fit","dev"):
        p=OLD/f"C0/{phase}.npz"
        check(p,old[str(p)],inputs)
        data[phase]=arrays(p)
    split=OLD/"E1/calib_split.npz"
    sealed(split,OLD/"FINAL_OUTPUT_SEAL.json",inputs)
    part=arrays(split)
    idx=part["calib_index"]
    t=[x.rsplit("|",1)[0] for x in data["fit"]["ids"]]
    tracks=np.asarray(t)
    train,calib=part["fit85_index"],idx
    assert len(train)==18665 and len(calib)==3054
    np.testing.assert_array_equal(np.sort(np.r_[train,calib]),np.arange(len(t)))
    assert not set(tracks[train])&set(tracks[calib])
    outputs={}
    audits={}
    for seed in (42,43):
        run=OLD/f"runs/E1B_E64_s{seed}"
        path=run/"final.pt"
        contract=run/"contract.json"
        for p in (path,contract):
            sealed(p,run/"FINAL_LOCK.json",inputs)
        spec=read(contract)["spec"]
        assert spec["split"]=="fit85" and spec["epochs"]==64 and spec["seed"]==seed
        prior=run/"calib/predictions.npz"
        sealed(prior,run/"EVALUATION_DONE.json",inputs)
        saved=arrays(prior)
        np.testing.assert_array_equal(data["fit"]["ids"][idx],saved["ids"])
        model,_=model_for(path,device,seed)

        def batch(d,where):
            x,z,_,tf=prediction(model,*tensors(d,where,device))
            return as_numpy(world_paths(x,tf)),as_numpy(z[:,0])

        max_p=max_z=0.
        for i in range(0,len(idx),64):
            p,z=batch(data["fit"],idx[i:i+64])
            max_p=max(max_p,float(abs(p-saved["paths"][i:i+64]).max()))
            max_z=max(max_z,float(abs(z-saved["logits"][i:i+64]).max()))
        assert max(max_p,max_z)<=1e-4,(seed,max_p,max_z)
        dev=data["dev"]
        parts=[batch(dev,slice(i,i+64)) for i in range(0,len(dev["ids"]),64)]
        f=ROOT/f"E1B_dev_s{seed}.npz"
        save_npz(f,paths=np.concatenate([p for p,z in parts]),logits=np.concatenate([z for p,z in parts]),ids=dev["ids"])
        outputs[f.name]=sha(f)
        audits[str(seed)]=dict(calib_windows=len(idx),dev_windows=len(dev["ids"]),max_coordinate_m=max_p,
                               max_logit=max_z,checkpoint=inputs[str(path)])
        print(f"E1B {seed} CALIB replay max path {max_p}, logit {max_z}",flush=True)
    write(ROOT/"HELDOUT_REPLAY.json",dict(status="PASS",device=device,inputs=inputs,outputs=outputs,
                                        checks=audits,fit85_windows=len(train),calib_windows=len(calib),
                                        fit85_tracks=len(set(tracks[train])),calib_tracks=len(set(tracks[calib]))))


if __name__=="__main__":
    main()
