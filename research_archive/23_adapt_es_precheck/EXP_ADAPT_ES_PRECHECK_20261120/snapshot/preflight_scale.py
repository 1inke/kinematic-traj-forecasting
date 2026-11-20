"""A single archived-decoder equivalence check before CPU analysis."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
WIDTH = ROOT.parent/"EXP_WIDTH_CAUSE_DIAG_EXEC_20261120"
sys.path.insert(0,str(WIDTH))

import numpy as np
import torch
from run_width import OLD, arrays, check, load_baseline, as_numpy, read, write
from width_model import h3_forward_diag, model_for, prediction, tensors, world_paths


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device="cuda:0"
    source = read(WIDTH/"PREFLIGHT.json")["files"]
    provenance={}
    for s in ("fit","dev"):
        f=OLD/f"C0/{s}.npz"
        check(f,source[str(f)],provenance)
    data={s:arrays(OLD/f"C0/{s}.npz") for s in ("fit","dev")}
    path,old=load_baseline(42,data,provenance)
    model,_=model_for(path,device,42)
    d=data["dev"]
    picked=np.sort(np.random.default_rng(20261120).choice(len(d["ids"]),200,replace=False))
    err={str(x):0. for x in (.4,.8,1.,1.4)}
    capture={}

    def hook(module,args):
        capture["decoder_inputs"]=args

    h=model.decoder.register_forward_pre_hook(hook)
    try:
        for idx in np.array_split(picked,4):
            _,_,_,tf=prediction(model,*tensors(d,idx,device))
            x,p=capture["decoder_inputs"]
            last=d["past"][idx,0,-1].astype(np.float64)[:,None,None]
            for a in (.4,.8,1.,1.4):
                out,_,_=h3_forward_diag(model.decoder,x,p,scale_mult=p.new_full((len(p),1),a))
                pred=as_numpy(world_paths(out[:,:1],tf))
                expected=last+a*(old["dev"][0][idx]-last)
                diff=float(np.max(abs(pred-expected)))
                assert diff<=1e-4,(a,diff)
                err[str(a)]=max(err[str(a)],diff)
    finally:
        h.remove()
    write(ROOT/"SCALE_EQUIVALENCE.json",dict(status="PASS",device=device,seed=42,n=len(picked),
                                          dev_indices=picked.tolist(),errors_m=err,files=provenance))
    print(err,flush=True)


if __name__=="__main__":
    main()
