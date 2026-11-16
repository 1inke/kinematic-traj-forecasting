"""Full raw-observation pipeline timing, 50 warmup and 500 measured iterations."""
import argparse
import time
import numpy as np
from common import *
from train_core import *
from causal_dataset import load_recording,past_input,RECORDINGS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('name');ap.add_argument('arm');a=ap.parse_args()
    torch.set_num_threads(1)
    # Resolve the exact historical official TRAIN root, without opening any test labels.
    source=read_json(ROOT/'qoder_work/E2E_MOTION_RESEARCH_RESET_EXEC_20261102/source_manifest.json')['files']
    rawpaths={Path(p).stem.removesuffix('_train'):Path(p) for p in source if p.endswith('_train.txt')}
    catalog={r:load_recording(rawpaths[r])[0] for r in RECORDINGS}
    data=dict(np.load(HERE/'C0/fit.npz'));fixed=data['ids'][:64]
    run=HERE/'runs'/a.name
    if a.name=='OLD42':run=ROOT/'qoder_work/E2E_MOTION_RESEARCH_RESET_EXEC_20261102/B0_seed42'
    state=torch.load(run/'final.pt',map_location='cpu',weights_only=False)
    rows={}
    for device,batch in [('cpu',1),('cuda',1),('cuda',64)]:
        t=Trainer(data,np.arange(64),device=device,arm=a.arm);model=t.model
        model.load_state_dict(state['ema'],strict=True);model.eval()
        def once():
            past=[];counts=[]
            for identity in fixed[:batch]:
                r,p,s=str(identity).split('|');h,_,n,_=past_input(catalog[r],int(p),int(s)+7);past.append(h);counts.append(n)
            p=torch.tensor(np.stack(past),device=device);n=torch.tensor(counts,device=device)
            pred,logits,_,transform=prediction(model,p,n);world=world_paths(pred,transform)
            selected=logits[:,0].argmax(-1);return world[torch.arange(batch,device=device),selected]
        times=[]
        with torch.inference_mode():
            for i in range(550):
                if device=='cuda':torch.cuda.synchronize()
                start=time.perf_counter();once()
                if device=='cuda':torch.cuda.synchronize()
                if i>=50:times.append((time.perf_counter()-start)*1000)
        rows[f'{device}_batch{batch}']=dict(warmup=50,measure=500,p50_ms=float(np.quantile(times,.5)),p95_ms=float(np.quantile(times,.95)),
            batch_latency_not_per_window=True,raw_neighbor_heading_full_encoder_decoder_score_inverse_top1=True,
            excludes='One-time raw label parsing and offline GT evaluation',cpu_threads=1)
        del t,model;torch.cuda.empty_cache()
    write_json(HERE/'cost'/f'{a.name}.json',dict(name=a.name,arm=a.arm,at=now(),latency=rows))


if __name__=='__main__':main()
