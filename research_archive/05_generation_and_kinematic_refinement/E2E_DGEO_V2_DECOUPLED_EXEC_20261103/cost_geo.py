"""Actual raw-observation predictor cost, full attention contractions included."""
import collections
import time
from geo_common import *
from geo_factory import make_geo_model
from train_core_geo import torch,np,bf
from causal_dataset import load_recording,past_input,RECORDINGS
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import FlopCounterMode
from preflight import guard_module


class Operators(TorchDispatchMode):
    def __init__(self):super().__init__();self.ops=collections.Counter()
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        self.ops[str(func)]+=1
        return func(*args,**(kwargs or {}))


def main():
    torch.set_num_threads(1);start=time.perf_counter()
    rg=guard_module();guard=rg.Guard(read_json(RUN/'MEMORY_ESTIMATE.json')['GPU_E']);guard.check(force=True)
    data=dict(np.load(BASE/'C0/fit.npz'))
    catalog={r:load_recording(bf.ASTRA/'datasets/eth_ucy/labels/eth'/(r+'_train.txt'))[0] for r in RECORDINGS}
    def construct(batch):
        histories=[];counts=[]
        for identity in data['ids'][:batch]:
            r,p,s=identity.split('|');h,_,n,_=past_input(catalog[r],int(p),int(s)+7)
            histories.append(h);counts.append(n)
        return np.stack(histories),np.array(counts,dtype=np.int64)
    all_results={}
    for arm in ARMS:
        results={}
        for device,batch in [('cpu',1),('cuda',1),('cuda',64)]:
            model,_=make_geo_model(arm,device);model.eval()
            def once():
                raw,counts=construct(batch)
                pred,score,_,tr=bf.prediction(model,torch.tensor(raw,device=device),torch.tensor(counts,device=device))
                return bf.world_paths(pred,tr),score[:,0].argmax(-1)
            times=[]
            with torch.inference_mode():
                for i in range(550):
                    if device=='cuda':torch.cuda.synchronize()
                    t=time.perf_counter();once()
                    if device=='cuda':torch.cuda.synchronize()
                    if i>=50:times.append(time.perf_counter()-t)
                    if i%100==0:guard.check()
                old=torch.backends.mha.get_fastpath_enabled();torch.backends.mha.set_fastpath_enabled(False)
                try:
                    ops=Operators()
                    with ops,FlopCounterMode(display=False) as counter:once()
                finally:torch.backends.mha.set_fastpath_enabled(old)
            results[f'{device}_batch{batch}']=dict(warmup=50,measure=500,CPU_threads=1,
                p50_ms=float(np.quantile(times,.5)*1000),p95_ms=float(np.quantile(times,.95)*1000),
                mean_ms=float(np.mean(times)*1000),batch_num_valid=construct(batch)[1].tolist(),
                actual_ATen_FMA2=int(counter.get_total_flops()),MAC_equivalent=counter.get_total_flops()/2,
                counted_operators={str(k):int(v) for k,v in counter.flop_counts.get('Global',{}).items()},
                all_executed_operators=dict(ops.ops),
                definition='Raw causal lookup/padding+transfer+heading+whole fresh F16/h3/new factor/score+inverse/top1. Disk parse excluded. Attention contractions counted; nonlinear/NumPy scalar operations listed or timed but excluded from FLOP arithmetic. Not Linear-only or exact all-scalar instructions.',
                trace_only_MHA_fastpath_disabled=True)
            parameters=sum(p.numel() for p in model.parameters());del model
        all_results[arm]=dict(parameters=parameters,over_305000_target=parameters>305000,measurements=results)
        print('COST_ARM_PASS',arm,flush=True)
    guard.check(force=True)
    write_json(RUN/'COST_ACTUAL_PREDICTOR.json',dict(status='MEASURED_ALL4_ACTUAL_RAW_OBSERVATION_PIPELINES',
        at=now(),elapsed_seconds=time.perf_counter()-start,arms=all_results,resource=guard.observations,
        initial_fresh_not_final_accuracy=True,not_deployment_certificate=True))


if __name__=='__main__':main()

