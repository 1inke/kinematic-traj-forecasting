"""Actual full predictor operator and raw-observation timing; no cached tokens."""
import collections
import time
from common import *
from train_core import *
from causal_dataset import load_recording, past_input, RECORDINGS
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import FlopCounterMode


class Operators(TorchDispatchMode):
    def __init__(self):super().__init__();self.ops=collections.Counter()
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        self.ops[str(func)]+=1
        return func(*args,**(kwargs or {}))


def main():
    torch.set_num_threads(1)
    started=time.perf_counter()
    data=dict(np.load(HERE/'C0/fit.npz'))
    catalog={r:load_recording(ASTRA/'datasets/eth_ucy/labels/eth'/(r+'_train.txt'))[0] for r in RECORDINGS}
    fixed=data['ids'][:64]
    def construct(batch):
        histories=[];counts=[]
        for identity in fixed[:batch]:
            r,p,s=identity.split('|');h,_,n,_=past_input(catalog[r],int(p),int(s)+7)
            histories.append(h);counts.append(n)
        return np.stack(histories),np.array(counts,dtype=np.int64)
    results={};operator_profile={}
    for device,batch in [('cpu',1),('cuda',1),('cuda',64)]:
        model,cfg=make_model(device);model.eval()
        def once():
            raw,counts=construct(batch)
            p=torch.tensor(raw,device=device);n=torch.tensor(counts,device=device)
            pred,logits,_,tr=prediction(model,p,n)
            world=world_paths(pred,tr)
            selected=logits[:,0].argmax(-1)
            return world,selected
        measurements=[]
        with torch.inference_mode():
            for i in range(550):
                if device=='cuda':torch.cuda.synchronize()
                t=time.perf_counter();once()
                if device=='cuda':torch.cuda.synchronize()
                if i>=50:measurements.append(time.perf_counter()-t)
            # Disable MHA fast path only for trace coverage; deployments above use default.
            previous=torch.backends.mha.get_fastpath_enabled()
            torch.backends.mha.set_fastpath_enabled(False)
            try:
                ops=Operators()
                with ops, FlopCounterMode(display=False) as counter:
                    once()
            finally:torch.backends.mha.set_fastpath_enabled(previous)
        all_ops=dict(ops.ops)
        counted={str(op):int(v) for op,v in counter.flop_counts.get('Global',{}).items()}
        matrix_ops={k:v for k,v in all_ops.items() if any(s in k for s in ('mm.','matmul','addmm','convolution','attention'))}
        operator_profile[f'{device}_batch{batch}']={'observed_all_operators':all_ops,'matrix_and_attention_ops':matrix_ops,
            'counted_FLOPs':counted,'FMA2_total':int(counter.get_total_flops()),
            'MAC_equivalent':counter.get_total_flops()/2,
            'definition':'Actual executed ATen mm/addmm/bmm/conv/SDPA formulas with FMA=2; includes attention contractions, not only Linear hooks. Pointwise/nonlinear/NumPy neighbor ops are listed/excluded from FLOP total and included in measured latency. Not an exact count of all scalar instructions.',
            'profiling_fastpath_disabled_only_for_trace':True}
        results[f'{device}_batch{batch}']={'warmup':50,'measure':500,'cpu_threads':1,
            'p50_ms':float(np.quantile(measurements,.5)*1000),'p95_ms':float(np.quantile(measurements,.95)*1000),
            'mean_ms':float(np.mean(measurements)*1000),'GPU_sync':device=='cuda',
            'includes':'past-only neighbor construction from raw recording lookup, padding, transfer, heading/center, full fresh F16+h3+score, inverse heading, top1 selection',
            'excludes':'one-time raw disk label parse; offline GT clustering and statistical audits',
            'batch_num_valid':construct(batch)[1].tolist(),
            'graph_stats':{k:float(v) if torch.is_tensor(v) and v.numel()==1 else str(v) for k,v in vars(model.last_graph_stats).items()}}
        del model
    rg_path=ROOT/'qoder_work/K20_FINE_MODE_ERROR_DIAG_20261027/resource_guard_v3.py'
    import importlib.util
    spec=importlib.util.spec_from_file_location('cost_resource',rg_path);rg=importlib.util.module_from_spec(spec);spec.loader.exec_module(rg)
    write_json(HERE/'COST_ACTUAL_PREDICTOR.json',{'status':'MEASURED_FRESH_B0_ARCHITECTURE_INITIAL_WEIGHTS',
        'at':now(),'elapsed_seconds':time.perf_counter()-started,'parameters':295474,
        'trainable':294897,'unique_predictor_parameters_no_external_encoder_added':True,
        'latency':results,'operator_profile':operator_profile,'resource_snapshot':rg.snapshot(),
        'not_accuracy_or_deployment_success_certificate':True})
    print('COST_COMPLETE',time.perf_counter()-started,results,flush=True)


if __name__=='__main__':main()
