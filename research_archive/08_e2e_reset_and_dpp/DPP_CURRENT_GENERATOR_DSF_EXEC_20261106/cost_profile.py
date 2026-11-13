"""Observed-array-to-world-path cost, explicitly not raw-neighbor-search cost."""
import time
from run_common import *
from current_h3_adapter import CurrentDSFPipeline

def main():
    torch.set_num_threads(1);reports=[]
    with np.load(BASE/'C0/fit.npz',allow_pickle=False) as f:past=torch.tensor(f['past'][:32]);valid=torch.tensor(f['num_valid'][:32])
    for device,batch in (('cpu',1),('cuda',1),('cuda',32)):
        adapter,_=load_current(device);model=CurrentDSFPipeline(adapter).to(device).eval();times=[]
        with torch.no_grad():
            for j in range(550):
                if device=='cuda':torch.cuda.synchronize()
                t=time.perf_counter();bundle=adapter.encode(past[:batch].to(device),valid[:batch].to(device));z=model.sampler(bundle['sampler_context']);world,logits=decode_logits(adapter,z,bundle);world.cpu();logits.cpu()
                if device=='cuda':torch.cuda.synchronize()
                if j>=50:times.append(time.perf_counter()-t)
        reports.append(dict(device=device,batch=batch,warmup=50,measure=500,mean_seconds=float(np.mean(times)),p95_seconds=float(np.quantile(times,.95)),scope='prepared causal neighbor array -> CPUtoGPU -> original normalization/F16/nativeh3 + DSF + controlledh3 -> world paths -> GPUtoCPU',raw_recording_IO_and_neighbor_search_included=False,not_complete_raw_stream_end_to_end=True))
    write(RUN/'COST_ACTUAL_ARRAY_PIPELINE.json',dict(status='MEASURED_EXPLICIT_SCOPE_NOT_RAW_STREAM_DEPLOYMENT',parameters=dict(frozen=295474,new_DSF=173440,whole=468914,old305000_target_pass=False),measurements=reports))

if __name__=='__main__':main()
