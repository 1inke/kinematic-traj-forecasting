"""Explicit CPU-only supplemental timing, no root science source mutation."""
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import contracts as c
from checks import Bank
from output_probe import probe
from evaluate_saved import save_port

def main():
    d=c.data();b=Bank(d);z=c.load_npz(c.ROOT/'preflight_v2/saved128/paths.npz');u0=np.zeros_like(z['u']);u1=z['u'];times=[]
    for _ in range(4):
        t=time.perf_counter();out,v,alpha,trace=probe(u0,u1,d,b);times.append(time.perf_counter()-t)
    t=time.perf_counter();save_port(c.ROOT/'prelaunch_extension/saved_bundle_timing',out,d,b,extra=dict(benchmark_only=True));save=time.perf_counter()-t
    t=time.perf_counter();c.old.verify(c.json_read(c.ROOT/'INPUT_SOURCE_LOCK.json')['files']);hashio=time.perf_counter()-t
    old=c.json_read(c.ROOT/'BENCHMARK_V2.json')
    extra=old['O0_seconds']+12*max(times)+51*save+3*hashio+60
    total=old['conservative_proposal_trial_seconds']+old['conservative_all_trial_independent_review_seconds']+max(extra,old['extra_save_O0_U2_SHA_seconds'])
    c.write(c.ROOT/'prelaunch_extension/RESULT.json',dict(status='PASS_SUPPLEMENTAL_ACTUAL_U2_SAVE_SHA_TIMING' if total<=1800 else 'BUDGET_STOP',
        U2_full12windowwise_9alphas_seconds=times,U2_sample_population_windows=128,full128_CQS_clusters_save_audit_seconds=save,
        inherited_full_source_SHA_IO_seconds=hashio,conservative_total_seconds=total,budget=1800,admitted=total<=1800,
        base_PF_SHA=c.sha(c.ROOT/'PREFLIGHT_V2.json'),base_benchmark_SHA=c.sha(c.ROOT/'BENCHMARK_V2.json'),
        helper_SHA=c.sha(__file__),root_code=c.contract(d)['code'],
        field_note='BENCHMARK_V2 full128_independent_scalar_audit_seconds is vectorized direct second implementation, no bank; original scalar128 separately checked in preflight',
        optimizer_updates=0,scientific_proposals=0,no_model_forward=True))
    assert total<=1800
    print('FULL TIMING ADMISSION',total,'seconds',flush=True)
if __name__=='__main__':main()
