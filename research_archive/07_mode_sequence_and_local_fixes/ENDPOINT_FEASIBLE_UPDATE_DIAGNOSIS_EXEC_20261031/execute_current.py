"""Finite three fresh models, no control retraining, no formal continuation."""
import time
import datetime
import json
import numpy as np
import contracts as c
import torch
from checks import Bank
from feasible_update import Engine
from train import infer,checkpoint_new
from preflight_checks import trees_equal
from evaluate_saved import save_port
from output_probe import probe

def main():
    t=time.perf_counter();pf=c.json_read(c.ROOT/'PREFLIGHT_V2.json');bench=c.json_read(c.ROOT/'BENCHMARK_V2.json')
    assert pf['status']=='PASS_IMPLEMENTATION_PREFLIGHT' and bench['admitted'] and bench['conservative_total_seconds']<=1800
    inputs=c.json_read(c.ROOT/'INPUT_SOURCE_LOCK.json')['files'];c.old.verify(inputs)
    source=dict(inputs)
    for p in c.ROOT.rglob('*'):
        if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts and 'failures' not in p.parts:source[str(p.resolve())]=c.sha(p)
    c.write(c.ROOT/'source_manifest.json',dict(status='PRODUCTION_SOURCE_LOCK',files=source,count=len(source)))
    d=c.data();bank=Bank(d);g=c.resource(c.json_read(c.ROOT/'MEMORY_ESTIMATE.json')['estimate_bytes']);g.check(force=True)
    c.write(c.ROOT/'PRELAUNCH_REVIEW.json',dict(status='PASS_ALL_REQUIRED_GATES_BEFORE_SCIENCE',PREFLIGHT=c.sha(c.ROOT/'PREFLIGHT_V2.json'),conservative_seconds=bench['conservative_total_seconds'],
        DA1=dict(verdict='PASS',issue='zero lost is forced, rescue is the actual finding',strongest_counterargument='finite local search may stall although feasible networks exist'),
        no_auto_formal=True,all_full128_independent_group_checks_before_first_scientific_proposal=True))
    proposals=[];rejections=[];allgroups=[]
    for seed in (42,43,44):
        folder=c.ROOT/f'F_RS_{seed}';folder.mkdir(exist_ok=False);e=Engine(d,bank,seed)
        olddone=c.json_read(c.PREV/'tiny'/f'RS_{seed}/done.json')
        oldinitial=c.trusted_checkpoint(c.PREV/'tiny'/f'RS_{seed}/initial_checkpoint.pt',olddone['files']['initial_checkpoint.pt'])
        assert trees_equal(e.model.state_dict(),oldinitial['model'])
        for point in (0,64,128,192,256):
            while e.updates<point:
                g.check();r=e.step()
                if e.updates%32==0:print('F-RS',seed,'proposal',e.updates,'accepted',e.accepted,'trials',len(e.trial_u),flush=True)
            c.atomic_checkpoint(folder/f'checkpoint_{point:03d}.pt',e.snapshot())
            z,v=bank.evaluate(infer(e.model,d['z'],'cuda:0'));assert v['all_pass']
            rr=save_port(folder/f'point_{point:03d}',z,d,bank,extra=dict(proposal=point,accepted=e.accepted,rejected=e.rejected))
            allgroups.extend([dict(method='F_RS',seed=seed,point=point,**x) for x in rr])
        # Batch clocks must reproduce the old RS experiment, not its accepted updates.
        oldstate=c.trusted_checkpoint(c.PREV/'tiny'/f'RS_{seed}/checkpoint.pt',olddone['files']['checkpoint.pt'])
        assert e.batch_order==oldstate['batch_order'] and trees_equal(e.sampler.get_state(),oldstate['sampler'])
        c.npz_new(folder/'all_trial_u.npz',u=np.stack(e.trial_u) if e.trial_u else np.empty((0,128,20,2),np.float32))
        records=[]
        for r in e.records:
            rr={k:v for k,v in r.items() if k!='preupdate_prediction'};records.append(rr)
            proposals.append(dict(seed=seed,**{k:r[k] for k in ('proposal','accepted','alpha','loss','unclipped_norm','delta_l2','optimizer_steps')},trials=len(r['trials'])))
            for tr in r['trials']:
                if not tr['all_pass']:rejections.append(dict(seed=seed,proposal=r['proposal'],alpha=tr['alpha'],changed=tr['changed'],u_index=tr.get('u_index'),bits=json.dumps(tr.get('bits',{})),reason=tr.get('reason','SCIENTIFIC_GATES')))
        c.write(folder/'proposal_history.json',records)
        c.npz_new(folder/'preupdate_trace.npz',u=np.stack([x['preupdate_prediction'] for x in e.records]),batch_indices=np.array(e.batch_order))
        for x in e.probes:
            n=x['proposal'];sub=c.ROOT/'U2_output_probe'/f'{seed}_{n:03d}';sub.mkdir(parents=True,exist_ok=False)
            c.npz_new(sub/'inputs.npz',u_current=x['u_current'],u_full_proposal=x['u_full_proposal'])
            out,v,alpha,trace=probe(x['u_current'],x['u_full_proposal'],d,bank)
            for label,oo in [('local',out),('alpha0',bank.decode(x['u_current'])),('alpha1',bank.decode(x['u_full_proposal']))]:
                rr=save_port(sub/label,oo,d,bank,extra=dict(GT_DIAGNOSTIC_ONLY=True,proposal=n,label=label))
                allgroups.extend([dict(method='U2_'+label,seed=seed,point=n,**q) for q in rr])
            c.write(sub/'CHOICE.json',dict(status='GT_DIAGNOSTIC_ONLY',alpha=alpha,trace=trace,global_verdict=v,not_shared_parameter_realizable=True))
        c.write(folder/'done.json',dict(status='ALL_256_PROPOSALS_COMMITTED_REVIEW_REQUIRED',proposals=e.updates,accepted=e.accepted,rejected=e.rejected,
            contract=c.contract(d),files={str(p.relative_to(folder)):c.sha(p) for p in folder.rglob('*') if p.is_file()}))
    c.csv(c.ROOT/'PROPOSALS.csv',proposals);c.csv(c.ROOT/'TRIAL_REJECTIONS.csv',rejections);c.csv(c.ROOT/'ALL_GROUPS.csv',allgroups)
    c.old.verify(source,g.check);g.check(force=True)
    c.write(c.ROOT/'RESOURCE_SCIENCE.json',dict(estimate=g.estimate,observations=g.observations))
    c.write(c.ROOT/'COMPUTATION.json',dict(status='ALL_SCIENTIFIC_PROPOSALS_AND_U2_SAVED_REVIEW_REQUIRED',proposals=768,new_models=3,formal_models=0,seconds=time.perf_counter()-t))
    print('ALL SCIENCE SAVED - REVIEW REQUIRED',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        c.write(c.ROOT/('FAILURE_EXECUTION_'+datetime.datetime.now().strftime('%H%M%S')+'.json'),dict(traceback=traceback.format_exc()))
        raise
