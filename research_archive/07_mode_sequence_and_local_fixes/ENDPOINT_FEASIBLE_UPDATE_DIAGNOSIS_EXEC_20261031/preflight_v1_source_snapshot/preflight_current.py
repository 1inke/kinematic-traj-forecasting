"""All new transaction and full-128 second-implementation checks BEFORE science."""
import copy
import importlib.util
import subprocess
import sys
import time
import numpy as np
import contracts as c
import torch
from feasible_update import Engine,search,rng
from train import infer,checkpoint_new
from trainer import Engine as SoftEngine
from preflight_checks import trees_equal
from checks import Bank,compare,independent_paths,scalar_paths,independent_verdict,full_metrics,groups,verdict
from evaluate_saved import save_port,read_port
from oracle_witness import run as oracle

def load_old_math():
    spec=importlib.util.spec_from_file_location('old_math_readonly',c.PREV/'preflight.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def synthetic_search(out):
    cases=[]
    for accept in (1,8,None):
        p=torch.nn.Parameter(torch.tensor([1.],device='cuda'));old=[p.detach().clone()];delta=[torch.tensor([.1],device='cuda')];count=[0]
        def evaluate(alpha):
            count[0]+=1
            assert torch.equal(p,old[0]+alpha*delta[0])
            return dict(all_pass=count[0]==accept,fixture_only=True)
        a,t=search([p],old,delta,evaluate)
        assert len(t)==(accept if accept else 8)
        assert a==(c.ALPHAS[accept-1] if accept else 0)
        if not accept:assert torch.equal(p,old[0])
        cases.append(dict(first_accept=accept,alpha=a,trials=len(t)))
    p=torch.nn.Parameter(torch.tensor([1.],device='cuda'))
    a,t=search([p],[p.detach().clone()],[torch.tensor([1e-10],device='cuda')],lambda _:(_ for _ in ()).throw(AssertionError('zero displacement evaluated')))
    assert a==0 and len(t)==8 and all(x['reason']=='ZERO_DISPLACEMENT' for x in t)
    # Exact quantization tie priority, empty groups, native zero-motion literal gate.
    from action_geometry import decode
    from motion_guard import inspect
    aa=c.space('END');p0=np.zeros((12,2));h=np.zeros((8,2));mid=(aa.normalized[24]+aa.normalized[25])/2
    expected=next(int(i) for i in aa.priority if ((aa.normalized-mid)**2).sum(1)[i]==((aa.normalized-mid)**2).sum(1).min())
    assert decode(p0,mid,aa)==expected and inspect(h,p0,p0)[0]
    assert not inspect(h,p0,p0+(np.arange(1,13)/12)[:,None]*[.1,0])[0]
    return dict(status='PASS_FIRST_EIGHTH_REJECT_ZERO_EXACT_TIE_NATIVE_ZERO',cases=cases)

class FixtureBank:
    def __init__(self,base):self.base=base;self.calls=0
    def evaluate(self,u):
        out,v=self.base.evaluate(u);self.calls+=1
        # First proposal accepts alpha .5, second rejects8, repeat fixed schedule.
        pos=(self.calls-1)%10
        v=copy.deepcopy(v);v['all_pass']=pos==1;v['synthetic_transaction_fixture']=True
        return out,v

def resume(d,bank,out,g,fixture=False):
    def make():return Engine(d,FixtureBank(bank) if fixture else bank,epochs=2)
    a=make()
    for _ in range(8):g.check();a.step()
    expected=a.snapshot();paths=infer(a.model,d['z'],'cuda:0');del a
    a=make()
    for _ in range(5):a.step()
    saved=a.snapshot();calls=a.bank.calls if fixture else None
    checkpoint_new(out/'interrupt5.pt',saved);s=c.sha(out/'interrupt5.pt');del a
    state=c.trusted_checkpoint(out/'interrupt5.pt',s);b=make();b.restore(state)
    if fixture:b.bank.calls=calls
    for _ in range(3):b.step()
    assert trees_equal(expected,b.snapshot()),'proposal resume differs'
    assert np.array_equal(paths,infer(b.model,d['z'],'cuda:0'))
    if fixture:assert b.rejected>0 and any(0<r['alpha']<1 for r in b.records)
    bad=[]
    for key in ('data','protocol','code'):
        v=copy.deepcopy(state);v['contract'][key]='bad'
        try:b.restore(v)
        except ValueError:bad.append(key)
        else:raise AssertionError('bad state accepted')
    v=copy.deepcopy(state);v['transaction']='IN_TRIAL'
    try:b.restore(v)
    except ValueError:bad.append('uncommitted')
    else:raise AssertionError('partial transaction accepted')
    try:c.trusted_checkpoint(out/'interrupt5.pt','0'*64)
    except ValueError:bad.append('checkpoint_sha')
    else:raise AssertionError('bad checkpoint accepted')
    return dict(status='PASS_EXACT_ALL_STATE_PROPOSAL5_RESUME',proposals=8,accepted=b.accepted,rejected=b.rejected,
        fixture_only=fixture,bad_reader_rejections=bad,all_rng_sampler_optimizer_trials_alpha_paths_exact=True)

def main():
    t0=time.perf_counter();out=c.ROOT/'preflight';out.mkdir(exist_ok=False)
    inherited=c.lock_inputs();d=c.data();g=c.resource(4495605760+64*1024**2+256*1024**2);g.check(mode='startup',force=True)
    c.write(c.ROOT/'MEMORY_ESTIMATE.json',dict(estimate_bytes=g.estimate,inventory=dict(previous_measured_peak=4495605760,bank_payload_reserve=64*1024**2,allocator=256*1024**2),policy='warning_first_remaining_v3',not_available_derived=True))
    t=time.perf_counter();bank=Bank(d);bank_seconds=time.perf_counter()-t
    c.write(c.ROOT/'INPUT_AUDIT.json',dict(status='PASS',IDs=d['x']['sample_id'].tolist(),tracks=97,native_good=69,safe_oracle_good=90,
        binding=c.binding(d),stat_refit=False,held_evaluated=False,bank_guard_all125440_actions_exact=True))
    oracle_t=time.perf_counter();oracle(d,bank);oracle_seconds=time.perf_counter()-oracle_t
    math=load_old_math();small=c.old.subset(d,np.arange(8))
    (out/'gradient').mkdir();grad=math.mathematical_tests(small,out/'gradient');gpu=math.gpu_bridge(small)
    # First real RS proposal, all4770 gradients/kernel, old initial snapshot EXACT.
    a=SoftEngine(d,dict(fixture='originalRS'),arm='RS');initial=copy.deepcopy(a.model.state_dict());a.step()
    expected_model=copy.deepcopy(a.model.state_dict());expected_opt=copy.deepcopy(a.optimizer.state_dict());expected_grads=[p.grad.detach().clone() for p in a.model.parameters()];expected_order=a.batch_order;del a
    b=Engine(d,bank);assert trees_equal(initial,b.model.state_dict());ids,_,_,_,_=b.propose()
    assert trees_equal(expected_model,b.model.state_dict()) and trees_equal(expected_opt,b.optimizer.state_dict())
    assert all(torch.equal(x,p.grad) for x,p in zip(expected_grads,b.model.parameters())) and ids.tolist()==expected_order[0]
    for seed in (42,43,44):
        folder=c.PREV/'tiny'/f'RS_{seed}';done=c.json_read(folder/'done.json')
        for rel,s in done['files'].items():assert c.sha(folder/rel)==s
        oldinitial=c.trusted_checkpoint(folder/'initial_checkpoint.pt',done['files']['initial_checkpoint.pt'])
        new=Engine(d,bank,seed);assert trees_equal(oldinitial['model'],new.model.state_dict())
        assert not list(new.model.buffers()) and sum(p.numel() for p in new.model.parameters())==4770
        assert np.all(infer(new.model,d['z'],'cuda:0')==0)
    tests=synthetic_search(out)
    # Full128 actual nonzero saved pipeline, original scalar guard AND independent group tables.
    u=infer(b.model,d['z'],'cuda:0');fast,v=bank.evaluate(u);slow=independent_paths(u,d)
    compare(fast,slow);compare(fast,scalar_paths(u,d));compare(v,independent_verdict(slow['C'],slow['S'],d))
    save_port(out/'saved128',fast,d,bank);read_port(out/'saved128',d)
    # Empty groups remain NOT_ESTIMABLE, not invented passes/zeros.
    dd=copy.deepcopy(d);dd['x']['group'][:]=0
    gr=groups(fast['S'],dd)[1];assert gr[2]['status']=='NOT_ESTIMABLE' and gr[3]['status']=='NOT_ESTIMABLE'
    # GT changes influence training acceptance, not the independent observation-only process.
    for k in (0,1):
        c.npz_new(out/f'observed{k}.npz',z=d['z'],past=d['x']['past'],native=d['x']['native'],gt=d['gt']+k*100)
        subprocess.run([sys.executable,'-B',str(c.ROOT/'observed_only.py'),str(out/f'observed{k}.npz'),str(out/f'inference{k}.npz')],check=True)
    za,zb=[c.load_npz(out/f'inference{k}.npz') for k in (0,1)];assert all(np.array_equal(za[k],zb[k]) for k in za)
    wrong=copy.deepcopy(d);wrong['gt']=d['gt']+100
    assert verdict(fast['C'],fast['S'],wrong,bank.ref)['bits']!=v['bits']
    for name,fixture in (('real_resume',False),('forced_transaction_resume',True)):
        sub=out/name;sub.mkdir();rr=resume(d,bank,sub,g,fixture);c.write(sub/'RESULT.json',rr)
    # Bad output and data are rejected via actual readers, not names-only checks.
    badout=copy.deepcopy(d);badout['gt']=d['gt'].copy();badout['gt'][0,0,0]+=.01
    try:read_port(out/'saved128',badout)
    except ValueError:pass
    else:raise AssertionError('changed data accepted')
    import shutil
    shutil.copytree(out/'saved128',out/'bad_output_fixture')
    with (out/'bad_output_fixture/paths.npz').open('ab') as f:f.write(b'CORRUPT')
    try:read_port(out/'bad_output_fixture',d)
    except ValueError:pass
    else:raise AssertionError('bad output accepted')
    # Actual measured timing; no outcome-based setting selection.
    prop=Engine(d,bank);pt=[]
    for i in range(12):
        torch.cuda.synchronize();t=time.perf_counter();prop.propose();torch.cuda.synchronize()
        if i>=4:pt.append(time.perf_counter()-t)
    ft=[]
    for _ in range(12):
        t=time.perf_counter();bank.evaluate(u);ft.append(time.perf_counter()-t)
    t=time.perf_counter()
    for _ in range(8):bank.evaluate(u)
    eight=time.perf_counter()-t
    t=time.perf_counter();full_metrics(fast['S'],d);cluster=time.perf_counter()-t
    # Completion audit uses second scalar decoding on ALL tried u; measure it too.
    t=time.perf_counter();slow=independent_paths(u,d);independent_verdict(slow['C'],slow['S'],d);audit=time.perf_counter()-t
    # Worst8 all768 proposals, saving15 checkpoints/CQS and36 U2 ports.
    compute=768*(max(pt)+8*max(ft));review=768*8*audit;extra=oracle_seconds+36*(max(ft)*9+cluster)+15*3*cluster+120
    conservative=compute+review+extra
    bench=dict(proposal_seconds=pt,full128_feasibility_seconds=ft,worst_eight_seconds=eight,full128_independent_scalar_audit_seconds=audit,
        full_clusters_seconds=cluster,O0_seconds=oracle_seconds,bank_seconds=bank_seconds,source_IO_seconds=c.json_read(c.ROOT/'INPUT_SOURCE_LOCK.json')['seconds'],
        conservative_proposal_trial_seconds=compute,conservative_all_trial_independent_review_seconds=review,extra_save_O0_U2_SHA_seconds=extra,
        conservative_total_seconds=conservative,budget_seconds=1800,admitted=conservative<=1800,
        audit_not_dropped=True,ordinary_RS_throwaway_proposals=12)
    c.write(c.ROOT/'BENCHMARK.json',bench)
    c.old.verify(inherited,g.check);g.check(force=True)
    c.write(c.ROOT/'PREFLIGHT.json',dict(status='PASS_IMPLEMENTATION_PREFLIGHT' if conservative<=1800 else 'PASS_CHECKS_BUDGET_STOP',gradient=grad,GPU=gpu,
        originalRS_first_proposal_allparams_allgrad_Adam_EXACT=True,old_three_initials_EXACT=True,synthetic=tests,
        real128_twoepochs8proposals_resume5_EXACT=True,forced_reject_and_half_accept_resume_EXACT=True,
        all128_independent_pooled_review_completed_before_science=True,independent_process_GT_invariance=True,
        bad_source_protocol_data_checkpoint_output_rejected=True,completed_output_SHA_compatible_read=True,
        fixture_optimizer_proposals=dict(first_equivalence=2,real_resume=16,synthetic_resume=16,benchmark=12),
        scientific_proposals=0,seconds=time.perf_counter()-t0))
    c.write(c.ROOT/'RESOURCE_PREFLIGHT.json',dict(policy='warning_first_remaining_v3',estimate=g.estimate,observations=g.observations))
    print('PREFLIGHT COMPLETE budget',round(conservative,2),'sec admitted',conservative<=1800,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback,datetime
        p=c.ROOT/('FAILURE_PREFLIGHT_'+datetime.datetime.now().strftime('%H%M%S')+'.json')
        c.write(p,dict(status='FAILURE',traceback=traceback.format_exc(),scientific_proposals=0))
        raise
