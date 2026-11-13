"""Additional prelaunch contract/statistics review; no optimizer or model forward."""
import common as c,model as m,objectives as o,bootstrap as b
import numpy as np,torch,ast,time

def main():
    started=time.perf_counter();pre=c.read(c.H/'PREFLIGHT.json');assert pre['status']=='PASS' and pre['planning_minutes']<=30;files=c.read(c.H/'source_manifest.json')['files'];c.verify(files);data=c.load();ii=data['split']['FIT'];maxref=0.;success=0
    for first in range(0,len(ii),256):
        c.guard();ix=ii[first:first+256];native=m.inputs(data,ix)[1];gt=torch.from_numpy(data['gt'][ix]).to(m.DEVICE)
        with torch.inference_mode():keep,d=o.keep_components(native.double(),native,gt)
        independent=o.numpy_components(data['native'][ix],data['native'][ix],data['gt'][ix]);assert keep.item()==0;assert np.array_equal(d['good0'].cpu().numpy(),independent['good0']);maxref=max(maxref,c.close(d['reference'].cpu().numpy(),independent['reference']));success+=int(d['valid'].sum())
    # Test the actual production bootstrap and independent population/track loops on
    # copies of saved GPU smoke outputs. Repeated fixture seeds are not new models.
    home=c.H;fixture=home/'prelaunch_bootstrap_fixture';fixture.mkdir(exist_ok=True)
    try:
        for cell in ('A','B'):
            paths=c.npz(home/'preflight_smoke'/cell/'paths.npz');aux=c.npz(home/'preflight_smoke'/cell/'aux.npz');aux.update({k:paths[k] for k in ('sample_id','groups','track')})
            for seed in m.SEEDS:
                folder=fixture/'runs'/cell/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
                for split in c.SIZES:np.savez_compressed(folder/(split+'_paths.npz'),**paths);np.savez_compressed(folder/(split+'_aux.npz'),**aux)
        c.H=fixture;prod=b.compute(False);ind=b.compute(True);err=c.compare(prod,ind)
        invalid=b.intervals(np.full((20000,1),np.nan),np.array([0.]),24)[0];assert invalid['valid_draws']==0 and not invalid['usable'] and invalid['CI_family'] is None
    finally:c.H=home
    # Run-completion branch is the real source's c.valid_done -> source + every SHA.
    tree=ast.parse((home/'train.py').read_text(encoding='utf-8'));assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='valid_done' for n in ast.walk(tree));assert all(x['status']=='EXACT' for x in pre['real_resume'])
    c.verify(files);review=dict(status='PASS',source_count=len(files),source_digest=c.digest(files),PREFLIGHT_SHA=c.sha(home/'PREFLIGHT.json'),all_FIT_native_G0_and_reference_Torch_numpy_checked=True,FIT_native_success=success,native_reference_max_error=maxref,actual_production_bootstrap_shared_draws_and_independent_trackloop_full20k_fixture_max_error=err,zero_denominator_draws_invalid_no_resample=True,fixture_seed_copies_not_independent_models=True,actual_runner_completed_SHA_branch_and_bad_source_output_preflight=True,inline_DA1='Keep could be general regularization or identity retreat; rescue noninferiority and fixed strict gates prevent calling lost-only improvement a generation breakthrough.',inline_DA2='All24 U1 jobs/96 group records including all20 counterevidence reviewed in U1_EVIDENCE_REVIEW. No all-directions-wrong or unique cause claim.',inline_DA3='Family24 not reduced, source locked, no adaptive checkpoint/keep coefficient. Saved subset resume is not full-FIT two epochs. New-head-only, reused populations, three seeds and motion failure limitations retained. Same assistant, not external review.',no_optimizer_or_model_forward=True,old_automation_unchanged=True,elapsed_seconds=time.perf_counter()-started)
    review['review_script_SHA']=c.sha(__file__);c.write('PRELAUNCH_REVIEW.json',review);print('PRELAUNCH_REVIEW PASS',review['elapsed_seconds'],maxref,err,flush=True)

if __name__=='__main__':
    try:main()
    except Exception:c.failure('STOP_T1_PRELAUNCH');raise
