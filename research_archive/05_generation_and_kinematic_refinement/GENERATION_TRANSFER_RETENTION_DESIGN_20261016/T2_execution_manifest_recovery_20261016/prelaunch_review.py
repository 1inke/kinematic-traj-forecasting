"""Actual production-bootstrap/independent-trackloop test on saved fixtures. No optimizer/forward."""
import common as c,model as m,bootstrap as b
import numpy as np,ast,time

def main():
    started=time.perf_counter();pre=c.read(c.H/'PREFLIGHT.json');assert pre['status']=='PASS' and pre['planning_minutes']<=30;files=c.read(c.H/'source_manifest.json')['files'];c.verify(files)
    import production_contract
    production_manifest=production_contract.check(c.load(),c.digest(files),write_name='PRODUCTION_MANIFEST_PRELAUNCH.json')
    home=c.H;fixture=home/'prelaunch_bootstrap_fixture';fixture.mkdir(exist_ok=True)
    try:
        p=c.npz(home/'preflight_smoke/C/paths.npz');a=c.npz(home/'preflight_smoke/C/aux.npz');a.update({k:p[k] for k in ('sample_id','groups','track')})
        for cell in ('A','C'):
            for seed in m.SEEDS:
                folder=fixture/'runs'/cell/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
                for split in c.SIZES:np.savez_compressed(folder/(split+'_paths.npz'),**p);np.savez_compressed(folder/(split+'_aux.npz'),**a)
        c.H=fixture;b.FIXTURE=True;prod=b.compute(False);ind=b.compute(True);err=c.compare(prod,ind)
        invalid=b.intervals(np.full((20000,1),np.nan),np.array([0.]),24)[0];assert invalid['valid_draws']==0 and not invalid['usable'] and invalid['CI_family'] is None
    finally:c.H=home;b.FIXTURE=False
    # Real main's registered 20k draw-order construction must equal immutable T1 draws.
    old=c.npz(c.T1/'bootstrap_draws.npz');gen=np.random.Generator(np.random.PCG64(20261016));sd=np.empty_like(old['modelseed']);td=np.empty_like(old['tracks'])
    for first in range(0,20000,128):
        n=min(128,20000-first);sd[first:first+n]=gen.integers(0,3,(n,3));td[first:first+n]=gen.integers(0,len(old['track_order']),(n,len(old['track_order'])))
    assert np.array_equal(sd,old['modelseed']) and np.array_equal(td,old['tracks'])
    tree=ast.parse((home/'train.py').read_text());assert any(isinstance(n,ast.Attribute) and n.attr=='valid_done' for n in ast.walk(tree))
    assert c.read(home/'A_REUSE_CONTRACT.json')['status']=='PASS' and all(x['status']=='EXACT' for x in pre['real_resume'])
    c.verify(files);c.guard()
    c.write('PRELAUNCH_REVIEW.json',dict(status='PASS',source_count=len(files),source_digest=c.digest(files),PREFLIGHT_SHA=c.sha(home/'PREFLIGHT.json'),full20k_actual_production_and_independent_trackloop_fixture_max_error=err,actual_T1_seed_and_track_draw_order_EXACT=True,fixture_repeated_seeds_not_new_independent_models=True,zero_denominator_invalid_no_resample=True,registered_family24_9_3_unchanged=True,actual_completed_source_SHA_branch_review=True,inline_DA1='Reversible feature changes standardization/conditioning; not extra information or pure motion semantics.',inline_DA2='T1 keep rescues/retention tradeoff and historical resource deviation retained; C may not generalize and must not force D.',inline_DA3='D not triggered by T1; no new loss/old-model forward, no threshold relaxation. All checks by same assistant, not external review.',actual_production_manifest_construction=production_manifest,no_optimizer_or_model_forward=True,elapsed_seconds=time.perf_counter()-started))
    print('PRELAUNCH PASS',time.perf_counter()-started,err,flush=True)

if __name__=='__main__':
    try:main()
    except Exception:c.failure('STOP_T2_PRELAUNCH');raise
