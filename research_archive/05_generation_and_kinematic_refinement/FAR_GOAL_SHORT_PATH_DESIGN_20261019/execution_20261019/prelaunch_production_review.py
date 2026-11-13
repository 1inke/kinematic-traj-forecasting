"""Exercise the ACTUAL production manifest/save/H2/skip branches before OS launch.

Only new prelaunch_smoke directory receives writes. Locked scientific files and
all prior digests verified before/after. Not independent external review.
"""
import copy,time,traceback
import common as c
c.E=c.read('MEMORY_ESTIMATE.json')['estimated_total_peak_bytes']
import numpy as np
import torch
import model as m
import training as tr
import diagnose_goals as dg
import run_series as run
import contracts
import evaluate_saved as ev

def main():
    begin=time.perf_counter();home=c.H;sm=c.read('source_manifest.json');c.verify(sm['files']);pre=c.read('PREFLIGHT.json');assert pre['status']=='PASS' and pre['source_digest']==sm['digest']
    d=c.load();h=c.npz(home/'H0_labels.npz');d.update(m20=h['m20'],y20=h['y20']);stats=c.npz(home/'shared_statistics.npz');ix=d['split']['outer'][:128]
    smoke=home/'prelaunch_smoke';assert not smoke.exists();smoke.mkdir()
    # Explicit data copies, not aliases masking a missing production input.
    for key in ('protocol.json','shared_statistics.npz'):
        import shutil
        shutil.copyfile(home/key,smoke/key)
    c.H=smoke;run.SOURCE=sm['digest'];run.JOBS={};manifests=[];reviews=[]
    try:
        paired={}
        for H,arm in [(12,'A'),(20,'B')]:
            m.setup(42);model=m.GoalConditionedShortPath(stats).cuda();initial=c.digest({k:c.array_sha(v.cpu().numpy()) for k,v in model.state_dict().items()})
            paired[arm]=initial
            for seed in (42,43,44):
                man=run.manifest(arm,H,seed,d,stats,initial);assert man['H']==H and man['arm']==arm and man['FIT_m20_SHA']==c.array_sha(d['m20'][d['split']['FIT']]);assert man['statistics_SHA']==c.sha(home/'shared_statistics.npz')
                manifests.append(man)
            # Actual finite FIT update then frozen production prediction API;
            # no scientific outcome choice/benchmark model retained.
            p,v,y,y20,mask=tr.tensors(d,d['split']['FIT'][:256]);opt=m.optimizer(model);opt.zero_grad(set_to_none=True);loss,_=m.objective(model(p,v),y,y20,mask,H);loss.backward();opt.step()
            out=dg.predict(model,d,ix);normalkey=arm+'/normal/outer';done=run.save_job(normalkey,'outer',ix,out,d,H)
            assert contracts.check_done(done,sm['digest'],done['contract']);oldhash=c.sha(smoke/'jobs'/normalkey/'paths.npz');again=run.save_job(normalkey,'outer',ix,out,d,H);assert again==done and oldhash==c.sha(smoke/'jobs'/normalkey/'paths.npz')
            donor=dg.donors(d['ids'][ix],d['track'][ix])
            for condition in ('CV_goal','cross_track_goal','GT_goal_local_diagnostic'):
                if condition=='CV_goal':take=ix;normal=out;override=np.repeat((4.8*d['v0'][ix].astype(np.float32))[:,None],20,1);extra={'override_z':override}
                elif condition=='cross_track_goal':take=ix;normal=out;override=out['z'][donor];extra=dict(override_z=override,donor_id=d['ids'][ix][donor],donor_track=d['track'][ix][donor])
                else:
                    take=ix[d['m20'][ix]];target=d['gt'][take,-1] if H==12 else d['y20'][take];override=np.repeat(((12/H)*target).astype(np.float32)[:,None],20,1);normal={k:v[d['m20'][ix]] for k,v in out.items()};extra={'override_z':override}
                response=dg.predict(model,d,take,override);run.save_job(arm+'/'+condition+'/outer','outer',take,response,d,H,extra,normal)
            for key,dd in list(run.JOBS.items()):
                if not key.startswith(arm+'/'):continue
                f=smoke/'jobs'/key;x=c.npz(f/'paths.npz');q=c.npz(f/'quality.npz');record=c.read(f/'metrics.json');mt,aux=ev.evaluate(x['paths'],x['gt'],x['groups'],x['native'],x['sample_id'],ind=True);ev.compare(mt,record['metrics']);ev.compare(aux,q)
                assert abs(m.reconstruct(x['z'],x['process_coeff'],x['v0'])-x['paths']).max()<=2e-12
                if 'donor_id' in x:assert np.all(x['donor_track']!=x['track'])
            reviews.append(dict(arm=arm,H=H,actual_production_manifest=True,actual_save_skip_and_H2_branches=True,all_saved_quality_cluster_motion_independently_exact=True))
        assert paired['A']==paired['B'];assert len(run.JOBS)==8
        c.write('REVIEW.json',dict(status='PASS',reviews=reviews,paired_init_EXACT=True,manifests=manifests,smoke_jobs=8,not_scientific_jobs=True))
    finally:c.H=home
    c.verify(sm['files']);c.write('PRODUCTION_BRANCH_REVIEW.json',dict(status='PASS',at=c.now(),seconds=time.perf_counter()-begin,source_digest=sm['digest'],locked_sources_initial_final_exact=True,helper_SHA=c.sha(__file__),smoke_review_SHA=c.sha(smoke/'REVIEW.json'),actual_production_manifests_all6=True,actual_production_H2_save_skip_branches=8,not_full_training_or_external_review=True,resource=c.RESOURCE))
    print('PASS_ACTUAL_PRODUCTION_BRANCH_REVIEW',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        c.write('PRELAUNCH_FAILURE.json',dict(at=c.now(),traceback=traceback.format_exc(),resource=c.RESOURCE));raise
