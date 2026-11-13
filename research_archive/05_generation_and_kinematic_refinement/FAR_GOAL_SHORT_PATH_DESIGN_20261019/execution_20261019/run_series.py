"""Only this new finite H1/H2 series; never imports an old runner/model."""
import time,copy,traceback,math
import common as c
c.E=c.read('MEMORY_ESTIMATE.json')['estimated_total_peak_bytes']
import model as m
import training as tr
import evaluate_saved as ev
import diagnose_goals as dg
import contracts
import bootstrap_stats as st
import torch
import numpy as np

JOBS={};SOURCE=None
def save_job(key,split,ix,out,d,H=0,extra=None,normal=None):
    target=c.H/'jobs'/key;done=target/'done.json';contract=dict(key=key,split=split,H=H,ids_SHA=c.array_sha(d['ids'][ix]))
    if done.exists():
        dd=c.read(done);contracts.check_done(dd,SOURCE,contract);JOBS[key]=dd;return dd
    c.stage('EVALUATE_SAVE',job=key,completed=len(JOBS),planned=82)
    assert out['path'].shape==(len(ix),20,12,2) and np.isfinite(out['path']).all()
    payload=dict(sample_id=d['ids'][ix],paths=out['path'],gt=d['gt'][ix],native=d['native'][ix].astype(float),groups=d['groups'][ix],track=d['track'][ix],scene=d['scene'][ix],sequence=d['meta']['sequence'][ix],m20=d['m20'][ix],y20=d['y20'][ix],v0=d['v0'][ix].astype(np.float32),**{k:v for k,v in out.items() if k!='path'})
    if H:payload['physical_goal']=(np.float32(H/12)*out['z']).astype(np.float32)
    if extra:payload.update(extra)
    c.save(str(target.relative_to(c.H)/'paths.npz'),**payload)
    metrics,aux=ev.evaluate(payload['paths'],payload['gt'],payload['groups'],payload['native'],payload['sample_id'])
    c.save(str(target.relative_to(c.H)/'quality.npz'),**aux)
    strata={}
    # ALL remains primary; both long-label strata keep their original denominators.
    for label,mask in [('m20_true',payload['m20']),('m20_false',~payload['m20'])]+[('scene:'+str(s),payload['scene']==s) for s in np.unique(payload['scene'])]:
        if mask.any():strata[label]=summary_subset(payload,aux,mask)
        else:strata[label]=dict(n=0,not_measured=True)
    record=dict(metrics=metrics,strata=strata,goals=dg.diagnostics(d,ix,out,H,normal) if H else None)
    c.write(target/'metrics.json',record)
    dd=dict(status='COMPLETE',source_digest=SOURCE,contract=contract,files={str(target/f):c.sha(target/f) for f in ('paths.npz','quality.npz','metrics.json')},n=len(ix),at=c.now())
    c.write(done,dd);JOBS[key]=dd;c.write('JOBS.json',JOBS);return dd

def summary_subset(payload,aux,mask):
    groups=payload['groups'][mask];a,f,j,g,_,_,_=ev.errors(payload['native'][mask],payload['gt'][mask]);base=g.any(1);vectors=aux['vectors'][mask];good=vectors[:,0].astype(bool)
    result={}
    for name,sub in c.masks(groups):
        if not sub.any():result[name]=dict(n=0);continue
        mot=ev.motion(payload['paths'][mask][sub]);nm=ev.motion(payload['native'][mask][sub]);result[name]=dict(n=int(sub.sum()),good=int(good[sub].sum()),native_good=int(base[sub].sum()),rescue=int((good[sub]&~base[sub]).sum()),lost=int((~good[sub]&base[sub]).sum()),coverage=float(good[sub].mean()),minADE=float(vectors[sub,1].mean()),minFDE=float(vectors[sub,2].mean()),J=float(vectors[sub,3].mean()),effective_good=float(aux['clusters'][mask][sub,1].mean()),motion=mot,native_motion=nm,motion_failed=[k for k in mot if mot[k]>1.01*nm[k]])
    return result

def manifest(arm,H,seed,d,stats,initial):
    return dict(arm=arm,H=H,seed=seed,source_digest=SOURCE,protocol_SHA=c.sha(c.H/'protocol.json'),statistics_SHA=c.sha(c.H/'shared_statistics.npz'),FIT_ids_SHA=c.array_sha(d['ids'][d['split']['FIT']]),FIT_m20_SHA=c.array_sha(d['m20'][d['split']['FIT']]),initialization_SHA=initial,parameters=sum(p.numel() for p in m.GoalConditionedShortPath(stats).parameters()),epochs=25,steps_per_epoch=74,optimizer=c.read('protocol.json')['optimizer'],representation='P2observed96 + observed v0 only; new goal-conditioned K20',dtype=c.read('protocol.json')['geometry_dtype'],native_identity=False)

def train_run(arm,H,seed,d,stats):
    folder=c.H/'runs'/f'{arm}_seed{seed}';folder.mkdir(parents=True,exist_ok=True);done=folder/'done.json';m.setup(seed);model=m.GoalConditionedShortPath(stats).cuda();init=c.digest({k:c.array_sha(v.detach().cpu().numpy()) for k,v in model.state_dict().items()});man=manifest(arm,H,seed,d,stats,init)
    opt=m.optimizer(model);gen=torch.Generator().manual_seed(seed);history=[];epoch=0;step=0
    if done.exists():
        dd=c.read(done);contracts.check_done(dd,SOURCE,man);cp=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False);assert cp['contract']==man and cp['epoch']==25 and cp['step']==1850;model.load_state_dict(cp['model']);return model
    if (folder/'checkpoint.pt').exists():
        cp=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False);assert cp['contract']==man;model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);rr=cp['rng'];rr['cpu']=rr['cpu'].cpu();rr['loader']=rr['loader'].cpu();rr['cuda']=[x.cpu() for x in rr['cuda']];tr.restore(rr,gen);history=cp['history'];epoch=cp['epoch'];step=cp['step']
    else:c.write(folder/'run_manifest.json',man)
    for ep in range(epoch,25):
        c.stage('TRAIN',arm=arm,seed=seed,epoch=ep,completed_jobs=len(JOBS));start=time.perf_counter();row=tr.epoch(model,opt,gen,d,d['split']['FIT'],H);step+=row['batches'];row.update(epoch=ep+1,global_step=step,seconds=time.perf_counter()-start,resource=c.RESOURCE.copy());history.append(row)
        # Epoch-boundary exact optimizer/all-RNG resume only; no best selector.
        cp=tr.snapshot(model,opt,gen,history,ep+1,step,man);tmp=folder/'checkpoint.tmp';torch.save(cp,tmp);c.os.replace(tmp,folder/'checkpoint.pt');c.write(folder/'history.json',history)
        text='\n'.join(c.json.dumps(c.safe(r),ensure_ascii=False,allow_nan=False) for r in history)+'\n';(folder/'history.jsonl').write_text(text,encoding='utf-8')
    assert len(history)==25 and step==1850
    c.write(done,dict(status='COMPLETE',source_digest=SOURCE,contract=man,files={str(folder/f):c.sha(folder/f) for f in ('checkpoint.pt','run_manifest.json','history.json','history.jsonl')},at=c.now()))
    return model

def audit_all(d):
    start=time.perf_counter();records=[]
    for key,dd in JOBS.items():
        c.stage('INDEPENDENT_SAVED_AUDIT',job=key,completed=len(records),total=len(JOBS));contracts.check_done(dd,SOURCE);folder=c.H/'jobs'/key;payload=c.npz(folder/'paths.npz');quality=c.npz(folder/'quality.npz');stored=c.read(folder/'metrics.json')
        ids=payload['sample_id'];lookup={int(v):i for i,v in enumerate(d['ids'])};ix=np.array([lookup[int(v)] for v in ids]);assert np.array_equal(payload['gt'],d['gt'][ix]) and np.array_equal(payload['native'],d['native'][ix].astype(float)) and np.array_equal(payload['m20'],d['m20'][ix])
        if 'z' in payload:
            err=float(abs(m.reconstruct(payload['z'],payload['process_coeff'],payload['v0'])-payload['paths']).max());assert err<=2e-12
        else:err=0.
        mt,aux=ev.evaluate(payload['paths'],payload['gt'],payload['groups'],payload['native'],ids,ind=True);ev.compare(mt,stored['metrics']);ev.compare(aux,quality)
        if 'donor_id' in payload:assert np.all(payload['donor_track']!=payload['track'])
        records.append(dict(key=key,status='PASS',n=len(ids),reconstruction_max_error=err,all_complete_link_quality_and_five_motion=True))
    assert len(records)==82
    return dict(status='PASS',jobs=len(records),records=records,elapsed_seconds=time.perf_counter()-start,no_model_or_gradient_second_replay=True,not_external_review=True)

def primary_bootstrap(independent=False):
    vectors={};reference=None
    for arm in ('A','B'):
        vv=[]
        for seed in (42,43,44):
            f=c.H/'jobs'/f'{arm}_seed{seed}'/'normal'/'outer';x=c.npz(f/'paths.npz');q=c.npz(f/'quality.npz')
            if reference is None:reference=x
            else:assert np.array_equal(x['sample_id'],reference['sample_id']) and np.array_equal(x['track'],reference['track'])
            vv.append(q['vectors'])
        vectors[arm]=np.stack(vv)
    n=c.npz(c.H/'jobs/native/normal/outer/quality.npz')['vectors'];vectors['native']=np.repeat(n[None],3,axis=0)
    return st.bootstrap(vectors,reference['track'],reference['groups'],independent=independent)

def main():
    global SOURCE,JOBS
    start=time.perf_counter();c.guard('startup');sm=c.read('source_manifest.json');SOURCE=sm['digest'];assert c.digest(sm['files'])==SOURCE;c.stage('H1_INITIAL_SOURCE_SHA');c.verify(sm['files']);d=c.load();h=c.npz(c.H/'H0_labels.npz');d.update(y20=h['y20'],m20=h['m20']);stats=c.npz(c.H/'shared_statistics.npz');JOBS=c.read('JOBS.json') if (c.H/'JOBS.json').exists() else {}
    for name,ix in d['split'].items():save_job('native/normal/'+name,name,ix,dict(path=d['native'][ix].astype(float)),d)
    paired={}
    for seed in (42,43,44):
        for arm,H in [('A',12),('B',20)]:
            model=train_run(arm,H,seed,d,stats);man=c.read(c.H/'runs'/f'{arm}_seed{seed}'/'run_manifest.json')
            if seed in paired:assert paired[seed]==man['initialization_SHA'],'PAIRED_INIT_MISMATCH'
            else:paired[seed]=man['initialization_SHA']
            for split,ix in d['split'].items():
                key=f'{arm}_seed{seed}/normal/{split}';folder=c.H/'jobs'/key
                if (folder/'done.json').exists():out={k:v for k,v in c.npz(folder/'paths.npz').items() if k in ('z','process_coeff','paths')};out['path']=out.pop('paths')
                else:out=dg.predict(model,d,ix)
                save_job(key,split,ix,out,d,H)
                if split=='FIT':continue
                donor=dg.donors(d['ids'][ix],d['track'][ix]);assert np.all(donor>=0),'H2_NO_DIFFERENT_TRACK_PARTNER'
                for condition in ('CV_goal','cross_track_goal','GT_goal_local_diagnostic'):
                    if condition=='GT_goal_local_diagnostic':
                        mask=d['m20'][ix];take=ix[mask];target=d['gt'][take,-1] if H==12 else d['y20'][take];override=np.repeat(((12/H)*target).astype(np.float32)[:,None,:],20,axis=1);normal={k:v[mask] for k,v in out.items()};extra=dict(override_z=override,diagnostic_only=np.array(True))
                    elif condition=='CV_goal':
                        take=ix;override=np.repeat((4.8*d['v0'][ix].astype(np.float32))[:,None,:],20,axis=1);normal=out;extra=dict(override_z=override,diagnostic_only=np.array(True))
                    else:
                        take=ix;override=out['z'][donor];normal=out;extra=dict(override_z=override,donor_id=d['ids'][ix][donor],donor_track=d['track'][ix][donor],diagnostic_only=np.array(True))
                    assert len(take)>0,'H2_EMPTY_LONG_LABEL_COHORT'
                    key=f'{arm}_seed{seed}/{condition}/{split}'
                    if not (c.H/'jobs'/key/'done.json').exists():response=dg.predict(model,d,take,override)
                    else:
                        z=c.npz(c.H/'jobs'/key/'paths.npz');response={k:z[k] for k in ('z','process_coeff')};response['path']=z['paths']
                    save_job(key,split,take,response,d,H,extra,normal)
            del model;torch.cuda.empty_cache()
    c.stage('PRIMARY_BOOTSTRAP_20000_FAMILY16');boot,draws=primary_bootstrap();c.write('PRIMARY_COMPARISONS.json',boot);c.save('PRIMARY_DRAWS.npz',**draws)
    audit=audit_all(d);c.write('ARITHMETIC_AUDIT.json',audit)
    c.stage('INDEPENDENT_BOOTSTRAP_RECOMPUTE');bi,di=primary_bootstrap(independent=True);ev.compare(boot,bi)
    for k,v in draws.items():assert np.allclose(v,di[k],atol=2e-12,rtol=0,equal_nan=True),k
    c.write('BOOTSTRAP_AUDIT.json',dict(status='PASS',draws=20000,family=16,all_saved_draws_recomputed=True,independent_trackloop=True))
    # Audit all checkpoint contracts without any second model forward/backward.
    model_files={}
    for seed in (42,43,44):
        for arm,H in [('A',12),('B',20)]:
            folder=c.H/'runs'/f'{arm}_seed{seed}';dd=c.read(folder/'done.json');contracts.check_done(dd,SOURCE);cp=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False);assert cp['epoch']==25 and cp['step']==1850 and len(cp['history'])==25;assert len(c.read(folder/'history.json'))==25;assert cp['contract']==dd['contract'];model_files.update(dd['files'])
    c.stage('FINAL_SOURCE_OUTPUT_SHA');c.verify(sm['files']);outputs={p:h for dd in JOBS.values() for p,h in dd['files'].items()};outputs.update(model_files);c.verify(outputs)
    c.write('PRESERVATION.json',dict(status='PASS',source_count=len(sm['files']),old_and_current_sources_initial_final_exact=True,output_files=len(outputs),models=6,history_rows=150,optimizer_steps=11100,all_old_modules_and_sources_unchanged=True));c.write('RESOURCE.json',dict(status='PASS',**c.RESOURCE))
    records={key:c.read(c.H/'jobs'/key/'metrics.json') for key in JOBS}
    result=dict(status='PASS_FAR_GOAL_SHORT_PATH_EXECUTION',at=c.now(),elapsed_seconds=time.perf_counter()-start,formal_studies_completed=1,models=6,epochs=150,steps=11100,jobs=len(JOBS),output_sha=outputs,source_digest=SOURCE,source_count=len(sm['files']),records=records,primary=boot,scientific_major_not_yet_manually_reviewed=True,manual_analysis_pending=True)
    c.write('RESULT.json',result);c.write('runtime_state.json',dict(status='COMPUTATION_COMPLETE_PENDING_ANALYSIS',at=c.now(),completed_jobs=len(JOBS),formal_studies_completed=1,source_digest=SOURCE,not_external_review=True))
