import common as c,model as m,metrics as v
import numpy as np,torch,time,gc

def save_job(prefix,net,data,ix,arm,source):
    output=c.H/(prefix+'_paths.npz');ap=c.H/(prefix+'_aux.npz');jp=c.H/(prefix+'_metrics.json');dp=c.H/(prefix+'_done.json');output.parent.mkdir(parents=True,exist_ok=True)
    if dp.exists():r=c.read(dp);c.valid_done(r,source);return r
    assert not any(p.exists() for p in (output,ap,jp)),'PRESERVE_PARTIAL_OUTPUT_STOP'
    if net is None:z=dict(paths=data['native'][ix],native=data['native'][ix])
    else:
        before={k:t.detach().cpu().clone() for k,t in net.state_dict().items()};p,raw,u,r=m.generated(net,data,ix,arm);m.exact(before,{k:t.detach().cpu() for k,t in net.state_dict().items()});z=dict(paths=p,raw=raw,controls=u,correction=r,native=data['native'][ix])
    z.update({k:data[k][ix] for k in ('sample_id','gt','groups','track','scene','sequence')});met,aux=v.evaluate(z);aux.update({k:z[k] for k in ('sample_id','groups','track')});np.savez_compressed(output,**z);np.savez_compressed(ap,**aux);c.write(jp,met)
    files={str(p):c.sha(p) for p in (output,ap,jp)};row=dict(status='PASS',source_digest=source,arm=arm,n=len(ix),output=str(output),aux=str(ap),metrics=str(jp),files=files);c.write(dp,row);return row

def main():
    start=time.perf_counter();pre=c.read(c.H/'PREFLIGHT.json');assert pre['status']=='PASS' and pre['planning_minutes']<=30
    files=c.read(c.H/'source_manifest.json')['files'];source=c.digest(files);assert source==pre['source_digest'];c.stage('U0_INITIAL_SOURCE_SHA');c.verify(files);data=c.load();st=c.npz(c.H/'statistics.npz');jobs={};runs={}
    for split,ix in data['split'].items():
        c.stage('U0_NATIVE_SAVE',split=split);jobs['native/'+split]=save_job('native/'+split,None,data,ix,'native20',source)
    for seed in m.SEEDS:
        for arm in m.ARMS:
            directory=c.H/f'runs/{arm}/seed{seed}';directory.mkdir(parents=True,exist_ok=True);done=directory/'done.json';key=f'{arm}/seed{seed}'
            if done.exists():row=c.read(done);c.valid_done(row,source);runs[key]=row;jobs.update(row['jobs']);continue
            manifest=dict(source_digest=source,statistics_SHA=c.sha(c.H/'statistics.npz'),arm=arm,seed=seed,epochs=25,batch=256,split_ID_digest=c.digest(data['ids'][data['split']['FIT']].tolist()),optimizer='AdamW.001wd.0001foreachFalse',factor='temporal_operator',no_projection=True)
            m.seed(seed);net=m.Refiner(st).to(m.DEVICE);opt=m.optimizer(net);g=torch.Generator().manual_seed(seed);history=[];cp=directory/'checkpoint.pt'
            if cp.exists():history=m.restore(cp,net,opt,g,manifest)
            for e in range(len(history),25):
                c.stage('TRAIN',arm=arm,seed=seed,epoch=e,epochs=25,completed_runs=len(runs));history.append(m.epoch(net,opt,data,data['split']['FIT'],g,e+1,arm));m.checkpoint(cp,net,opt,g,history,manifest);c.write(directory/'history.json',history)
            assert len(history)==25 and all(x['n']==18909 and x['optimization_steps']==74 for x in history)
            c.write(directory/'run_manifest.json',manifest);runjobs={}
            for split,ix in data['split'].items():
                c.stage('EVALUATE_SAVE',arm=arm,seed=seed,split=split,completed_runs=len(runs));jk=f'{key}/{split}';row=save_job(f'runs/{arm}/seed{seed}/{split}',net,data,ix,arm,source);jobs[jk]=row;runjobs[jk]=row;c.write('JOBS.json',jobs)
            rf={str(p):c.sha(p) for p in (cp,directory/'history.json',directory/'run_manifest.json')}
            rf.update({p:h for r in runjobs.values() for p,h in r['files'].items()});row=dict(status='PASS',source_digest=source,files=rf,jobs=runjobs,checkpoint_SHA=c.sha(cp),epochs=25);c.write(done,row);runs[key]=row;c.write('RUNS.json',runs);del net,opt;gc.collect();torch.cuda.empty_cache()
    assert len(jobs)==28 and len(runs)==6;c.write('JOBS.json',jobs)
    from statistics_ci import compute
    c.stage('U0_PRIMARY_BOOTSTRAP');boot=compute();c.write('BOOTSTRAP.json',boot)
    from audit_saved import main as audit
    audit();c.stage('U0_FINAL_SOURCE_SHA');c.verify(files);output={p:h for r in jobs.values() for p,h in r['files'].items()};output.update({p:h for r in runs.values() for p,h in r['files'].items()});output[str(c.H/'bootstrap_draws.npz')]=c.sha(c.H/'bootstrap_draws.npz');c.verify(output)
    cr=next(x for x in boot['rows'] if x['comparison']=='coupled-native');accuracy=bool(cr['estimate'][0]>=.05 and cr['CI_family12'][0][0]>0 and np.all(cr['per_seed'][:,0]>0) and np.all(cr['per_seed'][:,1:3]<=0))
    cells=[c.read(c.H/f'runs/{m.ARMS[1]}/seed{s}/{split}_metrics.json') for s in m.SEEDS for split in data['split']];safe=all(r['motion_pass'] for cell in cells for r in cell.values());preserve=all(r['lost_rate_native_good']<=.01 for cell in cells for r in cell.values());protected=all(r['ADE']<=1.01*r['native_ADE'] and r['FDE']<=1.01*r['native_FDE'] for cell in cells for group,r in cell.items() if group!='change')
    decision=dict(status='ACCURACY_POTENTIAL_WITH_MOTION_COST' if accuracy and not safe else 'PROMISING_REQUIRE_CONFIRMATION' if accuracy and safe and preserve and protected else 'ACCURACY_OR_PROTECTION_NOT_ESTABLISHED',accuracy_potential=accuracy,all_motion_safe=safe,native_good_preserved_1pct=preserve,protected_ADE_FDE_1pct=protected,major_claimed=False,automatic_next_training=False)
    c.write('DECISION.json',decision);c.write('PRESERVATION.json',dict(status='PASS',source_count=len(files),inherited_S0=1863,source_digest=source,initial_final_unchanged=True));c.write('RESOURCE.json',dict(status='PASS',**c.RESOURCE));c.write('RESULT.json',dict(status='PASS_U0_UNPROJECTED_TEMPORAL_REFINEMENT',decision=decision,jobs=jobs,runs=runs,scientific_jobs=24,native_reference_jobs=4,histories=150,steps=11100,outputs_SHA=output,source_count=len(files),source_digest=source,elapsed_seconds=time.perf_counter()-start,new_formal_experiments=1));c.stage('SUCCESS',decision=decision['status'])

if __name__=='__main__':
    try:main()
    except Exception:c.failure('STOP_U0');raise
