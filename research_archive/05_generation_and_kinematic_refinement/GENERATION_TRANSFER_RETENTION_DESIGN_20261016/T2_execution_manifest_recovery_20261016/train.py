import common as c,model as m,metrics as v,objectives as o
import numpy as np,torch,time,gc

def save_job(prefix,net,data,ix,cell,source):
    output=c.H/(prefix+'_paths.npz');ap=c.H/(prefix+'_aux.npz');jp=c.H/(prefix+'_metrics.json');dp=c.H/(prefix+'_done.json');output.parent.mkdir(parents=True,exist_ok=True)
    if dp.exists():row=c.read(dp);c.valid_done(row,source);return row
    assert not any(p.exists() for p in (output,ap,jp)),'PRESERVE_PARTIAL_OUTPUT_STOP'
    if net is None:z=dict(paths=data['native'][ix],native=data['native'][ix]);pop=None
    else:
        before={k:x.detach().cpu().clone() for k,x in net.state_dict().items()};p,raw,u,r=m.generated(net,data,ix);m.exact(before,{k:x.detach().cpu() for k,x in net.state_dict().items()});z=dict(paths=p,raw=raw,controls=u,correction=r,native=data['native'][ix]);pop=v.population(net,data,ix,cell);m.exact(before,{k:x.detach().cpu() for k,x in net.state_dict().items()})
    z.update({k:data[k][ix] for k in ('sample_id','gt','groups','track','scene','sequence')});met,aux=v.evaluate(z);aux.update({k:z[k] for k in ('sample_id','groups','track')});np.savez_compressed(output,**z);np.savez_compressed(ap,**aux);c.write(jp,met)
    files={str(p):c.sha(p) for p in (output,ap,jp)};row=dict(status='PASS',source_digest=source,cell=cell,arm='native20' if net is None else 'integrated_acceleration',n=len(ix),output=str(output),aux=str(ap),metrics=str(jp),population=pop,files=files,weights_all_buffers_eval_EXACT=True);c.write(dp,row);return row

def main():
    started=time.perf_counter();pre=c.read(c.H/'PREFLIGHT.json');assert pre['status']=='PASS' and pre['planning_minutes']<=30;c.guard('startup');files=c.read(c.H/'source_manifest.json')['files'];source=c.digest(files);assert source==pre['source_digest'];c.stage('T2_INITIAL_SOURCE_SHA');c.verify(files);data=c.load();st=c.npz(c.H/'statistics_velocity.npz');prior=c.read(c.T1/'JOBS.json');jobs={k:r for k,r in prior.items() if k.startswith('A/') or k.startswith('native/')};runs={}
    assert len(jobs)==16
    for row in jobs.values():c.valid_done(row,c.read(c.T1/'PREFLIGHT.json')['source_digest'])
    c.write('JOBS.json',jobs)
    for seed in m.SEEDS:
        for cell in m.CELLS:
            key=f'{cell}/seed{seed}';folder=c.H/'runs'/key;folder.mkdir(parents=True,exist_ok=True);done=folder/'done.json'
            if done.exists():row=c.read(done);c.valid_done(row,source);runs[key]=row;jobs.update(row['jobs']);continue
            manifest=dict(source_digest=source,protocol_SHA=c.sha(c.H/'protocol.json'),statistics_SHA=c.sha(c.H/'statistics_velocity.npz'),cell=cell,representation='velocity',keep=cell=='B',seed=seed,epochs=25,batch=256,FIT_ID_digest=c.digest(data['ids'][data['split']['FIT']].tolist()),optimizer='AdamW.001wd.0001foreachFalse',kernel='integrated_acceleration',only_factor='native24_position_to_velocity',no_projection=True)
            m.seed(seed);net=m.Refiner(st).to(m.DEVICE);opt=m.optimizer(net);g=torch.Generator().manual_seed(seed);cp=folder/'checkpoint.pt';history=[]
            if cp.exists():history=m.restore(cp,net,opt,g,manifest)
            for e in range(len(history),25):
                c.stage('TRAIN',cell=cell,seed=seed,epoch=e,epochs=25,completed_runs=len(runs));history.append(m.epoch(net,opt,data,data['split']['FIT'],g,e+1,cell));m.checkpoint(cp,net,opt,g,history,manifest);c.write(folder/'history.json',history)
            assert len(history)==25 and all(r['n']==18909 and r['optimization_steps']==74 for r in history)
            c.write(folder/'run_manifest.json',manifest);np.savez_compressed(folder/'statistics.npz',**st);np.savez_compressed(folder/'split_ids.npz',**{s:data['ids'][ii] for s,ii in data['split'].items()});c.write(folder/'assignment_stats.json',[{k:r[k] for k in ('epoch','success','keep_active','g0_counts','winner_counts','keep_slot_counts','winner_keep_cross','component_ADE','component_FDE','component_tie')} for r in history]);runjobs={}
            for split,ii in data['split'].items():
                c.stage('EVALUATE_SAVE',cell=cell,seed=seed,split=split,completed_runs=len(runs));jk=key+'/'+split;row=save_job('runs/'+jk,net,data,ii,cell,source);jobs[jk]=row;runjobs[jk]=row;c.write('JOBS.json',jobs)
            c.write(folder/'metrics.json',{split:c.read(row['metrics']) for split,row in zip(data['split'],runjobs.values())})
            paths=[cp,*[folder/x for x in ('history.json','run_manifest.json','statistics.npz','split_ids.npz','assignment_stats.json','metrics.json')]];rf={str(p):c.sha(p) for p in paths};rf.update({p:h for row in runjobs.values() for p,h in row['files'].items()});row=dict(status='PASS',source_digest=source,files=rf,jobs=runjobs,checkpoint_SHA=c.sha(cp),epochs=25,steps=1850);c.write(done,row);runs[key]=row;c.write('RUNS.json',runs);del net,opt;gc.collect();torch.cuda.empty_cache()
    assert len(jobs)==28 and len(runs)==3;c.write('JOBS.json',jobs)
    import bootstrap
    c.stage('T2_PRIMARY_BOOTSTRAP');stats=bootstrap.compute();c.write('STATISTICS.json',stats)
    from evaluate_saved import audit
    audit(data,files,source,jobs,runs);c.stage('T2_FINAL_SOURCE_SHA');c.verify(files);outputs={p:h for row in jobs.values() for p,h in row['files'].items()};outputs.update({p:h for row in runs.values() for p,h in row['files'].items()});outputs[str(c.H/'bootstrap_draws.npz')]=c.sha(c.H/'bootstrap_draws.npz');c.verify(outputs)
    c.write('PRESERVATION.json',dict(status='PASS',source_count=len(files),inherited_T1=2304,T1_resource_deviation_preserved=True,source_digest=source,initial_final_unchanged=True,old_models_code_results_untouched=True));c.guard();c.write('RESOURCE.json',dict(status='PASS',**c.RESOURCE));c.write('RESULT.json',dict(status='PASS_T2_TRANSFER_RETENTION',jobs=jobs,runs=runs,science_jobs=24,native_jobs=4,models=3,history_rows=75,steps=5550,reused_A_models=3,reused_A_jobs=12,outputs_SHA=outputs,statistics=stats,source_digest=source,source_count=len(files),elapsed_seconds=time.perf_counter()-started,manual_review_pending=True,new_formal_experiments=1));c.stage('SUCCESS',TRANSFER_SIGNAL=stats['decision']['TRANSFER_SIGNAL'])

if __name__=='__main__':
    try:main()
    except Exception:c.failure('STOP_T2');raise
