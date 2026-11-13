import common as c,model as m,metrics as v
import numpy as np,torch,time,gc
def manifest(data,arm,seed,source):
    assert arm in m.ARMS and seed in m.SEEDS
    return dict(stage='I1_DIRECT_READOUT',arm=arm,seed=seed,source_digest=source,protocol_SHA=c.sha(c.H/'protocol.json'),statistics_SHA=c.sha(c.H/'statistics.npz'),initialization_SHA=c.read(c.H/'INITIALIZATIONS.json')[str(seed)],FIT_ID_digest=c.digest(data['ids'][data['split']['FIT']].tolist()),objective='DIRECT_READOUT_MSE_B_K20_T12_SUMXY',output='native_float64+raw_float32_to_float64_UNBOUNDED',inputs=146,parameters=28632,epochs=25,batch=256,history='static_last_velocity' if arm=='S' else 'true_observed_velocity_acceleration',GT_groups_ID_logits_not_forward=True,old_F16_not_loaded=True,optimizer=dict(name='AdamW',lr=.001,wd=.0001,betas=[.9,.999],eps=1e-8,foreach=False))
def saved(row):
    pieces=[]
    for p in row['chunks']:c.guard();pieces.append(c.npz(p))
    out={k:np.concatenate([z[k] for z in pieces]) for k in pieces[0]};return out
def save_job(prefix,net,data,ix,arm,source):
    folder=c.H/prefix;done=folder/'done.json';folder.mkdir(parents=True,exist_ok=True)
    if done.exists():row=c.read(done);c.valid_done(row,source);return row
    assert not list(folder.glob('chunk*.npz')),'PARTIAL_OUTPUT_PRESERVED_STOP'
    before=None if net is None else {k:x.detach().cpu().clone() for k,x in net.state_dict().items()};paths=[]
    for start in range(0,len(ix),2048):
        ii=ix[start:start+2048];z={k:data[k][ii] for k in ('native','past','sample_id','gt','groups','track','scene','sequence')}
        if net is None:z['paths']=z['native'] if arm=='native' else v.cv_paths(z['past'])
        else:z['paths'],z['raw']=m.generated(net,data,ii,arm)
        assert np.isfinite(z['paths']).all();p=folder/f'chunk{start//2048:03d}.npz';np.savez_compressed(p,**z);paths.append(str(p));del z;gc.collect()
    if net is not None:m.exact(before,{k:x.detach().cpu() for k,x in net.state_dict().items()})
    temp=dict(chunks=paths);z=saved(temp);metrics,aux=v.evaluate(z);ap=folder/'aux.npz';jp=folder/'metrics.json';np.savez_compressed(ap,**aux);c.write(jp,metrics)
    files={str(p):c.sha(p) for p in [*paths,ap,jp]};row=dict(status='PASS',source_digest=source,arm=arm,n=len(ix),chunks=paths,aux=str(ap),metrics=str(jp),files=files,eval_all_weights_buffers_EXACT=True,chunk_max_rows=2048);c.write(done,row);return row
def audit(data,files,source,jobs,runs):
    c.stage('I1_INDEPENDENT_SAVED_AUDIT');t=time.perf_counter();records=[]
    for key,row in jobs.items():
        c.stage('I1_SAVED_AUDIT_JOB',job=key,done=len(records),total=len(jobs));c.valid_done(row,source);z=saved(row);ii=data['split'][key.split('/')[-1]]
        for k in ('native','past','sample_id','gt','groups','track','scene','sequence'):assert np.array_equal(z[k],data[k][ii]),('ID_PAYLOAD',key,k)
        v.validate(z,c.read(row['metrics']),c.npz(row['aux']));records.append(dict(job=key,n=len(ii),PASS=True));del z;gc.collect()
    for key,row in runs.items():
        c.valid_done(row,source);folder=c.H/'runs'/key;state=torch.load(folder/'checkpoint.pt',weights_only=False,map_location='cpu');hist=c.read(folder/'history.json');m.exact(state['history'],hist);assert len(hist)==25 and sum(x['optimization_steps'] for x in hist)==1850
        assert all(x['n']==18909 and x['objective']=='DIRECT_READOUT_MSE' for x in hist);assert all(torch.isfinite(x).all() for x in state['model'].values());assert state['model']['native_identity_flag'].dtype==torch.bool
        st=c.npz(c.H/'statistics.npz');m.exact(state['model']['mean'],torch.from_numpy(st['mean']));m.exact(state['model']['std'],torch.from_numpy(st['std']));assert state['manifest']==c.read(folder/'run_manifest.json')
    import bootstrap
    c.stage('I1_INDEPENDENT_20K_BOOTSTRAP');c.compare(c.read(c.H/'STATISTICS.json'),bootstrap.compute(True));c.verify(files)
    c.write('ARITHMETIC_AUDIT.json',dict(status='PASS',jobs=len(jobs),science_jobs=24,reference_jobs=8,models=6,history_rows=150,steps=11100,records=records,complete_quality_original_GTfree_clusters_five_motion_MSE_and_20k_CI=True,no_second_model_or_gradient_replay=True,elapsed_seconds=time.perf_counter()-t))
def main():
    start=time.perf_counter();pre=c.read(c.H/'PREFLIGHT.json');assert pre['status']=='PASS' and pre['planning_minutes']<=30;c.guard('startup');files=c.read(c.H/'source_manifest.json')['files'];source=c.digest(files);assert source==pre['source_digest'];c.stage('I1_INITIAL_SOURCE_SHA');c.verify(files);data=c.load();st=c.npz(c.H/'statistics.npz');jobs={};runs={}
    assert np.array_equal(st['training_ids'],data['ids'][data['split']['FIT']])
    for arm in ('native','CV'):
        for split,ii in data['split'].items():jobs[arm+'/'+split]=save_job('references/'+arm+'/'+split,None,data,ii,arm,source);c.write('JOBS.json',jobs)
    for seed in m.SEEDS:
        for arm in m.ARMS:
            key=f'{arm}/seed{seed}';folder=c.H/'runs'/key;folder.mkdir(parents=True,exist_ok=True);dp=folder/'done.json'
            if dp.exists():row=c.read(dp);c.valid_done(row,source);runs[key]=row;jobs.update(row['jobs']);continue
            contract=manifest(data,arm,seed,source);m.seed(seed);net=m.Readout(st).to(m.DEVICE);assert m.state_SHA(net)==contract['initialization_SHA'];opt=m.optimizer(net);g=torch.Generator().manual_seed(seed);cp=folder/'checkpoint.pt';hist=[]
            if cp.exists():hist=m.restore(cp,net,opt,g,contract)
            for e in range(len(hist),25):
                c.stage('I1_TRAIN',arm=arm,seed=seed,epoch=e,epochs=25,completed_runs=len(runs));hist.append(m.epoch(net,opt,data,data['split']['FIT'],g,e+1,arm));m.checkpoint(cp,net,opt,g,hist,contract);c.write(folder/'history.json',hist)
            assert len(hist)==25;c.write(folder/'run_manifest.json',contract);np.savez_compressed(folder/'statistics.npz',**st);np.savez_compressed(folder/'split_ids.npz',**{k:data['ids'][ii] for k,ii in data['split'].items()});rj={}
            for split,ii in data['split'].items():
                c.stage('I1_EVALUATE_SAVE',arm=arm,seed=seed,split=split);jk=key+'/'+split;row=save_job('runs/'+jk,net,data,ii,arm,source);jobs[jk]=row;rj[jk]=row;c.write('JOBS.json',jobs)
            rf={str(p):c.sha(p) for p in [cp,folder/'history.json',folder/'run_manifest.json',folder/'statistics.npz',folder/'split_ids.npz']};rf.update({p:h for row in rj.values() for p,h in row['files'].items()});row=dict(status='PASS',source_digest=source,files=rf,jobs=rj,epochs=25,steps=1850);c.write(dp,row);runs[key]=row;c.write('RUNS.json',runs);del net,opt;gc.collect();torch.cuda.empty_cache()
    assert len(runs)==6 and len(jobs)==32
    import bootstrap
    c.stage('I1_PRIMARY_BOOTSTRAP');stats=bootstrap.compute();c.write('STATISTICS.json',stats);audit(data,files,source,jobs,runs);c.stage('I1_FINAL_SOURCE_SHA');c.verify(files);outputs={p:h for row in jobs.values() for p,h in row['files'].items()};outputs.update({p:h for row in runs.values() for p,h in row['files'].items()});outputs[str(c.H/'bootstrap_draws.npz')]=c.sha(c.H/'bootstrap_draws.npz');c.verify(outputs)
    c.guard();c.write('PRESERVATION.json',dict(status='PASS',source_count=len(files),source_digest=source,initial_final_unchanged=True,old_models_code_results_untouched=True));c.write('RESOURCE.json',dict(status='PASS',**c.RESOURCE));c.write('RESULT.json',dict(status='PASS_I1_DIRECT_INFORMATION_READOUT',models=6,history_rows=150,steps=11100,science_jobs=24,reference_jobs=8,jobs=jobs,runs=runs,outputs_SHA=outputs,source_digest=source,source_count=len(files),statistics=stats,elapsed_seconds=time.perf_counter()-start,baseline_eligible=False,manual_review_pending=True));c.stage('SUCCESS',INFORMATION_READOUT_SIGNAL=stats['INFORMATION_READOUT_SIGNAL'])
if __name__=='__main__':
    try:main()
    except Exception:c.failure('STOP_I1');raise
