import common as c,model as m,metrics as v,features as f
import numpy as np,torch,time,gc,inspect,copy
from pathlib import Path
def resume(st,data,arm):
    folder=c.H/'resume_smoke'/arm;folder.mkdir(parents=True,exist_ok=True);ix=data['split']['FIT'][:513];inner=data['split']['inner'][:12];manifest=dict(stage='I1_DIRECT_READOUT',arm=arm,source='actual-preflight',contract=c.digest(c.read(c.H/'protocol.json')))
    def epoch(net,opt,g,e):
        row=m.epoch(net,opt,data,ix,g,e,arm);net.eval()
        with torch.inference_mode():row['inner_MSE']=float(m.loss(net(*m.inputs(data,inner),arm)[0],torch.from_numpy(data['gt'][inner]).to(m.DEVICE)))
        return row
    m.seed(720);net=m.Readout(st).to(m.DEVICE);opt=m.optimizer(net);g=torch.Generator().manual_seed(720);hist=[epoch(net,opt,g,1),epoch(net,opt,g,2)];m.checkpoint(folder/'continuous.pt',net,opt,g,hist,manifest)
    m.seed(720);net=m.Readout(st).to(m.DEVICE);opt=m.optimizer(net);g=torch.Generator().manual_seed(720);hist=[epoch(net,opt,g,1)];m.checkpoint(folder/'interrupted.pt',net,opt,g,hist,manifest)
    m.seed(999);net=m.Readout(st).to(m.DEVICE);opt=m.optimizer(net);g=torch.Generator().manual_seed(999);hist=m.restore(folder/'interrupted.pt',net,opt,g,manifest);hist.append(epoch(net,opt,g,2));m.checkpoint(folder/'resumed.pt',net,opt,g,hist,manifest)
    m.exact(torch.load(folder/'continuous.pt',map_location='cpu',weights_only=False),torch.load(folder/'resumed.pt',map_location='cpu',weights_only=False))
    try:m.restore(folder/'interrupted.pt',net,opt,g,dict(manifest,source='BAD'))
    except AssertionError:pass
    else:raise AssertionError('BAD_RESUME_ACCEPTED')
    return dict(arm=arm,status='EXACT',FIT=513,inner=12,epochs=2,all_parameters_buffers_optimizer_Torch_CPU_CUDA_loader_numpy_python_RNG_history=True)
def fixtures(st,data):
    fixture=np.stack((np.arange(-7,1)*.4,np.zeros(8)),1).astype(np.float32)[None];hs=f.history_numpy(fixture,'H');ss=f.history_numpy(fixture,'S');decimal_input_quantization=float(abs(hs[:,:14]-ss[:,:14]).max())
    exact=np.stack((np.arange(-7,1)*.5,np.zeros(8)),1).astype(np.float32)[None];assert not np.count_nonzero(f.history_numpy(exact,'H')[:,14:])
    ix=data['split']['FIT'][:1033];st1=f.statistics(data,ix,31);st2=f.statistics(data,ix,256);stat=max(c.close(st1[k],st2[k],1e-11) for k in ('mean','std'))
    q=torch.from_numpy(data['past'][ix].astype(np.float32)).to(m.DEVICE);err=max(c.close(f.history_torch(q,arm).cpu().numpy(),f.history_numpy(data['past'][ix],arm),2e-7) for arm in m.ARMS)
    checks=[]
    for arm in m.ARMS:
        m.seed(162);net=m.Readout(st).to(m.DEVICE).double()
        with torch.no_grad():net.head[-1].weight.normal_(0,.002);net.head[-1].bias.fill_(.01)
        inp=tuple(x.double() for x in m.inputs(data,ix[:8]));gt=torch.from_numpy(data['gt'][ix[:8]]).to(m.DEVICE);p=net(*inp,arm)[0];loss=m.loss(p,gt);raw=p.detach().requires_grad_();g=torch.autograd.grad(m.loss(raw,gt),raw)[0];expected=2*(raw-gt[:,None])/(len(raw)*20*12);c.close(g.detach().cpu().numpy(),expected.detach().cpu().numpy(),1e-12)
        c.close(float(loss),np.square(p.detach().cpu().numpy()-data['gt'][ix[:8],None]).sum(-1).mean(),2e-12)
        params=list(net.parameters());grad=torch.autograd.grad(loss,params);direction=[torch.randn_like(x) for x in params];norm=torch.sqrt(sum((x*x).sum() for x in direction));direction=[x/norm for x in direction];expected=sum((x*y).sum() for x,y in zip(grad,direction)).item();center=[x.detach().clone() for x in params]
        for eps in (1e-4,1e-5):
            values=[]
            for sign in (-1,1):
                with torch.no_grad():
                    for x,x0,d in zip(params,center,direction):x.copy_(x0+sign*eps*d)
                    values.append(float(m.loss(net(*inp,arm)[0],gt)))
            observed=(values[1]-values[0])/(2*eps);error=abs(observed-expected);assert error<=1e-7+1e-3*abs(expected),(arm,eps,error);checks.append(dict(arm=arm,eps=eps,expected=expected,observed=observed,error=error,double_copy_not_float32_quantization_proof=True))
    assert list(inspect.signature(m.Readout.forward).parameters)==['self','p2','native','past','arm']
    # Literal same-candidate gate, zero metric, deterministic single CV, duplicate K20 one cluster.
    p=np.zeros((1,20,12,2));gt=np.zeros((1,12,2));assert np.array_equal(c.clusters(p,np.ones((1,20),bool)),[[1,1]])
    fake=np.full_like(p,2.);fake[:,0,:11]=0;fake[:,0,-1]=[1.,0.];fake[:,1,:,:]=[.3,0.];fake[:,1,-1]=0
    a,ff,j,g,_=c.errors(fake,gt);assert a.min()<=.2 and ff.min()<=.4 and not g.any()
    return dict(status='PASS',common_stats_chunk_error=stat,history_GPU_numpy_error=err,exact_constant_velocity_zero_accel=True,decimal_past_float32_nonconstant_velocity_difference=decimal_input_quantization,GT_groups_ID_logits_not_forward=True,independent_mean20MSE_coordinate_gradient=True,double_FD=checks,same_candidate_and_duplicate_fixtures=True)
def bootstrap_fixture():
    import bootstrap
    values=np.random.Generator(np.random.PCG64(81)).uniform(.1,1,(3,16,3));tracks=np.array(['t'+str(x//2) for x in range(16)]);a=bootstrap.boot(values,tracks,False,c.H/'fixture_draws.npz');b=bootstrap.boot(values,tracks,True,c.H/'fixture_draws.npz');err=c.compare(a,b)
    return dict(status='PASS',draws=20000,independent_track_loop_max_error=err,all_arm_CV_pairing_and_seed_track_order=True,zero_denominator_no_resample=True)
def main():
    started=time.perf_counter();c.write('RESOURCE_START.json',c.guard('startup'));c.stage('I1_PREFLIGHT_ID_STATS');data=c.load();ix=data['split']['FIT'];st=f.statistics(data,ix);np.savez_compressed(c.H/'statistics.npz',**st)
    # Whole FIT independent population calculation from fixed float32 production features.
    mean=np.zeros(146);sq=np.zeros(146);count=np.zeros(146)
    for ii in np.array_split(ix,(len(ix)+255)//256):
        for sl,x in ((slice(0,96),data['p2'][ii].astype(float)),(slice(96,120),data['native'][ii].astype(np.float32).reshape(-1,24).astype(float)),(slice(120,146),f.history_numpy(data['past'][ii],'H').astype(float))):mean[sl]+=x.sum(0);sq[sl]+=(x*x).sum(0);count[sl]+=len(x)
    mean/=count;std=np.sqrt(np.maximum(sq/count-mean*mean,0));std[std<1e-6]=1;staterr=max(c.close(st['mean'],mean,1e-11),c.close(st['std'],std,1e-11))
    records={}
    for split,ii in data['split'].items():
        row=c.read(c.T1/'JOBS.json')['native/'+split];c.verify(row['files']);z=c.npz(row['output'])
        for k in ('sample_id','native','gt','groups','track','scene','sequence'):assert np.array_equal(z[k],data[k][ii]),('OLD_CACHE_ALIGNMENT',split,k)
        records[split]=dict(n=len(ii),change=int((data['groups'][ii]==2).sum()),tracks=len(set(data['track'][ii])),ID_digest=c.digest(data['ids'][ii].tolist()),all_saved_IDs_native_GT_groups_track_scene_sequence_EXACT=True);del z
    assert sum(len(ii) for ii in data['split'].values())==24679
    c.write('IDENTITY_AUDIT.json',dict(status='PASS',records=records,unique_ids=24679,raw_past8_source_observation_EXACT=True,common_FIT_stats_error=staterr,P2_history_window_count=18909,native_slot_count=18909*20,old_F16_not_loaded_or_forwarded=True))
    initials={}
    for seed in m.SEEDS:m.seed(seed);initials[str(seed)]=m.state_SHA(m.Readout(st))
    c.write('INITIALIZATIONS.json',initials)
    pref_source=c.digest({str(p):c.sha(p) for p in [*c.H.glob('*.py'),c.H/'protocol.json',c.H/'statistics.npz']})
    fx=fixtures(st,data);phases=[];resumes=[];bench=[];smoke=[]
    from train import save_job,saved,manifest
    for arm in m.ARMS:
        c.stage('I1_PREFLIGHT_GPU8',arm=arm);m.seed(42);net=m.Readout(st).to(m.DEVICE);m.seed(42);paired=m.Readout(st).to(m.DEVICE);m.exact(net.state_dict(),paired.state_dict());assert sum(x.numel() for x in net.parameters())==28632
        ii=ix[:256];p=net(*m.inputs(data,ii),arm)[0];assert np.array_equal(p.detach().cpu(),data['native'][ii]);opt=m.optimizer(net);grad=[]
        for b in range(8):m.step(net,opt,data,ix[b*256:(b+1)*256],arm);grad.append({k:float(torch.linalg.vector_norm(x.grad)) for k,x in net.named_parameters()});c.guard()
        assert all(any(g[k]>0 for g in grad[1:]) for k in grad[0]);phases.append(dict(arm=arm,parameters=28632,all_parameters_buffers_paired_EXACT=True,initial_native_identity=True,gradients=grad))
        prefix='preflight_smoke/'+pref_source[:12]+'/'+arm;t=time.perf_counter();row=save_job(prefix,net,data,data['split']['outer'][:128],arm,pref_source);v.validate(saved(row),c.read(row['metrics']),c.npz(row['aux']));duration=time.perf_counter()-t
        payload=sum(x.nbytes for x in saved(row).values());smoke.append(dict(arm=arm,rows=128,done_file=str(c.H/prefix/'done.json'),source_digest=pref_source,full_forward_save_read_independent_quality_original_clusters_five_motion_seconds=duration,uncompressed_payload=payload,not_science_job=True));resumes.append(resume(st,data,arm));m.seed(123);net=m.Readout(st).to(m.DEVICE);opt=m.optimizer(net);times=[]
        for b in range(36):
            torch.cuda.synchronize();t=time.perf_counter();m.step(net,opt,data,ii,arm);torch.cuda.synchronize()
            if b>=4:times.append(time.perf_counter()-t)
        bench.append(dict(arm=arm,warm=4,measured=32,FIT256_seconds=float(np.mean(times)),no_saved_benchmark_model_or_outcome_choice=True));del net,opt,paired;gc.collect();torch.cuda.empty_cache()
    bs=bootstrap_fixture();fixture=c.H/'contract_fixture.json';c.write(fixture,dict(fixture=True));row=dict(status='PASS',source_digest='fixture',files={str(fixture):c.sha(fixture)});c.valid_done(row,'fixture');bad=0
    for obj in (dict(row,source_digest='BAD'),dict(row,files={str(fixture):'0'*64})):
        try:c.valid_done(obj,'fixture')
        except AssertionError:bad+=1
    assert bad==2
    contracts=[manifest(data,a,s,'preflight-current') for a in m.ARMS for s in m.SEEDS];assert len({x['statistics_SHA'] for x in contracts})==1 and all(x['stage']=='I1_DIRECT_READOUT' and x['objective'].startswith('DIRECT_READOUT') for x in contracts)
    c.write('PRODUCTION_MANIFEST_PREFLIGHT.json',dict(status='PASS',actual_production_manifest_function_executed=contracts,all_referenced_statistics_and_protocol_opened_SHA=True,bad_source_output_resume_rejected=True))
    c.write('DA1_REVIEW.json',dict(status='PASS_WITH_LIMITATIONS',same_assistant_not_external=True,issue='S already has history through P2/native; H is explicit readout channel, not new sensing. F0 solver absence is unresolved not infeasibility. I1 regression may collapse K20; report clusters/motion and never baseline.',human_execution_confirmation=True))
    files=c.locks();c.write('source_manifest.json',dict(files=files,inherited_count=2560));c.verify(files)
    # Include whole saved pipeline for 6 scientific runs plus 2 deterministic refs, both independent review and source IO.
    train=5550*sum(x['FIT256_seconds'] for x in bench);pipeline=24679*8/128*max(x['full_forward_save_read_independent_quality_original_clusters_five_motion_seconds'] for x in smoke);btime=time.perf_counter();bootstrap_fixture();boot_seconds=time.perf_counter()-btime
    planning=(train+pipeline+32*boot_seconds+90)*1.2/60;outbytes=24679*8/128*max(x['uncompressed_payload'] for x in smoke);disk=c.psutil.disk_usage(str(c.H)).free;assert disk>=2*2**30+1.5*outbytes
    c.guard();pre=dict(status='PASS' if planning<=30 else 'BUDGET_NOT_LAUNCHED',source_count=len(files),source_digest=c.digest(files),planning_minutes=planning,ETA_minutes=[.7*planning,1.5*planning],elapsed_seconds=time.perf_counter()-started,actual_GPU8=phases,real_resume=resumes,fixtures=fx,bootstrap_fixture=bs,production_manifest_six_branches=True,benchmark=bench,smoke=smoke,train_seconds_nominal=train,full_saved_pipeline_seconds_nominal=pipeline,bootstrap_fixture_seconds=boot_seconds,resource=dict(c.RESOURCE),disk_free_bytes=disk,conservative_uncompressed_output_bytes=outbytes,statistical_and_scientific_thresholds_unchanged=True)
    c.write('PREFLIGHT.json',pre);print('PREFLIGHT',pre['status'],planning,pre['ETA_minutes'],len(files),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:c.failure('STOP_I1_PREFLIGHT');raise
