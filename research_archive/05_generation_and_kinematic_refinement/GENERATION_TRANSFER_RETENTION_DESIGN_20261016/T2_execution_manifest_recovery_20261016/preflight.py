import common as c,model as m,metrics as v,objectives as o
import numpy as np,torch,time,gc,inspect,copy,zipfile

def fixtures(st,data):
    ix=data['split']['FIT'][:8];p=torch.from_numpy(data['native'][ix]).to(m.DEVICE);gt=torch.from_numpy(data['gt'][ix]).to(m.DEVICE)
    lp=o.objective(p.clone().requires_grad_(),gt)[0];p1=p.clone().requires_grad_();g1=torch.autograd.grad(o.objective(p1,gt)[0],p1)[0];p2=p.clone().requires_grad_();g2=torch.autograd.grad(o.objective_independent_torch(p2,gt),p2)[0]
    c.close(g1.cpu().numpy(),g2.cpu().numpy(),1e-10);c.close(lp.item(),o.objective_numpy(p.cpu().numpy(),data['gt'][ix])[0])
    fixtures=[]
    cases=[]
    for kind in ('empty','single_worse','multiple','improved','boundary','all_zero','bad_better_than_good','mixed'):
        native=np.full((3,20,12,2),0.,float);native[...,0]=.5;pred=native.copy();y=np.zeros((3,12,2))
        if kind!='empty':native[:,0,:,0]=.1;pred[:,0,:,0]=.12
        if kind=='multiple':native[:,1,:,0]=.15;pred[:,1,:,0]=.11
        if kind=='improved':pred[:,0,:,0]=.08
        if kind=='boundary':
            # Exact endpoint-defined J=1; constant .2 has backend-dependent mean rounding.
            native[:,0,:,0]=.1;native[:,0,-1,0]=.4;pred[:,0,:,0]=.1;pred[:,0,-1,0]=.42
        if kind=='all_zero':native[:]=0;pred[:]=0
        if kind=='bad_better_than_good':pred[:,1,:,0]=0
        if kind=='mixed':native[1:]=.5;pred[1:]=.6
        n=torch.tensor(native,device=m.DEVICE);pp=torch.tensor(pred,device=m.DEVICE,requires_grad=True);gg=torch.tensor(y,device=m.DEVICE);kl,kd=o.keep_components(pp,n,gg);nn=o.numpy_components(pred,native,y)
        c.close(kl.item(),nn['keep']);assert np.array_equal(kd['slot'].detach().cpu(),nn['slot']);grad=torch.autograd.grad(kl,pp)[0];assert torch.isfinite(grad).all();assert torch.count_nonzero(grad[~kd['valid']])==0
        if kind in ('empty','all_zero','improved'):assert kl.item()==0 and torch.count_nonzero(grad)==0
        if kind=='bad_better_than_good':assert kd['slot'].tolist()==[0,0,0] and torch.count_nonzero(grad[:,1])==0
        fixtures.append(dict(case=kind,keep=kl.item(),batch_denominator=3,finite_gradient=True))
    # Exact ADE/FDE component tie: first11=.15625, endpoint=.34375 gives ADE=.171875=FDE/2.
    x=torch.zeros((1,2,12,2),device=m.DEVICE,dtype=torch.float64,requires_grad=True)
    with torch.no_grad():x[:,:,:11,0]=.15625;x[:,:,-1,0]=.34375
    jj=o.path_J(x,torch.zeros((1,12,2),device=m.DEVICE));gg=torch.autograd.grad(jj.sum(),x)[0]
    c.close(gg[0,:,0,0].cpu().numpy(),np.full(2,1/(12*.2)));assert torch.equal(jj[:,0],jj[:,1])
    # Independent representation and quantization contracts (C still not trained).
    from features import numpy_features,candidate_features
    p32=data['native'][data['split']['FIT'][:1033]].astype(np.float32);p64=p32.astype(float);prev=np.concatenate((np.zeros_like(p64[:,:,:1]),p64[:,:,:-1]),2);vd=(p64-prev)/.4;vp=numpy_features(p64,'velocity');rebuild=np.cumsum(vp.astype(float)*.4,axis=2);bound=8*np.finfo(np.float32).eps*.4*np.cumsum(abs(vd),axis=2)
    c.close(np.cumsum(vd*.4,axis=2),p64);assert np.all(abs(rebuild-p64)<=bound)
    assert np.array_equal(candidate_features(torch.from_numpy(p32).to(m.DEVICE),'velocity').cpu().numpy().reshape(vp.shape),vp)
    # Same production features, alternate accumulation chunking only.
    feats=np.concatenate((np.repeat(data['p2'][data['split']['FIT'][:1033]],20,0),vp.reshape(-1,24)),1).astype(float)
    sums=[]
    for chunk in (31,256):
        s=np.zeros(120);sq=np.zeros(120)
        for start in range(0,len(feats),chunk):s+=feats[start:start+chunk].sum(0);sq+=(feats[start:start+chunk]**2).sum(0)
        sums.append((s/len(feats),sq/len(feats)))
    err=max(c.close(a,b,1e-11) for a,b in zip(*sums))
    # Original A fixed nonzero output and all-parameter L0 gradient exact.
    namespace=dict(c=c,np=np,torch=torch,nn=m.nn,ARMS=m.ARMS,kernels=m.kernels)
    c.extract(c.PARENT/'model.py',('Refiner',),namespace);m.seed(162);net=m.Refiner(c.npz(c.T1/'statistics_position.npz'),representation='position').to(m.DEVICE);old=namespace['Refiner'](c.npz(c.T1/'statistics_position.npz')).to(m.DEVICE);old.load_state_dict(net.state_dict())
    with torch.no_grad():net.head[-1].weight.normal_(0,.002);net.head[-1].bias.fill_(.01)
    old.load_state_dict(net.state_dict());z=net(*m.inputs(data,ix))[0];zz=old(*m.inputs(data,ix),'integrated_acceleration')[0];assert torch.equal(z,zz)
    a=torch.autograd.grad(o.loss(z,m.inputs(data,ix)[1],gt,'A')[0],tuple(net.parameters()));b=torch.autograd.grad(o.objective(zz,gt)[0],tuple(old.parameters()));m.exact(a,b)
    # Fixed double-copy FD; record all active faces before either epsilon, no search.
    checks=[]
    for cell in m.CELLS:
        m.seed(162);net=m.Refiner(st).to(m.DEVICE).double()
        with torch.no_grad():net.head[-1].weight.normal_(0,.002);net.head[-1].bias.fill_(.01)
        p2,base=m.inputs(data,ix);p2=p2.double();base=base.double();gt=gt.double();out=net(p2,base)[0];loss,d=o.loss(out,base,gt,cell)
        def faces(out):
            e=torch.linalg.vector_norm(out-gt[:,None],dim=-1);aa=e.mean(-1)/.2;ff=e[...,-1]/.4;pair=torch.triu_indices(20,20,1,device=out.device);dist=torch.linalg.vector_norm(out[:,pair[0]]-out[:,pair[1]],dim=-1).mean(-1);_,dd=o.keep_components(out,base,gt)
            control_face=torch.linalg.vector_norm(torch.tanh(net(p2,base)[1]).double(),dim=-1)<=1
            return [o.objective(out,gt)[3],aa>=ff,dist<.3,dd['active'],dd['slot'],control_face]
        face=faces(out);params=list(net.parameters());gr=torch.autograd.grad(loss,params);dirs=[torch.randn_like(x) for x in params];norm=torch.sqrt(sum((x*x).sum() for x in dirs));dirs=[x/norm for x in dirs];expected=sum((x*y).sum() for x,y in zip(gr,dirs)).item();center=[x.detach().clone() for x in params]
        for eps in (1e-4,1e-5):
            values=[];unchanged=True
            for sign in (-1,1):
                with torch.no_grad():
                    for x,x0,di in zip(params,center,dirs):x.copy_(x0+sign*eps*di)
                    out=net(p2,base)[0];values.append(o.loss(out,base,gt,cell)[0].item());unchanged &= all(torch.equal(x,y) for x,y in zip(face,faces(out)))
            observed=(values[1]-values[0])/(2*eps);error=abs(observed-expected);assert unchanged,'NONSMOOTH_FD_UNDETERMINED_STOP';assert error<=1e-7+1e-3*abs(expected),(cell,eps,error)
            checks.append(dict(cell=cell,eps=eps,error=error,expected=expected,observed=observed,all_objective_faces_unchanged=True,double_copy_not_production_float32=True))
    assert list(inspect.signature(m.Refiner.forward).parameters)==['self','p2','native']
    return dict(status='PASS',keep=fixtures,original_A_output_and_all_parameter_gradient_EXACT=True,fullK20_numpy_torch_scalar_coordinategradient=True,component_tie_ADE_first=True,representation_double_inverse=True,velocity_float32_bound=True,feature_accumulation_error=err,double_FD=checks,GT_groups_IDs_not_forward=True)

def test_resume(st,data,cell):
    ix=data['split']['FIT'][:513];cal=data['split']['inner'][:12];folder=c.H/'resume_smoke'/cell;folder.mkdir(parents=True,exist_ok=True);manifest=dict(cell=cell,contract=c.digest(c.read(c.H/'protocol.json')),source='preflight-current')
    def epoch(net,opt,g,e):
        row=m.epoch(net,opt,data,ix,g,e,cell);net.eval()
        with torch.inference_mode():row['inner_loss']=o.loss(net(*m.inputs(data,cal))[0],m.inputs(data,cal)[1],torch.from_numpy(data['gt'][cal]).to(m.DEVICE),cell)[0].item()
        return row
    m.seed(720);net=m.Refiner(st).to(m.DEVICE);opt=m.optimizer(net);g=torch.Generator().manual_seed(720);hist=[epoch(net,opt,g,1),epoch(net,opt,g,2)];m.checkpoint(folder/'continuous.pt',net,opt,g,hist,manifest)
    m.seed(720);net=m.Refiner(st).to(m.DEVICE);opt=m.optimizer(net);g=torch.Generator().manual_seed(720);hist=[epoch(net,opt,g,1)];m.checkpoint(folder/'interrupted.pt',net,opt,g,hist,manifest)
    m.seed(999);net=m.Refiner(st).to(m.DEVICE);opt=m.optimizer(net);g=torch.Generator().manual_seed(999);hist=m.restore(folder/'interrupted.pt',net,opt,g,manifest);hist.append(epoch(net,opt,g,2));m.checkpoint(folder/'resumed.pt',net,opt,g,hist,manifest)
    m.exact(torch.load(folder/'continuous.pt',weights_only=False,map_location='cpu'),torch.load(folder/'resumed.pt',weights_only=False,map_location='cpu'))
    try:m.restore(folder/'interrupted.pt',net,opt,g,dict(manifest,source='bad'))
    except AssertionError:pass
    else:raise AssertionError('BAD_RESUME_ACCEPTED')
    return dict(cell=cell,status='EXACT',FIT=513,inner=12,epochs=2,all_parameters_bool_scalers_kernel_optimizer_TorchCPU_CUDA_loader_numpy_python_RNG_history_diagnostics=True,not_fullFIT_two_epochs=True)

def estimate():
    prior=c.read(c.T1/'RESOURCE.json');base=int(prior['peak_private_bytes'])
    with zipfile.ZipFile(c.T1/'runs/A/seed42/FIT_paths.npz') as archive:payload=sum(x.file_size for x in archive.infolist())
    scratch=20000*(3+956)*4;velocity=256*20*12*2*8*3;e=base+payload+scratch+velocity
    c.write('MEMORY_ESTIMATE.json',dict(estimated_total_peak_bytes=e,startup_required_bytes=1.3*e,measured_T1_lifetime_private_peak_bytes=base,T1_RESOURCE_SHA=c.sha(c.T1/'RESOURCE.json'),extra_saved_FIT_uncompressed_NPZ_payload_bytes=payload,trackdraw_scratch_bytes=scratch,velocity_batch_temporaries_bytes=velocity,not_free_RAM_selected=True,no_retrospective_T1_change=True))
    c.write('MEMORY_GUARD_FIXTURES.json',c.policy.fixtures());return c.guard('startup')

def reuse_contract(data,st):
    pre=c.read(c.T1/'PREFLIGHT.json');assert pre['status']=='PASS'
    c.verify(c.read(c.T1/'source_manifest.json')['files'])
    assert c.read(c.T1/'ARITHMETIC_AUDIT.json')['status']=='PASS'
    assert c.read(c.T1/'RESULT.json')['status']=='PASS_T1_TRANSFER_RETENTION'
    assert c.read(c.T1/'STATISTICS.json')['decision']['RETENTION_SIGNAL'] is False
    prior=c.read(c.T1/'RUNS.json');jobs=c.read(c.T1/'JOBS.json');position=c.npz(c.T1/'statistics_position.npz')
    for key in ('mean','std','counts'):assert np.array_equal(st[key][:96],position[key][:96]),('P2_STATS_CHANGED',key)
    assert np.array_equal(st['training_ids'],position['training_ids'])
    records=[]
    for seed in m.SEEDS:
        key=f'A/seed{seed}';row=prior[key];c.valid_done(row,pre['source_digest']);folder=c.T1/'runs'/key;manifest=c.read(folder/'run_manifest.json')
        assert manifest['cell']=='A' and manifest['representation']=='position' and manifest['keep'] is False and manifest['epochs']==25 and manifest['batch']==256
        assert manifest['FIT_ID_digest']==c.digest(data['ids'][data['split']['FIT']].tolist())
        hist=c.read(folder/'history.json');assert len(hist)==25 and all(r['n']==18909 and r['optimization_steps']==74 for r in hist)
        for split,ii in data['split'].items():
            jr=jobs[key+'/'+split];c.valid_done(jr,pre['source_digest']);z=c.npz(jr['output'])
            for k in ('native','gt','sample_id','groups','track','scene','sequence'):assert np.array_equal(z[k],data[k][ii]),('A_ID_OR_PAYLOAD',key,split,k)
            del z;gc.collect()
        records.append(dict(seed=seed,checkpoint_SHA=row['checkpoint_SHA'],four_saved_jobs_exact_IDs_and_SHA=True,history_rows=25,steps=1850))
    c.write('A_REUSE_CONTRACT.json',dict(status='PASS',source_digest=pre['source_digest'],records=records,only_A_reused_no_B_warmstart=True,T1_resource_deviation_remains=True,not_A_retraining=True,no_A_model_forward=True))
    return position

def precision_fixtures(data):
    from features import numpy_features
    cases={}
    for name,v in dict(constant=[1.,0.],zero=[0.,0.],reverse=[1.,0.],stop=[1.,0.],turn=[1.,0.]).items():
        vel=np.tile(v,(12,1))
        if name=='reverse':vel[6:]*=-1
        if name=='stop':vel[6:]=0
        if name=='turn':vel[6:]=[0.,1.]
        p=np.cumsum(vel*.4,0).astype(np.float32)[None,None]
        d=np.diff(np.concatenate((np.zeros_like(p[:,:,:1],dtype=float),p.astype(float)),2),axis=2)/.4
        c.close(np.cumsum(d*.4,2),p,2e-12);q=numpy_features(p,'velocity').astype(float);bound=8*np.finfo(np.float32).eps*.4*np.cumsum(abs(d),2)
        assert np.all(abs(np.cumsum(q*.4,2)-p)<=bound);cases[name]=dict(double_inverse_max=float(abs(np.cumsum(d*.4,2)-p).max()),quantized_inverse_max=float(abs(np.cumsum(q*.4,2)-p).max()),pointwise_bound_PASS=True)
    # Whole FIT inverse/quantization proof, no filtering; same registered production features.
    maxd=0.;maxq=0.;ix=data['split']['FIT']
    for start in range(0,len(ix),256):
        c.guard();p=data['native'][ix[start:start+256]].astype(np.float32);d=np.diff(np.concatenate((np.zeros_like(p[:,:,:1],dtype=float),p.astype(float)),2),axis=2)/.4;q=numpy_features(p,'velocity').astype(float)
        maxd=max(maxd,c.close(np.cumsum(d*.4,2),p,2e-12));error=abs(np.cumsum(q*.4,2)-p);maxq=max(maxq,float(error.max()));assert np.all(error<=8*np.finfo(np.float32).eps*.4*np.cumsum(abs(d),2))
    return dict(status='PASS',cases=cases,all_FIT_double_inverse_max=maxd,all_FIT_quantized_inverse_max=maxq,base_never_replaced_by_reconstruction=True)

def main():
    start=time.perf_counter();resource_start=estimate();c.write('RESOURCE_START.json',resource_start);c.stage('T2_PREFLIGHT_SOURCE_ID_STATS');data=c.load();st=c.stats(data,'velocity');np.savez_compressed(c.H/'statistics_velocity.npz',**st)
    position=reuse_contract(data,st);precision=precision_fixtures(data)
    import production_contract
    production_contract.check(data,'preflight-current',write_name='PRODUCTION_MANIFEST_PREFLIGHT.json')
    # Independent whole-FIT sums from immutable arrays, same registered 256 feature backend.
    from features import numpy_features
    ix=data['split']['FIT'];ss=np.zeros(120);sq=np.zeros(120);nn=np.zeros(120)
    for first in range(0,len(ix),256):
        ii=ix[first:first+256];p=data['p2'][ii].astype(float);vv=numpy_features(data['native'][ii],'velocity').reshape(-1,24).astype(float)
        ss[:96]+=np.sum(p,0);sq[:96]+=np.sum(p**2,0);nn[:96]+=len(p);ss[96:]+=np.sum(vv,0);sq[96:]+=np.sum(vv**2,0);nn[96:]+=len(vv)
    mean=ss/nn;sd=np.sqrt(np.maximum(sq/nn-mean**2,0));sd=np.where(sd<1e-6,1,sd);stat_error=max(c.close(st['mean'],mean,1e-11),c.close(st['std'],sd,1e-11))
    c.write('IDENTITY_AUDIT.json',dict(status='PASS',unique_ids=sum(c.SIZES.values()),records={name:dict(n=len(ii),change=int((data['groups'][ii]==2).sum()),tracks=len(set(data['track'][ii])),ID_digest=c.digest(data['ids'][ii].tolist())) for name,ii in data['split'].items()},P2_first96_stats_EXACT=True,own_native24_velocity_statistics=True,base_float32_to_float64_max=float(np.max(abs(data['native']-data['native'].astype(np.float32).astype(float)))),whole_FIT_statistics_error=stat_error,precision=precision))
    fx=fixtures(st,data);phases=[];resumes=[];bench=[];smoke=[]
    for cell in m.CELLS:
        c.stage('T2_PREFLIGHT_GPU8',cell=cell);m.seed(42);net=m.Refiner(st).to(m.DEVICE);before={k:x.detach().cpu().clone() for k,x in net.state_dict().items()};assert sum(x.numel() for x in net.parameters())==25304
        m.seed(42);control=m.Refiner(position,representation='position').to(m.DEVICE)
        m.exact(dict(net.named_parameters()),dict(control.named_parameters()))
        for k in ('kernelbank','native_identity_flag'):m.exact(before[k],control.state_dict()[k].cpu())
        for k in ('mean','std'):m.exact(before[k][:96],control.state_dict()[k][:96].cpu())
        ix=data['split']['FIT'][:256];assert np.array_equal(net(*m.inputs(data,ix))[0].detach().cpu(),data['native'][ix]);opt=m.optimizer(net);grads=[]
        for b in range(8):
            m.step(net,opt,data,data['split']['FIT'][b*256:(b+1)*256],cell);grads.append({k:float(torch.linalg.vector_norm(x.grad)) for k,x in net.named_parameters()});c.guard()
        assert all(any(g[k]>0 for g in grads[1:]) for k in grads[0]);phases.append(dict(cell=cell,parameter_count=25304,paired_weights_P2stats_kernel_bool_EXACT=True,candidate24_stats_intentionally_different=True,native_identity_EXACT=True,gradients=grads))
        pp,bb=m.inputs(data,ix[:8]);out=net(pp,bb)[0];gt=torch.from_numpy(data['gt'][ix[:8]]).to(m.DEVICE)
        first=torch.autograd.grad(o.loss(out,bb,gt,cell)[0],tuple(net.parameters()),retain_graph=True);second=torch.autograd.grad(o.objective(out,gt)[0],tuple(net.parameters()));m.exact(first,second)
        detached=out.detach().requires_grad_();g1=torch.autograd.grad(o.loss(detached,bb,gt,cell)[0],detached,retain_graph=True)[0];g2=torch.autograd.grad(o.objective_independent_torch(detached,gt),detached)[0];independent_gradient_error=c.close(g1.cpu().numpy(),g2.cpu().numpy(),1e-10)
        phases[-1].update(actual_nonzero_C_all_parameter_L0_gradient_EXACT=True,independent_coordinate_gradient_error=independent_gradient_error)
        ii=data['split']['outer'][:128];before={k:x.detach().cpu().clone() for k,x in net.state_dict().items()};t=time.perf_counter();p,raw,u,r=m.generated(net,data,ii);m.exact(before,{k:x.detach().cpu() for k,x in net.state_dict().items()});z=dict(paths=p,raw=raw,controls=u,correction=r,**{k:data[k][ii] for k in ('native','sample_id','gt','groups','track','scene','sequence')});met,aux=v.evaluate(z);folder=c.H/'preflight_smoke'/cell;folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'paths.npz',**z);np.savez_compressed(folder/'aux.npz',**aux);v.validate(c.npz(folder/'paths.npz'),met,c.npz(folder/'aux.npz'),'integrated_acceleration');duration=time.perf_counter()-t
        pop=v.population(net,data,ii,cell);ref=o.numpy_components(p,z['native'],z['gt']);c.close(pop['base'],ref['base']);c.close(pop['total'],ref['base'])
        smoke.append(dict(cell=cell,seconds=duration,all_weights_buffers_eval_EXACT=True,full_GPU_saved_outer128_independent_quality_clusters_five_motion_direction_controls=True,bytes=sum(x.nbytes for x in z.values())+sum(x.nbytes for x in aux.values())))
        resumes.append(test_resume(st,data,cell));m.seed(123);net=m.Refiner(st).to(m.DEVICE);opt=m.optimizer(net);times=[]
        for b in range(36):
            torch.cuda.synchronize();t=time.perf_counter();m.step(net,opt,data,ix,cell);torch.cuda.synchronize()
            if b>=4:times.append(time.perf_counter()-t)
        bench.append(dict(cell=cell,warm=4,measure=32,FIT256_seconds=float(np.mean(times)),no_saved_benchmark_model=True));del net,opt,control;gc.collect();torch.cuda.empty_cache()
    folder=c.H/'contract_fixture';folder.mkdir(exist_ok=True);c.write(folder/'valid.json',dict(fixture=True));p=folder/'valid.json';row=dict(status='PASS',source_digest='fixture',files={str(p):c.sha(p)});c.valid_done(row,'fixture');rejected=0
    for bad in (dict(row,source_digest='bad'),dict(row,files={str(p):'0'*64})):
        try:c.valid_done(bad,'fixture')
        except AssertionError:rejected+=1
    assert rejected==2
    files,inherited=c.locks();c.write('source_manifest.json',dict(files=files,inherited_parent=inherited,inherited_T1_base=2304));c.verify(files)
    elapsed=time.perf_counter()-start;train=5550*sum(x['FIT256_seconds'] for x in bench);pipeline=24679*6/128*np.mean([x['seconds'] for x in smoke]);planning=(train+pipeline+max(40,elapsed)+90)*1.2/60;bytes_est=24679*3/128*np.mean([x['bytes'] for x in smoke]);disk=c.psutil.disk_usage(str(c.H)).free;assert disk>=2*2**30+1.5*bytes_est
    c.guard();pre=dict(status='PASS' if planning<=30 else 'STOP_BUDGET',source_count=len(files),source_digest=c.digest(files),inherited_T1=inherited,planning_minutes=planning,ETA_minutes=[.7*planning,1.5*planning],elapsed_seconds=elapsed,resource=c.RESOURCE,resource_start=resource_start,fixtures=fx,actual_GPU8=phases,real_resume=resumes,benchmark=bench,smoke=smoke,train_seconds_nominal=train,full_pipeline_seconds_nominal=pipeline,disk_free_bytes=disk,uncompressed_outputs_bytes=bytes_est,completed_contract_SKIP_bad_source_output_resume_rejected=True,old_F16_not_loaded_or_forwarded=True,T1_RESOURCE_deviation_preserved=True)
    c.write('PREFLIGHT.json',pre);print('T2_PREFLIGHT',pre['status'],planning,pre['ETA_minutes'],len(files),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:c.failure('STOP_T2_PREFLIGHT');raise
