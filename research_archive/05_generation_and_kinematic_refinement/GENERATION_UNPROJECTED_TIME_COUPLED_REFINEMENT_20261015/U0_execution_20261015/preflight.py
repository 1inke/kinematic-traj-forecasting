import common as c,model as m,metrics as v
import numpy as np,torch,time,copy,gc,inspect

def inner(net,data,ix,arm):
    net.eval()
    with torch.inference_mode():return m.objective(net(*m.inputs(data,ix),arm)[0],torch.from_numpy(data['gt'][ix]).to(m.DEVICE))[0].item()

def test_resume(st,data,arm):
    ix=data['split']['FIT'][:513];cal=data['split']['inner'][:12];manifest=dict(test='real513FIT12inner2epoch',arm=arm,source='preflight-current',contract=c.digest(c.read(c.H/'protocol.json')));folder=c.H/'resume_smoke'/arm;folder.mkdir(parents=True,exist_ok=True)
    m.seed(720);net=m.Refiner(st).to(m.DEVICE);opt=m.optimizer(net);g=torch.Generator().manual_seed(720);hist=[]
    for e in (1,2):r=m.epoch(net,opt,data,ix,g,e,arm);r['inner_loss']=inner(net,data,cal,arm);hist.append(r)
    m.checkpoint(folder/'continuous.pt',net,opt,g,hist,manifest)
    m.seed(720);net=m.Refiner(st).to(m.DEVICE);opt=m.optimizer(net);g=torch.Generator().manual_seed(720);r=m.epoch(net,opt,data,ix,g,1,arm);r['inner_loss']=inner(net,data,cal,arm);m.checkpoint(folder/'interrupted.pt',net,opt,g,[r],manifest)
    m.seed(999);net=m.Refiner(st).to(m.DEVICE);opt=m.optimizer(net);g=torch.Generator().manual_seed(999);hist=m.restore(folder/'interrupted.pt',net,opt,g,manifest);r=m.epoch(net,opt,data,ix,g,2,arm);r['inner_loss']=inner(net,data,cal,arm);hist.append(r);m.checkpoint(folder/'resumed.pt',net,opt,g,hist,manifest)
    a=torch.load(folder/'continuous.pt',map_location='cpu',weights_only=False);b=torch.load(folder/'resumed.pt',map_location='cpu',weights_only=False);m.exact(a,b)
    rejected=False
    try:m.restore(folder/'interrupted.pt',net,opt,g,dict(manifest,source='wrong'))
    except AssertionError:rejected=True
    assert rejected
    return dict(arm=arm,status='EXACT',real_FIT=513,real_inner=12,epochs=2,all_parameters_buffers_bool_scalers_optimizer_TorchCPU_CUDA_loader_NumPy_Python_RNG_history=True,not_fullFIT_two_epochs=True,bad_resume_contract_rejected=True)

def fixtures(st,data):
    K=m.kernels();b=.20*(np.arange(1,13)/12.)**2;c.close(K.sum(-1),np.stack((b,b)))
    rng=np.random.default_rng(15);u=rng.uniform(-.2,.2,(3,20,12,2));native=data['native'][data['split']['FIT'][:3]]
    for arm in m.ARMS:
        raw=np.arctanh(u).astype(np.float32);out,controls,res=m.reconstruct(native,raw,arm)
        c.close(res,np.einsum('tj,bkjd->bktd',K[m.ARMS.index(arm)],controls))
        assert np.all(np.linalg.norm(res,axis=-1)<=b+2e-12)
    assert list(inspect.signature(m.Refiner.forward).parameters)==['self','p2','native','arm']
    # Independent complete K20 objective scalar/coordinate gradients on actual paths.
    ix=data['split']['FIT'][:8];p=torch.from_numpy(data['native'][ix]).to(m.DEVICE).requires_grad_();gt=torch.from_numpy(data['gt'][ix]).to(m.DEVICE);loss=m.objective(p,gt)[0];g=torch.autograd.grad(loss,p)[0];other=m.objective_independent_torch(p,gt);g2=torch.autograd.grad(other,p)[0];assert abs(loss.item()-m.objective_numpy(p.detach().cpu().numpy(),data['gt'][ix])[0])<=2e-12 and torch.max(abs(g-g2)).item()<=1e-10
    # Defined tie fixture first physical winner; zero errors have finite Torch grads.
    z=torch.zeros((2,20,12,2),device=m.DEVICE,dtype=torch.float64,requires_grad=True);gt0=torch.zeros((2,12,2),device=m.DEVICE,dtype=torch.float64);tie=m.objective(z,gt0);assert torch.equal(tie[3],torch.zeros(2,dtype=torch.long,device=m.DEVICE)) and torch.isfinite(torch.autograd.grad(tie[0],z)[0]).all()
    fd=[]
    for arm in m.ARMS:
        m.seed(162);net=m.Refiner(st).to(m.DEVICE).double()
        with torch.no_grad():net.head[-1].weight.normal_(0,.002);net.head[-1].bias.fill_(.01)
        p2,base=m.inputs(data,ix);p2=p2.double();base=base.double();p=net(p2,base,arm)[0];loss=m.objective(p,gt)[0];params=list(net.parameters());grads=torch.autograd.grad(loss,params);dirs=[torch.randn_like(x) for x in params];norm=torch.sqrt(sum((x*x).sum() for x in dirs));dirs=[x/norm for x in dirs];expected=sum((g*d).sum() for g,d in zip(grads,dirs)).item();center=[x.detach().clone() for x in params];winner=m.objective(p,gt)[3];checks=[]
        for eps in (1e-4,1e-5):
            values=[];ws=[]
            for sign in (-1,1):
                with torch.no_grad():
                    for x,x0,d in zip(params,center,dirs):x.copy_(x0+sign*eps*d)
                    out=net(p2,base,arm)[0];loss2=m.objective(out,gt);values.append(loss2[0].item());ws.append(loss2[3])
            assert all(torch.equal(winner,w) for w in ws),'FD_WINNER_CHANGED_UNDETERMINED'
            observed=(values[1]-values[0])/(2*eps);err=abs(observed-expected);assert err<=1e-7+1e-3*abs(expected),(arm,eps,err,expected);checks.append(dict(eps=eps,expected=expected,observed=observed,error=err,winner_face_unchanged=True))
        fd.append(dict(arm=arm,checks=checks,double_copy_not_production_float32_FD=True))
    return dict(status='PASS',independent_integral_recurrence=True,position_budget=True,no_GT_forward=True,fullK20_objective_scalar_and_coordinate_grad=True,tie_first_slot_and_zero_norm_finite=True,finite_difference=fd,kernel_ranks=[int(np.linalg.matrix_rank(k)) for k in K],kernel_conditions=[float(np.linalg.cond(k)) for k in K])

def main():
    start=time.perf_counter();c.stage('U0_PREFLIGHT_RESOURCE_IDS');data=c.load();st=c.stats(data);np.savez_compressed(c.H/'statistics.npz',**st)
    counts={name:dict(n=len(ix),tracks=len(set(data['track'][ix])),groups={k:int(mask.sum()) for k,mask in c.masks(data['groups'][ix])},ID_digest=c.digest(data['sample_id'][ix].tolist())) for name,ix in data['split'].items()};c.write('IDENTITY_AUDIT.json',dict(status='PASS',records=counts,past_last_origin_zero=True,P2_observed_cache_ID_exact=True,GT_groups_not_forward=True,FIT_only_stats=True))
    files,inherited=c.locks();fx=fixtures(st,data);phases=[];resume=[];bench=[];smoke=[]
    for arm in m.ARMS:
        c.stage('U0_PREFLIGHT_GPU8',arm=arm);m.seed(42);net=m.Refiner(st).to(m.DEVICE);paired={k:x.detach().cpu().clone() for k,x in net.state_dict().items()};assert sum(x.numel() for x in net.parameters())==25304
        if arm==m.ARMS[0]:init=paired
        else:m.exact(init,paired)
        ix=data['split']['FIT'][:256];before=net(*m.inputs(data,ix),arm)[0].detach().cpu().numpy();assert np.array_equal(before,data['native'][ix])
        opt=m.optimizer(net);grads=[]
        for b in range(8):
            ii=data['split']['FIT'][b*256:(b+1)*256];m.step(net,opt,data,ii,arm);grads.append({k:float(torch.linalg.vector_norm(x.grad).item()) for k,x in net.named_parameters()});c.guard()
        assert all(any(r[k]>0 for r in grads[1:]) for k in grads[0])
        phases.append(dict(arm=arm,trainable25304=True,initial_native_EXACT=True,paired_all_parameters_bool_scalers_kernelbank=True,gradients=grads))
        ii=data['split']['FIT'][:128];t=time.perf_counter();p,raw,u,r=m.generated(net,data,ii,arm);z=dict(paths=p,native=data['native'][ii],raw=raw,controls=u,correction=r,**{k:data[k][ii] for k in ('sample_id','gt','groups','track','scene','sequence')});met,aux=v.evaluate(z);path=c.H/'preflight_smoke'/arm;path.mkdir(parents=True,exist_ok=True);np.savez_compressed(path/'paths.npz',**z);np.savez_compressed(path/'aux.npz',**aux);v.validate(c.npz(path/'paths.npz'),met,c.npz(path/'aux.npz'),arm);duration=time.perf_counter()-t;smoke.append(dict(arm=arm,actual_GPU128_saved_full_quality_original_clusters_motion_reconstruction=True,seconds=duration,uncompressed_bytes=sum(x.nbytes for x in z.values())+sum(x.nbytes for x in aux.values())))
        resume.append(test_resume(st,data,arm));m.seed(123);net=m.Refiner(st).to(m.DEVICE);opt=m.optimizer(net);times=[]
        for b in range(36):
            torch.cuda.synchronize();t=time.perf_counter();m.step(net,opt,data,ix,arm);torch.cuda.synchronize()
            if b>=4:times.append(time.perf_counter()-t)
        bench.append(dict(arm=arm,warm=4,measured=32,FIT256_seconds=float(np.mean(times)),no_saved_benchmark_model=True));del net,opt;gc.collect();torch.cuda.empty_cache()
    c.write('fixtures/valid.json',dict(fixture=True));p=str(c.H/'fixtures/valid.json');row=dict(status='PASS',source_digest='fixture',files={p:c.sha(p)});assert c.valid_done(row,'fixture');reject=0
    for bad in (dict(row,source_digest='bad'),dict(row,files={p:'0'*64})):
        try:c.valid_done(bad,'fixture')
        except AssertionError:reject+=1
    assert reject==2
    c.verify(files);elapsed=time.perf_counter()-start;train=5550*sum(x['FIT256_seconds'] for x in bench);evaltime=24679*7/128*np.mean([x['seconds'] for x in smoke]);planning=(train+evaltime+2*max(10,elapsed-train*.01)+90)*1.20/60;disk=c.psutil.disk_usage(str(c.H)).free/2**30;size=24679*6/128*np.mean([x['uncompressed_bytes'] for x in smoke])/2**30;assert disk>=2+1.5*size
    pre=dict(status='PASS' if planning<=30 else 'STOP_BUDGET',source_count=len(files),inherited_S0=inherited,source_digest=c.digest(files),planning_minutes=planning,ETA_minutes=[.7*planning,1.5*planning],fixtures=fx,GPU8=phases,real_resume=resume,smoke=smoke,benchmark=bench,train_seconds_nominal=train,full24_saved_pipeline_seconds_nominal=evaltime,elapsed_seconds=elapsed,resource=c.RESOURCE,disk_free_GiB=disk,uncompressed_saved_arrays_GiB=size,scientific_jobs=24,native_jobs=4,completeSHA_skip_badsource_output_resume_rejected=True,old_models_not_forwarded_or_updated=True)
    c.write('source_manifest.json',dict(files=files,inherited_S0=inherited));c.write('PREFLIGHT.json',pre);print('U0_PREFLIGHT',pre['status'],planning,pre['ETA_minutes'],len(files),c.RESOURCE,flush=True)

if __name__=='__main__':
    try:main()
    except Exception:c.failure('STOP_U0_PREFLIGHT');raise
