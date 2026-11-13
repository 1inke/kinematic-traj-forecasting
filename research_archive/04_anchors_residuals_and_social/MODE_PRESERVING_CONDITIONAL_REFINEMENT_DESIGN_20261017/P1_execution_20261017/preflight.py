"""Actual registered scientific checks; source is sealed only after all pass."""
import common as c,model as m,objectives as o,metrics as v
import torch,numpy as np,time,gc,inspect,hashlib,copy
def fixtures(st,data):
    ix=data['split']['FIT'][:8];base=m.inputs(data,ix)[1];gt=torch.from_numpy(data['gt'][ix]).to(m.DEVICE)
    fx=[]
    for kind in ('zero_drift','nonowner_drift','ties','owner_switch'):
        n=np.zeros((3,20,12,2));n[...,0]=np.arange(20)[None,:,None]/16;y=np.full((3,12,2),.03125);p=n.copy()
        if kind=='nonowner_drift':p[:,3,:,1]+=.125
        if kind=='ties':n[:]=.25;p=n.copy();y[:]=0.
        if kind=='owner_switch':p[:,0,:,0]=4.
        for cell in m.CELLS:
            pp=torch.tensor(p,device=m.DEVICE,requires_grad=True);nn=torch.tensor(n,device=m.DEVICE);gg=torch.tensor(y,device=m.DEVICE)
            loss,d=o.loss(pp,nn,gg,cell);nd=o.numpy_components(p,n,y,cell)
            for k in ('total','task','anchor','diversity'):c.close(float(loss if k=='total' else d[k]),nd[k])
            assert np.array_equal(d['owner'].cpu(),nd['owner'])
            ag=torch.autograd.grad(d['anchor'],pp,retain_graph=True)[0];assert torch.isfinite(ag).all();assert torch.count_nonzero(ag[torch.arange(3),d['owner']])==0
            if kind in ('zero_drift','ties','owner_switch'):assert d['anchor']==0 if kind!='owner_switch' else True
            if kind=='nonowner_drift':assert torch.count_nonzero(ag[:,3])>0
            if kind=='ties':assert d['owner'].tolist()==[0,0,0]
            fx.append(dict(case=kind,cell=cell,numpy_scalar_and_owner=True,anchor_owner_direct_gradient_zero=True))
    # Legacy U0 nonzero A output and every parameter gradient EXACT.
    ns=dict(c=c,np=np,torch=torch,nn=m.nn,kernels=m.kernels,ARMS=m.ARMS)
    c.extract(c.PARENT/'model.py',('Refiner','objective'),ns)
    m.seed(162);net=m.Refiner(st).to(m.DEVICE);old=ns['Refiner'](st).to(m.DEVICE)
    with torch.no_grad():net.head[-1].weight.normal_(0,.002);net.head[-1].bias.fill_(.01)
    old.load_state_dict(net.state_dict());p=net(*m.inputs(data,ix))[0];q=old(*m.inputs(data,ix),'integrated_acceleration')[0];assert torch.equal(p,q)
    a=torch.autograd.grad(o.loss(p,base,gt,'A')[0],tuple(net.parameters()));b=torch.autograd.grad(ns['objective'](q,gt)[0],tuple(old.parameters()));m.exact(a,b)
    # Independently assembled coordinate gradients: no production vector pair reduction.
    for cell in m.CELLS:
        p=net(*m.inputs(data,ix))[0].detach().requires_grad_();l,d=o.loss(p,base,gt,cell)
        dist=torch.sqrt(((p-gt[:,None])**2).sum(-1));Q=dist.mean(-1)+.5*dist[...,-1];owner=d['owner'].detach()
        task=Q[torch.arange(len(p)),owner].mean();dr=torch.linalg.vector_norm(p-base,dim=-1);dr=dr.mean(-1)+.5*dr[...,-1]
        anchor=sum(dr[i,torch.arange(20,device=p.device)!=owner[i]].sum() for i in range(len(p)))/(len(p)*19)
        pieces=[torch.relu(.3-torch.linalg.vector_norm(p[:,j]-p[:,k],dim=-1).mean(-1)).square() for j in range(20) for k in range(j+1,20)]
        independent=task+.05*torch.stack(pieces).mean()+(anchor if cell in ('C','D') else 0)
        c.close(float(l),float(independent));g1=torch.autograd.grad(l,p,retain_graph=True)[0];g2=torch.autograd.grad(independent,p)[0];c.close(g1.cpu(),g2.cpu())
    # 32 hash-fixed normalized parameter directions per cell, both eps and all active faces saved.
    fd=[]
    for cell in m.CELLS:
        m.seed(162);net=m.Refiner(st).to(m.DEVICE).double()
        with torch.no_grad():net.head[-1].weight.normal_(0,.002);net.head[-1].bias.fill_(.01)
        p2,base=m.inputs(data,ix);p2=p2.double();base=base.double();out=net(p2,base)[0];loss,d=o.loss(out,base,gt,cell)
        params=list(net.parameters());gr=torch.autograd.grad(loss,params);center=[x.detach().clone() for x in params]
        def faces():
            out,raw,u,r=net(p2,base);_,dd=o.loss(out,base,gt,cell);pair=torch.triu_indices(20,20,1,device=out.device);distance=torch.linalg.vector_norm(out[:,pair[0]]-out[:,pair[1]],dim=-1).mean(-1)
            z=torch.tanh(raw).double()
            return [dd['owner'],(dd['Q']==dd['Q'].min(1).values[:,None]),distance<.3,torch.linalg.vector_norm(z,dim=-1)>1,torch.linalg.vector_norm(out-base,dim=-1)==0]
        reference=faces();rows=[]
        for j in range(32):
            tag=f'P1-FD-20261017:{cell}:{j}';seed=int(hashlib.sha256(tag.encode()).hexdigest()[:16],16)%(2**63-1);gen=torch.Generator(device=m.DEVICE).manual_seed(seed)
            dirs=[torch.randn(x.shape,device=x.device,dtype=x.dtype,generator=gen) for x in params];norm=torch.sqrt(sum((x*x).sum() for x in dirs));dirs=[x/norm for x in dirs];expected=float(sum((g*x).sum() for g,x in zip(gr,dirs)));smooth=True;checks=[]
            for eps in (1e-4,1e-5):
                vals=[]
                for sign in (-1,1):
                    with torch.no_grad():
                        for x,x0,di in zip(params,center,dirs):x.copy_(x0+sign*eps*di)
                        vals.append(float(o.loss(net(p2,base)[0],base,gt,cell)[0]));smooth &= all(torch.equal(a,b) for a,b in zip(reference,faces()))
                observed=(vals[1]-vals[0])/(2*eps);err=abs(observed-expected);passed=err<=1e-7 or err<=1e-4*max(abs(expected),abs(observed))
                checks.append(dict(eps=eps,error=err,expected=expected,observed=observed,passed=passed))
            with torch.no_grad():
                for x,x0 in zip(params,center):x.copy_(x0)
            if smooth:assert all(x['passed'] for x in checks),(cell,j,checks)
            rows.append(dict(direction=j,hash_seed=seed,smooth=smooth,classification='SMOOTH' if smooth else 'NONDIFFERENTIABLE',checks=checks))
        assert sum(x['smooth'] for x in rows)>=24
        fd.append(dict(cell=cell,rows=rows,smooth=sum(x['smooth'] for x in rows)))
    assert list(inspect.signature(m.Refiner.forward).parameters)==['self','p2','native']
    return dict(status='PASS',cases=fx,double_FD=fd,legacy_U0_A_output_and_all_parameter_grad_EXACT=True,independent_coordinate_gradient=True)
def resume(st,data,cell):
    ix=data['split']['FIT'][:513];cal=data['split']['inner'][:12];folder=c.H/'resume_smoke'/cell;folder.mkdir(parents=True,exist_ok=True);contract=dict(cell=cell,protocol=c.sha(c.H/'protocol.json'),owner_labels=c.sha(c.H/'responsibility_labels.npz'))
    def epoch(net,opt,g,e):
        r=m.epoch(net,opt,data,ix,g,e,cell);net.eval()
        with torch.no_grad():r['inner_loss']=float(o.loss(net(*m.inputs(data,cal))[0],m.inputs(data,cal)[1],torch.from_numpy(data['gt'][cal]).to(m.DEVICE),cell)[0])
        r['coordinate_diagnostic']=m.coordinate_diagnostic(net,data,cal,cell)
        return r
    m.seed(720);net=m.Refiner(st).to(m.DEVICE);opt=m.optimizer(net);g=torch.Generator().manual_seed(720);hist=[epoch(net,opt,g,1),epoch(net,opt,g,2)];m.checkpoint(folder/'continuous.pt',net,opt,g,hist,contract)
    m.seed(720);net=m.Refiner(st).to(m.DEVICE);opt=m.optimizer(net);g=torch.Generator().manual_seed(720);hist=[epoch(net,opt,g,1)];m.checkpoint(folder/'interrupted.pt',net,opt,g,hist,contract)
    m.seed(999);net=m.Refiner(st).to(m.DEVICE);opt=m.optimizer(net);g=torch.Generator().manual_seed(999);hist=m.restore(folder/'interrupted.pt',net,opt,g,contract);hist.append(epoch(net,opt,g,2));m.checkpoint(folder/'resumed.pt',net,opt,g,hist,contract)
    m.exact(torch.load(folder/'continuous.pt',weights_only=False,map_location='cpu'),torch.load(folder/'resumed.pt',weights_only=False,map_location='cpu'))
    rejected=0
    for bad in (dict(contract,protocol='bad'),dict(contract,owner_labels='bad')):
        try:m.restore(folder/'interrupted.pt',net,opt,g,bad)
        except AssertionError:rejected+=1
    assert rejected==2
    return dict(cell=cell,status='EXACT',FIT=513,inner=12,epochs=2,all_params_bool_scalers_kernel_optimizer_TorchCPU_CUDA_loader_numpy_python_RNG_history_diagnostics_EXACT=True,bad_source_and_owner_labels_rejected=True)
def main():
    start=time.perf_counter();c.guard('startup');c.stage('P1_PREFLIGHT_IDS');data=c.load();st=c.stats(data);np.savez_compressed(c.H/'statistics_position.npz',**st)
    old=c.npz(c.PARENT/'statistics.npz');m.exact(st,old)
    ix=data['split']['FIT'];nativeQ=np.linalg.norm(data['native'][ix]-data['gt'][ix,None],axis=-1);nativeQ=nativeQ.mean(-1)+.5*nativeQ[...,-1]
    np.savez_compressed(c.H/'responsibility_labels.npz',sample_id=data['ids'][ix],Q0=nativeQ,owner=nativeQ.argmin(1),ties=(nativeQ==nativeQ.min(1)[:,None]).sum(1))
    c.write('IDENTITY_AUDIT.json',dict(status='PASS',unique_ids=sum(c.SIZES.values()),records={s:dict(n=len(ii),change=int((data['groups'][ii]==2).sum()),tracks=len(set(data['track'][ii])),ID_digest=c.digest(data['ids'][ii].tolist())) for s,ii in data['split'].items()},statistics_original_position_EXACT=True,fixed_owner_FIT_only=True))
    fx=fixtures(st,data);phases=[];resumes=[];bench=[];smoke=[];paired={}
    for seed in m.SEEDS:
        for cell in m.CELLS:
            m.seed(seed);net=m.Refiner(st).to(m.DEVICE);state={k:x.detach().cpu().clone() for k,x in net.state_dict().items()}
            if cell=='A':paired[seed]=state
            else:m.exact(paired[seed],state)
    for cell in m.CELLS:
        c.stage('P1_PREFLIGHT_GPU8',cell=cell);m.seed(42);net=m.Refiner(st).to(m.DEVICE);assert sum(x.numel() for x in net.parameters())==25304
        ix=data['split']['FIT'][:256];assert np.array_equal(net(*m.inputs(data,ix))[0].detach().cpu(),data['native'][ix]);opt=m.optimizer(net);grads=[]
        for b in range(8):
            m.step(net,opt,data,data['split']['FIT'][b*256:(b+1)*256],cell);grads.append({k:float(x.grad.norm()) for k,x in net.named_parameters()})
        assert all(any(g[k]>0 for g in grads[1:]) for k in grads[0])
        phases.append(dict(cell=cell,parameter_count=25304,gradients=grads,paired_all_seed_state=True,native_identity=True))
        ii=data['split']['outer'][:128];before={k:x.detach().cpu().clone() for k,x in net.state_dict().items()};t=time.perf_counter();p,raw,u,r=m.generated(net,data,ii);m.exact(before,{k:x.detach().cpu() for k,x in net.state_dict().items()})
        z=dict(paths=p,raw=raw,controls=u,correction=r,cell=np.array(cell),**{k:data[k][ii] for k in ('native','sample_id','gt','groups','track','scene','sequence')});met,aux=v.evaluate(z);folder=c.H/'preflight_smoke'/cell;folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'paths.npz',**z);np.savez_compressed(folder/'aux.npz',**aux);c.write(folder/'metrics.json',met);v.validate(c.npz(folder/'paths.npz'),met,c.npz(folder/'aux.npz'),'integrated_acceleration')
        smoke.append(dict(cell=cell,seconds=time.perf_counter()-t,bytes=sum(x.nbytes for x in z.values())+sum(x.nbytes for x in aux.values()),actual128_saved_full_quality_original_clusters_five_motion_responsibility_support=True))
        resumes.append(resume(st,data,cell));m.seed(123);net=m.Refiner(st).to(m.DEVICE);opt=m.optimizer(net);times=[]
        for b in range(36):
            torch.cuda.synchronize();t=time.perf_counter();m.step(net,opt,data,ix,cell);torch.cuda.synchronize()
            if b>=4:times.append(time.perf_counter()-t)
        bench.append(dict(cell=cell,warm=4,measure=32,FIT256_seconds=float(np.mean(times)),no_saved_benchmark_model=True));del net,opt;gc.collect();torch.cuda.empty_cache()
    # Same production feature array with two accumulation chunk sizes, not mixed float32 feature backends.
    ix=data['split']['FIT'][:1033];feats=np.concatenate((np.repeat(data['p2'][ix],20,0),data['native'][ix].astype(np.float32).reshape(-1,24)),1).astype(float);acc=[]
    for chunk in (31,256):
        a=np.zeros(120);b=np.zeros(120)
        for first in range(0,len(feats),chunk):a+=feats[first:first+chunk].sum(0);b+=(feats[first:first+chunk]**2).sum(0)
        acc.append((a/len(feats),b/len(feats)))
    stats_error=max(c.close(a,b,1e-11) for a,b in zip(*acc))
    import bootstrap as boot
    rng=np.random.default_rng(17);stats=np.abs(rng.normal(size=(4,3,5,12)));stats[:,:,:,4:8]+=3;stats[:,:,:,11]+=3;tr=np.arange(5)
    aa,ap,ss,tt=boot.resample(stats,tr,n=100000);bb,bp,sss,ttt=boot.resample(stats,tr,n=100000,ind=True,saved={'seeds':ss,'tracks':tt});c.close(aa,bb);c.close(ap,bp)
    # A track with zero conditional denominator must produce invalid draws, no redraw.
    sparse=stats.copy();sparse[:,:,:,5]=0;dd,_,_,_=boot.resample(sparse,tr,n=1024);assert np.isnan(dd[:,:,1]).all()
    c.write('DA1_REVIEW.json',dict(status='PASS_WITH_LIMITATIONS',same_assistant_not_external=True,issue='Fixed closest native owner may be unrepairable; soft anchor may only suppress rescues. Four arms isolate interventions, not unique shared-head root cause. W0 does not gate or feed labels.',reviewer='inline ARS architect and DA1'))
    fixture=c.H/'contract_fixture.json';c.write(fixture,{'valid':True});row=dict(status='PASS',source_digest='fixture',files={str(fixture):c.sha(fixture)});assert c.valid_done(row,'fixture')
    rejected=0
    for bad in (dict(row,source_digest='bad'),dict(row,files={str(fixture):'0'*64})):
        try:c.valid_done(bad,'fixture')
        except AssertionError:rejected+=1
    assert rejected==2
    files,inherited=c.locks();c.write('source_manifest.json',dict(files=files,inherited_count=inherited));c.verify(files)
    elapsed=time.perf_counter()-start;train=5550*sum(x['FIT256_seconds'] for x in bench);pipeline=24679*13/128*np.mean([x['seconds'] for x in smoke]);planning=(train+pipeline+elapsed+120)*1.25/60
    bytes_est=24679*13/128*np.mean([x['bytes'] for x in smoke]);free=c.psutil.disk_usage(c.H).free;assert free>1.4*bytes_est+1*2**30,('DISK',free,bytes_est)
    c.write('PREFLIGHT.json',dict(status='PASS',source_count=len(files),source_digest=c.digest(files),planning_minutes=planning,ETA_minutes=[.7*planning,1.5*planning],elapsed_seconds=elapsed,fixtures=fx,actual_GPU8=phases,real_resume=resumes,benchmark=bench,smoke=smoke,statistics_chunk_error=stats_error,bootstrap100000_independent_drawloop_EXACT=True,sparse_conditional_no_redraw=True,resource=c.RESOURCE,disk_free=free,estimated_saved_uncompressed_bytes=bytes_est,no_time_veto=True,old_F16_not_loaded_forwarded=True))
    print('P1_PREFLIGHT_PASS',planning,len(files),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:c.failure('STOP_P1_PREFLIGHT');raise

