"""Actual GPU, NumPy full parameter gradient, fixed FD, exact interruption and saved pipeline."""
from common import *
from objectives import loss,numpy_loss_gradient
from train_stage import step,blob,restore,predict
from evaluate_saved import save_population
import independent_gradient

def ix_hash(d,n,tag):return np.array(sorted(range(len(d['ids'])),key=lambda i:hashlib.sha256(f'{tag}|{int(d["ids"][i])}'.encode()).digest())[:n])
def nonzero(m):
    g=torch.Generator(device='cpu').manual_seed(2026110199)
    with torch.no_grad():
        m.head.weight.copy_((torch.randn(m.head.weight.shape,generator=g)*.05).to(m.head.weight));m.head.bias.copy_((torch.randn(m.head.bias.shape,generator=g)*.1).to(m.head.bias))
    return m
def face(info,norm):return [(z['winner'].detach().cpu(),z['hinge'].detach().cpu()) for z in info]+[bool(norm>5)]
def fd_and_numpy(groups,d,ix):
    m=nonzero(new_model(42,groups,'cpu')).double();x=torch.from_numpy(d['x'][ix[:8]].astype(float));y=torch.from_numpy(d['gt'][ix[:8]].astype(float))
    p=m(x);v,inf=loss(p,y);v.backward();value,paths,grad,nfaces=independent_gradient.check(m,x.numpy(),y.numpy())
    assert abs(float(v.detach())-value)<=1e-10
    patherr=float(abs(paths-p.detach().numpy()).max());assert patherr<=1e-12
    graderr=max(float(abs(grad[k]-z.grad.numpy()).max()) for k,z in m.named_parameters());assert graderr<=1e-10,graderr
    # Also verify independent all-path gradients (separate from chain rule).
    pp=p.detach().clone().requires_grad_(True);z,_=loss(pp,y);z.backward();_,dg,_=numpy_loss_gradient(pp.detach().numpy(),y.numpy());assert np.max(abs(pp.grad.numpy()-dg))<=1e-10
    base=[a.detach().clone() for a in m.parameters()];gg=torch.Generator().manual_seed(2026110107);rows=[]
    norm=float(torch.sqrt(sum((z.grad**2).sum() for z in m.parameters())));baseface=face(inf,norm)
    for direction in range(4):
        ds=[torch.randn(a.shape,generator=gg,dtype=torch.float64)*.01 for a in base];ref=float(sum((z.grad*b).sum() for z,b in zip(m.parameters(),ds)))
        for eps in (1e-4,1e-5):
            values=[];faces=[]
            for sign in (1,-1):
                with torch.no_grad():
                    for z,a,b in zip(m.parameters(),base,ds):z.copy_(a+sign*eps*b)
                vv,ii=loss(m(x),y);gs=torch.autograd.grad(vv,list(m.parameters()));nn=float(torch.sqrt(sum((a*a).sum() for a in gs)))
                values.append(float(vv.detach()));faces.append(face(ii,nn))
            fixed=all(equal(baseface,f) for f in faces);assert fixed,('FIXED_FD_ACTIVITY_CHANGED_NO_EPS_SEARCH',groups,direction,eps)
            actual=(values[0]-values[1])/(2*eps);err=abs(actual-ref);tol=1e-6+1e-3*abs(ref);assert err<=tol
            rows.append(dict(direction=direction,eps=eps,reference=ref,FD=actual,error=err,tolerance=tol,winner_hinge_clip_faces_exact=fixed))
    # Tied/bound query counterfactual, not an additional scientific model.
    if groups==1:
        for z,a in zip(m.parameters(),base):
            with torch.no_grad():z.copy_(a)
        p=m(x);single,_=loss(p,y);tied,_=loss(torch.cat([p,p,p],1),y)
        gs=torch.autograd.grad(single,list(m.parameters()),retain_graph=True);gt=torch.autograd.grad(tied,list(m.parameters()))
        assert abs(float(single-tied))<=1e-12 and max(float(abs(a-b).max()) for a,b in zip(gs,gt))<=1e-12
        # Original ContextK20 nonzero bridge scalar, predictions, all gradients.
        oldm=old.ContextK20(*stats()).double();oldm.load_state_dict(m.state_dict());oldp=oldm(x);ov,_=old.base_loss(oldp,y);nv,_=loss(m(x),y)
        og=torch.autograd.grad(ov,list(oldm.parameters()));ng=torch.autograd.grad(nv,list(m.parameters()))
        assert torch.equal(oldp,m(x)[:,0]) and torch.equal(ov,nv)
        assert all(torch.equal(a,b) for a,b in zip(og,ng))
    return dict(groups=groups,numpy_scalar_error=abs(float(v.detach())-value),numpy_path_error=patherr,all_parameter_gradient_error=graderr,FD=rows)

def resume(groups,d,inn,ix,iy,out):
    orders=[];gen=torch.Generator().manual_seed(991)
    for _ in range(2):orders.append(torch.randperm(513,generator=gen).numpy())
    orders=np.stack(orders);x=d['x'][ix[:513]];y=d['gt'][ix[:513]].astype(np.float32)
    c=dict(source={p.name:sha(p) for p in HERE.glob('*.py')},stats=sha(HERE/'statistics.npz'),IDs=digest(d['ids'][ix[:513]]),inner_IDs=digest(inn['ids'][iy[:12]]),groups=groups,order=digest(orders))
    def run(interrupted):
        m=new_model(42,groups);opt=optimizer(m);g=torch.Generator().manual_seed(42);hist=[];evals=[];count=0
        for ep,order in enumerate(orders):
            for cursor in range(0,513,256):
                ii=order[cursor:cursor+256];a,b,info=step(m,opt,torch.from_numpy(x[ii]).cuda(),torch.from_numpy(y[ii]).cuda());count+=1
                hist.append(dict(loss=a,norm=b,winners=[q['winner'].cpu().numpy() for q in info],batchIDs=digest(d['ids'][ix[:513]][ii])))
                if interrupted and count==4:
                    z=blob(m,opt,g,hist,ep,cursor+len(ii),orders,c);save_torch(out/'interrupted.pt',z);s=sha(out/'interrupted.pt');del m,opt
                    random.random();np.random.random();torch.rand(4);torch.rand(4,device='cuda')
                    m=new_model(17,groups);opt=optimizer(m);loaded=trusted(out/'interrupted.pt',s,c);hist,ee,cc=restore(loaded,m,opt,g,c,orders);assert ee==ep and cc==cursor+len(ii)
            evals.append(predict(m,inn['x'][iy[:12]]))
        return dict(model=state(m),optimizer=copy.deepcopy(opt.state_dict()),rng=capture_rng(g),history=hist,inner=np.stack(evals))
    a=run(False);b=run(True);assert equal(a,b),'EXACT_RESUME_ALL_PARAMS_ADAM_RNG_HISTORY_EVAL'
    return dict(status='PASS_EXACT',groups=groups,FIT=513,inner=12,epochs=2,updates=6,interrupted_after=4,all_params_buffers_optimizer_CPU_CUDA_Python_NumPy_loader_RNG_order_cursor_history_eval_EXACT=True)

def bad_readers():
    folder=HERE/'preflight/readers';folder.mkdir(parents=True,exist_ok=True);m=new_model(42,1,'cpu');op=optimizer(m);g=torch.Generator().manual_seed(42)
    c=dict(source='abc',IDs='id',stats='stat',groups=1,main_group=0,initial='init',orders='order');z=blob(m,op,g,[],0,0,np.arange(8),c);save_torch(folder/'valid.pt',z);s=sha(folder/'valid.pt')
    assert trusted(folder/'valid.pt',s,c)['contract']==c;rejected=[]
    for k in c:
        bad=copy.deepcopy(c);bad[k]='intentionally bad'
        try:trusted(folder/'valid.pt',s,bad)
        except ValueError:rejected.append(k)
        else:raise AssertionError('bad contract accepted '+k)
    try:trusted(folder/'valid.pt','0'*64,c)
    except ValueError:rejected.append('checkpoint_sha')
    else:raise AssertionError('bad checkpoint SHA accepted')
    npz(folder/'saved.npz',x=np.arange(4));rel=(folder/'saved.npz').relative_to(HERE).as_posix();receipt=dict(contract=c,files={rel:sha(folder/'saved.npz')})
    write(folder/'valid_done.json',receipt);assert done(folder/'valid_done.json',c)==receipt
    receipt['files'][rel]='0'*64;write(folder/'bad_done.json',receipt)
    try:done(folder/'bad_done.json',c)
    except ValueError:rejected.append('output_reader_sha')
    else:raise AssertionError('bad saved output accepted')
    return rejected

def main():
    setup();folder=HERE/'preflight';folder.mkdir(exist_ok=False);g=Guard(6*2**30);g.check(force=True);start=time.perf_counter();d=data();fit=d['FIT'];inn=d['inner'];ix=ix_hash(fit,513,'group-query-pf-fit-v1');iy=ix_hash(inn,12,'group-query-pf-inner-v1')
    results=[];resume_reports=[];throughput={};gpu_batches={};save_timing=[]
    for arm,G in [('Q1',1),('Q3',3)]:
        stage('Q_ACTUAL_PREFLIGHT',arm=arm);rows=folder/arm;rows.mkdir();m=new_model(42,G);opt=optimizer(m);phase=[]
        # Paired group0 states must be exact before any updates.
        initial=state(m);expected=torch.load(HERE/f'initial/{arm}_42.pt',weights_only=True);assert equal(initial,expected)
        for b in range(8):
            ids=np.arange(b*256,(b+1)*256);v,n,info=step(m,opt,torch.from_numpy(fit['x'][ids]).cuda(),torch.from_numpy(fit['gt'][ids].astype(np.float32)).cuda())
            phase.append({k:dict(norm=float(p.grad.norm()),finite=bool(torch.isfinite(p.grad).all())) for k,p in m.named_parameters()})
            assert len(info)==G and all(z['winner'].shape==(256,) for z in info);g.check()
        assert all(any(ph[k]['norm']>0 for ph in phase[1:]) for k in phase[0]),'DISCONNECTED_PARAMETER'
        xx=torch.from_numpy(fit['x'][ix[:128]]).cuda();export=m.export().cuda()
        with torch.inference_mode():
            a=m(xx);assert a.shape==(128,G,20,12,2) and torch.equal(a[:,0],m(xx,False)) and torch.equal(a[:,0],export(xx))
        # GT cannot enter forward; inputs and source signatures saved in receipt.
        before=state(m);sel={k:v[ix[:128]] for k,v in fit.items()};t=time.perf_counter()
        save_population(rows/'saved128',a.cpu().numpy(),sel,g);save_timing.append(dict(arm=arm,N=128,seconds=time.perf_counter()-t))
        assert equal(before,state(m));gpu_batches[arm]=phase;del m,opt,export,a;torch.cuda.empty_cache()
        results.append(fd_and_numpy(G,fit,ix));resume_reports.append(resume(G,fit,inn,ix,iy,rows))
        # Fresh throwaway optimizer, no selected model or benchmark reuse.
        m=new_model(42,G);opt=optimizer(m);x=torch.from_numpy(fit['x'][:256]).cuda();y=torch.from_numpy(fit['gt'][:256].astype(np.float32)).cuda();times=[]
        for b in range(36):
            torch.cuda.synchronize();t=time.perf_counter();step(m,opt,x,y);torch.cuda.synchronize()
            if b>=4:times.append(time.perf_counter()-t)
        throughput[arm]=dict(warmup=4,measured=32,mean=float(np.mean(times)),max=float(np.max(times)),parameter_count=sum(p.numel() for p in m.parameters()))
        del m,opt;torch.cuda.empty_cache();g.check(force=True)
    readers=bad_readers();g.check(force=True)
    write(HERE/'PREFLIGHT_GPU.json',dict(status='PASS',GPU8_batches=gpu_batches,independent_double=results,resume=resume_reports,readers_rejected=readers,
        bound_tied_query_counterfactual='PASS_Q1_mean3_bound_groups_value_and_sharedgradient_tol1e-12',Q1_original_nonzero_paths_scalar_all_gradients='EXACT',export_main_only='EXACT_same_backend_batch',GT_forward='No GT argument; fixed features only',zero_head_not_native_identity=True))
    write(HERE/'PREFLIGHT_RESOURCE.json',dict(observations=g.observations,own_private_peak=max(r['snapshot']['peak_private'] for r in g.observations),own_RSS_peak=max(r['snapshot']['peak_rss'] for r in g.observations)))
    write(HERE/'BENCHMARK_PARTIAL.json',dict(training=throughput,saved_pipeline128=save_timing,total_preflight_seconds=time.perf_counter()-start,
        fixture_optimizer_updates=2*(8+12+36),scientific_updates=0,full_budget_admission='PENDING_CPU_FULL_PIPELINE_AND_SHA_TIMING'))
    print('ACTUAL GPU PREFLIGHT PASS',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        p=HERE/f'PREFLIGHT_FAILURE_{time.time_ns()}.json';write(p,dict(at=now(),traceback=traceback.format_exc(),scientific_updates=0));raise
