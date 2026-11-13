from train import *
from independent import numpy_path_gradient
import shutil,traceback
def exact(a,b):
    if isinstance(a,torch.Tensor):assert a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,np.ndarray):assert a.dtype==b.dtype and np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b
def analytic_fixtures():
    rng=np.random.default_rng(20261027);n=np.cumsum(rng.normal(0,.04,(4,20,12,2)),axis=2);y=n[:,0]+rng.normal(0,.01,(4,12,2));y[2:]+=2.;p=n+rng.normal(0,.015,n.shape);sc=np.array([.3,.2,.1]);pp=torch.tensor(p,requires_grad=True);v,t=objective(pp,torch.from_numpy(y),torch.from_numpy(n),torch.from_numpy(sc));nv,nt=numpy_objective(p,y,n,sc);gg=torch.autograd.grad(v,pp)[0].numpy();ng=numpy_path_gradient(p,y,n,sc)
    assert abs(v.item()-nv)<=1e-10 and max(abs(t[k].item()-nt[k]) for k in t)<=1e-10 and abs(gg-ng).max()<=1e-10
    # Fixed K copies: soft surrogate may improve without improving literal minJ.
    j=np.r_[0.,np.ones(19)*4];a=-.1*np.log(np.exp(-j/.1).sum());b=-.1*np.log(np.ones(20).sum());assert b<a and min(j)==0
    return dict(independent_scalar_error=abs(v.item()-nv),full_path_gradient_error=float(abs(gg-ng).max()),native_good_and_failure_keep_masks=True,copy_surrogate_not_coverage_proven=True)
def FD(data):
    report=[];x,n,y=data.batch(np.arange(4),'cpu');x=x.double();y=n[:,0]+.37
    for arm in ARMS:
        m,_,_=initialize(20261027,arm,data,'cpu',True)
        with torch.no_grad():m.offset.weight.normal_(0,.02);m.offset.bias.fill_(.01)
        def faces(p,aux):
            _,_,j=candidate_errors(p,y);_,_,nj=candidate_errors(n,y);assert float(torch.linalg.vector_norm(aux['anchors'],dim=-1).max())<.9
            return ((nj<=1).tolist(),(j>nj).tolist(),(mean_motion(p)>mean_motion(n)).tolist())
        p,aux=m(x,n);base_faces=faces(p,aux);value,_=objective(p,y,n,m.motion_scales);value.backward();grads=[q.grad.clone() for q in m.parameters()];ng=numpy_path_gradient(p.detach().numpy(),y.numpy(),n.numpy(),m.motion_scales.numpy());nv,_=numpy_objective(p.detach().numpy(),y.numpy(),n.numpy(),m.motion_scales.numpy());assert abs(value.item()-nv)<=1e-10
        m.zero_grad(set_to_none=True);m(x,n)[0].backward(torch.from_numpy(ng));chain=max(float(abs(q.grad-g).max()) for q,g in zip(m.parameters(),grads));assert chain<=1e-10
        base=[q.detach().clone() for q in m.parameters()];gen=torch.Generator().manual_seed(20261027);checks=[]
        for direction in range(4):
            ds=[torch.randn(q.shape,generator=gen,dtype=torch.float64) for q in m.parameters()];norm=torch.sqrt(sum((q*q).sum() for q in ds));ds=[q/norm for q in ds];ref=sum(float((q*g).sum()) for q,g in zip(ds,grads))
            for eps in (1e-4,1e-5):
                vals=[]
                for sign in (1,-1):
                    with torch.no_grad():
                        for q,b,d in zip(m.parameters(),base,ds):q.copy_(b+sign*eps*d)
                        pred,aux=m(x,n);assert faces(pred,aux)==base_faces,'FD_ACTIVE_FACE_SWITCH_NO_EPS_SEARCH';v,_=objective(pred,y,n,m.motion_scales);vals.append(v.item())
                error=abs((vals[0]-vals[1])/(2*eps)-ref);assert error<=1e-6+1e-3*abs(ref);checks.append(dict(direction=direction,eps=eps,error=error,reference=ref))
                with torch.no_grad():
                    for q,b in zip(m.parameters(),base):q.copy_(b)
        report.append(dict(arm=arm,chain_error=chain,FD=checks,all_active_masks_EXACT=base_faces,fixed_nonzero_fixture_no_epsilon_search=True))
    return report
def resume(folder,g):
    d=Data(513);report=[]
    for arm in ARMS:
        ct=dict(contract(42,arm),fixture513=True,epochs=2);m,o,r=initialize(42,arm,d);st=fresh()
        while st['epoch']<2:step(m,o,r,d,st)
        cont=snapshot(m,o,r,st,ct);m,o,r=initialize(42,arm,d);st=fresh()
        while st['updates']<19:step(m,o,r,d,st)
        f=folder/(arm+'_interrupt19.pt');torch.save(snapshot(m,o,r,st,ct),f);m,o,r=initialize(42,arm,d);st=restore(f,m,o,r,ct)
        while st['epoch']<2:step(m,o,r,d,st)
        exact(cont,snapshot(m,o,r,st,ct));report.append(dict(arm=arm,actual513train_34updates_interrupt19_ALLstate_RNG_history_EXACT=True));g.check()
    good=folder/'completed_reader';good.mkdir();torch.save(snapshot(m,o,r,st,ct),good/'checkpoint.pt');np.save(good/'outputs.npy',np.arange(3));finish(good,ct,updates=34);reader(good,ct,2);reject=[]
    for key in ('source','split','stats','arm','prepared'):
        bad={**ct,key:'bad'}
        try:reader(good,bad,2)
        except AssertionError:reject.append(key)
        else:raise AssertionError('Reader accepted bad '+key)
    bad=folder/'bad_output';shutil.copytree(good,bad);np.save(bad/'outputs.npy',np.ones(3))
    try:reader(bad,ct,2)
    except AssertionError:reject.append('output')
    else:raise AssertionError('Reader accepted bad output')
    bad=folder/'bad_checkpoint';shutil.copytree(good,bad);z=torch.load(bad/'checkpoint.pt',map_location='cpu',weights_only=False);z['contract']['source']='bad';torch.save(z,bad/'checkpoint.pt');doc=read_json(bad/'done.json');doc['outputs']['checkpoint.pt']=sha256(bad/'checkpoint.pt');write_json(bad/'done.json',doc)
    try:reader(bad,ct,2)
    except AssertionError:reject.append('checkpoint_embedded')
    else:raise AssertionError('Bad checkpoint accepted')
    try:restore(good/'checkpoint.pt',m,o,r,{**ct,'source':'bad'})
    except ValueError:reject.append('resume')
    else:raise AssertionError('Bad resume accepted')
    return report,reject
def main():
    start=time.perf_counter();h=verify();g=guard();folder=HERE/('gpu_fixture_'+str(time.time_ns()));folder.mkdir();d=Data();an=analytic_fixtures();fd=FD(d);res,reject=resume(folder,g);models={};first=None;benchmark={};depend=[]
    for arm in ARMS:
        m,o,r=initialize(42,arm,d);state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        if first is None:first=state
        else:exact(first,state)
        st=fresh();x,n,y=d.batch(np.arange(32),'cuda');p,aux=m(x,n);assert torch.equal(p,n.double());assert not aux['residual'].any();connected=set()
        for j in range(8):
            step(m,o,r,d,st);connected|={k for k,v in m.named_parameters() if v.grad.abs().sum()>0}
            if arm=='B1':assert not m.qkv.weight.grad[:64].any() and not m.qkv.bias.grad[:64].any()
        assert connected==set(dict(m.named_parameters()))
        m.eval()
        with torch.no_grad():
            p,aux=m(x,n);xp=x.clone();xp[:,1]+=.1;pp,_=m(xp,n);cross=float(abs(pp[:,0]-p[:,0]).max());perm=torch.arange(19,-1,-1,device='cuda');rp,_=m(x[:,perm],n[:,perm]);perr=float(abs(rp-p[:,perm]).max());assert perr<=2e-7
            if arm=='B1':assert cross==0
            else:assert cross>1e-12
        timings=[]
        for j in range(36):
            torch.cuda.synchronize();a=time.perf_counter();step(m,o,r,d,st);torch.cuda.synchronize()
            if j>=4:timings.append(time.perf_counter()-a)
        benchmark[arm]=dict(warm=4,measured=32,mean_seconds=float(np.mean(timings)));depend.append(dict(arm=arm,actualGPU8batch32=True,native_initial_identity=True,phased_active_tensors_positivefinite=True,B1_QK_structurally_zero_expected=arm=='B1',cross_candidate_dependency=cross,permutation_error=perr));models[arm]=m;g.check()
    from evaluate import run
    a=time.perf_counter();f=run(42,'opt',models,folder/'saved128',128,g);elapsed=time.perf_counter()-a;assert verify()==h
    write_json(HERE/'PREFLIGHT_GPU.json',dict(status='PASS',source=h,seconds=time.perf_counter()-start,analytic=an,parameter_FD=fd,resume=res,reader_rejections=reject,completed_SHA_skip_actual=True,paired_all_init_params_buffers=True,dependency=depend,benchmark=benchmark,saved128_folder=str(f.relative_to(HERE)),saved128_GPU_seconds=elapsed,parameters=sum(p.numel() for p in models['B1'].parameters()),train_count=d.count,steps_per_run=8*((d.count+31)//32),not_formal_or_fulltrain_twoepoch=True));resource(HERE/'PREFLIGHT_GPU_RESOURCE.json',g);print('PASS GPU',time.perf_counter()-start,flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as ex:failure('GPU_preflight',ex);raise
