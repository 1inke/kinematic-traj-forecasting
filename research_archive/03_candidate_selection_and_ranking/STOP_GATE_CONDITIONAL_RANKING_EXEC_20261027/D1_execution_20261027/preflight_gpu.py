"""Actual current-source loss/gradient/resume/manifest/GPU benchmark gates."""
from train_paired import *
import shutil,traceback
def exact(a,b,path='root'):
    if isinstance(a,torch.Tensor):assert a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu()),path
    elif isinstance(a,np.ndarray):assert a.dtype==b.dtype and np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:exact(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):exact(x,y,path+'/'+str(i))
    else:assert a==b,path
def objective_fixtures():
    e=np.zeros((9,401),bool);e[:,0]=True;t=np.zeros_like(e);t[:,0]=True;e[1:,1:7]=True
    t[2,0]=False;t[2,1:7]=True
    for row in (3,4,5,6,7,8):t[row,0]=False;t[row,1:3]=True
    s=np.zeros(e.shape);s[1,0]=4;s[2,1]=3;s[3,0]=10;s[4,1]=10;s[5,1:3]=2;s[5,3:5]=3;s[6,0]=-1000;s[6,1]=1000;s[7,0]=1000;s[7,1]=-1000
    records=[]
    for arm in ARMS:
        score=torch.tensor(s,dtype=torch.float64,requires_grad=True);ee=torch.from_numpy(e);tt=torch.from_numpy(t);value=loss(score,ee,tt,arm);grad=torch.autograd.grad(value,score)[0].numpy();v,g=numpy_loss_gradient(s,e,t,arm)
        assert abs(float(value)-v)<=1e-10 and abs(grad-g).max()<=1e-10
        if arm in ('S','D'):
            assert abs(float(loss(score+17,ee,tt,arm))-float(value))<=1e-10
            assert torch.equal(choose(score+17,ee),choose(score,ee))
        if arm in ('I','S'):assert torch.equal(value,legacy['loss'](score,ee,tt,arm))
        records.append(dict(arm=arm,scalar_error=abs(float(value)-v),score_gradient_error=float(abs(grad-g).max()),tie_first_and_extremes=True))
    score=torch.zeros((1,401),dtype=torch.float64,requires_grad=True);ee=torch.zeros((1,401),dtype=torch.bool);ee[:,0]=True;tt=ee.clone()
    v=loss(score,ee,tt,'D');gg=torch.autograd.grad(v,score)[0];assert v==0 and not gg.any()
    # Positive opportunity and high STOP => STOP-down gradient, undiluted .5 gate.
    ee=torch.ones((1,401),dtype=torch.bool);tt=torch.zeros_like(ee);tt[:,1]=True
    score=torch.zeros((1,401),dtype=torch.float64,requires_grad=True);v=loss(score,ee,tt,'D');gg=torch.autograd.grad(v,score)[0];assert gg[0,0]==.25
    tt.zero_();tt[:,0]=True;v=loss(score,ee,tt,'D');gg=torch.autograd.grad(v,score)[0];assert gg[0,0]==-.25
    # D gate/rank depend on maxima, not the number of duplicate negatives.
    ss=torch.zeros((1,401),dtype=torch.float64);ss[:,1]=3;ss[:,2:]=-2
    aa=torch.zeros_like(ee);aa[:,:3]=True;bb=ee.clone();tt.zero_();tt[:,1]=True
    assert torch.equal(decomposed(ss,aa,tt)[0],decomposed(ss,bb,tt)[0])
    return records
def parameter_fixture():
    seed_all(20261027);m=ActionEditor(np.zeros(323),np.ones(323)).double()
    with torch.no_grad():
        for p in m.parameters():p.fill_(.01)
    x=torch.zeros((4,401,324),dtype=torch.float64);x[:,:,0]=-.5;x[:,0,0]=.5;x[:,0,323]=1;x[:,1,0]=2;x[:,2,0]=1
    e=torch.ones((4,401),dtype=torch.bool);t=torch.zeros_like(e);t[:,1:3]=True;base=[p.detach().clone() for p in m.parameters()];g=torch.Generator().manual_seed(20261027);directions=[]
    for j in range(4):
        ds=[torch.randn(p.shape,generator=g,dtype=torch.float64) for p in m.parameters()];n=torch.sqrt(sum((v*v).sum() for v in ds));directions.append([v/n for v in ds])
    reports=[]
    for arm in ARMS:
        m.zero_grad(set_to_none=True);scores=m(x);v=loss(scores,e,t,arm);v.backward();der=[p.grad.clone() for p in m.parameters()];nv,ng=numpy_loss_gradient(scores.detach().numpy(),e.numpy(),t.numpy(),arm)
        assert abs(v.item()-nv)<=1e-10;m.zero_grad(set_to_none=True);m(x).backward(torch.from_numpy(ng));assert max(float(abs(p.grad-d).max()) for p,d in zip(m.parameters(),der))<=1e-10
        if arm in ('I','S'):
            m.zero_grad(set_to_none=True);ov=legacy['loss'](m(x),e,t,arm);ov.backward();assert v.item()==ov.item() and all(torch.equal(p.grad,d) for p,d in zip(m.parameters(),der))
        face=faces(scores.detach(),e,t);checks=[]
        for j,ds in enumerate(directions):
            ref=sum(float((a*b).sum()) for a,b in zip(der,ds))
            for eps in (1e-4,1e-5):
                vv=[]
                for sign in (1,-1):
                    with torch.no_grad():
                        for p,b,d in zip(m.parameters(),base,ds):p.copy_(b+sign*eps*d)
                        ss=m(x);assert faces(ss,e,t)==face,'FD_ACTIVE_FACE_SWITCH_NO_EPS_SEARCH';vv.append(float(loss(ss,e,t,arm)))
                numeric=(vv[0]-vv[1])/(2*eps);assert abs(numeric-ref)<=1e-6+1e-3*abs(ref)
                checks.append(dict(direction=j,eps=eps,error=abs(numeric-ref),reference=ref,active_faces_EXACT=True))
                with torch.no_grad():
                    for p,b in zip(m.parameters(),base):p.copy_(b)
        reports.append(dict(arm=arm,checks=checks,independent_full_parameter_chain=True,legacy_IS_all_parameter_grad_EXACT=arm!='D'))
    return reports
def resume(folder,g):
    data=OptData(42,513);reports=[];last=None
    for arm in ARMS:
        ct=contract(42,arm,source0(),513,2);m,o,r=initialize(42,data);st=fresh_state()
        while st['updates']<34:step(m,o,r,data,arm,st)
        continuous=snapshot(m,o,r,st,ct)
        m,o,r=initialize(42,data);st=fresh_state()
        while st['updates']<19:step(m,o,r,data,arm,st)
        f=folder/f'{arm}_interrupt19.pt';torch.save(snapshot(m,o,r,st,ct),f)
        m,o,r=initialize(42,data);st=restore(f,m,o,r,ct)
        while st['updates']<34:step(m,o,r,data,arm,st)
        exact(continuous,snapshot(m,o,r,st,ct));reports.append(dict(arm=arm,actual513opt34updates_interrupt19=True,all_parameters_statbuffers_optimizer_CPU_CUDA_loader_Python_NumPy_RNG_ordercursor_history_EXACT=True));last=(m,o,r,st,ct)
        g.check()
    m,o,r,st,ct=last;f=folder/'reader_complete';f.mkdir();torch.save(snapshot(m,o,r,st,ct),f/'checkpoint.pt');np.save(f/'output.npy',np.arange(5));finish(f,ct);validate_completed(f,ct);rejected=[]
    for key in ('source','split','statistics','features','labels','orders','candidate_pool'):
        bad=dict(ct);bad[key]='bad'
        try:validate_completed(f,bad)
        except (ValueError,AssertionError):rejected.append(key)
        else:raise AssertionError('Reader accepted bad '+key)
    for key in ('checkpoint','output'):
        bad=folder/('bad_'+key);shutil.copytree(f,bad)
        if key=='output':np.save(bad/'output.npy',np.ones(5))
        else:
            z=torch.load(bad/'checkpoint.pt',map_location='cpu',weights_only=False);z['contract']['source']='bad';torch.save(z,bad/'checkpoint.pt');d=read_json(bad/'done.json');d['outputs']['checkpoint.pt']=sha256(bad/'checkpoint.pt');write_json(bad/'done.json',d)
        try:validate_completed(bad,ct)
        except (ValueError,AssertionError):rejected.append(key)
        else:raise AssertionError('Reader accepted corrupt '+key)
    try:restore(f/'checkpoint.pt',m,o,r,{**ct,'source':'bad'})
    except ValueError:rejected.append('resume')
    else:raise AssertionError('Bad resume accepted')
    data.close();return reports,rejected
def main():
    start=time.perf_counter();verify_prepared();g=guard();h=source0();assert verify_manifest(HERE)==h
    folder=HERE/('preflight_gpu_'+str(time.time_ns()));folder.mkdir();obj=objective_fixtures();fd=parameter_fixture();res,reject=resume(folder,g)
    data=OptData(42);models={};paired=None;gpu=[];bench={}
    for arm in ARMS:
        m,o,r=initialize(42,data);init={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        if paired is None:paired=init
        else:exact(paired,init)
        x,e,t=data.batch(np.arange(32),'cuda');assert not m(x).any() and not choose(m(x),e).any()
        state=fresh_state();connected=set()
        for j in range(8):
            step(m,o,r,data,arm,state)
            if j==0:assert all(not p.grad.any() for n,p in m.named_parameters() if not n.startswith('net.4'))
            connected|={n for n,p in m.named_parameters() if p.grad.abs().sum()>0}
        assert connected==set(dict(m.named_parameters()));models[arm]=m;gpu.append(dict(arm=arm,actual8batch32_full401=True,phased_all_tensor_positivefinite=True,all_initial_params_buffers_paired=True,native_STOP_identity=True))
        if arm in ('I','S'):
            x,e,t=data.batch(np.arange(32),'cuda');m.zero_grad(set_to_none=True);v=loss(m(x),e,t,arm);v.backward();gg=[p.grad.clone() for p in m.parameters()];m.zero_grad(set_to_none=True);oldv=legacy['loss'](m(x),e,t,arm);oldv.backward()
            assert torch.equal(v,oldv) and all(torch.equal(p.grad,a) for p,a in zip(m.parameters(),gg))
        timing=[]
        for j in range(36):
            torch.cuda.synchronize();a=time.perf_counter();step(m,o,r,data,arm,state);torch.cuda.synchronize()
            if j>=4:timing.append(time.perf_counter()-a)
        bench[arm]=dict(warm=4,measured=32,mean_seconds=float(np.mean(timing)),max_seconds=max(timing));g.check()
    from evaluate_saved import save_models
    a=time.perf_counter();out=save_models(42,'opt',128,models,folder/'saved128');save_seconds=time.perf_counter()-a;data.close();assert verify_manifest(HERE)==h
    write_json(HERE/'PREFLIGHT_GPU.json',dict(status='PASS',source=h,seconds=time.perf_counter()-start,objective_fixtures=obj,FD=fd,resume=res,readers_actual_rejected=reject,completed_model_SHA_skip_actual=True,GPU8=gpu,benchmark=bench,saved128_folder=str(out.relative_to(HERE)),saved128_GPU_seconds=save_seconds,not_production_or_full4096_two_epochs=True))
    resource(HERE/'PREFLIGHT_GPU_RESOURCE.json',g);print('PASS CURRENT GPU PREFLIGHT',time.perf_counter()-start,flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as ex:
        out=HERE/('failure_gpu_'+str(time.time_ns()));out.mkdir();write_json(out/'FAILURE.json',dict(error=str(ex),traceback=traceback.format_exc(),preserve_all=True))
        for p in HERE.glob('*.py'):shutil.copyfile(p,out/p.name)
        raise
