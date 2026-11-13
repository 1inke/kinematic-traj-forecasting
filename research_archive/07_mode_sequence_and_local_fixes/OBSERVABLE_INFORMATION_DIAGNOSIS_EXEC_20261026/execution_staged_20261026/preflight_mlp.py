"""Actual GPU/FD/exact-resume tests, not scientific fits or old training replay."""
from mlp import *
import ast,shutil

def fd_checks():
    # Reuse the exact registered old parameter fixture without its legacy main/imports.
    tree=ast.parse((OLD/'preflight.py').read_text(encoding='utf-8'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parameter_fd')
    ns=dict(torch=torch,np=np,seed_all=seed_all,ActionEditor=ed.ActionEditor,loss=ed.loss,numpy_loss_and_score_grad=ed.numpy_loss_and_score_grad,ARMS=('I','S'))
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<exact-old-parameter-fd>','exec'),ns);original=ns['parameter_fd']()
    seed_all(20261026);m=ed.ActionEditor(np.zeros(323),np.ones(323)).double()
    with torch.no_grad():
        for p in m.parameters():p.fill_(.01)
    x=torch.zeros(4,32,324,dtype=torch.float64);x[:,:,0]=torch.linspace(-.5,2,32);x[:,0,323]=1
    y=torch.zeros(4,32,dtype=torch.bool);y[:,1:17]=True;w=torch.full((4,32),1/32,dtype=torch.float64);s=m(x);loss=weighted_loss(s,w,y);loss.backward();gr=[p.grad.clone() for p in m.parameters()]
    ss=s.detach().numpy();yy=y.numpy();ww=w.numpy();nv=np.sum(ww*np.where(yy,np.logaddexp(0,-ss),np.logaddexp(0,ss)),1).mean();ng=ww*(1/(1+np.exp(-ss))-yy)/4
    assert abs(nv-float(loss))<1e-10;m.zero_grad(set_to_none=True);m(x).backward(torch.from_numpy(ng));assert max(float(abs(a-p.grad).max()) for a,p in zip(gr,m.parameters()))<1e-10
    base=[p.detach().clone() for p in m.parameters()];gen=torch.Generator().manual_seed(20261026);rec=[]
    for i in range(4):
        ds=[torch.randn(p.shape,generator=gen,dtype=torch.float64) for p in m.parameters()];norm=torch.sqrt(sum((d*d).sum() for d in ds));ds=[d/norm for d in ds];ref=sum(float((a*d).sum()) for a,d in zip(gr,ds))
        for eps in (1e-4,1e-5):
            vals=[]
            for sign in (1,-1):
                with torch.no_grad():
                    for p,b,d in zip(m.parameters(),base,ds):p.copy_(b+sign*eps*d)
                    vals.append(float(weighted_loss(m(x),w,y)))
            with torch.no_grad():
                for p,b in zip(m.parameters(),base):p.copy_(b)
            numeric=(vals[0]-vals[1])/(2*eps);assert abs(ref-numeric)<=1e-6+1e-3*abs(ref);rec.append(dict(direction=i,eps=eps,error=abs(ref-numeric)))
    return dict(original_I_S=original,weighted_BCE=rec)

def main():
    folder=HERE/('preflight_mlp_'+str(time.time_ns()));folder.mkdir();candidate=verify();t=time.perf_counter();report=dict(source=candidate,at=now(),FD=fd_checks());timings={};resumes=[];models={}
    initial=None
    for kind,arm,limit,steps in (('tiny','I',None,8),('tiny','S',None,8),('probe','M_full',513,34)):
        d=Data(kind,42,limit);eng=Engine(d,arm,42)
        original=torch.load(OLD/'prepared_v1/initial_seed42.pt',map_location='cpu',weights_only=True);exact(original,eng.model.state_dict())
        if initial is None:initial=copy.deepcopy(eng.model.state_dict())
        else:exact(initial,eng.model.state_dict())
        x,a,y=d.batch(np.arange(32));scores=eng.model(x);assert not scores.any()
        if kind=='tiny':assert not ed.choose(scores,a).any();assert np.array_equal(normalize(d.x[:32],42),torch.cat(((x[...,:323]-eng.model.feature_mean)/eng.model.feature_std,x[...,323:]),-1).cpu().numpy())
        connected={n:False for n,p in eng.model.named_parameters()}
        def check(e):
            for n,p in e.model.named_parameters():connected[n]|=bool(torch.isfinite(p.grad).all() and (p.grad!=0).any())
        eng.run(8,check,False);assert all(connected.values());models[arm]=eng
        # Exact interruption mid-epoch, including sampler position/order and all RNG.
        continuous=Engine(d,arm,42);continuous.run(steps,progress=False);expected=continuous.snapshot(dict(fixture=arm,source=candidate))
        partial=Engine(d,arm,42);partial.run(3,progress=False);ck=folder/f'{arm}_resume.pt';torch.save(partial.snapshot(dict(fixture=arm,source=candidate)),ck)
        resumed=Engine(d,arm,42);resumed.restore(torch.load(ck,map_location='cpu',weights_only=False),dict(fixture=arm,source=candidate));resumed.run(steps,progress=False);exact(expected,resumed.snapshot(dict(fixture=arm,source=candidate)))
        try:resumed.restore(torch.load(ck,map_location='cpu',weights_only=False),dict(fixture=arm,source='BAD'))
        except ValueError:pass
        else:raise AssertionError('Bad resume accepted')
        resumes.append(dict(kind=kind,arm=arm,windows=d.n,epochs=2,updates=steps,params_buffers_optimizer_CPU_CUDA_loader_numpy_python_RNG_history_sampler='EXACT',actual_GPU8_connected=connected))
        # 4 warm and32 measured genuine optimizer updates, throwaway only.
        bench=Engine(d,arm,42);bench.run(4,progress=False);torch.cuda.synchronize();start=time.perf_counter();bench.run(36,progress=False);torch.cuda.synchronize();timings[arm]=(time.perf_counter()-start)/32
        del continuous,partial,resumed,bench;G.check()
    # Actual saved128 scores + immutable original observations, independent CPU audit.
    from evaluate_saved import edit,independent_check,build_features
    with np.load(HERE/'prepared/tiny.npz') as q:z={k:q[k] for k in q.files};x=z.pop('features');z.pop('opt_row')
    for arm in ('I','S'):
        e=models[arm];e.model.eval();s=[];before=copy.deepcopy(e.model.state_dict())
        with torch.inference_mode():
            for i in range(0,128,32):s.append(e.model(torch.as_tensor(x[i:i+32],device=DEVICE)).cpu().numpy())
        exact(before,e.model.state_dict());s=np.concatenate(s);chosen,paths=edit(z,s);p=folder/f'{arm}_saved128.npz';save_npz(p,scores=s,chosen=chosen,paths=paths)
        with np.load(p) as q:independent_check(z,q['scores'],q['chosen'],q['paths'])
    # A new feature reconstruction matches ALL original raw opt feature bytes.
    with np.load(HERE/'prepared/observations_opt.npz') as q:obs={k:q[k] for k in q.files}
    old=next(aux_chunks(42,'opt'));rebuilt=build_features(old,obs,0);assert np.array_equal(rebuilt,old_features(42,np.arange(128)))
    # Actual done/source and damaged-output rejection, preserving all fixtures.
    fixture=folder/'reader';fixture.mkdir();save_npz(fixture/'output.npz',a=np.arange(3));c=dict(source=candidate,IDs='fixture',stats='fixture',pool=42,updates=8);torch.save(models['I'].snapshot(c),fixture/'final.pt');finish(fixture,c);read_model(fixture,c)
    badc=dict(c,source='BAD')
    try:completed(fixture,badc)
    except ValueError:pass
    else:raise AssertionError('Bad source accepted')
    bad=folder/'bad_output';shutil.copytree(fixture,bad);np.savez_compressed(bad/'output.npz',a=np.arange(4))
    try:completed(bad,c)
    except ValueError:pass
    else:raise AssertionError('Bad output accepted')
    for field in ('IDs','stats'):
        try:read_model(fixture,dict(c,**{field:'BAD'}))
        except ValueError:pass
        else:raise AssertionError('Bad '+field+' accepted')
    badmodel=folder/'bad_checkpoint';shutil.copytree(fixture,badmodel);state=torch.load(badmodel/'final.pt',map_location='cpu',weights_only=False);state['contract']['source']='BAD';torch.save(state,badmodel/'final.pt');finish(badmodel,c)
    try:read_model(badmodel,c)
    except ValueError:pass
    else:raise AssertionError('Bad embedded model accepted')
    assert verify()==candidate;resource_report(folder,G);report.update(status='PASS',resumes=resumes,seconds=time.perf_counter()-t,step_seconds=timings,old_initial_allparams_buffers_EXACT=True,normalized_actual_float32_bytes_EXACT=True,saved128_allquality_clusters_motion=True,actual_full_feature_rebuild_EXACT=True,actual_readers_reject_bad_source_output_resume=True,resource=read_json(folder/'RESOURCE.json'),no_formal_fit=True)
    write_json(folder/'PREFLIGHT.json',report);write_json(HERE/'PREFLIGHT_MLP.json',dict(report,folder=folder.name));print('PASS MLP PREFLIGHT',report['seconds'],timings,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:fail(e);raise
