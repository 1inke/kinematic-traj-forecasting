"""Actual GPU/math/resume/saved-pipeline preflight for the new three-objective study."""
import copy
import hashlib
import io
import json
import time
import traceback
import numpy as np
import common as c
import torch
from scipy.special import erf
from model import ParameterRegressor,window_mse
from train import seed_everything,infer,checkpoint_new,checkpoint_load_trusted
from preflight_checks import trees_equal
from objectives import objective,numpy_loss_and_grad
from trainer import Engine
from inference import predict_paths
from evaluate import sample_metrics,evaluate_population

def tensors(d,device='cuda:0',double=False):
    ty=torch.float64 if double else torch.float32
    return [torch.as_tensor(d[k] if k!='native' else d['x']['native'],device=device,dtype=ty if k in ('z','target') else torch.float64)
            for k in ('z','target','native','gt','nj','best')]

def parameters_numpy(m):return {k:v.detach().cpu().numpy().copy() for k,v in m.named_parameters()}

def independent_network(params,x,dout=None):
    x=np.asarray(x,float).reshape(-1,40)
    w0,b0,w1,b1,w2,b2=[params['layers.'+key] for key in ('0.weight','0.bias','2.weight','2.bias','4.weight','4.bias')]
    gelu=lambda z:.5*z*(1+erf(z/np.sqrt(2)))
    prime=lambda z:.5*(1+erf(z/np.sqrt(2)))+z*np.exp(-z*z/2)/np.sqrt(2*np.pi)
    z0=x@w0.T+b0;a0=gelu(z0);z1=a0@w1.T+b1;a1=gelu(z1);u=np.tanh(a1@w2.T+b2)
    if dout is None:return u.reshape(-1,20,2)
    g2=np.asarray(dout).reshape(-1,2)*(1-u*u);g1=(g2@w2)*prime(z1);g0=(g1@w1)*prime(z0)
    return dict(zip(('layers.0.weight','layers.0.bias','layers.2.weight','layers.2.bias','layers.4.weight','layers.4.bias'),
        (g0.T@x,g0.sum(0),g1.T@a0,g1.sum(0),g2.T@a1,g2.sum(0))))

def fingerprint(activity):return {k:v.detach().cpu().numpy().copy() for k,v in activity.items()}

def smooth_fixture():
    rng=np.random.default_rng(20261030);x=rng.normal(0,.1,(4,20,40));target=np.full((4,20,2),.05)
    end=np.array([.10,.12,.80,.90])[:,None]+np.arange(20)[None]*.002
    native=np.zeros((4,20,12,2));native[:,:,:,0]=end[:,:,None]*(np.arange(1,13)/12)
    gt=np.zeros((4,12,2));d=np.linalg.norm(native,axis=-1);nj=np.maximum(d.mean(-1)/.2,d[...,-1]/.4)
    return dict(z=x,target=target,x=dict(native=native),gt=gt,nj=nj,best=np.maximum(nj-.05,0))

def mathematical_tests(real8,out):
    report=[];fixtures={'smooth_fixed':smooth_fixture(),'real_first8':real8}
    for name,d in fixtures.items():
        seed_everything(42,True);m=ParameterRegressor().to(device='cuda:0',dtype=torch.float64)
        with torch.no_grad():
            rng=torch.Generator(device='cpu').manual_seed(20261030)
            m.layers[-1].weight.copy_((torch.randn(m.layers[-1].weight.shape,generator=rng,dtype=torch.float64)*.002).cuda())
            m.layers[-1].bias.copy_(torch.tensor([.20,.15],device='cuda',dtype=torch.float64))
        xx,yy,n,gt,nj,best=tensors(d,double=True);params=parameters_numpy(m)
        reference_u=independent_network(params,d['z']);assert np.max(np.abs(reference_u-m(xx).detach().cpu().numpy()))<1e-10
        for arm in ('M','R','RS'):
            m.zero_grad(set_to_none=True);loss,parts,act=objective(m(xx),yy,n,gt,nj,best,arm);loss.backward()
            ref,dout,refparts=numpy_loss_and_grad(reference_u,d['target'],d['x']['native'],d['gt'],d['best'],arm)
            grads=independent_network(params,d['z'],dout)
            maxg=max(float(np.max(np.abs(v.grad.cpu().numpy()-grads[k]))) for k,v in m.named_parameters())
            scalarerr=abs(float(loss.detach().cpu())-ref)
            assert scalarerr<=1e-10 and maxg<=1e-10,(name,arm,scalarerr,maxg)
            oldparams=[v.detach().clone() for v in m.parameters()];analytic=[v.grad.detach().clone() for v in m.parameters()]
            rng=np.random.default_rng(20261030);directions=[]
            for repeat in range(4):
                ds=[torch.tensor(rng.normal(size=tuple(v.shape)),device='cuda',dtype=torch.float64) for v in m.parameters()]
                norm=torch.sqrt(sum((v*v).sum() for v in ds));directions.append([v/norm for v in ds])
            fd=[];base_activity=fingerprint(act)
            for k,ds in enumerate(directions):
                an=float(sum((a*b).sum() for a,b in zip(analytic,ds)).cpu())
                for eps in (1e-4,1e-5):
                    values=[];activities=[]
                    for sign in (1,-1):
                        with torch.no_grad():
                            for p,v,dr in zip(m.parameters(),oldparams,ds):p.copy_(v+sign*eps*dr)
                            ll,_,aa=objective(m(xx),yy,n,gt,nj,best,arm);values.append(float(ll.cpu()));activities.append(fingerprint(aa))
                    same=all(trees_equal(base_activity,a) for a in activities);numeric=(values[0]-values[1])/(2*eps)
                    status='PASS' if same else 'NONSMOOTH_CHECK_INCONCLUSIVE'
                    if same:assert abs(numeric-an)<=1e-6+1e-3*abs(an),(name,arm,eps,numeric,an)
                    elif name=='smooth_fixed':raise AssertionError('prespecified smooth fixture switched; no eps search')
                    fd.append(dict(direction=k,epsilon=eps,analytic=an,numeric=numeric,error=abs(numeric-an),activity_unchanged=same,status=status))
            with torch.no_grad():
                for p,v in zip(m.parameters(),oldparams):p.copy_(v)
            c.npz_new(out/f'{name}_{arm}_activity.npz',**base_activity)
            report.append(dict(fixture=name,arm=arm,scalar_max_error=scalarerr,all4770_grad_max_error=maxg,FD=fd))
    # Unit tests for exact nonsmooth conventions, not cross-face finite differences.
    u=torch.zeros((2,20,2),dtype=torch.float64,device='cuda',requires_grad=True)
    native=torch.zeros((2,20,12,2),dtype=torch.float64,device='cuda');gt=torch.zeros((2,12,2),dtype=torch.float64,device='cuda')
    target=torch.zeros_like(u);nj=torch.zeros((2,20),device='cuda',dtype=torch.float64);best=nj.clone()
    for arm in ('M','R','RS'):
        loss,parts,act=objective(u,target,native,gt,nj,best,arm);grad=torch.autograd.grad(loss,u,retain_graph=True)[0]
        assert loss==0 and torch.equal(grad,torch.zeros_like(grad)) and (act['component']==0).all()
    a=torch.tensor([[1.,1.,2.]],device='cuda',requires_grad=True);a.min(1).values.sum().backward();assert torch.equal(a.grad,torch.tensor([[1.,0.,0.]],device='cuda'))
    h=torch.tensor([0.],device='cuda',requires_grad=True);torch.relu(h).square().backward();assert h.grad.item()==0
    for fixture in ('all_good','all_bad','mixed'):
        d=smooth_fixture()
        if fixture=='all_good':d['x']['native'][:]=d['x']['native'][0]
        if fixture=='all_bad':d['x']['native'][:]=d['x']['native'][3]
        de=np.linalg.norm(d['x']['native']-d['gt'][:,None],axis=-1);d['nj']=np.maximum(de.mean(-1)/.2,de[...,-1]/.4);d['best']=d['nj'].copy()
        zz,yy,n,gt,nj,best=tensors(d,double=True);uu=torch.full((4,20,2),.1,device='cuda',dtype=torch.float64,requires_grad=True)
        ll,parts,_=objective(uu,yy,n,gt,nj,best,'RS');ll.backward();assert torch.isfinite(ll) and torch.isfinite(uu.grad).all()
        if fixture=='all_good':assert parts['rescue']==0
        if fixture=='all_bad':assert parts['retention']==0
    return dict(status='PASS_INDEPENDENT_DOUBLE_SCALAR_ALL4770_GRADS_FIXED_FD_AND_EXACT_SUBGRADIENTS',tests=report,
        empty_groups=True,max_ADE_first=True,min_firstslot=True,norm0_zero_subgradient=True,hinge0_zero=True)

def gpu_bridge(d):
    results=[];initial=[]
    xx,yy,n,gt,nj,best=tensors(d)
    for arm in ('M','R','RS'):
        seed_everything(42,True);m=ParameterRegressor().cuda();initial.append(copy.deepcopy(m.state_dict()))
        assert (m(xx)==0).all()
        for phase in ('zero_last','fixed_nonzero_last'):
            if phase=='fixed_nonzero_last':
                with torch.no_grad():m.layers[-1].weight.copy_(torch.linspace(-.01,.01,m.layers[-1].weight.numel(),device='cuda').reshape_as(m.layers[-1].weight))
            m.zero_grad(set_to_none=True);u=m(xx);loss,parts,activity=objective(u,yy,n,gt,nj,best,arm);loss.backward()
            grads={key:float(p.grad.abs().max().cpu()) for key,p in m.named_parameters()}
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            if phase=='fixed_nonzero_last':assert all(v>0 for v in grads.values())
            if arm=='M':
                got=[p.grad.clone() for p in m.parameters()];value=loss.detach().clone();m.zero_grad(set_to_none=True)
                old=window_mse(m(xx),yy);old.backward();assert torch.equal(old,value) and all(torch.equal(p.grad,v) for p,v in zip(m.parameters(),got))
            results.append(dict(arm=arm,phase=phase,layer_max_abs_grad=grads,finite=True))
    assert all(trees_equal(initial[0],v) for v in initial[1:])
    # Loss may use GT; prediction/decode API does not, and fixed inference is invariant.
    return dict(status='PASS_ACTUAL_GPU8_ALL3_ARMS_PHASED_PAIRED_M_EXACT',results=results,
        hidden_zero_at_zero_last_is_expected=True,model_buffers=list(m.named_buffers()),GT_not_forward=True)

def contract(d,arm,stage):
    return dict(stage=stage,arm=arm,loss_version='setutility-v1-coeff1-1',source_code=c.code_sha(),
        protocol=c.sha(c.ROOT/'protocol.json'),**c.binding(d))

def resume_tests(d,val,out,g):
    rows=[]
    for arm in ('M','R','RS'):
        con=contract(d,arm,'preflight_hash513_twoepoch');full=Engine(d,con,arm,epochs=2)
        while not full.complete:g.check();full.step()
        expected=full.snapshot();v1=infer(full.model,val['z'],'cuda:0',g.check)
        del full
        first=Engine(d,con,arm,epochs=2)
        for _ in range(19):first.step()
        path=out/f'{arm}_interrupt19.pt';checkpoint_new(path,first.snapshot());digest=c.sha(path);del first
        state=checkpoint_load_trusted(path);resumed=Engine(d,con,arm,epochs=2);resumed.restore(state)
        while not resumed.complete:g.check();resumed.step()
        assert trees_equal(expected,resumed.snapshot()) and np.array_equal(v1,infer(resumed.model,val['z'],'cuda:0',g.check))
        bad=[]
        for key in ('source_code','statistics','IDs','original_support','best','loss_version'):
            corrupted=copy.deepcopy(state);corrupted['contract'][key]='deliberately_bad'
            try:resumed.restore(corrupted)
            except ValueError:bad.append(key)
            else:raise AssertionError('bad resume accepted '+key)
        assert c.sha(path)==digest
        rows.append(dict(arm=arm,updates=34,interrupted=19,all_state_exact=True,validation12_exact=True,bad_contract_rejections=bad,
            real_train_IDs=d['x']['sample_id'].tolist(),real_val_IDs=val['x']['sample_id'].tolist(),stat_buffers='original model has0 buffers; external stats/inputbytes bound by SHA'))
        del resumed,expected,state
    return dict(status='PASS_ALL3_REAL_HASH513_PLUS12_TWOEPOCH_EXACT_RESUME',arms=rows,not_full3103_twoepochs=True)

def saved_pipeline(d,out,g):
    seed_everything(42,True);m=ParameterRegressor().cuda()
    with torch.no_grad():m.layers[-1].weight.copy_(torch.linspace(-.01,.01,64,device='cuda').reshape(2,32))
    u=infer(m,d['z'],'cuda:0',g.check);pred=predict_paths(d['x']['past'],d['x']['native'],u,'END',g.check)
    path=out/'actual128_saved.npz';c.npz_new(path,sample_id=d['x']['sample_id'],**pred);saved=c.load_npz(path)
    from audit_saved import check_saved
    check_saved(d['x'],saved,'END',g.check)
    # Independent complete-link via condensed pdist, all128x3ports.
    from scipy.cluster.hierarchy import linkage,fcluster
    from scipy.spatial.distance import pdist
    ports=[d['x']['native']+.4*(np.arange(1,13)/12)[None,None,:,None]*u.astype(float)[:,:,None,:],pred['raw_paths'],pred['guarded_paths']]
    started=time.perf_counter();groups=[]
    for name,p in zip(('C','Q','S'),ports):
        sm,rr=evaluate_population(d['x'],d['gt'],p,g.check);groups+=rr
        for i in range(128):
            de=np.linalg.norm(p[i]-d['gt'][i],axis=-1);good=(de.mean(1)<=.2)&(de[:,-1]<=.4)
            ds=np.stack([pdist(p[i,:,t]) for t in range(12)],1);labs=fcluster(linkage(np.maximum(ds.mean(1)/.1,ds[:,-1]/.2),method='complete'),1,criterion='distance')
            assert np.array_equal(good,sm['good'][i]) and len(np.unique(labs[good]))==sm['effective_good'][i]
        c.npz_new(out/f'{name}_metrics.npz',**sm)
    # Reader rejects mismatched file/source/IDs rather than trusting a success name.
    expected=c.sha(path)
    def reader(p,source_sha,expected_ids):
        if source_sha!=c.code_sha() or c.sha(p)!=expected:raise ValueError('bad source/output SHA')
        z=c.load_npz(p)
        if not np.array_equal(z['sample_id'],expected_ids):raise ValueError('bad IDs')
        return z
    reader(path,c.code_sha(),d['x']['sample_id'])
    for key,source,ids in [('source','bad',d['x']['sample_id']),('IDs',c.code_sha(),d['x']['sample_id'][::-1])]:
        try:reader(path,source,ids)
        except ValueError:pass
        else:raise AssertionError('bad reader accepted '+key)
    badpath=out/'bad_output_fixture.npz';c.npz_new(badpath,sample_id=d['x']['sample_id'],deliberate_bad=np.array([1]))
    try:reader(badpath,c.code_sha(),d['x']['sample_id'])
    except ValueError:pass
    else:raise AssertionError('bad output SHA accepted')
    return dict(status='PASS_ACTUAL128_SAVE_READBACK_CQS_QUALITY_COMPLETE_LINK_MOTION',seconds=time.perf_counter()-started,groups=groups,
        readers_bad_source_ID_output_rejected=True)

def main():
    started=time.perf_counter();c.input_lock()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('actual CUDA required')
    torch.zeros(1,device='cuda:0');torch.cuda.synchronize()
    d=c.data();ids=c.load_npz(c.OLD/'tiny_v2/indices.npz')['indices'];tiny=c.subset(d,ids)
    g0=c.resource();snap=g0.sampler();prior=3797450752
    payload=sum(v.nbytes for v in (d['z'],d['target'],d['nj'],d['best'],d['gt'],d['x']['native']))
    estimate=max(prior,snap['private'])+payload+128*1024**2+32*1024**2+256*1024**2
    g=c.resource(estimate);g.check(force=True)
    c.write(c.ROOT/'MEMORY_ESTIMATE.json',dict(estimate_bytes=estimate,previous_actual_GPU_private_peak=prior,current_measured_context_private=snap['private'],
        fulltrain_payload_bytes=payload,double_graph_reserve=128*1024**2,checkpoint_reserve=32*1024**2,allocator_reserve=256*1024**2,
        no_estimate_from_available=True,policy='warning_first_remaining_v3'))
    out=c.ROOT/'preflight';out.mkdir(exist_ok=False)
    real=c.subset(d,np.arange(8));c.write(out/'GPU8.json',gpu_bridge(real))
    c.write(out/'GRADIENT.json',mathematical_tests(real,out))
    pick=lambda sample,split,count:np.array(sorted(range(len(sample)),key=lambda i:(hashlib.sha256(f'endpoint-set-utility-resume-v1|{split}|{int(sample[i])}'.encode()).hexdigest(),int(sample[i])))[:count])
    ti=pick(d['x']['sample_id'],'train',513)
    vx=c.load_npz(c.P0/'prepared/X_val.npz',('sample_id',))['sample_id'];vi=pick(vx,'val',12);val=c.data('val',vi)
    c.write(out/'RESUME.json',resume_tests(c.subset(d,ti),val,out,g))
    pt=time.perf_counter();pipeline=saved_pipeline(tiny,out,g);pipeline['full_save_and_review_seconds']=time.perf_counter()-pt;c.write(out/'PIPELINE.json',pipeline)
    steps={}
    # Independent throwaway models, never reused for scientific model or selected by outcome.
    for arm in ('M','R','RS'):
        e=Engine(tiny,contract(tiny,arm,'throwaway_benchmark'),arm)
        for _ in range(4):e.step()
        torch.cuda.synchronize();t=time.perf_counter()
        for _ in range(32):e.step()
        torch.cuda.synchronize();steps[arm]=(time.perf_counter()-t)/32
        del e
    train_T=3*256*sum(steps.values());population_T=9*pipeline['full_save_and_review_seconds']
    sha_seconds=c.json_read(c.ROOT/'INPUT_SOURCE_LOCK.json')['seconds']
    planning_T=train_T+2*population_T+3*sha_seconds+90
    # F estimate is conservative population throughput scaled, and missing held labels explicit.
    planning_F=3*776*sum(steps.values())+2*9*(3103+993+3825+1856)/128*pipeline['full_save_and_review_seconds']+4*sha_seconds+180
    c.write(c.ROOT/'BENCHMARK.json',dict(per_step_seconds=steps,warm=4,measured=32,models_discarded_not_scientific=True,
        tiny_planning_seconds=planning_T,tiny_ETA=[.7*planning_T,1.5*planning_T],formal_planning_seconds=planning_F,
        formal_unmeasured_components='hold/inner END49 labels if branch triggers; separate actual preflight and workload estimate required before F',
        F16_end_to_end='NOT_REMEASURED'))
    c.verify(c.json_read(c.ROOT/'INPUT_SOURCE_LOCK.json')['files'],g.check);g.check(force=True)
    c.write(out/'RESOURCE.json',dict(policy='warning_first_remaining_v3',estimate_bytes=estimate,observations=g.observations))
    c.write(c.ROOT/'PREFLIGHT.json',dict(status='PASS_ACTUAL_THREE_OBJECTIVE_PREFLIGHT',elapsed_seconds=time.perf_counter()-started,
        actualGPU8_all3=True,independent_double_all4770=True,real_hash513_12_all3_exact_resume=True,actual128_saved_pipeline=True,
        statistical_buffer_scope='external original fixed mean/std SHA and normalized input bytes, model has0buffers',
        tiny_planning_seconds=planning_T,formal_not_admitted_by_this_receipt=True,no_held_effects=True))
    c.write(c.ROOT/'RESEARCH_REVIEW_DA1.json',dict(status='PASS_WITH_LIMITS',same_assistant_not_external=True,
        issue='New auxiliary objectives can win by doing nothing; continuous utility may not survive END49/guard.',
        resolution='Frozen three arms/coefficient1, require additional rescue, zero lost RS and all protected error/motion gates; no fourth variant or prior90% rewrite.'))
    print('PREFLIGHT PASS, tiny planning seconds',planning_T,'formal lower planning',planning_F,flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        c.write(c.ROOT/('FAILURE_PREFLIGHT_'+time.strftime('%H%M%S')+'.json'),dict(traceback=traceback.format_exc(),code_sha=c.code_sha()))
        raise
