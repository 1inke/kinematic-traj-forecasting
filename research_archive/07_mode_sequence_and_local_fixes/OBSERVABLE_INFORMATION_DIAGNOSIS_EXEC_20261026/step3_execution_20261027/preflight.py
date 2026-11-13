"""Current-source actual GPU/FD/ALL-state resume and saved128 construction.
No registered scientific effects are selected from these throwaway fixtures.
"""
from core import *
from feature_blocks import assemble,read_stats
from train import Engine,Data,exact,seed_all,G,torch,DiagnosticEditor,weighted_loss
import copy,time
from scipy.special import erf,expit

def independent_chain(m,x,w,y):
    # Double independent NumPy scalar, score derivative and complete parameter chain.
    p={k:v.detach().cpu().numpy().copy() for k,v in m.state_dict().items()}
    n=(x-p['feature_mean'])/p['feature_std']*p['active_mask']
    a=n@p['net.0.weight'].T+p['net.0.bias']
    ga=.5*a*(1+erf(a/np.sqrt(2)))
    b=ga@p['net.2.weight'].T+p['net.2.bias'];gb=.5*b*(1+erf(b/np.sqrt(2)))
    s=(gb@p['net.4.weight'].T)[...,0]
    value=float((w*np.where(y,np.logaddexp(0,-s),np.logaddexp(0,s))).sum(-1).mean())
    ds=w*(expit(s)-y)/len(x)
    grad4=(gb*ds[...,None]).sum((0,1))[None]
    dbg=ds[...,None]*p['net.4.weight'][0]
    db=dbg*(.5*(1+erf(b/np.sqrt(2)))+b*np.exp(-b*b/2)/np.sqrt(2*np.pi))
    grad2=np.einsum('bai,baj->ij',db,ga);bias2=db.sum((0,1))
    da=(db@p['net.2.weight'])*(.5*(1+erf(a/np.sqrt(2)))+a*np.exp(-a*a/2)/np.sqrt(2*np.pi))
    grad0=np.einsum('bai,baj->ij',da,n);bias0=da.sum((0,1))
    return value,ds,dict(zip(('net.0.weight','net.0.bias','net.2.weight','net.2.bias','net.4.weight'),(grad0,bias0,grad2,bias2,grad4)))

def main():
    require12();start=time.perf_counter();stamp=time.time_ns();out=HERE/f'preflight_{stamp}';out.mkdir()
    admission=read_json(HERE/'FEATURE_PROVENANCE.json');arms=admission['admissible_arms'];source0=verify();records=[];bench=[]
    data=Data(42);shared={};snapshot_contract=dict(source=source0,fixture='513opt2epoch',stat=sha256(HERE/'prepared/stats_seed42.npz'))
    for arm in arms:
        seed_all(42);e=Engine(data,arm,42);shared[arm]={k:v.clone() for k,v in e.model.state_dict().items() if k!='active_mask'}
        for old in shared.values():exact(old,shared[arm])
        x,w,y=data.batch(np.arange(32));initial=e.model(x)
        assert torch.equal(initial,torch.zeros_like(initial))
        assert torch.equal(e.model.normalized(x)[...,~e.model.active_mask],torch.zeros_like(e.model.normalized(x)[...,~e.model.active_mask]))
        # Eight actual32-window batches, phase0 first-layer zero is expected.
        phased=[]
        for step in range(8):
            x,w,y=data.batch(np.arange(step*32,(step+1)*32));e.opt.zero_grad(set_to_none=True);loss=weighted_loss(e.model(x),w,y);loss.backward()
            grads={k:float(p.grad.norm()) for k,p in e.model.named_parameters()}
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in e.model.parameters())
            inactive=~e.model.active_mask;assert torch.count_nonzero(e.model.net[0].weight.grad[:,inactive])==0
            if step>=2:assert all(v>0 for v in grads.values())
            phased.append(grads);torch.nn.utils.clip_grad_norm_(e.model.parameters(),5);e.opt.step();G.check()
        # Fixed nonzero double model. Parameter FD two registered eps, no search.
        m=copy.deepcopy(e.model).cpu().double();xx=x[:2,:8].cpu().double().detach().requires_grad_(True)
        ww=w[:2,:8].cpu().double();yy=y[:2,:8].cpu();m.zero_grad();scores=m(xx);scores.retain_grad()
        value=weighted_loss(scores,ww,yy);value.backward();nv,nds,ng=independent_chain(m,xx.detach().numpy(),ww.numpy(),yy.numpy())
        assert abs(float(value)-nv)<=1e-10 and np.max(abs(scores.grad.numpy()-nds))<=1e-10
        pg={k:p.grad.clone() for k,p in m.named_parameters()};chain_error=max(float(np.max(abs(pg[k].numpy()-ng[k]))) for k in pg);assert chain_error<=1e-10
        gen=torch.Generator().manual_seed(20261027);direction={k:torch.randn(p.shape,generator=gen,dtype=p.dtype) for k,p in m.named_parameters()}
        norm=torch.sqrt(sum((v*v).sum() for v in direction.values()));direction={k:v/norm for k,v in direction.items()}
        ref=sum(float((pg[k]*direction[k]).sum()) for k in pg);base={k:p.detach().clone() for k,p in m.named_parameters()};fd=[]
        for eps in (1e-4,1e-5):
            vals=[]
            for sign in (1,-1):
                with torch.no_grad():
                    for k,p in m.named_parameters():p.copy_(base[k]+sign*eps*direction[k])
                vals.append(float(weighted_loss(m(xx),ww,yy)))
            numerical=(vals[0]-vals[1])/(2*eps);assert abs(numerical-ref)<=1e-6+1e-3*abs(ref);fd.append(dict(eps=eps,reference=ref,numerical=numerical))
        del m,xx,ww,yy;G.check()
        # Real first513, 2 epochs=34 updates. Interrupt after19 crosses epoch.
        subset=Data(42,513);continuous=Engine(subset,arm,42);continuous.run(34);expected=continuous.snapshot(snapshot_contract)
        del continuous
        interrupted=Engine(subset,arm,42);interrupted.run(19);saved=interrupted.snapshot(snapshot_contract);checkpoint=out/f'{arm}_resume.pt';torch.save(saved,checkpoint);del interrupted,saved
        resumed=Engine(subset,arm,999);restored=torch.load(checkpoint,map_location='cpu',weights_only=False);resumed.restore(restored,snapshot_contract);resumed.run(34);exact(expected,resumed.snapshot(snapshot_contract))
        bad=False
        try:resumed.restore(restored,dict(snapshot_contract,bad_source=True))
        except ValueError:bad=True
        assert bad;del restored,expected,resumed;G.check()
        # Actual nonzero trained throwaway e: 128 full401 scores/paths saved.
        z=next(aux_chunks(42,'opt'));assert len(z['sample_id'])==128
        basefeatures=old_features(42,np.arange(128))
        with np.load(HERE/'prepared/observed_blocks_opt.npz') as zobs:token=zobs['token'][:128];neighbor=zobs['neighbor'][:128]
        e.model.eval();before=copy.deepcopy(e.model.state_dict());scores=[]
        with torch.inference_mode():
            for lo in range(0,128,32):scores.append(e.model(torch.as_tensor(assemble(basefeatures[lo:lo+32],token[lo:lo+32],neighbor[lo:lo+32]),device='cuda')).cpu().numpy())
        exact(before,e.model.state_dict());score=np.concatenate(scores);choice=np.where(z['eligible'],score,-np.inf).argmax(1);paths=z['native'].copy()
        for i,a in enumerate(choice):
            if a:r,l=divmod(int(a)-1,20);paths[i,r]=z['kin'][i,l]
        save_npz(out/f'{arm}_saved128.npz',scores=score,chosen=choice,paths=paths,**{k:z[k] for k in ('sample_id','track_key','group','native','kin','gt','eligible','B','target')})
        # Actual fixed4warm32measured production-sized32-window steps.
        e.model.train();times=[]
        for i in range(36):
            torch.cuda.synchronize();t=time.perf_counter();bx,bw,by=data.batch(np.arange(32));e.opt.zero_grad(set_to_none=True);ll=weighted_loss(e.model(bx),bw,by);ll.backward();torch.nn.utils.clip_grad_norm_(e.model.parameters(),5);e.opt.step();torch.cuda.synchronize()
            if i>=4:times.append(time.perf_counter()-t)
        bench.append(dict(arm=arm,actual32window_sampled_step_seconds=float(np.mean(times)),warm=4,measure=32))
        records.append(dict(arm=arm,actual8batch32=True,phased_gradient=phased,FD=fd,independent_parameter_chain_max=chain_error,
            real513_twoepoch34update_ALLstate_exact=True,bad_resume_rejected=True,saved128=str((out/f'{arm}_saved128.npz').relative_to(HERE)),saved_SHA=sha256(out/f'{arm}_saved128.npz')))
        del e;G.check()
    assert verify()==source0
    result=dict(status='PASS_GPU_AND_RESUME_PENDING_CPU_SAVED_REVIEW',at=now(),source=source0,seconds=time.perf_counter()-start,
        records=records,benchmark=bench,fixture_not_formal_training=True,not_full4096_twoepochs=True,
        no_saved_benchmark_model=True,no_effect_selection=True)
    write_json(HERE/'PREFLIGHT_GPU.json',result);resource_report(out/'RESOURCE.json',G)
    print(json.dumps(dict(status=result['status'],seconds=result['seconds'],benchmark=bench),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
