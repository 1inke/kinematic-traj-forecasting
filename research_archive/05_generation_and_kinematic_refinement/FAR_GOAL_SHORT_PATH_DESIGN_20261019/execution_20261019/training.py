import copy,random
import common as c
import model as m
import torch
import numpy as np

def tensors(d,ix,device='cuda:0'):
    return (torch.as_tensor(d['p2'][ix],device=device),torch.as_tensor(d['v0'][ix],dtype=torch.float32,device=device),torch.as_tensor(d['gt'][ix],dtype=torch.float64,device=device),torch.as_tensor(d['y20'][ix],dtype=torch.float64,device=device),torch.as_tensor(d['m20'][ix],dtype=torch.bool,device=device))
def rng(generator):return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),loader=generator.get_state())
def restore(state,generator):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state_all(state['cuda']);generator.set_state(state['loader'])
def snapshot(model,opt,gen,history,epoch,step,contract):return dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=rng(gen),history=copy.deepcopy(history),epoch=epoch,step=step,contract=contract)
def exact(a,b):
    if isinstance(a,torch.Tensor):assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b,equal_nan=True)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b,(a,b)
def epoch(model,opt,generator,d,ix,horizon):
    model.train();order=torch.randperm(len(ix),generator=generator).numpy();total=np.zeros(4);counts=np.zeros(20,np.int64);ties=0;maskn=0;gradient={};batches=0
    for offset in range(0,len(ix),256):
        c.guard();take=ix[order[offset:offset+256]];p2,v,y,y20,mask=tensors(d,take);opt.zero_grad(set_to_none=True);out=model(p2,v);loss,parts=m.objective(out,y,y20,mask,horizon);assert torch.isfinite(loss);loss.backward()
        norms={}
        for name,p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(),name
            norms[name]=float(p.grad.double().square().sum().sqrt())
        opt.step();n=len(take);total+=n*np.array([float(loss.detach()),float(parts['path'].detach()),float(parts['goal'].detach()),float(parts['diversity'].detach())]);counts+=np.bincount(parts['winner'].detach().cpu().numpy(),minlength=20);ties+=int((parts['ties']>1).sum());maskn+=int(mask.sum());batches+=1
        for k,vv in norms.items():gradient[k]=gradient.get(k,0.)+vv
    prob=counts/counts.sum();entropy=float(-np.sum(prob[prob>0]*np.log(prob[prob>0])))
    return dict(n=len(ix),batches=batches,loss=float(total[0]/len(ix)),L_path=float(total[1]/len(ix)),L_goal=float(total[2]/len(ix)),L_div=float(total[3]/len(ix)),m20_count=maskn,winner_counts=counts.tolist(),winner_entropy=entropy,dead_slots=np.flatnonzero(counts==0).tolist(),tie_windows=ties,mean_parameter_gradnorm={k:v/batches for k,v in gradient.items()})

@torch.no_grad()
def subset_eval(model,d,ix,horizon):
    model.eval();p,v,y,y20,mask=tensors(d,ix);loss,parts=m.objective(model(p,v),y,y20,mask,horizon)
    return dict(n=len(ix),loss=float(loss),components={k:float(parts[k]) for k in ('path','goal','diversity')},winner=parts['winner'].cpu().tolist())

def resume_test(stats,d,horizon):
    # Both arms: REAL 513 FIT +12 inner windows, not full FIT twice.
    fit=d['split']['FIT'][:513];inner=d['split']['inner'][:12];contract={'test':True,'horizon':horizon,'FIT_ids':c.array_sha(d['ids'][fit])}
    m.setup(800+horizon);a=m.GoalConditionedShortPath(stats).cuda();o=m.optimizer(a);g=torch.Generator().manual_seed(800+horizon);hist=[]
    hist.append(dict(train=epoch(a,o,g,d,fit,horizon),inner=subset_eval(a,d,inner,horizon)))
    cp=snapshot(a,o,g,hist,1,3,contract)
    hist.append(dict(train=epoch(a,o,g,d,fit,horizon),inner=subset_eval(a,d,inner,horizon)))
    full=snapshot(a,o,g,hist,2,6,contract)
    # Actual serialization/deserialization including optimizer and all RNG.
    path=c.H/f'preflight_resume_H{horizon}.pt';torch.save(cp,path);saved=torch.load(path,map_location='cpu',weights_only=False)
    b=m.GoalConditionedShortPath(stats).cuda();bo=m.optimizer(b);b.load_state_dict(saved['model']);bo.load_state_dict(saved['optimizer']);gg=torch.Generator();rr=saved['rng'];rr['cpu']=rr['cpu'].cpu();rr['loader']=rr['loader'].cpu();rr['cuda']=[x.cpu() for x in rr['cuda']];restore(rr,gg);bh=saved['history']
    bh.append(dict(train=epoch(b,bo,gg,d,fit,horizon),inner=subset_eval(b,d,inner,horizon)))
    exact(full,snapshot(b,bo,gg,bh,2,6,contract))
    return dict(status='PASS_EXACT',horizon=horizon,FIT=513,inner=12,epochs=2,steps=6,all_params_buffers_optimizer_all_RNG_history_exact=True,serialized_checkpoint_sha=c.sha(path))
