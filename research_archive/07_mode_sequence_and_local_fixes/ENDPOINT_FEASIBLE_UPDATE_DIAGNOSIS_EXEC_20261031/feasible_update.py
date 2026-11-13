"""Transactional original-RS Adam proposal with eight fixed global steps."""
import copy
import random
import numpy as np
import torch
from contracts import ALPHAS,contract,verify_contract
from train import Engine as OriginalEngine,infer
from objectives import objective
from preflight_checks import trees_equal

def rng():return copy.deepcopy(dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
def set_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in r['cuda']])

def search(parameters,old,delta,evaluate):
    """One immutable-origin affine trial sequence. Synthetic tests use this same function."""
    trials=[]
    for alpha in ALPHAS:
        with torch.no_grad():
            for p,v,dd in zip(parameters,old,delta):p.copy_(v+alpha*dd)
        changed=any(not torch.equal(p,v) for p,v in zip(parameters,old))
        if not changed:
            trials.append(dict(alpha=alpha,changed=False,all_pass=False,reason='ZERO_DISPLACEMENT'));continue
        result=evaluate(alpha);trials.append(dict(alpha=alpha,changed=True,**result))
        if result['all_pass']:return alpha,trials
    with torch.no_grad():
        for p,v in zip(parameters,old):p.copy_(v)
    return 0.,trials

class Engine(OriginalEngine):
    def __init__(self,d,bank,seed=42,epochs=64):
        super().__init__(d['z'],d['target'],contract(d),seed,'cuda:0',epochs,32)
        self.d=d;self.bank=bank;self.accepted=0;self.rejected=0;self.records=[];self.trial_u=[];self.probes=[]
        self.aux={k:torch.from_numpy(np.array(d[k],copy=True)) for k in ('gt','nj','best')}
        self.aux['native']=torch.from_numpy(np.array(d['x']['native'],copy=True))
    def propose(self):
        if self.complete:raise StopIteration
        if self.order is None:self.order=torch.randperm(len(self.x),generator=self.sampler)
        ids=self.order[self.cursor:self.cursor+32]
        self.model.train();u=self.model(self.x[ids].to(self.device))
        loss,parts,activity=objective(u,self.y[ids].to(self.device),*[self.aux[k][ids].to(self.device) for k in ('native','gt','nj','best')],'RS')
        self.optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),5.,error_if_nonfinite=True)
        self.optimizer.step()
        return ids,u.detach().cpu().numpy().copy(),float(loss.detach().cpu()),{k:float(v.detach().cpu()) for k,v in parts.items()},float(norm.cpu())
    def step(self):
        # S0 excludes sampler clock; consuming the next batch is outside rollback.
        old=[p.detach().clone() for p in self.model.parameters()];opt0=copy.deepcopy(self.optimizer.state_dict());S0=rng()
        number=self.updates+1;probe=number in (64,128,192,256)
        current=infer(self.model,self.d['z'],self.device) if probe else None
        assert trees_equal(S0,rng())
        ids,pred,value,parts,norm=self.propose();S1=rng();opt1=copy.deepcopy(self.optimizer.state_dict())
        full=[p.detach().clone() for p in self.model.parameters()];delta=[b-a for a,b in zip(old,full)]
        fullu=infer(self.model,self.d['z'],self.device) if probe else None
        assert trees_equal(S1,rng())
        def evaluate(alpha):
            uu=infer(self.model,self.d['z'],self.device)
            _,result=self.bank.evaluate(uu)
            index=len(self.trial_u);self.trial_u.append(uu.copy())
            assert trees_equal(S1,rng()),'trial consumed RNG'
            return dict(result,u_index=index)
        alpha,trials=search(list(self.model.parameters()),old,delta,evaluate)
        if alpha:
            self.optimizer.load_state_dict(opt1);self.accepted+=1
        else:
            self.optimizer.load_state_dict(opt0);self.rejected+=1
        set_rng(S1)
        # Proposal clock, batch cursor and sampler advance once, even on rejection.
        self.batch_order.append(ids.tolist());self.cursor+=len(ids);self.updates+=1;self.loss_sum+=value*len(ids)
        if self.cursor==len(self.x):
            self.history.append(dict(epoch=self.epoch+1,loss=self.loss_sum/len(self.x),proposals=self.updates,accepted=self.accepted))
            self.epoch+=1;self.cursor=0;self.order=None;self.loss_sum=0.
        for v in self.optimizer.state.values():assert int(v['step'])==self.accepted
        self.records.append(dict(proposal=number,accepted=bool(alpha),alpha=alpha,batch_ids=ids.tolist(),loss=value,components=parts,unclipped_norm=norm,
            delta_l2=float(torch.sqrt(sum((x.double()**2).sum() for x in delta)).cpu()),trials=trials,
            optimizer_steps=self.accepted,preupdate_prediction=pred))
        if probe:self.probes.append(dict(proposal=number,u_current=current,u_full_proposal=fullu))
        return self.records[-1]
    def snapshot(self):
        s=super().snapshot();s.update(accepted=self.accepted,rejected=self.rejected,records=copy.deepcopy(self.records),trial_u=copy.deepcopy(self.trial_u),probes=copy.deepcopy(self.probes),transaction='COMMITTED_ONLY')
        return s
    def restore(self,state):
        verify_contract(state['contract'],self.d)
        if state.get('transaction')!='COMMITTED_ONLY':raise ValueError('uncommitted transaction checkpoint')
        super().restore(state)
        for k in ('accepted','rejected','records','trial_u','probes'):setattr(self,k,copy.deepcopy(state[k]))
        assert self.accepted+self.rejected==self.updates
        for v in self.optimizer.state.values():assert int(v['step'])==self.accepted
