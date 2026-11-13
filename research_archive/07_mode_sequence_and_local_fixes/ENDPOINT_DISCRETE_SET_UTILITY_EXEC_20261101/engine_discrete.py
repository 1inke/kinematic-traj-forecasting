"""Deterministic fresh utility regression with full committed optimizer/RNG state."""
import copy
import random
import numpy as np
import contracts_discrete as c
import torch
from train import seed_everything
from model_discrete import ActionUtilityHead,utility_loss

def infer(model,z,check=lambda:None):
    model.eval();out=[]
    with torch.no_grad():
        for s in range(0,len(z),32):
            check();out.append(model(torch.from_numpy(z[s:s+32]).cuda()).cpu().numpy())
    return np.concatenate(out)

class Engine:
    def __init__(self,d,arm,seed=42,epochs=64,stage='tiny'):
        if arm not in ('G','H') or seed not in (42,43,44):raise ValueError('arm/seed')
        seed_everything(seed,True);self.model=ActionUtilityHead().cuda();self.initial=copy.deepcopy(self.model.state_dict())
        assert sum(p.numel() for p in self.model.parameters())==6800 and not list(self.model.buffers())
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=.001,weight_decay=.0001)
        self.d=d;self.arm=arm;self.seed=seed;self.epochs=epochs;self.stage=stage;self.contract=c.contract(d,arm,stage)
        self.x=torch.from_numpy(np.array(d['z'+arm],copy=True));self.y=torch.from_numpy(np.array(d['value'],copy=True));self.valid=torch.from_numpy(np.array(d['valid'],copy=True))
        self.sampler=torch.Generator(device='cpu').manual_seed(seed+901)
        self.epoch=self.cursor=self.updates=0;self.order=None;self.loss_sum=0.;self.history=[];self.batch_order=[];self.trace=[]
    @property
    def complete(self):return self.epoch>=self.epochs
    def step(self):
        if self.complete:raise StopIteration
        if self.order is None:self.order=torch.randperm(len(self.x),generator=self.sampler)
        ids=self.order[self.cursor:self.cursor+32];self.model.train();score=self.model(self.x[ids].cuda())
        y,v=self.y[ids].cuda(),self.valid[ids].cuda();loss=utility_loss(score,y,v)
        if not torch.isfinite(loss) or not torch.isfinite(score).all():raise FloatingPointError('nonfinite loss/score')
        self.optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),5.,error_if_nonfinite=True);self.optimizer.step()
        self.trace.append(dict(update=self.updates+1,batch_indices=ids.tolist(),score=score.detach().cpu().numpy().copy(),loss=float(loss.detach().cpu()),unclipped_grad_norm=float(norm.cpu())))
        value=float(loss.detach().cpu());self.batch_order.append(ids.tolist());self.cursor+=len(ids);self.updates+=1;self.loss_sum+=value*len(ids)
        if self.cursor==len(self.x):
            self.epoch+=1;self.history.append(dict(epoch=self.epoch,loss=self.loss_sum/len(self.x),updates=self.updates));self.order=None;self.cursor=0;self.loss_sum=0.
        return value
    def snapshot(self):
        return copy.deepcopy(dict(contract=self.contract,arm=self.arm,seed=self.seed,epochs=self.epochs,model=self.model.state_dict(),initial=self.initial,
            optimizer=self.optimizer.state_dict(),sampler=self.sampler.get_state(),epoch=self.epoch,cursor=self.cursor,updates=self.updates,order=self.order,
            history=self.history,batch_order=self.batch_order,trace=self.trace,loss_sum=self.loss_sum,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),transaction='COMMITTED'))
    def restore(self,state):
        if state['contract']!=self.contract or state['seed']!=self.seed or state['epochs']!=self.epochs or state.get('transaction')!='COMMITTED':raise ValueError('source/split/stat/actionorder/protocol/arm/seed mismatch')
        self.model.load_state_dict(state['model'],strict=True);self.initial=copy.deepcopy(state['initial']);self.optimizer.load_state_dict(state['optimizer']);self.sampler.set_state(state['sampler'].cpu())
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
        for k in ('epoch','cursor','updates','history','batch_order','trace','loss_sum'):setattr(self,k,copy.deepcopy(state[k]))
        self.order=None if state['order'] is None else state['order'].cpu().clone()
