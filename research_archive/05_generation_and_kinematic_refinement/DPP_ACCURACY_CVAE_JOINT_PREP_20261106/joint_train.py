"""FIT-only paired trainer. ALL optimizer and sampling RNG state is resumable."""
from pathlib import Path
import math, random
import numpy as np
import torch
from joint_common import (protocol,sha,HERE,STAGE,read,seed_all,batch,cpu,exact,save_checkpoint)
from joint_models import make
from joint_losses import loss

class Trainer:
    def __init__(self,family,arm,seed,data,future,source_sha,device='cuda',epochs=None,indices=None):
        self.spec=protocol();self.family=family;self.arm=arm;self.seed=seed
        self.device=device;self.data=data;self.future=future
        self.indices=np.arange(len(data['ids'])) if indices is None else np.asarray(indices)
        if len(future)!=len(data['ids']) or family not in self.spec['families'] or arm not in self.spec['arms']:raise ValueError('TRAIN_CONTRACT')
        self.epochs=self.spec['epochs'] if epochs is None else epochs;self.batch=self.spec['batch']
        self.model=make(family,seed,device)
        o=self.spec['optimizer'];self.params=[p for p in self.model.parameters() if p.requires_grad]
        self.optimizer=torch.optim.Adam(self.params,lr=o['lr'],betas=tuple(o['betas']),eps=o['eps'],weight_decay=o['weight_decay'],foreach=o['foreach'])
        self.loader=torch.Generator().manual_seed(seed+1000)
        self.cal=read(STAGE/'CALIBRATION.json');self.epoch=0;self.cursor=0;self.updates=0;self.order=None
        self.history=[];self.orders=[];self.connected=set()
        self.initial_frozen=cpu(self.model.frozen_state())
        self.contract=dict(family=family,arm=arm,seed=seed,epochs=self.epochs,batch=self.batch,
            ids=data['ids'][self.indices].tolist(),source_sha=source_sha,protocol_sha=sha(HERE/'protocol.json'))
    def step(self):
        if self.order is None or self.cursor==len(self.order):
            if self.epoch>=self.epochs:raise StopIteration
            self.order=self.indices[torch.randperm(len(self.indices),generator=self.loader).numpy()]
            self.epoch+=1;self.cursor=0;self.orders.append(self.data['ids'][self.order].tolist())
        ix=self.order[self.cursor:self.cursor+self.batch]
        b=batch(self.data,ix,self.device);target=torch.as_tensor(self.future[ix],dtype=torch.float32,device=self.device)
        self.model.train()
        if self.family=='existing':self.model.adapter.eval()
        self.optimizer.zero_grad(set_to_none=True)
        value,d=loss(self.model,self.family,self.arm,b,target,self.cal,self.spec)
        if not torch.isfinite(value):raise ValueError('NONFINITE_LOSS')
        value.backward()
        for name,p in self.model.named_parameters():
            if not p.requires_grad:
                if p.grad is not None:raise ValueError('FROZEN_PARAMETER_GRAD '+name)
            elif p.grad is None or not torch.isfinite(p.grad).all():raise ValueError('BAD_GRAD '+name)
            elif p.grad.abs().any():self.connected.add(name)
        norm=torch.nn.utils.clip_grad_norm_(self.params,self.spec['optimizer']['clip_norm'])
        if not torch.isfinite(norm):raise ValueError('NONFINITE_GRAD_NORM')
        self.optimizer.step();self.cursor+=len(ix);self.updates+=1
        row=dict(epoch=self.epoch,update=self.updates,cursor=self.cursor,loss=float(value.detach()),grad_norm=float(norm))
        row.update({k:float(d[k].detach()) for k in ('accuracy','EC','qkin','reconstruction','KL')})
        self.history.append(row);return row
    def snapshot(self):
        return cpu(dict(contract=self.contract,model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),
            epoch=self.epoch,cursor=self.cursor,updates=self.updates,order=self.order,orders=self.orders,
            history=self.history,connected=sorted(self.connected),rng=dict(python=random.getstate(),numpy=np.random.get_state(),
            cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],loader=self.loader.get_state())))
    def restore(self,s):
        exact(s['contract'],self.contract)
        self.model.load_state_dict(s['model'],strict=True);self.optimizer.load_state_dict(s['optimizer'])
        for name in ('epoch','cursor','updates','order','orders','history'):setattr(self,name,cpu(s[name]))
        self.connected=set(s['connected']);r=s['rng']
        random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu'].cpu())
        if r['cuda']:torch.cuda.set_rng_state_all([v.cpu() for v in r['cuda']])
        self.loader.set_state(r['loader'].cpu())
    def check_frozen(self):exact(self.initial_frozen,cpu(self.model.frozen_state()))

@torch.no_grad()
def predict(model,data,device,guard=None):
    model.eval();state=cpu(model.state_dict());paths=[];controls=[]
    for start in range(0,len(data['ids']),32):
        y,a=model.generate(batch(data,slice(start,start+32),device))
        if not torch.isfinite(y).all():raise ValueError('NONFINITE_INFERENCE')
        paths.append(y.double().cpu().numpy())
        controls.append(a['control' if 'control' in a else 'z'].cpu().numpy())
        if guard:guard.check()
    exact(state,cpu(model.state_dict()))
    return np.concatenate(paths),np.concatenate(controls)
