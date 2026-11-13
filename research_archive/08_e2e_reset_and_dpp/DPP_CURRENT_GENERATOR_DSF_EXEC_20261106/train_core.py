import math
from run_common import *

class Trainer:
    def __init__(self,adapter,cache,indices,arm,seed,epochs=20,batch=32):
        self.adapter=adapter;self.cache=cache;self.indices=np.asarray(indices);self.arm=arm;self.seed=seed;self.epochs=epochs;self.batch=batch
        self.device=next(adapter.model.parameters()).device;self.sampler=new_sampler(seed,self.device)
        self.optimizer=torch.optim.Adam(self.sampler.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
        self.generator=torch.Generator().manual_seed(seed);self.epoch=0;self.cursor=0;self.updates=0;self.order=None;self.history=[];self.order_history=[];self.connected=set()
        self.cal=read(STAGE_A/'CALIBRATION.json');self.contract=contract(arm,seed,self.indices,cache)
    def step(self):
        if self.order is None or self.cursor==len(self.order):
            if self.epoch>=self.epochs:raise StopIteration
            self.order=self.indices[torch.randperm(len(self.indices),generator=self.generator).numpy()];self.cursor=0;self.epoch+=1
            self.order_history.append(self.cache['ids'][self.order].tolist())
        ix=self.order[self.cursor:self.cursor+self.batch];bundle=batch_bundle(self.cache,ix,self.device)
        self.sampler.train();self.optimizer.zero_grad(set_to_none=True)
        loss,diagnostics=objective(self.adapter,self.sampler,bundle,self.arm,self.cal)
        if not torch.isfinite(loss):raise ValueError('NONFINITE_LOSS')
        loss.backward()
        for name,p in self.sampler.named_parameters():
            if p.grad is None or not torch.isfinite(p.grad).all():raise ValueError('MISSING_OR_NONFINITE_DSF_GRAD '+name)
            if p.grad.abs().any():self.connected.add(name)
        if any(p.grad is not None for p in self.adapter.model.parameters()):raise ValueError('OLD_MODEL_GRADIENT')
        norm=torch.nn.utils.clip_grad_norm_(self.sampler.parameters(),5.)
        if not torch.isfinite(norm):raise ValueError('NONFINITE_NORM')
        self.optimizer.step();self.updates+=1;self.cursor+=len(ix)
        row=dict(epoch=self.epoch,update=self.updates,cursor=self.cursor,loss=float(loss.detach()),qkin=float(diagnostics['qkin'].mean().detach()),EC=float(diagnostics['EC'].mean().detach()),control_abs_mean=float(diagnostics['control'].abs().mean().detach()),grad_norm=float(norm))
        self.history.append(row);return row
    def snapshot(self):
        return cpu(dict(contract=self.contract,sampler=self.sampler.state_dict(),optimizer=self.optimizer.state_dict(),epoch=self.epoch,cursor=self.cursor,updates=self.updates,order=self.order,history=self.history,order_history=self.order_history,connected=sorted(self.connected),rng=dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),loader=self.generator.get_state())))
    def load(self,state):
        verify_contract(state['contract'],self.contract);self.sampler.load_state_dict(state['sampler'],strict=True);self.optimizer.load_state_dict(state['optimizer'])
        for k in ('epoch','cursor','updates','order','history','order_history'):setattr(self,k,copy.deepcopy(state[k]))
        self.connected=set(state['connected']);r=state['rng'];random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in r['cuda']]);self.generator.set_state(r['loader'].cpu())

@torch.no_grad()
def infer(adapter,sampler,cache,guard=None):
    sampler.eval();before=cpu(sampler.state_dict());ys=[];ls=[];zs=[]
    for start in range(0,len(cache['ids']),32):
        b=batch_bundle(cache,slice(start,start+32),next(sampler.parameters()).device);z=sampler(b['sampler_context']);y,l=decode_logits(adapter,z,b)
        ys.append(y.double().cpu().numpy());ls.append(l.cpu().numpy());zs.append(z.cpu().numpy())
        if guard:guard.check()
    exact(before,cpu(sampler.state_dict()),'EVAL_SAMPLER_PARAMETERS_BUFFERS');return np.concatenate(ys),np.concatenate(ls),np.concatenate(zs)
