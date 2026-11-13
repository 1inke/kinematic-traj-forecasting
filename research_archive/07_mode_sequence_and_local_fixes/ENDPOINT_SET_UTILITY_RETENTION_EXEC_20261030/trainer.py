"""New source-bound Engine; M calls original kernel verbatim, no warmstart."""
import copy
import numpy as np
import torch
from train import Engine as OriginalEngine,array_digest
from objectives import objective

class Engine(OriginalEngine):
    def __init__(self,d,contract,arm,seed=42,device='cuda:0',epochs=64):
        super().__init__(d['z'],d['target'],contract,seed,device,epochs,batch=32)
        if arm not in ('M','R','RS'):raise ValueError('arm')
        self.arm=arm;self.aux={key:torch.from_numpy(np.array(d[key],copy=True)) for key in ('gt','nj','best')}
        self.aux['native']=torch.from_numpy(np.array(d['x']['native'],copy=True))
        self.aux_binding={k:array_digest(v.numpy()) for k,v in self.aux.items()}
        self.diagnostics=[]
        self.last_prediction=None
        if arm=='M':self.model.register_forward_hook(self.capture)

    def capture(self,model,args,out):self.last_prediction=out.detach().clone()

    def arguments(self,ids):return [self.aux[k][ids].to(self.device) for k in ('native','gt','nj','best')]

    def step(self):
        if self.arm=='M':
            order=self.order if self.order is not None else None
            cursor=self.cursor
            result=super().step()
            ids=self.batch_order[-1]
            with torch.no_grad():_,parts,activity=objective(self.last_prediction,self.y[ids].to(self.device),*self.arguments(ids),'M')
            component_norms=dict(mse=result['unclipped_grad_norm'],retention=0.,rescue=0.)
        else:
            if self.complete:raise StopIteration
            if self.order is None:self.order=torch.randperm(len(self.x),generator=self.sampler)
            ids=self.order[self.cursor:self.cursor+self.batch]
            self.model.train();u=self.model(self.x[ids].to(self.device))
            self.last_prediction=u.detach().clone()
            loss,parts,activity=objective(u,self.y[ids].to(self.device),*self.arguments(ids),self.arm)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite objective')
            self.optimizer.zero_grad(set_to_none=True)
            component_norms={}
            for key,value in parts.items():
                if (key=='rescue' and self.arm=='R') or not value.requires_grad:component_norms[key]=0.;continue
                grad=torch.autograd.grad(value,tuple(self.model.parameters()),retain_graph=True,allow_unused=True)
                component_norms[key]=float(torch.sqrt(sum((v.detach().double().square().sum() for v in grad if v is not None),torch.zeros((),device=self.device,dtype=torch.float64))).cpu())
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),5.,error_if_nonfinite=True)
            self.optimizer.step();self.batch_order.append(ids.tolist());self.cursor+=len(ids);self.updates+=1
            value=float(loss.detach().cpu());self.loss_sum+=value*len(ids)
            if self.cursor==len(self.x):
                self.history.append(dict(epoch=self.epoch+1,loss=self.loss_sum/len(self.x),updates=self.updates))
                self.epoch+=1;self.cursor=0;self.order=None;self.loss_sum=0.
            result=dict(loss=value,unclipped_grad_norm=float(norm.detach().cpu()))
        self.diagnostics.append(dict(update=self.updates,components={k:float(v.detach().cpu()) for k,v in parts.items()},
            preupdate_prediction=self.last_prediction.cpu().numpy().copy(),
            grad_norms_in_actual_objective=component_norms,unclipped_grad_norm=result['unclipped_grad_norm'],clipped=result['unclipped_grad_norm']>5,
            activity={k:v.cpu().numpy().copy() for k,v in activity.items()}))
        return result

    def snapshot(self):
        state=super().snapshot();state.update(arm=self.arm,aux_binding=self.aux_binding,diagnostics=copy.deepcopy(self.diagnostics));return state

    def restore(self,state):
        if state['arm']!=self.arm or state['aux_binding']!=self.aux_binding:raise ValueError('loss/support/oracle/GT binding mismatch')
        super().restore(state);self.diagnostics=copy.deepcopy(state['diagnostics'])
