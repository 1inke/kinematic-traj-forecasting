from pathlib import Path
import copy,hashlib,math,random,time
import numpy as np
import torch
from probe_io import ROOT,now,read,write,save_torch,npz,seal,verify,sha
from probe_model import Refiner,predict,baseline_loss,extra_loss,make_basis
from probe_metrics import metrics,aggregate

def cpu_tree(v):
    if torch.is_tensor(v):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu_tree(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu_tree(x) for x in v]
    if isinstance(v,tuple):return tuple(cpu_tree(x) for x in v)
    return copy.deepcopy(v)
def initialize(arm,seed,device):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    return Refiner(arm).to(device)
class Trainer:
    def __init__(self,arm,seed,cache,head,criterion,device,epochs=20,batch=256):
        self.arm,self.seed,self.cache,self.head,self.criterion,self.device=arm,seed,cache,head,criterion,device
        self.epochs,self.batch=epochs,batch;self.model=initialize(arm,seed,device)
        self.opt=torch.optim.AdamW(self.model.parameters(),lr=.001,weight_decay=.0001,foreach=False)
        self.steps=math.ceil(len(cache['y'])/batch);self.total=epochs*self.steps
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.opt,lambda i:.5*(1+math.cos(math.pi*min(i/self.total,1))))
        self.generator=torch.Generator().manual_seed(20261117+seed)
        self.epoch=0;self.updates=0;self.history=[];self.connected=set()
    def epoch_step(self):
        start=time.monotonic();order=torch.randperm(len(self.cache['y']),generator=self.generator);order_hash=hashlib.sha256(order.numpy().astype('<i8').tobytes()).hexdigest()
        self.model.train();total_loss=0.;extra_value=0.;active=0;gate=0;count=0
        for first in range(0,len(order),self.batch):
            index=order[first:first+self.batch].to(self.device);paths,z,shift=predict(self.model,self.head,self.cache,index)
            y=self.cache['y'][index];loss,detail=baseline_loss(paths,y,z,self.criterion)
            extra=loss.new_zeros(())
            if self.model.reallocate:
                extra,allocation=extra_loss(paths,y,z);loss=loss+extra;active+=int(allocation['active'].sum());gate+=int(allocation['gate'].sum())
            if not torch.isfinite(loss):raise ValueError('NONFINITE_LOSS')
            self.opt.zero_grad(set_to_none=True);loss.backward()
            grads=[p.grad.flatten() for p in self.model.parameters() if p.grad is not None]
            if not torch.isfinite(torch.cat(grads)).all():raise ValueError('NONFINITE_GRADIENT')
            self.opt.step();self.scheduler.step();self.updates+=1;n=len(index);count+=n
            total_loss+=float(loss.detach())*n;extra_value+=float(extra.detach())*n
        self.epoch+=1
        for name,p in self.model.named_parameters():
            if p.grad is not None and bool(p.grad.abs().any()):self.connected.add(name)
        row=dict(epoch=self.epoch,updates=self.updates,loss=total_loss/count,extra_loss=extra_value/count,allocation_active_rate=active/count,allocation_gate_rate=gate/count,order_hash=order_hash,seconds=time.monotonic()-start,lr=self.opt.param_groups[0]['lr'])
        self.history.append(row);return row
    def snapshot(self,contract):
        return cpu_tree(dict(contract=contract,model=self.model.state_dict(),optimizer=self.opt.state_dict(),scheduler=self.scheduler.state_dict(),generator=self.generator.get_state(),epoch=self.epoch,updates=self.updates,history=self.history,connected=sorted(self.connected),python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
    def restore(self,state,contract):
        assert state['contract']==contract
        self.model.load_state_dict(state['model']);self.opt.load_state_dict(state['optimizer']);self.scheduler.load_state_dict(state['scheduler']);self.generator.set_state(state['generator'].cpu())
        self.epoch,self.updates=state['epoch'],state['updates'];self.history=copy.deepcopy(state['history']);self.connected=set(state['connected'])
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'].cpu())
        if state['cuda_rng'] and torch.cuda.is_available():torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])

@torch.no_grad()
def evaluate(model,head,data):
    cache,meta=data['tensors'],data['meta'];paths=[];logits=[];shifts=[]
    if model is not None:model.eval()
    for first in range(0,len(cache['y']),256):
        index=slice(first,first+256)
        if model is None:
            shift=torch.zeros_like(cache['relative'][index]);p=torch.einsum('tp,nkpc->nktc',make_basis().to(cache['relative'].device),cache['relative'][index]);z=cache['logits'][index]
        else:p,z,shift=predict(model,head,cache,index)
        paths.append(p.cpu().numpy());logits.append(z.cpu().numpy());shifts.append(shift.cpu().numpy())
    paths=np.concatenate(paths);logits=np.concatenate(logits);shift=np.concatenate(shifts)
    w=metrics(paths,cache['y'].cpu().numpy(),logits,meta['heading_valid'],shift)
    return dict(groups=aggregate(w,meta['group'],meta['labels'])),w,dict(paths=paths,logits=logits,shift=shift)

def latency(model,head,data):
    import statistics
    nthreads=torch.get_num_threads();torch.set_num_threads(1)
    m=copy.deepcopy(model).cpu().eval();h=copy.deepcopy(head).cpu().eval();cache={k:v[:1].detach().cpu() for k,v in data['tensors'].items()}
    result={}
    with torch.inference_mode():
        for name,fn in [('refiner_only',lambda:m(cache['mode'],cache['relative'],cache['scale'],cache['logits'])),('with_frozen_score_and_interpolation',lambda:predict(m,h,cache))]:
            for _ in range(50):fn()
            times=[]
            for _ in range(200):
                started=time.perf_counter();fn();times.append((time.perf_counter()-started)*1000)
            result[name]=dict(median_ms=float(np.median(times)),p90_ms=float(np.quantile(times,.9)),repeats=200,warmup=50,cpu_threads=1,batch=1)
    torch.set_num_threads(nthreads);return result

def save_evaluation(folder,result,w,pred,data):
    meta=data['meta'];folder.mkdir(parents=True,exist_ok=True)
    npz(folder/'window_metrics.npz',**w,ids=meta['ids'],group=meta['group'],labels=meta['labels'])
    npz(folder/'predictions_local.npz',**pred,ids=meta['ids'])
    write(folder/'RESULT.json',result)

def run_one(arm,seed,data,head,criterion,device,contract):
    folder=ROOT/'runs'/f'{arm}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'DONE.json').exists():verify(read(folder/'DONE.json')['files']);return read(folder/'RESULT.json')
    trainer=Trainer(arm,seed,data['fit']['tensors'],head,criterion,device)
    initial=cpu_tree(trainer.model.state_dict());ref=ROOT/'runs'/f'A1_s{seed}'/'initial.pt'
    if ref.exists():
        expected=torch.load(ref,map_location='cpu',weights_only=False)
        for name,value in initial.items():assert torch.equal(value,expected[name]),name
    if not (folder/'initial.pt').exists():save_torch(folder/'initial.pt',initial)
    receipts=sorted((folder/'checkpoints').glob('epoch*.json')) if (folder/'checkpoints').exists() else []
    if receipts:
        receipt=read(receipts[-1]);verify(receipt['files']);state=torch.load(next(iter(receipt['files'])),map_location='cpu',weights_only=False);trainer.restore(state,contract)
    while trainer.epoch<20:
        write(ROOT/'CURRENT.json',dict(at=now(),phase='TRAIN',arm=arm,seed=seed,epoch=trainer.epoch+1,epochs=20))
        row=trainer.epoch_step();print(f'{arm} seed={seed} epoch={row["epoch"]}/20 loss={row["loss"]:.6f} seconds={row["seconds"]:.3f}',flush=True)
        if arm!='A1':
            base=read(ROOT/'runs'/f'A1_s{seed}'/'TRAINING.json')['history'][trainer.epoch-1];assert base['order_hash']==row['order_hash']
        path=folder/'checkpoints'/f'epoch{trainer.epoch:02d}.pt';save_torch(path,trainer.snapshot(contract));write(path.with_suffix('.json'),dict(epoch=trainer.epoch,files=seal([path])))
        write(folder/'TRAINING.json',dict(arm=arm,seed=seed,history=trainer.history,epochs=trainer.epoch,updates=trainer.updates,connected=sorted(trainer.connected)))
        if (ROOT/'PAUSE.request').exists():write(ROOT/'CURRENT.json',dict(at=now(),phase='PAUSED',arm=arm,seed=seed,epoch=trainer.epoch));return None
    save_torch(folder/'final.pt',trainer.snapshot(contract))
    write(ROOT/'CURRENT.json',dict(at=now(),phase='EVALUATE',arm=arm,seed=seed))
    result,w,pred=evaluate(trainer.model,head,data['dev']);result.update(arm=arm,seed=seed,epochs=20,updates=trainer.updates,parameter_count=sum(p.numel() for p in trainer.model.parameters()),parameters_with_gradient=sum(p.numel() for name,p in trainer.model.named_parameters() if name in trainer.connected),latency=latency(trainer.model,head,data['dev']),training_seconds=sum(v['seconds'] for v in trainer.history))
    save_evaluation(folder,result,w,pred,data['dev'])
    write(folder/'DONE.json',dict(at=now(),files=seal([folder/'final.pt',folder/'RESULT.json',folder/'window_metrics.npz',folder/'predictions_local.npz',folder/'TRAINING.json',folder/'initial.pt'])))
    return result
