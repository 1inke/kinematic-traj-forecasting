"""Explicit fixed-final trainer with resumable scheduler, EMA and all RNG."""
import copy
import os
from baseline_factory import *
from common import *


def cpu_tree(x):
    if torch.is_tensor(x): return x.detach().cpu().clone()
    if isinstance(x, dict): return {k: cpu_tree(v) for k,v in x.items()}
    if isinstance(x, list): return [cpu_tree(v) for v in x]
    if isinstance(x, tuple): return tuple(cpu_tree(v) for v in x)
    return copy.deepcopy(x)


def exact(a, b, where='root'):
    if torch.is_tensor(a):
        if not torch.equal(a.cpu(), b.cpu()): raise ValueError('NOT_EXACT ' + where)
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a,b): raise ValueError('NOT_EXACT ' + where)
    elif isinstance(a, dict):
        if a.keys() != b.keys(): raise ValueError('KEY_MISMATCH ' + where)
        for k in a: exact(a[k], b[k], where + '/' + str(k))
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b): raise ValueError('LEN_MISMATCH ' + where)
        for k,(x,y) in enumerate(zip(a,b)): exact(x,y,where+'/'+str(k))
    elif a != b: raise ValueError('VALUE_MISMATCH ' + where)


class Trainer:
    def __init__(self, data, indices, batch=64, epochs=8, device='cuda', seed=42):
        self.data, self.indices = data, np.asarray(indices)
        self.device, self.batch, self.epochs = device, batch, epochs
        self.model, self.cfg = make_model(device, seed)
        self.loss_fn = ModeQueryLoss(self.cfg).to(device)
        self.ema = ModelEMA(self.model, .995)
        self.steps_per_epoch = math.ceil(len(indices) / batch)
        self.optimizer, self.scheduler = optimizer_scheduler(self.model, self.steps_per_epoch, epochs)
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch, self.cursor, self.updates = 0, 0, 0
        self.order = None
        self.history, self.batch_orders = [], []
        self.connected = set()

    def step(self):
        if self.order is None or self.cursor == len(self.order):
            if self.epoch >= self.epochs:
                raise StopIteration
            self.order = self.indices[torch.randperm(len(self.indices), generator=self.generator).numpy()]
            self.cursor = 0; self.epoch += 1
        idx = self.order[self.cursor:self.cursor+self.batch]
        self.model.train()
        p = torch.as_tensor(self.data['past'][idx], device=self.device)
        y = torch.as_tensor(self.data['future'][idx], device=self.device)
        n = torch.as_tensor(self.data['num_valid'][idx], device=self.device)
        out, logits, target, _ = prediction(self.model, p, n, y, augment=True)
        loss, detail = loss_value(out, target, logits, self.loss_fn)
        if not torch.isfinite(loss): raise ValueError('NONFINITE_TRAIN_LOSS')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name,param in self.model.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all(): raise ValueError('NONFINITE_GRAD '+name)
                if param.grad.abs().any(): self.connected.add(name)
        grad = torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 5.)
        if not torch.isfinite(grad): raise ValueError('NONFINITE_GRAD_NORM')
        self.optimizer.step(); self.ema.update(self.model); self.scheduler.step()
        self.updates += 1; self.cursor += len(idx)
        self.batch_orders.append(self.data['ids'][idx].tolist())
        row = dict(epoch=self.epoch, update=self.updates, cursor=self.cursor,
            loss=float(loss.detach()), regression=float(detail['regression']),
            classification=float(detail['classification']), grad_norm=float(grad),
            lr=self.optimizer.param_groups[0]['lr'])
        self.history.append(row)
        return row

    def snapshot(self, contract):
        return cpu_tree(dict(contract=contract, model=self.model.state_dict(), ema=self.ema.module.state_dict(),
            optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(),
            loss_buffers=self.loss_fn.state_dict(), rng=rng_state(self.generator),
            epoch=self.epoch,cursor=self.cursor,updates=self.updates,order=self.order,
            history=self.history,batch_orders=self.batch_orders,connected=sorted(self.connected)))

    def load(self, state, contract):
        exact(state['contract'],contract)
        self.model.load_state_dict(state['model'], strict=True)
        self.ema.module.load_state_dict(state['ema'], strict=True)
        self.loss_fn.load_state_dict(state['loss_buffers'], strict=True)
        self.optimizer.load_state_dict(state['optimizer']); self.scheduler.load_state_dict(state['scheduler'])
        self.epoch,self.cursor,self.updates=state['epoch'],state['cursor'],state['updates']
        self.order=state['order']; self.history=copy.deepcopy(state['history']); self.batch_orders=copy.deepcopy(state['batch_orders'])
        self.connected=set(state['connected'])
        restore_rng(state['rng'],self.generator)


def save_checkpoint(path, state):
    path=Path(path)
    if path.exists(): raise FileExistsError(path)
    temporary=path.with_suffix(path.suffix+'.new')
    with temporary.open('xb') as f:
        torch.save(state,f); f.flush(); os.fsync(f.fileno())
    os.rename(temporary,path)


@torch.no_grad()
def infer(model, data, batch=64, guard=None):
    before=cpu_tree(model.state_dict()); model.eval()
    paths, scores=[],[]
    for start in range(0,len(data['ids']),batch):
        idx=slice(start,start+batch)
        p=torch.as_tensor(data['past'][idx],device=next(model.parameters()).device)
        n=torch.as_tensor(data['num_valid'][idx],device=p.device)
        out,logits,_,transform=prediction(model,p,n)
        paths.append(world_paths(out,transform).double().cpu().numpy()); scores.append(logits[:,0].cpu().numpy())
        if guard: guard.check()
    exact(before,cpu_tree(model.state_dict()),'EVAL_PARAMETERS_BUFFERS')
    return np.concatenate(paths),np.concatenate(scores)
