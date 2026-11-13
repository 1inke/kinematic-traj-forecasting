"""Deterministic training kernel with exact mid-epoch state. No auto-training."""
import copy
import hashlib
import os
import random
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from model import ParameterRegressor, window_mse


def array_digest(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def seed_everything(seed, use_cuda=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


class Engine:
    """Production defaults frozen; small epochs/batch override only for unit fixtures."""
    def __init__(self, x, y, contract, seed=42, device='cpu', epochs=8, batch=32):
        x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        if x.shape[1:] != (20, 40) or y.shape != x.shape[:2]+(2,) or not len(x):
            raise ValueError('training tensor contract')
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('nonfinite training tensors')
        if seed not in (42, 43, 44) or epochs <= 0 or batch <= 0:
            raise ValueError('invalid training configuration')
        self.device = torch.device(device)
        self.contract = copy.deepcopy(contract)
        self.binding = dict(x=array_digest(x), y=array_digest(y), seed=seed,
                            epochs=epochs, batch=batch, device=str(self.device))
        seed_everything(seed, use_cuda=self.device.type == 'cuda')
        self.model = ParameterRegressor().to(self.device)
        self.initial_state = copy.deepcopy(self.model.state_dict())
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001, weight_decay=.0001)
        self.sampler = torch.Generator(device='cpu').manual_seed(seed+901)
        self.x, self.y = torch.from_numpy(x), torch.from_numpy(y)
        self.epochs, self.batch = epochs, batch
        self.epoch = self.cursor = self.updates = 0
        self.order = None
        self.loss_sum = 0.
        self.history, self.batch_order = [], []

    @property
    def complete(self):
        return self.epoch >= self.epochs

    def step(self):
        if self.complete:
            raise StopIteration
        if self.order is None:
            self.order = torch.randperm(len(self.x), generator=self.sampler)
        ids = self.order[self.cursor:self.cursor+self.batch]
        self.model.train()
        prediction = self.model(self.x[ids].to(self.device))
        loss = window_mse(prediction, self.y[ids].to(self.device))
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite loss')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5., error_if_nonfinite=True)
        self.optimizer.step()
        self.batch_order.append(ids.tolist())
        self.cursor += len(ids)
        self.updates += 1
        value = float(loss.detach().cpu())
        self.loss_sum += value*len(ids)
        if self.cursor == len(self.x):
            self.history.append(dict(epoch=self.epoch+1, loss=self.loss_sum/len(self.x),
                                     updates=self.updates))
            self.epoch += 1
            self.cursor = 0
            self.order = None
            self.loss_sum = 0.
        return dict(loss=value, unclipped_grad_norm=float(norm.detach().cpu()))

    def snapshot(self):
        return copy.deepcopy(dict(contract=self.contract, binding=self.binding,
            model=self.model.state_dict(), initial_state=self.initial_state,
            optimizer=self.optimizer.state_dict(), sampler=self.sampler.get_state(),
            python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else [],
            epoch=self.epoch, cursor=self.cursor, updates=self.updates, order=self.order,
            loss_sum=self.loss_sum, history=self.history, batch_order=self.batch_order))

    def restore(self, state):
        if state['contract'] != self.contract or state['binding'] != self.binding:
            raise ValueError('resume source/data/statistics/protocol/seed/device mismatch')
        self.model.load_state_dict(state['model'], strict=True)
        self.initial_state = copy.deepcopy(state['initial_state'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.sampler.set_state(state['sampler'].cpu())
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if state['cuda_rng']:
            if not torch.cuda.is_available():
                raise ValueError('CUDA RNG state cannot resume on CPU-only runtime')
            torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']])
        for name in ('epoch', 'cursor', 'updates', 'loss_sum', 'history', 'batch_order'):
            setattr(self, name, copy.deepcopy(state[name]))
        self.order = None if state['order'] is None else state['order'].cpu().clone()


def checkpoint_new(path, state):
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f:
        torch.save(state, f)


def checkpoint_load_trusted(path):
    # Full RNG/optimizer checkpoint uses pickle. Only load local hash-verified artifacts.
    return torch.load(path, map_location='cpu', weights_only=False)


def infer(model, x, device='cpu', check=lambda: None):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(x), 32):
            check()
            values.append(model(torch.from_numpy(x[start:start+32]).to(device)).cpu().numpy())
    return np.concatenate(values)
