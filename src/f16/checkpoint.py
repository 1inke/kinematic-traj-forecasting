"""Opt-in complete new-run checkpoints. No old checkpoint is upgraded in place."""
import random
import numpy as np
from .io import sha256

def capture_rng(generator):
    import torch
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],loader=generator.get_state())

def restore_rng(state,generator):
    import torch
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch_cpu'].cpu());generator.set_state(state['loader'].cpu())
    if state['torch_cuda']:
        if not torch.cuda.is_available():raise ValueError('CUDA RNG checkpoint cannot be silently restored on CPU')
        torch.cuda.set_rng_state_all([x.cpu() for x in state['torch_cuda']])

def save_checkpoint(path,model,optimizer,generator,history,contract):
    import torch
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=capture_rng(generator),history=history,contract=contract),path)
    return sha256(path)

def load_checkpoint(path,model,optimizer,generator,expected_contract):
    import torch
    x=torch.load(path,map_location='cpu',weights_only=False)
    if x['contract']!=expected_contract:raise ValueError('Resume source/ID/protocol/stats contract mismatch')
    model.load_state_dict(x['model']);optimizer.load_state_dict(x['optimizer']);restore_rng(x['rng'],generator)
    return x['history']
