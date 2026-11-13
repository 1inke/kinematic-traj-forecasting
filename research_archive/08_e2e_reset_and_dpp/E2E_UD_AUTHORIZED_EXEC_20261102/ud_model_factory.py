from ud_common import *
import baseline_factory as bf
from train_core import cpu_tree,exact
from decoder_adapter import UDDecoder
import torch


def make_ud_model(arm,device='cuda',seed=42):
    model,cfg=bf.make_model(device,seed)
    original=cpu_tree(model.state_dict())
    devices=[torch.cuda.current_device()] if str(device).startswith('cuda') else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(20261102+seed)
        model.decoder=UDDecoder(model.decoder,arm).to(device)
    for name,value in original.items():exact(value,model.state_dict()[name],name)
    return model,cfg


def new_branch(model):
    return model.decoder.update if hasattr(model.decoder,'update') else model.decoder.interaction
