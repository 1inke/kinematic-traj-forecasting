from geo_common import *
import baseline_factory as bf
from train_core import cpu_tree,exact
from geo_modules import DGeoDecoder
import torch


def make_geo_model(arm,device='cuda',seed=42):
    m,cfg=bf.make_model(device,seed);before=cpu_tree(m.state_dict())
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if str(device).startswith('cuda') else []):
        torch.manual_seed(20261103+seed);m.decoder=DGeoDecoder(m.decoder,arm).to(device)
    for name,value in before.items():exact(value,m.state_dict()[name],name)
    return m,cfg

