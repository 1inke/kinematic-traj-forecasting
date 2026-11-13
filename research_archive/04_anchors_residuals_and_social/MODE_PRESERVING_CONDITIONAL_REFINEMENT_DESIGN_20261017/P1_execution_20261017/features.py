import numpy as np
import torch

def numpy_features(native,representation):
    p=native.astype(np.float32)
    if representation=='position':return p
    assert representation=='velocity'
    x=p.astype(np.float64);previous=np.concatenate((np.zeros_like(x[...,:1,:]),x[...,:-1,:]),axis=-2)
    return ((x-previous)/.4).astype(np.float32)

def candidate_features(native,representation):
    p=native.float()
    if representation=='position':return p.flatten(-2)
    assert representation=='velocity'
    x=p.double();previous=torch.cat((torch.zeros_like(x[...,:1,:]),x[...,:-1,:]),dim=-2)
    return ((x-previous)/.4).float().flatten(-2)

