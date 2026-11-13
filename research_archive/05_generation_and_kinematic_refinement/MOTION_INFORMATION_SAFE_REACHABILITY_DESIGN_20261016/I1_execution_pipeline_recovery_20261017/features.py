import numpy as np
def history_numpy(past,arm):
    assert arm in ('S','H')
    q=past.astype(np.float32).astype(np.float64);v=np.diff(q,axis=1)/.4;a=np.diff(v,axis=1)/.4
    if arm=='S':v=np.repeat(v[:,-1:],7,1);a=np.zeros_like(a)
    return np.concatenate((v.reshape(-1,14),a.reshape(-1,12)),1).astype(np.float32)
def history_torch(past,arm):
    import torch
    assert arm in ('S','H');q=past.float().double() if past.dtype!=torch.float64 else past
    v=(q[:,1:]-q[:,:-1])/.4;a=(v[:,1:]-v[:,:-1])/.4
    if arm=='S':v=v[:,-1:].expand(-1,7,-1);a=torch.zeros_like(a)
    return torch.cat((v.reshape(-1,14),a.reshape(-1,12)),1).to(past.dtype)
def statistics(data,indices,chunk=256):
    sums=np.zeros(146);sq=np.zeros(146);counts=np.zeros(146)
    for start in range(0,len(indices),chunk):
        ii=indices[start:start+chunk]
        blocks=[data['p2'][ii].astype(np.float32).astype(float),data['native'][ii].astype(np.float32).reshape(-1,24).astype(float),history_numpy(data['past'][ii],'H').astype(float)]
        for sl,x in zip((slice(0,96),slice(96,120),slice(120,146)),blocks):sums[sl]+=x.sum(0);sq[sl]+=(x*x).sum(0);counts[sl]+=len(x)
    mean=sums/counts;sd=np.sqrt(np.maximum(sq/counts-mean*mean,0))
    return dict(mean=mean,std=np.where(sd<1e-6,1.,sd),counts=counts,training_ids=data['ids'][indices])
