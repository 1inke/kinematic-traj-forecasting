"""Same per-window MLP, constant GEMM shape; no scientific tolerance change.

The original MLP has no batch-dependent normalization. Padding is never saved
or scored. Keeping computational batch256 avoids float32 shape-dependent GEMM
roundoff when checking caller-requested chunk sizes.
"""
import numpy as np

def predict(net,past,device,batch=256):
    import torch
    if not 1<=batch<=256:raise ValueError('Requested chunk must be1..256')
    out=[]
    with torch.inference_mode():
        for i in range(0,len(past),batch):
            rows=np.asarray(past[i:i+batch],dtype=np.float32);padded=np.zeros((256,8,2),np.float32);padded[:len(rows)]=rows
            pred=net(torch.from_numpy(padded).to(device))['paths'][:len(rows)].cpu().numpy();out.append(pred)
    answer=np.concatenate(out)
    if answer.shape!=(len(past),20,12,2) or not np.isfinite(answer).all():raise ValueError('Invalid frozen K20 output')
    return answer
