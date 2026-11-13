"""NumPy64 saved-control reconstruction, not another model/GPU forward."""
import numpy as np
from joint_common import module,OLD

def reconstructor(model,family):
    if family=='existing':
        weights={k:v.detach().cpu().numpy().astype(np.float64) for k,v in model.adapter.model.decoder.state_dict().items()}
        independent=module('joint_old_decoder_numpy',OLD/'independent.py')
        return lambda z,data,sl:independent.decoder_np(z,data,sl,weights)
    if family!='cvae':raise ValueError('RECONSTRUCTION_FAMILY')
    weights={k:v.detach().cpu().numpy().astype(np.float64) for k,v in model.decoder.state_dict().items()}
    def decode(z,data,sl):
        context=data['context'][sl,0].astype(np.float64);z=np.asarray(z,np.float64)
        x=np.concatenate([np.broadcast_to(context[:,None],(len(z),20,64)),z],-1)
        for layer in ('0','2','4'):
            x=x@weights[layer+'.weight'].T+weights[layer+'.bias']
            if layer!='4':x=np.maximum(x,0)
        local=x.reshape(len(z),20,12,2)
        return np.einsum('bktd,bcd->bktc',local,data['matrix'][sl].astype(np.float64))+data['origin'][sl,None,None].astype(np.float64)
    return decode
