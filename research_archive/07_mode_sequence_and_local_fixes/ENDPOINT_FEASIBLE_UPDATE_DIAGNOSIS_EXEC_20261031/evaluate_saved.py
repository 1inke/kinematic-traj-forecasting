"""Exclusive result bundles and explicit source-compatible SHA readers."""
import numpy as np
import contracts as c
from checks import full_metrics,verdict

def save_port(folder,out,d,bank,extra=None):
    folder.mkdir(parents=True,exist_ok=False)
    arrays={k:v for k,v in out.items() if isinstance(v,np.ndarray)};rows=[]
    for name in ('C','Q','S'):
        z,r=full_metrics(out[name],d)
        arrays.update({name+'_'+k:v for k,v in z.items()})
        rows.extend([dict(port=name,**x) for x in r])
    arrays['sample_id']=d['x']['sample_id']
    c.npz_new(folder/'paths.npz',**arrays)
    v=verdict(out['C'],out['S'],d,bank.ref)
    c.write(folder/'METRICS.json',dict(groups=rows,verdict=v,extra=extra or {}))
    c.write(folder/'done.json',dict(status='SAVED_FULL_QUALITY_AND_INDEPENDENT_POOLED',contract=c.contract(d),
        files={p.name:c.sha(p) for p in folder.iterdir() if p.is_file()}))
    return rows

def read_port(folder,d):
    done=c.json_read(folder/'done.json');c.verify_contract(done['contract'],d)
    for rel,s in done['files'].items():
        if c.sha(folder/rel)!=s:raise ValueError('output SHA mismatch')
    z=c.load_npz(folder/'paths.npz')
    if not np.array_equal(z['sample_id'],d['x']['sample_id']):raise ValueError('output ID mismatch')
    for k in ('C','Q','S'):
        if z[k].shape!=(128,20,12,2) or not np.isfinite(z[k]).all():raise ValueError('output shape/finite mismatch')
    return z
