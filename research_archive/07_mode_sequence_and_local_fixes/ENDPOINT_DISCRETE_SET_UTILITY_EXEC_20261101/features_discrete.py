"""Observation-only 40 frozen dimensions plus train3103-only geometry8."""
import numpy as np
import contracts_discrete as c

def geometry8(native):
    p=np.asarray(native,float);delta=np.linalg.norm(p[:,:,None]-p[:,None,:],axis=-1)
    D=np.maximum(delta.mean(-1)/.10,delta[...,-1]/.20)
    D[:,np.arange(20),np.arange(20)]=np.inf
    # Stable original physical-slot order breaks exact distance ties.
    order=np.argsort(D,axis=-1,kind='stable')[...,:8]
    return np.take_along_axis(D,order,-1)

def fit8(native,chunk=31):
    count=0;mean=np.zeros(8);M2=np.zeros(8)
    for start in range(0,len(native),chunk):
        x=geometry8(native[start:start+chunk]).reshape(-1,8);n=len(x);bm=x.mean(0);bs=((x-bm)**2).sum(0)
        if count:
            diff=bm-mean;M2+=bs+diff*diff*count*n/(count+n);mean+=diff*n/(count+n)
        else:mean=bm;M2=bs
        count+=n
    std=np.sqrt(M2/count);std[std<1e-6]=1.
    return dict(mean=mean,std=std,count=np.array(count))

def transform(d,stats8,arm):
    stats40=c.load_npz(c.P0/'prepared/statistics.npz')
    raw=c.raw_features(d['x']['past'],d['x']['native'])
    z40=(raw-stats40['mean'])/stats40['std']
    geo=geometry8(d['x']['native']);z8=(geo-stats8['mean'])/stats8['std']
    z=np.concatenate((z40,z8),-1)
    if arm=='G':z[...,:16]=0.
    elif arm!='H':raise ValueError('arm')
    return z.astype(np.float32)

def independent_geometry8(native):
    out=[]
    for window in native:
        rows=[]
        for k in range(20):
            values=[]
            for j in range(20):
                if j==k:continue
                e=np.sqrt(((window[k]-window[j])**2).sum(-1))
                values.append((max(e.mean()/.1,e[-1]/.2),j))
            rows.append([v for v,j in sorted(values)[:8]])
        out.append(rows)
    return np.array(out)
