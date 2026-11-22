import numpy as np
B='C:/Users/linke/Documents/New project/qoder_work/EXP_DECEL_DIAG_EXEC_20261117/populations/C0_dev/B0_s{}/decel_window_metrics.npz'
rng=np.random.default_rng(7)
betas=np.round(np.arange(-3,3.01,0.25),2)
def es(P,L,w):
    a=(w*np.sqrt(P**2+L**2)).sum(1)
    dp=P[:,:,None]-P[:,None];dl=L[:,:,None]-L[:,None]
    s=0.5*(w[:,:,None]*w[:,None]*np.sqrt(dp**2+dl**2)).sum((1,2));return a-s
def top1(P,L,w):
    i=w.argmax(1);n=np.arange(len(i));return np.sqrt(P[n,i]**2+L[n,i]**2)
for s in (42,43,44,45,46):
    d=np.load(B.format(s),allow_pickle=True);v=d['direction_valid']
    P=d['candidate_prog_error_t12'][v];L=d['candidate_lat_error_t12'][v];w=d['candidate_probability'][v];r=d['candidate_r'][v]
    ids=d['ids'][v];tr=np.array(['|'.join(i.split('|')[:2]) for i in ids]);spd=d['observed_speed'][v]
    sb=np.digitize(spd,np.quantile(spd,[.2,.4,.6,.8]))
    ut=np.unique(tr);fold=dict(zip(ut,rng.permutation(len(ut))%5));f=np.array([fold[t] for t in tr])
    rc=r-np.median(r,1,keepdims=True)
    def tilt(b,m): 
        z=np.log(w[m]+1e-12)+b*rc[m];z-=z.max(1,keepdims=True);e=np.exp(z);return e/e.sum(1,keepdims=True)
    es0=es(P,L,w);t0=top1(P,L,w);es1=np.empty(len(w));t1=np.empty(len(w));chosen=[]
    for k in range(5):
        for c in range(5):
            trm=(f!=k)&(sb==c);te=(f==k)&(sb==c)
            vals=[es(P[trm],L[trm],tilt(b,trm)).mean() for b in betas];b=betas[int(np.argmin(vals))];chosen.append(b)
            wt=tilt(b,te);es1[te]=es(P[te],L[te],wt);t1[te]=top1(P[te],L[te],wt)
    wm0=(w*P).sum(1)
    print(s,'endpoint wES change %.2f%%'%(100*(es1.sum()-es0.sum())/es0.sum()),' top1 FDE change %.2f%%'%(100*(t1.sum()-t0.sum())/t0.sum()),' betas by bin(fold0..):',np.array(chosen).reshape(5,5)[0])

