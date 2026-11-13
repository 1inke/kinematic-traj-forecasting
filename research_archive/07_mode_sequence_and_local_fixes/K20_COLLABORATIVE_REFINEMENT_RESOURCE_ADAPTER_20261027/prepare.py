from core import *
from scipy.interpolate import CubicSpline
def motion_features(p):
    v=np.diff(p,axis=2,prepend=np.zeros_like(p[:,:,:1]))/.4;a=np.diff(v,axis=2)/.4;j=np.diff(a,axis=2)/.4;s,aa,jj=[np.linalg.norm(x,axis=-1) for x in (v,a,j)]
    return np.stack((np.quantile(s,.95,axis=2),aa.mean(2),np.quantile(aa,.95,axis=2),jj.mean(2),np.quantile(jj,.95,axis=2)),axis=-1)
def raw_features(past,tokens,native):
    p=past.astype(np.float64);v=np.diff(p,axis=1)/.4;a=np.diff(v,axis=1)/.4;t=tokens.astype(np.float32)
    h=np.concatenate((p.reshape(len(p),16),v.reshape(len(p),14),a.reshape(len(p),12),t[:,-1].astype(float),t.mean(1).astype(float)),axis=1)
    return np.concatenate((native.reshape(-1,20,24),motion_features(native),np.broadcast_to(h[:,None],(len(p),20,234))),axis=-1).astype(np.float32)
def main():
    g=guard(True);out=HERE/'prepared';out.mkdir(exist_ok=True)
    if (out/'ready.json').exists():
        for n,h in read_json(out/'ready.json')['outputs'].items():assert sha256(out/n)==h
        return
    pops={};audit={}
    for split in SPLITS:
        z=raw_population(split,split=='opt');obs=load(OBS/f'prepared/observations_{split}.npz');assert np.array_equal(z['sample_id'],obs['sample_id']) and np.array_equal(z['track_key'],obs['track_key'])
        x=raw_features(obs['past'],obs['tokens'],z['native']);assert x.shape==(len(z['native']),20,263)
        z.update(raw=x,past=obs['past'],tokens=obs['tokens']);pops[split]=z;audit[split]=dict(n=len(x),IDs=arrsha(z['sample_id']),tracks=len(np.unique(z['track_key'])),held_GT_read=split=='opt');save(out/(split+'.npz'),**z);g.check()
    for a in SPLITS:
        for b in SPLITS:
            if a!=b:assert not np.intersect1d(pops[a]['sample_id'],pops[b]['sample_id']).size and not set(pops[a]['track_key'])&set(pops[b]['track_key'])
    z=pops['opt'];tracks=np.unique(z['track_key']);order=sorted(tracks,key=lambda s:hashlib.sha256(('collaborative-val-v1|'+s).encode()).hexdigest());vt=np.array(order[:int(np.ceil(.2*len(tracks)))],dtype=tracks.dtype);val=np.flatnonzero(np.isin(z['track_key'],vt));train=np.flatnonzero(~np.isin(z['track_key'],vt));save(out/'split.npz',train=train,val=val,sample_id=z['sample_id'],track_key=z['track_key'])
    # Equal windows and equal candidates; no per-window K normalization.
    x=z['raw'][train].astype(float);mu=x.mean((0,1));var=((x-mu)**2).mean((0,1));sd=np.sqrt(var);sd[sd<1e-6]=1
    mu2=np.mean(np.mean(x,axis=1),axis=0);var2=np.mean(np.mean(x*x,axis=1),axis=0)-mu2*mu2;assert np.max(abs(mu-mu2))<=1e-11 and np.max(abs(var-var2))<=1e-9
    p=z['native'][train];v=np.diff(p,axis=2,prepend=np.zeros_like(p[:,:,:1]))/.4;a=np.diff(v,axis=2)/.4;j=np.diff(a,axis=2)/.4;scales=np.array([max(1e-6,np.sqrt((w*w).sum(-1).mean())) for w in (v,a,j)])
    basis=CubicSpline([0,4,8,12],np.vstack((np.zeros((1,3)),np.eye(3))),bc_type=((1,np.zeros(3)),(2,np.zeros(3))))(np.arange(1,13));bound=np.abs(basis).sum(1).max();save(out/'statistics.npz',mean=mu,std=sd,motion_scales=scales,basis=basis,basis_l1_bound=np.array(bound))
    audit.update(train_windows=len(train),validation_windows=len(val),train_tracks=len(tracks)-len(vt),validation_tracks=len(vt),opt_total=len(x)+len(val),train_change=int((z['group'][train]==2).sum()),statistics_only_train=True,no_GT_in_input=True,all_original_IDs_retained=True,raw_float32=True,statistics_mean_crosscheck=float(abs(mu-mu2).max()),statistics_variance_crosscheck=float(abs(var-var2).max()))
    write_json(HERE/'INPUT_AUDIT.json',audit);write_json(out/'ready.json',dict(outputs={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='ready.json'},counts=audit));resource(HERE/'PREPARE_RESOURCE.json',g);print(audit,flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as ex:failure('prepare',ex);raise
