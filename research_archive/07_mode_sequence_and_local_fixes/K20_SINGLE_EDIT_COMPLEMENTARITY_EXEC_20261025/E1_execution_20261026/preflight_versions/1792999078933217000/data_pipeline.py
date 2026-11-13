"""Bounded opt-only feature preparation. Hold/inner labels are gated by all six final contracts."""
import shutil,time
import numpy as np
from env import *
from feature_builder import build,WindowMoments
CACHE=HERE/'prepared_v1'
def digest(x):return e0.digest(x)
def _verify_files(folder,manifest):
    for p,ss in manifest['outputs'].items():
        if sha256(folder/p)!=ss:raise ValueError('Prepared cache SHA '+p)
def prepare():
    require_e0();guard=Guard(ESTIMATE);guard.check('startup');source=e0.verify()
    if (CACHE/'ready.json').exists():
        manifest=read_json(CACHE/'ready.json');_verify_files(CACHE,manifest);return manifest
    if CACHE.exists() and any(CACHE.iterdir()):raise FileExistsError('Preserve partial prepared_v1; use new explicit version instead of overwrite')
    CACHE.mkdir(exist_ok=True)
    estimate_disk=12*1024**3
    if shutil.disk_usage(HERE).free<1.3*estimate_disk:raise OSError('Insufficient disk for immutable feature cache + output/temp allowance12GiB*1.3')
    registry=np.load(E0/'SPLIT_REGISTRY.npz');opt=registry['opt_row'];ids=registry['sample_id'][opt]
    from data_contract import ENC,SPLIT
    with np.load(ROOT/SPLIT) as z:raw=z['grouped_train_index'][opt]
    with np.load(ROOT/ENC) as z:past=z['past'][raw];tokens=z['tokens'][raw]
    outputs={};reports=[];start=time.perf_counter()
    for seed in SEEDS:
        pools=np.load(E0/f'seed{seed}/input_pools.npz');n=pools['native'];k=pools['kin'];assert np.array_equal(pools['sample_id'],ids)
        xfile=f'features_seed{seed}.npy';features=np.lib.format.open_memmap(CACHE/xfile,mode='w+',dtype=np.float32,shape=(4096,401,324))
        eligible=np.empty((4096,401),bool);target=np.empty_like(eligible);B=np.empty_like(eligible)
        wm=np.empty((4096,323));wv=np.empty_like(wm);mom=WindowMoments()
        for offset in range(0,4096,128):
            with np.load(E0/f'seed{seed}/chunk_{offset:05d}_{offset+128:05d}.npz') as z:
                assert np.array_equal(z['sample_id'],ids[offset:offset+128])
                motion=z['motion'];valid=z['eligible'];targets=z['target'];bb=z['B']
                for j in range(128):
                    i=offset+j;xx=build(past[i],tokens[i],n[i],k[i],motion[j]);mom.add(xx,valid[j]);v=xx[valid[j],:323]
                    wm[i]=v.mean(0);wv[i]=((v-wm[i])**2).mean(0);features[i]=xx.astype(np.float32)
                eligible[offset:offset+128]=valid;target[offset:offset+128]=targets;B[offset:offset+128]=bb
            features.flush();guard.check()
        mean,std=mom.finish()
        # Independent two-pass law of total variance, not production Welford recurrence.
        im=wm.mean(0);ivar=wv.mean(0)+((wm-im)**2).mean(0)
        assert (ivar>=0).all();isd=np.sqrt(ivar);isd[isd<1e-6]=1
        error_mean=float(abs(mean-im).max());error_std=float(abs(std-isd).max())
        if max(error_mean,error_std)>1e-10:raise ValueError('Independent full4096 window-balanced stats')
        features.flush();features._mmap.close()
        del features
        np.savez_compressed(CACHE/f'stats_seed{seed}.npz',mean64=mean,std64=std,mean32=mean.astype(np.float32),std32=std.astype(np.float32),sample_id=ids,window_mean=wm,window_variance=wv)
        np.savez_compressed(CACHE/f'labels_seed{seed}.npz',eligible=eligible,target=target,B=B,sample_id=ids)
        import torch
        seed_all(seed);from editor import ActionEditor
        model=ActionEditor(mean,std);torch.save(model.state_dict(),CACHE/f'initial_seed{seed}.pt')
        g=torch.Generator().manual_seed(seed);orders=torch.stack([torch.randperm(4096,generator=g) for _ in range(8)]).numpy();np.save(CACHE/f'orders_seed{seed}.npy',orders)
        for fn in (xfile,f'stats_seed{seed}.npz',f'labels_seed{seed}.npz',f'initial_seed{seed}.pt',f'orders_seed{seed}.npy'):outputs[fn]=sha256(CACHE/fn)
        reports.append(dict(seed=seed,n=4096,eligible_actions=int(eligible.sum()),mean_error=error_mean,std_error=error_std,stats_population='FIT_opt4096 only',feature_bytes=4096*401*324*4));guard.check()
    np.savez_compressed(CACHE/'identity.npz',sample_id=ids,track_key=registry['track_key'][opt],groups=registry['group'][opt],raw_index=raw)
    outputs['identity.npz']=sha256(CACHE/'identity.npz');assert e0.verify()==source
    out=dict(status='PASS_PREPARED_OPT_ONLY',source_e0_digest=source,protocol_SHA=sha256(HERE/'protocol.json'),outputs=outputs,reports=reports,elapsed_seconds=time.perf_counter()-start,resource=guard.observations,no_hold_inner_GT=True)
    write_json(CACHE/'ready.json',out);return out
class OptData:
    def __init__(self,seed,count=4096):
        self.seed=seed;self.count=count;self.features=np.load(CACHE/f'features_seed{seed}.npy',mmap_mode='r')
        with np.load(CACHE/f'labels_seed{seed}.npz') as z:self.eligible=z['eligible'][:count];self.target=z['target'][:count];self.B=z['B'][:count];self.ids=z['sample_id'][:count]
        with np.load(CACHE/f'stats_seed{seed}.npz') as z:self.mean=z['mean64'];self.std=z['std64']
        self.orders=np.load(CACHE/f'orders_seed{seed}.npy')
    def batch(self,indices,device):
        import torch
        ix=np.asarray(indices)
        return tuple(torch.from_numpy(x).to(device) for x in (np.array(self.features[ix],copy=True),self.eligible[ix].copy(),self.target[ix].copy()))
def contract(seed,arm,source):
    m=read_json(CACHE/'ready.json')
    if m['protocol_SHA']!=sha256(HERE/'protocol.json'):raise ValueError('Prepared protocol changed')
    return dict(source=source,seed=seed,arm=arm,protocol=sha256(HERE/'protocol.json'),split=m['outputs']['identity.npz'],statistics=m['outputs'][f'stats_seed{seed}.npz'],initial=m['outputs'][f'initial_seed{seed}.pt'],orders=m['outputs'][f'orders_seed{seed}.npy'],features=m['outputs'][f'features_seed{seed}.npy'],labels=m['outputs'][f'labels_seed{seed}.npz'],population_windows=4096,epochs=8,steps=1024,parameters=11488)
def verify_prepared():
    m=read_json(CACHE/'ready.json');_verify_files(CACHE,m);return m
if __name__=='__main__':
    try:print(json.dumps(prepare()))
    except BaseException as exc:
        import traceback
        write_json(HERE/('PREPARE_FAILURE_'+str(time.time_ns())+'.json'),dict(type=type(exc).__name__,message=str(exc),at=e0.now(),traceback=traceback.format_exc()));raise
