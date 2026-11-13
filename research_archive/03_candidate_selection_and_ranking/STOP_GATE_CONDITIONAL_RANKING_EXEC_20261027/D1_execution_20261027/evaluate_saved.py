"""New-editor GPU scoring; old candidate/label geometry reused by explicit ID/SHA."""
from core import *
from objective import *
fb={'np':np};pure(OLD/'feature_builder.py',('build',),fb)
def unlocked():
    lock=read_json(HERE/'ALL9_FINALS_LOCK.json');assert lock['source']==source0() and len(lock['locks'])==9
    for name,files in lock['locks'].items():
        for fn,h in files.items():assert sha256(HERE/'runs'/name/fn)==h
    return lock
def save_models(seed,split,limit=None,fixture_models=None,fixture_folder=None):
    g=guard();data=OptData(seed)
    if fixture_models is None:
        unlocked();models={}
        for arm in ARMS:
            m=ActionEditor(data.mean,data.std).cuda();ck=torch.load(HERE/f'runs/{arm}_seed{seed}/checkpoint.pt',map_location='cpu',weights_only=False);assert ck['contract']==contract(seed,arm,source0());m.load_state_dict(ck['model']);models[arm]=m
    else:assert split=='opt' and limit==128;models=fixture_models
    out=Path(fixture_folder) if fixture_folder else HERE/f'populations/seed{seed}/{split}';out.mkdir(parents=True,exist_ok=True)
    expected=dict(source=source0(),seed=seed,split=split,fixture=fixture_models is not None)
    if (out/'done.json').exists():completed(out,expected);return out
    if any(out.iterdir()):raise FileExistsError('Preserve prior partial population')
    obs=metadata(split);before={a:{k:v.detach().cpu().clone() for k,v in m.state_dict().items()} for a,m in models.items()};lo=0;chunks=[]
    for j,(oldname,z) in enumerate(aux(seed,split)):
        if limit is not None and lo>=limit:break
        assert np.array_equal(z['sample_id'],obs['sample_id'][lo:lo+len(z['sample_id'])])
        if split=='opt':x=np.array(data.features[lo:lo+len(z['sample_id'])],copy=True)
        else:x=np.array([fb['build'](p,t,n,k,mm) for p,t,n,k,mm in zip(obs['past'][lo:],obs['tokens'][lo:],z['native'],z['kin'],z['motion'])],np.float32)
        for arm,m in models.items():
            m.eval()
            with torch.no_grad():s=m(torch.from_numpy(x).cuda()).cpu().numpy()
            assert np.isfinite(s).all();a=np.where(z['eligible'],s,-np.inf).argmax(1);paths=z['native'].copy()
            for i,ac in enumerate(a):
                if ac:r,l=divmod(int(ac)-1,20);paths[i,r]=z['kin'][i,l]
            fn=f'{arm}_chunk{j:04d}.npz';save_npz(out/fn,sample_id=z['sample_id'],scores=s,chosen=a,paths=paths,eligible=z['eligible'],B=z['B'],target=z['target'])
            if fixture_models is None and arm in ('I','S'):
                with np.load(OLD/f'populations/seed{seed}/{split}'/(arm+'_'+oldname[len('aux_'):])) as legacy:
                    assert np.array_equal(s,legacy['scores']) and np.array_equal(a,legacy['action']) and np.array_equal(paths,legacy['paths']),'CONTROL_REPRODUCTION_UNRESOLVED final saved scores/path'
        chunks.append(j);lo+=len(z['sample_id']);mark('D1_SAVE_ALL401',seed=seed,split=split,windows=lo);g.check()
    assert lo==(limit if limit else len(obs['sample_id']))
    for arm,m in models.items():assert all(torch.equal(v.cpu(),before[arm][k]) for k,v in m.state_dict().items())
    resource(out/'RESOURCE_GPU.json',g);data.close();finish(out,expected,n=lo,chunks=chunks,arms=list(models),eval_all_weights_buffers_EXACT=True,old_IS_final_scores_paths_EXACT=fixture_models is None);return out
if __name__=='__main__':save_models(int(sys.argv[1]),sys.argv[2])
