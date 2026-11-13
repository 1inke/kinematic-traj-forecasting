from train import *
def run(seed,split,models=None,folder=None,limit=None,existing_guard=None):
    fixture=models is not None;assert not fixture or split=='opt';data=dataset(split,held=not fixture);g=existing_guard if existing_guard is not None else guard();g.check();stats=load(HERE/'prepared/statistics.npz')
    if not fixture:
        assert (HERE/'ALL6_FINALS_LOCK.json').exists();models={}
        for arm in ARMS:
            m=Refiner(stats,arm).cuda();ct=contract(seed,arm);ck=torch.load(HERE/f'runs/{arm}_seed{seed}/checkpoint.pt',map_location='cpu',weights_only=False);assert ck['contract']==ct;m.load_state_dict(ck['model']);models[arm]=m
    f=Path(folder) if folder else HERE/f'populations/seed{seed}/{split}';f.mkdir(parents=True,exist_ok=True);ct=dict(source=source(),seed=seed,split=split,fixture=fixture)
    if (f/'done.json').exists():completed(f,ct);return f
    assert not any(f.iterdir());count=limit or len(data['native']);before={a:{k:v.detach().cpu().clone() for k,v in m.state_dict().items()} for a,m in models.items()};chunks=[]
    for lo in range(0,count,128):
        hi=min(lo+128,count);x=torch.from_numpy(data['raw'][lo:hi]).cuda();n=torch.from_numpy(data['native'][lo:hi]).cuda();kw={k:data[k][lo:hi] for k in ('native','gt','sample_id','track_key','group')}
        for a,m in models.items():
            m.eval()
            with torch.no_grad():p,aux=m(x,n)
            kw.update({a+'_paths':p.cpu().numpy(),a+'_anchors':aux['anchors'].cpu().numpy(),a+'_gate':aux['gate'].cpu().numpy()})
        name=f'chunk{lo:05d}.npz';save(f/name,**kw);chunks.append(name);g.check();mark('COLLAB_SAVE',seed=seed,split=split,windows=hi)
    for a,m in models.items():assert all(torch.equal(v.cpu(),before[a][k]) for k,v in m.state_dict().items())
    resource(f/'RESOURCE_GPU.json',g);finish(f,ct,n=count,chunks=chunks,eval_all_weights_buffers_EXACT=True);return f
if __name__=='__main__':run(int(sys.argv[1]),sys.argv[2])
