from model import *
from src.f16.checkpoint import capture_rng,restore_rng
def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
class Data:
    def __init__(self,limit=None):
        self.z=dataset('opt');self.rows=load(HERE/'prepared/split.npz')['train'];self.stats=load(HERE/'prepared/statistics.npz')
        if limit:self.rows=self.rows[:limit]
        self.count=len(self.rows)
    def batch(self,ix,device):return tuple(torch.from_numpy(self.z[k][self.rows[ix]].copy()).to(device) for k in ('raw','native','gt'))
def initialize(seed,arm,data,device='cuda',double=False):
    seed_all(seed);m=Refiner(data.stats,arm).to(device)
    if double:m=m.double()
    o=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8,foreach=False);r=torch.Generator().manual_seed(seed);return m,o,r
def fresh():return dict(epoch=0,cursor=0,order=None,updates=0,sums={k:0. for k in ('total','set','keep','motion','small','gradnorm')},history=[])
def step(m,o,r,data,st):
    if st['order'] is None:st['order']=torch.randperm(data.count,generator=r).numpy()
    ix=st['order'][st['cursor']:st['cursor']+32];x,n,y=data.batch(ix,next(m.parameters()).device);m.train();o.zero_grad(set_to_none=True);p,aux=m(x,n);value,parts=objective(p,y,n,m.motion_scales);assert torch.isfinite(value);value.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());norm=torch.nn.utils.clip_grad_norm_(m.parameters(),5);o.step()
    for k,v in dict(total=value,gradnorm=norm,**parts).items():st['sums'][k]+=float(v.detach())*len(ix)
    st['cursor']+=len(ix);st['updates']+=1;ep=False
    if st['cursor']==data.count:
        st['epoch']+=1;st['history'].append(dict(epoch=st['epoch'],updates=st['updates'],windows=data.count,order_SHA=arrsha(st['order']),**{k:v/data.count for k,v in st['sums'].items()}));st.update(cursor=0,order=None,sums={k:0. for k in st['sums']});ep=True
    return ep
def snapshot(m,o,r,st,ct):return copy.deepcopy(dict(model=m.state_dict(),optimizer=o.state_dict(),rng=capture_rng(r),state=st,contract=ct))
def restore(path,m,o,r,ct):
    z=torch.load(path,map_location='cpu',weights_only=False)
    if z['contract']!=ct:raise ValueError('Bad source/ID/stat/arm resume')
    m.load_state_dict(z['model']);o.load_state_dict(z['optimizer']);restore_rng(z['rng'],r);return z['state']
def reader(folder,ct,epochs=8):
    d=completed(folder,ct);z=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False);assert z['contract']==ct and z['state']['epoch']==epochs and z['state']['updates']==d['updates'];return d
def run(seed,arm):
    data=Data();g=guard();ct=contract(seed,arm);f=HERE/f'runs/{arm}_seed{seed}';f.mkdir(parents=True,exist_ok=True)
    if (f/'done.json').exists():return reader(f,ct)
    m,o,r=initialize(seed,arm,data);st=fresh();old=sorted(f.glob('update*_resume.pt'))
    if old:st=restore(old[-1],m,o,r,ct)
    else:torch.save(m.state_dict(),f/'initial.pt')
    write_json(f/'manifest.json',dict(contract=ct,training_ids=data.z['sample_id'][data.rows].tolist(),fixedfinal8=True,old_models_not_used=True))
    while st['epoch']<8:
        ep=step(m,o,r,data,st)
        if st['updates']%32==0:
            path=f/f"update{st['updates']:05d}_resume.pt";assert not path.exists();torch.save(snapshot(m,o,r,st,ct),path);g.check()
        if ep:
            write_json(f/'history.json',st['history']);mark('COLLAB_TRAIN',arm=arm,seed=seed,epoch=st['epoch'],updates=st['updates'])
            if st['epoch'] in (1,4,8):
                path=f/f"opt128_epoch{st['epoch']}.npz"
                if not path.exists():
                    before={k:v.detach().clone() for k,v in m.state_dict().items()};m.eval();x,n,y=data.batch(np.arange(128),'cuda')
                    with torch.no_grad():p,aux=m(x,n)
                    assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items());save(path,paths=p.cpu().numpy(),native=n.cpu().numpy(),gt=y.cpu().numpy(),sample_id=data.z['sample_id'][data.rows[:128]],gate=aux['gate'].cpu().numpy(),anchors=aux['anchors'].cpu().numpy())
    torch.save(snapshot(m,o,r,st,ct),f/'checkpoint.pt');resource(f/'RESOURCE.json',g);finish(f,ct,epochs=8,updates=st['updates']);reader(f,ct)
def lock():
    data=Data();models={};paired=[]
    for seed in SEEDS:
        init=[];orders=[]
        for arm in ARMS:
            f=HERE/f'runs/{arm}_seed{seed}';reader(f,contract(seed,arm));models[str((f/'checkpoint.pt').relative_to(HERE))]=sha256(f/'checkpoint.pt');init.append(torch.load(f/'initial.pt',map_location='cpu',weights_only=True));orders.append([h['order_SHA'] for h in read_json(f/'history.json')])
        assert init[0].keys()==init[1].keys() and all(torch.equal(init[0][k],init[1][k]) for k in init[0]) and orders[0]==orders[1];paired.append(dict(seed=seed,all_initial_parameters_buffers_EXACT=True,batch_ID_orders_EXACT=True))
    write_json(HERE/'ALL6_FINALS_LOCK.json',dict(source=source(),models=models,paired=paired,at=now(),finals=6,epochs=48,updates=6*8*((data.count+31)//32),held_not_yet_evaluated=True))
if __name__=='__main__':
    if sys.argv[1]=='lock':lock()
    else:run(int(sys.argv[1]),sys.argv[2])
