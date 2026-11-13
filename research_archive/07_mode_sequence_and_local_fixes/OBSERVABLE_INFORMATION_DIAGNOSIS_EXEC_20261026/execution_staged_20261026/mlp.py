"""Original editor/loss; bounded per-window loader, exact sampler/RNG resume."""
from base import *
G=guard('MLP')
import torch,random,copy
from torch.nn import functional as F
from src.f16.checkpoint import capture_rng,restore_rng
ed=module('_observable_original_editor',OLD/'editor.py')
torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.use_deterministic_algorithms(True)
DEVICE='cuda'

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def exact(a,b):
    if torch.is_tensor(a):assert a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,np.ndarray):assert a.dtype==b.dtype and np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b,(a,b)

class Data:
    def __init__(self,kind,pool=42,limit=None):
        self.kind=kind;self.pool=pool
        if kind=='tiny':
            with np.load(HERE/'prepared/tiny.npz') as z:self.x=z['features'];self.e=z['eligible'];self.y=z['target']
            self.n=len(self.x)
        else:
            with np.load(HERE/f'prepared/actions_seed{pool}.npz') as z:self.a=z['actions'];self.w=z['weights'];self.y=z['labels']
            self.n=len(self.a)
        if limit:self.n=min(self.n,limit)
    def batch(self,ix):
        if self.kind=='tiny':return tuple(torch.as_tensor(v[ix],device=DEVICE) for v in (self.x,self.e,self.y))
        # No 2.13GB feature file copied into RAM: only selected action rows.
        mm=np.load(OLD/f'prepared_v1/features_seed{self.pool}.npy',mmap_mode='r')
        a=self.a[ix];x=np.array(mm[ix[:,None],np.maximum(a,0)],copy=True);mm._mmap.close()
        return (torch.as_tensor(x,device=DEVICE),torch.as_tensor(self.w[ix],device=DEVICE,dtype=torch.float32),torch.as_tensor(self.y[ix],device=DEVICE))

def weighted_loss(s,w,y):return (w*torch.where(y,F.softplus(-s),F.softplus(s))).sum(-1).mean()

class Engine:
    def __init__(self,data,arm,seed):
        seed_all(seed);self.data=data;self.arm=arm;self.seed=seed
        mean,std=stats(data.pool);self.model=ed.ActionEditor(mean,std).to(DEVICE)
        self.opt=torch.optim.AdamW(self.model.parameters(),lr=.001,weight_decay=.0001,foreach=False)
        self.loader=torch.Generator().manual_seed(seed);self.order=None;self.cursor=0;self.step=0;self.history=[]
    def snapshot(self,contract):
        return dict(model=copy.deepcopy(self.model.state_dict()),optimizer=copy.deepcopy(self.opt.state_dict()),rng=capture_rng(self.loader),order=self.order.clone() if self.order is not None else None,cursor=self.cursor,step=self.step,history=copy.deepcopy(self.history),contract=contract)
    def restore(self,state,contract):
        if state['contract']!=contract:raise ValueError('Resume contract mismatch')
        self.model.load_state_dict(state['model'],strict=True);self.opt.load_state_dict(state['optimizer']);restore_rng(state['rng'],self.loader)
        self.order=state['order'];self.cursor=state['cursor'];self.step=state['step'];self.history=state['history']
    def run(self,until,callback=None,progress=True):
        while self.step<until:
            if self.order is None or self.cursor==self.data.n:self.order=torch.randperm(self.data.n,generator=self.loader);self.cursor=0
            ix=self.order[self.cursor:self.cursor+32].numpy();self.cursor+=len(ix)
            x,a,y=self.data.batch(ix);self.model.train();self.opt.zero_grad(set_to_none=True);s=self.model(x)
            loss=ed.loss(s,a,y,self.arm) if self.data.kind=='tiny' else weighted_loss(s,a,y)
            if not torch.isfinite(loss):raise ArithmeticError('Nonfinite training loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.model.parameters()):raise ArithmeticError('Invalid legal gradient')
            norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),5);self.opt.step();self.step+=1
            self.history.append(dict(update=self.step,loss=float(loss.detach()),gradient_norm=float(norm),batch_ID_rows=ix.tolist()))
            if self.step%32==0:
                G.check()
                if progress:mark('MLP_TRAIN',kind=self.data.kind,arm=self.arm,seed=self.seed,update=self.step,until=until)
            if callback:callback(self)
    def tiny_reading(self,path):
        was=self.model.training;self.model.eval();scores=[];before=copy.deepcopy(self.model.state_dict())
        with torch.inference_mode():
            for lo in range(0,self.data.n,32):scores.append(self.model(torch.as_tensor(self.data.x[lo:lo+32],device=DEVICE)).cpu().numpy())
        exact(before,self.model.state_dict());self.model.train(was);save_npz(path,scores=np.concatenate(scores))

def contract(kind,arm,seed,pool):
    return dict(source=source(),kind=kind,arm=arm,seed=seed,pool=pool,prepared_sha=sha256(HERE/'prepared/done.json'),stats_sha=sha256(OLD/f'prepared_v1/stats_seed{pool}.npz'),original_editor=sha256(OLD/'editor.py'),updates=512 if kind=='tiny' else 1024)

def read_model(folder,c):
    d=completed(folder,c);state=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
    if state['contract']!=c or state['step']!=c['updates'] or len(state['history'])!=c['updates']:raise ValueError('Embedded final model/history/contract mismatch')
    for k,v in state['model'].items():
        if v.dtype.is_floating_point and not torch.isfinite(v).all():raise ValueError('Nonfinite final model '+k)
    mean,std=stats(c['pool']);assert np.array_equal(state['model']['feature_mean'].numpy(),mean) and np.array_equal(state['model']['feature_std'].numpy(),std)
    return state

def train(kind,arm,seed):
    pool=42 if kind=='tiny' else seed;folder=HERE/('step1' if kind=='tiny' else 'step2')/f'{arm}_seed{seed}';folder.mkdir(parents=True,exist_ok=True);c=contract(kind,arm,seed,pool)
    if (folder/'done.json').exists():read_model(folder,c);print('SHA_SKIP',folder);return
    if list(folder.iterdir()):raise RuntimeError('Partial run preserved; explicit compatible exact-resume required')
    data=Data(kind,pool);engine=Engine(data,arm,seed);write_json(folder/'manifest.json',c)
    if kind=='tiny':engine.tiny_reading(folder/'reading000.npz')
    def callback(e):
        if kind=='tiny' and e.step in (32,128,512):e.tiny_reading(folder/f'reading{e.step:03d}.npz')
        if e.step in ((128,512) if kind=='tiny' else (128,512,1024)):
            torch.save(e.snapshot(c),folder/f'checkpoint{e.step:04d}.pt')
            write_json(folder/'history.json',e.history)
    engine.run(c['updates'],callback);torch.save(engine.snapshot(c),folder/'final.pt');write_json(folder/'history.json',engine.history)
    resource_report(folder,G);finish(folder,c,updates=engine.step,epochs=128 if kind=='tiny' else 8);read_model(folder,c)

def predict(arm,seed,split):
    from evaluate_saved import build_features,save_population
    folder=HERE/'step2'/f'{arm}_seed{seed}';c=contract('probe',arm,seed,seed);state=read_model(folder,c)
    lock=read_json(HERE/'ALL_PROBE_FINALS_LOCK.json');assert lock['models'][f'{arm}_seed{seed}']==sha256(folder/'final.pt')
    mean,std=stats(seed);m=ed.ActionEditor(mean,std).to(DEVICE);m.load_state_dict(state['model']);m.eval();before=copy.deepcopy(m.state_dict())
    with torch.inference_mode():
        def score(x):
            return np.concatenate([m(torch.as_tensor(x[i:i+32],device=DEVICE)).cpu().numpy() for i in range(0,len(x),32)])
        save_population(arm,seed,split,score,G)
    exact(before,m.state_dict())

if __name__=='__main__':
    try:
        if sys.argv[1]=='train':train(sys.argv[2],sys.argv[3],int(sys.argv[4]))
        else:predict(sys.argv[2],int(sys.argv[3]),sys.argv[4])
    except BaseException as e:fail(e);raise
