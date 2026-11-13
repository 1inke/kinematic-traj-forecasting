from core import *
from feature_blocks import assemble,read_stats
G=guard('MLP')
import torch,random,copy
from model import DiagnosticEditor,weighted_loss
from src.f16.checkpoint import capture_rng,restore_rng
torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
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
    def __init__(self,seed,limit=None):
        self.seed=seed
        with np.load(PREVIOUS/f'prepared/actions_seed{seed}.npz') as z:self.actions=z['actions'];self.weights=z['weights'];self.target=z['labels'];self.ids=z['sample_id']
        with np.load(HERE/'prepared/observed_blocks_opt.npz') as z:
            assert np.array_equal(self.ids,z['sample_id']);self.token=z['token'];self.neighbor=z['neighbor']
        self.n=len(self.ids) if limit is None else min(limit,len(self.ids))
    def batch(self,ix):
        mm=np.load(OLD/f'prepared_v1/features_seed{self.seed}.npy',mmap_mode='r')
        act=self.actions[ix];base=np.array(mm[ix[:,None],np.maximum(act,0)],copy=True);mm._mmap.close()
        raw=assemble(base,self.token[ix],self.neighbor[ix])
        return (torch.as_tensor(raw,device=DEVICE),torch.as_tensor(self.weights[ix],device=DEVICE,dtype=torch.float32),torch.as_tensor(self.target[ix],device=DEVICE))
class Engine:
    def __init__(self,data,arm,seed):
        seed_all(seed);self.data=data;self.arm=arm;self.seed=seed
        self.model=DiagnosticEditor(*read_stats(data.seed),arm).to(DEVICE)
        self.opt=torch.optim.AdamW(self.model.parameters(),lr=.001,weight_decay=.0001,foreach=False)
        self.loader=torch.Generator().manual_seed(seed);self.order=None;self.cursor=0;self.step=0;self.history=[]
    def snapshot(self,c):return dict(model=copy.deepcopy(self.model.state_dict()),optimizer=copy.deepcopy(self.opt.state_dict()),rng=capture_rng(self.loader),order=self.order.clone() if self.order is not None else None,cursor=self.cursor,step=self.step,history=copy.deepcopy(self.history),contract=c)
    def restore(self,s,c):
        if s['contract']!=c:raise ValueError('Resume source/stat/arm contract mismatch')
        self.model.load_state_dict(s['model'],strict=True);self.opt.load_state_dict(s['optimizer']);restore_rng(s['rng'],self.loader)
        self.order=s['order'];self.cursor=s['cursor'];self.step=s['step'];self.history=s['history']
    def run(self,until,callback=None,progress=False):
        while self.step<until:
            if self.order is None or self.cursor==self.data.n:self.order=torch.randperm(self.data.n,generator=self.loader);self.cursor=0
            ix=self.order[self.cursor:self.cursor+32].numpy();self.cursor+=len(ix);x,w,y=self.data.batch(ix)
            self.opt.zero_grad(set_to_none=True);s=self.model(x);loss=weighted_loss(s,w,y)
            if not torch.isfinite(loss):raise ArithmeticError('Nonfinite loss')
            loss.backward()
            for p in self.model.parameters():
                if p.grad is None or not torch.isfinite(p.grad).all():raise ArithmeticError('Nonfinite/disconnected gradient')
            norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),5);self.opt.step();self.step+=1
            self.history.append(dict(update=self.step,loss=float(loss.detach()),gradient_norm=float(norm),batch_ID_rows=ix.tolist()))
            if self.step%32==0:
                G.check()
                if progress:write_json(HERE/'progress.json',dict(stage='STEP3_TRAIN',arm=self.arm,seed=self.seed,update=self.step,at=now()))
            if callback:callback(self)
def contract(arm,seed):
    return dict(source=source(),arm=arm,seed=seed,updates=1024,stats_SHA=sha256(HERE/f'prepared/stats_seed{seed}.npz'),
        table_SHA=sha256(PREVIOUS/f'prepared/actions_seed{seed}.npz'),feature_provenance_SHA=sha256(HERE/'FEATURE_PROVENANCE.json'),objective='original_shared_IPW_BCE',parameters=42208)
def read_model(folder,c):
    completed(folder,c);s=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
    if s['contract']!=c or s['step']!=1024 or len(s['history'])!=1024:raise ValueError('Final embedded contract/history')
    mean,std=read_stats(c['seed']);assert np.array_equal(s['model']['feature_mean'].numpy(),mean) and np.array_equal(s['model']['feature_std'].numpy(),std)
    return s
def main(arm,seed):
    assert arm in read_json(HERE/'FEATURE_PROVENANCE.json')['admissible_arms'];assert read_json(HERE/'PRELAUNCH_REVIEW.json')['status']=='PASS'
    f=HERE/'runs'/f'{arm}_seed{seed}';f.mkdir(parents=True,exist_ok=True);c=contract(arm,seed)
    if (f/'done.json').exists():read_model(f,c);return
    if list(f.iterdir()):raise FileExistsError('Preserve partial; compatible explicit ALL-state resume required')
    e=Engine(Data(seed),arm,seed);write_json(f/'manifest.json',c);torch.save(e.snapshot(c),f/'initial.pt')
    def callback(e):
        if e.step in (128,512,1024):torch.save(e.snapshot(c),f/f'checkpoint{e.step:04d}.pt');write_json(f/'history.json',e.history)
    e.run(1024,callback,True);torch.save(e.snapshot(c),f/'final.pt');write_json(f/'history.json',e.history)
    resource_report(f/'RESOURCE.json',G);finish(f,c,updates=1024,epochs=8);read_model(f,c)
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))
