"""Q scientific training only after complete preflight and source lock."""
from common import *
from objectives import loss

def step(m,opt,x,y):
    m.train();p=m(x);v,info=loss(p,y);opt.zero_grad(set_to_none=True);v.backward()
    n=torch.nn.utils.clip_grad_norm_(m.parameters(),5.)
    assert torch.isfinite(v) and torch.isfinite(n)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    opt.step();return float(v.detach()),float(n),info
def blob(m,opt,g,hist,epoch,cursor,orders,c):
    return dict(model=state(m),optimizer=copy.deepcopy(opt.state_dict()),rng=capture_rng(g),history=copy.deepcopy(hist),epoch=epoch,cursor=cursor,orders=np.asarray(orders).copy(),contract=copy.deepcopy(c))
def restore(z,m,opt,g,c,orders):
    reject(z['contract'],c)
    if not np.array_equal(z['orders'],orders):raise ValueError('SAMPLER_ORDER_MISMATCH')
    m.load_state_dict(z['model']);opt.load_state_dict(z['optimizer']);restore_rng(z['rng'],g)
    return copy.deepcopy(z['history']),z['epoch'],z['cursor']
def predict(m,x,all_groups=True):
    G=1+len(m.aux) if all_groups else 1;answer=np.empty((len(x),G,20,12,2),np.float32);m.eval()
    with torch.inference_mode():
        for i in range(0,len(x),256):
            n=min(256,len(x)-i);inp=np.zeros((256,210),np.float32);inp[:n]=x[i:i+n]
            value=m(torch.from_numpy(inp).to(next(m.parameters()).device),train_groups=all_groups)
            if not all_groups:value=value[:,None]
            answer[i:i+n]=value[:n].cpu().numpy()
    assert np.isfinite(answer).all();return answer
def gradient_diagnostic(m,d,run,ep,ix):
    m.train();x=torch.from_numpy(d['x'][ix]).cuda();y=torch.from_numpy(d['gt'][ix].astype(np.float32)).cuda();p=m(x)
    params=[v for k,v in m.named_parameters() if k.startswith(('head.','fuse.'))];vectors=[];norms=[]
    for group in range(p.shape[1]):
        v,_=loss(p[:,group:group+1],y);gs=torch.autograd.grad(v,params,retain_graph=True)
        a=torch.cat([z.flatten() for z in gs]).detach().cpu().double().numpy();vectors.append(a);norms.append(float(np.linalg.norm(a)))
    cosine=[[float(np.dot(a,b)/(na*nb)) if na and nb else None for b,nb in zip(vectors,norms)] for a,na in zip(vectors,norms)]
    npz(run/f'gradient_diagnostic_epoch{ep}.npz',indices=ix,sample_id=d['ids'][ix],paths=p.detach().cpu().numpy(),shared_gradients=np.stack(vectors))
    write(run/f'gradient_diagnostic_epoch{ep}.json',dict(epoch=ep,n=128,scope='shared head+fuse only',norms=norms,cosine=cosine,zero_norm_cosine='NA',updates=0))
    m.zero_grad(set_to_none=True)
def train_run(arm,s,data,g):
    run=HERE/f'runs/{arm}_{s}';run.mkdir(parents=True,exist_ok=False);groups={'Q1':1,'Q3':3}[arm];c=contract(arm,s)
    m=new_model(s,groups);initial=torch.load(HERE/f'initial/{arm}_{s}.pt',map_location='cpu',weights_only=True);m.load_state_dict(initial)
    opt=optimizer(m);loader=torch.Generator().manual_seed(s);orders=load_npz(HERE/f'orders/{s}.npz')['orders'];hist=[];d=data['FIT'];diag=load_npz(HERE/'diagnostic_ids.npz')['indices']
    gradient_diagnostic(m,d,run,0,diag)
    for ep,order in enumerate(orders):
        stage('Q_TRAIN',arm=arm,seed=s,epoch=ep,completed_updates=ep*74);t=time.perf_counter();counts=np.zeros((groups,20),np.int64);ties=np.zeros(groups,np.int64);batches=[]
        for i in range(0,len(order),256):
            idx=order[i:i+256];g.check();x=torch.from_numpy(d['x'][idx]).cuda();y=torch.from_numpy(d['gt'][idx].astype(np.float32)).cuda()
            v,n,info=step(m,opt,x,y)
            for k,z in enumerate(info):counts[k]+=torch.bincount(z['winner'],minlength=20).cpu().numpy();ties[k]+=int((z['ties']>1).sum())
            batches.append(dict(n=len(idx),IDs=digest(d['ids'][idx]),loss=v,preclip_norm=n,clip=n>5))
        entropy=[]
        for row in counts:
            pr=row/row.sum();entropy.append(float(-np.sum(pr[pr>0]*np.log(pr[pr>0]))/np.log(20)))
        hist.append(dict(epoch=ep+1,steps=len(batches),winner_counts=counts,winner_entropy=entropy,never_won=(counts==0).sum(1),tied_windows=ties,batches=batches,seconds=time.perf_counter()-t))
        write(run/f'history_epoch{ep+1}.json',hist)
        save_torch(run/f'epoch{ep+1}.pt',blob(m,opt,loader,hist,ep+1,0,orders,c))
        if ep+1 in (1,4,8):gradient_diagnostic(m,d,run,ep+1,diag)
    assert len(hist)==8 and sum(x['steps'] for x in hist)==592
    write(run/'history.json',hist);save_torch(run/'checkpoint.pt',blob(m,opt,loader,hist,8,0,orders,c));save_torch(run/'export_main_group0.pt',state(m.export()))
    # Exact export at same fixed compute256, including nonzero trained weights.
    exp=m.export().cuda();before=state(m);xx=torch.from_numpy(d['x'][diag[:8]]).cuda()
    with torch.inference_mode():assert torch.equal(m(xx,False),exp(xx)) and torch.equal(m(xx)[:,0],exp(xx))
    del exp
    from evaluate_saved import save_population
    for split,pop in data.items():
        stage('Q_EVALUATE_SAVE_INDEPENDENT_AUDIT',arm=arm,seed=s,split=split)
        paths=predict(m,pop['x']);save_population(run/split,paths,pop,g);assert equal(before,state(m));del paths
    outputs={p.relative_to(HERE).as_posix():sha(p) for p in run.rglob('*') if p.is_file()}
    write(run/'done.json',dict(status='SUCCESS',contract=c,files=outputs,epochs=8,updates=592,all_eval_weights_buffers_exact=True,export_group0_exact=True))
    del m,opt;torch.cuda.empty_cache();g.check(force=True)

def main():
    from budget_admission import validate
    setup();validate(HERE)
    mf=read(HERE/'source_manifest.json');E=read(HERE/'MEMORY_ESTIMATE.json')['GPU_E'];g=Guard(E)
    stage('Q_INITIAL_SOURCE_SHA');verify(mf['files'],g.check);d=data();started=time.perf_counter()
    from evaluate_saved import save_population
    for name,v in d.items():
        stage('Q_NATIVE_REFERENCE_FULL_SAVED_AUDIT',split=name);save_population(HERE/f'native/{name}',v['native'][:,None],v,g)
    for s in SEEDS:
        for arm in ('Q1','Q3'):train_run(arm,s,d,g)
    stage('Q_FINAL_SOURCE_SHA');verify(mf['files'],g.check);g.check(force=True)
    write(HERE/'SCIENCE_RESOURCE.json',dict(status='PASS',policy='warning_first_v3',E=E,observations=g.observations))
    write(HERE/'SCIENCE_DONE.json',dict(status='SUCCESS_SCIENCE_FULL_SAVED_ARITHMETIC',models=6,epochs=48,updates=3552,seconds=time.perf_counter()-started,source_digest=mf['digest']))
if __name__=='__main__':main()
