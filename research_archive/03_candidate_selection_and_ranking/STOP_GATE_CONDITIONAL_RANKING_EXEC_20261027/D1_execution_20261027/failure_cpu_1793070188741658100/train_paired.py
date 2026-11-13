import copy,random,math,time
from core import *
from objective import *
from src.f16.checkpoint import capture_rng,restore_rng
seedns={'np':np,'random':random};pure(OLD/'env.py',('seed_all',),seedns);seed_all=seedns['seed_all']
def initialize(seed,data,device='cuda'):
    seed_all(seed);m=ActionEditor(data.mean,data.std);m.load_state_dict(torch.load(CACHE/f'initial_seed{seed}.pt',map_location='cpu',weights_only=True));m.to(device)
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8,foreach=False);gen=torch.Generator().manual_seed(seed)
    return m,opt,gen
def fresh_state():return dict(epoch=0,cursor=0,order=None,updates=0,total=0.,selected=0,stops=0,gradnorm_total=0.,gate_total=0.,rank_total=0.,history=[])
def snapshot(m,opt,gen,state,contract):return copy.deepcopy(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=capture_rng(gen),state=state,contract=contract))
def restore(path,m,opt,gen,expected):
    z=torch.load(path,map_location='cpu',weights_only=False)
    if z['contract']!=expected:raise ValueError('Resume source/stats/pool/order/objective mismatch')
    m.load_state_dict(z['model']);opt.load_state_dict(z['optimizer']);restore_rng(z['rng'],gen);return z['state']
def step(m,opt,gen,data,arm,state):
    if state['order'] is None:
        order=torch.randperm(data.count,generator=gen).numpy();state['order']=order
        if data.count==4096:assert np.array_equal(order,data.orders[state['epoch']])
    ix=state['order'][state['cursor']:state['cursor']+32];m.train();x,e,t=data.batch(ix,next(m.parameters()).device)
    opt.zero_grad(set_to_none=True);scores=m(x);v=loss(scores,e,t,arm);assert torch.isfinite(v) and torch.isfinite(scores).all()
    ch=choose(scores.detach(),e);state['selected']+=int(t.gather(1,ch[:,None]).sum());state['stops']+=int((ch==0).sum())
    v.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    gradnorm=torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();state['total']+=float(v.detach())*len(ix);state['gradnorm_total']+=float(gradnorm)*len(ix)
    if arm=='D':
        _,components=decomposed(scores.detach(),e,t);state['gate_total']+=float(components['gate'].sum());state['rank_total']+=float(components['rank'].sum())
    state['cursor']+=len(ix);state['updates']+=1;completed_epoch=False
    if state['cursor']==data.count:
        n=data.count;epoch=state['epoch']+1;state['history'].append(dict(epoch=epoch,steps=state['updates'],epoch_steps=math.ceil(n/32),windows=n,loss=state['total']/n,selected_target_count=state['selected'],STOP_count=state['stops'],order_SHA=arrsha(state['order']),gradient_norm_mean=state['gradnorm_total']/n,gate_component=state['gate_total']/n if arm=='D' else None,conditional_rank_component=state['rank_total']/n if arm=='D' else None))
        state.update(epoch=epoch,cursor=0,order=None,total=0.,selected=0,stops=0,gradnorm_total=0.,gate_total=0.,rank_total=0.);completed_epoch=True
    return completed_epoch
def infer_opt(m,data):
    before={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};ss=[];m.eval()
    for lo in range(0,data.count,128):
        with torch.no_grad():ss.append(m(data.batch(np.arange(lo,min(lo+128,data.count)),next(m.parameters()).device)[0]).cpu().numpy())
    assert all(torch.equal(v.cpu(),before[k]) for k,v in m.state_dict().items())
    s=np.concatenate(ss);a=np.where(data.eligible,s,-np.inf).argmax(1);return s,a
def validate_completed(folder,expected):
    d=completed(folder,expected);ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert ck['contract']==expected and ck['state']['epoch']==expected['epochs'] and ck['state']['updates']==expected['steps'];return d
def bridge(m,history,seed,arm):
    if arm=='D':return dict(status='NOT_APPLICABLE_NEW_OBJECTIVE')
    ck=torch.load(OLD/f'runs/{arm}_seed{seed}/checkpoint.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(v.detach().cpu(),ck['model'][k]) for k,v in m.state_dict().items()),'CONTROL_REPRODUCTION_UNRESOLVED weights'
    for old,new in zip(ck['history'],history):
        assert all(new[k]==v for k,v in old.items()),'CONTROL_REPRODUCTION_UNRESOLVED common history'
    return dict(status='PASS_OLD_IS_FINAL_WEIGHTS_COMMON_HISTORY_EXACT',old_checkpoint_SHA=sha256(OLD/f'runs/{arm}_seed{seed}/checkpoint.pt'))
def run(seed,arm):
    g=guard();data=OptData(seed);expected=contract(seed,arm,source0());folder=HERE/f'runs/{arm}_seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'done.json').exists():return validate_completed(folder,expected)
    m,opt,gen=initialize(seed,data);state=fresh_state();resumes=sorted(folder.glob('update*_resume.pt'))
    if resumes:state=restore(resumes[-1],m,opt,gen,expected)
    write_json(folder/'run_manifest.json',dict(contract=expected,only_factor='registered I/S/D supervision organization',all401=True,no_GT_forward=True,raw_input_stats_initial_orders_reused_SHA=True))
    while state['epoch']<8:
        ep=step(m,opt,gen,data,arm,state)
        if state['updates']%32==0:
            f=folder/f"update{state['updates']:04d}_resume.pt"
            if f.exists():raise FileExistsError('Preserve existing state')
            torch.save(snapshot(m,opt,gen,state,expected),f);g.check()
        if ep:
            write_json(folder/'history.json',state['history']);mark('D1_TRAIN',arm=arm,seed=seed,epoch=state['epoch'],update=state['updates'])
            if state['epoch'] in (1,4,8):
                f=folder/f"opt_reading_epoch{state['epoch']}.npz"
                if not f.exists():
                    s,a=infer_opt(m,data);save_npz(f,scores=s,chosen=a,sample_id=data.ids)
    assert state['updates']==1024
    # A crash between an update checkpoint and its diagnostic readout never
    # justifies retraining. Reconstruct only missing registered opt readouts
    # from this run's own complete RNG-bound snapshots, preserving live RNG.
    rng_before=capture_rng(gen)
    for epoch in (1,4,8):
        f=folder/f'opt_reading_epoch{epoch}.npz'
        if not f.exists():
            diagnostic=copy.deepcopy(m);ck=torch.load(folder/f'update{epoch*128:04d}_resume.pt',map_location='cpu',weights_only=False);assert ck['contract']==expected;diagnostic.load_state_dict(ck['model']);s,a=infer_opt(diagnostic,data);save_npz(f,scores=s,chosen=a,sample_id=data.ids);del diagnostic
    restore_rng(rng_before,gen)
    write_json(folder/'CONTROL_BRIDGE.json',bridge(m,state['history'],seed,arm));torch.save(snapshot(m,opt,gen,state,expected),folder/'checkpoint.pt')
    resource(folder/'RESOURCE.json',g);data.close();finish(folder,expected,epochs=8,steps=1024);return validate_completed(folder,expected)
def lock_finals():
    locks={}
    for seed in SEEDS:
        for arm in ARMS:
            f=HERE/f'runs/{arm}_seed{seed}';validate_completed(f,contract(seed,arm,source0()));locks[f'{arm}_seed{seed}']={k:sha256(f/k) for k in ('checkpoint.pt','done.json','history.json','run_manifest.json','CONTROL_BRIDGE.json')}
    write_json(HERE/'ALL9_FINALS_LOCK.json',dict(status='ALL9_FINAL8_LOCKED_BEFORE_HELD',source=source0(),locks=locks,at=now()));return locks
if __name__=='__main__':run(int(sys.argv[1]),sys.argv[2])
