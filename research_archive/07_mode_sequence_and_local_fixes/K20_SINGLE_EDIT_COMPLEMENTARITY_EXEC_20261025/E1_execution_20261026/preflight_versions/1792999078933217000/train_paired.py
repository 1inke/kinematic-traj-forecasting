"""Six fresh fixed-final editors. No hold/inner labels and no generator training."""
import math,time,shutil,copy
import numpy as np
import torch
from env import *
from editor import ActionEditor,loss,choose
from data_pipeline import CACHE,OptData,contract
def initialize(seed,data,device):
    seed_all(seed);model=ActionEditor(data.mean,data.std)
    model.load_state_dict(torch.load(CACHE/f'initial_seed{seed}.pt',map_location='cpu',weights_only=True),strict=True);model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8,foreach=False)
    generator=torch.Generator().manual_seed(seed)
    return model,optimizer,generator
def train_epochs(model,optimizer,generator,data,epochs,history,arm,folder=None,expected_contract=None,guard=None):
    device=next(model.parameters()).device
    for epoch in range(len(history),epochs):
        model.train();order=torch.randperm(data.count,generator=generator).numpy()
        if data.count==4096 and not np.array_equal(order,data.orders[epoch]):raise ValueError('Registered paired order mismatch')
        total=0.;selected_good=0;stops=0;steps=0
        for start in range(0,data.count,32):
            ix=order[start:start+32];x,valid,target=data.batch(ix,device)
            optimizer.zero_grad(set_to_none=True);scores=model(x);objective=loss(scores,valid,target,arm)
            if not torch.isfinite(scores).all() or not torch.isfinite(objective):raise FloatingPointError('Nonfinite train scores/loss')
            action=choose(scores.detach(),valid);selected_good+=int(target.gather(1,action[:,None]).sum());stops+=int((action==0).sum())
            objective.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite/disconnected train gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();total+=float(objective.detach())*len(ix);steps+=1
        history.append(dict(epoch=epoch+1,steps=(epoch+1)*math.ceil(data.count/32),epoch_steps=steps,windows=data.count,loss=total/data.count,selected_target_count=selected_good,STOP_count=stops,order_SHA=array_sha(order)))
        if folder is not None:
            path=folder/f'epoch{epoch+1:02d}_resume.pt'
            if path.exists():raise FileExistsError('Historical epoch checkpoint must not be overwritten')
            save_checkpoint(path,model,optimizer,generator,history,expected_contract)
            write_json(folder/'history.json',history)
            stage('E1_TRAIN',arm=arm,seed=expected_contract['seed'],epoch=epoch+1,steps=history[-1]['steps'])
        if guard:guard.check()
    return history
def read_completed(folder,expected):
    done=read_json(folder/'done.json')
    if done['contract']!=expected:raise ValueError('Completed source/split/stats/initial/orders/arm contract')
    for name,ss in done['outputs'].items():
        if sha256(folder/name)!=ss:raise ValueError('Completed output SHA '+name)
    if done['epoch_rows']!=expected['epochs'] or done['steps']!=expected['steps']:raise ValueError('Completed epoch/step count')
    checkpoint=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    if checkpoint['contract']!=expected or len(checkpoint['history'])!=expected['epochs'] or checkpoint['history'][-1]['steps']!=expected['steps']:raise ValueError('Embedded checkpoint contract/history')
    if sha256(folder/'stats.npz')!=expected['statistics']:raise ValueError('Completed statistics file contract')
    with np.load(folder/'stats.npz') as z:ids=z['sample_id'][:expected['population_windows']]
    if not np.array_equal(read_json(folder/'split_ids.json')['FIT_opt'],ids):raise ValueError('Completed explicit split IDs')
    return done
def run(seed,arm,source,guard):
    folder=HERE/'runs'/f'{arm}_seed{seed}';folder.mkdir(parents=True,exist_ok=True);expected=contract(seed,arm,source)
    if (folder/'done.json').exists():return read_completed(folder,expected)
    data=OptData(seed);model,optimizer,generator=initialize(seed,data,'cuda');history=[]
    resumes=sorted(folder.glob('epoch*_resume.pt'))
    if resumes:
        history=load_checkpoint(resumes[-1],model,optimizer,generator,expected)
        if len(history)>8:raise ValueError('Resume epoch count')
    manifest=dict(contract=expected,only_factor='independent vs set-ranking loss',GT_forward=False,allK20=True,zero_last_nativeidentity=True,paired_seed=seed)
    write_json(folder/'run_manifest.json',manifest)
    train_epochs(model,optimizer,generator,data,8,history,arm,folder,expected,guard)
    assert len(history)==8 and history[-1]['steps']==1024
    save_checkpoint(folder/'checkpoint.pt',model,optimizer,generator,history,expected)
    write_json(folder/'assignment_stats.json',[{k:v for k,v in h.items() if k in ('epoch','selected_target_count','STOP_count','windows')} for h in history])
    shutil.copyfile(CACHE/f'stats_seed{seed}.npz',folder/'stats.npz')
    write_json(folder/'split_ids.json',dict(FIT_opt=data.ids.tolist(),population='4096 fixed hash-ordered only',hold_inner_evaluated=False))
    files=[p.name for p in folder.iterdir() if p.is_file() and p.name!='done.json']
    done=dict(status='COMPLETE_FINAL8_TRAINING_ONLY',contract=expected,epoch_rows=8,steps=1024,outputs={p:sha256(folder/p) for p in files})
    write_json(folder/'done.json',done);read_completed(folder,expected);return done
def lock_all_finals(source):
    locks={}
    for seed in SEEDS:
        for arm in ARMS:
            folder=HERE/'runs'/f'{arm}_seed{seed}';read_completed(folder,contract(seed,arm,source))
            locks[f'{arm}_seed{seed}']=dict(checkpoint=sha256(folder/'checkpoint.pt'),done=sha256(folder/'done.json'),manifest=sha256(folder/'run_manifest.json'),history=sha256(folder/'history.json'))
    write_json(HERE/'ALL6_FINALS_LOCK.json',dict(status='ALL6_FIXED_FINAL8_LOCKED',source=source,locks=locks,hold_inner_GT_unlock_permitted=True));return locks
