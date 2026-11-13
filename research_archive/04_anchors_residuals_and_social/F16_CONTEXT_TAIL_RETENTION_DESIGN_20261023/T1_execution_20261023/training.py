from core import *

def epoch(m,o,g,d,order,arm,previous=None,guard=None):
    m.train();n=len(order);sums={k:0. for k in ('loss','base','task','div','aux','relative_excess','preclip_grad_norm')}
    slots=np.full((len(d['ids']),2),-1,np.int16);qh=np.zeros(20,np.int64);jh=qh.copy();batches=[];tiesq=tiesj=tiesc=active=clipped=steps=0
    strata={k:dict(n=0,base_task_sum=0.,excess_sum=0.) for k in ('native_success','native_failure')}
    for i in range(0,n,256):
        if guard:guard.check()
        ix=order[i:i+256];device=next(m.parameters()).device
        x=torch.from_numpy(np.ascontiguousarray(d['x'][ix])).to(device);y=torch.from_numpy(np.ascontiguousarray(d['gt'][ix].astype(np.float32))).to(device)
        nj=torch.from_numpy(np.ascontiguousarray(d['native_j'][ix])).to(device)
        p=m(x);value,info=objective(p,y,nj,arm);assert torch.isfinite(value)
        o.zero_grad(set_to_none=True);value.backward();grad=torch.nn.utils.clip_grad_norm_(m.parameters(),5.)
        assert torch.isfinite(grad) and all(t.grad is not None and torch.isfinite(t.grad).all() for t in m.parameters())
        o.step();b=len(ix);steps+=1;clipped+=int(grad>5)
        qslot=info['Qslot'].detach().cpu().numpy();jslot=info['Jslot'].detach().cpu().numpy();slots[ix]=np.stack((qslot,jslot),1)
        qh+=np.bincount(qslot,minlength=20);jh+=np.bincount(jslot,minlength=20)
        ex=info['excess'].detach().cpu().numpy();qq=info['Q'].min(1).values.detach().cpu().numpy();native_good=d['native_j'][ix]<=1
        for key,mask in [('native_success',native_good),('native_failure',~native_good)]:
            strata[key]['n']+=int(mask.sum());strata[key]['base_task_sum']+=float(qq[mask].sum());strata[key]['excess_sum']+=float(ex[mask].sum())
        active+=int((info['excess']>0).sum());tiesq+=int(info['Q_extra_ties'].sum());tiesj+=int(info['J_extra_ties'].sum());tiesc+=int(info['component_ties'].sum())
        vals={k:float(info[k].detach()) for k in ('base','task','div')};vals.update(loss=float(value.detach()),aux=.2*float(ex.mean()) if arm=='P_relative' else 0.,relative_excess=float(ex.mean()),preclip_grad_norm=float(grad))
        for k,v in vals.items():sums[k]+=v*b
        batches.append(dict(n=b,**vals,active=int((ex>0).sum()),preclip_norm=float(grad),clip=bool(grad>5)))
    switch=None if previous is None else dict(Q=float((slots[order,0]!=previous[order,0]).mean()),J=float((slots[order,1]!=previous[order,1]).mean()))
    rec=dict(**{k:v/n for k,v in sums.items()},n=n,steps=steps,grad_clip_rate=clipped/steps,WTA=qh.tolist(),J_responsibility=jh.tolist(),
        WTA_extra_ties=tiesq,J_extra_ties=tiesj,J_component_ties=tiesc,hinge_active_fraction=active/n,switch=switch,strata=strata,batches=batches)
    return rec,slots

def quick(m,d,idx):
    p=predict(m,d['x'][idx]);e=errors(p,d['gt'][idx]);ne=errors(d['native'][idx],d['gt'][idx]);rows={}
    for group,num in [('overall',None),('stable',0),('transition',1),('change',2)]:
        mask=np.ones(len(idx),bool) if num is None else d['groups'][idx]==num
        rows[group]=dict(n=int(mask.sum()),good=int(e['coverage'][mask].sum()),native_good=int(ne['coverage'][mask].sum()),rescue=int((e['coverage']&~ne['coverage']&mask).sum()),lost=int((~e['coverage']&ne['coverage']&mask).sum()))
    return rows

def train_run(arm,s,data,guard):
    con=contract(arm,s);run=HERE/f'runs/{arm}_seed{s}';run.mkdir(parents=True,exist_ok=True)
    if (run/'done.json').exists():return valid_done(run/'done.json',con)
    m=new_model(s);o=optimizer(m);g=torch.Generator().manual_seed(s);hist=[];steps=0;start=0;slots=None
    if (run/'resume.pt').exists():
        blob=torch.load(run/'resume.pt',weights_only=False,map_location='cpu');restore(blob,m,o,g,con);hist=blob['history'];steps=blob['step'];start=blob['epoch'];slots=blob['assignment']
    orders=np.load(HERE/f'batch_orders/seed{s}.npz')['orders'];diagidx=np.load(HERE/'diagnostic_ids.npz')['FIT_index']
    for ep in range(start,8):
        stage('TRAIN',status='RUNNING',arm=arm,seed=s,epoch=ep)
        rec,slots=epoch(m,o,g,data['FIT'],orders[ep],arm,slots,guard);steps+=rec['steps'];rec.update(epoch=ep+1,diagnostic=quick(m,data['FIT'],diagidx));hist.append(rec)
        write_json(run/'history.json',hist);torch.save(checkpoint(m,o,g,ep+1,steps,hist,slots,con),run/'resume.pt')
    assert len(hist)==8 and steps==592
    torch.save(checkpoint(m,o,g,8,steps,hist,slots,con),run/'checkpoint.pt')
    write_json(run/'run_manifest.json',dict(contract=con,epochs=8,steps=steps,trainable=82008,GT_native_in_forward=False,teacher_coordinate_loss=False,base='Qhard+.05 endpoint linear hinge',aux='0.2mean relu(Jstudent-detach Jnative)' if arm=='P_relative' else 'none',optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001,betas=[.9,.999],eps=1e-8,clip_norm=5),software=dict(torch=torch.__version__,numpy=np.__version__,device=torch.cuda.get_device_name()),dtype='float32',eval_dtype='float64',initial='old locked fresh zero output, not native identity'))
    for rel in ('statistics.npz','split_ids.npz'):shutil.copyfile(HERE/rel,run/rel)
    np.savez_compressed(run/'assignment_stats.npz',sample_id=data['FIT']['ids'],last_preupdate_QJslots=slots)
    reproduction={'history_common_fields':True,'paths_bitwise':True};jobs={};before=frozen_state(m)
    if arm=='A_replay':
        original=read_json(OLD/f'runs/A_seed{s}/history.json')
        for a,b in zip(hist,original):
            for key in ('loss','base','task','div','preclip_grad_norm','n','steps','grad_clip_rate','WTA','WTA_extra_ties','diagnostic'):
                if not equal(a[key],b[key]):reproduction['history_common_fields']=False
        write_json(run/'CONTROL_REPRODUCTION.json',reproduction)
    from evaluate_saved import save_evaluate
    for split,d in data.items():
        stage('EVALUATE_SAVE_AUDIT',arm=arm,seed=s,split=split)
        paths=predict(m,d['x']);assert equal(before,frozen_state(m));guard.check()
        if arm=='A_replay':
            with np.load(OLD/f'runs/A_seed{s}/predictions_{split}.npz') as z:
                assert np.array_equal(z['sample_id'],d['ids']);reproduction['paths_bitwise'] &= np.array_equal(paths,z['paths'])
            write_json(run/'CONTROL_REPRODUCTION.json',reproduction)
        jobs[split]=save_evaluate(run,split,paths,d,guard)
        del paths
    if arm=='A_replay' and not all(reproduction.values()):raise RuntimeError('CONTROL_REPRODUCTION_UNRESOLVED: no repeat training; preserve outputs')
    outputs={p.relative_to(HERE).as_posix():sha256(p) for p in run.iterdir() if p.is_file() and p.name not in ('done.json','resume.pt')}
    done=dict(status='SUCCESS',contract=con,outputs=outputs,metrics=jobs,eval_weights_bool_scalers_exact=True,control_reproduction=reproduction if arm=='A_replay' else None)
    write_json(run/'done.json',done);del m,o;torch.cuda.empty_cache();return done
