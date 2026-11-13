"""Fixed eight-epoch new students; paired initial states and immutable permutations."""
from core import *

def tensors(data,teacher,idx,device):
    return [torch.from_numpy(np.ascontiguousarray(x[idx])).to(device) for x in (data['x'],data['gt'].astype(np.float32),teacher['R'],teacher['R_mask'],teacher['N'],teacher['N_mask'])]

def numpy_base(p,gt):
    e=np.sqrt(((p.astype(float)-gt[:,None].astype(float))**2).sum(-1));q=e.mean(-1)+.5*e[:,:,-1]
    ends=p.astype(float)[:,:,-1];dist=np.sqrt(((ends[:,PAIR[0]]-ends[:,PAIR[1]])**2).sum(-1))
    return float(q.min(1).mean()+.05*np.maximum(.3-dist,0).mean())

def quick(m,data,idx):
    p=predict(m,data['x'][idx]);er=errors(p,data['gt'][idx]);ne=errors(data['native'][idx],data['gt'][idx]);rows={}
    for label,g in [('overall',None),('stable',0),('transition',1),('change',2)]:
        mask=np.ones(len(idx),bool) if g is None else data['groups'][idx]==g
        rows[label]=dict(n=int(mask.sum()),good=int(er['coverage'][mask].sum()),native_good=int(ne['coverage'][mask].sum()),rescue=int((er['coverage']&~ne['coverage']&mask).sum()),lost=int((~er['coverage']&ne['coverage']&mask).sum()))
    return rows

def epoch(m,opt,g,data,teacher,order,arm,guard=None):
    m.train();n=len(order);sums={k:0. for k in ['loss','base','task','div','aux','preclip_grad_norm','R_sum','N_sum']};nr=nt=clipped=steps=ties=0;wta=np.zeros(20,np.int64);rwin=np.zeros(20,np.int64);nwin=np.zeros(20,np.int64);tt=0;batch_log=[]
    for i in range(0,n,256):
        if guard is not None:guard.check()
        idx=order[i:i+256];x,y,R,rm,N,nm=tensors(data,teacher,idx,next(m.parameters()).device)
        p=m(x);objective,info=loss(p,y,R,rm,N,nm,arm);assert torch.isfinite(objective)
        opt.zero_grad(set_to_none=True);objective.backward()
        grad=torch.nn.utils.clip_grad_norm_(m.parameters(),5.)
        assert torch.isfinite(grad) and all(t.grad is None or torch.isfinite(t.grad).all() for t in m.parameters())
        opt.step();steps+=1;clipped+=int(grad>5);nr+=int(rm.sum());nt+=int(nm.sum());ties+=int((info['winner_ties']-1).sum())
        wta+=torch.bincount(info['winner'],minlength=20).cpu().numpy()
        for key,target in [('R_winner',rwin),('N_winner',nwin)]:
            if info[key] is not None:target+=torch.bincount(info[key],minlength=20).cpu().numpy()
        tt+=info['teacher_ties']
        for k in ['base','task','div','aux','R_sum','N_sum']:sums[k]+=float(info[k].detach())*len(idx)
        sums['loss']+=float(objective.detach())*len(idx);sums['preclip_grad_norm']+=float(grad)*len(idx)
        batch_log.append(dict(n=len(idx),n_R=int(rm.sum()),n_N=int(nm.sum()),loss=float(objective.detach()),R_sum=float(info['R_sum'].detach()),N_sum=float(info['N_sum'].detach()),preclip_norm=float(grad),clip=bool(grad>5)))
    return dict(**{k:v/n for k,v in sums.items()},n=n,steps=steps,n_R=nr,n_N=nt,grad_clip_rate=clipped/steps,WTA=wta.tolist(),R_teacher_responsibility=rwin.tolist(),N_teacher_responsibility=nwin.tolist(),WTA_extra_ties=ties,teacher_extra_ties=tt,batches=batch_log,R_N_sum_aggregation='batch_N_weighted_sum_diagnostic_not_population_mean')

def optimizer(m):return torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)

def actual_assignment(p,data,teacher=None):
    e=errors(p,data['gt']);ne=errors(data['native'],data['gt']);q=e['ADE']+.5*e['FDE'];rows={};matches=[]
    for name,mask in [('native_success',ne['coverage']),('native_failure_teachable',~ne['coverage'] & (teacher['N_mask'].any(1) if teacher is not None else np.zeros(len(p),bool))),('native_failure_no_teacher',~ne['coverage'] & ~(teacher['N_mask'].any(1) if teacher is not None else np.zeros(len(p),bool)))]:
        rows[name]=dict(n=int(mask.sum()),good=int(e['coverage'][mask].sum()),rescue=int((e['coverage']&~ne['coverage']&mask).sum()),lost=int((~e['coverage']&ne['coverage']&mask).sum()))
    counts={}
    if teacher is not None:
        for label in ['R','N']:
            rr,ss=np.nonzero(teacher[label+'_mask']);wins=[]
            for i in range(0,len(rr),256):
                distance=np.linalg.norm(teacher[label][rr[i:i+256],ss[i:i+256]].astype(float)[:,None]-p[rr[i:i+256]].astype(float),axis=-1)
                qq=distance.mean(-1)+.5*distance[:,:,-1];wins.extend(qq.argmin(1).tolist())
            counts[label]=dict(count=len(rr),student_responsibility=np.bincount(wins,minlength=20).tolist(),row=rr,teacher_slot=ss,student_slot=np.array(wins,dtype=np.int16))
    return dict(subsets=rows,WTA=np.bincount(e['Q_winner'],minlength=20).tolist(),WTA_extra_ties=int(((q==q.min(1,keepdims=True)).sum(1)-1).sum()),teachers=counts)

def train_run(arm,s,data,teacher,guard):
    run=HERE/f'runs/{arm}_seed{s}';run.mkdir(parents=True,exist_ok=True);c=contract(arm,s)
    if (run/'done.json').exists():return valid_done(run/'done.json',c)
    m=new_model(s);initial=torch.load(HERE/f'initial_states/seed{s}.pt',weights_only=True);m.load_state_dict(initial)
    opt=optimizer(m);g=torch.Generator().manual_seed(s);orders=np.load(HERE/f'batch_orders/seed{s}.npz')['orders'];hist=[];assignment=[];step=0;start_epoch=0
    if (run/'resume.pt').exists():
        blob=torch.load(run/'resume.pt',weights_only=False,map_location='cpu');restore(blob,m,opt,g,c);hist=blob['history'];assignment=blob['assignment'];start_epoch=blob['epoch'];step=blob['step']
    diag=np.load(HERE/'diagnostic_ids.npz')['FIT_index']
    for ep in range(start_epoch,8):
        stage('TRAIN',status='RUNNING',arm=arm,seed=s,epoch=ep)
        t=time.perf_counter();rec=epoch(m,opt,g,data['FIT'],teacher,orders[ep],arm,guard);step+=rec['steps'];rec.update(epoch=ep+1,diagnostic=quick(m,data['FIT'],diag))
        dp=predict(m,data['FIT']['x'][diag]);sub={k:v[diag] for k,v in data['FIT'].items()};st={k:v[diag] for k,v in teacher.items() if isinstance(v,np.ndarray)}
        da=actual_assignment(dp,sub,st)
        for v in da['teachers'].values():
            for key in ['row','teacher_slot','student_slot']:v.pop(key)
        rec['diagnostic_assignment']=da;hist.append(rec);assignment.append(dict(epoch=ep+1,WTA=rec['WTA'],R_teacher_responsibility=rec['R_teacher_responsibility'],N_teacher_responsibility=rec['N_teacher_responsibility']))
        write_json(run/'history.json',hist);write_json(run/'assignment_history.json',assignment)
        blob=checkpoint(m,opt,g,ep+1,step,hist,assignment,c);torch.save(blob,run/'resume.pt')
        print('EPOCH',arm,s,ep+1,'seconds',time.perf_counter()-t,flush=True)
    assert len(hist)==8 and step==8*74
    torch.save(checkpoint(m,opt,g,8,step,hist,assignment,c),run/'checkpoint.pt')
    write_json(run/'run_manifest.json',dict(contract=c,checkpoint_sha256=sha256(run/'checkpoint.pt'),epochs=8,optimizer_steps=step,initial_output='all_zero_not_native_identity',paired_initial_exact=True,final_only=True))
    before=frozen_state(m);jobs={}
    for name,d in data.items():
        stage('EVALUATE_SAVE_AUDIT',arm=arm,seed=s,split=name)
        p=predict(m,d['x']);guard.check();pred=run/f'predictions_{name}.npz'
        np.savez_compressed(pred,paths=p,native=d['native'],gt=d['gt'],sample_id=d['ids'],groups=d['groups'],track=d['track'])
        metrics,er=evaluate(p,d['gt'],d['groups'],d['native'],d['ids'],d['track'])
        write_json(run/f'metrics_{name}.json',metrics)
        np.savez_compressed(run/f'sample_metrics_{name}.npz',**er)
        # Full independent old auditor is deliberately not replaced with training metrics.
        audit=independent.audit_file(pred,run/f'metrics_{name}.json',frozen_cache=prior.PATHS['cache'],frozen_labels=prior.PATHS['labels'],frozen_metadata=prior.PATHS['metadata'],frozen_manifest=ROOT/'config/data_manifest.v1.json',root=ROOT)
        assert audit['status']=='PASS',audit
        write_json(run/f'audit_{name}.json',audit)
        tt=teacher if name=='FIT' else inner_teacher(d,s)
        aa=actual_assignment(p,d,tt)
        # Save explicit row/slot responsibilities without bloating text summaries.
        arrays={}
        for key,v in aa['teachers'].items():
            for k in ['row','teacher_slot','student_slot']:arrays[key+'_'+k]=v.pop(k)
        np.savez_compressed(run/f'teacher_assignment_{name}.npz',sample_id=d['ids'],**arrays)
        write_json(run/f'assignment_{name}.json',aa);jobs[name]=metrics
        assert equal(before,frozen_state(m));guard.check();del p,er
    outputs={p.relative_to(HERE).as_posix():sha256(p) for p in run.iterdir() if p.is_file() and p.name not in ['done.json','resume.pt']}
    done=dict(status='SUCCESS',contract=c,outputs=outputs,metrics=jobs,eval_weights_bool_scalers_exact=True)
    write_json(run/'done.json',done);del m,opt;torch.cuda.empty_cache();return done

def inner_teacher(d,s):
    kin=np.load(prior.kin_path('inner',s));ne=errors(d['native'],d['gt']);ke=errors(kin,d['gt'])
    return dict(R=d['native'].astype(np.float32),R_mask=ne['good'],N=kin,N_mask=ke['good']&~ne['coverage'][:,None])
