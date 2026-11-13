"""Prospectively fixed observational diagnostics, not another training objective."""
from ud_common import *
from train_core_ud import torch,np,bf,cpu_tree,exact
from preflight import hash_indices


def observed(past):
    v=np.diff(past.astype(np.float64),axis=1)/.4;s=np.linalg.norm(v,axis=-1)
    denom=s[:,:-1]*s[:,1:];c=(v[:,:-1]*v[:,1:]).sum(-1)
    angles=np.arccos(np.clip(np.divide(c,denom,out=np.ones_like(c),where=denom>1e-12),-1,1))
    return angles[:,-3:].max(1),np.abs(np.diff(s,axis=1))[:,-3:].max(1)


def calibrate():
    fit=dict(np.load(BASE/'C0/fit.npz'));turn,speed=observed(fit['past'][:,0])
    write_json(RUN/'MECHANISM_CONTRACT.json',dict(status='LOCKED_FIT_PAST_ONLY_BEFORE_PILOTS',
        turn_threshold=float(np.quantile(turn,.75)),speed_change_threshold=float(np.quantile(speed,.75)),
        criterion='last3 consecutive observed velocity transitions; greater than fit-only linear75% quantile; no future calibration',
        dev_hash='ud-mechanism-v1',windows=128,perturbation_m=.05,
        perturbation='Target last world observation +/- .05 lateral to unperturbed last velocity (fallback x heading), keep causal neighbor identities/coordinates fixed, recompute heading normalization; sensitivity only, not full reranked deployment',
        main_candidates=['U-Delta','D-DiffGate'],
        subjective_clear_mechanism_clause='No automatic promotion based on subjective mechanism narrative; full objective pilot screen is computed. Borderline mechanism evidence requires explicit prospective next-stage review, not threshold search.',
        zero_positive_excess_p95='NOT_ESTIMABLE_NO_FLOOR; cannot declare relative tail noninferiority or automatic screen pass',
        physical_slots='Not semantic intent identities; signed difference attention is not probability or causal proof'))


@torch.no_grad()
def collect_mechanism(model,arm,fit,dev,guard):
    model.eval();before=cpu_tree(model.state_dict());rule=read_json(RUN/'MECHANISM_CONTRACT.json')
    out=RUN/'mechanisms'/arm;out.mkdir(parents=True,exist_ok=False)
    diagnostics={}
    if arm.startswith('U'):
        for split,data in [('fit',fit),('dev',dev)]:
            blocks={}
            for start in range(0,len(data['ids']),64):
                idx=slice(start,start+64)
                bf.prediction(model,torch.tensor(data['past'][idx],device='cuda'),torch.tensor(data['num_valid'][idx],device='cuda'))
                for k,v in model.decoder.update.diagnostics.items():blocks.setdefault(k,[]).append(v.cpu().numpy())
                guard.check()
            blocks={k:np.concatenate(v) for k,v in blocks.items()};turn,speed=observed(data['past'][:,0])
            maneuver=(turn>rule['turn_threshold'])|(speed>rule['speed_change_threshold'])
            blocks.update(ids=data['ids'],observed_turn=turn,observed_speed_change=speed,observed_maneuver=maneuver,group=data['group'])
            np.savez_compressed(out/f'{split}_response.npz',**blocks)
            strata={}
            for name,mask in [('all',np.ones(len(maneuver),bool)),('observed_maneuver',maneuver),('not_observed_maneuver',~maneuver),('future_change_without_observed_maneuver',(data['group']==2)&~maneuver)]:
                strata[name]=dict(n=int(mask.sum()),values={k:float(v[mask,-4:].mean()) if v.ndim==2 and mask.any() else (float(v[mask].mean()) if mask.any() else None) for k,v in blocks.items() if k in model.decoder.update.diagnostics})
            diagnostics[split]=strata
        idx=hash_indices(dev['ids'],rule['dev_hash'],128);raw=dev['past'][idx].copy();n=dev['num_valid'][idx]
        velocity=raw[:,0,-1]-raw[:,0,-2];norm=np.linalg.norm(velocity,axis=-1,keepdims=True)
        direction=np.divide(velocity,norm,out=np.tile(np.array([[1.,0.]],dtype=raw.dtype),(len(raw),1)),where=norm>1e-6)
        lateral=np.stack([-direction[:,1],direction[:,0]],-1)
        references=[('current',model)]
        baseline,_=bf.make_model('cuda');baseline.load_state_dict(torch.load(BASE/'B0_seed42/final.pt',map_location='cpu',weights_only=False)['ema'],strict=True);baseline.eval()
        references.append(('B0',baseline));sensitivity={}
        for label,net in references:
            captures={}
            def context_hook(module,args,value):captures['context']=value[:,0].detach()
            handle=net.decoder.context.register_forward_hook(context_hook)
            dh=None
            if hasattr(net.decoder,'update'):
                def delta_hook(module,args,value):captures['delta']=value.detach()
                dh=net.decoder.update.register_forward_hook(delta_hook)
            records=[]
            for sign in (0,-1,1):
                x=raw.copy();x[:,0,-1]+=sign*.05*lateral
                pred,score,_,_=bf.prediction(net,torch.tensor(x,device='cuda'),torch.tensor(n,device='cuda'))
                context=captures['context']+captures.get('delta',0)
                records.append((context.cpu().numpy(),pred[:,0,:,[3,7,11]].cpu().numpy()))
            handle.remove()
            if dh:dh.remove()
            c0,a0=records[0];responses=[]
            for sign,(c,a) in zip((-1,1),records[1:]):
                d0=a0[...,2,:]-a0[...,0,:];d=a[...,2,:]-a[...,0,:]
                valid=(np.linalg.norm(d,axis=-1)>1e-8)&(np.linalg.norm(d0,axis=-1)>1e-8)
                angle=np.arctan2(d0[...,0]*d[...,1]-d0[...,1]*d[...,0],(d0*d).sum(-1))
                responses.append(dict(sign=sign,context_norm_per_m=(np.linalg.norm(c-c0,axis=-1)/.05).tolist(),
                    anchor_heading_change_rad=[[float(value) if ok else None for value,ok in zip(row,good)] for row,good in zip(angle,valid)],valid_heading=valid.tolist()))
            sensitivity[label]=responses
        del baseline
        write_json(out/'PERTURBATION.json',dict(ids=dev['ids'][idx].tolist(),rule=rule['perturbation'],responses=sensitivity))
    else:
        idx=hash_indices(dev['ids'],rule['dev_hash'],128);sums=[];weights=[];updates=[];gates=[]
        for start in range(0,len(idx),64):
            take=idx[start:start+64];bf.prediction(model,torch.tensor(dev['past'][take],device='cuda'),torch.tensor(dev['num_valid'][take],device='cuda'))
            d=model.decoder.interaction.diagnostics
            w=d['attention_weights'].reshape(len(take),9,4,20,20)[:,0].cpu().numpy()
            weights.append(w);sums.extend(w.sum(-1).reshape(-1).tolist());updates.extend(d['update_norm'].reshape(len(take),9,20)[:,0].cpu().numpy().reshape(-1).tolist());gates.append(float(d['gate_mean']))
        w=np.concatenate(weights);np.savez_compressed(out/'attention_fixed128.npz',ids=dev['ids'][idx],weights=w)
        diagnostics=dict(n=128,negative_fraction=float((w<0).mean()),positive_fraction=float((w>0).mean()),
            row_sum_min=float(np.min(sums)),row_sum_max=float(np.max(sums)),gate_mean_including_context_agents=float(np.mean(gates)),
            target_update_norm_mean=float(np.mean(updates)),signed_attention_not_probability=True)
    exact(before,cpu_tree(model.state_dict()),'MECHANISM_NO_WEIGHT_BUFFER_MUTATION')
    write_json(out/'DIAGNOSIS.json',dict(arm=arm,descriptive_only=True,diagnostics=diagnostics,model_weights_before_after_exact=True))
    write_json(out/'done.json',dict(status='SUCCESS',files=seal(out.iterdir())))
