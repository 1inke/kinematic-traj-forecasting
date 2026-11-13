"""Completion supplement: independent CPU saved-output/CI/provenance review, no model forward."""
from pathlib import Path
import sys,json,hashlib,csv
sys.dont_write_bytecode=True
import numpy as np
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[2];OUT=Path(__file__).resolve().parent
OLD=ROOT/'qoder_work/F16_CONTEXT_COMPLEMENTARITY_DESIGN_20261022/execution_20261022'
D0=HERE.parent/'execution_20261023'
sys.path.insert(0,str(ROOT/'src'))
from f16.io import write_json
from f16.resource import Guard
SEEDS=(42,43,44);ARMS=('A_replay','P_relative')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def same(a,b,tol=1e-10):
    a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape
    if a.dtype.kind in 'buiUS' or b.dtype.kind in 'buiUS':assert np.array_equal(a,b);return 0.
    assert np.array_equal(np.isnan(a),np.isnan(b));ok=np.isfinite(a);delta=float(np.max(np.abs(a[ok]-b[ok]))) if ok.any() else 0.;assert delta<=tol,delta;return delta
def quality(paths,y):
    e=np.sqrt(np.sum((paths.astype(float)-y.astype(float)[:,None])**2,axis=-1));a=e.sum(-1)/12;f=e[:,:,-1];j=np.maximum(a/.2,f/.4);q=a+.5*f;row=np.arange(len(e));qs=q.argmin(1);js=j.argmin(1);good=(a<=.2)&(f<=.4)
    return dict(ADE=a,FDE=f,J=j,good=good,coverage=good.any(1),Q_winner=qs,J_winner=js,Q_ADE=a[row,qs],Q_FDE=f[row,qs],Q_segments=e[row,qs].reshape(-1,3,4).mean(-1),J_segments=e[row,js].reshape(-1,3,4).mean(-1),Jmin=j.min(1),minADE=a.min(1),minFDE=f.min(1))
def masks(groups):return [('overall',np.ones(len(groups),bool)),('stable',groups==0),('transition',groups==1),('change',groups==2)]
def csvout(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def source_verify():
    mf=read(HERE/'source_manifest.json');ext=read(HERE/'source_extension_prelaunch.json');assert ext['base_source_digest']==mf['digest']
    for key,v in mf['files'].items():assert sha(v['path'])==v['sha256'],key
    for key,value in ext['extension_files'].items():assert sha(ROOT/key)==value,key
    return mf,ext

def main():
    guard=Guard(4*2**30);guard.check('startup');mf,ext=source_verify()
    assert read(HERE/'worker_state.json')['status']==read(HERE/'exit_status.json')['status']=='SUCCESS'
    assert read(HERE/'RESULT.json')['status']=='PASS_EXECUTION_T1_RELATIVE_REGRESSION'
    for name in ('ARITHMETIC_AUDIT.json','PRESERVATION.json','RESOURCE.json'):assert read(HERE/name)['status']=='PASS'
    raw={};records=[];motions=[];subsets=[];checkpoints=[];outputs={};maxerr=0.;total_epochs=steps=0
    for arm in ARMS:
        for seed in SEEDS:
            run=HERE/f'runs/{arm}_seed{seed}';done=read(run/'done.json');con=done['contract'];assert con['arm']==arm and con['seed']==seed and con['source_digest']==mf['digest']
            for rel,value in done['outputs'].items():assert sha(HERE/rel)==value,rel;outputs[rel]=value
            for key,fp in [('statistics',HERE/'statistics.npz'),('IDs',HERE/'split_ids.npz'),('initial',HERE/f'initial_states/seed{seed}.pt'),('orders',HERE/f'batch_orders/seed{seed}.npz'),('native_J',HERE/'native_J_FIT.npz'),('protocol',HERE/'protocol.json')]:assert con[key]==sha(fp)
            history=read(run/'history.json');assert len(history)==8 and [h['epoch'] for h in history]==list(range(1,9))
            assert sum(h['steps'] for h in history)==592 and all(h['n']==18909 for h in history);total_epochs+=len(history);steps+=592
            for h in history:
                for key in ('loss','base','task','div','relative_excess','aux'):
                    derived=sum(b[key]*b['n'] for b in h['batches'])/18909;maxerr=max(maxerr,same(derived,h[key]))
                assert sum(h['WTA'])==sum(h['J_responsibility'])==18909
                assert abs(h['loss']-h['base']-h['aux'])<=1e-7
            if arm=='A_replay':
                rep=read(run/'CONTROL_REPRODUCTION.json');assert rep['history_common_fields'] and rep['paths_bitwise']
                oldhist=read(OLD/f'runs/A_seed{seed}/history.json')
                for a,b in zip(history,oldhist):
                    for key in ('loss','base','task','div','preclip_grad_norm','n','steps','grad_clip_rate','WTA','WTA_extra_ties','diagnostic'):assert a[key]==b[key],key
            for split,n in [('FIT',18909),('inner',1856)]:
                guard.check();audit=read(run/f'audit_{split}.json');da=read(run/f'diagnostic_audit_{split}.json');assert audit['status']==da['status']=='PASS'
                assert audit['independent_implementation'] and da['independent_code'];maxerr=max(maxerr,da['max_abs_difference'])
                with np.load(run/f'predictions_{split}.npz') as z:d={k:z[k].copy() for k in z.files}
                assert d['paths'].shape==(n,20,12,2) and np.isfinite(d['paths']).all() and len(np.unique(d['sample_id']))==n
                with np.load(HERE/'split_ids.npz') as z:same(d['sample_id'],z[split])
                if arm=='A_replay':
                    with np.load(OLD/f'runs/A_seed{seed}/predictions_{split}.npz') as z:assert np.array_equal(d['paths'],z['paths'])
                e=quality(d['paths'],d['gt']);ne=quality(d['native'],d['gt'])
                with np.load(run/f'sample_metrics_{split}.npz') as z:
                    for key in z.files:maxerr=max(maxerr,same(e[key],z[key]))
                if split=='inner':
                    raw[(arm,seed)]={k:e[k] for k in ('coverage','minADE','minFDE','Jmin')}
                    if seed==42 and arm=='A_replay':inner_ids=d['sample_id'];inner_tracks=d['track'];inner_groups=d['groups'];native={k:ne[k] for k in raw[(arm,seed)]}
                    else:same(d['sample_id'],inner_ids);same(d['track'],inner_tracks);same(d['groups'],inner_groups)
                metric=read(run/f'metrics_{split}.json');pop=read(run/f'population_loss_{split}.json');summary=read(run/f'diagnostics_{split}.json')
                smap={g['group']:g for g in summary['groups']}
                with np.load(D0/f'D0_D1/seed{seed}_{split}/per_window.npz') as z:
                    same(z['sample_id'],d['sample_id']);frozen_labels=z['partition'].copy();old_quality={k:z['student_'+k].copy() for k in ('minADE','minFDE','Jmin')}
                for group,mask in masks(d['groups']):
                    m=metric[group];gm=ne['coverage'];good=e['coverage'];rescue=~gm&good&mask;lost=gm&~good&mask;nwin=int(mask.sum())
                    same(m['good'],int(good[mask].sum()));same(m['native_good'],int(gm[mask].sum()));same(m['rescue'],int(rescue.sum()));same(m['lost'],int(lost.sum()))
                    same(np.array(m['rescue_IDs']),d['sample_id'][rescue]);same(np.array(m['lost_IDs']),d['sample_id'][lost])
                    r=float(np.maximum(e['Jmin'][mask]-ne['Jmin'][mask],0).mean());maxerr=max(maxerr,same(pop['groups'][group]['R'],r))
                    row=dict(arm=arm,seed=seed,split=split,group=group,n=nwin,tracks=m['tracks'],native_good=m['native_good'],good=m['good'],rescue=m['rescue'],lost=m['lost'],lost_rate_native_good=m['lost_rate_native_good'],ADE=m['ADE'],FDE=m['FDE'],J=m['J'],R=r,distinct=m['distinct'],effective_good=m['effective_good'],motion_pass=m['motion_pass'])
                    for key in ('ADE','FDE','J'):maxerr=max(maxerr,same(row[key],float(e[{'ADE':'minADE','FDE':'minFDE','J':'Jmin'}[key]][mask].mean())))
                    records.append(row)
                    for key,val in m['motion'].items():motions.append(dict(arm=arm,seed=seed,split=split,group=group,metric=key,value=val,native=m['native_motion'][key],relative_harm_pct=(val/m['native_motion'][key]-1)*100,pass_literal=val<=1.01*m['native_motion'][key]))
                    for label,name in enumerate(('retained','rescue','lost','neither')):
                        take=mask&(frozen_labels==label);den=int(take.sum());entry=dict(arm=arm,seed=seed,split=split,group=group,frozen_old_A_partition=name,n=den,tracks=len(np.unique(d['track'][take])),good=int(good[take].sum()))
                        for key in ('minADE','minFDE','Jmin'):
                            delta=e[key]-old_quality[key];entry[key+'_delta_vs_oldA_mean']=float(delta[take].mean()) if den else None;entry[key+'_contribution_vs_oldA']=float(delta[take].sum()/nwin)
                        subsets.append(entry)
                del d,e,ne
    assert total_epochs==48 and steps==3552
    # Checkpoint metadata only: no model instance, forward, optimizer or CUDA replay.
    import torch
    for arm in ARMS:
        for seed in SEEDS:
            run=HERE/f'runs/{arm}_seed{seed}';blob=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False);initial=torch.load(HERE/f'initial_states/seed{seed}.pt',map_location='cpu',weights_only=True)
            assert blob['epoch']==8 and blob['step']==592 and blob['history']==read(run/'history.json')
            for key in ('mean','std'):assert torch.equal(blob['model'][key],initial[key])
            assert sum(v.numel() for k,v in blob['model'].items() if k not in ('mean','std'))==82008
            assert all(torch.isfinite(v).all() for v in blob['model'].values())
            assert set(blob['rng'])=={'python','numpy','torch_cpu','torch_cuda','loader'}
            for state in blob['optimizer']['state'].values():assert float(state['step'])==592 and all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v))
            checkpoints.append(dict(arm=arm,seed=seed,checkpoint_sha256=sha(run/'checkpoint.pt'),epoch=8,steps=592,all_stats_buffers_exact=True,optimizer_param_groups=blob['optimizer']['param_groups']))
            del blob,initial
    guard.check()
    boot=read(HERE/'BOOTSTRAP.json')
    with np.load(HERE/'BOOTSTRAP_DRAWS.npz') as z:td=z['track_draws'];sd=z['seed_draws'];stored=z['paired_values'];fixed=z['fixed_seed_values'];track_pop=z['track_population']
    unique,inv=np.unique(inner_tracks,return_inverse=True);same(unique,track_pop);assert td.shape==(20000,len(unique)) and sd.shape==(20000,3)
    rng=np.random.default_rng(20261023)
    for i in range(20000):assert np.array_equal(td[i],rng.integers(len(unique),size=len(unique))) and np.array_equal(sd[i],rng.integers(3,size=3))
    defs=[('primary','change','R','P-A')]+[(f'{comp}:{group}:{key}',group,key,comp) for comp in ('P-A','P-native') for group in ('overall','change') for key in ('coverage','minADE','minFDE','reference_success_lost_rate')]
    cirows=[];cierror=0.
    for j,(name,group,key,comp) in enumerate(defs):
        expected=boot['results'][j];assert (expected['name'],expected['group'],expected['metric'],expected['contrast'])==(name,group,key,comp)
        mask=np.ones(len(inv),bool) if group=='overall' else inner_groups==2;nums=[];dens=[]
        for seed in SEEDS:
            pp=raw[('P_relative',seed)];ref=raw[('A_replay',seed)] if comp=='P-A' else native
            if key=='R':v=np.maximum(pp['Jmin']-native['Jmin'],0)-np.maximum(ref['Jmin']-native['Jmin'],0);dm=mask
            elif key=='reference_success_lost_rate':v=(ref['coverage']&~pp['coverage']).astype(float);dm=mask&ref['coverage']
            else:v=pp[key].astype(float)-ref[key].astype(float);dm=mask
            nums.append([float(v[dm&(inv==t)].sum()) for t in range(len(unique))]);dens.append([int((dm&(inv==t)).sum()) for t in range(len(unique))])
        nums=np.array(nums);dens=np.array(dens);nn=nums[:,td].sum(-1).T;dd=dens[:,td].sum(-1).T;rat=np.divide(nn,dd,out=np.full(nn.shape,np.nan),where=dd>0)
        fv=rat.mean(1);bv=np.array([rat[k,sd[k]].mean() for k in range(20000)]);cierror=max(cierror,same(bv,stored[:,j]),same(fv,fixed[:,j]));point=float((nums.sum(1)/dens.sum(1)).mean());cierror=max(cierror,same(point,expected['point']))
        row=dict(name=name,group=group,metric=key,contrast=comp,point=point,tracks=len(np.unique(inner_tracks[mask])))
        for tag,val in [('paired_track_seed',bv),('fixed_seed_track_only',fv)]:
            valid=np.isfinite(val);same(int(valid.sum()),expected[tag]['valid']);same(int((~valid).sum()),expected[tag]['invalid'])
            lower,upper=np.quantile(val[valid],[.025,.975]);cierror=max(cierror,same([lower,upper],expected[tag]['CI95']));row[tag+'_CI95_lower']=float(lower);row[tag+'_CI95_upper']=float(upper)
            if j:
                low,up=np.quantile(val[valid],[.0015625,.9984375]);cierror=max(cierror,same([low,up],expected[tag]['CI_family16']));row[tag+'_family16_lower']=float(low);row[tag+'_family16_upper']=float(up)
        cirows.append(row)
    assert len(cirows)==17
    mf2,ext2=source_verify();assert mf2==mf and ext2==ext and all(sha(HERE/k)==v for k,v in outputs.items())
    guard.check();csvout('ALL_GROUP_RESULTS.csv',records);csvout('ALL_MOTION.csv',motions);csvout('FROZEN_OLD_A_SUBSETS.csv',subsets)
    # Uniform CSV schema keeps all16 corrected results plus separate primary.
    csvout('SECONDARY16_CI.csv',cirows[1:]);write_json(OUT/'PRIMARY_CI.json',cirows[0]);write_json(OUT/'CHECKPOINTS.json',checkpoints)
    review=dict(status='PASS_SAVED_COMPLETION_SUPPLEMENT',fresh_runs=6,scientific_population_jobs=12,epoch_rows=48,optimizer_steps=3552,source_count=len(mf['files']),extension_source_count=len(ext['extension_files']),source_digest=mf['digest'],done_outputs_SHA_count=len(outputs),source_and_outputs_before_after_unchanged=True,max_saved_quality_summary_error=maxerr,all_20k_primary_secondary16_and_fixedseed_review_maxerror=cierror,all_20k_track_seed_draw_order_exact=True,all_checkpoint_stats_buffers_exact=True,all_A_paths_bitwise_original_and_common_history_exact=True,registered_full_cluster_motion_and_full_geometry_audit_reports_all_PASS=True,no_full_cluster_or_full_geometry_third_replay=True,no_model_instance_forward_backward_optimizer=True,checkpoint_CPU_metadata_only=True,all_groups=records,motion_rows=motions,frozen_oldA_subsets=subsets,CI=cirows,resource_estimate_completion_CPU_bytes=4*2**30,completion_CPU_resource_observations=guard.observations)
    write_json(OUT/'REVIEW.json',review);print(json.dumps({k:v for k,v in review.items() if k not in ('all_groups','motion_rows','frozen_oldA_subsets','CI','completion_CPU_resource_observations')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
