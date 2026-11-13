"""CPU-only second implementation of final subsets/assignments/losses.

Full independent geometry/physics/clusters audit is already done per job by
audit/verify.py. This entry verifies those reports/source SHAs and separately
recomputes every saved per-window scalar and extra teacher/assignment diagnostic.
No torch, model, optimizer or replay of the old training process.
"""
from common import *
import time
from audit.verify import compare

def independent_extras(z):
    p=z['paths'].astype(float);native=z['native'].astype(float);gt=z['gt'].astype(float);groups=z['groups'];kin=z['teacher_kin'].astype(float);n=len(p)
    def calc(a):
        distance=np.sqrt(np.sum((a-gt[:,None])**2,axis=-1));ade=np.sum(distance,axis=-1)/12;fde=distance[:,:,-1];j=np.maximum(ade/.2,fde/.4);good=(ade<=.2)&(fde<=.4);q=ade+.5*fde;return distance,ade,fde,j,good,q
    e,a,f,j,g,q=calc(p);_,na,nf,nj,ng,nq=calc(native);kc=kin.shape[1]>0;kg=calc(kin)[4] if kc else np.empty((n,0),bool);native_cov=ng.any(1);covered=g.any(1);qw=q.argmin(1);jw=j.argmin(1);ii=np.arange(n)
    per=dict(ADE=a,FDE=f,J=j,good=g,covered=covered,native_covered=native_cov,Q_winner=qw,J_winner=jw,Q_segments=e[ii,qw].reshape(n,3,4).mean(-1),J_segments=e[ii,jw].reshape(n,3,4).mean(-1),teacher_KIN_good=kg)
    assignment={}
    for name,mask in [('overall',np.ones(n,bool)),('stable',groups==0),('transition',groups==1),('change',groups==2)]:
        success=mask&covered;assignment[name]=dict(N=int(mask.sum()),WTA_Q_winner_counts=np.bincount(qw[mask],minlength=20).tolist(),WTA_exact_tie_windows=int(np.sum(np.sum(q==q.min(1)[:,None],axis=1)[mask]>1)),any_good_candidate_counts=g[mask].sum(0).tolist(),first_good_physical_slot_distribution=np.bincount(g.argmax(1)[success],minlength=20).tolist(),success_J_closest_slot_distribution=np.bincount(jw[success],minlength=20).tolist())
    totals={key:dict(count=0,sum=0.,assign=np.zeros(20,np.int64),ties=0) for key in ['R','N','B']}
    if kc:
        for start in range(0,n,64):
            stop=min(n,start+64);teachers=np.concatenate([native[start:stop],kin[start:stop]],1).astype(np.float32).astype(float);delta=teachers[:,:,None]-p[start:stop,None];norm=np.sqrt(np.sum(delta*delta,axis=-1));cost=np.sum(norm,axis=-1)/12+.5*norm[:,:,:,-1];minimum=cost.min(-1);win=cost.argmin(-1);empty=np.zeros((stop-start,20),bool);masks={'R':np.concatenate([ng[start:stop],empty],1)&native_cov[start:stop,None],'N':np.concatenate([empty,kg[start:stop]],1)&~native_cov[start:stop,None],'B':np.concatenate([ng[start:stop],kg[start:stop]],1)}
            for key,mask in masks.items():
                total=totals[key];total['count']+=int(mask.sum());total['sum']+=float(minimum[mask].sum());total['assign']+=np.bincount(win[mask],minlength=20);total['ties']+=int(np.sum((np.sum(cost==minimum[:,:,None],axis=-1)[mask])>1))
    teacher=dict(available=kc)
    if kc:
        teacher.update(teacher_cache_is_GT_posthoc_label_only=True,R_native_success_teacher_count=totals['R']['count'],N_native_failed_KIN_good_teacher_count=totals['N']['count'],B_union_good_teacher_count=totals['B']['count'],B_extra_KIN_in_native_success_count=int((kg&native_cov[:,None]).sum()))
        for key in ['R','N','B']:teacher[key+'_nearest_student_counts']=totals[key]['assign'].tolist();teacher[key+'_exact_teacher_ties']=totals[key]['ties']
    else:teacher['reason']='No verified aligned KIN cache; never substitute FIT rows'
    terms={k:v['sum']/v['count'] if v['count'] else 0. for k,v in totals.items()};ep=p[:,:,-1];upper=np.triu_indices(20,1);div=float(np.maximum(.3-np.sqrt(np.sum((ep[:,upper[0]]-ep[:,upper[1]])**2,axis=-1)),0).mean());task=float(q.min(1).mean());base=task+.05*div
    loss=dict(WTA_Q=task,endpoint_linear_diversity=div,base=base,R_population_teacher_mean=terms['R'],N_population_teacher_mean=terms['N'],B_population_teacher_mean=terms['B'],hypothetical_full_population_A=base,hypothetical_full_population_B=base+terms['B'],hypothetical_full_population_C=base+terms['R'],hypothetical_full_population_D=base+terms['N'],hypothetical_full_population_E=base+terms['R']+terms['N'])
    subsets={}
    for label,take in [('native_success',native_cov),('native_failure_with_good_KIN',~native_cov&kg.any(1) if kc else None),('native_failure_without_good_KIN',~native_cov&~kg.any(1) if kc else None)]:
        if take is None:subsets[label]={'available':False};continue
        sub={}
        for name,mask in [('overall',np.ones(n,bool)),('stable',groups==0),('transition',groups==1),('change',groups==2)]:
            mask=mask&take;den=int(mask.sum());num=int(covered[mask].sum());sub[name]=dict(N=den,good=num,coverage=num/den if den else None,lost=int(np.sum(native_cov&~covered&mask)),rescue=int(np.sum(~native_cov&covered&mask)),ADE=float(a[mask].min(1).mean()) if den else None,FDE=float(f[mask].min(1).mean()) if den else None,J=float(j[mask].min(1).mean()) if den else None)
        subsets[label]={'available':True,'groups':sub}
    return dict(assignment=assignment,teacher=teacher,population_loss=loss,subsets=subsets),per

def without_prose(x):
    if isinstance(x,dict):return {k:without_prose(v) for k,v in x.items() if k not in ['warning']}
    return x

def main():
    start=time.monotonic();verify_sources();cfg=read_json(HERE/'protocol.json');guard=Guard(cfg['resource']['estimate_bytes']);guard.check('startup');manifest=read_json(HERE/'frozen_eval/checkpoint_replay_manifest.json')
    partial='--available' in sys.argv
    if not partial and (manifest['status']!='SUCCESS_FROZEN22' or len(manifest['records'])!=22):raise RuntimeError('FROZEN22_NOT_COMPLETE')
    cache_dir=HERE/'completion_review/jobs';cache_dir.mkdir(parents=True,exist_ok=True)
    before={};checks=[];maxdiff=0.;fields=0
    for record in manifest['records']:
        guard.check();key=record['job'];base=HERE/'frozen_eval';done=read_json(base/'done'/f'{key}.json')
        for rel,digest in done['outputs'].items():
            path=HERE/rel
            if sha256(path)!=digest:raise ValueError('BAD_SAVED_OUTPUT '+rel)
            before[rel]=digest
        audit=read_json(base/'audit'/f'{key}.json')
        if audit['status']!='PASS' or audit['script_sha']!=sha256(ROOT/'audit/verify.py') or not audit['frozen_labels_verified']:raise ValueError('INDEPENDENT_METRIC_AUDIT_NOT_CURRENT')
        cache=cache_dir/f'{key}.json'
        if cache.exists():
            previous=read_json(cache)
            if previous['script_sha256']!=sha256(__file__) or previous['outputs']!=done['outputs']:raise RuntimeError('STALE_COMPLETION_REVIEW_CACHE')
            item=previous['check'];checks.append(item);fields+=item['extra_diagnostics']['n_fields'];maxdiff=max(maxdiff,item['extra_diagnostics']['max_abs_diff'],max(c['max_abs_diff'] for c in item['perwindow']));continue
        with np.load(base/'predictions'/f'{key}.npz',allow_pickle=False) as z:
            extras,per=independent_extras(z)
            with np.load(base/'sample_metrics'/f'{key}.npz',allow_pickle=False) as saved:
                if not np.array_equal(saved['sample_id'],z['sample_id']):raise ValueError('SAMPLE_METRIC_ID')
                perchecks=[]
                for name,arr in per.items():
                    value=saved[name];delta=0. if np.array_equal(arr,value) else float(np.max(np.abs(arr.astype(float)-value.astype(float))));pass_=bool(delta<=cfg['audit_float_tolerance'])
                    if not pass_:raise ValueError('PERWINDOW_MISMATCH '+key+'/'+name)
                    perchecks.append(dict(field=name,max_abs_diff=delta,pass_=pass_))
            stored=read_json(base/'metrics'/f'{key}_diagnostics.json');a=compare(without_prose(stored),extras,cfg['audit_float_tolerance'])
            if a['status']!='PASS':raise ValueError('EXTRA_DIAGNOSTICS_MISMATCH '+str((key,a)))
            fields+=a['n_fields'];maxdiff=max(maxdiff,a['max_abs_diff'],max(c['max_abs_diff'] for c in perchecks));item=dict(job=key,status='PASS',extra_diagnostics=a,perwindow=perchecks);checks.append(item);write_json(cache,dict(script_sha256=sha256(__file__),outputs=done['outputs'],check=item))
        print(f'CPU independent extra audit {len(checks)}/22 {key}',flush=True)
    if any(sha256(HERE/p)!=v for p,v in before.items()):raise RuntimeError('SAVED_OUTPUT_CHANGED_DURING_REVIEW')
    verify_sources();guard.check();dest=HERE/'completion_review';dest.mkdir(exist_ok=True);write_json(dest/'REVIEW.json',dict(status='PASS_CURRENT_SAVED_ARITHMETIC',jobs=22,max_abs_diff=maxdiff,extra_scalar_fields=fields,checks=checks,source_inputs_unchanged=True,saved_output_SHA_before_after_equal=True,saved_outputs_sha256=before,script_sha256=sha256(__file__),seconds=time.monotonic()-start,resource=guard.observations,no_torch_or_model_or_optimizer=True,not_external_review=True,not_historical_training_provenance_certification=True))
    if partial:
        # A partial cache is never reported as complete22. Keep its scope explicit.
        value=read_json(dest/'REVIEW.json');value.update(status='PARTIAL_AUDIT_ONLY',jobs=len(checks));write_json(dest/'PARTIAL_REVIEW.json',value)
        # Do not publish a final-looking REVIEW while production is ongoing.
        (dest/'REVIEW.json').unlink()
        print(f'PARTIAL extra review:{len(checks)} jobs checked; no final completion claim.',flush=True);return
    progress('COMPLETED_AND_INDEPENDENTLY_REVIEWED_PENDING_P3_REPORT',done_jobs=22);print(f'REVIEW PASS {time.monotonic()-start:.1f}s maxdiff={maxdiff}',flush=True)

if __name__=='__main__':main()
