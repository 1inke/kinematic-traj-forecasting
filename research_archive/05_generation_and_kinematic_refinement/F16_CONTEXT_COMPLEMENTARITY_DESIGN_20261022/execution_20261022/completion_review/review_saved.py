"""CPU final supplement: contracts, saved arithmetic and descriptive tables only.

No model class, forward, optimizer step, gradient or second full cluster replay.
The worker's eight full independent cluster/motion audits are SHA-validated.
"""
from pathlib import Path
import sys,hashlib,json,csv,time
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from f16.io import sha256,read_json,write_json
from f16.resource import Guard

def check_source():
    m=read_json(HERE/'source_manifest.json')
    for key,v in m['files'].items():assert sha256(v['path'])==v['sha256'],('SOURCE',key)
    ext=read_json(HERE/'source_extension_prelaunch.json');assert ext['base_source_digest']==m['digest']
    for key,v in ext['extension_files'].items():assert sha256(ROOT/key)==v
    return m,ext

def floatsame(a,b):
    if isinstance(a,dict):return set(a)==set(b) and all(floatsame(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(floatsame(x,y) for x,y in zip(a,b))
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    return a==b

def csvout(name,rows):
    with (HERE/'completion_review'/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    start=time.perf_counter();guard=Guard(6*2**30);guard.check('startup');manifest,extension=check_source()
    # Record every existing scientific output before reading, not transient monitor/control files.
    paths=[]
    for s in [42,43,44]:paths += [p for p in (HERE/f'runs/A_seed{s}').iterdir() if p.is_file()]
    paths += [HERE/f'native_{split}.npz' for split in ['FIT','inner']]
    paths += [HERE/f'native_metrics_{split}.json' for split in ['FIT','inner']]+[HERE/f'native_audit_{split}.json' for split in ['FIT','inner']]
    paths += [HERE/n for n in ['RESULT.json','DECISION.json','RESULT_ANALYSIS.md','RESULT_ANALYSIS.json','ARITHMETIC_AUDIT.json','PRESERVATION.json','RESOURCE.json','source_manifest.json','source_extension_prelaunch.json']]
    before={p.relative_to(HERE).as_posix():sha256(p) for p in paths}
    native={split:read_json(HERE/f'native_metrics_{split}.json') for split in ['FIT','inner']};ids=np.load(HERE/'split_ids.npz');stats=np.load(HERE/'statistics.npz');decision=read_json(HERE/'DECISION.json')
    import torch
    # CPU deserialize own complete checkpoints; do not instantiate ContextK20 or optimizer.
    rows=[];motions=[];subsets=[];assignments=[];history_rows=[];contracts=[];reviews=[];largest=0.;gates=[]
    for split in ['FIT','inner']:
        a=read_json(HERE/f'native_audit_{split}.json');assert a['status']=='PASS' and a['frozen_labels_verified']
        for key,digest in a['inputs_sha'].items():assert sha256(ROOT/key)==digest
        reviews.append(dict(job='native_'+split,audit_status=a['status'],maxdiff=a['max_abs_diff']))
    for seed in [42,43,44]:
        run=HERE/f'runs/A_seed{seed}';done=read_json(run/'done.json');assert done['status']=='SUCCESS'
        for key,digest in done['outputs'].items():assert sha256(HERE/key)==digest,('RUN_OUTPUT',key)
        c=done['contract'];assert c['arm']=='A' and c['seed']==seed and c['source_digest']==manifest['digest']
        for key,path in [('protocol','protocol.json'),('stats','statistics.npz'),('ids','split_ids.npz'),('initial',f'initial_states/seed{seed}.pt'),('orders',f'batch_orders/seed{seed}.npz'),('teacher',f'teachers/seed{seed}.npz')]:assert c[key]==sha256(HERE/path)
        blob=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False)
        hist=read_json(run/'history.json');ah=read_json(run/'assignment_history.json')
        assert blob['contract']==c and blob['epoch']==8 and blob['step']==592 and len(hist)==8 and len(ah)==8 and floatsame(blob['history'],hist) and floatsame(blob['assignment'],ah)
        assert [r['epoch'] for r in hist]==list(range(1,9)) and sum(r['steps'] for r in hist)==592
        assert all(r['n']==18909 and r['steps']==74 and sum(v['n'] for v in r['batches'])==18909 for r in hist)
        assert set(blob['rng'])=={'python','numpy','torch_cpu','torch_cuda','loader'} and len(blob['rng']['torch_cuda'])>0
        assert torch.equal(blob['model']['mean'],torch.from_numpy(stats['mean'])) and torch.equal(blob['model']['std'],torch.from_numpy(stats['std']))
        assert sum(v.numel() for k,v in blob['model'].items() if k not in ['mean','std'])==82008
        assert all(torch.isfinite(v).all() for v in blob['model'].values())
        assert all(int(v['step'])==592 for v in blob['optimizer']['state'].values())
        contracts.append(dict(seed=seed,epochs=8,steps=592,parameters=82008,checkpoint_sha256=sha256(run/'checkpoint.pt'),all_contract_components_SHA=True,CPU_only_metadata_state_read=True))
        for r in hist:history_rows.append(dict(seed=seed,epoch=r['epoch'],loss=r['loss'],base=r['base'],task=r['task'],div=r['div'],clip_rate=r['grad_clip_rate'],preclip_grad_norm=r['preclip_grad_norm'],WTA_nonzero_slots=int((np.array(r['WTA'])>0).sum()),extra_WTA_ties=r['WTA_extra_ties'],diagnostic_good=r['diagnostic']['overall']['good'],diagnostic_change_good=r['diagnostic']['change']['good'],diagnostic_n=r['diagnostic']['overall']['n']))
        del blob
        for split in ['FIT','inner']:
            guard.check();metrics=read_json(run/f'metrics_{split}.json');audit=read_json(run/f'audit_{split}.json');aa=read_json(run/f'assignment_{split}.json')
            assert audit['status']=='PASS' and audit['frozen_labels_verified']
            for key,digest in audit['inputs_sha'].items():assert sha256(ROOT/key)==digest
            reviews.append(dict(job=f'A_{seed}_{split}',audit_status=audit['status'],maxdiff=audit['max_abs_diff']))
            with np.load(run/f'predictions_{split}.npz') as z,np.load(run/f'sample_metrics_{split}.npz') as saved:
                p=z['paths'].astype(float);y=z['gt'];base=z['native'];group=z['groups'];sid=z['sample_id'];track=z['track'];assert np.array_equal(sid,ids[split])
                dis=np.sqrt(((p-y[:,None])**2).sum(-1));a=dis.sum(2)/12;f=dis[:,:,-1];j=np.maximum(a/.2,f/.4);good=(a<=.2)&(f<=.4);cover=good.any(1)
                bd=np.sqrt(((base-y[:,None])**2).sum(-1));ba=bd.sum(2)/12;bf=bd[:,:,-1];bc=((ba<=.2)&(bf<=.4)).any(1)
                for key,value in [('ADE',a),('FDE',f),('J',j),('good',good),('coverage',cover),('Q_winner',(a+.5*f).argmin(1)),('J_winner',j.argmin(1))]:
                    difference=float(np.max(np.abs(saved[key].astype(float)-value.astype(float))));largest=max(largest,difference);assert difference<=2e-12,(seed,split,key,difference)
                assert np.array_equal(np.bincount((a+.5*f).argmin(1),minlength=20),aa['WTA'])
                for gi,g in enumerate(['overall','stable','transition','change']):
                    take=np.ones(len(p),bool) if gi==0 else group==gi-1;m=metrics[g];b=native[split][g];n=int(take.sum());rescue=int((cover&~bc&take).sum());lost=int((~cover&bc&take).sum())
                    assert m['n']==n and m['good']==int(cover[take].sum()) and m['rescue']==rescue and m['lost']==lost and m['native_good']==int(bc[take].sum())
                    assert m['rescue_IDs']==sid[cover&~bc&take].tolist() and m['lost_IDs']==sid[~cover&bc&take].tolist()
                    checks=dict(nonempty=n>0,coverage=m['coverage']>=.99*b['coverage'],ADE=m['ADE']<=1.01*b['ADE'],FDE=m['FDE']<=1.01*b['FDE']);checks.update({key:m['motion'][key]<=1.01*b['motion'][key] for key in m['motion']})
                    registered=next(r for r in decision['rows'] if r['seed']==seed and r['split']==split and r['group']==g);assert registered['checks']==checks and registered['passed']==all(checks.values())
                    gates.append(dict(seed=seed,split=split,group=g,checks=checks,passed=all(checks.values())))
                    rows.append(dict(seed=seed,split=split,group=g,n=n,tracks=m['tracks'],native_good=m['native_good'],A_good=m['good'],native_coverage=b['coverage'],A_coverage=m['coverage'],coverage_delta_pp=100*(m['coverage']-b['coverage']),rescue=rescue,lost=lost,lost_rate_native_success=m['lost_rate_native_good'],retained=m['retained'],native_ADE=b['ADE'],A_ADE=m['ADE'],ADE_harm_pct=100*(m['ADE']/b['ADE']-1),native_FDE=b['FDE'],A_FDE=m['FDE'],FDE_harm_pct=100*(m['FDE']/b['FDE']-1),native_J=b['J'],A_J=m['J'],distinct=m['distinct'],effective_good=m['effective_good'],all_motion=m['motion_pass'],failed_checks=','.join(k for k,v in checks.items() if not v)))
                    for key,v in m['motion'].items():motions.append(dict(seed=seed,split=split,group=g,metric=key,n=n,native=b['motion'][key],student=v,harm_pct=100*(v/b['motion'][key]-1) if b['motion'][key] else None,literal_pass=v<=1.01*b['motion'][key]))
                for key,v in aa['subsets'].items():subsets.append(dict(seed=seed,split=split,subset=key,**v))
                assignments.append(dict(seed=seed,split=split,WTA=aa['WTA'],WTA_nonzero=int((np.array(aa['WTA'])>0).sum()),extra_ties=aa['WTA_extra_ties'],teachers=aa['teachers']))
                assert sum(v['n'] for v in aa['subsets'].values())==len(p)
            del p,dis,base
    assert len(rows)==24 and len(motions)==120 and len(reviews)==8
    assert not any(r['passed'] for r in gates) and decision['status']=='BASELINE_NOT_READY' and not decision['all_pass']
    check_source();after={p.relative_to(HERE).as_posix():sha256(p) for p in paths};assert before==after
    guard.check();csvout('ALL_GROUP_RESULTS.csv',rows);csvout('ALL_MOTION.csv',motions);csvout('SUBSETS.csv',subsets);csvout('EPOCH_DIAGNOSTICS.csv',history_rows)
    write_json(HERE/'completion_review/ASSIGNMENTS.json',assignments)
    summary={}
    for split in ['FIT','inner']:
        summary[split]={}
        for group in ['overall','stable','transition','change']:
            rr=[r for r in rows if r['split']==split and r['group']==group];summary[split][group]=dict(n=rr[0]['n'],native_good=rr[0]['native_good'],A_good=[r['A_good'] for r in rr],coverage_gain_pp=[r['coverage_delta_pp'] for r in rr],mean_coverage_gain_pp=float(np.mean([r['coverage_delta_pp'] for r in rr])),ADE_harm_pct=[r['ADE_harm_pct'] for r in rr],FDE_harm_pct=[r['FDE_harm_pct'] for r in rr],rescue=[r['rescue'] for r in rr],lost=[r['lost'] for r in rr],lost_rate=[r['lost_rate_native_success'] for r in rr])
    report=dict(status='PASS_SAVED_COMPLETION_SUPPLEMENT',base_sources=420,prelaunch_extension_sources=1,old_and_production_artifact_SHA_before=before,old_and_production_artifact_SHA_after=after,all_unchanged=True,checkpoints=contracts,saved_full_independent_audit_reports_validated=reviews,full_audits_not_repeated=True,saved_slot_counts_maxdiff=largest,all_gate_checks_recomputed=gates,all_S1_groups_failed=True,S2='SKIPPED_BASELINE_GATE',source_digest=manifest['digest'],completion_helper_sha=sha256(__file__),no_model_class_or_forward_backward_optimizer_steps=True,no_new_data_or_population=True,summary=summary,all_motion_pass_groups=0,all_motion_total_groups=24,accel_mean_failed_groups=sum(1 for r in motions if r['metric']=='accel_mean' and not r['literal_pass']),description='S1 is descriptive deterministic gate; S2 primary and secondary CI not estimated because B/E were not trained',elapsed_seconds=time.perf_counter()-start,resource=guard.observations)
    write_json(HERE/'completion_review/REVIEW.json',report)
    print(json.dumps(dict(status=report['status'],saved_arithmetic_maxdiff=largest,summary=summary,contracts=contracts,all_group_motion='0/24PASS',acceleration_mean_FAIL='24/24'),ensure_ascii=False,indent=2))

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback;write_json(HERE/'completion_review/FAILURE.json',dict(traceback=traceback.format_exc()));raise
