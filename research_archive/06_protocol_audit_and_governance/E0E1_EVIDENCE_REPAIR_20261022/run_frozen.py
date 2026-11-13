"""P2 only: frozen22 jobs. Never import or call old run_e1/main/train."""
import time,gc,traceback
from common import *
from diagnostics import analyze

def main():
    from f16.eval import evaluate
    from audit.verify import audit_file
    import torch
    cfg=read_json(HERE/'protocol.json')
    if read_json(HERE/'PREFLIGHT.json')['status']!='PASS_FROZEN_EVALUATION_ONLY':raise RuntimeError('PREFLIGHT_REQUIRED')
    guard=Guard(cfg['resource']['estimate_bytes']);guard.check('startup');verify_sources();started=time.monotonic();data=data_load();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;device=read_json(HERE/'PREFLIGHT.json')['device'];records=[];assignment={};replayrows=[];frozen=HERE/'frozen_eval'
    for sub in ['predictions','sample_metrics','metrics','audit','done']:(frozen/sub).mkdir(parents=True,exist_ok=True)
    contract=sha256(HERE/'source_manifest_before.json');code_contract={p.name:sha256(p) for p in HERE.glob('*.py')};pinned={row['path']:row for row in read_json(HERE/'PREFLIGHT.json')['KIN_complete_cache_row_replay']}
    for seed,arm in RUNS:
        net=load_student(seed,arm,device);state=frozen_state(net)
        for split in ['inner','FIT']:
            guard.check();key=f'{arm}_seed{seed}_{split}';ix=data['split'][split];done=frozen/'done'/f'{key}.json'
            if done.exists():
                item=read_json(done)
                if item['input_contract']!=contract or item['code_contract']!=code_contract:raise ValueError('BAD_SOURCE_SKIP_REJECTED')
                for filename,digest in item['outputs'].items():
                    if sha256(HERE/filename)!=digest:raise ValueError('BAD_OUTPUT_SKIP_REJECTED')
                records.append(item['record']);assignment[key]=read_json(frozen/'metrics'/f'{key}_diagnostics.json');continue
            progress('FROZEN_FORWARD',job=key,done_jobs=len(records));t0=time.monotonic();pred=predict(net,data['past'][ix],device);assert_frozen(net,state);kinfile=kin_path(split,seed);kin=None
            if kinfile.exists():
                evidence=pinned.get(str(kinfile))
                if evidence is None or evidence['file_sha256']!=sha256(kinfile) or evidence['registered_IDs_sha256']!=array_sha(data['ids'][ix]):raise ValueError('KIN_EXPLICIT_MAPPING_REQUIRED')
                kin=np.load(kinfile)
            pathfile=frozen/'predictions'/f'{key}.npz';arrays=dict(paths=pred,native=data['native'][ix],gt=data['gt'][ix],groups=data['groups'][ix],sample_id=data['ids'][ix],track=data['track'][ix],sequence=data['metadata']['sequence'][ix],past_last=data['past'][ix,-1],teacher_kin=kin if kin is not None else np.empty((len(ix),0,12,2),np.float32),teacher_global_registry_row=ix)
            np.savez_compressed(pathfile,**arrays);progress('DIRECT_METRICS_AND_ASSIGNMENT',job=key,done_jobs=len(records));metrics,e=evaluate(pred,data['gt'][ix],data['groups'][ix],data['native'][ix],data['ids'][ix],data['track'][ix]);diag,perwindow=analyze(pred,data['native'][ix],data['gt'][ix],data['groups'][ix],kin);resultfile=frozen/'metrics'/f'{key}.json';write_json(resultfile,metrics);diagfile=frozen/'metrics'/f'{key}_diagnostics.json';write_json(diagfile,diag);samples=frozen/'sample_metrics'/f'{key}.npz';np.savez_compressed(samples,sample_id=data['ids'][ix],**perwindow)
            if split=='inner':
                cmp=reconcile(metrics,old_result(seed)['results'][arm]['eval']);write_json(frozen/'metrics'/f'{key}_replay.json',cmp);replayrows.append(dict(job=key,**cmp))
                if cmp['status']!='RECONCILED':raise RuntimeError('REPLAY_NOT_RECONCILED '+key)
            progress('INDEPENDENT_SAVED_PATH_AUDIT',job=key,done_jobs=len(records));audit=audit_file(pathfile,resultfile,cfg['audit_float_tolerance'],frozen_cache=PATHS['cache'],frozen_labels=PATHS['labels'],frozen_metadata=PATHS['metadata'],frozen_manifest=ROOT/'config/data_manifest.v1.json',root=ROOT);auditpath=frozen/'audit'/f'{key}.json';write_json(auditpath,audit)
            if audit['status']!='PASS':raise RuntimeError('SAVED_ARITHMETIC_AUDIT_FAILED '+key)
            assert_frozen(net,state);guard.check();record=dict(job=key,seed=seed,arm=arm,split=split,N=len(ix),IDs_sha=array_sha(data['ids'][ix]),checkpoint_sha=sha256(checkpoint(seed,arm)),strict_weights_buffers_exact=True,device=device,dtype='float32',seconds=time.monotonic()-t0,metrics=metrics,independent_path_audit=audit['status'],independent_fields=audit['n_fields'],max_abs_diff=audit['max_abs_diff'])
            outputs={p.relative_to(HERE).as_posix():sha256(p) for p in [pathfile,resultfile,diagfile,samples,auditpath]};write_json(done,dict(input_contract=contract,code_contract=code_contract,outputs=outputs,record=record));records.append(record);assignment[key]=diag
            print(f'{len(records)}/22 {key} good={metrics["change"]["good"]}/{metrics["change"]["n"]} motion={metrics["change"]["motion_pass"]} audit={audit["status"]} {record["seconds"]:.1f}s',flush=True)
            write_json(frozen/'checkpoint_replay_manifest.json',dict(status='RUNNING',records=records));del pred,kin,arrays,e,perwindow;gc.collect()
        assert_frozen(net,state);del net,state;gc.collect()
    if len(records)!=22:raise RuntimeError('INCOMPLETE22')
    verify_sources();guard.check();write_json(HERE/'source_manifest_after.json',dict(files=capture(current_sources()),historic_inputs_unchanged=True));rows=[]
    for r in records:
        for group,m in r['metrics'].items():rows.append(dict(job=r['job'],seed=r['seed'],arm=r['arm'],split=r['split'],group=group,**{k:m[k] for k in ['n','good','native_good','rescue','lost','retained','coverage','ADE','FDE','J','distinct','effective_good','motion_pass']}))
    write_csv(frozen/'metrics_by_group.csv',rows);write_json(frozen/'assignment_stats.json',assignment);write_json(frozen/'checkpoint_replay_manifest.json',dict(status='SUCCESS_FROZEN22',records=records,actual_seconds=time.monotonic()-started,input_contract=contract,device=device));write_json(HERE/'REPLAY_RECONCILIATION.json',dict(status='ALL_INNER_RECONCILED',rows=replayrows));write_json(HERE/'RESOURCE.json',dict(policy=cfg['resource'],observations=guard.observations));progress('COMPLETED_FROZEN22_PENDING_COMPLETION_REVIEW',done_jobs=22,no_training=True)
    print(f'FROZEN22_COMPLETE {time.monotonic()-started:.1f}s. No new training; P3 review pending.',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        write_json(HERE/'FAILURE.json',dict(stage='CURRENT_FROZEN_EVALUATION_FAILURE',traceback=traceback.format_exc()));progress('STOP_CURRENT_FAILURE');raise
