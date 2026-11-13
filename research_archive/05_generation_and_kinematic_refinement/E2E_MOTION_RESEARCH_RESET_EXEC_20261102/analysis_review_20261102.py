"""Read-only saved-result interpretation supplement; no model/optimizer imports.

Original science sources and reports are not edited. New summaries are descriptive
only and do not introduce a new success gate or choose a different checkpoint.
"""
import csv
import json
from datetime import datetime
import numpy as np
from common import HERE, sha, seal, verify, read_json, write_json, write_csv, now
from causal_dataset import motion


def main():
    out=HERE/'analysis_review_20261102'
    out.mkdir(exist_ok=False)
    manifest=read_json(HERE/'source_manifest.json')
    preservation=read_json(HERE/'PRESERVATION.json')
    review=read_json(HERE/'completion_review/REVIEW.json')
    lock=read_json(HERE/'B0_FINAL_LOCK.json')
    files={}
    for group in (manifest['files'],preservation['sources'],preservation['outputs'],
                  review['root_and_outputs'],lock['files'],lock['epoch_checkpoints']):
        for p,s in group.items():
            if p in files and files[p]!=s:raise ValueError('MANIFEST_CONTRADICTION '+p)
            files[p]=s
    verify(files)
    result=read_json(HERE/'RESULT.json');exit=read_json(HERE/'WORKER_EXIT.json')
    if exit['status']!='SUCCESS' or result['status']!='SUCCESS_C0_B0_COMPLETE_REVIEWED_PILOT':raise ValueError('NOT_COMPLETE')
    if result['review']!=sha(HERE/'completion_review/REVIEW.json'):raise ValueError('REVIEW_DIGEST_MISMATCH')
    rootfiles=['RESULT.json','WORKER_EXIT.json','WORKER_STATE.json','PRESERVATION.json',
               'completion_review/REVIEW.json','SCIENCE_DONE.json','SCIENCE_RESOURCE.json',
               'COST_ACTUAL_PREDICTOR.json','C0/EVALUATION_CONTRACT.json','C0/MOTION_REFERENCE.json']
    root_manifest=seal([HERE/p for p in rootfiles])
    reference=read_json(HERE/'C0/EVALUATION_CONTRACT.json')['tail_thresholds']
    allrows=[];recordings=[];diagnostics={};examples=[]
    for split in ('fit','dev'):
        folder=HERE/'populations'/split
        verify(read_json(folder/'done.json')['files'])
        metrics=read_json(folder/'metrics.json')
        with np.load(folder/'predictions.npz') as saved:
            p=saved['paths'];past=saved['past'];gt=saved['future'];ids=saved['ids'];g=saved['groups'];logits=saved['logits']
        with np.load(folder/'window_metrics.npz') as w:
            ade=w['ADE'];fde=w['FDE'];j=w['J'];good=w['good'];covered=w['coverage'];jwin=w['J_winner']
        n=len(ids);row=np.arange(n);top=logits.argmax(1)
        topgood=good[row,top]
        if not np.isfinite(p).all() or len(np.unique(ids))!=n:raise ValueError('BAD_SAVED_VALUES_IDS')
        # No repeated full cluster review: the verified original independent audit covers it.
        gt_motion=motion(past,gt)
        predicted=motion(np.broadcast_to(past[:,None],(n,20,8,2)),p)
        family=np.array([i.split('|')[0] for i in ids])
        group_masks=[('overall',np.ones(n,bool))]+[(name,g==k) for k,name in enumerate(('stable','transition','change'))]
        for name,mask in group_masks:
            r=metrics[name]
            if int(mask.sum())!=r['n'] or int(covered[mask].sum())!=r['good']:raise ValueError('COUNT_MISMATCH')
            ntop=int(topgood[mask].sum());ncover=int(covered[mask].sum())
            allrows.append(dict(split=split,group=name,n=int(mask.sum()),tracks=r['target_tracks'],
                good=ncover,coverage=r['coverage'],minADE=r['minADE'],minFDE=r['minFDE'],
                score_top1_good=ntop,score_top1_coverage=float(topgood[mask].mean()),
                score_top1_ADE=r['score_top1_ADE'],score_top1_FDE=r['score_top1_FDE'],
                oracle_good_but_top1_missed=int((covered&~topgood&mask).sum()),
                fraction_covered_retrieved_by_top1=ntop/ncover if ncover else None,
                distinct=r['distinct'],effective_good=r['effective_good']))
        for name in np.unique(family):
            for group,mask in group_masks:
                m=mask&(family==name)
                if not m.any():continue
                recordings.append(dict(split=split,recording=name,group=group,n=int(m.sum()),
                    good=int(covered[m].sum()),coverage=float(covered[m].mean()),
                    minADE=float(ade[m].min(1).mean()),minFDE=float(fde[m].min(1).mean()),
                    score_top1_ADE=float(ade[row,top][m].mean()),score_top1_FDE=float(fde[row,top][m].mean())))
        def panel(motion_data,mask):
            return {key:dict(denominator=int(value[mask].size),mean=float(value[mask].mean()),
                p95=float(np.quantile(value[mask],.95)),p99_5=float(np.quantile(value[mask],.995)),
                rate_above_fit_p99_5=float((value[mask]>reference[key]).mean())) for key,value in motion_data.items()}
        diagnostics[split]={'groups':{name:{'raw_GT':panel(gt_motion,mask),
            'allK20_predicted':panel(predicted,mask),
            'score_top1_predicted':panel({k:v[row,top] for k,v in predicted.items()},mask),
            'J_oracle_predicted':panel({k:v[row,jwin] for k,v in predicted.items()},mask)}
            for name,mask in group_masks},
            'acceleration_per_future_difference':[float(predicted['acceleration'][...,i].mean()) for i in range(11)],
            'jerk_per_future_difference':[float(predicted['jerk'][...,i].mean()) for i in range(10)],
            'score_selected_slot_counts':np.bincount(top,minlength=20).tolist(),
            'GT_J_winner_slot_counts':np.bincount(jwin,minlength=20).tolist()}
        # Exhaustive ranking disagreement counts above, a fixed complete top50 worst table below.
        worst=np.argsort(-j.min(1),kind='stable')[:50]
        for i in worst:
            examples.append(dict(split=split,id=str(ids[i]),group=int(g[i]),
                minJ=float(j[i].min()),minADE=float(ade[i].min()),minFDE=float(fde[i].min()),
                score_slot=int(top[i]),J_slot=int(jwin[i]),score_slot_J=float(j[i,top[i]]),
                score_slot_ADE=float(ade[i,top[i]]),score_slot_FDE=float(fde[i,top[i]])))
        del p,predicted
    epochs=read_json(HERE/'B0_seed42/EPOCH_SUMMARY.json')
    loss_drop=1-epochs[-1]['mean_train_loss']/epochs[0]['mean_train_loss']
    resource=read_json(HERE/'SCIENCE_RESOURCE.json')
    worker_seconds=(datetime.fromisoformat(exit['finished_at'])-datetime.fromisoformat(exit['started_at'])).total_seconds()
    summary=dict(status='ANALYZED_C0_B0_PILOT_COMPLETE_NO_MECHANISM_CLAIM',at=now(),
        source_SHA_checked=len(manifest['files']),all_saved_SHA_checked=len(files),
        worker_seconds=worker_seconds,science_seconds=read_json(HERE/'SCIENCE_DONE.json')['elapsed_seconds'],
        original_planning_seconds=read_json(HERE/'PRELAUNCH_REVIEW.json')['planning_seconds'],
        epoch_train_loss=[e['mean_train_loss'] for e in epochs],loss_drop_fraction=loss_drop,
        complete_results=allrows,raw_GT_and_predictions=diagnostics,
        production_own_process_memory=resource['snapshot'],
        retained_observation_warnings=sorted({w for x in resource['observations'] for w in x['warnings']}),
        stderr=[{'name':p.name,'bytes':p.stat().st_size} for p in (HERE/'logs').glob('*stderr.log')],
        interpretation=[
            'One fresh model trained and saved audited successfully, not demonstrated to beat any matched baseline.',
            'Lower training objective demonstrates optimization, not full convergence or generalization.',
            'Score-top1/oracle gap is descriptive ranking head limitation, not proof a newly trained scorer would solve errors. No new scorer authorized.',
            'Linear anchor intervals explain much low future mean acceleration; junction discontinuity and tail rates must be reported separately.',
            'Fit/dev population mixture and raw-GT difficulty differ; dev>fit coverage is not itself evidence of leakage or superior generalization.',
            'No matched old-native predictions on new source IDs, no retention/major comparison or baseline promotion; B0 self-loss0 is tautology.',
            'One seed, overlap windows and unresolved sister-footage identity preclude significance, new blind-scene or causal freezing-root claims.'],
        no_new_training=True,no_model_forward_backward=True,no_full_clusters_rerun=True,
        no_PID_task_queries=True,no_delete=True,same_assistant_analysis_not_external_review=True)
    write_csv(out/'ALL_GROUPS.csv',allrows)
    write_csv(out/'RECORDING_GROUPS.csv',recordings)
    write_csv(out/'WORST_CASES_WITH_SCORE.csv',examples)
    write_json(out/'ANALYSIS.json',summary)
    verify(files);verify(root_manifest)
    write_json(out/'REVIEW.json',{'status':'PASS_CURRENT_SOURCE_OUTPUT_SHA_AND_DESCRIPTIVE_SAVED_SUPPLEMENT',
        'files_checked':len(files),'original_sources':len(manifest['files']),
        'original_reports_sources_unchanged':True,'additional_source':seal([__file__]),
        'original_roots':root_manifest,'supplement':seal([out/n for n in ('ANALYSIS.json','ALL_GROUPS.csv','RECORDING_GROUPS.csv','WORST_CASES_WITH_SCORE.csv')]),
        'at':now(),'no_new_training_or_model_replay':True,'same_assistant_not_external_review':True})
    print('SHA_VERIFIED',len(files),'WORKER_SECONDS',worker_seconds,'TRAIN_LOSS_DROP',loss_drop,flush=True)
    for r in allrows:
        if r['split']=='dev':print('DEV',r,flush=True)
    for s in ('fit','dev'):
        x=diagnostics[s]['groups']['overall']
        print('MOTION',s,'GT',x['raw_GT'],'PRED',x['allK20_predicted'],flush=True)
    print('ACCEL_PER_STEP',diagnostics['dev']['acceleration_per_future_difference'],flush=True)


if __name__=='__main__':main()
