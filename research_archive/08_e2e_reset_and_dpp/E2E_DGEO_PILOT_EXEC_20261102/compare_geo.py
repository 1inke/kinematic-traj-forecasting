"""Saved-data paired analysis. Single seed, no confirmatory p/CI claims."""
import csv
import time
from geo_common import *
import numpy as np

GROUPS=('overall','stable','transition','change')
TAILS=('speed','acceleration','jerk','junction_acceleration','junction_jerk')


def summarize_pair(a,b,ga,gb,pa,pb,groupmask):
    ma=np.asarray(a['coverage'],bool);mb=np.asarray(b['coverage'],bool)
    rescue=groupmask&ma&~mb;lost=groupmask&~ma&mb
    ia=np.arange(len(ma));ta=pa['logits'].argmax(1);tb=pb['logits'].argmax(1)
    ca=a['good'][ia,ta];cb=b['good'][ia,tb]
    delta=a['J'].min(1)-b['J'].min(1)
    frequencies=np.bincount(a['J_winner'][groupmask],minlength=20);prob=frequencies/frequencies.sum()
    return dict(n=int(groupmask.sum()),reference_good=int((mb&groupmask).sum()),
        rescue=int(rescue.sum()),lost=int(lost.sum()),lost_rate_among_reference_good=float(lost.sum()/(mb&groupmask).sum()) if (mb&groupmask).sum() else None,
        rescue_ids=pa['ids'][rescue].tolist(),lost_ids=pa['ids'][lost].tolist(),
        lost_J_damage=delta[lost].tolist(),lost_reference_Jmargin=(1-b['J'].min(1)[lost]).tolist(),
        coverage_delta=ga['coverage']-gb['coverage'],minADE_delta=ga['minADE']-gb['minADE'],
        minFDE_delta=ga['minFDE']-gb['minFDE'],minJ_delta=ga['minJ']-gb['minJ'],
        method_score_top1_good=int((ca&groupmask).sum()),reference_score_top1_good=int((cb&groupmask).sum()),
        method_conditional_top1_recall=float((ca&groupmask).sum()/(ma&groupmask).sum()) if (ma&groupmask).sum() else None,
        reference_conditional_top1_recall=float((cb&groupmask).sum()/(mb&groupmask).sum()) if (mb&groupmask).sum() else None,
        method_distinct=ga['distinct'],method_effective_good=ga['effective_good'],
        J_oracle_physical_slot_counts=frequencies.tolist(),J_oracle_slot_entropy=float(-(prob[prob>0]*np.log(prob[prob>0])).sum()),
        physical_slot_not_semantic_intent=True)


def pilot_screen(groups,reference):
    o,c=groups['overall'],groups['change'];bo,bc=reference['overall'],reference['change']
    ade=(o['minADE']<bo['minADE'] or c['minADE']<bc['minADE']) and o['minADE']<=1.01*bo['minADE'] and c['minADE']<=1.01*bc['minADE']
    fde=all(groups[g]['minFDE']<=1.01*reference[g]['minFDE'] for g in ('overall','stable','transition'))
    risk=[]
    for g in GROUPS:
        for k in TAILS:
            a=groups[g]['tail_risk']['allK20'][k];b=reference[g]['tail_risk']['allK20'][k]
            estimable=b['positive_excess_p95']>0 and reference[g]['target_tracks']>=20
            risk.append(dict(group=g,metric=k,rate_delta_pp=100*(a['exceedance_rate']-b['exceedance_rate']),
                positive_excess_p95=a['positive_excess_p95'],reference_positive_excess_p95=b['positive_excess_p95'],
                rate_pass=a['exceedance_rate']<=b['exceedance_rate']+.01,
                magnitude_pass=a['positive_excess_p95']<=1.01*b['positive_excess_p95'] if estimable else None,
                status='ESTIMABLE' if estimable else 'NOT_ESTIMABLE_NO_FLOOR'))
    complete=all(r['magnitude_pass'] is not None for r in risk)
    tail=complete and all(r['rate_pass'] and r['magnitude_pass'] for r in risk)
    return dict(continuous_ADE_pass=bool(ade),FDE_noninferiority_pass=bool(fde),tail_noninferiority_pass=bool(tail),
        full_pilot_screen=bool(ade and fde and tail),tail_comparisons=risk,
        zero_reference_policy='No floor/epsilon, no declaration of relative noninferiority from 0/0',
        single_seed_significance='NOT_ASSESSED',not_major_or_baseline_promotion=True)


def main():
    t=time.perf_counter();out=RUN/'completion_review';out.mkdir(exist_ok=False)
    source=read_json(RUN/'source_manifest.json');verify(source['files'])
    import torch
    all_rows=[];comparisons={};screens={};assets=[];root_references={};paired_controls={}
    for arm in ARMS:
        run=RUN/'runs'/arm;done=read_json(run/'done.json');verify(done['files'])
        if done['updates']!=2720 or done['epochs']!=8:raise ValueError('INCOMPLETE_RUN')
        lock=read_json(run/'FINAL_LOCK.json');verify(lock['files'])
        with (run/'history.csv').open(encoding='utf-8-sig') as f:hist=list(csv.DictReader(f))
        if len(hist)!=2720:raise ValueError('HISTORY_COUNT')
        orders=read_json(run/'batch_orders.json')['batch_IDs']
        if orders!=read_json(BASE/'B0_seed42/batch_orders.json')['batch_IDs']:raise ValueError('UNPAIRED_ORDER')
        state=torch.load(run/'final.pt',map_location='cpu',weights_only=False)
        if state['contract']['source_manifest']!=sha(RUN/'source_manifest.json') or state['updates']!=2720:raise ValueError('FINAL_SOURCE_OR_UPDATE')
        if any(not torch.isfinite(p).all() for p in state['model'].values() if p.is_floating_point()):raise ValueError('NONFINITE_FINAL')
        del state
        comparisons[arm]={}
        for split in ('fit','dev'):
            folder=RUN/'populations'/arm/split;old=BASE/'populations'/split
            receipt=read_json(folder/'done.json');verify(receipt['files']);verify(read_json(old/'done.json')['files'])
            audit=read_json(folder/'audit.json')
            if audit['status']!='PASS_ALL_WINDOWS_INDEPENDENT_QUALITY_PDIST_FULLK20_MOTION':raise ValueError('AUDIT_INCOMPLETE')
            a=dict(np.load(folder/'window_metrics.npz'));b=dict(np.load(old/'window_metrics.npz'))
            pa=dict(np.load(folder/'predictions.npz'));pb=dict(np.load(old/'predictions.npz'))
            if not np.array_equal(pa['ids'],pb['ids']) or not np.array_equal(pa['future'],pb['future']):raise ValueError('NOT_MATCHED_ID_GT')
            groups=read_json(folder/'metrics.json');reference=read_json(old/'metrics.json');comparisons[arm][split]={}
            for j,g in enumerate(GROUPS):
                mask=np.ones(len(pa['ids']),bool) if j==0 else pa['groups']==j-1
                comparisons[arm][split][g]=summarize_pair(a,b,groups[g],reference[g],pa,pb,mask)
                r=groups[g];all_rows.append(dict(arm=arm,split=split,group=g,n=r['n'],good=r['good'],coverage=r['coverage'],minADE=r['minADE'],minFDE=r['minFDE'],minJ=r['minJ'],distinct=r['distinct'],effective_good=r['effective_good'],**r['motion']))
            if split=='dev':screens[arm]=pilot_screen(groups,reference)
            assets.extend(folder.iterdir())
        verify(read_json(RUN/'mechanisms'/arm/'done.json')['files']);assets.extend((RUN/'mechanisms'/arm).iterdir())
        assets.extend(p for p in run.iterdir() if p.is_file())
    for method,control in [('DG-GeoGate','DG-Base'),('DG-GeoDiff','DG-GeoGate'),('DG-GeoScore','DG-GeoGate')]:
        ga=read_json(RUN/'populations'/method/'dev/metrics.json');gb=read_json(RUN/'populations'/control/'dev/metrics.json')
        pa=dict(np.load(RUN/'populations'/method/'dev/predictions.npz'));pb=dict(np.load(RUN/'populations'/control/'dev/predictions.npz'))
        a=dict(np.load(RUN/'populations'/method/'dev/window_metrics.npz'));b=dict(np.load(RUN/'populations'/control/'dev/window_metrics.npz'))
        paired_controls[method+'-minus-'+control]={}
        for j,g in enumerate(GROUPS):
            mask=np.ones(len(pa['ids']),bool) if j==0 else pa['groups']==j-1
            paired_controls[method+'-minus-'+control][g]=summarize_pair(a,b,ga[g],gb[g],pa,pb,mask)
    write_csv(RUN/'ALL_GROUPS.csv',all_rows)
    write_json(RUN/'PAIRED_COMPARISON_VS_B0.json',comparisons);write_json(RUN/'PAIRED_MECHANISM_CONTROLS.json',paired_controls)
    write_json(RUN/'PILOT_SCREEN.json',screens)
    write_json(RUN/'DGEO_MECHANISM_DIAGNOSIS_CORE.json',{arm:read_json(RUN/'mechanisms'/arm/'DIAGNOSIS.json') for arm in ARMS})
    write_json(RUN/'DGEO_INITIAL_MECHANISM_DIAGNOSIS.json',{arm:read_json(RUN/'initial_mechanisms'/arm/'DIAGNOSIS.json') for arm in ARMS})
    decision=dict(status='FOUR_PAIRED_PILOTS_COMPLETE_REVIEWED',models=4,epochs=32,updates=10880,seed=42,
        screen=screens,conditional_seed_review={a:'ELIGIBLE_OWN_NEXT_STAGE_ADMISSION' if screens[a]['full_pilot_screen'] else 'NO_AUTOMATIC_PROMOTION_REVIEW_COUNTEREVIDENCE' for a in ('DG-GeoGate','DG-GeoDiff','DG-GeoScore')},
        no_V_F_or_seed43_44_started=True,no_significance_claim=True,baseline_promoted=False,
        concern='Single seed, overlapping targets/windows, same development source and repeated dev use. Same nominal parameter scale is not equal effective capacity. Conditional mechanisms are descriptive, not causal proof.',
        DA2_same_assistant_not_external='Improvements can be optimizer/extra-capacity effects; compare fixed adjacent DG-Base/GeoGate controls. No both/U+D training. Actual target-only FLOPs and shared parameter counts must be measured, not inferred from discarded neighbor outputs.',
        DA3_same_assistant_not_external='Execution PASS does not imply pilot, old-major, safety or generalization PASS. Failed/zero-denominator gates retained. Later stages require their own prelaunch receipts; no subjective threshold search.')
    write_json(RUN/'DECISION.json',decision);write_json(RUN/'RESULT_ANALYSIS.json',decision)
    lines=['# DGeo authorized four paired pilots','', 'Completed four fresh seed42 8-epoch pilots: 32 epochs / 10880 scientific updates. Fixed final EMA, no dev checkpoint selection.', '', '|arm|dev group|N|good|coverage|minADE|minFDE|','|---|---|---:|---:|---:|---:|---:|']
    for r in all_rows:
        if r['split']=='dev':lines.append(f"|{r['arm']}|{r['group']}|{r['n']}|{r['good']}|{r['coverage']:.6f}|{r['minADE']:.6f}|{r['minFDE']:.6f}|")
    lines+=['','## ARS DA2 / DA3 (same assistant, not external peer review)','',decision['DA2_same_assistant_not_external'],decision['DA3_same_assistant_not_external'],decision['concern'],
        '', 'All paired rescue/lost IDs, reference-success denominators, J margin damage, top1 conditional recall, all five tail panels and signed attention/geometric mode responses are saved. Positive attention is not semantic intent. Fixed-hash128 initial/final geometric diagnostics were sealed before scientific training.',
        '','All four DGeo measured parameter counts are <=305000 without removing the B0 backbone. Exact per-arm costs are sealed. COST_ACTUAL_PREDICTOR measures whole raw-observation pipelines, not only new layers; operator counts exclude scalar nonlinear/NumPy FLOPs but include their latency.',
        '', 'Saved dual arithmetic and original complete-link/pdist clustering/pooled motion were performed per full population. This completion verifies receipts, every output SHA, checkpoint metadata and paired orders; it does not replay all model gradients. Single seed yields no confirmatory significance claim.',
        '', 'The current worker ends after these four pilots. Conditional multi-seed or combined implementation is not silently launched. All failures, old models and results retained; baseline/historycycle11/long-term goal unchanged.']
    with (RUN/'RESULT_ANALYSIS.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    verify(source['files'])
    roots=[RUN/n for n in ('ALL_GROUPS.csv','PAIRED_COMPARISON_VS_B0.json','PAIRED_MECHANISM_CONTROLS.json','PILOT_SCREEN.json','DGEO_MECHANISM_DIAGNOSIS_CORE.json','DGEO_INITIAL_MECHANISM_DIAGNOSIS.json','DECISION.json','RESULT_ANALYSIS.json','RESULT_ANALYSIS.md','COST_ACTUAL_PREDICTOR.json')]
    write_json(out/'REVIEW.json',dict(status='PASS_ALL4_SOURCE_OUTPUT_SHA_CHECKPOINT_ORDER_AND_SAVED_QA_RECEIPTS',
        at=now(),elapsed_seconds=time.perf_counter()-t,checkpoint_metadata_loads=4,
        no_model_forward_or_gradient_replay=True,source_unchanged=True,files=seal(assets+roots)))
    write_json(RUN/'RESULT.json',dict(status='SUCCESS_AUTHORIZED_ALL4_DGEO_PILOTS_REVIEWED',at=now(),models=4,epochs=32,updates=10880,
        review=sha(out/'REVIEW.json'),scientific_success='SEE_PILOT_SCREEN_NOT_EXECUTION_STATUS'))
    write_json(RUN/'PRESERVATION.json',dict(status='PASS',sources=source['files'],outputs=seal(assets+roots+[out/'REVIEW.json',RUN/'RESULT.json'])))
    print('ALL4_COMPLETION_REVIEW_PASS',flush=True)


if __name__=='__main__':main()
