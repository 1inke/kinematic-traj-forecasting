"""CPU source/checkpoint/accounting review, no second model or gradient replay."""
import csv
import time
from common import *
import numpy as np


def main():
    started=time.perf_counter()
    out=HERE/'completion_review';out.mkdir(exist_ok=False)
    manifest=read_json(HERE/'source_manifest.json');verify(manifest['files'])
    lock=read_json(HERE/'B0_FINAL_LOCK.json');verify(lock['files']);verify(lock['epoch_checkpoints'])
    if lock['science_updates']!=2720:raise ValueError('BAD_UPDATE_ACCOUNTING')
    run=HERE/'B0_seed42'
    with (run/'history.csv').open(encoding='utf-8-sig') as f:history=list(csv.DictReader(f))
    if len(history)!=2720 or int(history[-1]['epoch'])!=8:raise ValueError('HISTORY_INCOMPLETE')
    orders=read_json(run/'batch_orders.json')['batch_IDs']
    fit=dict(np.load(HERE/'C0/fit.npz'));dev=dict(np.load(HERE/'C0/dev.npz'))
    for epoch in range(8):
        identities=[s for batch in orders[epoch*340:(epoch+1)*340] for s in batch]
        if sorted(identities)!=sorted(fit['ids'].tolist()):raise ValueError('ORDER_NOT_ONE_COMPLETE_FIT_EPOCH')
    populations={};all_outputs=[]
    for name,data in [('fit',fit),('dev',dev)]:
        folder=HERE/'populations'/name
        done=read_json(folder/'done.json');verify(done['files'])
        audit=read_json(folder/'audit.json')
        if audit['status']!='PASS_ALL_WINDOWS_INDEPENDENT_QUALITY_PDIST_FULLK20_MOTION' or audit['windows']!=len(data['ids']):
            raise ValueError('FULL_SAVED_AUDIT_INCOMPLETE')
        with np.load(folder/'predictions.npz') as saved:
            if not np.array_equal(saved['ids'],data['ids']) or not np.array_equal(saved['future'],data['future']):
                raise ValueError('OUTPUT_ID_OR_GT_CONTRACT')
        populations[name]=read_json(folder/'metrics.json')
        all_outputs.extend(folder.iterdir())
    # CPU metadata load is explicit; not a model instantiation or optimizer replay.
    import torch
    checkpoint=torch.load(run/'final.pt',map_location='cpu',weights_only=False)
    if checkpoint['updates']!=2720 or checkpoint['epoch']!=8 or checkpoint['cursor']!=len(fit['ids']):
        raise ValueError('FINAL_STATE_INCOMPLETE')
    if any(not torch.isfinite(p).all() for p in checkpoint['model'].values() if p.is_floating_point()):
        raise ValueError('NONFINITE_FINAL_WEIGHTS')
    if checkpoint['contract']['source_manifest']!=sha(HERE/'source_manifest.json'):raise ValueError('FINAL_SOURCE_INCOMPATIBLE')
    epochs=read_json(run/'EPOCH_SUMMARY.json')
    decision={'status':'COMPLETED_B0_DEVELOPMENT_PILOT_NOT_MECHANISM_OR_MAJOR_SUCCESS',
        'C0':'PASS_WITHIN_RECORDING_SOURCE_TIME_TARGET_AUDIT_SISTER_FOOTAGE_UNVERIFIED',
        'fresh_seed42':True,'epochs':8,'updates':2720,'parameters':295474,
        'selection':'FINAL_EMA_EPOCH8_NO_DEV_CHECKPOINT_SELECTION',
        'train_loss_first_epoch':epochs[0]['mean_train_loss'],'train_loss_last_epoch':epochs[-1]['mean_train_loss'],
        'continuous_and_motion_metrics':populations,
        'legacy_shadow':'NOT_COMPARABLE_TO_OLD_NATIVE_ON_NEW_RAW_SPLIT; old Q FAIL unchanged',
        'inference':'Single-seed pipeline and learning trend only, not convergence or significance.',
        'next':'Review B0 dev continuous errors, coverage, empirical tails and actual cost before a separately registered U or D pilot. No automatic U+D/V/60 models.',
        'baseline_promoted':False,'historycycle11_unchanged':True,'long_term_goal_not_claimed_complete':True}
    write_json(HERE/'DECISION.json',decision)
    write_json(HERE/'RESULT_ANALYSIS.json',decision)
    # Generated evidence-based summary; companion inline reviews are same-assistant roles.
    text=['# C0 + B0 fresh end-to-end pilot', '',
        'Completed one fresh seed42 model, 8 fixed epochs / 2720 updates. Final epoch8 EMA, no dev selection.', '',
        f'Fit {len(fit["ids"])} windows; dev {len(dev["ids"])} windows. Same recording time and target disjoint. Sister recording footage independence unverified; not a new blind benchmark.', '',
        f'Train loss epoch1 {epochs[0]["mean_train_loss"]:.6f} -> epoch8 {epochs[-1]["mean_train_loss"]:.6f}. Pilot learning trend is not proof of full convergence.', '',
        '|split|group|N|good|coverage|minADE|minFDE|', '|---|---|---:|---:|---:|---:|---:|']
    for split,groups in populations.items():
        for name,r in groups.items():
            text.append(f'|{split}|{name}|{r["n"]}|{r.get("good",0)}|{r.get("coverage",0):.6f}|{r.get("minADE",0):.6f}|{r.get("minFDE",0):.6f}|')
    text += ['', 'All K20 complete-link clusters, same-slot ADE/FDE/J, five pooled motion, junction, allK/top1/oracle tails and worst cases are saved per population. Old native outputs have no matched new-source contract and are NOT_COMPARABLE. CV is not relabelled as native.', '',
        '## DA2 — counterevidence (same assistant, not external peer review)', '',
        '1. Single seed and correlated overlapping windows cannot support a significance or generalization claim. Sister-footage aliases remain physically unverified even though exact raw coordinate/time and offset-independent full-track signatures found no duplicates.',
        '2. Global p99.5 from realized GT is an empirical reference, not a physiological safety limit; all K20 alternatives need not match a unique realized future distribution. Zero-reference noninferiority remains non-estimable.',
        '3. Fresh random whole-model exposure differs from historical Q in source, neighbor eligibility, decoder and optimization. A better B0 would not isolate freezing as the sole cause.', '',
        '## DA3 — final disposition (same assistant)', '',
        'The pipeline and saved numerical audit can pass while prediction quality or motion risk is poor. No old major/retention gate is retrospectively changed; no baseline promotion. B0 is the prospective matched comparator, not a verified solution.',
        'C0 uses official TRAIN source only; official ETH test was inventoried by SHA, not parsed or evaluated. GPU arithmetic, all-RNG resume and saved independent audits are separately documented; this CPU completion does not replay all model gradients.',
        'No automatic U/D/V or formal 60-model validation is launched by this worker. All failure/source/output files retained.']
    with (HERE/'RESULT_ANALYSIS.md').open('x',encoding='utf-8') as f:f.write('\n'.join(text)+'\n')
    write_json(out/'REVIEW.json',{'status':'PASS_CPU_COMPLETE_OUTPUT_SHA_SOURCE_ACCOUNTING_CHECKPOINT_METADATA',
        'at':now(),'elapsed_seconds':time.perf_counter()-started,'full_saved_arithmetic':'Already independently recomputed once per population in save pipeline; receipt and every file SHA verified here, not a second whole-save bound.',
        'model_forward_backward':0,'metadata_checkpoints_loaded':1,'same_assistant_not_external_review':True,
        'final_source_unchanged':True,'root_and_outputs':seal(all_outputs+[HERE/'SCIENCE_DONE.json',HERE/'SCIENCE_RESOURCE.json',HERE/'B0_FINAL_LOCK.json',HERE/'DECISION.json',HERE/'RESULT_ANALYSIS.json',HERE/'COST_ACTUAL_PREDICTOR.json'])})
    verify(manifest['files'])
    write_json(HERE/'RESULT.json',{'status':'SUCCESS_C0_B0_COMPLETE_REVIEWED_PILOT',
        'at':now(),'completed_fresh_models':1,'epochs':8,'updates':2720,
        'scientific_major_success':'NOT_CLAIMED','all_old_results_unchanged':True,
        'no_U_D_V_formal_models':True,'review':sha(out/'REVIEW.json')})
    write_json(HERE/'PRESERVATION.json',{'status':'PASS','sources':manifest['files'],
        'outputs':seal([p for p in HERE.rglob('*') if p.is_file() and 'runtime_cache' not in str(p) and p.suffix not in ('.log','.new') and p.name not in ('WORKER_STATE.json','runtime_control.json')])})
    print('COMPLETION_REVIEW_PASS',time.perf_counter()-started,flush=True)


if __name__=='__main__':main()
