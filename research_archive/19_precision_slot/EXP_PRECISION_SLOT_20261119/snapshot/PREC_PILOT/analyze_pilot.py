"""Evaluate paired fixed-budget controls and learned prior arms."""
from pathlib import Path
import sys
import numpy as np
from run_pilot import ROOT, DIAG, OLD, CONTROLS, NEW, ARM_PARENT, load, control_folder, read, replace_json, save_npz
from prior_decoder import CV_SLOT
sys.path.insert(0,str(DIAG))
from run_prec_diag import quality, maneuver_labels, groups, fit_reference, pooled, motion_arrays, motion_summary
from run_prec_diag import motion_differences, window_metrics, summarize


def analyze(data, runs):
    labels = maneuver_labels(data['past'],data['future'])
    masks = groups(data,labels)
    y,past = data['future'].astype(float),data['past'][:,0].astype(float)
    truth = {g:pooled(motion_arrays(y[m,None],past[m])) for g,m in masks.items() if m.any()}
    fit = fit_reference(OLD/'C0/MOTION_REFERENCE.json')
    tail = read(OLD/'C0/EVALUATION_CONTRACT.json')['tail_thresholds']
    cv = past[:,-1,None]+np.arange(1,13)[None,:,None]*(past[:,-1]-past[:,-2])[:,None]
    rows = np.arange(len(cv))
    report = {}
    windows = {}
    for arm in CONTROLS+NEW:
        folder = control_folder(arm) if arm in CONTROLS else ROOT/'runs'/arm
        z = load(folder/'dev_predictions.npz')
        np.testing.assert_array_equal(z['ids'],data['ids'])
        np.testing.assert_array_equal(z['group'],data['group'])
        paths,logits = z['paths'].astype(float),z['logits']
        top = logits.argmax(1)
        cases = [(arm,paths,np.full(len(cv),CV_SLOT) if arm in NEW else None)]
        if arm in CONTROLS:
            target = np.argsort(-logits,axis=1,kind='stable')[:,-1]
            edited = paths.copy()
            edited[rows,target] = cv
            cases.append((arm+'_offline_CV',edited,target))
        for key,path,slot in cases:
            q = quality(path,y)
            v = window_metrics(q,top)
            if slot is not None:
                v.update(prior_is_top1=top==slot,
                         prior_is_minADE=q['ADE'].argmin(1)==slot,
                         prior_is_minFDE=q['FDE'].argmin(1)==slot,
                         prior_good=q['good'][rows,slot],
                         prior_only_good=q['good'][rows,slot]&(q['good'].sum(1)==1))
            report[key] = dict(groups=summarize(v,masks),
                               motion=motion_summary(path,past,q,top,masks,fit,tail,truth))
            windows[key]=v
            save_npz(ROOT/'windows'/f'{key}.npz',ids=data['ids'],group=data['group'],maneuver=labels,**v)
    compare_pairs = [('E3_self','B0'),('Prior','B0'),('Prior_self','E3_self'),('Prior_self','Prior'),('Prior_self','B0'),
                     ('Prior','B0_offline_CV'),('Prior_self','E3_self_offline_CV'),
                     ('B0_offline_CV','B0'),('E3_self_offline_CV','E3_self')]
    comparisons = {}
    for arm,ref in compare_pairs:
        panel = {}
        for group,new in report[arm]['groups'].items():
            base = report[ref]['groups'][group]
            m = masks[group]
            covered,prior_covered = windows[arm]['covered'][m],windows[ref]['covered'][m]
            gains = {k:(base[k]-new[k])/base[k] if base[k]>0 else None for k in ('minADE','minFDE','top1_ADE','top1_FDE')}
            panel[group] = dict(relative_gains=gains,coverage_delta_pp=100*(new['covered']-base['covered']),
                                rescued=int((covered&~prior_covered).sum()),lost=int((prior_covered&~covered).sum()))
        err = []
        for g,k,tolerance in [('overall','top1_FDE',.02),('change','minFDE',.01),('turn_any','minFDE',.01),('decel_stop','minFDE',.01)]:
            delta = panel[g]['relative_gains'][k]
            if delta is None or -delta > tolerance:
                err.append(dict(group=g,metric=k,relative_harm=None if delta is None else -delta,limit=tolerance))
        motion,undefined = motion_differences(report[ref]['motion'],report[arm]['motion'])
        comparisons[arm+'_vs_'+ref] = dict(groups=panel,error_failures=err,motion_failures=motion,
                                            undefined_motion=undefined,protection_pass=not(err or motion or undefined))
    paired = comparisons['Prior_self_vs_E3_self']['groups']['overall']['relative_gains']
    gains_pass = all(paired[k]>=.015 for k in ('minADE','minFDE'))
    guarded = ['Prior_vs_B0','Prior_self_vs_E3_self','Prior_self_vs_B0']
    protection_pass = all(comparisons[k]['protection_pass'] for k in guarded)
    outcome = dict(screening_gain_pass=gains_pass,protection_pass=protection_pass,
                   recommend_more_seeds=gains_pass and protection_pass,compared_for_protection=guarded,
                   single_seed=True)
    result = dict(study='EXP_PREC_PILOT_20261119',status='COMPLETE',seed=42,epochs=32,K=20,
                  fit_windows=21719,dev_windows=len(y),reused_controls=list(CONTROLS),new_runs=list(NEW),
                  CV_slot_zero_based=CV_SLOT,models=report,comparisons=comparisons,outcome=outcome,
                  train={k:v['train'] for k,v in runs.items()})
    replace_json(ROOT/'RESULT.json',result)
    lines=['# PREC-PILOT', '',
           '32 轮，seed42，C0 开发评估。下表的误差单位为米。', '',
           '| 方案 | minADE | minFDE | Top1 FDE | 联合覆盖 | 转向 minFDE | 变向 minFDE |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for arm,r in report.items():
        a=r['groups']['overall']
        lines.append(f"| {arm} | {a['minADE']:.6f} | {a['minFDE']:.6f} | {a['top1_FDE']:.6f} | {a['covered']:.2%} | "
                     f"{r['groups']['turn_any']['minFDE']:.6f} | {r['groups']['change']['minFDE']:.6f} |")
    lines+=['','| 比较 | 相对 ADE 改善 | 相对 FDE 改善 | 误差保护触发 | 运动学保护触发 |',
             '|---|---:|---:|---:|---:|']
    for k,r in comparisons.items():
        g=r['groups']['overall']['relative_gains']
        lines.append(f"| {k} | {g['minADE']:.3%} | {g['minFDE']:.3%} | {len(r['error_failures'])} | {len(r['motion_failures'])} |")
    lines+=['',f"组合对 E3_self 的双指标筛选通过 {gains_pass}。保护通过 {protection_pass}。建议扩展 seed {outcome['recommend_more_seeds']}。",
             '', 'CV 槽使用率、覆盖新增与损失及分组触发细节见 RESULT.json。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
