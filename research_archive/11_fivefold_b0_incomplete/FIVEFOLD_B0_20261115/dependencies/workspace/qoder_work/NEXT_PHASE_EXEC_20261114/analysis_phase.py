"""Saved-output-only E1 decisions and paired E2--E4 comparisons."""
import itertools
import math
import numpy as np
from common import *
import exchange_null as ex

OLD=ROOT/'qoder_work/E2E_MOTION_RESEARCH_RESET_EXEC_20261102'
ARMS=['E2_natural','E2_clamped','E2_clamped_launch','E3_self','E3_set','E3_set_nobias','T1','T2','T3']
GROUPS=ex.GROUPS
MOTION=['accel_mean','jerk_mean','junction_accel_mean','speed_p95']


def prediction_file(name,split='dev'):
    if name=='OLD42':return OLD/'populations'/split/'predictions.npz'
    return HERE/'runs'/name/split/'predictions.npz'


def population(name,split):
    if name=='OLD42':return read_json(OLD/'populations'/split/'metrics.json')
    return read_json(HERE/'runs'/name/split/'metrics.json')


def qread(name,split,data):
    path=prediction_file(name,split)
    with np.load(path,allow_pickle=False) as p:
        if not np.array_equal(p['ids'],data['ids']):raise ValueError('ID_ORDER_MISMATCH')
        if len(np.unique(p['ids']))!=len(data['ids']):raise ValueError('DUPLICATE_IDS')
        return ex.quality(p['paths'].astype(np.float64),data['future'].astype(np.float64))


def null(names,split):
    data=dict(np.load(HERE/'C0'/f'{split}.npz'))
    qs=ex.QList([qread(n,split,data) for n in names]);qs.group=data['group']
    pairs,summary=ex.null_distribution(qs)
    rows=[population(n,split) for n in names]
    for g in GROUPS:
        summary[g]['signal_threshold_pp']=2*summary[g]['coverage_diff_pp']['sd']/math.sqrt(3)
        summary[g]['tol']=max(.03,2*summary[g]['lost_rate']['sd'])
        summary[g]['effective_good_diff_sd']=float(np.std([b[g]['effective_good']-a[g]['effective_good'] for a,b in itertools.permutations(rows,2)],ddof=1))
    return dict(reference_names=names,summary=summary,pairs=pairs)


def e1a():
    names=['OLD42']+[f'E1A_s{s}' for s in range(43,47)]
    result={split:null(names,split) for split in ('fit','dev')}
    data=dict(np.load(HERE/'C0/dev.npz'));ref=qread('OLD42','dev',data)
    result['decomposition']={}
    for name in ['E1A_order_only','E1A_init_only','E1A_rerun']:
        result['decomposition'][name]=ex.pair_exchange(ref['coverage'],qread(name,'dev',data)['coverage'],data['group'])
    with np.load(prediction_file('OLD42')) as a,np.load(prediction_file('E1A_rerun')) as b:
        result['rerun_bitwise']={k:bool(np.array_equal(a[k],b[k])) for k in ('paths','logits','ids')}
        result['rerun_max_path_error']=float(np.max(np.abs(a['paths']-b['paths'])))
    result['interpretation_limits']=['Null pair differences share runs and are dependent; five seeds give a coarse descriptive scale.',
        'Nonzero exchange alone does not prove the legacy gate mathematically unreachable.',
        'Paired accuracy/DPP arms are different objectives and do not isolate training randomness.',
        'A hardware/library change can also explain failure to reproduce the historical seed42.']
    write_json(HERE/'E1/E1A_NULL.json',result)


def choose_budget():
    rows={}
    for E in (8,16,32,64):
        seeds=[read_json(HERE/'runs'/f'E1B_E{E}_s{s}'/'calib/summary_v2.json') for s in (42,43)]
        rows[str(E)]=dict(overall_minADE=float(np.mean([x['overall']['minADE'] for x in seeds])),
                         change_coverage=float(np.mean([x['change']['coverage'] for x in seeds])))
    comparisons=[];selected=64
    for E in (8,16,32):
        a,b=rows[str(E)],rows[str(2*E)]
        improvement=1-b['overall_minADE']/a['overall_minADE']
        coverage=b['change_coverage']-a['change_coverage']
        qualifies=bool(improvement<.01 and coverage<.01)
        comparisons.append(dict(E=E,minADE_relative_improvement=improvement,change_coverage_improvement_pp=100*coverage,qualifies=qualifies))
        if qualifies and selected==64:selected=E
    result=dict(E_star=selected,calibration_only=True,rows=rows,comparisons=comparisons,
        possibly_not_saturated=selected==64,rule='smallest E with both doubling gains <1% minADE and <1pp change coverage; else64')
    write_json(HERE/'E1/BUDGET_DECISION.json',result);return selected


def seal_contract(E,names):
    nn={split:null(names,split) for split in ('fit','dev')}
    contract=dict(version='E2E_V2',at=now(),E_star=E,null=nn,
        reference_names=names,method_seeds=[42,43,44],signal='3 signs agree, abs(mean coverage delta pp)>2*nullSD/sqrt(3); descriptive screen, not formal significance',
        excess_lost='per group: mean same-seed lost_rate minus B0 ordered-pair mean; harm if >max(.03,2*nullSD_lost)',
        motion='all 4 metrics on J-oracle/same-group GT and allK20/FIT-GT: abs(log ratio) worsening>0.05; any seed/group/port triggers harm',
        motion_metrics=MOTION,tail='any seed/group/port C0 p99.5 exceedance rate increases >.01',
        E2_primary='junction allK abslog distance decreases in all seeds and mean improvement>2*paired-seed SD; no V2 harm and overall minADE<=1.01 B0',
        E3_primary='set-self change coverage signal; set-B0 no harm; if self/set both beat B0 without interaction signal report capacity effect only',
        T1_primary='overall effective-good 3 signs agree, mean>2*null effective-good SD/sqrt(3)',
        T2_T3_primary='change coverage positive signal and no harm',
        bootstrap=dict(draws=20000,seed=20261114,unit='whole dev target track then paired model seed',family=176,adjustment='Bonferroni descriptive intervals; no null-scale gate substituted'),
        T3='Human-confirmed rotate each future displacement by kappa*t, integrate, preserve each step speed',
        legacy='Compute lost<=1%, five pooled motion<=1.01 B0, change coverage+5pp and ADE/FDE each -5%; never revise historical verdicts',
        no_baseline_upgrade=True,no_official_test=True,no_factor_combination=True)
    write_json(HERE/'E2E_V2_CONTRACT.json',contract)
    write_json(HERE/'E2E_V2_SEAL.json',dict(at=now(),files=seal([HERE/'E2E_V2_CONTRACT.json',HERE/'E1/BUDGET_DECISION.json',HERE/'E1/E1A_NULL.json']),
        condition='SEALED_BEFORE_ANY_FORMAL_E2_E3_E4_TRAINING'))


def signal(ds,sd):
    ds=np.asarray(ds,float);mean=float(ds.mean());threshold=2*sd/math.sqrt(3)
    hit=bool(mean!=0 and np.all(np.sign(ds)==np.sign(mean)) and abs(mean)>threshold)
    return dict(per_seed=ds.tolist(),mean=mean,threshold=threshold,signal=hit,positive=hit and mean>0,negative=hit and mean<0)


def compare(names,refs,split,contract):
    data=dict(np.load(HERE/'C0'/f'{split}.npz'));results={}
    mq=[qread(n,split,data) for n in names];bq=[qread(n,split,data) for n in refs]
    mr=[population(n,split) for n in names];br=[population(n,split) for n in refs]
    refstats=ex.fit_reference(HERE/'C0/MOTION_REFERENCE.json')
    # Summaries include same-population GT and oracle motion; all sources are saved arrays.
    ms=[];bs=[]
    for nn,bag in [(names,ms),(refs,bs)]:
        for name in nn:
            with np.load(prediction_file(name,split)) as p:
                summary,_=ex.summarize(dict(paths=p['paths'],logits=p['logits']),data,refstats)
            bag.append(summary)
    harms=[]
    for group,mask in ex.group_masks(data['group']).items():
        floor=contract['null'][split]['summary'][group]
        exchange=[ex.pair_exchange(b['coverage'],m['coverage'],data['group'])[group] for b,m in zip(bq,mq)]
        sig=signal([x['coverage_diff_pp'] for x in exchange],floor['coverage_diff_pp']['sd'])
        excess=float(np.mean([x['lost_rate'] for x in exchange]))-floor['lost_rate']['mean']
        motion_harm=[];tail_harm=[]
        for i in range(3):
            for port in ('oracle_vs_GT','allK20_vs_fitGT'):
                for key in MOTION:
                    a,b=ms[i][group][port][key]['abs_log_ratio'],bs[i][group][port][key]['abs_log_ratio']
                    if a is None or b is None:raise ValueError('UNDEFINED_GT_RATIO_NO_FLOOR')
                    if a-b>.05:motion_harm.append(dict(seed=42+i,port=port,key=key,increase=a-b))
            for port in mr[i][group]['tail_risk']:
                for key in mr[i][group]['tail_risk'][port]:
                    diff=mr[i][group]['tail_risk'][port][key]['exceedance_rate']-br[i][group]['tail_risk'][port][key]['exceedance_rate']
                    if diff>.01:tail_harm.append(dict(seed=42+i,port=port,key=key,increase=diff))
        legacy=[]
        for i in range(3):
            motions={key:mr[i][group]['motion'][key]<=1.01*br[i][group]['motion'][key] for key in ('speed_p95','accel_mean','accel_p95','jerk_mean','jerk_p95')}
            legacy.append(dict(seed=42+i,lost_le_1pct=exchange[i]['lost_rate']<=.01,five_motion=motions,
                protected_ADE_FDE_le_1pct={key:mr[i][group][key]<=1.01*br[i][group][key] for key in ('minADE','minFDE')},
                coverage_gain_ge_5pp=exchange[i]['coverage_diff_pp']>=5,
                ADE_FDE_improve_ge_5pct={key:mr[i][group][key]<=.95*br[i][group][key] for key in ('minADE','minFDE')}))
        flags=dict(excess_lost=excess>floor['tol'],coverage_negative_signal=sig['negative'],motion_away_from_GT=bool(motion_harm),tail=bool(tail_harm))
        if any(flags.values()):harms.append(group)
        ef=signal([m[group]['effective_good']-b[group]['effective_good'] for m,b in zip(mr,br)],floor['effective_good_diff_sd'])
        results[group]=dict(coverage=sig,effective_good=ef,exchange=exchange,excess_lost=excess,tol=floor['tol'],harm_flags=flags,
            motion_harm=motion_harm,tail_harm=tail_harm,LEGACY_SHADOW=legacy,
            minADE_per_seed=[m[group]['minADE']-b[group]['minADE'] for m,b in zip(mr,br)],
            minFDE_per_seed=[m[group]['minFDE']-b[group]['minFDE'] for m,b in zip(mr,br)])
    # J-oracle slot entropy, fixed common-coverage top1 denominator, and segment errors.
    for i in range(3):
        common=mq[i]['coverage']&bq[i]['coverage']
        with np.load(prediction_file(names[i],split)) as mp,np.load(prediction_file(refs[i],split)) as bp:
            for group,mask in ex.group_masks(data['group']).items():
                h=np.bincount(mq[i]['J_winner'][mask],minlength=20).astype(float);h/=h.sum()
                c=common&mask; idx=np.flatnonzero(c)
                results[group].setdefault('mechanisms',[]).append(dict(seed=42+i,J_slot_entropy=float(-(h[h>0]*np.log(h[h>0])).sum()),
                    common_coverage_n=len(idx),method_top1_recall=float(mq[i]['good'][idx,mp['logits'][idx].argmax(1)].mean()) if len(idx) else None,
                    reference_top1_recall=float(bq[i]['good'][idx,bp['logits'][idx].argmax(1)].mean()) if len(idx) else None))
    return dict(method_names=names,reference_names=refs,groups=results,harm_groups=harms,no_harm=not harms)


def bootstrap_compare(names,refs):
    data=dict(np.load(HERE/'C0/dev.npz'));mq=[qread(n,'dev',data) for n in names];bq=[qread(n,'dev',data) for n in refs]
    mr=[population(n,'dev') for n in names];br=[population(n,'dev') for n in refs]
    # Cache only per-window effective-good, already verified by two complete-link implementations.
    def effective(n):
        path=(OLD/'populations/dev/window_metrics.npz' if n=='OLD42' else HERE/'runs'/n/'dev/window_metrics.npz')
        # clusters are not in legacy error npz; recompute once from saved predictions.
        import sys
        sys.path.insert(0,str(ROOT/'src'))
        from f16.eval import clusters
        with np.load(prediction_file(n)) as p:return clusters(p['paths'],ex.quality(p['paths'],data['future'])['good'])[:,1]
    me=[effective(n) for n in names];be=[effective(n) for n in refs]
    tracks=np.array(['|'.join(str(x).split('|')[:2]) for x in data['ids']]);unique,codes=np.unique(tracks,return_inverse=True)
    rng=np.random.default_rng(20261114);nt=len(unique);draws=20000
    ti=rng.integers(0,nt,(draws,nt));si=rng.integers(0,3,(draws,3))
    rows={}
    for group,mask in ex.group_masks(data['group']).items():
        counts=np.bincount(codes,weights=mask,minlength=nt);den=counts[ti].sum(1);valid=den>0
        if len(np.unique(tracks[mask]))<20 or valid.mean()<.95:raise ValueError('BOOTSTRAP_INSUFFICIENT_TRACKS')
        rows[group]={}
        for key in ('coverage','minADE','minFDE','effective_good'):
            if key=='effective_good':delta=np.array(me)-np.array(be)
            elif key=='coverage':delta=np.array([q[key] for q in mq],float)-np.array([q[key] for q in bq],float)
            else:
                k='ADE' if key=='minADE' else 'FDE';delta=np.array([q[k].min(1) for q in mq])-np.array([q[k].min(1) for q in bq])
            sums=np.array([np.bincount(codes,weights=x*mask,minlength=nt) for x in delta])
            # Shared resamples across all metrics/groups; then paired model seeds.
            selected=sums[:,ti].sum(2).T
            boot=selected[np.arange(draws)[:,None],si].mean(1)[valid]/den[valid]
            # Independent expanded-window formula on first 8 valid draws.
            for r in np.flatnonzero(valid)[:8]:
                multip=np.bincount(ti[r],minlength=nt)[codes]*mask
                direct=np.mean([(delta[s]*multip).sum()/multip.sum() for s in si[r]])
                if abs(direct-boot[np.flatnonzero(valid).tolist().index(r)])>2e-12:raise ValueError('BOOTSTRAP_INDEPENDENT_ARITHMETIC')
            rows[group][key]=dict(point=float(delta[:,mask].mean()),CI95=np.quantile(boot,[.025,.975]).tolist(),
                family176_CI=np.quantile(boot,[.025/176,1-.025/176]).tolist(),valid_draws=int(valid.sum()),tracks=int(len(np.unique(tracks[mask]))),draws=draws,
                seed=20261114,independent_expanded_window_checks=8)
    return rows


def final_report(E,refs):
    contract=read_json(HERE/'E2E_V2_CONTRACT.json');verify(read_json(HERE/'E2E_V2_SEAL.json')['files'])
    report=dict(E_star=E,comparisons={},interpretation='Exploratory repeated-development study; null scales are coarse screens, not formal significance. No baseline upgrade.')
    for arm in ARMS:
        names=[f'{arm}_s{s}' for s in (42,43,44)]
        row={split:compare(names,refs[:3],split,contract) for split in ('fit','dev')}
        row['bootstrap_dev']=bootstrap_compare(names,refs[:3])
        dev=row['dev'];noharm=dev['no_harm']
        if arm.startswith('E2_'):
            improvements=[];adeokay=[]
            for n,b in zip(names,refs[:3]):
                a=read_json(HERE/'runs'/n/'dev/summary_v2.json')['overall']
                if b=='OLD42':
                    data=dict(np.load(HERE/'C0/dev.npz'))
                    with np.load(prediction_file(b)) as p:z,_=ex.summarize(p,data,ex.fit_reference(HERE/'C0/MOTION_REFERENCE.json'))
                    bb=z['overall']
                else:bb=read_json(HERE/'runs'/b/'dev/summary_v2.json')['overall']
                improvements.append(bb['allK20_vs_fitGT']['junction_accel_mean']['abs_log_ratio']-a['allK20_vs_fitGT']['junction_accel_mean']['abs_log_ratio'])
                adeokay.append(a['minADE']<=1.01*bb['minADE'])
            h2a=bool(min(improvements)>0 and np.mean(improvements)>2*np.std(improvements,ddof=1))
            row['H2a']=dict(improvement_per_seed=improvements,signal=h2a)
            success=h2a and noharm and all(adeokay)
        elif arm=='T1':success=dev['groups']['overall']['effective_good']['positive'] and noharm
        elif arm in ('T2','T3','E3_self'):success=dev['groups']['change']['coverage']['positive'] and noharm
        else:success=False
        row['verdict']='SUCCESS_EXPLORATORY_SCREEN' if success else 'NO_SUCCESS_UNDER_PRESPECIFIED_SCREEN'
        report['comparisons'][arm]=row
        write_json(HERE/'analysis'/f'{arm}.json',row)
    for label,refarm in [('E3_interaction','E3_self'),('E3_geometry_bias','E3_set_nobias')]:
        names=[f'E3_set_s{s}' for s in (42,43,44)];r=[f'{refarm}_s{s}' for s in (42,43,44)]
        row={split:compare(names,r,split,contract) for split in ('fit','dev')}
        row['bootstrap_dev']=bootstrap_compare(names,r);report['comparisons'][label]=row
        write_json(HERE/'analysis'/f'{label}.json',row)
    setr=report['comparisons']['E3_set'];interaction=report['comparisons']['E3_interaction']['dev']['groups']['change']['coverage']['positive']
    setr['verdict']='SUCCESS_EXPLORATORY_SCREEN' if interaction and setr['dev']['no_harm'] else 'NO_SUCCESS_UNDER_PRESPECIFIED_SCREEN'
    report['E3_capacity_only']=bool(not interaction and report['comparisons']['E3_self']['dev']['groups']['change']['coverage']['positive'] and setr['dev']['groups']['change']['coverage']['positive'])
    write_json(HERE/'RESULT_ANALYSIS.json',report)
    text=['# 下一阶段实验结果',f'\nE*={E}；E1→合同封存→E2/E3/E4 已完成。',
          '\n重复使用的开发集，仅为探索性研究；噪声标尺不能当作正式显著性检验。五折和因素组合未执行，baseline 未升级。',
          '\n| 方法 | change覆盖差(pp) | E2E_V2伤害组 | 预定规则 |','|---|---:|---|---|']
    for arm in ARMS:
        r=report['comparisons'][arm];d=r['dev']
        text.append(f"|{arm}|{d['groups']['change']['coverage']['mean']:.4f}|{','.join(d['harm_groups']) or '无'}|{r['verdict']}|")
    text+=['\n完整 fit/dev、逐 seed、救回/丢失、GT参照运动、尾部、LEGACY_SHADOW、整track+seed区间见 RESULT_ANALYSIS.json 和 analysis/。',
           '\nT3 使用用户确认的逐步位移旋转方案。A7 是否复现历史 B0 及其库版本局限见 E1/E1A_NULL.json。']
    (HERE/'RESULT_ANALYSIS.md').write_text('\n'.join(text),encoding='utf-8')


if __name__=='__main__':
    import sys
    if sys.argv[1]=='e1a':e1a()
    elif sys.argv[1]=='final':final_report(int(sys.argv[2]),json.loads(sys.argv[3]))
    else:raise ValueError('UNKNOWN_ANALYSIS_MODE')
