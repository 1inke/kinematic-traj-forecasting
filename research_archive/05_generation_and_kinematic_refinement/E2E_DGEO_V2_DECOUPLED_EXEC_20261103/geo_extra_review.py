"""CPU saved-data relational support and unconditional-tail review, not model replay."""
from geo_common import *
import numpy as np
from causal_dataset import motion

GROUPS=('overall','stable','transition','change')


def quantile_checked(x,q):
    reference=float(np.quantile(x,q));s=np.sort(np.asarray(x).reshape(-1));t=(len(s)-1)*q;i=int(np.floor(t));j=min(i+1,len(s)-1)
    independent=float(s[i]+(t-i)*(s[j]-s[i]))
    if abs(reference-independent)>2e-12:raise ValueError('UNCONDITIONAL_QUANTILE_INDEPENDENT')
    return reference


def common_recall(populations):
    cov={k:((p['ADE']<=.2)&(p['FDE']<=.4)).any(1) for k,p in populations.items()}
    intersection=np.logical_and.reduce(list(cov.values()));result={};reference=next(iter(populations.values()))
    for j,g in enumerate(GROUPS):
        mask=intersection if j==0 else intersection&(reference['groups']==j-1)
        result[g]=dict(n=int(mask.sum()),tracks=int(len(set('|'.join(s.split('|')[:2]) for s in reference['ids'][mask]))),recall={})
        for name,p in populations.items():
            top=p['logits'].argmax(1);good=(p['ADE'][np.arange(len(top)),top]<=.2)&(p['FDE'][np.arange(len(top)),top]<=.4)
            result[g]['recall'][name]=dict(top1_good=int(good[mask].sum()),denominator=int(mask.sum()),conditional_recall=float(good[mask].mean()) if mask.any() else None)
    return result


def read_population(name,split):
    root=BASE/'populations'/split if name=='B0' else (V1/'populations'/'DG-GeoDiff'/split if name=='V1-DG-GeoDiff' else RUN/'populations'/name/split)
    done=read_json(root/'done.json');verify(done['files'])
    p=dict(np.load(root/'predictions.npz'));e=dict(np.load(root/'window_metrics.npz'))
    return dict(p,**{k:e[k] for k in ('ADE','FDE','J','coverage')})


def extra_math_fixture():
    # Exact self-reference and different coverage denominator examples, no fabricated empirical result.
    ids=np.array(['a|1|1','a|2|1']);z=np.zeros((2,20));a=np.ones((2,20));f=np.ones((2,20))
    a[0,0]=.1;f[0,0]=.2;a[1,1]=.1;f[1,1]=.2
    p=dict(ADE=a,FDE=f,logits=z,groups=np.array([2,2]),ids=ids)
    q=dict(p,logits=z.copy());q['logits'][1,1]=1
    report=common_recall({'B0':p,'method':q})
    if report['change']['n']!=2 or report['change']['recall']['B0']['top1_good']!=1 or report['change']['recall']['method']['top1_good']!=2:raise ValueError('COMMON_COVERAGE_FIXTURE')
    return dict(status='PASS_EXACT_COMMON_COVERAGE_DENOMINATOR_FIXTURE')


def factorial_contrast(a,b,c,d):
    return dict(A_at_native=b-a,A_at_Taylor=d-c,B_at_no_geo=c-a,B_at_maneuver=d-b,
        A_mean_effect=((b-a)+(d-c))/2,B_mean_effect=((c-a)+(d-b))/2,
        interaction=d-b-c+a,single_seed_descriptive_only=True)


def extra_screen(s,unconditional,arm,gm,gb):
    r=s['tail_comparisons']
    positive=[x for x in r if x['metric']!='jerk'];zeros=[x for x in r if x['metric']=='jerk']
    est=len(positive)==16 and all(x['status']=='ESTIMABLE' and x['rate_pass'] and x['magnitude_pass'] is True for x in positive)
    zero_rates=len(zeros)==4 and all(x['rate_pass'] for x in zeros)
    jerk_rows=[x for x in unconditional if x['arm']==arm and x['split']=='dev' and x['metric']=='jerk']
    no_new_jerk=len(jerk_rows)==4 and all(x['new_exceed']==0 for x in jerk_rows)
    benefit=all(gm[g]['coverage']>=gb[g]['coverage'] and gm[g]['effective_good']>=gb[g]['effective_good'] for g in ('overall','change'))
    return dict(strict_literal_C0_screen=s['full_pilot_screen'],
        estimable_16_plus_zero_jerk_pass=bool(s['continuous_ADE_pass'] and s['FDE_noninferiority_pass'] and est and zero_rates),
        all16_estimable_pass=bool(est),zero_reference_rate_tolerance_pass=bool(zero_rates),strict_zero_jerk_new_exceed_eq0=bool(no_new_jerk),
        zero_rate_tolerance_is_not_literal_zero_or_relative_noninferiority=True,
        overall_change_coverage_effectivegood_nonlower=bool(benefit),request_only_no_automatic_followup=True)


def main():
    import itertools
    manifest=read_json(RUN/'source_manifest.json');verify(manifest['files'])
    thresholds=read_json(BASE/'C0/EVALUATION_CONTRACT.json')['tail_thresholds']
    unconditional=[];support={};diagnoses={};pair_history={}
    for arm in ARMS:
        verify(read_json(RUN/'runs'/arm/'done.json')['files'])
        diagnoses[arm]=dict(initial=read_json(RUN/'initial_mechanisms'/arm/'DIAGNOSIS.json'),final=read_json(RUN/'mechanisms'/arm/'DIAGNOSIS.json'))
        for split in ('fit','dev'):
            p=read_population(arm,split);b=read_population('B0',split)
            if not np.array_equal(p['ids'],b['ids']) or not np.array_equal(p['future'],b['future']):raise ValueError('MATCHED_ID_GT')
            n=len(p['ids']);past=p['past'].astype(np.float64)
            mm=motion(np.broadcast_to(past[:,None],(n,20,8,2)),p['paths'].astype(np.float64))
            bm=motion(np.broadcast_to(past[:,None],(n,20,8,2)),b['paths'].astype(np.float64))
            for j,g in enumerate(GROUPS):
                mask=np.ones(n,bool) if j==0 else p['groups']==j-1
                for key,threshold in thresholds.items():
                    x=mm[key][mask];y=bm[key][mask];old=y>threshold;new=x>threshold
                    unconditional.append(dict(arm=arm,split=split,group=g,metric=key,Nwindows=int(mask.sum()),
                        correlated_candidate_time_readings=int(x.size),threshold=threshold,
                        p95=quantile_checked(x,.95),p99=quantile_checked(x,.99),baseline_p95=quantile_checked(y,.95),baseline_p99=quantile_checked(y,.99),
                        old_exceed=int(old.sum()),new_exceed=int(new.sum()),both_exceed=int((old&new).sum()),
                        resolved_old_only=int((old&~new).sum()),introduced_new_only=int((~old&new).sum()),
                        method_excess_p95_on_fixed_old_support=float(np.quantile(np.maximum(x[old]-threshold,0),.95)) if old.any() else None,
                        physical_slot_time_alignment_not_semantic_identity=True))
            del p,b,mm,bm
    populations={name:read_population(name,'dev') for name in ('B0','V1-DG-GeoDiff')+ARMS}
    for a,b in itertools.combinations(populations,2):
        support[a+'__'+b]=common_recall({a:populations[a],b:populations[b]})
    support['ALL_COMMON']=common_recall(populations)
    from compare_geo import summarize_pair
    for arm in ARMS:
        pair_history[arm]={}
        for split in ('fit','dev'):
            pa=read_population(arm,split);pb=read_population('V1-DG-GeoDiff',split)
            aa=dict(np.load(RUN/'populations'/arm/split/'window_metrics.npz'))
            bb=dict(np.load(V1/'populations/DG-GeoDiff'/split/'window_metrics.npz'))
            ga=read_json(RUN/'populations'/arm/split/'metrics.json');gb=read_json(V1/'populations/DG-GeoDiff'/split/'metrics.json')
            pair_history[arm][split]={}
            for j,g in enumerate(GROUPS):
                mask=np.ones(len(pa['ids']),bool) if j==0 else pa['groups']==j-1
                pair_history[arm][split][g]=summarize_pair(aa,bb,ga[g],gb[g],pa,pb,mask)
    screens=read_json(RUN/'PILOT_SCREEN.json');dual={}
    for arm in ARMS:
        gm=read_json(RUN/'populations'/arm/'dev/metrics.json');gb=read_json(BASE/'populations/dev/metrics.json')
        dual[arm]=extra_screen(screens[arm],unconditional,arm,gm,gb)
    factor={}
    for split in ('fit','dev'):
        gm={arm:read_json(RUN/'populations'/arm/split/'metrics.json') for arm in ARMS}
        factor[split]={}
        for group in GROUPS:
            factor[split][group]={}
            for key in ('coverage','minADE','minFDE','minJ','effective_good','distinct'):
                a,b,c,d=[gm[arm][group][key] for arm in ARMS]
                factor[split][group][key]=factorial_contrast(a,b,c,d)
    write_csv(RUN/'UNCONDITIONAL_TAILS.csv',unconditional)
    write_json(RUN/'COMMON_COVERED_SUPPORT_TOP1.json',support)
    write_json(RUN/'V1_REFERENCE_SUPPORT_EXCHANGE.json',pair_history)
    write_json(RUN/'FACTORIAL_EFFECTS.json',factor)
    write_json(RUN/'DUAL_SCREEN.json',dual)
    write_json(RUN/'DGEO_MECHANISM_DIAGNOSIS.json',dict(initial_final=diagnoses,common_support=support,dual_screen=dual,
        factorial='FACTORIAL_EFFECTS.json',unconditional='UNCONDITIONAL_TAILS.csv',not_causal_proof=True,same_assistant_not_external=True))
    lines=['# DGeo V2 completed four-arm factorial pilots','',
        'Four fresh seed42 x8 fixed final EMA; no automatic seeds43/44 or next architecture. Training artifacts, original complete-link/pdist, full quality/motion/junction dual receipts preserved.',
        '','|arm|overall ADE|change ADE|overall good|change good|strict literal screen|new estimable/rate screen|',
        '|---|---:|---:|---:|---:|---|---|']
    for arm in ARMS:
        m=read_json(RUN/'populations'/arm/'dev/metrics.json')
        lines.append(f"|{arm}|{m['overall']['minADE']:.7f}|{m['change']['minADE']:.7f}|{m['overall']['good']}|{m['change']['good']}|{dual[arm]['strict_literal_C0_screen']}|{dual[arm]['estimable_16_plus_zero_jerk_pass']}|")
    lines+=['','## ARS DA2 / DA3, same assistant NOT external',
        'A nonlinear feature interaction need not be nonzero for every history; two independent residual bounds do not ensure whole path safety. Taylor residual still interpolates through piecewise-linear anchors, not smooth constant curvature.',
        'Factor B bundles a changed basis, physical envelope, and scale decoupling. Mean effects are descriptive single-seed factorial contrasts, not significance or pure isolation of every submechanism. Historical V1 uses different new-module seed, so V1 comparisons are not a fully paired one-factor causal proof.',
        'Rate tolerance on zero-reference jerk is <=1pp, not literal zero. Both that proposed new diagnostic screen and strict old screen are retained; neither upgrades baseline or automatically triggers seeds. Explicit strict_zero_jerk flag and all unconditional/support exchange counts are saved.',
        'No claim that a negative outcome proves stable-gradient drowning, information absence, expert routing necessity, or a unique root. Single seed, overlapping windows, repeated development and sister recording independence UNKNOWN. No confirmatory p/CI.',
        'Original major/protection/coverage and motion criteria, baseline/historycycle11 and unfinished long-term objective unchanged. All evidence retained.']
    with (RUN/'DGEO_RESULT_ANALYSIS.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    roots=[RUN/n for n in ('UNCONDITIONAL_TAILS.csv','COMMON_COVERED_SUPPORT_TOP1.json','V1_REFERENCE_SUPPORT_EXCHANGE.json','FACTORIAL_EFFECTS.json','DUAL_SCREEN.json','DGEO_MECHANISM_DIAGNOSIS.json','DGEO_RESULT_ANALYSIS.md')]
    verify(manifest['files'])
    write_json(RUN/'GEO_EXTRA_REVIEW.json',dict(status='PASS_DGEO_COMMON_SUPPORT_UNCONDITIONAL_TAIL_AND_FULL_SAVED_SHA',files=seal(roots),sources_unchanged=True,no_new_training_or_model_replay=True,statistical_inference='single seed descriptive'))
if __name__=='__main__':main()
