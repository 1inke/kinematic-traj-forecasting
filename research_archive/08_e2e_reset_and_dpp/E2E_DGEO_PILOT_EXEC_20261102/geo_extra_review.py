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
    root=BASE/'populations'/split if name=='B0' else RUN/'populations'/name/split
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


def main():
    manifest=read_json(RUN/'source_manifest.json');verify(manifest['files'])
    thresholds=read_json(BASE/'C0/EVALUATION_CONTRACT.json')['tail_thresholds']
    unconditional=[];support={};all_sha={};diagnoses={}
    for arm in ARMS:
        verify(read_json(RUN/'runs'/arm/'done.json')['files'])
        diagnoses[arm]=dict(initial=read_json(RUN/'initial_mechanisms'/arm/'DIAGNOSIS.json'),final=read_json(RUN/'mechanisms'/arm/'DIAGNOSIS.json'))
        for split in ('fit','dev'):
            p=read_population(arm,split);b=read_population('B0',split)
            if not np.array_equal(p['ids'],b['ids']) or not np.array_equal(p['future'],b['future']):raise ValueError('NOT_PAIRED_ID_GT')
            past=p['past'].astype(np.float64);n=len(past)
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
                        physical_slot_time_alignment_not_semantic_path_identity=True))
            if split=='dev':support[arm+'-B0']=common_recall({'B0':b,arm:p})
            del p,b,mm,bm
    gate=read_population('DG-GeoGate','dev');score=read_population('DG-GeoScore','dev');base=read_population('B0','dev')
    support['GeoScore-GeoGate']=common_recall({'B0':base,'DG-GeoGate':gate,'DG-GeoScore':score})
    # Dedicated support uses pair-only intersection; triple is separately retained.
    pair=common_recall({'DG-GeoGate':gate,'DG-GeoScore':score})
    support['GeoScore-GeoGate_PAIR_ONLY']=pair
    controls=read_json(RUN/'PAIRED_MECHANISM_CONTROLS.json');screen=read_json(RUN/'PILOT_SCREEN.json');gains={}
    for method,reference in [('DG-GeoGate','DG-Base'),('DG-GeoDiff','DG-GeoGate'),('DG-GeoScore','DG-GeoGate')]:
        a=read_json(RUN/'populations'/method/'dev/metrics.json');b=read_json(RUN/'populations'/reference/'dev/metrics.json')
        if method!='DG-GeoScore':gain=a['overall']['minADE']<b['overall']['minADE'] and a['change']['minADE']<=1.01*b['change']['minADE']
        else:
            r=pair['change'];gain=r['n']>0 and r['recall']['DG-GeoScore']['top1_good']>r['recall']['DG-GeoGate']['top1_good'] and all(a[g]['coverage']>=b[g]['coverage'] and a[g]['minADE']<=1.01*b[g]['minADE'] and a[g]['minFDE']<=1.01*b[g]['minFDE'] for g in ('overall','change'))
        gains[method]=dict(reference=reference,point_mechanism_gain=bool(gain),original_full_pilot_screen=screen[method]['full_pilot_screen'],
            eligible_to_request_own_three_seed_admission=bool(gain and screen[method]['full_pilot_screen']),no_auto_new_training=True)
    write_csv(RUN/'UNCONDITIONAL_TAILS.csv',unconditional)
    write_json(RUN/'COMMON_COVERED_SUPPORT_TOP1.json',support)
    write_json(RUN/'DGEO_MECHANISM_DIAGNOSIS.json',dict(initial_final=diagnoses,common_support=support,prospective_gain_rules=gains,
        unconditional='UNCONDITIONAL_TAILS.csv',not_causal_proof=True,same_assistant_not_external=True))
    note=['# DGeo 四臂执行与机制评审','',
        '四fresh seed42固定8轮完成。训练/损失/EMA/数据/clip5/64D与原B0配对不变；原筛选结果见PILOT_SCREEN，原始全组与支持交换见ALL_GROUPS/PAIRED_COMPARISON。此补充只CPU保存数据，不再训练/模型前后向/全聚簇。','',
        '|arm|overall ADE|change ADE|overall coverage|change coverage|原联合门|新增机制配对点信号|', '|---|---:|---:|---:|---:|---|---|']
    for arm in ARMS:
        m=read_json(RUN/'populations'/arm/'dev/metrics.json')
        note.append(f"|{arm}|{m['overall']['minADE']:.6f}|{m['change']['minADE']:.6f}|{m['overall']['coverage']:.6f}|{m['change']['coverage']:.6f}|{screen[arm]['full_pilot_screen']}|{gains.get(arm,{}).get('point_mechanism_gain','composite control')}|")
    note+=['','## ARS DA2（同助手，非外审）','',
        '几何+中心化是联合干预，不能分别证明哪个独因果。DG-Base相对B0同时有容量/初值/部分评分特征解耦变化。非零增益出口没有改变初始轨迹，但改变了第一次proposal_delta梯度；不能称全部优化几何保持不变。GeoScore更改新评分残差且可能经共同回归骨干影响候选集，因此共同覆盖和各自覆盖分母同时报告。','',
        '负权和off-diagonal cosine只是算子/表示描述，接近0不等有效模式、负权多不等去噪成功。实际结果优先于方向文档里未独立证实的固定数值、信息论归因或因果陈述。','',
        '## ARS DA3与最终边界（同助手，非外审）','',
        '不能用无条件p95/p99替代原条件尾部门，或用不同条件人群变化直接断言物理风险已被修好。零positive-excess参考仍NOT_ESTIMABLE，不加epsilon/floor。physical-slot相配不等同一语义路径。任何旧支持丢失均按原逐ID记录，不只报净增。','',
        '单seed42、dev相关窗/同track/反复开发、姐妹录像独立性UNKNOWN，不是独立场景泛化或显著性。此worker在四臂结束，三seed仅记满足原门后可申请自身预检/预算，无后台自动续训。原major/baseline/historycycle11/未完goal不变。']
    with (RUN/'DGEO_RESULT_ANALYSIS.md').open('x',encoding='utf-8') as f:f.write('\n'.join(note)+'\n')
    roots=[RUN/n for n in ('UNCONDITIONAL_TAILS.csv','COMMON_COVERED_SUPPORT_TOP1.json','DGEO_MECHANISM_DIAGNOSIS.json','DGEO_RESULT_ANALYSIS.md')]
    verify(manifest['files'])
    write_json(RUN/'GEO_EXTRA_REVIEW.json',dict(status='PASS_DGEO_COMMON_SUPPORT_UNCONDITIONAL_TAIL_AND_FULL_SAVED_SHA',
        files=seal(roots),sources_unchanged=True,statistical_inference='single-seed descriptive only',no_new_training_or_model_replay=True))


if __name__=='__main__':main()
