"""Old success/protection rules disclosed separately from DSF EC diagnostics."""
from run_common import *
FIVE=('speed_p95','accel_mean','accel_p95','jerk_mean','jerk_p95')

def main():
    comparisons=read(RUN/'PRIMARY_COMPARISON.json');rows=[]
    for split in ('fit','dev'):
        native=read(RUN/'populations/native'/split/'metrics.json')
        for arm in ('uniform','kin'):
            for seed in (42,43,44):
                folder=RUN/'populations'/f'{arm}_seed{seed}'/split;m=read(folder/'metrics.json');support=read(folder/'support_and_quality.json')
                for group in ('overall','stable','transition','change'):
                    a=m[group];b=native[group];s=support[group]
                    motion={key:dict(model=a['motion'][key],native=b['motion'][key],literal_PASS=a['motion'][key]<=1.01*b['motion'][key]) for key in FIVE}
                    protect_errors=group=='change' or (a['minADE']<=1.01*b['minADE'] and a['minFDE']<=1.01*b['minFDE'])
                    rows.append(dict(arm=arm,seed=seed,split=split,group=group,N=a['n'],good=a['good'],native_good=b['good'],rescue=s['rescue'],lost=s['lost'],native_good_lost_rate=s['lost_rate'],native_good_protection_PASS=s['lost_rate'] is not None and s['lost_rate']<=.01,protected_ADEFDE_PASS=protect_errors,five_motion=motion,all_five_motion_PASS=all(v['literal_PASS'] for v in motion.values()),change_cov_5pp_PASS=100*(a['good']-b['good'])>=5*a['n'] if group=='change' else None,change_ADEFDE_5percent_PASS=(a['minADE']<=.95*b['minADE'] and a['minFDE']<=.95*b['minFDE']) if group=='change' else None))
    primary={r['metric']:r for r in comparisons['rows'] if r['comparison']=='kin-native'}
    write(RUN/'PROTECTION_GATES.json',dict(status='COMPLETE_LITERAL_OLD_GATES_NOT_RELAXED',rows=rows,kin_native_absolute_CI=dict(coverage_lower_positive=primary['coverage']['family18'][0]>0,ADE_upper_negative=primary['minADE']['family18'][1]<0,FDE_upper_negative=primary['minFDE']['family18'][1]<0),strict_old_major_success=False,strict_old_major_reason='Only DSF seeds are varied; one shared pretrained generator, no new three-fresh-generator pipeline validation, scene/joint unresolved, exploratory reused dev. Point estimates and literal gates disclosed without automatic baseline promotion.',no_safety_epsilon_or_floor=True))
    report=['# DSF 数值结果与原保护门','', '|seed|uniform dev change good|kin dev change good|native good|kin lost|kin rescue|','|---|---:|---:|---:|---:|---:|']
    for seed in (42,43,44):
        u=next(r for r in rows if r['arm']=='uniform' and r['seed']==seed and r['split']=='dev' and r['group']=='change');k=next(r for r in rows if r['arm']=='kin' and r['seed']==seed and r['split']=='dev' and r['group']=='change')
        report.append(f"|{seed}|{u['good']}|{k['good']}|{k['native_good']}|{k['lost']}|{k['rescue']}|")
    report+=['','全部48seed/split/group的原成功保留、protected ADE/FDE、五pooled-motion literal1%门见PROTECTION_GATES.json。主family18普通/校正/fixedseed CI完整保留，不挑阳性；与单一冻结基底共享的三DSF优化种子不等三新wholemodel重复。','', '场景/联合未触发，不自动baseline升级或CVAE训练。']
    with (RUN/'RESULT_NUMERIC_SUMMARY.md').open('x',encoding='utf-8') as f:f.write('\n'.join(report)+'\n')

if __name__=='__main__':main()
