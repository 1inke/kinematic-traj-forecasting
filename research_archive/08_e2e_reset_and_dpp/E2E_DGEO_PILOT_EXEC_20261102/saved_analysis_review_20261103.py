"""Explicit CPU analysis extension: SHA/readers and saved scalar/event recount, no model/cluster replay."""
import csv,json,hashlib
from pathlib import Path
from datetime import datetime
import numpy as np

R=Path(__file__).resolve().parent;B=R.parent/'E2E_MOTION_RESEARCH_RESET_EXEC_20261102';O=R/'analysis_review_20261103'
ARMS=('DG-Base','DG-GeoGate','DG-GeoDiff','DG-GeoScore');GROUPS=('overall','stable','transition','change')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(4*1024**2),b''):h.update(x)
    return h.hexdigest()
def save(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvout(p,rows):
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    O.mkdir(exist_ok=False);files={}
    for receipt in ('source_manifest.json','PRESERVATION.json','completion_review/REVIEW.json','GEO_EXTRA_REVIEW.json','FINAL_SEAL.json'):
        v=read(R/receipt)
        for key in ('files','sources','outputs'):
            for p,h in v.get(key,{}).items():
                if p in files and files[p]!=h:raise ValueError('INCONSISTENT_DIGEST')
                files[p]=h
    for p,h in files.items():
        if sha(p)!=h:raise ValueError('SOURCE_OR_OUTPUT_CHANGED '+p)
    state=read(R/'WORKER_STATE.json');exit=read(R/'WORKER_EXIT.json');result=read(R/'RESULT.json')
    if state['status']!='SUCCESS' or exit['status']!='SUCCESS' or result['status']!='SUCCESS_AUTHORIZED_ALL4_DGEO_PILOTS_REVIEWED':raise ValueError('NOT_FULLY_COMPLETE')
    paired=read(R/'PAIRED_COMPARISON_VS_B0.json');screens=read(R/'PILOT_SCREEN.json');mech=read(R/'DGEO_MECHANISM_DIAGNOSIS.json')
    base={s:read(B/'populations'/s/'metrics.json') for s in ('fit','dev')}
    rows=[];events=[];tails=[];resources=[];pops={};timing=[]
    for arm in ('B0',)+ARMS:
        for split in ('fit','dev'):
            root=B/'populations'/split if arm=='B0' else R/'populations'/arm/split
            done=read(root/'done.json')
            for p,h in done['files'].items():
                if sha(p)!=h:raise ValueError('SAVED_POPULATION_DIGEST')
            with np.load(root/'predictions.npz') as saved:
                data={k:saved[k] for k in ('ids','groups','logits','future')}
            e=dict(np.load(root/'window_metrics.npz'));old=dict(np.load(B/'populations'/split/'window_metrics.npz'))
            with np.load(B/'populations'/split/'predictions.npz') as saved:
                if not np.array_equal(saved['ids'],data['ids']) or not np.array_equal(saved['future'],data['future']):raise ValueError('UNPAIRED_SOURCE_ID_GT')
            good=(e['ADE']<=.2)&(e['FDE']<=.4);cov=good.any(1)
            if not np.array_equal(cov,e['coverage']):raise ValueError('NOT_SAME_CANDIDATE_COVERAGE')
            top=data['logits'].argmax(1);topgood=good[np.arange(len(cov)),top]
            metrics=read(root/'metrics.json')
            if split=='dev':pops[arm]=dict(data,good=good,cov=cov,topgood=topgood)
            for j,g in enumerate(GROUPS):
                mask=np.ones(len(cov),bool) if j==0 else data['groups']==j-1
                a=metrics[g];b=base[split][g];rescue=mask&cov&~old['coverage'];lost=mask&~cov&old['coverage']
                if int(cov[mask].sum())!=a['good'] or abs(float(e['ADE'][mask].min(1).mean())-a['minADE'])>2e-12:raise ValueError('GROUP_SCALARS')
                if arm!='B0':
                    q=paired[arm][split][g]
                    if q['rescue']!=int(rescue.sum()) or q['lost']!=int(lost.sum()) or q['rescue_ids']!=data['ids'][rescue].tolist() or q['lost_ids']!=data['ids'][lost].tolist():raise ValueError('EVENT_RECOUNT')
                rows.append(dict(arm=arm,split=split,group=g,n=a['n'],tracks=a['target_tracks'],good=a['good'],coverage_percent=100*a['coverage'],coverage_delta_pp=100*(a['coverage']-b['coverage']),
                    minADE=a['minADE'],minADE_relative_percent=100*(a['minADE']/b['minADE']-1),minFDE=a['minFDE'],minFDE_relative_percent=100*(a['minFDE']/b['minFDE']-1),
                    rescue=int(rescue.sum()),lost=int(lost.sum()),reference_good=b['good'],lost_reference_good_percent=100*lost.sum()/b['good'],top1_good=int(topgood[mask].sum()),
                    conditional_top1_percent=100*topgood[mask].sum()/a['good'],distinct=a['distinct'],effective_good=a['effective_good']))
            if arm!='B0':
                for i in np.flatnonzero(cov!=old['coverage']):
                    events.append(dict(arm=arm,split=split,id=str(data['ids'][i]),group=int(data['groups'][i]),event='rescue' if cov[i] else 'lost',
                        old_J=float(old['J'][i].min()),new_J=float(e['J'][i].min()),old_support_margin=float(1-old['J'][i].min()),J_damage=float(e['J'][i].min()-old['J'][i].min())))
        if arm=='B0':continue
        d=read(R/'runs'/arm/'done.json');rs=read(R/'runs'/arm/'RESOURCE.json')
        with (R/'runs'/arm/'history.csv').open(encoding='utf-8-sig') as f:h=list(csv.DictReader(f))
        if d['updates']!=2720 or len(h)!=2720 or any(sum(int(x['epoch'])==j for x in h)!=340 for j in range(1,9)):raise ValueError('RUN_ACCOUNTING')
        resources.append(dict(arm=arm,own_private_peak=rs['snapshot']['peak_private'],own_RSS_peak=rs['snapshot']['peak_rss'],observations=len(rs['observations']),warnings=sum(bool(x.get('warnings')) for x in rs['observations']),stops=sum(x['status']!='PASS' for x in rs['observations']),not_whole_tree_peak=True))
        timing.append(dict(arm=arm,science_save_mechanism_seconds=d['elapsed_seconds']))
        for x in screens[arm]['tail_comparisons']:
            tails.append(dict(arm=arm,**x,conditional_magnitude_relative_percent=100*(x['positive_excess_p95']/x['reference_positive_excess_p95']-1) if x['reference_positive_excess_p95']>0 else None))
    common=read(R/'COMMON_COVERED_SUPPORT_TOP1.json');new_common={}
    cases=[(a+'-B0',('B0',a)) for a in ARMS]+[('GeoScore-GeoGate',('B0','DG-GeoGate','DG-GeoScore')),('GeoScore-GeoGate_PAIR_ONLY',('DG-GeoGate','DG-GeoScore'))]
    for case,names in cases:
        intersection=np.logical_and.reduce([pops[n]['cov'] for n in names]);new_common[case]={}
        for j,g in enumerate(GROUPS):
            mask=intersection if j==0 else intersection&(pops['B0']['groups']==j-1)
            ref=common[case][g]
            if int(mask.sum())!=ref['n']:raise ValueError('COMMON_DENOMINATOR')
            new_common[case][g]=dict(n=int(mask.sum()),models={})
            for n in names:
                count=int(pops[n]['topgood'][mask].sum())
                if count!=ref['recall'][n]['top1_good']:raise ValueError('COMMON_TOP1_RECOUNT')
                new_common[case][g]['models'][n]=dict(hit=count,recall=float(count/mask.sum()) if mask.any() else None)
    # Fixed population of new/retained/lost opportunities: no model-specific denominator swap.
    top_exchange={}
    for arm in ARMS:
        a=pops[arm];b=pops['B0'];mask=b['groups']==2;both=mask&a['cov']&b['cov'];lost=mask&~a['cov']&b['cov'];resc=mask&a['cov']&~b['cov']
        top_exchange[arm]=dict(change_N=960,common_supported=int(both.sum()),common_B0_top1=int(b['topgood'][both].sum()),common_method_top1=int(a['topgood'][both].sum()),
            old_top1_removed_with_lost_support=int(b['topgood'][lost].sum()),new_top1_in_rescues=int(a['topgood'][resc].sum()),total_B0_top1=int(b['topgood'][mask].sum()),total_method_top1=int(a['topgood'][mask].sum()))
    seconds=(datetime.fromisoformat(exit['finished_at'])-datetime.fromisoformat(exit['started_at'])).total_seconds()
    pre=read(R/'PRELAUNCH_REVIEW.json');parts=pre['parts'];measured=sum(parts[k] for k in ('actual_numeric_preflight','actual_raw_cost','actual_pairing_regression','actual_late_saved_consumer'))+seconds
    csvout(O/'ALL_GROUPS_REVIEWED.csv',rows);csvout(O/'SUPPORT_EXCHANGE.csv',events);csvout(O/'TAIL_COMPARISONS.csv',tails)
    analysis=dict(status='ANALYZED_DGEO_OVERALL_GAIN_CHANGE_HARM_NO_FULL_SCREEN_PASS',source_count=167,verified_unique_files=len(files),models=4,epochs=32,updates=10880,
        worker_seconds=seconds,measured_preflight_cost_regression_plus_worker_seconds=measured,planning_seconds=pre['planning_seconds'],budget_seconds=2700,
        results=rows,common_top1_verified=new_common,top1_support_decomposition=top_exchange,screens=screens,mechanisms=mech['initial_final'],resources=resources,timing=timing,
        independent_review_scope='CPU saved scalar/event and all common-support recount+SHA only; existing complete-link/pdist/fullquality/pooled+junction dual receipts verified, no full model gradient or full clustering replay.',
        DA1='Prelaunch approved actual64/clip5/source/gradient stages/zeroNA preserved, new four-arm direction and45min budget explicit; same assistant role, not external review.',
        DA2='Cannot cherry-pick overall gains: GeoDiff +108 overall = stable+98 transition+26 change-16; change FDE+4.93% and oldgood loss67/350. Actual feature de-collinearity/negative attention works numerically but not the protected scientific objective.',
        DA3='GeoDiff has descriptive adjacent improvement vsGeoGate, not full success; GeoScore common-change ranking worsens111->104 on292 windows. Rare acceleration extremes and speed increase remain despite smoother junction and some bulk quantiles. No automatic seeds/V/F or causal/intent claims.',
        statistical_scope='One seed42/arm, dev4753 overlapping windows/248tracks, change960/102tracks, repeated dev/sisterfootage unknown. No confirmatory p/CI, no independent whole-scene generalization. Nominal structure and shared B0 weights do not isolate geometry from centering or other engineering changes.',
        next='Keep B0 comparator. Preserve GeoDiff overall signal and GeoGate common-change recall signal, but no original full-screen pass or automatic continuation. No new training under this analysis request.')
    save(O/'ANALYSIS.json',analysis)
    lines=['# DGeo四臂复核分析','',
        '结论：几何特征+模式中心化确实改变了表示，并带来比上一轮更明显的总体误差改善；但收益主要来自stable，change保护/覆盖和物理极端尾部没有解决，四臂均不满足原完整筛选门。差分相对普通几何门控有配对点增益；后验关系打分头本次未显示增益。',
        '',f'完整worker/exit/RESULT及GEO_EXTRA_REVIEW/FINAL_SEAL均SUCCESS。4fresh seed42各8轮/2720，共32轮10880更新；worker实耗{seconds/60:.3f}min，实际前检/成本/回归加worker合计约{measured/60:.3f}min，低于本次45min规划预算（不含代码实施及非计算对话时间）。原167来源与{len(files)}去重来源/输出SHA复核一致，全部8科学人口审核收据齐。',
        '', '|模型|总体ADE|总体FDE|总体成功窗|change ADE|change FDE|change成功窗|','|---|---:|---:|---:|---:|---:|---:|']
    for arm in ('B0',)+ARMS:
        a=next(x for x in rows if x['arm']==arm and x['split']=='dev' and x['group']=='overall');c=next(x for x in rows if x['arm']==arm and x['split']=='dev' and x['group']=='change')
        lines.append(f"|{arm}|{a['minADE']:.6f}|{a['minFDE']:.6f}|{a['good']}/4753|{c['minADE']:.6f}|{c['minFDE']:.6f}|{c['good']}/960|")
    lines+=['','## 1. 去共线成功，但不能等同预测目标成功','',
        '固定128 dev输入的qk off-diagonal cosine：DG-Base约.9865→.9960；GeoGate约-.0466→-.0398；GeoDiff约-.0459→-.0419。GeoDiff负权41.875%→39.634%，不再是上一轮1.59%。原所有新增参数第2步已连通，数值/初始等价与精确恢复通过。这支持“实现确实工作”，不证明去共线是全部精度变化的独因果。',
        '', '## 2. 总体收益是真实点估计，但主要来自stable','',
        'GeoDiff总体ADE相对B0约下降3.07%、FDE下降3.92%，coverage+108/4753=+2.272pp。其分解是stable+98、transition+26、change-16；change ADE恶化1.47%、FDE恶化4.93%，coverage350→334。GeoGate整体+85但change-31；GeoScore整体+72但change-48。此现象在FIT也存在：GeoGate/GeoDiff/GeoScore的change成功1495/1585/1457，均低于B0的1660，不应只解释为dev统计门过严。',
        '差分相对GeoGate总体ADE下降约.228%、总体多23/change多15成功窗，符合事先相邻比较的点信号；但仍没有修复相对B0的change退化。DG-Base的change360/960反而高于三种几何臂，但自身总体/稳定FDE及尾部门不通过，不自动挑它改名方法。',
        '', '## 3. 评分效果必须固定共同支持分母','',
        '各自已覆盖change集合：B0106/350=30.29%，GeoGate115/319=36.05%，GeoDiff101/334=30.24%，GeoScore104/302=34.44%。GeoGate相对B0共同覆盖267窗上113/267 vs99/267，仍有14次Top1净增，故不是全靠分母变小；但这不抵消其丢失83个B0成功change窗。',
        '新增关系打分器应对比GeoScore与GeoGate：共同覆盖292个change窗，GeoGate111/292=38.01%，GeoScore104/292=35.62%；三者共同覆盖253窗也111→104。GeoScore还比GeoGate少17个change成功窗，故该评分头本次未获支持。不同训练模型候选坐标也不同，共同窗口不等同一候选池，不能再称纯固定候选排序因果隔离。',
        '', '## 4. 支持交换与运动风险未解决','',
        'change rescue/lost：DG-Base32/22、GeoGate52/83、GeoDiff51/67、GeoScore45/93；GeoDiff旧好丢失67/350=19.14%，不是小净变化。总体matched事件及Jmargin全保留在SUPPORT_EXCHANGE.csv，不用净增掩盖。',
        'GeoDiff总体接续acc p95从1.4706降至1.3204，确有变平滑的部分；但速度p95从1.2828升至1.3791，速度超阈样本6760→12420（相同1140720候选×时间点分母，非独立人）。内部acc p95/p99有下降，但极端超阈631→2515，正excess条件p95幅度+82.72%。这不是所有运动整体恶化，也不是纯条件分母截断：更稀少的极端尾部确实扩大。C0的经验阈不是生理硬安全上限，原统计和验收规则不因解释改变。',
        '四臂rate增量均<=+1pp，但可估幅度项有真实失败；零参考jerk仍NOT_ESTIMABLE_NO_FLOOR，不补PASS。GeoGate/GeoDiff/GeoScore的原连续ADE门还因change超过1%害而失败，不能把未晋级只归于运动门。',
        '', '## ARS DA2/DA3与统计边界（同助手角色，非外审）','',analysis['DA2'],analysis['DA3'],analysis['statistical_scope'],
        '', '当前结论：保留B0为开发对照、不升级baseline，不自动补seed43/44/V/F或扫描宽度/LR/loss。可保留“几何联合干预有总体误差信号、差分相邻对照有点增益、GeoGate共同change排序有局部信号”，但不能宣称目标机动预测、旧支持保留或安全成功，也不能反推所有几何/差分方法无效。全部文件/模型/失败/日志/自动记录和旧PAUSED保持。']
    with (R/'RESULT_ANALYSIS_REVIEWED.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    outputs=[O/n for n in ('ANALYSIS.json','ALL_GROUPS_REVIEWED.csv','SUPPORT_EXCHANGE.csv','TAIL_COMPARISONS.csv')]+[R/'RESULT_ANALYSIS_REVIEWED.md']
    save(O/'REVIEW.json',dict(status='PASS_READONLY_SOURCE_OUTPUT_SHA_GROUP_EVENT_COMMON_SUPPORT_RECOUNT',verified_unique_files=len(files),source_count=167,
        source_extension={str(Path(__file__).resolve()):sha(__file__)},files={str(p):sha(p) for p in outputs},no_model_forward_backward=True,no_training=True,no_full_clustering_replay=True,same_assistant_not_external=True))
    print(json.dumps(dict(status='DGEO_SAVED_ANALYSIS_REVIEW_PASS',verified=len(files),worker_minutes=seconds/60,measured_full_components_minutes=measured/60,resources=resources,
        dev=[x for x in rows if x['split']=='dev' and x['group'] in ('overall','change')]),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
