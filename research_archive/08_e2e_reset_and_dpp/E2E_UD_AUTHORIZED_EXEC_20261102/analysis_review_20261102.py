"""New explicit CPU saved-data analysis extension. No training/model/pickle/cluster replay."""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parent
B=R.parent/'E2E_MOTION_RESEARCH_RESET_EXEC_20261102'
O=R/'analysis_review_20261102'
ARMS=('U-GRU','U-Delta','D-Gate','D-DiffGate')
GROUPS=('overall','stable','transition','change')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvout(p,rows):
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    O.mkdir(exist_ok=False)
    files={}
    for name in ('source_manifest.json','PRESERVATION.json','FINAL_SEAL.json','completion_review/REVIEW.json'):
        v=read(R/name)
        for key in ('files','sources','outputs'):
            for p,d in v.get(key,{}).items():
                if p in files and files[p]!=d:raise ValueError('INCONSISTENT_SHA '+p)
                files[p]=d
    for p,d in files.items():
        if sha(p)!=d:raise ValueError('SOURCE_OR_OUTPUT_CHANGED '+p)
    result=read(R/'RESULT.json');exit=read(R/'WORKER_EXIT.json')
    if result['status']!='SUCCESS_AUTHORIZED_ALL4_UD_PILOTS_REVIEWED' or exit['status']!='SUCCESS':raise ValueError('NOT_COMPLETE')
    base={s:read(B/'populations'/s/'metrics.json') for s in ('fit','dev')}
    paired=read(R/'PAIRED_COMPARISON_VS_B0.json');screens=read(R/'PILOT_SCREEN.json')
    rows=[];events=[];tails=[];resources=[];run_times=[]
    for arm in ('B0',)+ARMS:
        for split in ('fit','dev'):
            root=B/'populations'/split if arm=='B0' else R/'populations'/arm/split
            metrics=read(root/'metrics.json');e=dict(np.load(root/'window_metrics.npz'))
            p=dict(np.load(root/'predictions.npz'))
            if len(p['ids'])!=(21719 if split=='fit' else 4753):raise ValueError('POPULATION_COUNT')
            if not np.array_equal(e['coverage'],((e['ADE']<=.2)&(e['FDE']<=.4)).any(1)):raise ValueError('SAVED_SAME_SLOT_GOOD')
            old=dict(np.load(B/'populations'/split/'window_metrics.npz'))
            oldp=dict(np.load(B/'populations'/split/'predictions.npz'))
            if not np.array_equal(oldp['ids'],p['ids']) or not np.array_equal(oldp['future'],p['future']):raise ValueError('UNPAIRED_IDS_GT')
            top=p['logits'].argmax(1);topgood=e['good'][np.arange(len(top)),top]
            for j,g in enumerate(GROUPS):
                mask=np.ones(len(top),bool) if j==0 else p['groups']==j-1
                r=metrics[g];br=base[split][g]
                rescue=mask&e['coverage']&~old['coverage'];lost=mask&~e['coverage']&old['coverage']
                if int(e['coverage'][mask].sum())!=r['good']:raise ValueError('GROUP_GOOD_RECOUNT')
                if arm!='B0':
                    q=paired[arm][split][g]
                    if q['rescue']!=int(rescue.sum()) or q['lost']!=int(lost.sum()) or q['rescue_ids']!=p['ids'][rescue].tolist() or q['lost_ids']!=p['ids'][lost].tolist():raise ValueError('PAIRED_EVENTS_MISMATCH')
                rows.append(dict(arm=arm,split=split,group=g,n=int(mask.sum()),target_tracks=r['target_tracks'],good=r['good'],
                    coverage_percent=100*r['coverage'],coverage_delta_pp=100*(r['coverage']-br['coverage']),
                    minADE=r['minADE'],ADE_relative_change_percent=100*(r['minADE']/br['minADE']-1),
                    minFDE=r['minFDE'],FDE_relative_change_percent=100*(r['minFDE']/br['minFDE']-1),
                    rescued=int(rescue.sum()),lost=int(lost.sum()),reference_good=br['good'],
                    lost_reference_good_percent=100*lost.sum()/br['good'],top1_good=int(topgood[mask].sum()),
                    top1_conditional_recall_percent=100*topgood[mask].sum()/r['good'],
                    distinct=r['distinct'],effective_good=r['effective_good']))
            for i in np.flatnonzero(e['coverage']!=old['coverage']):
                events.append(dict(arm=arm,split=split,id=str(p['ids'][i]),group=int(p['groups'][i]),event='rescue' if e['coverage'][i] else 'lost',
                    reference_minJ=float(old['J'][i].min()),method_minJ=float(e['J'][i].min()),
                    reference_margin=float(1-old['J'][i].min()),deltaJ=float(e['J'][i].min()-old['J'][i].min()),
                    reference_Jslot=int(old['J_winner'][i]),method_Jslot=int(e['J_winner'][i])))
        if arm=='B0':continue
        d=read(R/'runs'/arm/'done.json');summary=read(R/'runs'/arm/'EPOCH_SUMMARY.json');res=read(R/'runs'/arm/'RESOURCE.json')
        if len(summary)!=8 or d['updates']!=2720:raise ValueError('RUN_ACCOUNTING')
        with (R/'runs'/arm/'history.csv').open(encoding='utf-8-sig') as f:history=list(csv.DictReader(f))
        if len(history)!=2720 or any(sum(int(x['epoch'])==j for x in history)!=340 for j in range(1,9)):raise ValueError('HISTORY_ACCOUNTING')
        for t in screens[arm]['tail_comparisons']:
            tails.append(dict(arm=arm,**t,magnitude_harm_percent=100*(t['positive_excess_p95']/t['reference_positive_excess_p95']-1) if t['reference_positive_excess_p95']>0 else None))
        resources.append(dict(arm=arm,private_peak=res['snapshot']['peak_private'],RSS_peak=res['snapshot']['peak_rss'],
            warning_observations=sum(bool(x.get('warnings')) for x in res['observations']),retained_observations=len(res['observations']),
            stop_observations=sum(x['status']!='PASS' for x in res['observations']),not_whole_tree_peak=True))
        run_times.append(dict(arm=arm,total_science_save_and_mechanism_seconds=d['elapsed_seconds'],
            summed_epoch_train_and_checkpoint_seconds=sum(x['elapsed_seconds'] for x in summary),epoch1_loss=summary[0]['mean_train_loss'],epoch8_loss=summary[-1]['mean_train_loss']))
    controls=read(R/'PAIRED_MECHANISM_CONTROLS.json');perturb={}
    for arm in ('U-GRU','U-Delta'):
        x=read(R/'mechanisms'/arm/'PERTURBATION.json');perturb[arm]={}
        for label,values in x['responses'].items():
            perturb[arm][label]=[dict(sign=v['sign'],context_norm_per_m_mean=float(np.mean(v['context_norm_per_m'])),
                anchor_heading_abs_rad_mean=float(np.mean([abs(z) for row in v['anchor_heading_change_rad'] for z in row if z is not None]))) for v in values]
    seconds=(datetime.fromisoformat(exit['finished_at'])-datetime.fromisoformat(exit['started_at'])).total_seconds()
    pre=read(R/'PRELAUNCH_REVIEW.json');parts=pre['parts']
    measured_pre=parts['actual_all4_preflight']+parts['actual_all4_raw_cost']+parts['actual_full_epoch_and_mechanism_regression']
    analysis=dict(status='ANALYZED_FOUR_PILOTS_NO_FULL_SCREEN_PASS',models=4,epochs=32,updates=10880,
        source_count=len(read(R/'source_manifest.json')['files']),unique_source_output_SHA_verified=len(files),
        worker_actual_seconds=seconds,planning_seconds=pre['planning_seconds'],
        measured_preflight_cost_regression_plus_worker_seconds=measured_pre+seconds,
        full_workflow_budget_deviation=dict(limit_seconds=1800,actual_measured_components_seconds=measured_pre+seconds,
            explanation='Worker alone below30min, but prelaunch contract included actual numeric preflight/cost/full-epoch regression in full budget. Their measured sum plus worker exceeds30min. Preserve admission estimate as historical; disclose estimation deviation, no claim of whole-workflow budget compliance. No rerun to clean record.'),
        results=rows,controls=controls,resources=resources,run_timing=run_times,perturbation=perturb,
        screens=screens,no_scientific_reexecution=True,no_completed_PID_task_queries=True,
        DA1='Pre-existing prelaunch review and explicit approved64/clip5/three-stage/D-overage contract retained, not retrospectively changed.',
        DA2='No cherry-picking: UDelta overall net+25 mainly stable+31 while change-10; DDiff effective-good rises mainly stable but coverage falls3 overall. DGate modest error gains exceed no significance criterion. Keep all rescue/lost IDs and physical-tail failures.',
        DA3='No full-screen arm; relative tail magnitude failures exist even excluding unestimable zero-jerk cells. Do not attribute stop only to 0/0 handling, infer permanent zero gradients, or promote V/F on tiny single-seed gains. Extra capacity/optimization and signed-attention reweighting are alternatives to mechanism explanation. Full workflow timing estimate also exceeded budget when measured preflight components included.',
        statistical_scope='One seed42 per arm, same dev4753 correlated windows/248tracks, change960/102tracks. Descriptive paired points only; no p values or confirmatory CIs. No new independent whole-scene/generalization or causal-intent claim.',
        next='Do not automatically add seeds/V/F or rename better control as candidate. Preserve B0 development comparator; original baseline/historycycle11/long-term goal unchanged.')
    csvout(O/'ALL_GROUPS_REVIEWED.csv',rows);csvout(O/'SUPPORT_EXCHANGE.csv',events);csvout(O/'TAIL_COMPARISONS.csv',tails)
    write(O/'ANALYSIS.json',analysis)
    table=['|模型|总体ADE(m)|总体FDE(m)|总体覆盖|change ADE(m)|change覆盖|change Top1条件召回|','|---|---:|---:|---:|---:|---:|---:|']
    for arm in ('B0',)+ARMS:
        o=next(x for x in rows if x['arm']==arm and x['split']=='dev' and x['group']=='overall');c=next(x for x in rows if x['arm']==arm and x['split']=='dev' and x['group']=='change')
        table.append(f"|{arm}|{o['minADE']:.6f}|{o['minFDE']:.6f}|{o['good']}/{o['n']} ({o['coverage_percent']:.3f}%)|{c['minADE']:.6f}|{c['good']}/{c['n']} ({c['coverage_percent']:.3f}%)|{c['top1_conditional_recall_percent']:.3f}%|")
    text=['# U/D四个单因素pilot：保存证据复核分析','',
        '结论：执行与数值审核通过，但四臂均未通过完整pilot门。U-Delta没有证明新息更新优于原B0；D有小幅误差线索，但差分版本没有建立对普通门控的总体优势，不自动推进三seed/V/F。',
        '',f'四fresh seed42固定8轮，共32epoch/10880更新。worker实耗{seconds/60:.3f}分钟，原规划{pre["planning_seconds"]/60:.3f}分钟；数值预检/成本/额外回归加worker实测合计{(measured_pre+seconds)/60:.3f}分钟，超过上位全流程30分钟预算估计。该偏差如实保留，不将worker单独未超30分钟冒称全流程达标。',
        '',*table,'',
        '## 1. U：新增分支可训练，但没有兑现change改善','',
        'U-Delta总体净多25成功窗，其中stable净多31、transition多4、change反而少10；change原350变340。相对B0总体ADE恶化约0.54%、change ADE恶化约0.22%。相对GRU总体ADE略好，但change更差、少11成功窗。不能用stable增益代表机动预测改善，也不能把这次阴性归为永久零梯度；三阶段实测与恢复门已经通过。',
        'U-Delta dev观测机动/非机动的innovation mean约1.808/1.794，beta约.38695/.38876；没有显现更强的机动写入均值。这是固定分层下的描述，不证明门完全没学或否定所有新息结构。GRU和Delta的状态范数不是同一量纲指标，不能直接比大小宣胜。',
        '', '## 2. D：普通门控是较温和的线索，差分特异优势未立','',
        'D-Gate总体ADE约下降0.25%、change ADE约下降0.41%，总体覆盖净多16、change仅多2。D-DiffGate change ADE约下降0.47%、change多3，但总体ADE略恶化、总体少3、stable少11。与普通Gate直接比，差分总体ADE更差约0.35%、总体少19个成功窗，change只多1个；不能挑change微小优势称差分机制被验证。',
        'B0 change条件Top1召回106/350=30.29%；D-Gate94/352=26.70%；D-DiffGate102/353=28.90%。候选交互并未在这次pilot解决排序漏选。不同模型已覆盖集合不同，条件比率与无条件Top1成功数同时保留。',
        'D-DiffGate总体effective-good约.549高于B0约.493，但change约.424低于B0约.430，且总体覆盖下降。更多好候选主要集中于已较容易的stable人口，不等于新窗口成功或语义模式理解。固定128注意力负权约1.588%、行和约.5，只证明实际启用差分算子，不证明去噪/互补因果。',
        '', '## 3. 净增不能隐藏旧成功丢失','',
        '相对B0总体 rescue/lost：U-GRU125/112、U-Delta122/97、D-Gate98/82、D-DiffGate71/74。change分别25/24、19/29、20/18、15/12。所有旧成功分母及逐ID的J裕量伤害保存在SUPPORT_EXCHANGE.csv和ALL_GROUPS_REVIEWED.csv。此处参考是匹配的新B0，不将其伪称旧native同人口。',
        '', '## 4. 尾部门没有通过，不是只有零分母阻碍','',
        '所有arm的超阈率增加均在+1个百分点范围，但正超出量p95有真实恶化：D-Gate stable速度约+2.58%；D-DiffGate stable速度约+2.84%、overall速度约+1.50%、overall接续jerk约+1.59%、change接续acc约+1.23%，均超过1%幅度门。U还存在更大的acc正超出幅度恶化。即使将无正jerk样本的NOT_ESTIMABLE单元暂不用于判定，也不能得出尾部门通过。',
        'B0若没有正超出样本，正超出条件p95相对比无法估计；保持NOT_ESTIMABLE_NO_FLOOR，不把0/0补成PASS，也不把它称成实际出现运动违规。fit-GT阈值是经验参照，不是生理硬上限。完整五pooled与junction各组表保留。',
        '', '## ARS DA2 / DA3与统计边界（同一助手角色，非外审）','',analysis['DA2'],analysis['DA3'],analysis['statistical_scope'],
        '','下一步：本请求只分析，不新增训练。没有足够证据按原门自动补43/44或进入V/F。D-Gate可保留为弱线索，但不能事后将对照改名主方法；U和差分D的具体假说本次未获支持，不外推所有运动状态/协同方法无效。所有旧模型、代码、失败、日志和自动记录保持，baseline不升级。',
        '',f'复核{len(files)}个去重来源/输出SHA全部一致；全部8科学人口保存算术/原完整complete-link+pdist/运动双实现收据齐。此次只读已保存数据、重数事件与核SHA，不训练、不全模型/梯度回放、不重复全聚簇，也不查已完成PID/task。']
    with (R/'RESULT_ANALYSIS_REVIEWED.md').open('x',encoding='utf-8') as f:f.write('\n'.join(text)+'\n')
    outputs=[O/n for n in ('ANALYSIS.json','ALL_GROUPS_REVIEWED.csv','SUPPORT_EXCHANGE.csv','TAIL_COMPARISONS.csv')]+[R/'RESULT_ANALYSIS_REVIEWED.md']
    write(O/'REVIEW.json',dict(status='PASS_SOURCE_OUTPUT_SHA_AND_PAIRED_SAVED_EVENT_RECOUNT',source_manifest_count=121,verified_unique_files=len(files),
        source_extension={str(Path(__file__).resolve()):sha(__file__)},files={str(p):sha(p) for p in outputs},
        no_model_forward_backward=True,no_training=True,no_full_cluster_reexecution=True,same_assistant_not_external_review=True))
    print(json.dumps(dict(status='ANALYSIS_REVIEW_PASS',verified=len(files),worker_minutes=seconds/60,measured_full_components_minutes=(measured_pre+seconds)/60,resources=resources,dev=[x for x in rows if x['split']=='dev' and x['group'] in ('overall','change')]),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
