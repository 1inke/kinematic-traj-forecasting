"""Bounded final saved-data report and identity fractions, no Torch/forward."""
import sys
from pathlib import Path
H=Path(__file__).resolve().parents[1];sys.path.insert(0,str(H))
import common as c
import numpy as np

def main():
    s=c.read(H/'completion_review/SUMMARY.json');rows=s['rows'];result=c.read(H/'RESULT.json');stats=c.read(H/'STATISTICS.json');ident=[]
    for key,row in c.read(H/'JOBS.json').items():
        if key.startswith('native/'):continue
        c.guard()
        with np.load(row['output'],allow_pickle=False) as z:
            same=z['paths']==z['native'];win=same.all((1,2,3));slots=same.all((2,3));groups=z['groups']
        for group,mask in c.masks(groups):ident.append(dict(job=key,group=group,n=int(mask.sum()),window_native_exact_fraction=float(win[mask].mean()),candidate_native_exact_fraction=float(slots[mask].mean())))
    c.write(H/'completion_review/IDENTITY_FRACTIONS.json',dict(status='PASS',records=ident,no_model_forward=True))
    change=[r for r in rows if r['group']=='change'];bouter=[r for r in change if r['cell']=='B' and r['split']=='outer'];boutmean=float(np.mean([(r['good']-r['native_good'])/r['n'] for r in bouter]));resource=s['resource_deviation'];major=False
    decision=dict(status='T1_RETENTION_NOT_ESTABLISHED_RESOURCE_REVIEW_REQUIRED',numerical_completion=True,clean_execution_contract=False,RETENTION_SIGNAL=False,T2_scientifically_independent_but_resource_blocked=True,T2_started=False,T3_NOT_TRIGGERED=True,major_success=False,strict_formal_completed_counter=0,formal_attempts=1,completed_training_runs=6,automatic_next_action=False,human_decision_required='Approve a versioned guard correction/re-estimate based on measured footprint before T2; never rerun T1 or alter old RESOURCE/SHA.',not_unique_loss_or_head_cause=True)
    c.write(H/'NEXT_BRANCH.json',decision)
    header=['# T1结果：保留有局部改善，未达到预注册迁移信号；资源合同存在偏差','',
      '分析日期：2026-10-16，日本标准时间。六个fresh训练已结束；本报告不把计算SUCCESS等同于模型成功或全部执行合同通过。',
      '', '## 1. 执行与证据范围','',
      'worker与exit均SUCCESS：12:30:48.489445→12:59:25.782480，实际28分37.293秒。13:08:19.1928605单次完成核验确认进程已结束、任务Ready、XML0trigger、stderr0；不再查询该PID。',
      '', 'A/B各3seed、每run25epoch，共6模型/150history/11100更新；24科学jobs＋4native参考，84主要保存输出。原完整ARITHMETIC_AUDIT对全28job质量、原GT-free complete-link聚簇、五pooledmotion、方向/分层、keep、controls及全部统计抽样独立复算PASS。2304sources初末SHA不变。',
      '', '补充completion_review只做CPU/NumPy：全28job保存质量与计数max0、全原源和完成输出SHA前后验证；12个A科学job的所有保存数组与历史U0 coupled bitwiseEXACT。新鲜重复执行不构成独立泛化，未重放所有旧模型/梯度，也不是外部评审。完整96组/全部ID清单见completion_review/SUMMARY.json。',
      '', '## 2. 原生变化组完整好候选覆盖','',
      '标准固定为同一候选ADE≤0.2 AND FDE≤0.4；三个数依次seed42/43/44。不同split分母不能混用。','',
      '|人口|N|原生good|A good|B good|A rescue/lost|B rescue/lost|','|---|---:|---:|---|---|---|---|']
    for split in c.SIZES:
        aa=[r for r in change if r['cell']=='A' and r['split']==split];bb=[r for r in change if r['cell']=='B' and r['split']==split]
        aa.sort(key=lambda r:r['seed']);bb.sort(key=lambda r:r['seed']);header.append(f"|{split}|{aa[0]['n']}|{aa[0]['native_good']}|"+'/'.join(str(r['good']) for r in aa)+'|'+'/'.join(str(r['good']) for r in bb)+'|'+'; '.join(f"{r['rescue']}/{r['lost']}" for r in aa)+'|'+'; '.join(f"{r['rescue']}/{r['lost']}" for r in bb)+'|')
    header += ['', '## 3. Primary outer-change机制门','', '固定family24，即便C/D未执行也不缩家族；20000 PCG64(seed20261016)同ID paired track×3seed抽样，全部primary有效draw20000。普通95%不是替代门。','', '|B−A指标|点估计|family24同时区间|','|---|---:|---|']
    names=['coverage差(pp)','lost率差(pp，分母原生成功110)','rescue率差(pp，分母原生失败143)','固定native-j0实际ΔJ差']
    for i,r in enumerate(stats['primary']['rows']):
        scale=100 if i<3 else 1;header.append(f"|{names[i]}|{r['estimate']*scale:.6f}|[{r['CI_family'][0]*scale:.6f}, {r['CI_family'][1]*scale:.6f}]|")
    header += ['', 'RETENTION_SIGNAL未通过：lost平均只降2.1212pp，未达3pp；三个seed分别降4.5455/1.8182/0pp，不是三seed严格下降。同时区间跨0；rescue均值略降且非劣区间下界约−8.48pp低于−2pp门；coverage第三seed下降。不能把宽区间说成“保留项已证明完全无效”，也不能据点估计宣传已迁移。',
      '', 'B对native绝对coverage平均仅+0.131752pp，family3区间[−4.741778,+4.515928]pp；不是每seed≥+5pp。ADE/FDE/minJ的family9差区间均跨0，无≥5%误差改进证据。因此无重大模型成功；原baseline/historycycle11/goal不升级。',
      '', '## 4. 训练内与迁移、保护—救回的交换','',
      'FIT change的B把lost由124/114/119降至78/74/67，但rescue同时由358/374/343降至290/324/295。原生失败上的生成收益被部分压缩，不能只报lost减少。FIT整体good却增加（A11657/11638/11593→B11700/11743/11722），这是反例，不能说所有训练组都变差。',
      '', 'outer change B的good112/111/108对A107/105/113：前两seed改善，第三下降。inner change169/171/174对A173/173/174更弱或相同。CAL change232/232/234全部优于A221/214/225，仍全部低于native243；CAL仅事后反例，不可升为primary或选择权重。',
      '', 'B outer change的native-success lost仍13/14/16，即11.82/12.73/14.55%，远高于1%保护门。B固定原生J最佳候选的实际ΔJ比A在三seed均改善，但其family24区间跨0，且相对native本身仍是正ΔJ（变差），不能用此替代完整coverage。',
      '', 'B outer change平均修正幅度约0.0464–0.0493m（A约0.0559–0.0568m），并非全部回退identity。精确原生相等窗口/候选比例逐组保存在IDENTITY_FRACTIONS.json；实际质量交换而非仅基于“近identity”的主观标签判断。',
      '', '## 5. 运动学与有效候选簇','',
      '所有seed/split/group五pooled指标保留literal native×1.01标准，不因当前去投影而降低要求。下表逐组列出失败项；平均加速度改善不等于全门通过。有效good簇也不能抵消完整覆盖或成功保护失败。','',
      '|cell|seed|split|group|N|good/native|R/L|有效good簇均值|失败motion|protected ADE/FDE通过|',
      '|---|---:|---|---|---:|---|---|---:|---|---|']
    for r in rows:header.append(f"|{r['cell']}|{r['seed']}|{r['split']}|{r['group']}|{r['n']}|{r['good']}/{r['native_good']}|{r['rescue']}/{r['lost']}|{r['effective_good']:.6f}|{','.join(r['motion_failure_metrics']) or 'PASS'}|{r['protected_error_pass']}|")
    bmotion=[r for r in rows if r['cell']=='B'];allfail={key:sum(key in r['motion_failure_metrics'] for r in bmotion) for key in bmotion[0]['motion']}
    header += ['', 'B运动学失败计数（48个seed×split×group，五项分别统计）：`'+str(allfail)+'`。',
      '', 'outer change B平均加速度相对native仍约+10.81%至+11.32%，A约+12.29%至+12.52%；代价减小但仍严格失败。固定少量FIT batch的L0/keep输出坐标梯度cos为正，不能称参数梯度冲突已排除，也不能当有限AdamW因果证明。训练history为preupdate加权值，不能当最终人口loss；后者保存在各job done.population。',
      '', '## 6. 资源偏差：停止下一训练而不覆盖原报告','',
      f"本任务E={resource['estimated_peak_bytes']}bytes（{resource['estimated_peak_bytes']/2**30:.6f}GiB），1.3E={resource['approved_process_limit_bytes']/2**30:.6f}GiB；原RESOURCE登记peak_private_bytes={resource['recorded_peak_private_bytes']}（{resource['recorded_peak_private_bytes']/2**30:.6f}GiB），比上限高约5.03%。原报告仍标PASS，但guard只比较当前private/RSS，累计记录的生命周期peak没有参与超限判断。",
      '', '没有保留每次当前snapshot/峰值时间，不能反推峰值发生时刻或确定具体分配来源；需核对peak口径并在新版本补足已记录峰值检查/保守估算。可用物理内存最低约13.40GiB，故不能声称电脑内存耗尽或要求用户关闭程序。',
      '', '原RESOURCE/PREFLIGHT/SHA/worker结果全部保留，不把原SUCCESS改写成失败，也不把资源合同偏差抹掉。新RESOURCE_DEVIATION_REVIEW.json明确该偏差，严格无偏差执行完成暂不记1；六训练已完成/数学分析完成另列。T2尚未实施/启动，须等直接人类确认新版本资源审核与重估，不重训T1，不静默提高E或改固定1.3倍。',
      '', '## 7. 同助手反方检查（非外审）','',
      '**DA2 — 证据与反例：** 不能因outer机制门失败就抹去CAL保留改善、FIT整体覆盖增加和fixed-slot三seed点估计改善；但也不能用这些事后人口替代primary。负面论据最强处是第三seed覆盖与救回下降、保护仍远超门及motion持续失败。',
      '', '**DA3 — 最终压力测试：** 该keep也可视为一般正则，无法独证“缺失保护loss就是根因”。三seed/重复窗口/F16已见/target非context独立/outerCAL适应性复用+无新holdout限制全部保留。资源报告PASS与峰值矛盾是执行合同重大问题，必须阻止自动T2，不因科学结果已可读就放行。',
      '', '## 8. 当前分支','',
      'T1：RETENTION_SIGNAL_NOT_ESTABLISHED；D因T1门失败不触发。T2在科学问题上仍独立可做，但当前RESOURCE_REVIEW_REQUIRED，未实施、未预检、无worker或ETA。下一次只能先在新目录核对峰值口径、修正guard/估算并保留原源，再按原C对A方案走实际GPU/恢复/预算门；不扫keep权重或新MoE/scorer。',
      '', '来源：本目录RESULT.json、ARITHMETIC_AUDIT.json、STATISTICS.json、PRESERVATION.json、RESOURCE.json、completion_review/SUMMARY.json/REVIEW.json/IDENTITY_FRACTIONS.json及各run保存路径/metrics/history；辅助分析AI执行，同助手反方审核，非外部独立评审。']
    (H/'RESULT_ANALYSIS.md').write_text('\n'.join(header)+'\n',encoding='utf-8')
    c.write(H/'RESULT_ANALYSIS.json',dict(**decision,primary=stats['primary'],absolute=stats['absolute'],gate=s['retention_gate'],rows=rows,training=s['training'],A_repeat=s['A_repeat'],resource_deviation=resource,identity=ident,inline_DA2='Preserve CAL/FIT overall/fixed-slot counterexamples; do not replace primary or claim unique loss cause.',inline_DA3='Resource peak/current-guard inconsistency blocks next launch; preserve worker SUCCESS and original RESOURCE PASS as provenance, add explicit deviation instead of editing them.',not_external_review=True))
    print('REPORT_SAVED',dict(B_outer_native_coverage_delta=boutmean,motion_failures=allfail,identity_rows=len(ident),T2_started=False,T3_triggered=False),flush=True)

if __name__=='__main__':main()
