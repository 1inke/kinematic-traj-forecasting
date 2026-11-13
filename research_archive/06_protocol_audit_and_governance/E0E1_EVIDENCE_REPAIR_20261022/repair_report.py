"""P1 correction table and P3 report, only after actual frozen/review evidence."""
from common import *

def main():
    report=read_json(HERE/'frozen_eval/checkpoint_replay_manifest.json');review=read_json(HERE/'completion_review/REVIEW.json')
    if report['status']!='SUCCESS_FROZEN22' or review['status']!='PASS_CURRENT_SAVED_ARITHMETIC':raise RuntimeError('FINAL_REVIEW_REQUIRED')
    original=ROOT/'qoder_work/互补蒸馏_E0E1_实验分析报告_20261020.md';e0stats=read_json(HERE/'E0_STATISTICS_CORRECTED.json');rows=[]
    corrections=[
      ('C01','干净的+8pp上界已证明','现有有限native/KIN池上的GT贪心可达见证；cal_v11只隔离当前划分，native上游仍暴露。','E0_STATISTICS_CORRECTED.json;EXPOSURE_AUDIT.csv','COVERAGE_RECOMPUTED_SOURCE_BOUNDED','底座逐checkpoint同期cache生产来源不完整'),
      ('C02','K0-C是数学最优上界','K0-C没有最优性证明；只能称已找到可行见证，非所有可能学生上界。','original k0/k0c_greedy.py','SOURCE_METHOD_CHECK','无全局最优性证书'),
      ('C03','clean/dropped增益接近证明污染无影响','两组组成不匹配、不随机化，不作污染因果解释。','E0_STATISTICS_CORRECTED.json','DESIGN_BOUNDARY','不能从两组点估计消除混杂'),
      ('C04','三seed原CI[6.47,10.15]直接联合引用','共享track固定三seed95%[5.8411,10.9568]pp；paired-seed敏感性[5.5556,11.2710]pp。','E0_STATISTICS_CORRECTED.json','20K_SHARED_TRACK_POSTHOC_RECOMPUTE','事后探索非新确认性'),
      ('C05','seed42全部筛选门满足后晋级','E42运动学FAIL，后续43/44是筛选失败后的探索追加，不是合规成功晋级。','frozen_eval/checkpoint_replay_manifest.json;original DESIGN.md','OLD_JSON_AND_FROZEN_REPLAY','原DESIGN为DRAFT，未见同期锁源'),
      ('C06','学生容量不足已证实','当前history-only、57432参数、8epoch配置未可靠成功；信息/优化/容量/预算未分离。','frozen_eval/assignment_stats.json;original models.py','SOURCE_AND_FROZEN_FINAL_DIAGNOSTICS','没有收敛或容量上限证明'),
      ('C07','26%失败窗口可教','按窗口而非路径计数：1943/8051、1952/8051、1899/8051=24.13/24.25/23.59%。','frozen_eval/assignment_stats.json','FROZEN_TEACHER_SUBSET_RECOMPUTE','有限稀疏线索非唯一根因'),
      ('C08','B/E仅改变是否显式分解','B额外包含native成功窗KIN好教师；E两组各均值相加，也改变目标集合及尺度。','frozen_eval/assignment_stats.json;ALGEBRA_ALL_PARAMETER_FIXTURE.json','SOURCE_AND_ALL_PARAMETER_IDENTITY_FIXTURE','新重加权训练需另冻结，不能把恒等式当新机制'),
      ('C09','输入全锁源且能完整续训','当前补SHA不能追溯旧训练锁源；旧state_dict/arm/seed不含optimizer/全部RNG。','CACHE_LINEAGE.json;RUN_INVENTORY.csv','POSTHOC_INVENTORY','旧权重只用于冻结评价'),
      ('C10','旧FATAL表示当前全部训练失败','旧FIT-local索引错误与后续11权重分列；本轮也保留自己的数值/reader失败及修复。','original FATAL.txt;IMPLEMENTATION_TRANSITION_V2.json;IMPLEMENTATION_TRANSITION_V3.json','TIME_AND_SOURCE_SEPARATED','没有删除失败或改原结果'),
      ('C11','原W=True等于本轮全部独立核验','本轮E0只复核覆盖/索引/关联CI；P2学生22套完整聚簇五motion已独立审核，两范围不混。','E0_STATISTICS_CORRECTED.json;completion_review/REVIEW.json','EXPLICIT_REVIEW_SCOPE','E0全W未新增完整审核')]
    for cid,claim,corrected,evidence,level,unresolved in corrections:rows.append(dict(claim_id=cid,original_claim=claim,corrected_claim=corrected,evidence_path=evidence,verification_level=level,unresolved=unresolved))
    write_csv(HERE/'CORRECTIONS.csv',rows)
    records=report['records'];by={(r['seed'],r['arm'],r['split']):r for r in records};assign=read_json(HERE/'frozen_eval/assignment_stats.json');table=[];subtable=[];traits=[]
    for r in records:
        m=r['metrics'];a=assign[r['job']];table.append(f"|{r['arm']}/{r['seed']}|{r['split']}|{m['overall']['good']}/{m['overall']['n']}|{m['change']['good']}/{m['change']['n']}|{m['change']['rescue']}/{m['change']['lost']}|{'PASS' if m['change']['motion_pass'] else 'FAIL'}|")
        for group in ['overall','change']:
            ns=a['subsets']['native_success']['groups'][group];fail=a['subsets']['native_failure_with_good_KIN']['groups'][group];nt=a['subsets']['native_failure_without_good_KIN']['groups'][group];subtable.append(dict(job=r['job'],group=group,native_success_N=ns['N'],retained=ns['good'],lost=ns['lost'],lost_rate=ns['lost']/ns['N'] if ns['N'] else None,teachable_failure_N=fail['N'],teachable_rescued=fail['good'],teachable_completion_rate=fail['coverage'],no_teacher_failure_N=nt['N'],no_teacher_rescue=nt['good']))
    write_csv(HERE/'SUBSET_COMPARISONS.csv',subtable)
    for split in ['FIT','inner']:
        for seed in SEEDS:
            for contrast,a,b in [('E-B','E','B'),('E-A','E','A'),('B-A','B','A')]:
                ta=by[(seed,a,split)]['metrics'];tb=by[(seed,b,split)]['metrics'];aa=assign[f'{a}_seed{seed}_{split}']['subsets'];ab=assign[f'{b}_seed{seed}_{split}']['subsets']
                traits.append(dict(split=split,seed=seed,contrast=contrast,overall_coverage_delta_pp=100*(ta['overall']['coverage']-tb['overall']['coverage']),change_coverage_delta_pp=100*(ta['change']['coverage']-tb['change']['coverage']),change_lost_delta=ta['change']['lost']-tb['change']['lost'],change_teachable_completion_delta_pp=100*(aa['native_failure_with_good_KIN']['groups']['change']['coverage']-ab['native_failure_with_good_KIN']['groups']['change']['coverage'])))
    write_json(HERE/'DESCRIPTIVE_CONTRASTS.json',dict(status='DESCRIPTIVE_NOT_NEW_HYPOTHESIS_TEST',contrasts=traits));write_csv(HERE/'DESCRIPTIVE_CONTRASTS.csv',traits)
    lineage=read_json(HERE/'CACHE_LINEAGE.json');pre=read_json(HERE/'PREFLIGHT.json')
    for entry in lineage['entries']:
        match=next((r for r in pre['KIN_complete_cache_row_replay'] if r['path']==entry['artifact_path']),None)
        if match:entry.update(replay_verified=True,numeric_row_mapping_evidence=match,original_historical_lock_not_repaired=True)
    write_json(HERE/'CACHE_LINEAGE_VERIFIED.json',lineage)
    text=f'''# E0/E1 修订报告与冻结补评（2026-10-22）

本报告纠正原总报告，不覆盖历史文件。原文SHA256：`{sha256(original)}`。逐条纠错与来源见CORRECTIONS.csv、CACHE_LINEAGE_VERIFIED.json、EXPOSURE_AUDIT.csv。

## 事实一：E0仍是机会见证，不是数学上界或无污染证明

cal_v11 change443窗native146→182/180/186、lost0；dropped257窗native97→117/117/116、lost0。均值+8.2769pp。共享50track固定3seed95%CI[{e0stats['shared_track_fixed_seed_CI_pp'][0]:.4f},{e0stats['shared_track_fixed_seed_CI_pp'][1]:.4f}]pp；共同track+paired-seed敏感性[{e0stats['shared_track_paired_seed_CI_pp'][0]:.4f},{e0stats['shared_track_paired_seed_CI_pp'][1]:.4f}]pp。20k原RNG约定保留，均为事后探索。

当前划分v1.1确获作者批准，不再混同旧“尚无v1.1”；但h3/v2实际split均见cal_v11全部1620身份/时间窗。新划分不会重训或抹去底座暴露。E0实际CPV文件与原输入清单的另处同名文件分列；本轮完整9套KIN缓存数值行重放最大{max(r['numeric_full_row_replay_maxdiff'] for r in pre['KIN_complete_cache_row_replay']):.9g}，当前映射可用，不替代同期训练/缓存锁源。E0没有本轮全W/聚簇审核，原W=True保留为原报告层。

## 事实二：11已有学生×FIT/inner的最终冻结结果

|arm/seed|人口|overall good/N|change good/N|change rescue/lost|change五motion|
|---|---|---:|---:|---:|---|
'''+ '\n'.join(table)+f'''

所有22个checkpoint×split都严格加载57432参数，全部参数与buffer前后EXACT，eval/inference-mode，无optimizer或参数更新。inner计数全部对齐原JSON；seed42未存native_good_count，明确用good−rescue+lost恒等式交叉核对，不伪称旧字段存在。差异细目见REPLAY_RECONCILIATION.json。

保存的同candidate质量、完整K20 GT-free聚簇、五pooledmotion已由独立NumPy/SciPy入口核验；额外全部逐窗标量、教师/槽责任、原生成功/可教失败/不可教失败子集和最终人口loss分项另有CPU第二实现审核。当前审核max_abs_diff={review['max_abs_diff']:.9g}，旧输入前后SHA相同。不是历史训练源追溯认证，也不是外部审稿。

## 口径与边界

WTA与教师最近匹配分开，责任是最终模型后验，不是历epoch使用率。每slot成功数与distinct不等于有效互补。minADE/minFDE可来自不同slot，good/J和分段要求同完整candidate。运动学12v/11a/10j、dt.4、linear pooled p95、literal1.01；未对学生做事后安全过滤。原总loss history不可倒推出逐epoch coverage，最终全人口teacher比值也不等于训练batch均值。

B/E原教师集合与归一化均不同，因此不能纯归因“是否分解”。CPU合成全参数fixture证明同teacher同权重union/decomposed标量及全部参数梯度等价；这不是值得再花训练预算的独立机制。新的R/N组间重加权会改变优化几何，不是恒等式。

## 推断（不是唯一根因）

当前history-only学生未使用F16交互表征；训练8epoch也没有收敛证明。将“57k容量不足×监督稀疏”写成已证根因是不成立的。须同时检查FIT拟合、inner迁移、保留/补全权衡及运动代价；完整描述性contrast和子集分母已保存，不只引seed42change最好结果。

## 未证明与后续

未证明未知场景/whole-model未见泛化、运动理解、数学最优上界、学生能力上限或新baseline。此阶段结束于P3决策，不自动训练P4。若后续选择组间权重实验，必须另冻结同信息/同teacher/同总lambda/同预算与CI家族；不得同时加宽、过采样、换表征或加物理项。
'''
    (HERE/'REPORT_REVISED.md').write_text(text,encoding='utf-8')
    print('P1 revised report,11 corrections,subset denominators and descriptive contrasts saved.')

if __name__=='__main__':main()
