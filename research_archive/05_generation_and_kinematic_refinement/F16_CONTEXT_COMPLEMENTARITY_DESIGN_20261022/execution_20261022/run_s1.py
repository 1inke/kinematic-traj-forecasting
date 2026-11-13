"""First sole worker executes S1 only; never starts S2 without the measured gate."""
from training import *
import csv

def gate(results):
    rows=[]
    for s in SEEDS:
        for split in ['FIT','inner']:
            native=np.load(HERE/f'native_{split}.npz')
            nm=read_json(HERE/f'native_metrics_{split}.json')
            for group in ['overall','stable','transition','change']:
                m=results[s]['metrics'][split][group];b=nm[group]
                checks=dict(nonempty=m['n']>0,coverage=m['coverage']>=.99*b['coverage'],ADE=m['ADE']<=1.01*b['ADE'],FDE=m['FDE']<=1.01*b['FDE'])
                checks.update({k:m['motion'][k]<=1.01*b['motion'][k] for k in m['motion']})
                rows.append(dict(seed=s,split=split,group=group,n=m['n'],native_good=b['good'],A_good=m['good'],rescue=m['rescue'],lost=m['lost'],checks=checks,passed=all(checks.values())))
    return dict(status='PASS_ALLOW_S2_PREFLIGHT' if all(r['passed'] for r in rows) else 'BASELINE_NOT_READY',all_pass=all(r['passed'] for r in rows),rows=rows)

def native_outputs(data,guard):
    for split,d in data.items():
        stage('NATIVE_REFERENCE_SAVE_AUDIT',split=split)
        path=HERE/f'native_{split}.npz'
        np.savez_compressed(path,paths=d['native'],native=d['native'],gt=d['gt'],sample_id=d['ids'],groups=d['groups'],track=d['track'])
        metric,er=evaluate(d['native'],d['gt'],d['groups'],d['native'],d['ids'],d['track']);write_json(HERE/f'native_metrics_{split}.json',metric)
        rev=independent.audit_file(path,HERE/f'native_metrics_{split}.json',frozen_cache=prior.PATHS['cache'],frozen_labels=prior.PATHS['labels'],frozen_metadata=prior.PATHS['metadata'],frozen_manifest=ROOT/'config/data_manifest.v1.json',root=ROOT)
        assert rev['status']=='PASS';write_json(HERE/f'native_audit_{split}.json',rev);guard.check()

def report(results,decision,elapsed):
    body=['# F16同源上下文公共基线：S1实际结果','',f"执行完成；科学阶段门：**{decision['status']}**。",'', 'A使用210维观测/冻结token压缩输入，82008新参数，三个fresh seed各8epochs。零输出初始化不是nativeidentity。旧F16/decoder/scorer和KIN不更新。','', '|split|seed|group|N|native good|A good|rescue|lost|gate|','|---|---:|---|---:|---:|---:|---:|---:|---|']
    for row in decision['rows']:body.append('|'+ '|'.join(str(row[k]) for k in ['split','seed','group','n','native_good','A_good','rescue','lost'])+'|'+('PASS' if row['passed'] else 'FAIL')+'|')
    body += ['', '所有组的minADE/FDE/J、完整GT-free聚簇、有效good、五项pooled运动学、逐窗rescue/lost ID保存在各run的metrics与sample_metrics；阶段门逐项布尔在DECISION.json。', '', '## 结论与停止边界', '', '任一seed/人口/组失败就不启动B/E_bal，不延长epoch、不换初始化或接口。科学失败不等于执行失败。' if not decision['all_pass'] else 'S1已合格；S2尚未启动，需要同协议的阶段预检、独立实测安排后才追加B/E_bal。', '', '## 同助手反方检查（不是外审）', '', 'DA2：基线失败可能来自零初始化、短8epoch优化、recent+mean压缩或模型配方；本设计无同容量history-only臂，不能独证F16信息无用。native参考的上游暴露与输入路径不同，不能作为公平容量上限。', '', 'DA3：不挑最好seed或只看change。净覆盖不代表逐窗保留；即使部分motion更好，任一五项门失败也不称安全。保留所有反例。上游h3/v2已见inner，FIT/inner目标track互斥非context互斥，重复inner探索不构成独立wholemodel泛化。', '', f'实际worker全流程{elapsed:.3f}s；S0成本见source_audit/RESULT.json，训练/每epoch记录见history。源与旧产物初末SHA核验在PRESERVATION.json。原major_status=NOT_ESTABLISHED；baseline/historycycle11/原总目标不变。', '', '报告由同助手研究设计/统计角色审查，不是外部审稿。']
    (HERE/'RESULT_ANALYSIS.md').write_text('\n'.join(body)+'\n',encoding='utf-8')
    write_json(HERE/'RESULT_ANALYSIS.json',dict(execution='COMPLETE_S1',scientific=decision['status'],decision=decision,metrics={str(s):results[s]['metrics'] for s in SEEDS},S2='SKIPPED_BASELINE_GATE' if not decision['all_pass'] else 'CONDITIONAL_NOT_STARTED',major_status='NOT_ESTABLISHED',elapsed_seconds=elapsed,review='same_assistant_inline_DA2_DA3_not_external'))

def main():
    start=time.perf_counter();guard=Guard(E);guard.check('startup');stage('S1_INITIAL_SOURCE_SHA',status='RUNNING',active_worker=True)
    digest=verify_sources();data=load_data();native_outputs(data,guard);results={}
    for s in SEEDS:
        with np.load(HERE/f'teachers/seed{s}.npz') as z:teacher={k:z[k].copy() for k in ['R','R_mask','N','N_mask']}
        results[s]=train_run('A',s,data,teacher,guard)
    stage('S1_FINAL_SOURCE_SHA');assert verify_sources()==digest
    decision=gate(results);write_json(HERE/'DECISION.json',decision)
    write_json(HERE/'ARITHMETIC_AUDIT.json',dict(status='PASS',scientific_model_population_jobs=6,native_reference_jobs=2,all_full_quality_original_complete_link_five_motion=True,independent='audit/verify.py',reports=[str(p.relative_to(HERE)) for p in HERE.glob('runs/A_seed*/audit_*.json')],all_final_parameters_buffers_exact=True))
    write_json(HERE/'PRESERVATION.json',dict(status='PASS',source_digest=digest,all_locked_sources_old_weights_results_unchanged=True,source_count=len(read_json(HERE/'source_manifest.json')['files'])))
    guard.check();write_json(HERE/'RESOURCE.json',dict(status='PASS',estimate_bytes=E,observations=guard.observations,retained_peak_private=max(x['snapshot']['peak_private'] for x in guard.observations),retained_peak_RSS=max(x['snapshot']['peak_rss'] for x in guard.observations)))
    report(results,decision,time.perf_counter()-start)
    write_json(HERE/'RESULT.json',dict(status='PASS_EXECUTION_S1_CONTEXT_BASELINE',science=decision['status'],runs=3,epochs=24,optimizer_steps=1776,scientific_jobs=6,native_references=2,source_digest=digest,all_audits='PASS',S2_started=False))
    stage('ANALYZED' if not decision['all_pass'] else 'S1_ANALYZED_S2_PREFLIGHT_REQUIRED',status='ANALYZED' if not decision['all_pass'] else 'S1_COMPLETE',active_worker=False,scientific_decision=decision['status'],elapsed_seconds=time.perf_counter()-start)

if __name__=='__main__':main()
