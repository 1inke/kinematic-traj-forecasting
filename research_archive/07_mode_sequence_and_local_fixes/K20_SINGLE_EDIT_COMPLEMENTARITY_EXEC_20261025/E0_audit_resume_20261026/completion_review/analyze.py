"""Completion source/output contracts and full saved group aggregation, no model or re-enumeration."""
from pathlib import Path
import sys,csv,json
import numpy as np
HERE=Path(__file__).resolve().parent;RUN=HERE.parent
sys.path.insert(0,str(RUN));import resume
r=resume.adapter;c=resume.c
def main():
    if (RUN/'RESULT_ANALYSIS.json').exists():raise FileExistsError('Existing final report retained; use new version if needed')
    initial=r.verify();parent,done=r.parent_done()
    result=c.read_json(RUN/'RESULT.json');audit=c.read_json(RUN/'ARITHMETIC_AUDIT.json');resource=c.read_json(RUN/'RESOURCE.json')
    assert result['scientific_decision']=='E0_PASS_ALLOW_E1_OWN_PREFLIGHT' and audit['status']=='PASS_ALL_4915200_ACTIONS_AND_ORACLE'
    assert sum(v['nonSTOP_actions'] for v in audit['records'])==4915200 and len(audit['records'])==3
    assert c.read_json(RUN/'worker_state.json')['status']==c.read_json(RUN/'exit_status.json')['status']=='SUCCESS'
    registry=np.load(r.PARENT/'SPLIT_REGISTRY.npz');opt=registry['opt_row'];tracks=registry['track_key'][opt];groups=registry['group'][opt]
    population={};rows=[];checks=[]
    for seed,d in zip(c.SEEDS,done):
        npositive=0;trackpositive=[];N=0
        for file in d['chunks']:
            with np.load(r.PARENT/f'seed{seed}'/file) as z:
                keep=z['B'].any(1);start=int(z['start']);npositive+=int(keep.sum());trackpositive.extend(tracks[start:start+len(keep)][keep]);N+=len(keep)
        m=c.read_json(r.PARENT/f'seed{seed}'/'POPULATION_METRICS.json');population[str(seed)]=m;change=m['change']
        gate=dict(positive_windows=100*npositive>=5*N,positive_tracks=len(set(trackpositive))>=50,change_windows=int((groups==2).sum())>=300,change_tracks=len(set(tracks[groups==2]))>=30,oracle_change_coverage=100*(change['good']-change['native_good'])>=2*change['n'],native_good_lost_zero=all(x['lost']==0 for x in m.values()),allgroups_five_pooled_motion=all(x['motion'][k]<=1.01*x['native_motion'][k] for x in m.values() for k in c.NAMES))
        assert gate==d['gate']['checks'] and all(gate.values())
        checks.append(dict(seed=seed,checks=gate,positive_windows=npositive,positive_tracks=len(set(trackpositive)),total=N))
        for group,v in m.items():
            row=dict(seed=seed,group=group,**{k:v[k] for k in ('n','tracks','native_good','good','rescue','lost','ADE','FDE','J','distinct','effective_good')},coverage_gain_pp=100*(v['good']-v['native_good'])/v['n'])
            row.update({k+'_ratio':v['motion'][k]/v['native_motion'][k] if v['native_motion'][k] else None for k in c.NAMES});rows.append(row)
    assert r.verify()==initial;r.parent_done()
    with (HERE/'ALL12_GROUPS.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    final=dict(status='ANALYZED_E0_PASS_ALLOW_E1_OWN_PREFLIGHT',at=c.now(),scientific_decision=result['scientific_decision'],checks=checks,all_group_metrics=population,complete_action_audit=audit,source_digest=initial,parent_source_digest=parent,source_entries=650,helper_SHA=c.sha256(__file__),all_old_files_unchanged=True,no_forward_training=True,formal_training_completed=0,resource=resource,same_assistant_not_external_review=True,DA2='Oracle attainable opportunity is not observable learnability; all positives retained and all groups disclosed.',DA3='Per-window safety is not pooled safety; final12groupsx5 literal gates separately passed; no hold/inner label unlocked.',limitations=['FIT_opt only; frozen generators may have seen hold/inner','3 seeds share same windows; no independent scene confirmation','no end-to-end cost or learned editor demonstrated'],E1='AUTHORIZED_IMPLEMENTATION_AND_OWN_ACTUAL_PREFLIGHT_NEXT',E2='NOT_STARTED')
    c.write_json(RUN/'RESULT_ANALYSIS.json',final)
    lines=['# E0完成：一次编辑机会门通过','','2026-10-26。状态：ANALYZED_E0_PASS_ALLOW_E1_OWN_PREFLIGHT。原两次失败保留，修复后只恢复保存审核，没有重新枚举或训练。','','## 核验范围','','三seed×4096固定FIT_opt×400，共4,915,200替换动作逐项独立重构；所有mask/标签/oracle EXACT，数值max4.44e-16。完整K20原GT-free complete-link及独立pdist、12组五项pooled运动学、源650与全部父输出SHA/PRESERVATION/RESOURCE通过。worker/exit均SUCCESS，14:31:06.675972→14:35:02.942506，约3分56秒。生命周期private峰1,873,444,864bytes/RSS381,390,848bytes，是本恢复审核峰，不是原枚举或未来训练峰。','','## 全组结果','','|seed|组|n|native good|oracle good|净增pp|rescue|lost|五motion|','|---|---|---:|---:|---:|---:|---:|---:|---|']
    for x in rows:lines.append(f"|{x['seed']}|{x['group']}|{x['n']}|{x['native_good']}|{x['good']}|{x['coverage_gain_pp']:.4f}|{x['rescue']}|{x['lost']}|PASS|")
    lines+=['','正窗口：2292/2486/2310，涉及423/415/421条track，均超过5%和50track门。变化组977窗/216tracks满足规模门，固定oracle净增9.519%/10.747%/10.440%（百分点），均高于2pp；全部组原好lost0。未改oracle排序或逐窗口/pooled阈值。全部12组详表、Q/J单槽时段、完整救回ID与簇统计在JSON/ALL12_GROUPS.csv及父保存记录。','','## 结论与下一步','','固定native20+KIN20内，最多一次替换确实存在合格的GT知情见证，因此可进入E1自己的实现与真实预检。不是已学会选择、部署成功或整个生成问题解决；未训练I/S，也未看hold/inner标签。E1仍需324特征/FIT_opt统计、paired11488MLP、真实GPU梯度/FD、双方513窗2epoch全optimizer/RNG一致、坏合同拒绝、实际保存质量和资源吞吐门全部通过后，才能六fresh8epoch。','','## 同助手反证DA2/DA3（非外审）','','- 有GT好动作不等于观测可辨识，旧多步oracle与本次不能当严格配对因果。','- 变化组阳性不掩盖其它组；四组与每seed的全部五motion均另核，未把窗口门当pooled证明。','- 所有权修复只改变内存生命周期，原资源失败和旧来源不洗掉。650源继承与旧输出初末SHA相同。','- 只检验新编辑器的条件迁移。冻结模型既有暴露、相关窗口、3seed与反复开发集限制仍在，未做独立泛化或真实端到端成本认证。','','所有文件、模型、结果、失败与日志保留，无删除。baseline/historycycle11与未完成总目标不变。']
    with (RUN/'RESULT_ANALYSIS.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    c.write_json(HERE/'REVIEW.json',dict(status='PASS_COMPLETION_SOURCE_OUTPUT_AND_GATES',at=c.now(),source_digest=initial,full4915200_action_audit_from_worker=True,all12_group_gate_recompute=True,outputs={p.name:c.sha256(p) for p in [RUN/'RESULT_ANALYSIS.md',RUN/'RESULT_ANALYSIS.json',HERE/'ALL12_GROUPS.csv']},new_helper_SHA=c.sha256(__file__),no_new_math_or_model_forward=True))
    print(json.dumps(dict(status=final['status'],rows=len(rows),checks=checks)))
if __name__=='__main__':main()
