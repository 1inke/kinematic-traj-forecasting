"""Final additive seal, including the required five-port parameter supplement."""
import time
import attribution_core as c

def main():
    g=c.guard();g.check(mode='startup',force=True)
    earlier=c.json_read(c.ROOT/'FINAL_PRESERVATION.json')
    for rel,s in earlier['files'].items():assert c.sha(c.ROOT/rel)==s
    assert c.json_read(c.ROOT/'FIVE_PORT_PARAMETER_SUPPLEMENT.json')['status']=='PASS_ALL12800_PORT_ERRORS_AND125440_EXISTING_ACTION_UTILITIES'
    assert c.json_read(c.ROOT/'completion_review_v2/REVIEW.json')['status']=='PASS_INDEPENDENT_SAVED_ATTRIBUTION'
    base=c.json_read(c.ROOT/'SOURCE_EXTENSION.json')['files'];c.verify_files(base,g.check)
    additional={str(p.resolve()):c.sha(p) for p in c.ROOT.glob('*.py') if str(p.resolve()) not in base}
    c.write_json(c.ROOT/'FINAL_SOURCE_ADDITIONS.json',dict(files=additional,
        no_inherited_digest_replacement=True,parent_base_manifest_sha=c.sha(c.ROOT/'SOURCE_EXTENSION.json')))
    c.verify_files(additional,g.check)
    resources={}
    for name in ('RESOURCE_COMPUTATION.json','completion_review_v2/RESOURCE.json','RESOURCE_FIGURES.json','RESOURCE_FINALIZATION.json'):
        rr=c.json_read(c.ROOT/name);obs=rr['observations'];ss=[o['snapshot'] for o in obs]
        resources[name]=dict(peak_private=max(v['peak_private'] for v in ss),peak_RSS=max(v['peak_rss'] for v in ss),
            min_available_physical=min(v['physical'] for v in ss),min_available_commit=min(v['commit'] for v in ss),
            warnings=sorted(set(w for o in obs for w in o['warnings'])))
    g.check(force=True)
    c.write_json(c.ROOT/'RESOURCE.json',dict(status='PASS_WARNING_FIRST_V3',stage_resources=resources,
        max_single_process_private=max(v['peak_private'] for v in resources.values()),
        not_sum_or_simultaneous_peak=True,figure_commit_1p3_warning_preserved=True,no_memory_lock_or_OS_mutation=True,
        final_seal_observations=g.observations))
    content='''# 最终完成索引

状态：PASS_OFFLINE_ATTRIBUTION；建议PRIORITIZE_SET_UTILITY；0训练/0模型前向。

- [完整结论报告](RESULT_ANALYSIS.md) / [机器可读分析](RESULT_ANALYSIS.json)
- [五端口每候选参数误差、quality与失败位](five_port_candidate_parameters.csv)：12800行，注明actual float32教师与exact END49参数两个口径。
- [全部49动作已有安全/效用数据](all49_saved_action_utility.npz)：125440既有J/safe表单元及距离；无新的全动作几何枚举。
- [最新完整保存审核](completion_review_v2/REVIEW.json) / [全部20组及48收益单元补充审核](completion_review_v2/GROUP_SUPPLEMENT.json)
- [全部5120反事实当前版本](counterfactual_metrics_v2.npz)：junction先转float64；v1及失败证据均保留，见报告第9节。
- [全部7丢失+8救回的15页PDF](cases/ALL_15_CASES.pdf) / [案例索引](CASE_INDEX.md)
- [资源记录](RESOURCE.json)：v3预警制；绘图阶段出现提交内存1.3规划余量预警，未出现紧急/持续压力STOP；未锁内存或改OS。阶段峰值是各进程生命周期读数，不能相加冒充同时总峰值。
- 当前最终封存是FINAL_PRESERVATION_V2.json与FINAL_SEAL_CHECK_V2.json，包含首轮130文件封存和后补完整参数表等；所有旧封存文件保持原样。

同助手第二套实现与反方审核，不是外部审稿。旧tiny90%STOP、TIME停止、P3不触发、正式G/H未测均不变；不自动启动任何后续训练。
'''
    with (c.ROOT/'FINAL_COMPLETION.md').open('x',encoding='utf-8') as f:f.write(content)
    files={str(p.relative_to(c.ROOT)):c.sha(p) for p in c.ROOT.rglob('*') if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts}
    c.write_json(c.ROOT/'FINAL_PRESERVATION_V2.json',dict(status='PASS_OFFLINE_ATTRIBUTION_ALL_DELIVERABLES',
        inherited_source_entries=5140,base_source_entries=len(base),additional_source_entries=len(additional),
        prior130_file_seal_preserved=True,files=files,immutable_output_count=len(files),no_deletion=True,
        no_old_models_or_paused_tasks_modified=True))
    for rel,s in files.items():assert c.sha(c.ROOT/rel)==s
    c.write_json(c.ROOT/'FINAL_SEAL_CHECK_V2.json',dict(status='PASS_ALL_DELIVERABLE_SHA_READBACK',
        files_verified=len(files),manifest_sha=c.sha(c.ROOT/'FINAL_PRESERVATION_V2.json'),
        inherited_source_entries=5140,completion_status='ANALYZED_PASS_OFFLINE_ATTRIBUTION',
        decision='PRIORITIZE_SET_UTILITY',next_training_authorized=False,finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    print('FINAL V2 ALL DELIVERABLES PASS',len(files),'files',flush=True)

if __name__=='__main__':main()
