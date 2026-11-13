"""Evidence-linked finite report and immutable final readback. No model loading."""
from collections import Counter
import csv
from pathlib import Path
import time
import numpy as np
import attribution_core as c

def table(name):return list(csv.DictReader((c.ROOT/name).open(encoding='utf-8-sig')))

def main():
    g=c.guard();g.check(mode='startup',force=True)
    review=c.json_read(c.ROOT/'completion_review/REVIEW.json')
    assert review['status']=='PASS_INDEPENDENT_SAVED_ATTRIBUTION'
    groupreview=c.json_read(c.ROOT/'completion_review/GROUP_SUPPLEMENT.json')
    assert groupreview['status']=='PASS_ALL20_GROUPS_AND_NATIVE_STRATA'
    qa=c.json_read(c.ROOT/'VISUAL_REVIEW.json');assert qa['status']=='PASS_ALL15_CASES_AND_SUMMARY_VISUAL_REVIEW'
    z=c.load_npz(c.ROOT/'derived_ports.npz');cov=z['good'].any(2);m=z['good'][0].sum(1)
    dm=z['initial_error']-z['final_error'];dj=z['J'][0]-z['J'][3]
    st=np.where(z['good'][0],np.where(m[:,None]==1,0,1),np.where(m[:,None]>0,2,3))
    lost=np.flatnonzero(cov[0]&~cov[3]);rescue=np.flatnonzero(~cov[0]&cov[3]);ac=c.space('END')
    assert len(lost)==7 and len(rescue)==8 and np.all(m[lost]==1) and not cov[1,lost].any()
    identity_lost=sum(bool(np.all(z['teacher_action'][i,z['good'][0,i]]==ac.identity)) for i in lost)
    # Extra explicit summaries, all records retained in base CSV/NPZ.
    ut=c.load_npz(c.ROOT/'neighbor_utility_metrics.npz')
    compat=[]
    for name in ('G_main','H_main','G_all','H_all','hash_main','hash_all'):
        for top in (1,5):
            same=ut[name+'_same_action'][:,:top];gap=ut[name+'_executed_gap'][:,:top];ok=ut[name+'_proposal_safe'][:,:top]
            valid=np.isfinite(same);different=valid&(same==0);near=gap<=.01
            compat.append(dict(map=name,top=top,pairs=int(valid.sum()),different_label_pairs=int(different.sum()),
                different_label_executed_near_oracle=int((different&near).sum()),
                different_label_executed_near_oracle_fraction=(different&near).sum()/different.sum() if different.any() else None,
                different_label_safe_proposal_near_oracle=int((different&near&(ok==1)).sum()),
                warning='Fallback identity can preserve Native without showing reference action compatibility; safe-proposal count is separate.'))
    c.csv_new(c.ROOT/'neighbor_functional_equivalence.csv',compat)
    transitions=[]
    for a,b in ((0,1),(1,2),(2,3),(0,3)):
        transitions.append(dict(from_port=c.PORTS[a],to_port=c.PORTS[b],retained=int((cov[a]&cov[b]).sum()),
            lost=int((cov[a]&~cov[b]).sum()),rescue=int((~cov[a]&cov[b]).sum()),unresolved=int((~cov[a]&~cov[b]).sum())))
    c.csv_new(c.ROOT/'stage_transition_counts.csv',transitions)
    stagecodes=dict(Counter(''.join(map(str,row)) for row in cov[:4].T.astype(int)))
    strata=[]
    for k,name in enumerate(c.STRATA):
        mask=st==k;wm=mask.any(1)
        strata.append(dict(stratum=name,candidates=int(mask.sum()),windows=int(wm.sum()),tracks=int(len(np.unique(z['track_key'][wm]))),
            delta_MSE=dm[mask].mean(),delta_MSE_signed_sum=dm[mask].sum(),delta_MSE_contribution=dm[mask].sum()/dm.sum(),
            delta_J=dj[mask].mean(),delta_J_signed_sum=dj[mask].sum(),delta_J_contribution=dj[mask].sum()/dj.sum(),
            support_became_bad=int((mask&z['good'][0]&~z['good'][3]).sum())))
    # The guidance chooses one priority, not an automatic experiment or unique-cause assertion.
    decision=dict(status='PRIORITIZE_SET_UTILITY',execution_status='PASS_OFFLINE_ATTRIBUTION',
        justification='All7 final lost windows had one Native-good support and were already uncovered at C; average gains concentrate in non-support slots;5/7 lost supports have identity teacher.',
        competing_explanations=['fixed MSE fitting remains incomplete','quantization has additional reversible boundaries','local teacher labels differ without exact duplicate hard conflicts'],
        not_selected_as_primary={'decoding':'Q-to-S rescues3 windows and loses0; none of final7 losses first appears in Q/S',
            'input_label_compatibility':'NN functional utility improves vs hash and exact float32 duplicate groups are0; this is not proof information is sufficient',
            'fit_diagnosis':'still plausible but fitting more ordinary candidate error can further dominate rare unique supports; not disproven'},
        automatic_next_training=False,old_tiny_gate='STOP_TINY_FIT_V2_UNCHANGED',formal_G_H='NOT_TESTED',TIME='STOP_UNCHANGED',P3='NOT_TRIGGERED')
    c.write_json(c.ROOT/'DECISION.json',decision)
    ci=c.json_read(c.ROOT/'DESCRIPTIVE_CI.json');dup=c.json_read(c.ROOT/'exact_input_conflicts.json')
    ports=table('ports_summary.csv');groups=table('all_groups_quality_motion.csv');gains=table('gain_allocation.csv');ns=table('neighbor_summary.csv')
    cf=c.load_npz(c.ROOT/'counterfactual_metrics.npz');cf_summary={}
    for mode in ('restore','edit'):
        v=cf[mode+'_covered']
        cf_summary[mode]=dict(K20_sets=2560,rescue_sets_vs_native=int((v&~cov[0,:,None]).sum()),lost_sets_vs_native=int((~v&cov[0,:,None]).sum()),
            rescue_sets_vs_S=int((v&~cov[3,:,None]).sum()),lost_sets_vs_S=int((~v&cov[3,:,None]).sum()),
            unique_rescued_windows_vs_S=int((v&~cov[3,:,None]).any(1).sum()),unique_lost_windows_vs_S=int((~v&cov[3,:,None]).any(1).sum()))
    report=dict(execution_status='PASS_OFFLINE_ATTRIBUTION',decision=decision,population=dict(windows=128,tracks=97,candidates=2560,active=2057,identity=503),
        ports=ports,all20_group_metrics=groups,all48_gain_cells=gains,stage_transitions=transitions,stage_codes=stagecodes,
        support_population=dict(zero=int((m==0).sum()),unique=int((m==1).sum()),multiple=int((m>=2).sum()),allgood_candidates=int(z['good'][0].sum())),
        lost_ids=z['sample_id'][lost].tolist(),rescue_ids=z['sample_id'][rescue].tolist(),all_final_lost_at_C=True,
        identity_teacher_among_final_lost=int(identity_lost),native_support_strata=strata,
        continuous_outside_radius_candidates=int((np.linalg.norm(.4*z['continuous_param'],axis=-1)>.4).sum()),
        candidate_guard_failures={key:int(z['failed'][j].any(-1).sum()) for j,key in enumerate(c.PORTS)},
        counterfactuals=cf_summary,neighbor_summary=ns,neighbor_functional_equivalence=compat,
        exact_input_conflicts={k:{kk:vv for kk,vv in v.items() if kk!='records'} for k,v in dup.items()},descriptive_CI=ci,
        source_count=c.json_read(c.ROOT/'COMPUTATION_RESULT.json')['source_count'],independent_saved_review=review,
        new_training=0,new_model_forward=0,new_all49_geometry_enumeration=0,held_GT_access=False,
        limitations=['posthoc128 train subset / one H42 tiny /97 correlated tracks','F16 previously saw upstream data','H/G metric changes not causal history effect',
            'GT oracle and support restoration not deployable','No exact duplicate conflict does not establish information sufficiency','no new confirmation or p-value fishing'],
        ethics='Existing de-identified trajectories only; no new data/participants. AI-assisted analysis, same-assistant DA and second implementation, not external peer review.')
    c.write_json(c.ROOT/'RESULT_ANALYSIS.json',report)
    lines=['# 端点修正失败归因：关键支持保护应优先于继续压低平均候选误差','',
        '状态：**PASS_OFFLINE_ATTRIBUTION**；首选建议：**PRIORITIZE_SET_UTILITY**。仅给出下一轮单因素设计方向，不自动训练。','',
        '## 1. 结论与证据范围','',
        '已完成A0/A1/A2/A3：原128训练窗、97条sequence-target tracks、全部2560候选；640个五端口窗口、5120个固定K20单槽反事实、全部近邻/哈希参照和5000整track探索抽样。0训练、0前向、没有读取模型/pickle。原tiny未过90%的STOP、TIME停止、P3未触发及正式G/H未测均不变。',
        '', '最清楚的结果不是“没有学会”：平均参数误差和许多失败槽J确实改善；但7个最终丢失窗都依赖唯一Native-good槽，并且在连续预测C时已坏。五端口轨迹与指标见[候选逐阶段表](candidate_stage_audit.csv)、[阶段窗口表](window_stage_transitions.csv)。','',
        '## 2. 连续→量化→过滤：不是过滤首先毁掉这7例','',
        '|端口|好窗/128|救回|丢失|平均minADE|平均minFDE|平均minJ|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in ports:lines.append('|'+row['port']+'|'+'|'.join(row[k] if k in ('good','rescue','lost') else f'{float(row[k]):.6f}' for k in ('good','rescue','lost','minADE','minFDE','minJ'))+'|')
    lines+=['', 'N→C救8丢9；C→Q救2丢3；Q→S救3丢0。阶段码完整保留逆转，不能将7个丢失强塞成互斥原因箱。最终7例均为1→0→0→0，5例关键槽的教师就是identity（不修正），另2例为active教师但预测不对。7例来自5条track，其中同一条track贡献3例；不是7个独立行人样本。',
        '', 'C有77/2560候选超出.4m圆形动作域；C的470个候选违反至少一个guard位、Q为398个，S/O为0。全人口四组五pooled motion与junction通过，不等于C每条轨迹合法。C仅中间表示诊断，不能据此取消量化或安全标准。O为GT safe oracle，90好窗/21救回/0丢失是可达机会，不是可学或部署证据。','',
        '## 3. 平均收益落在哪里','',
        '|Native固定层|候选/窗口|ΔMSE均值|ΔJ均值|ΔMSE有符号贡献|ΔJ有符号贡献|',
        '|---|---:|---:|---:|---:|---:|']
    for r in strata:lines.append(f'|{r["stratum"]}|{r["candidates"]}/{r["windows"]}|{r["delta_MSE"]:.6f}|{r["delta_J"]:.6f}|{r["delta_MSE_contribution"]:.2%}|{r["delta_J_contribution"]:.2%}|')
    lines+=['', 'ΔMSE=初始误差−末误差，ΔJ=Native−S，正数更好。总和可以出现负贡献或单层贡献超过100%；未取绝对值或裁负。所有分组/active与identity交叉及候选加权、窗内平均后窗口加权版本均在[gain_allocation.csv](gain_allocation.csv)。',
        '', '36个唯一支持槽总体ΔJ=-0.08819，8槽变坏，其中一窗被其他槽补救，因此最终丢7窗。唯一支持identity的19槽平均ΔMSE=-0.09128、ΔJ=-0.20576，6槽最终变坏；这与“把原本应当不动的关键槽移坏”相符，但不证明换某个loss必然解决。多支持87槽/33窗表现较稳，不能把所有原好槽一概说受损。',
        '', '原失败59窗的1180槽平均J改善0.32557，但仅8窗真正跨过完整coverage边界。S精确教师动作命中30.27%，safe J在oracle+.01内32.19%；参数误差、动作命中和完整集合覆盖是不同指标，不能替代。','',
        '## 4. 两类单槽反事实','',
        '全部128×20的restore与edit均为K20，固定其余19条，完整重算quality、原complete-link聚簇、五motion及junction。restore相对S恰有7个集合救回7个lost窗，同时8个集合会撤掉S新获得的覆盖。Native背景仅单槽修正有9个救回集合和8个丢失集合；计数单位为集合，不能与独立窗口或可加贡献混淆。',
        '', '**恢复原成功槽必然救回是定义性结果**，不能部署GT选槽；也未搜索2^20组合或声称Shapley/训练因果。完整表：[restore](single_slot_restore.csv)、[edit](single_slot_edit.csv)。','',
        '## 5. 输入近邻：有局部相容性，尚无精确输入硬冲突证据','',
        'G/H近邻及哈希参照先于本轮GT/教师关联封存。全部同track候选排除，每reference track至多1条；主口径同sequence，另给跨sequence敏感性。60候选（3窗）只有2条同sequence参考track，保留真实数量，未跨sequence补齐。',
        '', '同sequence H最近1条：平均教师参数距离0.4063、动作相同38.36%、提议安全82.81%、执行J-gap0.22456；最近5条均值分别为0.4977、30.86%、75.59%、0.26676。哈希参照top5执行J-gap0.48436；近邻相对参照约-0.21760。两组实际距离分布均报告，哈希不保证逐查询更远，也不构成随机化因果对照。',
        '', '不把不同标签直接称功能冲突：[交叉效用表](neighbor_functional_equivalence.csv)同时列出不同标签但执行近oracle、且提议自身安全的数量，避免把fallback identity混成原提议可用。G/H各2560个float32输入均无精确重复，经验重复输入MSE下界均为0；仅表示未找到这类硬冲突，不能证明信息充分，更不是Bayes误差为0。','',
        '## 6. 全事件图与统计','',
        '[全部15事件索引](CASE_INDEX.md)和[合并案例PDF](cases/ALL_15_CASES.pdf)包含全部7 lost+8 rescue：past、GT、所有20槽、所有Native-good支持、C/Q/S/O和7位guard原因。原生物理slot不等于语义意图。',
        '', '![阶段与收益](figures/stage_and_gain.png)','',
        '六个预定描述量共用5000整track重抽样，均5000有效；不做新p值/确认性检验。S−N覆盖差0.78125pp，探索95%区间约[-6.67,+7.69]pp；原好丢失率10.14%，区间约[1.64,20.34]%；不能凭净增1称可靠提升。详见[DESCRIPTIVE_CI.json](DESCRIPTIVE_CI.json)。','',
        '## 7. 下一方向为什么选集合效用，而不先怪安全门','',
        '优先建议固定结构/解码，在下一份精确协议中单因素检验集合效用/成功保留监督。理由是稀少的唯一支持槽受损，而普通失败槽的大量收益支配平均目标；这只是比继续同时扩大网络/预算更有针对性的证据排序。',
        '', '竞争解释仍保留：固定映射拟合未达90%；量化确实新增3个暂时损失也救回2个；邻近输入仍可能缺少条件。安全门并非当前7例最早损伤点，且Q→S净救3无新增丢失，不支持放宽1.01。没有精确重复冲突，不代表未来完全可观测；近邻相关性不等于H对G增量。',
        '', '## 8. 完成性、反方审核与限制','',
        '来源继承5140旧digest和全部父最终输出；新来源初末SHA一致，全部新增输出封存后逐文件读回。checkpoint仅文件SHA。独立第二实现复核640端口、5120反事实的质量/完整聚簇/运动、全部近邻排除/tie/hash、精确冲突下界及5000×6抽样算术；补核全部20组pooled/junction和48收益单元。这里“独立”指同助手第二套算术实现，不是外部审稿或模型/梯度再次回放。',
        '', 'DA1：看到7lost后的探索，完整128/全部事件约束选择偏差。DA2：GT恢复和oracle的定义性优势不是创新，单槽反事实不代表可学策略。DA3：平均收益错位并不证实任何新loss必胜，C越界、距离混杂、样本稀少、同track相关与已参与拟合均明确保留。',
        '', '仅既有去标识数据、无新数据参与者；AI辅助分析和同助手反方审核如实披露。旧模型/结果/失败/日志/枚举/自动记录未删，旧PAUSED未动，baseline/historycycle11/长期目标不变。本有限离线诊断结束，不自动新增训练。']
    with (c.ROOT/'RESULT_ANALYSIS.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    c.write_json(c.ROOT/'RESEARCH_REVIEW_DA2_DA3.json',dict(status='PASS_WITH_ACKNOWLEDGED_LIMITATIONS',same_assistant_not_external=True,
        DA2=dict(issue='Counterfactual restore and oracle success is tautological; ordinary candidate gains do not prove a new objective will win.',resolution='No GT deployment or automatic training; primary priority is conditional design guidance, not unique cause.'),
        DA3=dict(issue='A weak conclusion could cherry-pick7losses or treat zero duplicate bound as observability proof.',resolution='All128, 15cases, multi-support positives, quantization reversals, all NN scopes, identity fallback and no-hard-conflict caveat retained.')))
    # Register every additive helper/QA extension without replacing the base source digests.
    source=dict(c.json_read(c.ROOT/'SOURCE_EXTENSION.json')['files'])
    extras={str(p.resolve()):c.sha(p) for p in c.ROOT.glob('*.py') if str(p.resolve()) not in source}
    c.write_json(c.ROOT/'COMPLETION_SOURCE_EXTENSION.json',dict(files=extras,base_manifest_sha=c.sha(c.ROOT/'SOURCE_EXTENSION.json'),no_base_digest_replacement=True))
    c.verify_files(source,g.check);c.verify_files(extras,g.check);g.check(force=True)
    c.write_json(c.ROOT/'RESOURCE_FINALIZATION.json',dict(status='PASS',policy='warning_first_remaining_v3',estimate=g.estimate,observations=g.observations))
    c.write_json(c.ROOT/'runtime_control.json',dict(status='ANALYZED_PASS_OFFLINE_ATTRIBUTION',active_worker=False,
        new_OS_task=False,new_automation=False,training=0,forward=0,old_paused_unchanged=True,no_deletion=True,next_action='Finite diagnosis ended; no automatic training.'))
    files={str(p.relative_to(c.ROOT)):c.sha(p) for p in c.ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    c.write_json(c.ROOT/'FINAL_PRESERVATION.json',dict(status='PASS_OFFLINE_ATTRIBUTION_SOURCE_AND_OUTPUT_SHA',source_count=len(source),additional_source_count=len(extras),
        files=files,all_old_sources_unchanged=True,all_parent_outputs_unchanged=True,all_new_output_files=len(files)))
    for rel,s in files.items():assert c.sha(c.ROOT/rel)==s
    c.write_json(c.ROOT/'FINAL_SEAL_CHECK.json',dict(status='PASS_ALL_NEW_OUTPUT_SHA_READBACK',files_verified=len(files),
        preservation_sha=c.sha(c.ROOT/'FINAL_PRESERVATION.json'),source_base_sha=c.sha(c.ROOT/'SOURCE_EXTENSION.json')))
    print('FINAL PASS',decision['status'],'sealed files',len(files),flush=True)

if __name__=='__main__':main()
