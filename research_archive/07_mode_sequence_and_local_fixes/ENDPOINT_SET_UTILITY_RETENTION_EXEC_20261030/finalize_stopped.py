"""Finite utility-screen STOP report; never turns a tiny failure into formal admission."""
import csv
from pathlib import Path
import time
import numpy as np
import common as c

def main():
    decision=c.json_read(c.ROOT/'TINY_DECISION.json')
    assert decision['status']=='STOP_TINY_UTILITY_NO_FORMAL'
    review=c.json_read(c.ROOT/'tiny_review/REVIEW.json');assert review['status']=='PASS_ALL_NINE_TINY_SAVED_ARITHMETIC'
    manifest=c.json_read(c.ROOT/'source_manifest.json');g=c.resource();g.check(force=True)
    analysis=c.json_read(c.ROOT/'TINY_ANALYSIS_DATA.json');summaries=analysis['summaries']
    old=c.load_npz(c.ATTR/'derived_ports.npz');ng=old['good'][0];nj=old['J'][0];covered=ng.any(1)
    proxy=[];errorfail=[];overview=[]
    for s in summaries:
        arm,seed=s['arm'],s['seed'];z=c.load_npz(c.ROOT/'tiny'/f'{arm}_{seed}'/'final.npz')
        assert np.array_equal(z['sample_id'],old['sample_id'])
        j=z['C_J'];ret=np.zeros(128);resc=np.zeros(128)
        for i in range(128):
            if covered[i]:ret[i]=max(j[i,ng[i]].min()-nj[i].min(),0.)**2
            else:resc[i]=max(j[i].min()-old['J'][4,i].min(),0.)**2
        proxy.append(dict(arm=arm,seed=seed,final_population_continuous_LR=ret.mean(),final_population_continuous_LS=resc.mean(),
            population='fixed128',training_preupdate_average=False,not_executed_coverage=True))
        r=s['S_groups']['overall']
        overview.append(dict(arm=arm,seed=seed,native_good=69,good=r['good'],rescue=r['rescue'],lost=r['lost'],lost_fraction=r['lost_fraction'],
            minADE=r['minADE'],minFDE=r['minFDE'],active_MSE_drop=s['active_MSE_drop'],nonidentity=s['nonidentity'],
            circle_outside=s['continuous_outside_circle'],guard_rejections=s['guard_rejections'],guard_failed_bits=s['guard_failed_bits']))
        for group,r in s['S_groups'].items():
            lim=1 if group=='change' else 1.01
            for metric in ('minADE','minFDE'):
                if r[metric]>lim*r['native_'+metric]:
                    errorfail.append(dict(arm=arm,seed=seed,group=group,metric=metric,value=r[metric],native=r['native_'+metric],limit_factor=lim,
                        harm_percent=100*(r[metric]/r['native_'+metric]-1)))
    c.csv_new(c.ROOT/'TINY_OVERVIEW.csv',overview);c.csv_new(c.ROOT/'FINAL_CONTINUOUS_PROXIES.csv',proxy);c.csv_new(c.ROOT/'ERROR_PROTECTION_FAILURES.csv',errorfail)
    c.write(c.ROOT/'CI.json',dict(status='NOT_RUN_TINY_GATE_STOP',family_size=6,draws_registered=20000,seed=20261030,
        H1='NOT_RUN',H2='NOT_RUN',H3='NOT_RUN',H4='NOT_RUN',H5='NOT_RUN',H6='NOT_RUN',
        reason='No formal fits, no hold/inner evaluation; no training-subset significance substituted',old_G_H='NOT_TESTED'))
    c.write(c.ROOT/'FINAL_RELEASE.json',dict(status='NOT_RELEASED_TINY_GATE_FAILED',tiny_models=9,tiny_updates=2304,
        formal_models=0,formal_updates=0,val_effects='NOT_RUN',hold_GT_effects='NOT_RUN',inner_GT_effects='NOT_RUN',
        preflight_val12_only=True,TIME='STOP_UNCHANGED',P3='NOT_TRIGGERED',original90_STOP_unchanged=True))
    formal=Path(c.ROOT/'formal');assert not formal.exists()
    finaldec=dict(status='ANALYZED_TINY_UTILITY_STOP_NO_FORMAL',execution_status='PASS_NINE_TINY_AND_INDEPENDENT_AUDIT',
        scientific_screen='FAIL_TINY_JOINT_GATE',gate=decision,
        interpretation='Retention reduces lost in2/3 seeds; rescue term increases rescues in3/3 vs R, but RS retains6-8 losses and harms protected transition FDE/change ADE. Partial training-set effects, not usable support-preserving solution.',
        no_formal_or_held=True,no_fourth_variant=True,no_posthoc_select_R=True,no_coefficient_epoch_or_width_retry=True)
    c.write(c.ROOT/'DECISION.json',finaldec)
    # The three checkpoints are inline research-integrity reviews, never independent external opinions.
    c.write(c.ROOT/'RESEARCH_REVIEW_DA2_DA3.json',dict(status='PASS_REPORT_WITH_LIMITS',same_assistant_not_external=True,
        DA2=dict(issue='Net coverage73 can hide6lost; lower continuous R/S proxy is not guarded retention.',
            resolution='All9 outcomes/CQS/groups, 1% criterion, transition/change harms and failed checks retained; no formal admission.'),
        DA3=dict(issue='A negative fixed coefficient experiment cannot disprove every set objective, nor demonstrate input irreducibility.',
            resolution='Conclusions limited to same4770 model, fixed128/3seed/256updates/lambda1; no sweep, newteacher, or broad causal claim.')))
    c.write(c.ROOT/'RESULT_ANALYSIS.json',dict(status=finaldec['status'],decision=finaldec,
        tiny_overview=overview,all_summaries=summaries,continuous_proxies=proxy,protection_failures=errorfail,
        actual_models=9,actual_updates=2304,actual_tiny_epochs=576,formal_models=0,held_GT_opened=False,
        M42_control=c.json_read(c.ROOT/'M42_CONTROL_REPRODUCTION.json'),independent_saved_review=review,
        preflight_current='preflight_v3',preflight_subset_updates=3*2*34,preflight_separate_from_scientific_models=True,
        bad_adapter_v1_and_receipt_v2_failures_preserved=True,baseline_history_goal_unchanged=True,
        no_old_F16_decoder_scorer_forward_or_training=True,no_deletion=True,old_paused_unchanged=True))
    lines=['# 集合效用与成功保留监督：tiny筛查未通过，停止正式分支','',
        '**状态：ANALYZED_TINY_UTILITY_STOP_NO_FORMAL。**九个tiny已全部完成并通过独立保存算术审核；科学效用联合门未过，正式九模型及held分析未启动。','',
        '## 1. 实际执行与不变项','',
        '原128训练窗、97条sequence-target tracks、2560候选，M/R/RS×42/43/44各fresh256更新（64epoch），共2304更新/576tiny epoch。相同4770 H网络、train3103旧统计、teacher、配对初始化/批序、AdamW .001/.0001、clip5、END49、literal guard和K20；唯一因素为目标配置，两个新系数固定1。',
        '', 'M为候选MSE，R加原好支持保留项，RS再加原坏窗救回项。MSE沿原float32路径，新连续几何/J为float64，无径向clip/STE或guard反向。旧90%拟合STOP并未改写，本轮也没有检验H−G历史增量。','',
        '## 2. 全九模型的真实安全执行结果','',
        '下表全部以guarded S端口判断完整samecandidate ADE≤.2 AND FDE≤.4；Native好窗均69/128。不能把净增掩盖原成功损失。','',
        '|seed|目标|好窗/128|救回|丢失|原好丢失率|active MSE下降|',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in overview:lines.append(f'|{r["seed"]}|{r["arm"]}|{r["good"]}|{r["rescue"]}|{r["lost"]}|{r["lost_fraction"]:.2%}|{r["active_MSE_drop"]:.2%}|')
    lines+=['','M42的初始化、最终参数、优化器、sampler、Python/NumPy/TorchCPU/CUDA RNG、history、全部批序均对旧H42 **bitwise EXACT**，连续预测也bitwise一致。它是兼容性复现，不是新增独立泛化重复。','',
        'R−M的lost差为0/-1/-1：2seed减少，seed42不变；R43的rescue从M的9减为8，保留存在折中。RS−R的rescue差为+1/+1/+2，三个seed都有小幅增量；但RS−M的完整好窗差为0/+1/+4，并非三seed严格改善。','',
        'RS的active参数MSE下降约71.64%/73.39%/73.35%，低于M的约80%，同时部分覆盖/误差更好；这说明参数拟合与集合效用并不等价，但不能因此放行保护失败。各模型仍未达到旧90%描述门，不把新机制结果冒旧任务PASS。','',
        '## 3. 预注册tiny门逐项判定','']
    for key,value in decision['checks'].items():lines.append(f'- {key}: **{"PASS" if value else "FAIL"}**')
    lines+=['', '核心阻碍：RS仍丢8/6/6个原好窗，约11.59%/8.70%/8.70%；tiny有69原好，≤1%明确等价丢0，不能改成“允许一例”。RS42还比M42多丢1例。所有RS的五pooled motion与junction四组都通过，但物理通过不意味着准确性保留。','',
        '准确性保护反例（只列RS；全部arm/group详见ALL_GROUPS.csv）：','',
        '|seed|group|指标|相对Native损害|允许因子|','|---|---|---|---:|---:|']
    for r in errorfail:
        if r['arm']=='RS':lines.append(f'|{r["seed"]}|{r["group"]}|{r["metric"]}|{r["harm_percent"]:.3f}%|{r["limit_factor"]}|')
    lines+=['', 'transition FDE三seed损害约10.55%/14.22%/17.25%，change ADE约1.11%/1.90%/1.10%；明显不是仅一次边界差一点。RS总体minADE/FDE有所改善，不能挽救这些分组保护失败。','',
        '## 4. 机制线索与连续代理局限','',
        '保留项对部分seed有小幅lost改善，救回项使R基础上的rescue增加，但没有把关键支持损伤消除。三个目标无clip触发，不能把失败归为已证明的梯度爆炸；各项真实梯度范数和活动面、每批预测及三个loss分量均在checkpoint和training_trace.npz中保留。',
        '', '训练前更新分量均值与final总体代理不是同一口径：[FINAL_CONTINUOUS_PROXIES.csv](FINAL_CONTINUOUS_PROXIES.csv)独立保存最终连续LR/LS，不能用较小代理宣称真实S coverage/保护已实现。连续值可能超圆域，量化与guard仍可能改变作用；[全部108组端口表](ALL_GROUPS.csv)、[432分层单元](SUPPORT_STRATA.csv)、[全部3456端口事件](WINDOW_EVENTS.csv)覆盖C/Q/S、Native支持四层、identity/active、原15事件及全部其他窗口。',
        '', '非identity执行约72.7–73.5%（RS），不是全identity平凡解；它仍未满足同时保留与救回。固定系数1可能形成目标竞争，当前结果不证明所有集合目标都无效，也不证明输入不足。','',
        '## 5. 真实预检、保存审核与来源','',
        '实际三arm GPU前8train分阶段梯度/配对初值/M原标量梯度EXACT；独立NumPy双精度scalar和全4770参数梯度误差最大约2.78e-16，固定4方向×eps1e-4/1e-5、max/min/norm/hinge活动面均原样记录，真实点与预设光滑夹具检查通过。另单测全好/全坏/混合/空组、exact ADE-first/max、最低slot/min、零norm与hinge0子梯度。',
        '', '三arm各真实hash513train+12val、两epoch34更新/中断19恢复：全部参数、外部统计绑定、优化器、TorchCPU/CUDA/loader/Python/NumPy RNG、batch/history/新诊断EXACT。此为子集预检，不是完整3103两轮，也不计科学tiny模型。原模型无统计buffers，统计在输入变换外部SHA绑定，报告未虚称存在新buffer。',
        '', '完整保存审核覆盖9×3×128=3456窗口端口/69120候选端口，C/Q/S几何、动作tie/guard位、同槽ADE/FDE/J、原SciPy全K20 complete-link/pdist、有效good簇、单J最优完整路径早中晚、全部group五pooled/junction及所有2304批loss分量。float32 MSE独立后端最大差5.96046e-8，低于预定2e-7后端界；双精度R/S最大差1.38778e-16，低于2e-12。两界均不用于good、motion或其他科学门，未加epsilon。',
        '', '初次静态括号错误的代码保留。实际预检v1在旧audit接口把callback当作arithmetic_atol而失败；v2因独占写保护拒绝重写已有资源收据而停止。两者均在科学tiny前，旧失败、fixture checkpoint、所有源码/报告均保留。当前v3仅显式修audit keyword和输出/资源收据namespace，完整数学/GPU/resume/saved pipeline重新跑PASS后才锁源训练，未把旧PASS重命名成新PASS。',
        '', '## 6. 耗时、资源及未执行项','']
    bench=c.json_read(c.ROOT/'BENCHMARK.json');comp=c.json_read(c.ROOT/'TINY_COMPUTATION.json')
    lines+=[f'阶段T事前规划{bench["tiny_planning_seconds"]:.1f}秒，实际九模型训练与保存流程{comp["elapsed_seconds"]:.1f}秒（含流程IO、不含此前代码实现/预检及后续独立审核）；独立tiny审核{review["elapsed_seconds"]:.1f}秒。不能拿单次优化或旧4.7秒训练时间冒整个项目耗时。',
        '', f'条件阶段F保守规划{bench["formal_planning_seconds"]/60:.1f}分钟，另有尚未实际测量的hold/inner END49教师构造；因tiny门失败，F不启动，未发生这些成本。旧F16原始端到端成本NOT_REMEASURED，4770是新模块而非整模总参数。',
        '', 'v3库存来自旧真实GPU峰+当前payload/doublegraph/checkpoint/allocator，1.3仅预警；不锁内存、不调OS/pagefile、不关闭用户软件、不删文件。资源峰值详见RESOURCE.json，磁盘事前0.5GiB保守×1.3检查通过。','',
        '**NOT_RUN：正式9模型、完整train/val/hold/inner的正式最终评价、held教师新构造、六项新hold主bootstrap、部署/major放行。**原G/H问题NOT_TESTED，TIME/P3不变。不能拿tiny中的改善伪装迁移效果或缩小family6补显著。','',
        '## 7. 结案与反方审核','',
        '本有限配置停止，不把R事后改为主方法进入正式，不加第四目标、不调系数/LR/网络宽度/训练步数。可保留“目标可改变局部保留/救回分配”的训练内线索，但当前组合未达到可靠原生支持保护。',
        '', 'DA1：设计基于已见tiny失败，完整三arm×三seed、冻结旧128与全部门；DA2：净增和物理PASS不可掩盖lost与transition/change损害；DA3：固定配置阴性不能排除整个集合目标家族，三seed/97相关track/训练内拟合/F16已见及整个项目重复开发限制保留。没有独立held泛化、SOTA、意图理解或唯一根因结论。',
        '', 'AI辅助实施；DA与第二套算术由同助手完成，不是外部审稿，不是所有模型/梯度再次回放。仅既有去标识数据、无新增参与者或外部数据，许可与伦理沿原项目要求；不编造IRB。全部旧模型/结果/失败/日志/自动记录保留，旧PAUSED不动，baseline/historycycle11/长期目标未升级或假完成。']
    with (c.ROOT/'RESULT_ANALYSIS.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    resources={}
    for name in ('preflight_v3/RESOURCE.json','RESOURCE_TINY.json','tiny_review/RESOURCE.json'):
        value=c.json_read(c.ROOT/name);obs=value['observations'];s=[v['snapshot'] for v in obs]
        resources[name]=dict(peak_private=max(v['peak_private'] for v in s),peak_RSS=max(v['peak_rss'] for v in s),
            min_physical=min(v['physical'] for v in s),min_commit=min(v['commit'] for v in s),warnings=sorted(set(w for v in obs for w in v['warnings'])))
    c.write(c.ROOT/'RESOURCE.json',dict(status='PASS_WARNING_FIRST_V3',stages=resources,
        max_single_process_private=max(v['peak_private'] for v in resources.values()),not_aggregate_simultaneous_peak=True,no_memory_reserved_or_OS_mutation=True))
    c.verify(manifest['files'],g.check)
    extras={str(p.resolve()):c.sha(p) for p in c.ROOT.glob('*.py') if str(p.resolve()) not in manifest['files']}
    c.write(c.ROOT/'COMPLETION_SOURCE_EXTENSION.json',dict(files=extras,base_manifest_sha=c.sha(c.ROOT/'source_manifest.json'),no_old_digest_replacement=True))
    c.verify(extras,g.check)
    c.write(c.ROOT/'runtime_control.json',dict(status='ANALYZED_TINY_UTILITY_STOP_NO_FORMAL',active_worker=False,models_completed=9,updates=2304,
        formal_models=0,held_GT_opened=False,new_OS_task=False,new_automation=False,no_deletion=True,old_paused_unchanged=True,
        next_action='Finite series stopped by preregistered tiny gate; no automatic fourth configuration or retuning.'))
    files={str(p.relative_to(c.ROOT)):c.sha(p) for p in c.ROOT.rglob('*') if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts}
    c.write(c.ROOT/'FINAL_PRESERVATION.json',dict(status='PASS_NINE_TINY_SOURCES_AND_OUTPUT_SHA',files=files,
        source_count=len(manifest['files']),additional_helper_sources=len(extras),all_old_sources_unchanged=True,no_deletion=True))
    for rel,s in files.items():assert c.sha(c.ROOT/rel)==s
    c.write(c.ROOT/'FINAL_SEAL_CHECK.json',dict(status='PASS_ALL_OUTPUT_SHA_READBACK',files_verified=len(files),
        manifest_sha=c.sha(c.ROOT/'FINAL_PRESERVATION.json'),decision='STOP_TINY_UTILITY_NO_FORMAL',
        finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    print('FINAL NINE TINY AUDIT PASS; SCIENTIFIC GATE STOP; formal0; sealed',len(files),flush=True)

if __name__=='__main__':main()
