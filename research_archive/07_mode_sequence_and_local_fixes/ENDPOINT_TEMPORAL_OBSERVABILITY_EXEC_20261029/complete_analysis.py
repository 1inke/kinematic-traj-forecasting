"""CPU saved-receipt / count synthesis. No model or gradient replay."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from io_contract import json_new,json_read,load_npz,sha,verify_manifest


def summarize_resource(path):
    z=json_read(path)
    if 'resource' in z:z=z['resource']
    obs=z['observations']
    return dict(checks_retained=len(obs),estimate_bytes=z.get('estimate'),
        peak_private=max(r['snapshot']['peak_private'] for r in obs),
        peak_RSS=max(r['snapshot']['peak_rss'] for r in obs),
        min_physical=min(r['snapshot']['physical'] for r in obs),
        min_commit=min(r['snapshot']['commit'] for r in obs),
        warnings=sorted(set(t for r in obs for t in r['warnings'])),
        all_recorded_status_PASS=all(r['status']=='PASS' for r in obs),
        observation_window_not_all_system_lifetime=True)


def main():
    verify_manifest(ROOT/'source_manifest.json');verify_manifest(ROOT/'learning_source_extension.json')
    verify_manifest(ROOT/'END_learning_preflight/RESULT.json')
    review=json_read(ROOT/'p0_completion_review/REVIEW.json')
    for name,digest in review['files'].items():
        if sha(ROOT/name)!=digest:raise AssertionError('review output SHA')
    if json_read(ROOT/'P0_WORKER_SUCCESS.json')['review_sha']!=sha(ROOT/'p0_completion_review/REVIEW.json'):raise AssertionError('worker/review binding')
    groups=json_read(ROOT/'p0_completion_review/ALL_GROUPS.json')['rows']
    decisions=json_read(ROOT/'p0_completion_review/DECISION.json')['branches']
    eventrows=[];ambiguity=[];motion=[]
    for branch in ('END','TIME'):
        for pop in ('train','val'):
            x=load_npz(ROOT/f'prepared/X_{pop}.npz')
            directory=ROOT/f'p0_completion_review/{branch}_{pop}'
            oracle=load_npz(directory/'ORACLES.npz')
            for port in ('geometry','motion','safe'):
                z=load_npz(directory/(port+'_metrics.npz'),('covered','native_covered','sample_id'))
                if not np.array_equal(z['sample_id'],x['sample_id']):raise AssertionError('event IDs')
                for i,sample in enumerate(x['sample_id']):
                    b,c=bool(z['native_covered'][i]),bool(z['covered'][i])
                    event='retained' if b and c else 'lost' if b else 'rescue' if c else 'unresolved'
                    eventrows.append(dict(branch=branch,population=pop,port=port,sample_id=int(sample),track=str(x['track_key'][i]),group=int(x['group'][i]),event=event))
            for name,mask in [('overall',np.ones(len(x['sample_id']),bool))]+[(n,x['group']==i) for i,n in enumerate(('stable','transition','change'))]:
                active=oracle['active'][mask]
                ambiguity.append(dict(branch=branch,population=pop,group=name,N=int(mask.sum()),candidate_denominator=int(active.size),
                    active_windows=int(active.any(1).sum()),active_candidates=int(active.sum()),
                    identity_candidates=int(active.size-active.sum()),exact_tie_candidates=int((oracle['exact_ties'][mask]>1).sum()),
                    near_tie_candidates=int((oracle['near_ties'][mask]>1).sum()),
                    mean_safe_count=float(oracle['safe_count'][mask].mean()),
                    median_diameter=float(np.median(oracle['diameter'][mask])),p95_diameter=float(np.quantile(oracle['diameter'][mask],.95))))
    for r in groups:
        item={k:r[k] for k in ('branch','population','port','group','N','native_good','good','rescue','lost','coverage_gain')}
        for k,value in r['motion'].items():
            base=r['native_motion'][k]
            item[k]=value;item['native_'+k]=base;item[k+'_ratio']=value/base if base else None
        for suffix in ('mean','p95'):
            value=r['junction_'+suffix];base=r['native_junction_'+suffix]
            item['junction_'+suffix]=value;item['native_junction_'+suffix]=base
            item['junction_'+suffix+'_ratio']=value/base if base else None
        item.update(motion_literal_pass=r['motion_literal_pass'],junction_literal_pass=r['junction_literal_pass'])
        motion.append(item)
    out=ROOT/'completion_review';out.mkdir(exist_ok=False)
    for filename,rows in [('P0_WINDOW_EVENTS.csv',eventrows),('P0_AMBIGUITY.csv',ambiguity),('FULL_MOTION_TABLE.csv',motion)]:
        with (out/filename).open('x',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    gpu=json_read(ROOT/'END_learning_preflight/GPU8.json')
    resumes=json_read(ROOT/'END_learning_preflight/RESUME.json')
    learning=json_read(ROOT/'LEARNING_PREFLIGHT_RESULT.json')
    strata=json_read(ROOT/'TINY_STRATUM_ADMISSION.json')['strata']
    resource={name:summarize_resource(path) for name,path in (
        ('P0_review',ROOT/'p0_completion_review/RESOURCE.json'),
        ('END_actual_GPU_preflight',ROOT/'END_learning_preflight/RESOURCE.json'))}
    counts=review['counts'];actions=sum(r['actions'] for r in counts)
    if actions!=40140800 or len(groups)!=48 or len(eventrows)!=24576:raise AssertionError('complete denominator')
    if strata['END']['train']['active_windows']!=3103 or strata['END']['train']['inactive_windows']!=0:raise AssertionError('tiny preflight stop basis')
    if learning['branches']['END']['status']!='STOP_TINY_STRATUM_INSUFFICIENT':raise AssertionError('END not stopped as measured')
    if learning['branches']['TIME']['status']!='STOP_LOW_SAFE_OPPORTUNITY_IN_THIS_ACTION_CLASS':raise AssertionError('TIME gate')
    summary=dict(status='ANALYZED_P0_END_TINY_SAMPLING_PROTOCOL_STOP_TIME_OPPORTUNITY_STOP',
        P0=dict(unique_opt_windows=4096,branch_population_rows=8192,actions_reviewed=actions,oracle_group_rows=48,decisions=decisions,
                independent_review_status=review['status'],all_diameter_errors_zero=all(r['max_diameter_error']==0 for r in counts)),
        actual_GPU_preflight=dict(status=gpu['status'],max_double_gradient_error=max(r['independent']['max_gradient_error'] for r in gpu['arms']),
                                  both_real513_12_resume=resumes['status'],new_MLP_only=True),
        tiny=dict(status='STOP_TINY_STRATUM_INSUFFICIENT',strata=strata,updates=0,MSE_drop_test_executed=False,
                  interpretation='The window-level inactive stratum is empty. This is sampling-contract feasibility, NOT a fit failure or evidence of unobservability.'),
        formal_models=0,held_GT_opened=False,P3='NOT_TRIGGERED_TIME_P0_FAILED_INDEPENDENTLY',
        primary_hold_CI='NOT_EXECUTED_NO_FORMAL_FINALS',resource=resource,
        decision_required=dict(proposed_only_not_executed='Clarify the second tiny group as 64 distinct remaining training windows, allowing active windows, selected by the unchanged fixed hash; retain 256 updates and the 90% gate.',
            scientific_thresholds_unchanged=True,requires_human_confirmation=True),
        limits=['Privileged GT oracle is not a learned improvement.','Zero-lost safe oracle is structural.','TIME stop applies only to this fixed action and safety class.',
                'All native trajectories already contain frozen history conditioning.','No held/inner generalization, major improvement or baseline upgrade is established.'],
        same_assistant_not_external=True,no_model_gradient_replay_in_completion=True,no_deletion=True)
    json_new(ROOT/'RESULT_ANALYSIS.json',summary)
    end=decisions['END'];tm=decisions['TIME']
    text='''# 端点／时间可观测性：已完成P0与端点实际GPU预检，等待tiny抽样合同澄清

## 当前结论

P0全部保存结果及独立复算通过。端点动作类有足够安全oracle机会；时间动作类未过原定研究价值门，停止该分支。
端点的GPU梯度和真实513/12恢复核验均通过，但tiny原实现要求的非活跃窗口层为空，故未执行tiny更新，也未放行正式训练。
这不是“模型拟合失败”，更不是“历史没有信息”。本系列尚未获得正式G/H对比或held覆盖改善证据。

| 分支／人口 | 原好窗 | 几何oracle好窗 | 加内部motion过滤 | 再加junction过滤 | 安全救回／轨迹 |
|---|---:|---:|---:|---:|---:|
'''
    for branch in ('END','TIME'):
        for pop in ('train','val'):
            d=decisions[branch][pop]['stages'];s=d['safe']
            text+=f"| {branch} / {pop} ({s['change_N']} change窗) | {s['native_good']} | {d['geometry']['good']} | {d['motion']['good']} | {s['good']} | {s['rescue']} / {s['rescue_tracks']} |\n"
    text+='''
训练change分母721，准入要求至少15救回且至少5条轨迹。END为134/721（+18.5853个百分点），TIME仅5/721（+0.6935个百分点）。验证集不用于调门，END验证44/256（+17.1875个百分点），TIME为0/256。所有oracle零lost由identity合法、同槽J最小化构造保证，不能当作学会保留原成功。

## 真实执行与审核

- 新执行锁4451来源，完整继承原4411旧digest；原18源码保留，新版本只批量化动作插值。实际hash128：END125440/TIME1128960动作与原实现路径逐位一致；独立几何最大误差0/1.776e-15，literal七失败位及教师选择EXACT。
- 全量独立复核40140800动作、8192个branch-population重复窗（4096唯一opt窗）、12个oracle人口／48组：同候选ADE/FDE/J、全部mask／tie／直径、K20 complete-link聚簇及有效good、五pooled motion和junction。安全oracle16组全部原五motion和新增junction人口门通过；不把逐候选通过逻辑上等同pooled门。
- END各候选可用动作平均train12.7378/val13.2023；train有50087/62060候选非identity，但3103/3103窗口至少一候选非identity。TIME的多解/tie和退化需完整保留，详见P0_AMBIGUITY.csv。
- 前台通道中断留下5个完整阶段和29个partial标签分片。原文件保留，在新命名空间逐字节复用，原冻结SHA、copy收据及全数学复算均通过。不是重做已完模型或删除失败。
- 独立CPU续算+审核worker 22:08:45.594371至22:19:10.310898双SUCCESS，约624.423秒；这不含此前前台与128预检，不冒称P0总耗时。唯一半程核验22:20:04.5588881发现已完整结束，不再查旧PID／task／log。
- 实际CUDA G/H各8窗分阶段梯度、零输出identity、配对初始化、G屏蔽列零梯度、独立double全部4770参数梯度及固定1e-4/1e-5差分通过；最大梯度误差5.551e-17。
- G/H各真实513训练窗+12验证窗、两epoch34更新、19步中断序列化恢复，全部模型／optimizer／TorchCPU-CUDA／loader／Python／NumPy RNG、批序和history EXACT。坏source/stat/split/output实际拒绝。不是全3103两轮，不加载旧F16/decoder/scorer，不是外部评审。

## 端点tiny为什么没有运行

原文为“64个含非identity教师动作的窗口+64个其余窗口”，现有冻结核心代码将第二层实现为不含非identity教师动作的窗口。真实END训练集3103窗全部属于第一层，第二层为0。候选级仍有11973个identity标签，不等于存在非活跃窗口。

因此登记STOP_TINY_STRATUM_INSUFFICIENT，tiny更新0，90%下降测试未执行；没有给出假的tiny拟合阴性或完整预检PASS。TIME已独立在P0停止，不能因为它有750个非活跃train窗就跨分支借样本。

建议仅供确认的最小澄清：第二组改为从第一组未抽中的train窗口继续按同一固定哈希取64个，允许仍为活跃窗口；保持128个不同train窗、256次更新、原90%门及全部正式学习合同不变。此澄清尚未执行，未换LR、宽度、标签、GT或评价阈值。现有P0与GPU恢复证据可保留，确认后按新source绑定补tiny及剩余保存/benchmark预检。

## 当前未执行

正式G/H六模型、完整8epoch效果、hold/inner GT本轮评价、六比较主CI、真实模块成本和P3联合均未执行；P3还因TIME的P0门失败而不触发。不得声明部署成功、major提升、独立whole-model泛化或升级baseline。没有新loss/scorer/unfreeze/data/额外变体，旧PAUSED和所有文件保留。

## 同助手反方审核DA2/DA3（非外审）

- DA2：134个安全oracle救回并不证明可学。GT选出的最优参数可能多解；H/G只检验显式历史增量，G输入本身已有冻结历史信息。TIME的50→24→5是按顺序的过滤统计，不是互斥的因果根因分解。
- DA2：窗口级无非活跃层是抽样合同可实现性问题，不能偷换为模型表达、优化或观测信息失败。相反也不能因安全oracle有机会就跳过tiny或改变样本层使预检自动通过。
- DA3：所有安全oracle通过、无lost的结论只对当前GT特权有限动作和人口成立。未训练/未held对比即不支持主要研究问题。原重大改善、原成功保护、全部motion和相关性标准不变；本轮报告明确保留这种未完成性。

## 完整证据

- p0_completion_review/REVIEW.json、ALL_GROUPS.json/csv、各oracle完整NPZ。
- completion_review/P0_WINDOW_EVENTS.csv：24576个重复port-window事件与ID，含全部rescue/lost/retained/unresolved。
- completion_review/P0_AMBIGUITY.csv、FULL_MOTION_TABLE.csv：所有组分母、多解及完整运动量比值。
- END_learning_preflight/GPU8.json、RESUME.json、READER_REJECTION.json、TINY.json及RESOURCE.json。
- TINY_STRATUM_ADMISSION.json、learning_source_extension.json：真实分层及29个原partial来源核验。

报告由AI助手辅助生成与审核；源码、原始数组及SHA收据是证据，不是外部独立评审或第二次全模型梯度回放。
'''
    with (ROOT/'RESULT_ANALYSIS.md').open('x',encoding='utf-8') as f:f.write(text)
    verify_manifest(ROOT/'source_manifest.json');verify_manifest(ROOT/'learning_source_extension.json')
    sources={str(Path(__file__).resolve()):sha(__file__),str(ROOT/'source_manifest.json'):sha(ROOT/'source_manifest.json'),str(ROOT/'learning_source_extension.json'):sha(ROOT/'learning_source_extension.json')}
    json_new(out/'ANALYSIS_EXTENSION.json',dict(files=sources,status='EXPLICIT_SAVED_COUNT_SYNTHESIS_EXTENSION'))
    ignored={'runtime_control.json','halfway_check.json','FINAL_PRESERVATION.json'}
    outputs={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.suffix!='.pyc' and p.name not in ignored and '__pycache__' not in p.parts}
    json_new(ROOT/'FINAL_PRESERVATION.json',dict(status='PASS_SOURCE_AND_ALL_IMMUTABLE_OUTPUT_SHA',source_count=4451,
        files=outputs,paths_relative_to_execution_root=True,source_final_unchanged=True,
        no_model_gradient_replay=True,no_deletion=True))
    print(json.dumps(dict(status=summary['status'],actions=actions,groups=len(groups),resources=resource,outputs=len(outputs)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
