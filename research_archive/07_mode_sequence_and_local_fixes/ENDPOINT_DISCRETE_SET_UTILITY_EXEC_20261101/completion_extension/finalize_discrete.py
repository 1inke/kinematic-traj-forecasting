"""Explicit final saved-data/source extension. No model forward or training."""
import csv
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import contracts_discrete as c
from preflight_checks import trees_equal

def wm(paths,past):
    h=past.astype(float);v=(paths-np.concatenate((np.zeros(paths.shape[:-2]+(1,2)),paths[...,:-1,:]),-2))/.4
    a=np.diff(v,axis=-2)/.4;j=np.diff(v,n=2,axis=-2)/(.4**2)
    v,a,j=[np.sqrt((x*x).sum(-1)) for x in (v,a,j)]
    jun=np.sqrt(((paths[:,:,0]/.4-((h[:,-1]-h[:,-2])/.4)[:,None])**2).sum(-1))/.4
    return np.stack((np.quantile(v,.95,axis=(1,2)),a.mean((1,2)),np.quantile(a,.95,axis=(1,2)),j.mean((1,2)),np.quantile(j,.95,axis=(1,2)),jun.mean(1),np.quantile(jun,.95,axis=1)),1)

def main():
    t=time.perf_counter();exit_receipt=c.json_read(c.ROOT/'WORKER_EXIT.json');assert exit_receipt['status']=='SUCCESS'
    manifest=c.json_read(c.ROOT/'source_manifest.json');c.verify(manifest['files'])
    original=c.json_read(c.ROOT/'FINAL_PRESERVATION.json')
    for rel,s in original['files'].items():assert c.sha(c.ROOT/rel)==s
    d=c.read_prepared('tiny');pf=c.json_read(c.ROOT/'PREFLIGHT.json');decision=c.json_read(c.ROOT/'TINY_DECISION.json')
    summaries=decision['all_seed_results'];windowrows=[];metas=[];init={};order={}
    native=(d['nj']<=1).any(1);nm=wm(d['x']['native'],d['x']['past'])
    for seed in (42,43,44):
        for arm in ('G','H'):
            f=c.ROOT/'tiny'/f'{arm}_{seed}';done=c.json_read(f/'done.json')
            state=c.trusted_checkpoint(f/'checkpoint_256.pt',done['files']['checkpoint_256.pt']);initial=c.trusted_checkpoint(f/'checkpoint_000.pt',done['files']['checkpoint_000.pt'])
            keys=('model','initial','optimizer','sampler','python_rng','numpy_rng','torch_rng','cuda_rng','epoch','cursor','updates','order','history','batch_order','trace','contract')
            assert all(k in state for k in keys) and trees_equal(initial['model'],state['initial'])
            assert state['contract']==c.contract(d,arm,'tiny') and state['updates']==256 and state['epoch']==64
            assert all(__import__('torch').isfinite(v).all() for v in state['model'].values())
            if seed not in init:init[seed]=state['initial'];order[seed]=state['batch_order']
            else:assert trees_equal(init[seed],state['initial']) and order[seed]==state['batch_order']
            metas.append(dict(arm=arm,seed=seed,params=6800,model_buffers=0,all_RNG_payloads_and_contract_present=True,initial_point0_EXACT=True,
                optimizer_steps=sorted(set(int(v['step']) for v in state['optimizer']['state'].values())),history64=True))
            z=c.load_npz(f/'point_256/output.npz');motion=wm(z['paths'],d['x']['past'])
            for i in range(128):
                event='retained' if native[i] and z['covered'][i] else 'lost' if native[i] else 'rescue' if z['covered'][i] else 'unresolved'
                windowrows.append(dict(arm=arm,seed=seed,sample_id=int(d['x']['sample_id'][i]),track=str(d['x']['track_key'][i]),group=int(d['x']['group'][i]),event=event,
                    choice=int(z['choice'][i]),slot=int(z['slot'][i]),action=int(z['action'][i]),score=float(z['selected_score'][i]),value=float(z['selected_value'][i]),regret=float(z['regret'][i]),
                    minADE=float(z['minADE'][i]),minFDE=float(z['minFDE'][i]),minJ=float(z['minJ'][i]),
                    **dict(zip(c.reference.NAMES,motion[i].tolist())),**dict(zip(('native_'+k for k in c.reference.NAMES),nm[i].tolist()))))
    c.csv(c.ROOT/'completion_extension/FINAL_WINDOW_EVENTS_AND_MOTION.csv',windowrows)
    resource=[]
    for name in ('RESOURCE_D0.json','RESOURCE_PREFLIGHT.json','RESOURCE_TINY.json','RESOURCE_REVIEW.json'):
        r=c.json_read(c.ROOT/name);obs=r['observations']
        resource.append(dict(stage=name,estimate=r['estimate'],own_lifetime_private_peak=max(x['snapshot']['peak_private'] for x in obs),own_lifetime_RSS_peak=max(x['snapshot']['peak_rss'] for x in obs),
            warnings=sorted(set(w for x in obs for w in x['warnings'])),STOP=any(x['status']!='PASS' for x in obs)))
    counts=c.json_read(c.ROOT/'EXECUTION_ACCOUNTING.json');actual=sum(counts['preflight_optimizer_updates'].values())+1536
    c.write(c.ROOT/'EXECUTION_ACCOUNTING_FINAL.json',dict(science=counts,actual_Adam_step_calls=actual,resource_stages=resource,
        parameter_scope='6800 new head only; no old generator forward',prePopen_admission=c.sha(c.ROOT/'RESOURCE_ADMISSION.json'),no_OS_pagefile_memorylock_or_userkill=True))
    c.write(c.ROOT/'completion_extension/REVIEW.json',dict(status='PASS_COMPLETE_MODEL_METADATA_FINAL_EVENT_MOTION_AND_ROOT_LOG_SOURCE_SHA',models=metas,window_rows=len(windowrows),
        full_source_count=manifest['count'],base_seal_sha=c.sha(c.ROOT/'FINAL_PRESERVATION.json'),helper_SHA=c.sha(__file__),all_old_PAUSED_untouched=True,
        no_forward_backward_optimizer=True,not_external_review=True,seconds=time.perf_counter()-t))
    result=c.json_read(c.ROOT/'RESULT_ANALYSIS.json');bench=c.json_read(c.ROOT/'BENCHMARK.json');run=c.json_read(c.ROOT/'TINY_COMPUTATION.json');rev=c.json_read(c.ROOT/'completion_review/REVIEW.json');D0=c.json_read(c.ROOT/'D0_RESULT.json')
    formal='NOT_TRIGGERED_T_GATE_FAILED' if decision['status']=='TINY_ACTION_UTILITY_STOP' else 'PENDING_OWN_PREFLIGHT_AND_30MIN_ADMISSION'
    with (c.ROOT/'RESULT_ANALYSIS_FINAL.md').open('x',encoding='utf-8') as f:
        f.write('# 有限端点动作集合效用：最终实验报告\n\n')
        f.write(f"结论：`{decision['status']}`。正式F阶段：`{formal}`。\n\n")
        f.write('## 固定方法与范围\n\n原tiny128窗/97相关tracks，Native69成功；新6800参数G/H各3fresh种子，48→64→32→48。原40维统计冻结，新8维集合距离仅train3103拟合；G在标准化后清零历史16维。原END49去掉重复identity后20×48单槽编辑+全局NOOP，原guard不变，其他19路径逐字节保留。\n\n')
        f.write('监督是所有合法动作的完整K20效用，正/负/零非空类别等权MSE，float64比较；没有唯一oracle教师、连续C、量化Q、额外保护loss或GT部署回退。NOOP阈值固定0，分数不是校准概率。\n\n')
        f.write('## 六个固定末步结果\n\n|arm|seed|覆盖/128|救回|丢失/69|NOOP|负效用选择|有限动作regret|\n|---|---|---|---|---|---|---|---|\n')
        for r in summaries:f.write(f"|{r['arm']}|{r['seed']}|{r['good']}|{r['rescue']}|{r['lost']}|{r['noop_fraction']:.3f}|{r['selected_negative']}|{r['regret']:.6f}|\n")
        f.write('\nH为事先指定主方法；不能改用最好G推进。H每seed必须lost0、rescue≥9/9/10并通过全部protected误差、change不坏、全组五pooled及junction门。\n\n')
        for gate in decision['H_gates']:f.write(f"- H{gate['seed']}：{gate['checks']}；联合通过={gate['pass_all']}。\n")
        f.write('\n所有seed×保存点0/64/128/192/256×四组见ALL_GROUPS.csv；全部3840窗口事件及选择分数/真实效用/救回丢失见WINDOW_EVENTS.csv，最终每窗运动成本另见completion_extension/FINAL_WINDOW_EVENTS_AND_MOTION.csv。不是只报净增或最优中途模型。\n\n')
        f.write('## D0机会与审计\n\n旧O0路径逐字节复现21rescue/0lost；新效用oracle另外计算，同样覆盖90且实际全组误差/运动通过。两者皆GT见证，不是可学习或可部署证书。\n\n')
        f.write(f"D0逐动作显式重构122880个K20集合，标签/符号/同路径覆盖与公式完全一致，maxerror={D0['full_actions']['maxerror']}；新8维train-only独立两遍与31/256chunk统计最大差{D0['new8_stats_max_two_pass_chunk_error']:.3g}。原train/val IDs和sequence|target tracks无交集；不是上下文独立或从未开发过的数据证明。\n\n")
        f.write('实际G/H GPU8 phased、独立NumPy全6800梯度/固定双eps1e-4/1e-5、配对初值/归一化后历史清零梯度0、NOOP/ties/全invalid分母/GT改动前向mask选择不变均通过。双方真实hash513train+12val两epoch34updates、第19中断，参数/Adam/externalstats/全部CPU-CUDA-Python-NumPy RNG/sampler/batch/history/trace EXACT。不是全train两轮。\n\n')
        f.write('两臂实际128非零输出在科学启动前完成独立CPU同路径质量、原SciPy完整K20 complete-link(.10/.20)+pdist、每组pooled五motion及junction、19固定槽、标签/选动作/regret保存审核。坏source/split/stat/actionorder/checkpoint/output真实reader拒绝。\n\n')
        f.write(f"完整结案审核6checkpoint/1536次preupdate损失/30保存点，算术最大loss差{rev['preupdate_double_max_error']:.3g}；{manifest['count']}来源初末SHA不变。保存算术不是所有模型梯度二次前向后向，同助手ARS DA1/2/3非外审。\n\n")
        f.write('## 代价、边界与结案\n\n')
        f.write(f"保守T启动规划{bench['T_conservative_seconds']:.2f}s，门1800s；实跑训练保存{run['seconds']:.2f}s，独立审核{rev['seconds']:.2f}s，另有D0/预检/封存。科学1536 Adam更新，预检208（恢复136+固定benchmark72），总实际1744；不混计。无失败前科学更新、无旧模型重训。\n\n")
        f.write(f"实际8窗口集合距离{bench['geometry8_8windows_seconds']:.6f}s，原960 guard{bench['original960mask_8windows_seconds']:.6f}s；6800参数/133120 MAC只指新MLP，不含原F16/数据IO等真实全管线成本，不称整模轻量部署成功。\n\n")
        f.write('resource_guard_v3为实测inventory估计，1.3仅预警；各受测进程自身Windows生命周期private/RSS峰见EXECUTION_ACCOUNTING_FINAL.json，不是全树合并峰。无固定6GiB硬停、无OS/pagefile/用户程序kill或内存锁。原独立按需XML0任务保留记录、不重复startup/half或恢复旧PAUSED，所有文件/模型/日志/失败都保留。\n\n')
        if formal=='NOT_TRIGGERED_T_GATE_FAILED':f.write('T联合门失败，按设计停止此固定端点小分支；F六模型/H1-H5五项held检验未触发，hold/inner GT未解析或评价。未改阈值/系数/seed/epoch，也没有挑G替代H。\n\n')
        f.write('新方法相对旧RS改变表示、目标和单槽政策，不作单因素因果归因。G已有F16历史处理后的native几何，H-G只检验额外显式历史且此处是训练内tiny；6800容量、8维摘要、类别重加权/960最大选择误估和有限优化均为竞争解释。不能由当前结果推导历史无信息/未来不可辨识，也不能把tiny改善当跨track泛化或心理意图理解。旧90%STOP/TIME/P3、major双5%及全部1%保护/相关性审核、baseline/historycycle11与长期目标保持。\n')
    c.write(c.ROOT/'RESULT_ANALYSIS_FINAL.json',dict(result,formal_status=formal,completion_extension=c.json_read(c.ROOT/'completion_extension/REVIEW.json'),accounting=c.json_read(c.ROOT/'EXECUTION_ACCOUNTING_FINAL.json')))
    c.verify(manifest['files'])
    files={str(p.relative_to(c.ROOT)):c.sha(p) for p in c.ROOT.rglob('*') if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts
        and p.name not in ('runtime_control.json','startup_check.json','halfway_check.json','completion_check.json')}
    c.write(c.ROOT/'FINAL_PRESERVATION_V2.json',dict(status='COMPLETE_SOURCE_SCIENCE_LOGS_WORKER_EXIT_SEAL',files=files,original_seal_retained=True))
    for rel,s in files.items():assert c.sha(c.ROOT/rel)==s
    c.write(c.ROOT/'FINAL_SEAL_CHECK_V2.json',dict(status='PASS',files=len(files),manifest=c.sha(c.ROOT/'FINAL_PRESERVATION_V2.json')))
    print('FINAL DISCRETE SERIES',decision['status'],formal,flush=True)
if __name__=='__main__':main()
