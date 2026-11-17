import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys,time,traceback
import numpy as np
import torch
from probe_io import ROOT,OLD,read,write,verify,seal,sha,now,npz
from probe_cache import create_cache,load_cache
from probe_train import run_one,evaluate,save_evaluation
from probe_metrics import decide
from probe_preflight import technical_preflight

def render(result):
    d=result['decision'];base=result['baseline']['groups'];runs=result['runs']
    lines=['# EXP-SET-INTERACT-PROBE：冻结缓存上的候选交互检验','',
        f"执行状态：**{result['execution_status']}**；机制判定：**{d['verdict']}**；保护状态：**{d.get('protection_status','PENDING')}**。",'',
        '所有结果来自冻结C0 B0@64 seed42缓存；修正模块seed为0/1/2，不能当作三个底座seed。严格逐候选输入和保护门分开报告，均已由用户确认。dev反复用于开发，只作探索，不作为独立测试或论文结论。','',
        '| 臂 | minADE | minFDE | Top1 FDE | 冗余近对 | 有效候选数 | 纵向超范围 | 范围内缺口 | 未覆盖率 | 修正幅度m |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def row(name,g):
        values=[g[k] for k in ('minADE','minFDE','top1FDE','redundant_pairs','effective_candidates','longitudinal_outside','longitudinal_inside_gap','uncovered','mean_shift')]
        return '| '+name+' | '+' | '.join('—' if v is None else f'{v:.6f}' for v in values)+' |'
    lines.append(row('A0',base['all']))
    for arm,items in runs.items():
        g={k:float(np.mean([v['groups']['all'][k] for v in items.values()])) if base['all'].get(k) is not None else None for k in base['all'] if isinstance(base['all'][k],(int,float)) or base['all'][k] is None}
        lines.append(row(arm,g))
    lines+=['','纵向比例仅在方向有效窗口计算；未覆盖是minFDE>.4，实际再分配门是赢家FDE>.4，两者分别保存。修正幅度是各锚点位移范数的均值。','',
        f"Q1 A1对A0平均相对FDE改善：{d['Q1_proxy_gain']:.4%}；sigma={d['sigma']:.4%}，thr={d['threshold']:.4%}（相对比例，样本标准差ddof=1）。",'']
    if d.get('stopped_after_A1'):
        lines+=['**代理检查未通过，按设计停止，剩余15次不运行，不解释交互或损失机制。**','']
    if 'comparisons' in d:
        lines+=['| 问题 | 比较 | FDE相对改善 | ADE相对改善 | 固定有效门 |','|---|---|---:|---:|---|']
        for name,b in d['comparisons'].items():lines.append(f"| {name} | {b['arm']} vs {b['reference']} | {b['mean_FDE_gain']:.4%} | {b['mean_ADE_gain']:.4%} | {'通过' if b['passes'] else '未通过'} |")
        lines+=['','| 臂 | Top1相对A0恶化 | stable FDE恶化 | decel_stop FDE恶化 | 保护 |','|---|---:|---:|---:|---|']
        for arm,b in d['protections'].items():lines.append(f"| {arm} | {b['checks']['top1']['relative_harm']:.4%} | {b['checks']['stable']['relative_harm']:.4%} | {b['checks']['decel_stop']['relative_harm']:.4%} | {'通过' if b['passes'] else '失败'} |")
    lines+=['','## D0：冻结B0 dev','', '| 分组 | n | 冗余近对 | 有效候选 | minFDE未覆盖 | 赢家FDE门 | 纵向超范围 | 范围内缺口 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for group,g in base.items():
        fmt=lambda k:'—' if g.get(k) is None else f'{g[k]:.5f}'
        lines.append(f"| {group} | {g['n']} | {fmt('redundant_pairs')} | {fmt('effective_candidates')} | {fmt('uncovered')} | {fmt('allocation_gate')} | {fmt('longitudinal_outside')} | {fmt('longitudinal_inside_gap')} |")
    lines+=['','## 成本','', '| 臂 | 注册参数 | 有梯度参数 | CPU单条修正ms中位数 |','|---|---:|---:|---:|']
    for arm,items in runs.items():
        vals=list(items.values());lines.append(f"| {arm} | {vals[0]['parameter_count']} | {vals[0]['parameters_with_gradient']} | {np.median([v['latency']['refiner_only']['median_ms'] for v in vals]):.6f} |")
    lines+=['','GPU只用于一次缓存前向与小修正模块；正式训练不执行编码器、不改变B0和score_head权重，不做数据增强。CPU时间仅是缓存输入上的修正模块耗时，不是整网速度。','',
        '原始设计、用户确认的补充约定见design_original与IMPLEMENTATION_NOTES.md。逐窗指标不含原始坐标；cache/和完整predictions_local含敏感分析输入，仅留本机。保护失败不更换机制类别，但禁止推进；即使有条件信号，也需要独立端到端实验，当前不自动执行B阶段。','',
        f"本次运行墙钟秒数（含缓存/预检/评价）：{result['elapsed_seconds']:.2f}。执行与科研判定分开：完成执行不代表H成立。",'']
    (ROOT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    start=time.monotonic();cfg=read(ROOT/'CONFIG.json');verify(read(ROOT/'SOURCE_MANIFEST.json')['files'])
    if (ROOT/'DONE.json').exists():verify(read(ROOT/'DONE.json')['files']);print('ALREADY_COMPLETE_VERIFIED');return
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=cfg['device'];create_cache(device);data,head,criterion=load_cache(device)
    technical_preflight(data,head,criterion,device)
    baseline_folder=ROOT/'A0'
    if (baseline_folder/'DONE.json').exists():verify(read(baseline_folder/'DONE.json')['files']);baseline=read(baseline_folder/'RESULT.json')
    else:
        baseline,w,pred=evaluate(None,head,data['dev']);baseline.update(arm='A0',seed=42,parameter_count=0)
        save_evaluation(baseline_folder,baseline,w,pred,data['dev']);write(baseline_folder/'DONE.json',dict(files=seal([baseline_folder/'RESULT.json',baseline_folder/'window_metrics.npz',baseline_folder/'predictions_local.npz'])))
    write(ROOT/'D0.json',baseline)
    head_before={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    contract=dict(source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),cache_seal_sha256=sha(ROOT/'cache/DONE.json'),epochs=20,batch=256,device=device)
    runs={}
    for arm in ('A1','A2','A3','A4','A5','A6'):
        runs[arm]={}
        for seed in (0,1,2):
            value=run_one(arm,seed,data,head,criterion,device,dict(contract,arm=arm,seed=seed))
            if value is None:return
            runs[arm][str(seed)]=value
            decision=decide(baseline,runs) if arm=='A1' and len(runs['A1'])==3 else None
            if decision is not None:write(ROOT/'Q1_GATE.json',decision)
        write(ROOT/'INTERIM.json',dict(completed_runs=sum(len(v) for v in runs.values()),runs=runs))
        if arm=='A1' and decide(baseline,runs)['verdict']=='PROXY_INVALID':break
    decision=decide(baseline,runs)
    for k,v in head.state_dict().items():assert torch.equal(v.detach().cpu(),head_before[k]),k
    assert all(p.grad is None for p in head.parameters())
    verify(read(ROOT/'SOURCE_MANIFEST.json')['files']);verify(read(ROOT/'cache/DONE.json')['files'])
    value=dict(study='EXP_SET_INTERACT_PROBE_20261117',at=now(),execution_status='STOPPED_AS_DESIGNED' if decision['verdict']=='PROXY_INVALID' else 'COMPLETE',decision=decision,baseline=baseline,runs=runs,completed_runs=sum(len(v) for v in runs.values()),planned_max_runs=18,skipped_runs=18-sum(len(v) for v in runs.values()),base_weights_and_score_head_frozen=True,cache_augmented=False,source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),elapsed_seconds=time.monotonic()-start,device=device,base_seed=42,refiner_seeds=[0,1,2],B_stage_started=False)
    write(ROOT/'RESULT.json',value);render(value)
    files=[ROOT/'RESULT.json',ROOT/'REPORT.md',ROOT/'D0.json',ROOT/'Q1_GATE.json',ROOT/'cache/DONE.json']
    files.extend(p for p in (ROOT/'runs').rglob('DONE.json'));files.append(ROOT/'A0/DONE.json')
    write(ROOT/'DONE.json',dict(at=now(),status='SUCCESS',files=seal(files),verdict=decision['verdict'],protection_status=decision.get('protection_status'),completed_runs=value['completed_runs']))
    write(ROOT/'CURRENT.json',dict(at=now(),phase='COMPLETE',verdict=decision['verdict'],protection_status=decision.get('protection_status'),completed_runs=value['completed_runs']))
    print(f'COMPLETE {decision["verdict"]} runs={value["completed_runs"]}',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'FAILURE.json',dict(at=now(),error=str(error),traceback=traceback.format_exc()));write(ROOT/'CURRENT.json',dict(at=now(),phase='FAILED',error=str(error)));raise
