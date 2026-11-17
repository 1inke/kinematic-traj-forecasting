"""EXP-DECEL-DIAG: read saved outputs only, no training/inference/GPU imports."""
from __future__ import annotations

from datetime import datetime,timezone,timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.dont_write_bytecode = True
import numpy as np
from decel_metrics import compute,summarize,decide,HISTORY_NAMES

ROOT = Path(__file__).resolve().parents[1]


def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,p)


def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def verify(files):
    for p,digest in files.items():
        if sha(p)!=digest:raise ValueError('INPUT_HASH_MISMATCH '+p)


def align(pred,data):
    ids=np.asarray(data['ids']);saved=np.asarray(pred['ids'])
    if len(np.unique(ids))!=len(ids) or len(np.unique(saved))!=len(saved):raise ValueError('DUPLICATE_IDS')
    if set(ids.tolist())!=set(saved.tolist()):raise ValueError('ID_UNIVERSE_MISMATCH')
    lookup={key:i for i,key in enumerate(saved.tolist())};order=np.array([lookup[key] for key in ids.tolist()])
    for name,expected in [('future',data['future']),('past',data['past'][:,0])]:
        if name in pred and not np.array_equal(pred[name][order],expected):raise ValueError('SAVED_'+name.upper()+'_MISMATCH')
    for name in ('group','groups'):
        if name in pred and not np.array_equal(pred[name][order],data['group']):raise ValueError('GROUP_MISMATCH')
    if 'scene' in pred and 'scene' in data and not np.array_equal(pred['scene'][order],data['scene']):raise ValueError('SCENE_MISMATCH')
    return pred['paths'][order],pred['logits'][order],int((order!=np.arange(len(order))).sum())


def save_npz(p,**kwargs):
    temp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('xb') as f:
        np.savez_compressed(f,**kwargs);f.flush();os.fsync(f.fileno())
    os.replace(temp,p)


def render(result):
    primary=result['datasets']['ETH']['AUG_s42']
    lines=['# EXP-DECEL-DIAG：候选生成与选择诊断','',
        f"主判定：**{primary['decision']['label']}**（主类 {primary['decision']['main_class']}，细分 {primary['decision']['B_subtype']}）。原文A/B数值条件未改。",'',
        '全部为已有预测与真值的CPU分析，没有训练、模型推理或GPU导入。细分只提供探索方向，不能证明历史本质不可预测，或打分头完全没有利用历史。','',
        '## 减速组主表（t=12，方向有效窗口）','',
        '| 数据/模型 | n | 全部过冲 | 纵向包络覆盖 | 纵向可达≤.4m | Top1过冲>.4m | 最优名次中位数 | P_dec均值 | 判定 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for dataset,models in result['datasets'].items():
        for model,item in models.items():
            b=item['cohorts']['deceleration']['all_history'];s=b['t12'];d=item['decision']
            if not b['n']:continue
            lines.append(f"| {dataset}/{model} | {b['n']} | {s['all_overshoot_rate']:.1%} | {s['envelope_coverage_rate']:.1%} | {s['longitudinal_reachable_rate']:.1%} | {s['top1_overshoot_rate']:.1%} | {s['best_rank_median']:.1f} | {b['mean_P_dec']:.3f} | {d['label']} |")
    lines+=['','## 最优候选的纵向与横向误差（ETH减速组）','',
        '| 模型 | 纵向带符号均值 | 纵向绝对均值 / p90 | 横向带符号均值 | 横向绝对均值 / p90 | minFDE |',
        '|---|---:|---:|---:|---:|---:|']
    for model,item in result['datasets']['ETH'].items():
        s=item['cohorts']['deceleration']['all_history']['t12'];p=s['best_prog_error'];q=s['best_lat_error']
        lines.append(f"| {model} | {p['signed_mean']:.4f} | {p['absolute_mean']:.4f} / {p['absolute_p90']:.4f} | {q['signed_mean']:.4f} | {q['absolute_mean']:.4f} / {q['absolute_p90']:.4f} | {s['best_displacement_error_mean']:.4f} |")
    lines+=['','纵横误差都以米计；带符号均值接近0不表示每个窗口准确。t=6的对应量在JSON逐格保留。','',
        '## 减速候选数量和匹配速度比','',
        '| 数据/模型 | 无减速候选 | 1–2条 | ≥3条 | 平均条数 | 存在速度比误差≤.15 | Top1 r均值 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for dataset,models in result['datasets'].items():
        for model,item in models.items():
            b=item['cohorts']['deceleration']['all_history'];c=b['n_decel_distribution']
            lines.append(f"| {dataset}/{model} | {c['zero']:.1%} | {c['one_to_two']:.1%} | {c['three_or_more']:.1%} | {b['mean_n_decel']:.2f} | {b['speed_ratio_reachable_rate']:.1%} | {b['mean_top1_r']:.3f} |")
    lines+=['','## Q3：历史趋势（ETH减速组）','',
        '| 模型/历史 | n | 纵向可达 | 全部过冲 | Top1过冲 | P_dec | 小样本 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for model,item in result['datasets']['ETH'].items():
        for name,b in item['cohorts']['deceleration']['by_history'].items():
            if not b['n']:
                lines.append(f'| {model}/{name} | 0 | — | — | — | — | 是 |');continue
            s=b['t12'];lines.append(f"| {model}/{name} | {b['n']} | {s['longitudinal_reachable_rate']:.1%} | {s['all_overshoot_rate']:.1%} | {s['top1_overshoot_rate']:.1%} | {b['mean_P_dec']:.3f} | {'是' if b['small_group'] else '否'} |")
    lines+=['','## 校准表：全部方向有效窗口','',
        '以下不只取未来减速的样本。完整JSON还包含各机动和历史趋势档的校准表。','',
        '| 数据/模型 | P_dec档 | n | 平均P_dec | 真实r_g≤.8比例 | 实际−预测 | 小样本 |',
        '|---|---|---:|---:|---:|---:|---|']
    for dataset,models in result['datasets'].items():
        for model,item in models.items():
            for b in item['cohorts']['all']['all_history']['calibration']:
                if not b['n']:
                    lines.append(f"| {dataset}/{model} | {b['lower']:.1f}–{b['upper']:.1f} | 0 | — | — | — | 是 |")
                else:
                    lines.append(f"| {dataset}/{model} | {b['lower']:.1f}–{b['upper']:.1f} | {b['n']} | {b['mean_P_dec']:.3f} | {b['actual_deceleration_rate']:.3f} | {b['gap_actual_minus_predicted']:+.3f} | {'是' if b['small_group'] else '否'} |")
    lines+=['','## 状态变化与对照（ETH速度增强）','',
        '| 机动 | 总窗 / 有效窗 | 纵向可达 | Top1过冲 | 最优纵向绝对误差 | 最优横向绝对误差 |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ('deceleration','stop','turn_and_speed','steady','acceleration','state_change'):
        item=primary['cohorts'][name];b=item['all_history']
        if not b['n']:continue
        s=b['t12'];lines.append(f"| {name} | {item['total_windows']} / {b['n']} | {s['longitudinal_reachable_rate']:.1%} | {s['top1_overshoot_rate']:.1%} | {s['best_prog_error']['absolute_mean']:.4f} | {s['best_lat_error']['absolute_mean']:.4f} |")
    lines+=['','## 判定证据和一致性','', '```json',json.dumps(primary['decision'],ensure_ascii=False,indent=2),'```','',
        'C0五seed分别保留，未把相同4753窗重复叠成更大样本。相同判定支持“C0也存在该现象”；不同判定不能仅凭这一对照归因ETH或速度偏移，因为训练池和数据分布也不同。','',
        '原文主判定为减速组；停止/变速转向及steady/acceleration是补充，不事后替换主对象或阈值。纵向可达只检查一维，不保证存在横向和终点也达标的完整候选。','',
        'B1只表示这些条件下出现与已有历史线索未被充分利用相符的现象；B2不能证明本质不确定。小样本分层不作显著性或因果结论。','',
        '主方向规则使用三步平均位移长度>.03m；前次三步总位移规则作为敏感性分析保留在JSON，不用它替换主判定。','',
        f"CPU分析耗时 {result['analysis_seconds']:.2f} 秒；数据集/模型共 {sum(len(v) for v in result['datasets'].values())} 组。原输入SHA及输出封存见SOURCE_MANIFEST.json、FINAL_OUTPUT_SEAL.json和独立核验文件。未启动其他实验或推送GitHub。",'']
    (ROOT/'DECEL_DIAG.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    started=time.monotonic();config=read(ROOT/'CONFIG.json')
    write(ROOT/'CURRENT.json',dict(at=now(),phase='VERIFY_INPUTS'))
    manifest=read(ROOT/'SOURCE_MANIFEST.json');verify(manifest['files'])
    if (ROOT/'RESULT.json').exists() and read(ROOT/'RESULT.json').get('status')=='SUCCESS':
        verify(read(ROOT/'FINAL_OUTPUT_SEAL.json')['files']);print('ALREADY_COMPLETE_VERIFIED');return
    helper=config['maneuver_source']
    spec=importlib.util.spec_from_file_location('frozen_eth_speed_math',helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sys.path.insert(0,config['fivefold_code'])
    from fivefold_data import load_split
    eth=load_split(config['data_dir'],'eth','test');c0=arrays(config['C0_dev'])
    populations=dict(ETH=eth,C0_dev=c0);datasets={};saved={};alignment=[]
    for item in config['populations']:
        name,model=item['dataset'],item['model']
        write(ROOT/'CURRENT.json',dict(at=now(),phase='ANALYZE',dataset=name,model=model))
        data=populations[name];pred=arrays(item['prediction'])
        paths,logits,reordered=align(pred,data)
        labels=module.maneuver_labels(data['past'],data['future'])
        w=compute(data['past'],data['future'],paths,logits)
        w.update(ids=data['ids'],original_group=data['group'],maneuver=labels)
        folder=ROOT/'populations'/name/model;folder.mkdir(parents=True,exist_ok=True)
        save_npz(folder/'decel_window_metrics.npz',**w)
        summary=summarize(w,labels);decision=decide(w,labels)
        datasets.setdefault(name,{})[model]=dict(n_total=len(labels),direction_valid=int(w['direction_valid'].sum()),
            direction_invalid=int((~w['direction_valid']).sum()),cohorts=summary,decision=decision,
            sensitivity_previous_total_direction=dict(cohorts=summarize(w,labels,'direction_valid_previous_total'),
                                                        decision=decide(w,labels,'direction_valid_previous_total')))
        alignment.append(dict(dataset=name,model=model,n=len(labels),reordered_rows=reordered,
                              prediction_sha256=sha(item['prediction']),IDs_and_saved_targets='PASS'))
        saved[name+'/'+model]=str(folder/'decel_window_metrics.npz')
    assert 'torch' not in sys.modules,'MODEL_LIBRARY_MUST_NOT_BE_IMPORTED'
    duration=time.monotonic()-started
    value=dict(study='EXP_DECEL_DIAG_20261117',status='COMPLETE',at=now(),source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
        definitions=read(ROOT/'RULES.json'),datasets=datasets,alignment_checks=alignment,
        analysis_seconds=duration,training=False,model_inference=False,torch_imported=False,GPU_used=False,
        raw_coordinates_in_output=False,per_window_files=saved)
    write(ROOT/'DECEL_DIAG.json',value);render(value)
    files={str(p):sha(p) for p in ROOT.rglob('*') if p.is_file() and
           (p.name in ('DECEL_DIAG.json','DECEL_DIAG.md') or p.name=='decel_window_metrics.npz')}
    write(ROOT/'FINAL_OUTPUT_SEAL.json',dict(at=now(),source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),files=files))
    write(ROOT/'RESULT.json',dict(at=now(),status='SUCCESS',primary_decision=datasets['ETH']['AUG_s42']['decision'],
        populations=len(saved),analysis_seconds=duration,training=False,model_inference=False,GPU_used=False))
    write(ROOT/'CURRENT.json',dict(at=now(),phase='COMPLETE',primary=datasets['ETH']['AUG_s42']['decision']['label']))
    print(json.dumps(dict(status='SUCCESS',primary=datasets['ETH']['AUG_s42']['decision'],analysis_seconds=duration),ensure_ascii=False))


if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(ROOT/'FAILURE.json',dict(at=now(),error=str(e),traceback=traceback.format_exc()))
        write(ROOT/'CURRENT.json',dict(at=now(),phase='FAILED',error=str(e)))
        raise
