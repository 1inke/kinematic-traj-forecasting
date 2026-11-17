"""CPU-only post-run audit and paired analysis; no inference or training."""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path('C:/Users/linke/Downloads/1/EXP_ETH_SPEED_20261116')
sys.path.insert(0, str(PACKAGE/'code'))
from speed_io import config, imports, now, read, sha, verify, write
from eth_speed_math import observed_speed, speed_bins, maneuver_labels, stratify, gain

c = config(); fd, fm = imports(c)
contract = read(ROOT/'INPUT_CONTRACT.json'); contract_sha = sha(ROOT/'INPUT_CONTRACT.json')
verify(contract['files'])
upstream = read(c['upstream_manifest'])['files']; verify(upstream)
diag_done = read(ROOT/'DIAG/DONE.json'); verify(diag_done['files'])
RD = ROOT/'SPEED_AUG/B0_wide_eth_E64_s42'
aug_done = read(RD/'DONE.json'); verify(aug_done['files'])
assert diag_done['input_contract_sha256'] == aug_done['input_contract_sha256'] == contract_sha
exit_record = read(ROOT/'LAST_EXIT.json')
assert exit_record['phase'] == 'aug' and exit_record['status'] == 'SUCCESS' and exit_record['exit_code'] == 0
result = read(RD/'RESULTS.json'); diag = read(ROOT/'DIAG/RESULTS.json')
baseline_dir = Path(c['baselines']['eth']['checkpoint']).parent
baseline_done = read(baseline_dir/'DONE.json'); verify(baseline_done['files'])
receipt = read(c['baselines']['eth']['receipt'])
assert result['order_hash'] == receipt['train']['order_hash']
assert result['contract']['train_index_sha256'] == receipt['contract']['train_index_sha']
epochs = [read(RD/f'ckpt_epoch{i:03d}.json') for i in range(1, 65)]
assert [e['updates'] for e in epochs] == [450*i for i in range(1, 65)]
for e in epochs:
    old = read(baseline_dir/f"ckpt_epoch{e['epoch']:03d}.json")
    assert e['order_hash'] == old['order_hash'], e['epoch']
assert sha(RD/'ckpt_epoch064.pt') == epochs[-1]['sha256']

population_checks = []


def verify_population(directory, data, profile, expected=None):
    with np.load(directory/'predictions.npz') as z:
        paths, logits = z['paths'].astype(np.float64), z['logits'].astype(np.float64)
        np.testing.assert_array_equal(z['ids'], data['ids'])
        np.testing.assert_array_equal(z['group'], data['group'])
        np.testing.assert_array_equal(z['scene'], data['scene'])
    assert paths.shape == (len(data['ids']), 20, 12, 2)
    assert np.isfinite(paths).all() and np.isfinite(logits).all()
    w = fm.per_window(paths, logits, data['future'], data['past'][:, 0])
    # Independent arithmetic, including joint coverage on the SAME candidate.
    distance = np.sqrt(np.square(paths-data['future'][:, None]).sum(-1))
    ade = distance.mean(-1); fde = distance[..., -1]; row = np.arange(len(paths))
    first = logits.argmax(-1)
    independent = dict(minADE=ade.min(-1), minFDE=fde.min(-1),
                       coverage=((ade <= .2) & (fde <= .4)).any(-1),
                       top1_good=(((ade <= .2) & (fde <= .4))[row, first]),
                       top1_ADE=ade[row, first], top1_FDE=fde[row, first])
    for key, value in independent.items():
        np.testing.assert_allclose(value, w[key], atol=1e-12, rtol=1e-12)
    with np.load(directory/'window_metrics.npz') as stored:
        np.testing.assert_array_equal(stored['ids'], data['ids'])
        for key in set(stored.files) & set(w):
            np.testing.assert_allclose(stored[key], w[key], atol=1e-7, rtol=1e-7)
    summary = stratify(w, data['past'], data['future'], profile, data['group'])
    if expected is not None:
        compare_tree(summary, expected)
    population_checks.append(dict(directory=str(directory), n=len(paths), passed=True))
    return dict(paths=paths, logits=logits, metrics=w, summary=summary,
                candidate_ADE=ade, candidate_FDE=fde)


def compare_tree(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        other = expected[key]
        if isinstance(value, dict): compare_tree(value, other)
        elif value is None: assert other is None
        else: np.testing.assert_allclose(value, other, atol=1e-10, rtol=1e-10)


test = fd.load_split(c['data_dir'], 'eth', 'test')
train = fd.load_split(c['data_dir'], 'eth', 'train'); index, val_index = fd.scene_tail_split(train['scene'])
val = fd.subset(train, val_index); del train
profile = diag['folds']['eth']['profile']
cases = {}
for fold in ('eth', 'hotel'):
    data = test if fold == 'eth' else fd.load_split(c['data_dir'], fold, 'test')
    for name, summary in diag['folds'][fold]['cases'].items():
        item = verify_population(ROOT/'DIAG'/fold/name, data, diag['folds'][fold]['profile'], summary)
        if fold == 'eth': cases[name] = item
aug = verify_population(RD/'test', test, profile, result['outputs']['test'])
aug_val = verify_population(RD/'val', val, profile, result['outputs']['val'])
# Old baseline uses different filenames; reconstruct without modifying its archive.
with np.load(baseline_dir/'val_predictions.npz') as z:
    np.testing.assert_array_equal(z['ids'], val['ids'])
    vb = fm.per_window(z['paths'], z['logits'], val['future'], val['past'][:, 0])
val_base = stratify(vb, val['past'], val['future'], profile, val['group'])
for k in ('minADE', 'minFDE', 'coverage', 'top1_ADE', 'top1_FDE'):
    np.testing.assert_allclose(val_base['overall'][k], read(baseline_dir/'metrics.json')['val']['overall'][k], atol=1e-7, rtol=1e-7)

base = cases['identity']; b, a = base['metrics'], aug['metrics']
bins = speed_bins(test['past'], profile); maneuvers = maneuver_labels(test['past'], test['future'])


def paired(mask):
    count = int(mask.sum())
    if not count: return dict(n=0)
    good_b, good_a = b['coverage'], a['coverage']
    out = dict(n=count, B0_covered=int(good_b[mask].sum()), AUG_covered=int(good_a[mask].sum()),
               rescued=int((mask & ~good_b & good_a).sum()), lost=int((mask & good_b & ~good_a).sum()),
               top1_B0_good=int(b['top1_good'][mask].sum()), top1_AUG_good=int(a['top1_good'][mask].sum()))
    for key in ('minADE', 'minFDE', 'top1_ADE', 'top1_FDE', 'same_path_FDE'):
        before, after = b[key][mask], a[key][mask]
        delta = after-before
        out[key] = dict(B0=float(before.mean()), AUG=float(after.mean()),
                        relative_gain=gain(float(before.mean()), float(after.mean())),
                        delta_sum=float(delta.sum()), improved_windows=int((delta < -1e-9).sum()),
                        worsened_windows=int((delta > 1e-9).sum()),
                        tied_windows=int((np.abs(delta) <= 1e-9).sum()),
                        B0_p95=float(np.quantile(before, .95)), AUG_p95=float(np.quantile(after, .95)),
                        B0_p99=float(np.quantile(before, .99)), AUG_p99=float(np.quantile(after, .99)))
    return out


all_mask = np.ones(len(test['ids']), bool)
paired_results = dict(overall=paired(all_mask),
    speed={name: paired(bins == i) for i, name in enumerate(('low', 'middle', 'high'))},
    maneuver={name: paired(maneuvers == name) for name in sorted(set(maneuvers))},
    original_group={name: paired(test['group'] == i) for i, name in enumerate(('stable', 'transition', 'change'))},
    speed_by_maneuver={name: {m: paired((bins == i) & (maneuvers == m)) for m in sorted(set(maneuvers))}
                      for i, name in enumerate(('low', 'middle', 'high'))})
total_ade_delta = paired_results['overall']['minADE']['delta_sum']
for family in ('speed', 'maneuver', 'original_group'):
    for block in paired_results[family].values():
        if block['n']:
            block['share_of_overall_ADE_delta'] = block['minADE']['delta_sum']/total_ade_delta

# Inspect signed endpoint error along observed heading; these are descriptive,
# and oracle paths use GT only to evaluate, never to generate a prediction.
direction = test['past'][:, 0, -1].astype(np.float64)-test['past'][:, 0, -4]
norm = np.linalg.norm(direction, axis=-1); heading_valid = norm > .03
direction /= np.maximum(norm[:, None], 1e-12)
endpoint = test['future'][:, -1].astype(np.float64)
endpoint_diagnostics = {}
for label, mask in [('all', all_mask), ('deceleration', maneuvers == 'deceleration'),
                    ('steady', maneuvers == 'steady'), ('change', test['group'] == 2)]:
    mask = mask & heading_valid; record = dict(n=int(mask.sum()))
    for name, item in [('B0', base), ('AUG', aug)]:
        row = np.arange(len(test['ids']))
        for selection, selected in [('Top1', item['logits'].argmax(-1)), ('oracle_FDE', item['candidate_FDE'].argmin(-1))]:
            error = item['paths'][row, selected, -1]-endpoint
            signed = (error*direction).sum(-1)
            record[name+'_'+selection] = dict(mean_signed_longitudinal_error=float(signed[mask].mean()),
                                               positive_error_fraction=float((signed[mask] > 0).mean()))
    endpoint_diagnostics[label] = record

timing = dict(training_session_seconds=epochs[-1]['session_elapsed_seconds'],
              epoch_seconds_min=float(min(e['epoch_wall_seconds'] for e in epochs)),
              epoch_seconds_max=float(max(e['epoch_wall_seconds'] for e in epochs)),
              epoch_seconds_median=float(np.median([e['epoch_wall_seconds'] for e in epochs])),
              reference_B0_seconds=receipt['train']['train_seconds_this_session'],
              controlled_speed_benchmark=False)
analysis = dict(at=now(), status='COMPLETE_VERIFIED', source_contract_sha256=contract_sha,
    integrity=dict(upstream_files=len(upstream), contract_files=len(contract['files']),
                   DIAG_files=len(diag_done['files']), AUG_files=len(aug_done['files']),
                   baseline_DONE_files=len(baseline_done['files']), recomputed_populations=population_checks,
                   all_64_epoch_order_hashes_match_B0=True, final_checkpoint_sha256=epochs[-1]['sha256']),
    fixed_verdict=result['comparisons'], paired=paired_results, endpoint_diagnostics=endpoint_diagnostics,
    val=dict(B0=val_base, AUG=aug_val['summary']), timing=timing,
    original_test_summaries=dict(B0=base['summary'], AUG=aug['summary'],
        B0_high_motion=cases['high_motion']['summary']),
    baseline_promoted=False, new_training=False, new_model_inference=False,
    interpretation='Speed range helps oracle candidates on ETH, especially fast steady motion; change/deceleration and Top1 endpoint errors remain problematic. Single seed, explored test, no causal proof.')
write(ROOT/'ANALYSIS_20261116.json', analysis)

def pct(x): return f'{100*x:+.2f}%'
def gain_text(before, after): return pct((after-before)/before)

lines = ['# ETH 速度诊断与宽范围增强：完成结果分析', '',
 '结论：宽范围速度增强在本次 ETH、seed42 的整体 minADE/minFDE 上有效，达到了预设探索条件；收益集中在高速与近似保持运动的窗口。Top1、覆盖率和减速/变化运动存在代价，暂不替换 baseline。', '',
 '实验于 2026-11-16 23:49:15 完成。64 轮、28800 次更新，固定最终 EMA；与 B0 使用相同 95% 训练索引，全部 64 轮批次顺序散列一致。任务退出 0，预测与指标保存完整。', '',
 '## 整体结果', '', '| 指标 | 原 B0 | 宽范围增强 | 变化（负数为误差下降） |', '|---|---:|---:|---:|']
for metric in ('minADE', 'minFDE', 'top1_ADE', 'top1_FDE'):
    x, y = base['summary']['overall'][metric], aug['summary']['overall'][metric]
    lines.append(f'| {metric}（m） | {x:.6f} | {y:.6f} | {gain_text(x,y)} |')
for metric, title in [('coverage', '同候选覆盖率'), ('top1_good', 'Top1 达标率')]:
    x, y = base['summary']['overall'][metric], aug['summary']['overall'][metric]
    lines.append(f'| {title} | {x:.2%} | {y:.2%} | {(y-x)*100:+.2f} 个百分点 |')
overall = paired_results['overall']
lines += ['', f"覆盖窗口 {overall['B0_covered']}→{overall['AUG_covered']}，救回 {overall['rescued']}、丢失 {overall['lost']}。Top1 达标窗口 {overall['top1_B0_good']}→{overall['top1_AUG_good']}。",
 'minADE/minFDE 是 20 条候选中分别取最小值；二者可以来自不同候选。Top1 是模型按分数选出的轨迹，整体误差恶化，不能把最优候选进步等同于实际选取轨迹更可靠。',
 f"逐窗比较：ADE 改善 {overall['minADE']['improved_windows']}、恶化 {overall['minADE']['worsened_windows']}；FDE 改善 {overall['minFDE']['improved_windows']}、恶化 {overall['minFDE']['worsened_windows']}。均值改善不代表多数窗口的 FDE 都改善。", '']
for family, title in [('speed', '按训练速度分界'), ('original_group', '按原运动变化分组'), ('maneuver', '按机动类型')]:
    lines += [f'## {title}', '', '| 组 | n | ADE B0→增强 | ADE 变化 | FDE B0→增强 | FDE 变化 |', '|---|---:|---:|---:|---:|---:|']
    for name, row in paired_results[family].items():
        if not row['n']: continue
        ade, fde = row['minADE'], row['minFDE']
        label = name + ('（小样本）' if row['n'] < 20 else '')
        lines.append(f"| {label} | {row['n']} | {ade['B0']:.4f}→{ade['AUG']:.4f} | {pct(-ade['relative_gain'])} | {fde['B0']:.4f}→{fde['AUG']:.4f} | {pct(-fde['relative_gain'])} |")
    lines.append('')
lines += [f"速度分界为训练观测 p50={profile['p50']:.4f}、p90={profile['p90']:.4f} m/s。高速组贡献约 {paired_results['speed']['high']['share_of_overall_ADE_delta']:.1%} 的整体 ADE 减少；分组共用原始观测速度，不在缩放后重新分组。", '',
 '减速组 FDE 恶化与变化组 FDE 恶化表明：扩大运动尺度范围没有同时解决速度状态的变化。机动标签使用未来真值，仅用于评价；分组互相关联，不代表独立因果机制。停止与纯转向样本少，只作描述。', '',
 '## 减速过冲与高速交叉分组', '',
 '在高速×近似保持组（56 窗），minFDE 从 1.1986 降至 0.6622 m；高速×减速组（56 窗）却从 0.6758 增至 0.9317 m。同属高速观测组，不同机动类型的变化方向相反；这不是组内速度分布完全匹配的比较，不能据此单独认定因果。', '',
 f"对观测最后三步方向长度大于 0.03 m 的 {endpoint_diagnostics['deceleration']['n']} 个减速窗口，Top1 终点误差在观测前进方向的均值由 {endpoint_diagnostics['deceleration']['B0_Top1']['mean_signed_longitudinal_error']:.3f} 增至 {endpoint_diagnostics['deceleration']['AUG_Top1']['mean_signed_longitudinal_error']:.3f} m；正值表示沿观测方向超过真实终点。这个结果与减速时选出的轨迹过冲加重一致，但属于事后描述，不能据此证明是哪一训练机制导致。", '',
 '对近似保持运动的相应 110 窗，Top1 沿观测方向的终点误差则从 -1.281 缩至 -0.663 m，表现为原先预测不足的偏差减小。两种状态对照提示，下一步应检查状态变化和候选分数，而非继续无条件扩大速度增强。', '',
 '## 尾部与零训练参照', '']
for metric in ('minADE', 'minFDE'):
    row = overall[metric]
    lines.append(f"- {metric} p95：{row['B0_p95']:.4f}→{row['AUG_p95']:.4f} m；p99：{row['B0_p99']:.4f}→{row['AUG_p99']:.4f} m。")
hm = cases['high_motion']['summary']['overall']
lines += ['', '| 方案 | minADE（m） | minFDE（m） | 覆盖率 | Top1 FDE（m） |', '|---|---:|---:|---:|---:|']
for name, block in [('原 B0',base['summary']['overall']),('零训练 high_motion',hm),('宽范围增强',aug['summary']['overall'])]:
    lines.append(f"| {name} | {block['minADE']:.6f} | {block['minFDE']:.6f} | {block['coverage']:.2%} | {block['top1_FDE']:.6f} |")
lines += ['', f"零训练 high_motion 的 minADE/minFDE 为 {hm['minADE']:.6f}/{hm['minFDE']:.6f}；本次增强为 {aug['summary']['overall']['minADE']:.6f}/{aug['summary']['overall']['minFDE']:.6f}。增强相比零训练 ADE 再降 3.28%，FDE 却增加约 0.31%。两者没有叠加，本轮不新增推理来搜索最佳组合。零训练方案的 Top1 误差也未优于原 B0，不能直接当部署升级方案；可以继续作为低成本的候选生成对照。", '',
 '预处理链的全点匹配、统一跳帧/坐标设置和视频元信息检查通过；ETH 缩放有效而 HOTEL 没有同样改善，支持速度范围是相关因素。仍未独立验证上游物理标定与视频对齐，也不能证明速度是唯一原因。', '',
 '## 验证集、耗时与证据边界', '']
for metric in ('minADE','minFDE','coverage','top1_ADE','top1_FDE'):
    x, y = val_base['overall'][metric], aug_val['summary']['overall'][metric]
    lines.append(f'- 原 scene_tail 验证集 {metric}：{x:.6f}→{y:.6f}。')
lines += ['', f"训练会话约 {timing['training_session_seconds']/60:.1f} 分钟；轮次约 {timing['epoch_seconds_min']:.1f}–{timing['epoch_seconds_max']:.1f} 秒，比预计 45–80 分钟长。轮速明显波动，缺少受控负载测速，不能把耗时差归因于增强倍率本身。", '',
 f"独立复核：456 项旧源、DIAG {len(diag_done['files'])} 项输出、增强 {len(aug_done['files'])} 项输出散列通过；11 组诊断和两组增强的保存预测对真实目标重算逐窗及分层指标通过，旧 B0 验证集也重建核对。全 64 轮批次顺序一致。未启动新训练、额外 seed、模型推理或旧五折。", '',
 '仅有一个 seed 和 ETH 折，而且测试结果已用于探索设计；不作显著性、五折泛化或真正理解运动状态的断言。U(0.5,2.0) 同时扩大范围并改变平均增强速度，不能分离二者的贡献。', '',
 '建议保留为速度增强的正向探索结果；若继续研究，优先检查减速/停止时的终点过冲与候选选择问题，再决定是否做少量配对确认。本轮不自动开展后续实验，也不升级 baseline。', '']
(ROOT/'RESULT_ANALYSIS_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
write(ROOT/'ANALYSIS_HEALTH_20261116.json', dict(at=now(), status='PASS',
      source_contract_sha256=contract_sha, final_checkpoint_sha256=epochs[-1]['sha256'],
      DIAG_sealed_files=len(diag_done['files']), AUG_sealed_files=len(aug_done['files']),
      recomputed_prediction_populations=len(population_checks),
      baseline_validation_recomputed=True, new_training=False, new_inference=False))
print(json.dumps(dict(status='COMPLETE_VERIFIED', verdict=result['comparisons'],
    paired_overall=overall, speed=paired_results['speed'], original_group=paired_results['original_group'],
    val_B0=val_base['overall'], val_AUG=aug_val['summary']['overall'], timing=timing,
    endpoint_diagnostics=endpoint_diagnostics), ensure_ascii=False, indent=2))
