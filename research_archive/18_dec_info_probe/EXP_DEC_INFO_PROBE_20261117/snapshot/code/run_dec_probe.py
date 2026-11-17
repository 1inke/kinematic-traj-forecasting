"""Standalone C0-only EXP-DEC-INFO-PROBE runner."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '2')
import sys
import time
import traceback
import statistics
import numpy as np
import torch
from dec_io import ROOT, PREVIOUS, ARMS, SEEDS, npz, read, write, seal, sha, verify, now
from dec_model import make_basis, predict
from dec_cache import build, load
from dec_train import Trainer, new_model, evaluate, train_one, future_shuffle_check
from dec_analysis import diagnose, permutation, proxy, decision, paired


def real_preflight(data, head, criterion, device, base_params):
    folder = ROOT / 'preflight'
    t = data['fit']['tensors']
    sample = {k: v[:256] for k, v in t.items()}
    checks = []
    with torch.no_grad():
        base = torch.einsum('tp,nkpc->nktc', make_basis().to(device), sample['relative'])
        for arm in ARMS:
            model = new_model(arm, 0, device)
            p, z, shift = predict(model, head, sample)
            assert torch.equal(p, base) and not bool(shift.any()), arm
            diff = float((z - sample['logits']).abs().max())
            assert diff <= 1e-4, (arm, diff)
            checks.append(dict(arm=arm, parameter_count=sum(p.numel() for p in model.parameters()),
                               init_logit_max_error=diff, init_path_exact=True,
                               future_shuffle=future_shuffle_check(model, head, data['dev']['tensors'])))
    counts = {c['arm']: c['parameter_count'] for c in checks}
    assert counts['A1'] == 7019 and len({counts[a] for a in ARMS[2:]}) == 1
    assert abs(counts['A1L']/counts['C1']-1) <= .05 and counts['C3']/base_params <= .10
    # Compare against the unchanged A1 class, and exact complete-epoch recovery for C3.
    from probe_train import Trainer as OldTrainer
    old = OldTrainer('A1', 0, sample, head, criterion, device, epochs=2, batch=64)
    new = Trainer('A1', 0, sample, head, criterion, device, epochs=2, batch=64)
    a, b = old.epoch_step(), new.epoch_step()
    assert a['loss'] == b['loss'] and a['order_hash'] == b['order_hash']
    assert all(torch.equal(v, new.model.state_dict()[k]) for k, v in old.model.state_dict().items())
    fixture = dict(scope='TECHNICAL_ONLY', device=device, source_manifest=sha(ROOT / 'SOURCE_MANIFEST.json'))
    c3 = Trainer('C3', 0, sample, head, criterion, device, epochs=2, batch=64)
    c3.epoch_step()
    state = c3.snapshot(fixture)
    from dec_io import save_torch
    p = folder / 'resume_fixture.pt'
    save_torch(p, state)
    a = c3.epoch_step()
    resumed = Trainer('C3', 0, sample, head, criterion, device, epochs=2, batch=64)
    resumed.restore(torch.load(p, map_location='cpu', weights_only=False), fixture)
    b = resumed.epoch_step()
    assert a['loss'] == b['loss'] and a['order_hash'] == b['order_hash']
    assert all(torch.equal(v, resumed.model.state_dict()[k]) for k, v in c3.model.state_dict().items())
    write(folder / 'REAL_CACHE.json', dict(status='PASS', at=now(), reference_A1_exact=True,
          C3_checkpoint_resume_exact=True, all_arms=checks, training_fixture_is_not_scientific_result=True,
          fixture_sha256=sha(p), base_parameters=base_params))
    print('REAL_PREFLIGHT PASS', flush=True)


def describe(result):
    d = result['decision']
    lines = ['# EXP-DEC-INFO-PROBE', '',
        f"执行：**{result['execution_status']}**；判定：**{d['verdict']}**；保护：**{d['protection_status']}**。", '',
        '单一冻结B0@64 seed42，C0开发集实验；未读取测试集。三个seed属于修正模块。'
        '原设计和运行前约定见 design_original 与 IMPLEMENTATION_NOTES.md。', '',
        '| 组别 | minADE20 | minFDE20 | Top1 FDE | 未覆盖率 | 修正幅度(m) | 注册参数 | 有梯度参数 | CPU修正(ms) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    keys = ('minADE','minFDE','top1FDE','uncovered','mean_shift')
    lines.append('| A0 | ' + ' | '.join(f'{result["baseline"]["all"][k]:.6f}' for k in keys) + ' | 0 | 0 | — |')
    for arm, data in result['runs'].items():
        v = list(data.values())
        row = [f'{statistics.mean(a["clean"]["all"][k] for a in v):.6f}' for k in keys]
        row += [str(v[0]['parameter_count']), str(v[0]['parameters_with_nonzero_gradient']),
                f'{statistics.mean(a["latency"]["refiner_only"]["median_ms"] for a in v):.4f}']
        lines.append('| ' + arm + ' | ' + ' | '.join(row) + ' |')
    p = d['proxy']
    lines += ['', f"P0 minFDE 改善 {p['P0']['minFDE']['mean']:.4%}；样本标准差 {p['sigma']:.4%}；thr={p['threshold']:.4%}。", '']
    if d['verdict'] == 'PROXY_INVALID':
        lines += ['P0 未通过；按设计停止其余15次，不解读读取额外信息的效果。', '']
    else:
        lines += [f"无额外信息参照 R = **{d['reference']}**。", '',
            '| 比较 | minFDE相对改善 | minADE相对改善 |', '|---|---:|---:|']
        for k in ('Q1','Q2','Q3_vs_R','Q3_vs_A0','Q4_vs_C4','Q5_vs_C2_descriptive'):
            v = d[k]
            lines.append(f"| {k} | {v['minFDE']['mean']:.4%} | {v['minADE']['mean']:.4%} |")
        lines += ['', '## 邻居置乱与社交分组', '',
            '| 组别 | 范围 | 原始minFDE(m) | 置乱minFDE(m) | 原始对C1改善 | 置乱对C1改善 |',
            '|---|---|---:|---:|---:|---:|']
        for arm, groups in result['shuffle_comparisons'].items():
            for g, v in groups.items():
                if v is not None:
                    lines.append(f"| {arm} | {g} | {v['clean_minFDE']:.6f} | {v['shuffled_minFDE']:.6f} | "
                                 f"{v['clean_vs_C1']['minFDE']['mean']:.4%} | {v['shuffled_vs_C1']['minFDE']['mean']:.4%} |")
        lines += ['', f"机制一致性：{d['mechanism']['passes']}；社交窗口贡献占优：{d['mechanism']['social_dominates']}；"
                  f"置乱使社交收益达到预定回落幅度：{d['mechanism']['shuffle_drops_social_gain']}。", '']
    dev = result['D0']['dev']
    lines += ['## D0：候选几何诊断（dev）', '',
              '| 分组 | 窗口数 | 社交窗口比例 | 社交窗内候选选邻集合不同 | C3平均邻居数 | C4平均邻居数 | C3/C4候选选邻数相同 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    def pct(v): return '—' if v is None else f'{v:.2%}'
    def num(v): return '—' if v is None else f'{v:.3f}'
    for g, v in dev.items():
        lines.append(f"| {g} | {v['n']} | {pct(v['social_rate'])} | {pct(v['candidate_sets_vary_among_social'])} | "
                     f"{num(v['c3_neighbors_per_candidate'])} | {num(v['c4_neighbors_per_candidate'])} | {pct(v['same_count_fraction'])} |")
    lines += ['', 'C3和C4均最多读3个邻居，实际数量可能不同。两臂差异反映选邻、关系特征和有效数量共同作用。'
              '社交标签使用未修正候选和匀速邻居，组内结果供机制诊断。', '',
              '置乱保留原窗口社交标签和评价目标，只更换同邻居数量的其他窗口的整组邻居输入。'
              'C1读到的是已做历史社交混合的目标token，因此其增益不能只归因于自身运动细节。', '',
              f"本次执行 {result['elapsed_seconds']:.2f} 秒，完成 {result['completed_runs']}/18 次。"
              '没有自动启动端到端验证、旧五折或新baseline。', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    started = time.monotonic()
    source = ROOT / 'SOURCE_MANIFEST.json'
    verify(read(source)['files'])
    if (ROOT / 'DONE.json').exists():
        verify(read(ROOT / 'DONE.json')['files'])
        print('ALREADY_COMPLETE_VERIFIED', flush=True)
        return
    device = read(ROOT / 'CONFIG.json')['device']
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert device == 'cuda:0' and torch.cuda.is_available()
    build(device)
    data, head, criterion = load(device)
    base_params = read(ROOT / 'cache/VALIDATION.json')['base_params']
    head_before = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    real_preflight(data, head, criterion, device, base_params)
    phase_files = [ROOT / 'cache/DONE.json', ROOT / 'preflight/REAL_CACHE.json', ROOT / 'preflight/resume_fixture.pt']
    d0, windows = {}, {}
    for split in ('fit','dev'):
        w, d0[split] = diagnose(data[split], make_basis().to(device))
        windows[split] = w
        p = ROOT / f'D0_{split}_windows.npz'
        meta = data[split]['meta']
        npz(p, **w, ids=meta['ids'], group=meta['group'], labels=meta['labels'])
        phase_files.append(p)
    write(ROOT / 'D0.json', d0)
    phase_files.append(ROOT / 'D0.json')
    social = windows['dev']['social']
    donor, buckets = permutation(windows['dev']['neighbor_count'])
    p = ROOT / 'NEIGHBOR_SHUFFLE.json'
    write(p, dict(seed=20261117, buckets=buckets, donor_row=donor.tolist(), matches_counts=True,
                  social_groups_fixed_to_clean_inputs=True))
    phase_files.append(p)
    base, w, pred = evaluate(None, head, data['dev'], social)
    meta = data['dev']['meta']
    npz(ROOT / 'A0/window_metrics.npz', **w, ids=meta['ids'], group=meta['group'], labels=meta['labels'], social=social)
    npz(ROOT / 'A0/predictions_local.npz', **pred, ids=meta['ids'])
    write(ROOT / 'A0/RESULT.json', base)
    phase_files += [ROOT / f'A0/{n}' for n in ('window_metrics.npz','predictions_local.npz','RESULT.json')]
    prev = read(PREVIOUS / 'D0.json')['groups']['all']
    for k in ('minADE','minFDE','top1FDE','uncovered'):
        assert abs(base['all'][k]-prev[k]) < 1e-10, k
    contract = dict(source_manifest_sha256=sha(source), cache_seal_sha256=sha(ROOT/'cache/DONE.json'),
                    epochs=20, batch=256, device=device)
    runs = {}
    p0 = None
    for arm in ARMS:
        runs[arm] = {}
        for seed in SEEDS:
            row = train_one(arm, seed, data, head, criterion, device, dict(contract, arm=arm, seed=seed), social, donor)
            if row is None:
                return
            runs[arm][str(seed)] = row
        if arm == 'A1':
            p0 = proxy(runs, base)
            previous_a1 = {str(s): read(PREVIOUS / f'runs/A1_s{s}/RESULT.json')['groups']['all']['minFDE'] for s in SEEDS}
            write(ROOT / 'P0_GATE.json', dict(**p0, previous_exact_seed_FDE=previous_a1,
                  new_seed_FDE={s: r['clean']['all']['minFDE'] for s, r in runs['A1'].items()}))
            phase_files.append(ROOT / 'P0_GATE.json')
            print(f'P0 passes={p0["passes"]} gain={p0["P0"]["minFDE"]["mean"]:.6%}', flush=True)
            if not p0['passes']:
                break
        write(ROOT / 'INTERIM.json', dict(at=now(), completed_runs=sum(len(v) for v in runs.values()), runs=runs))
    d = decision(runs, base, base_params, buckets)
    compared = {}
    for a in ('C2','C3','C4'):
        if a not in runs:
            continue
        compared[a] = {}
        for g in ('all','social','non_social','stable','transition','change','decel_stop'):
            c, s = paired(runs, base, a, 'C1', g), paired(runs, base, a, 'C1', g, 'shuffled')
            compared[a][g] = (dict(clean_vs_C1=c, shuffled_vs_C1=s,
                   clean_minFDE=float(np.mean([v['clean'][g]['minFDE'] for v in runs[a].values()])),
                   shuffled_minFDE=float(np.mean([v['shuffled'][g]['minFDE'] for v in runs[a].values()]))) if c and s else None)
    for k, v in head.state_dict().items():
        assert torch.equal(v.detach().cpu(), head_before[k])
    assert all(p.grad is None for p in head.parameters())
    verify(read(source)['files'])
    verify(read(ROOT / 'cache/DONE.json')['files'])
    for a, seeds in runs.items():
        for s in seeds:
            p = ROOT / f'runs/{a}_s{s}/DONE.json'
            verify(read(p)['files'])
            phase_files.append(p)
    result = dict(study='EXP_DEC_INFO_PROBE_20261117', at=now(), decision=d,
                  execution_status='COMPLETE' if p0['passes'] else 'STOPPED_AS_DESIGNED',
                  completed_runs=sum(len(v) for v in runs.values()), planned_max_runs=18,
                  baseline=base, runs=runs, D0=d0, shuffle_comparisons=compared,
                  frozen_base_parameters=base_params, score_head_unchanged=True,
                  base_seed=42, refiner_seeds=list(SEEDS), device=device,
                  test_data_accessed=False, B_stage_started=False,
                  elapsed_seconds=time.monotonic()-started, source_manifest_sha256=sha(source))
    write(ROOT / 'RESULT.json', result)
    describe(result)
    phase_files += [ROOT / 'RESULT.json', ROOT / 'REPORT.md']
    write(ROOT / 'DONE.json', dict(at=now(), status='SUCCESS', completed_runs=result['completed_runs'],
                                  verdict=d['verdict'], files=seal(phase_files)))
    write(ROOT / 'CURRENT.json', dict(at=now(), phase=result['execution_status'], completed_runs=result['completed_runs'],
                                     verdict=d['verdict'], protection_status=d['protection_status']))
    print('COMPLETE ' + d['verdict'], flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        write(ROOT / 'FAILURE.json', dict(at=now(), error=str(e), traceback=traceback.format_exc()))
        write(ROOT / 'CURRENT.json', dict(at=now(), phase='FAILED', error=str(e)))
        raise
