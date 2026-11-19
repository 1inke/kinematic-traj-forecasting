"""Offline comparison from sealed predictions, followed by output verification."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
OLD = WORK / 'qoder_work/NEXT_PHASE_EXEC_20261114'
DEC = WORK / 'qoder_work/EXP_DEC_INFO_PROBE_EXEC_20261117'
LABEL = Path('C:/Users/linke/Downloads/1/EXP_ETH_SPEED_20261116/code/eth_speed_math.py')
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(LABEL.parent))
from exchange_null import quality, group_masks, motion_arrays, pooled, fit_reference
from eth_speed_math import maneuver_labels
from prec_mechanism import analyze_mechanism

METRICS = ('minADE', 'minFDE')
MOTION = ('speed_p95', 'accel_mean', 'jerk_mean', 'junction_accel_mean')
SCHEMES = {'CV': (0,), 'stationary': (1,), 'CV_stationary': (0, 1)}
ARMS = {'B0': range(42, 47), 'E3_self': range(42, 45)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def status(stage, **values):
    write(ROOT/'CURRENT.json', dict(at=datetime.now().astimezone().isoformat(), phase=stage, **values))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as dataset:
        return {key: dataset[key] for key in dataset.files}


def input_manifest():
    historical = {}
    for p in (OLD/'SOURCE_MANIFEST.json', OLD/'FINAL_OUTPUT_SEAL.json', DEC/'SOURCE_MANIFEST.json',
              DEC/'cache/DONE.json', OLD/'runs/E1C_E64_s42/FINAL_LOCK.json'):
        historical.update(read(p)['files'])
    files = [LABEL, OLD/'exchange_null.py', OLD/'loss_independent.py', OLD/'C0/MOTION_REFERENCE.json',
             OLD/'C0/EVALUATION_CONTRACT.json', OLD/'runs/E1C_E64_s42/final.pt',
             *(DEC/'cache'/name for name in ('fit.pt', 'dev.pt', 'frozen_assets.pt'))]
    for split in ('fit', 'dev'):
        files.append(OLD/f'C0/{split}.npz')
    for arm, seeds in ARMS.items():
        for seed in seeds:
            parent = run_folder(arm, seed)
            receipt = read(parent/'EVALUATION_DONE.json')['files']
            for split in ('fit', 'dev'):
                p = parent/f'{split}/predictions.npz'
                historical[str(p)] = receipt[str(p)]
                files.append(p)
    digests = {}
    for p in files:
        assert sha(p) == historical[str(p)], p
        digests[str(p)] = historical[str(p)]
    for p in (ROOT/'PROTOCOL.md', ROOT/'run_prec_diag.py', ROOT/'prec_mechanism.py',
              *sorted((ROOT/'standalone').glob('*.ps1'))):
        digests[str(p)] = sha(p)
    return digests


def run_folder(arm, seed):
    return OLD/'runs'/(f'E1C_E64_s{seed}' if arm == 'B0' else f'E3_self_s{seed}')


def load_prediction(folder, split, data):
    saved = arrays(folder/f'{split}/predictions.npz')
    ids = data['ids']
    lookup = {v: i for i, v in enumerate(saved['ids'])}
    assert len(lookup) == len(ids) == len(saved['ids']) and set(lookup) == set(ids)
    index = np.array([lookup[v] for v in ids])
    for name, expected in [('future',data['future']),('past',data['past'][:,0]),('groups',data['group'])]:
        np.testing.assert_array_equal(saved[name][index], expected)
    paths, logits = saved['paths'][index], saved['logits'][index]
    assert paths.shape == (len(ids), 20, 12, 2) and logits.shape == (len(ids), 20)
    assert np.isfinite(paths).all() and np.isfinite(logits).all()
    return paths, logits


def slot_stats(q):
    sole = q['good'] & (q['good'].sum(1, keepdims=True) == 1)
    return dict(unique_joint_coverage=sole.sum(0), ADE_wins=np.bincount(q['ADE'].argmin(1), minlength=20),
                FDE_wins=np.bincount(q['FDE'].argmin(1), minlength=20))


def fixed_slots(paths, target):
    q = quality(paths, target)
    stats = slot_stats(q)
    rank = np.lexsort((np.arange(20), stats['ADE_wins']+stats['FDE_wins'], stats['unique_joint_coverage']))
    return rank[:2], {k: v.tolist() for k, v in stats.items()}


def window_metrics(q, top):
    row = np.arange(len(top))
    return dict(minADE=q['ADE'].min(-1), minFDE=q['FDE'].min(-1), top1_ADE=q['ADE'][row,top],
                top1_FDE=q['FDE'][row,top], covered=q['coverage'], top1_good=q['good'][row,top])


def window_exchange(base, removed, post, union, sorted_drop):
    d = dict(retired_slots=sorted_drop)
    for key in METRICS:
        d[key+'_net_gain_m'] = base[key] - post[key]
        d[key+'_delete_harm_m'] = removed[key] - base[key]
        d[key+'_add_after_delete_gain_m'] = removed[key] - post[key]
        d[key+'_lost_vs_union_m'] = post[key] - union[key]
        np.testing.assert_allclose(d[key+'_net_gain_m'],
                                  d[key+'_add_after_delete_gain_m']-d[key+'_delete_harm_m'], atol=1e-13)
    d.update(coverage_gained=post['covered'] & ~base['covered'], coverage_lost=base['covered'] & ~post['covered'],
             deleted_original_coverage=base['covered'] & ~removed['covered'],
             restored_after_deletion=base['covered'] & ~removed['covered'] & post['covered'])
    return d


def groups(data, labels):
    masks = group_masks(data['group'])
    masks.update({label:labels == label for label in sorted(set(labels))})
    masks['turn_any'] = np.isin(labels, ['turn','turn_and_speed'])
    masks['decel_stop'] = np.isin(labels, ['deceleration','stop'])
    return masks


def summarize(values, masks, base=None, change=None):
    output = {}
    for name, mask in masks.items():
        n = int(mask.sum())
        if not n:
            continue
        a = {k: float(v[mask].mean()) for k, v in values.items()}
        a['n'] = n
        a['no_joint_good_candidate_n'] = int((~values['covered'][mask]).sum())
        a['good_exists_top1_missed_n'] = int((values['covered'][mask] & ~values['top1_good'][mask]).sum())
        if base is not None:
            for key in ('minADE','minFDE','top1_FDE'):
                ref = base[name][key]
                a[key+'_relative_gain'] = (ref-a[key])/ref if ref>0 else None
            a['coverage_gain_pp'] = 100*(a['covered']-base[name]['covered'])
        if change is not None:
            for key in METRICS:
                value = change[key+'_net_gain_m'][mask]
                a[key+'_improved_n'] = int((value>1e-10).sum())
                a[key+'_worsened_n'] = int((value< -1e-10).sum())
                a[key+'_positive_gain_sum_m'] = float(value[value>0].sum())
                a[key+'_negative_gain_sum_m'] = float(-value[value<0].sum())
                for field in ('delete_harm_m','add_after_delete_gain_m','lost_vs_union_m'):
                    a[key+'_'+field+'_sum'] = float(change[key+'_'+field][mask].sum())
            for key in ('coverage_gained','coverage_lost','deleted_original_coverage','restored_after_deletion'):
                a[key+'_n'] = int(change[key][mask].sum())
        output[name] = a
    return output


def motion_summary(paths, past, q, top, masks, fit, tail, gt=None):
    motion = motion_arrays(paths, past)
    out = {}
    for name in ('overall','stable','transition','change','turn_any','decel_stop','stationary','steady'):
        if name not in masks or not (mask := masks[name]).any():
            continue
        row = np.flatnonzero(mask)
        ports = {'allK': {k:v[mask] for k,v in motion.items()},
                 'J_oracle':{k:v[row,q['J_winner'][mask]] for k,v in motion.items()},
                 'Top1':{k:v[row,top[mask]] for k,v in motion.items()}}
        out[name] = {}
        for port, values in ports.items():
            stats = pooled(values)
            reference = fit if port == 'allK' else gt[name]
            distance = {k: abs(float(np.log(stats[k]/reference[k]))) if min(stats[k],reference[k])>0 else None for k in MOTION}
            tails = {k: float((v>tail[k]).mean()) for k,v in values.items()}
            out[name][port] = dict(metrics=stats, abs_log_ratio=distance, exceedance_rate=tails)
    return out


def motion_differences(base, updated):
    flags = []
    undefined = []
    for name in ('overall','stable','transition','change'):
        for port, b in base[name].items():
            p = updated[name][port]
            if port != 'Top1':
                for k in MOTION:
                    x,y = p['abs_log_ratio'][k],b['abs_log_ratio'][k]
                    if x is None or y is None:
                        undefined.append(dict(group=name,port=port,metric=k))
                    elif x-y > .05:
                        flags.append(dict(group=name,port=port,metric=k,increase=x-y,kind='abs_log_ratio'))
            for k,x in p['exceedance_rate'].items():
                diff = x-b['exceedance_rate'][k]
                if diff > .01:
                    flags.append(dict(group=name,port=port,metric=k,increase=diff,kind='tail_rate'))
    return flags, undefined


def run_case(paths, logits, data, labels, indices, fitref, tail):
    y, past = data['future'].astype(float), data['past'][:,0].astype(float)
    row = np.arange(len(y))
    top = logits.argmax(-1)
    q = quality(paths, y)
    base = window_metrics(q, top)
    masks = groups(data,labels)
    b = summarize(base,masks)
    gt = {k:pooled(motion_arrays(y[m,None],past[m])) for k,m in masks.items() if m.any()}
    base_motion = motion_summary(paths,past,q,top,masks,fitref,tail,gt)
    # Stable descending sort, reversed for deletion, preserves the original Top1 under ties.
    bottom = np.argsort(-logits,axis=1,kind='stable')[:,::-1]
    cv = past[:,-1,None]+np.arange(1,13)[None,:,None]*(past[:,-1]-past[:,-2])[:,None]
    still = np.broadcast_to(past[:,-1,None],cv.shape)
    prior = np.stack((cv,still),axis=1)
    detail = dict(ids=data['ids'],group=data['group'],maneuver=labels,
                  CV_stationary_identical=(cv==still).all((1,2)),original_top1_index=top)
    detail.update({f'base__{k}':v for k,v in base.items()})
    result = {}
    for scheme, prior_index in SCHEMES.items():
        new = prior[:,prior_index]
        united = quality(np.concatenate((paths,new),axis=1),y)
        uv = window_metrics(united,top)
        result['union_'+scheme] = dict(K=paths.shape[1]+len(prior_index),groups=summarize(uv,masks,b))
        for k,v in uv.items():
            detail['union_'+scheme+'__'+k] = v
        for policy in ('lowest_logit','fixed_fit'):
            drop = bottom[:,:len(prior_index)] if policy == 'lowest_logit' else np.broadcast_to(indices[:len(prior_index)],(len(y),len(prior_index)))
            candidate = paths.copy()
            candidate[row[:,None],drop] = new
            qq = quality(candidate,y)
            values = window_metrics(qq,top)
            assert (values['minADE']>=uv['minADE']-1e-12).all() and (values['minFDE']>=uv['minFDE']-1e-12).all()
            if policy == 'lowest_logit':
                assert not np.any(drop==top[:,None])
                np.testing.assert_array_equal(values['top1_FDE'],base['top1_FDE'])
            masked = {k:v.copy() for k,v in q.items() if k in ('ADE','FDE','good')}
            masked['ADE'][row[:,None],drop] = np.inf
            masked['FDE'][row[:,None],drop] = np.inf
            masked['good'][row[:,None],drop] = False
            removed = {key:masked[metric].min(1) for key,metric in [('minADE','ADE'),('minFDE','FDE')]}
            removed['covered'] = masked['good'].any(1)
            exchange = window_exchange(base,removed,values,uv,drop)
            key = policy+'_'+scheme
            motion = motion_summary(candidate,past,qq,top,masks,fitref,tail,gt)
            flags, undefined = motion_differences(base_motion,motion)
            result[key] = dict(K=20, groups=summarize(values,masks,b,exchange),
                               motion=motion,motion_flags=flags,undefined_motion=undefined,
                               original_top1_replaced_n=int((drop==top[:,None]).any(1).sum()))
            for k,v in (values|exchange).items():
                detail[key+'__'+k] = v
    return dict(baseline=b, baseline_motion=base_motion, options=result), detail


def aggregate(all_runs):
    panel = {}
    for family, runs in all_runs.items():
        panel[family] = {}
        for key, sample in next(iter(runs.values()))['options'].items():
            cells = [v['options'][key] for v in runs.values()]
            by_group = {}
            for group, entry in sample['groups'].items():
                by_group[group] = {k:float(np.mean([c['groups'][group][k] for c in cells]))
                                   for k,v in entry.items() if v is not None and all(c['groups'][group][k] is not None for c in cells)}
            overall = by_group['overall']
            protected_failures = [dict(seed=seed,group=g,metric=k,harm=-c['groups'][g][k+'_relative_gain'])
                                  for seed,c in zip(runs,cells)
                                  for g,k,threshold in [('overall','top1_FDE',.02),('change','minFDE',.01),
                                                        ('turn_any','minFDE',.01),('decel_stop','minFDE',.01)]
                                  if -c['groups'][g][k+'_relative_gain']>threshold]
            panel[family][key] = dict(K=sample['K'], groups=by_group,
                gain_screen=overall['minADE_relative_gain']>=.02 and overall['minFDE_relative_gain']>=.025,
                protected_failures=protected_failures, motion_flags=sum(len(c.get('motion_flags',[])) for c in cells),
                undefined_motion=sum(len(c.get('undefined_motion',[])) for c in cells),
                original_top1_replaced_n=sum(c.get('original_top1_replaced_n',0) for c in cells))
    return panel


def render(result):
    lines = ['# PREC-DIAG', '',
             '使用 C0 已保存完整预测。以下改善为各 seed 对原模型的相对改善均值。', '',
             '| 底座 | 方案 | K | minADE改善 | minFDE改善 | change FDE改善 | Top1 FDE改善 | 联合覆盖变化 | 精度筛选 | 误差保护触发 | 运动学保护触发 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|']
    for arm, options in result['panel'].items():
        for key,v in options.items():
            a = v['groups']['overall']
            change = v['groups']['change']['minFDE_relative_gain']
            error = str(len(v['protected_failures'])) if v['K']==20 else '—'
            motion = str(v['motion_flags']) if v['K']==20 else '—'
            lines.append(f"| {arm} | {key} | {v['K']} | {a['minADE_relative_gain']:.3%} | {a['minFDE_relative_gain']:.3%} | "
                         f"{change:.3%} | {a['top1_FDE_relative_gain']:.3%} | {a['coverage_gain_pp']:+.3f} pp | "
                         f"{'达到' if v['gain_screen'] else '未达到'} | {error} | {motion} |")
    lines += ['', '逐窗口派生量、删除损失和各机动类型见 RESULT.json 与 windows。'
              '运动学触发记录按 seed、分组、指标以及评价口分别计数。K21/22 仅用于对照。', '',
              '低分替换保留原 logit，Top1 不变。固定槽也沿用原 logit，因此 Top1 变化只来自被替换槽获原最高分的窗口。', '',
              '参数可达性和拟合梯度见 MECHANISM.json。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    if (ROOT/'DONE.json').exists():
        assert all(sha(path)==digest for path,digest in read(ROOT/'DONE.json')['files'].items())
        print('ALREADY_COMPLETE',flush=True)
        return
    started = time.monotonic()
    status('VERIFY_INPUTS')
    inputs = input_manifest()
    write(ROOT/'INPUT_MANIFEST.json',dict(files=inputs))
    data = {split:arrays(OLD/f'C0/{split}.npz') for split in ('fit','dev')}
    assert [len(d['ids']) for d in data.values()]==[21719,4753]
    labels = {s:maneuver_labels(d['past'],d['future']) for s,d in data.items()}
    gtref = fit_reference(OLD/'C0/MOTION_REFERENCE.json')
    tails = read(OLD/'C0/EVALUATION_CONTRACT.json')['tail_thresholds']
    status('ANALYZE_SAVED',family='B0',seed=42)
    all_runs = {}
    out = ROOT/'windows'
    out.mkdir(exist_ok=True)
    for arm,seeds in ARMS.items():
        all_runs[arm] = {}
        for seed in seeds:
            status('ANALYZE_SAVED',family=arm,seed=seed)
            print(f'DIAG {arm} seed={seed}',flush=True)
            parent = run_folder(arm,seed)
            fit, _ = load_prediction(parent,'fit',data['fit'])
            rank, fit_stats = fixed_slots(fit,data['fit']['future'].astype(float))
            del fit
            paths, logits = load_prediction(parent,'dev',data['dev'])
            result, detail = run_case(paths, logits, data['dev'], labels['dev'], rank, gtref, tails)
            result.update(fixed_slots=rank.tolist(),fit_slot_stats=fit_stats)
            all_runs[arm][str(seed)] = result
            np.savez_compressed(out/f'{arm}_s{seed}.npz',**detail)
    status('MECHANISM')
    mechanism = analyze_mechanism(OLD,DEC,data,labels)
    write(ROOT/'MECHANISM.json',mechanism)
    result = dict(status='COMPLETE',families=all_runs,panel=aggregate(all_runs),duration_seconds=time.monotonic()-started,
                  windows_fit=21719,windows_dev=4753,new_training=False,new_model_forward=False,
                  evaluation_only=True,independent_test_accessed=False)
    write(ROOT/'RESULT.json',result)
    render(result)
    assert all(sha(p)==h for p,h in inputs.items())
    artifacts = [ROOT/name for name in ('INPUT_MANIFEST.json','MECHANISM.json','RESULT.json','REPORT.md')] + sorted(out.glob('*.npz'))
    write(ROOT/'DONE.json',dict(status='SUCCESS',files={str(p):sha(p) for p in artifacts}))
    status('COMPLETE',evaluated_model_seed_pairs=8)
    print('PREC_DIAG_COMPLETE',flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        status('FAILED',message=str(error))
        (ROOT/'FAILURE.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
