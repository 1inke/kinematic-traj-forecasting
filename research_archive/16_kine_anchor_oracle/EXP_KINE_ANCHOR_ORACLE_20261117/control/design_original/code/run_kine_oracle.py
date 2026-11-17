"""EXP-KINE-ANCHOR-ORACLE runner. CPU + NumPy only: no torch, no training, no model inference.

  python run_kine_oracle.py --data-dir <processed_data_diverse> --out <new empty dir>
        [--reference eth:B0_s42:<predictions.npz> ...] [--folds eth hotel ...] [--seeds 0 1 2]
  python run_kine_oracle.py --smoke --out <dir>      # synthetic data, checks the pipeline only

Outputs (write-once, the output dir must not exist): RESULT.json, REPORT.md, TEMPLATES.npz,
PER_WINDOW_<fold>_<split>.npz (per-window errors + labels, no raw coordinates).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
sys.dont_write_bytecode = True
import numpy as np

import kine_common as kc
import kine_repr as kr
from kine_decide import decide

STUDY = 'EXP_KINE_ANCHOR_ORACLE_20261117'
SENSITIVITY = [('S', dict(v0=0.15)), ('S', dict(v0=0.50)), ('K', dict(v0=0.15)), ('K', dict(v0=0.50)),
               ('K', dict(lam=0.5)), ('K', dict(lam=2.0))]


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def arm_name(family, v0=kr.V0, lam=kr.LAMBDA):
    tag = family
    if v0 != kr.V0:
        tag += f'_v0={v0:g}'
    if lam != kr.LAMBDA:
        tag += f'_lam={lam:g}'
    return tag


# ----------------------------------------------------------------------------- groups
def window_groups(d, profile):
    labels = kc.maneuver_labels(d['past'], d['future'])
    bins = kc.speed_bins(d['past'], profile)
    g = dict(overall=np.ones(len(labels), bool))
    for name in kc.MANEUVERS:
        g['maneuver:' + name] = labels == name
    g['decel_stop'] = (labels == 'deceleration') | (labels == 'stop')
    for i, name in enumerate(kc.SPEED_NAMES):
        g['speed:' + name] = bins == i
    for i, name in enumerate(kc.CHANGE_NAMES):
        g['change:' + name] = d['group'] == i
    return g, labels, bins


def block(ade, fde, mask):
    n = int(mask.sum())
    if not n:
        return dict(n=0)
    return dict(n=n, small=n < 20, minADE=float(ade[mask].mean()), minFDE=float(fde[mask].mean()),
                FDE_p90=float(np.quantile(fde[mask], .9)), reach_FDE_le_0p4=float((fde[mask] <= .4).mean()))


# ----------------------------------------------------------------------------- references
def reference_errors(path, d):
    with np.load(path, allow_pickle=False) as z:
        pred = {k: z[k] for k in z.files}
    ids, saved = np.asarray(d['ids']), np.asarray(pred['ids'])
    if set(ids.tolist()) != set(saved.tolist()) or len(np.unique(saved)) != len(saved):
        raise ValueError('REFERENCE_ID_MISMATCH ' + str(path))
    lookup = {key: i for i, key in enumerate(saved.tolist())}
    order = np.array([lookup[key] for key in ids.tolist()])
    if 'future' in pred and not np.allclose(pred['future'][order], d['future'], atol=1e-5):
        raise ValueError('REFERENCE_FUTURE_MISMATCH ' + str(path))
    paths = np.asarray(pred['paths'][order], np.float64)
    err = np.linalg.norm(paths - np.asarray(d['future'], np.float64)[:, None], axis=-1)
    return err.mean(-1).min(-1), err[..., -1].min(-1)


# ----------------------------------------------------------------------------- one fold
def run_fold(fold, train, test, seeds, references, out, log):
    fit_idx, val_idx = kc.scene_tail_split(train['scene'])
    splits = dict(fit=kc.subset(train, fit_idx), val=kc.subset(train, val_idx), test=test)
    profile = kc.fit_profile(splits['fit']['past'])
    frame = {}
    for name, d in splits.items():
        fl, vr, hv = kc.focal_frame(d['past'], d['future'])
        frame[name] = dict(future=fl, kp=kr.keypoints(fl), v_rec=vr, heading_valid=hv)
    arms = [(f, {}) for f in kr.FAMILIES] + SENSITIVITY
    res = dict(profile=profile, n=dict((k, len(v['future'])) for k, v in splits.items()),
               slow_share={k: float((frame[k]['v_rec'] < kr.V0).mean()) for k in ('fit', 'val', 'test')},
               heading_invalid_share={k: float((~frame[k]['heading_valid']).mean()) for k in ('fit', 'val', 'test')},
               splits={}, templates={})
    per_window, templates_npz = {}, {}
    for split in ('val', 'test'):
        groups, labels, bins = window_groups(splits[split], profile)
        res['splits'][split] = {}
        pw = dict(labels=labels, speed_bin=bins, change=splits[split]['group'],
                  v_rec=frame[split]['v_rec'], ids=splits[split]['ids'])
        for family, opt in arms:
            name = arm_name(family, **opt)
            v0, lam = opt.get('v0', kr.V0), opt.get('lam', kr.LAMBDA)
            per_seed, ades, fdes = [], [], []
            for seed in seeds:
                key = f'{fold}|{name}|{seed}'
                if key not in templates_npz:
                    templates_npz[key] = kr.fit_templates(family, frame['fit']['kp'],
                                                          kr.v_eff(frame['fit']['v_rec'], v0), seed, lam)
                t = templates_npz[key]
                cand = kr.candidates(family, t, kr.v_eff(frame[split]['v_rec'], v0), lam)
                a, f = kr.best_of(cand, frame[split]['future'])
                ades.append(a); fdes.append(f)
                per_seed.append({g: block(a, f, msk) for g, msk in groups.items()})
            res['splits'][split][name] = per_seed
            if not opt:
                pw[name + '_minADE'] = np.mean(ades, 0); pw[name + '_minFDE'] = np.mean(fdes, 0)
        # reference ceiling of B0's own scale mechanism (contains ground truth)
        oracle = []
        for seed in seeds:
            a, f = kr.oracle_scale_P(templates_npz[f'{fold}|P|{seed}'], frame[split]['future'])
            oracle.append({g: block(a, f, msk) for g, msk in groups.items()})
        res['splits'][split]['REF_P_ORACLE_SCALE'] = oracle
        if split == 'test':
            for rname, rpath in references.get(fold, []):
                a, f = reference_errors(rpath, splits['test'])
                res['splits'][split]['REF_' + rname] = [{g: block(a, f, msk) for g, msk in groups.items()}]
                res.setdefault('reference_sha256', {})[rname] = sha(rpath)
        per_window[split] = pw
        log(f'{fold} {split} done')
    res['templates']['K_seed0'] = kr.template_table('K', templates_npz[f'{fold}|K|{seeds[0]}'])
    for split, pw in per_window.items():
        np.savez_compressed(out / f'PER_WINDOW_{fold}_{split}.npz', **pw)
    return res, templates_npz, per_window


# ----------------------------------------------------------------------------- summary
def collect(results, seeds):
    summary = dict(folds={}, pooled_test={})
    for fold, r in results.items():
        summary['folds'][fold] = {}
        for split in ('val', 'test'):
            summary['folds'][fold][split] = {
                arm: dict(minADE=[s['overall']['minADE'] for s in per_seed],
                          minFDE=[s['overall']['minFDE'] for s in per_seed])
                for arm, per_seed in r['splits'][split].items() if len(per_seed) == len(seeds)}
    return summary


def pooled(per_window_all, arms, mask_fn):
    out = {}
    for arm in arms:
        a = np.concatenate([pw[arm + '_minADE'][mask_fn(pw)] for pw in per_window_all])
        f = np.concatenate([pw[arm + '_minFDE'][mask_fn(pw)] for pw in per_window_all])
        out[arm] = dict(n=int(len(a)), minADE=float(a.mean()) if len(a) else None,
                        minFDE=float(f.mean()) if len(f) else None)
    return out


def pooled_decel_per_seed(results, seeds):
    """Seed-wise pooled decel_stop means (window-weighted over folds)."""
    out = {}
    for arm in kr.FAMILIES:
        ade, fde = [], []
        for i in range(len(seeds)):
            n = sum(results[f]['splits']['test'][arm][i]['decel_stop'].get('n', 0) for f in results)
            if not n:
                raise ValueError('NO_DECEL_STOP_WINDOWS_IN_TEST')
            ade.append(sum(results[f]['splits']['test'][arm][i]['decel_stop'].get('minADE', 0) *
                           results[f]['splits']['test'][arm][i]['decel_stop'].get('n', 0) for f in results) / n)
            fde.append(sum(results[f]['splits']['test'][arm][i]['decel_stop'].get('minFDE', 0) *
                           results[f]['splits']['test'][arm][i]['decel_stop'].get('n', 0) for f in results) / n)
        out[arm] = dict(decel_stop=dict(minADE=ade, minFDE=fde))
    return out


# ----------------------------------------------------------------------------- report
def mean_block(per_seed, group, key):
    vals = [s[group].get(key) for s in per_seed if s[group].get('n', 0)]
    return float(np.mean(vals)) if vals else None


def fmt(x, nd=3):
    return '—' if x is None else f'{x:.{nd}f}'


def pct(x):
    return '—' if x is None else f'{100 * x:+.1f}%'


def render(result):
    d = result['decision']; nums = d['numbers']; R = result['folds']
    L = [f'# 【已完成】EXP-KINE-ANCHOR-ORACLE 结果（自动生成）', '',
         f"**判定：{d['verdict']}**：{d['meaning']}", '',
         f"> {d['caveat']}。全部为模板几何计算，没有训练、没有模型推理。", '',
         '## 预注册检查', '', '| 检查 | 结果 |', '|---|---|']
    for k, v in d['checks'].items():
        L.append(f'| {k} | {"通过" if v else "未通过"} |')
    L += ['', f"ETH测试 K 相对 P：minADE {pct(nums['eth_test_g_ADE'])}，minFDE {pct(nums['eth_test_g_FDE'])}；"
          f"三个聚类seed配对 minFDE：{', '.join(pct(g) for g in nums['eth_seed_pair_g_FDE'])}", '',
          f"五折验证（分布内）K 相对 P：minADE {pct(nums['fivefold_val_g_ADE'])}，minFDE {pct(nums['fivefold_val_g_FDE'])}", '',
          f"减速+停止（五折测试合并）minFDE：K {fmt(nums['decel_stop_minFDE']['main'])} / "
          f"P {fmt(nums['decel_stop_minFDE']['P'])} / P_AUG {fmt(nums['decel_stop_minFDE']['P_AUG'])}", '',
          '## 归因（不影响判定）', '']
    for k, v in d['attribution'].items():
        L.append(f"- {k}：**{v['label']}**（ETH测试 minFDE {pct(v['eth_test_g_FDE'])}，减速+停止 {pct(v['decel_stop_g_FDE'])}）")
    arms = ['P', 'P_AUG', 'S', 'K', 'REF_P_ORACLE_SCALE']
    for split, title in (('test', '测试（跨场景）'), ('val', '验证（分布内，scene_tail 5%）')):
        L += ['', f'## {title}：overall minADE / minFDE（聚类seed平均）', '',
              '| 折 | ' + ' | '.join(arms) + ' |', '|---' * (len(arms) + 1) + '|']
        for fold, r in R.items():
            cells = []
            for arm in arms:
                ps = r['splits'][split].get(arm)
                cells.append('—' if ps is None else f"{fmt(mean_block(ps, 'overall', 'minADE'))} / {fmt(mean_block(ps, 'overall', 'minFDE'))}")
            L.append(f'| {fold} | ' + ' | '.join(cells) + ' |')
    if 'eth' in R:
        L += ['', '## ETH 测试：按机动 / 速度分组的 minFDE', '', '| 组 | n | ' + ' | '.join(arms) + ' |',
              '|---' * (len(arms) + 2) + '|']
        sp = R['eth']['splits']['test']
        for g in ['overall', 'decel_stop'] + ['maneuver:' + m for m in kc.MANEUVERS] + ['speed:' + s for s in kc.SPEED_NAMES]:
            n = sp['P'][0][g].get('n', 0)
            L.append(f'| {g} | {n}{"（小样本）" if 0 < n < 20 else ""} | ' +
                     ' | '.join(fmt(mean_block(sp[a], g, 'minFDE')) for a in arms) + ' |')
        refs = [a for a in sp if a.startswith('REF_') and a != 'REF_P_ORACLE_SCALE']
        if refs:
            L += ['', '已训练模型参考（ETH 测试，overall minADE / minFDE）：' +
                  '；'.join(f"{a[4:]} {fmt(mean_block(sp[a], 'overall', 'minADE'))} / {fmt(mean_block(sp[a], 'overall', 'minFDE'))}" for a in refs)]
    L += ['', '## 敏感性（不参与判定；五折测试 overall minFDE 均值）', '', '| 臂 | minFDE |', '|---|---:|']
    for arm in sorted({a for r in R.values() for a in r['splits']['test'] if '=' in a}):
        L.append(f"| {arm} | {fmt(float(np.mean([mean_block(r['splits']['test'][arm], 'overall', 'minFDE') for r in R.values()])))} |")
    if 'eth' in R and R['eth']['templates'].get('K_seed0'):
        L += ['', '## ETH折训练集上的 K 模板（seed 0；按最后一段速度比排序）', '',
              '| # | 速度比 ρ1/ρ2/ρ3 | 方向 φ1/φ2/φ3（度） |', '|---:|---|---|']
        for i, t in enumerate(R['eth']['templates']['K_seed0']):
            L.append(f"| {i} | {' / '.join(f'{v:.2f}' for v in t['rho'])} | {' / '.join(f'{v:.0f}' for v in t['phi_deg'])} |")
    L += ['', '## 数据概况', '', '| 折 | fit/val/test 窗口 | 慢速(<V0)占比 fit/test | 朝向无效占比 test |', '|---|---|---|---|']
    for fold, r in R.items():
        L.append(f"| {fold} | {r['n']['fit']}/{r['n']['val']}/{r['n']['test']} | "
                 f"{r['slow_share']['fit']:.2f}/{r['slow_share']['test']:.2f} | {r['heading_invalid_share']['test']:.2f} |")
    return '\n'.join(L) + '\n'


# ----------------------------------------------------------------------------- smoke data
def synthetic_fold(rng, n, speed_mu, coupled=True):
    """Walkers with speed changes; returns raw-like (data[N,9,20,6], num[N]).
    coupled=False: future speed is drawn independently of the past (negative control: speed
    normalisation must NOT win there)."""
    data = np.zeros((n, kc.MAX_AGENTS, 20, 6), np.float32)
    v = rng.lognormal(np.log(speed_mu), .35, n)
    th = rng.uniform(-np.pi, np.pi, n)
    ratio = np.exp(rng.normal(0, .3, n))
    turn = rng.normal(0, .25, n)
    for i in range(n):
        fut_v = v[i] if coupled else rng.lognormal(np.log(speed_mu), .35)
        speeds = np.concatenate([np.full(8, v[i]), fut_v * np.linspace(1, ratio[i], 12)])
        heads = np.concatenate([np.full(8, th[i]), th[i] + np.linspace(0, turn[i], 12)])
        step = (speeds * kc.DT)[:, None] * np.stack([np.cos(heads), np.sin(heads)], -1)
        xy = np.cumsum(step, 0) + rng.normal(0, .01, (20, 2)) + rng.uniform(-5, 5, 2)
        data[i, 0, :, 4:] = xy
        data[i, 0, :, 0] = i % 4
    return data, np.ones(n, np.int64)


def make_smoke(root, coupled=True):
    rng = np.random.default_rng(7)
    root.mkdir(parents=True)
    for fold in kc.FOLDS:
        for split, n, mu in (('train', 900, .55), ('test', 300, 1.0 if fold == 'eth' else .6)):
            d, num = synthetic_fold(rng, n, mu, coupled)
            np.save(root / f'{fold}_data_{split}.npy', d); np.save(root / f'{fold}_num_{split}.npy', num)
    return root


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir'); ap.add_argument('--out', required=True)
    ap.add_argument('--folds', nargs='+', default=list(kc.FOLDS))
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--reference', nargs='*', default=[], help='fold:name:predictions.npz (test only)')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--smoke-negative', action='store_true', help='synthetic negative control')
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)          # write-once: refuse an existing directory
    t0 = time.time()

    def log(msg):
        line = f'[{now()}] {msg}'
        print(line, flush=True)
        with (out / 'RUN.log').open('a', encoding='utf-8') as f:
            f.write(line + '\n')

    a.smoke = a.smoke or a.smoke_negative
    data_dir = make_smoke(out / '_smoke_data', coupled=not a.smoke_negative) if a.smoke else Path(a.data_dir)
    if 'eth' not in a.folds:
        raise SystemExit('ETH fold is required by the decision rules')
    references = {}
    for spec in a.reference:
        fold, name, path = spec.split(':', 2)
        references.setdefault(fold, []).append((name, path))
    inputs = {}
    results, templates_all, per_window_test = {}, {}, []
    for fold in a.folds:
        files = kc.fold_files(data_dir, fold)
        inputs[fold] = {k: dict(path=str(p), sha256=sha(p)) for k, p in files.items()}
        train, test = kc.load_split(data_dir, fold, 'train'), kc.load_split(data_dir, fold, 'test')
        res, tpl, pw = run_fold(fold, train, test, a.seeds, references, out, log)
        results[fold] = res; templates_all.update(tpl); per_window_test.append(pw['test'])
    summary = collect(results, a.seeds)
    summary['pooled_test'] = pooled_decel_per_seed(results, a.seeds)
    decision = decide(summary, main='K')
    pooled_groups = {g: pooled(per_window_test, kr.FAMILIES, fn) for g, fn in {
        'decel_stop': lambda pw: np.isin(pw['labels'], ['deceleration', 'stop']),
        'steady': lambda pw: pw['labels'] == 'steady',
        'speed_high': lambda pw: pw['speed_bin'] == 2}.items()}
    result = dict(study=STUDY, finished=now(), seconds=round(time.time() - t0, 1), smoke=bool(a.smoke),
                  settings=dict(V0=kr.V0, LAMBDA=kr.LAMBDA, RHO_EPS=kr.RHO_EPS, KEY_STEPS=kr.KEY_STEPS,
                                AUG=dict(range=kr.AUG_RANGE, prob=kr.AUG_PROB, copies=kr.AUG_COPIES),
                                K=kc.K, seeds=a.seeds, sensitivity=[arm_name(f, **o) for f, o in SENSITIVITY]),
                  environment=dict(python=sys.version, numpy=np.__version__, platform=platform.platform()),
                  inputs=inputs, folds=results, summary=summary, pooled_test_seedmean=pooled_groups,
                  decision=decision, training_started=False, model_inference=False)
    with (out / 'RESULT.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    np.savez_compressed(out / 'TEMPLATES.npz', **{k.replace('|', '__'): v for k, v in templates_all.items()})
    (out / 'REPORT.md').write_text(render(result), encoding='utf-8')
    log(f"verdict={decision['verdict']} seconds={result['seconds']}")
    return result


if __name__ == '__main__':
    main()
