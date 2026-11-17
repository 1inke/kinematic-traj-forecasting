"""Pre-registered decision rules of EXP-KINE-ANCHOR-ORACLE (fixed before any real run).

Input `summary`:
  summary['folds'][fold][split][arm] = dict(minADE=[per seed], minFDE=[per seed])  (overall)
  summary['pooled_test'][arm]['decel_stop'] = dict(minADE=[per seed], minFDE=[per seed])
split in ('val', 'test'); arms include 'P', 'P_AUG', 'S', 'K'.
Gain g(X vs Y) = (Y - X) / Y  (positive = X has the smaller error).
"""
from __future__ import annotations

import numpy as np

ETH_FDE_GAIN = 0.05     # C1
ETH_ADE_GAIN = 0.03     # C1
WEAK_FDE_GAIN = 0.02    # WEAK band lower edge
FOLD_TOL = -0.02        # C3: no test fold worse than 2 %
VAL_TOL = -0.01         # C4: in-distribution (five-fold val mean) not worse than 1 %
ATTR = 0.02             # attribution band K vs S, S vs P


def m(x):
    return float(np.mean(x))


def gain(x, ref):
    return (ref - x) / ref if ref > 0 else 0.0


def _fold_mean(summary, split, arm, metric):
    return float(np.mean([m(summary['folds'][f][split][arm][metric]) for f in summary['folds']]))


def decide(summary, main='K'):
    folds = summary['folds']
    if 'eth' not in folds:
        raise ValueError('ETH_FOLD_REQUIRED')
    eth = folds['eth']['test']
    g_eth_fde = gain(m(eth[main]['minFDE']), m(eth['P']['minFDE']))
    g_eth_ade = gain(m(eth[main]['minADE']), m(eth['P']['minADE']))
    seed_pairs = [gain(a, b) for a, b in zip(eth[main]['minFDE'], eth['P']['minFDE'])]
    per_fold = {f: dict(g_ADE=gain(m(folds[f]['test'][main]['minADE']), m(folds[f]['test']['P']['minADE'])),
                        g_FDE=gain(m(folds[f]['test'][main]['minFDE']), m(folds[f]['test']['P']['minFDE'])))
                for f in folds}
    val_ade = gain(_fold_mean(summary, 'val', main, 'minADE'), _fold_mean(summary, 'val', 'P', 'minADE'))
    val_fde = gain(_fold_mean(summary, 'val', main, 'minFDE'), _fold_mean(summary, 'val', 'P', 'minFDE'))
    test_main = _fold_mean(summary, 'test', main, 'minFDE')
    test_aug = _fold_mean(summary, 'test', 'P_AUG', 'minFDE')
    dec = summary['pooled_test']
    dec_main, dec_p, dec_aug = (m(dec[a]['decel_stop']['minFDE']) for a in (main, 'P', 'P_AUG'))

    checks = dict(
        C1_eth_speed_shift=g_eth_fde >= ETH_FDE_GAIN and g_eth_ade >= ETH_ADE_GAIN,
        C2_not_worse_than_aug_fivefold=test_main <= test_aug,
        C3_no_test_fold_worse=all(v['g_ADE'] >= FOLD_TOL and v['g_FDE'] >= FOLD_TOL for v in per_fold.values()),
        C4_in_distribution_ok=val_ade >= VAL_TOL and val_fde >= VAL_TOL,
        C5_decel_ok=dec_main <= dec_p and dec_main <= dec_aug,
        C6_eth_all_seed_pairs_positive=all(g > 0 for g in seed_pairs),
    )
    c = checks
    if c['C1_eth_speed_shift'] and c['C3_no_test_fold_worse'] and c['C4_in_distribution_ok'] \
            and c['C6_eth_all_seed_pairs_positive'] and (c['C2_not_worse_than_aug_fivefold'] or c['C5_decel_ok']):
        verdict = 'SUPPORT'
    elif c['C3_no_test_fold_worse'] and c['C4_in_distribution_ok'] and g_eth_fde >= WEAK_FDE_GAIN:
        verdict = 'WEAK'
    else:
        verdict = 'NOT_SUPPORTED'

    def attribution(x, y):
        ge = gain(m(eth[x]['minFDE']), m(eth[y]['minFDE']))
        gd = gain(m(dec[x]['decel_stop']['minFDE']), m(dec[y]['decel_stop']['minFDE']))
        if ge >= ATTR or gd >= ATTR:
            label = f'{x}_BETTER'
        elif ge <= -ATTR or gd <= -ATTR:
            label = f'{y}_BETTER'
        else:
            label = 'SIMILAR'
        if ge >= ATTR and gd <= -ATTR or ge <= -ATTR and gd >= ATTR:
            label = 'MIXED'
        return dict(label=label, eth_test_g_FDE=ge, decel_stop_g_FDE=gd)

    return dict(
        verdict=verdict, main_arm=main, checks=checks,
        numbers=dict(eth_test_g_FDE=g_eth_fde, eth_test_g_ADE=g_eth_ade, eth_seed_pair_g_FDE=seed_pairs,
                     per_fold_test=per_fold, fivefold_val_g_ADE=val_ade, fivefold_val_g_FDE=val_fde,
                     fivefold_test_minFDE_main=test_main, fivefold_test_minFDE_P_AUG=test_aug,
                     decel_stop_minFDE=dict(main=dec_main, P=dec_p, P_AUG=dec_aug)),
        attribution=dict(K_vs_S=attribution('K', 'S'), S_vs_P=attribution('S', 'P')),
        meaning=dict(
            SUPPORT='机动模板在零训练条件下已显示优势：进入训练验证（单独写实验文档）',
            WEAK='速度归一化有一点作用但证据不够；不单独立为主线，可作为B的附属设计',
            NOT_SUPPORTED='模板层面没有优势：方向A降级，解码以B（候选间优化）为主')[verdict],
        caveat='模板层面的优势是必要条件，不保证训练后的解码器也更好（见05档KINEMATIC_DECODER系列）')
