"""Window statistics, matched neighbor permutations and predeclared decision."""
import numpy as np
import torch
from dec_model import risk_geometry


def masks(meta, social):
    group, labels = meta['group'], meta['labels']
    return dict(all=np.ones(len(group), dtype=bool), stable=group == 0, transition=group == 1,
                change=group == 2, decel_stop=np.isin(labels, ('deceleration', 'stop')),
                social=social, non_social=~social)


def diagnose(data, basis):
    t = data['tensors']
    output = {k: [] for k in ('neighbor_count', 'social', 'candidate_sets_vary', 'c3_count', 'c4_count',
                              'c3_and_c4_sets_equal')}
    with torch.no_grad():
        for lo in range(0, len(t['relative']), 256):
            ix = slice(lo, lo + 256)
            rel, pos, vel, valid = [t[k][ix] for k in ('relative', 'nbr_pos', 'nbr_vel', 'nbr_valid')]
            paths = torch.einsum('tp,nkpc->nktc', basis, rel)
            inds, keep, _, dca = risk_geometry(paths, pos, vel, valid)
            c3_sets = torch.zeros_like(dca, dtype=torch.bool).scatter_(-1, inds, keep)
            closest = pos.norm(dim=-1).masked_fill(~valid, torch.inf).argsort(stable=True)[..., :3]
            c4_keep = valid.gather(-1, closest)
            c4_sets = torch.zeros_like(valid).scatter_(-1, closest, c4_keep)
            rows = dict(neighbor_count=valid.sum(-1), social=((dca < 1.) & valid[:, None]).any(dim=(1, 2)),
                        candidate_sets_vary=(c3_sets != c3_sets[:, :1]).any(dim=(1, 2)),
                        c3_count=keep.sum(-1), c4_count=c4_keep.sum(-1),
                        c3_and_c4_sets_equal=(c3_sets == c4_sets[:, None]).all(-1))
            for k, v in rows.items():
                output[k].append(v.cpu().numpy())
    w = {k: np.concatenate(v) for k, v in output.items()}
    summaries = {}
    for name, m in masks(data['meta'], w['social']).items():
        n = int(m.sum())
        social = m & w['social']
        s = int(social.sum())
        summaries[name] = dict(n=n, neighbor_count_histogram=np.bincount(w['neighbor_count'][m], minlength=9).tolist(),
             social_n=s, social_rate=s / n if n else None, candidate_sets_vary_n=int(w['candidate_sets_vary'][social].sum()),
             candidate_sets_vary_among_social=float(w['candidate_sets_vary'][social].mean()) if s else None,
             c3_neighbors_per_candidate=float(w['c3_count'][m].mean()) if n else None,
             c4_neighbors_per_candidate=float(w['c4_count'][m].mean()) if n else None,
             c3_neighbor_count_histogram=np.bincount(w['c3_count'][m].ravel(), minlength=4).tolist(),
             c4_neighbor_count_histogram=(np.bincount(w['c4_count'][m], minlength=4) * 20).tolist(),
             same_count_fraction=float((w['c3_count'][m] == w['c4_count'][m, None]).mean()) if n else None,
             same_set_fraction=float(w['c3_and_c4_sets_equal'][m].mean()) if n else None)
    return w, summaries


def window_metrics(paths, scores, shift, y):
    d = np.linalg.norm(paths.astype(np.float64) - y.astype(np.float64)[:, None], axis=-1)
    f = d[..., -1]
    top = scores.argmax(-1)
    min_f = f.min(-1)
    return dict(minADE=d.mean(-1).min(-1), minFDE=min_f,
                top1FDE=f[np.arange(len(top)), top], uncovered=min_f > .4,
                mean_shift=np.linalg.norm(shift.astype(np.float64), axis=-1).mean(axis=(1, 2)))


def aggregate(w, meta, social):
    result = {}
    for name, m in masks(meta, social).items():
        n = int(m.sum())
        result[name] = dict(n=n, **{k: float(v[m].mean()) if n else None for k, v in w.items()},
                            minFDE_p90=float(np.quantile(w['minFDE'][m], .9)) if n else None)
    return result


def permutation(count):
    """One frozen label-free derangement per count, no cross-count donors."""
    rng = np.random.default_rng(20261117)
    dest = np.arange(len(count))
    buckets = {}
    for k in range(9):
        ix = np.flatnonzero(count == k)
        if len(ix) > 1 and k > 0:
            chosen = rng.permutation(ix)
            dest[chosen] = np.roll(chosen, -1)
        buckets[str(k)] = dict(n=len(ix), moved=int((dest[ix] != ix).sum()))
    assert np.array_equal(count[dest], count) and len(np.unique(dest)) == len(dest)
    return dest, buckets


def paired(runs, baseline, arm, ref, group='all', mode='clean'):
    vals = {}
    for metric in ('minFDE', 'minADE'):
        out = []
        for s in ('0', '1', '2'):
            x = runs[arm][s][mode][group][metric]
            y = baseline[group][metric] if ref == 'A0' else runs[ref][s]['clean'][group][metric]
            if x is None or y is None or y <= 0:
                return None
            out.append((y - x) / y)
        vals[metric] = dict(by_seed=out, mean=float(np.mean(out)))
    return vals


def effective(pair, threshold):
    return pair is not None and pair['minFDE']['mean'] >= threshold and pair['minADE']['mean'] >= -.002


def proxy(runs, baseline):
    q = paired(runs, baseline, 'A1', 'A0')
    f = q['minFDE']['mean']
    sd = float(np.std(q['minFDE']['by_seed'], ddof=1))
    return dict(P0=q, previous_reference=.0129, difference=f-.0129, sigma=sd,
                threshold=max(.005, 2*sd), passes=f >= .008 and abs(f - .0129) <= .003 and q['minADE']['mean'] >= -.002)


def decision(runs, baseline, base_params, shuffle_buckets):
    p0 = proxy(runs, baseline)
    if not p0['passes']:
        return dict(verdict='PROXY_INVALID', proxy=p0, eligible_to_advance=False,
                    protection_status='NOT_EVALUATED')
    thr = p0['threshold']
    ref = min(('A1', 'A1L'), key=lambda a: np.mean([v['clean']['all']['minFDE'] for v in runs[a].values()]))
    cs = ('C1', 'C2', 'C3', 'C4')
    relative = {a: paired(runs, baseline, a, ref) for a in cs}
    a0 = paired(runs, baseline, 'C3', 'A0')
    c4 = paired(runs, baseline, 'C3', 'C4')
    c2 = paired(runs, baseline, 'C3', 'C2')
    gate04 = effective(c4, thr)
    group_effect = {}
    for g in ('all', 'social', 'non_social'):
        clean = paired(runs, baseline, 'C3', 'C1', g)
        shuffled = paired(runs, baseline, 'C3', 'C1', g, 'shuffled')
        n = baseline[g]['n']
        def absolute(mode):
            if not n:
                return None
            return float(np.mean([runs['C1'][s]['clean'][g]['minFDE'] - runs['C3'][s][mode][g]['minFDE']
                                  for s in ('0', '1', '2')]))
        group_effect[g] = dict(n=n, vs_C1_clean=clean, vs_C1_shuffled=shuffled,
                               mean_absolute_clean_gain_m=absolute('clean'),
                               mean_absolute_shuffled_gain_m=absolute('shuffled'))
    s, u = group_effect['social'], group_effect['non_social']
    social_total = None if s['n'] < 20 else s['n'] * s['mean_absolute_clean_gain_m']
    nonsocial_total = None if u['n'] < 20 else u['n'] * u['mean_absolute_clean_gain_m']
    social_dominates = (social_total is not None and nonsocial_total is not None and
                        social_total > max(0., nonsocial_total))
    valid_permutation = all(k == '0' or x['n'] == x['moved'] for k, x in shuffle_buckets.items())
    clean, shuffled = s['vs_C1_clean'], s['vs_C1_shuffled']
    original_gain = clean['minFDE']['mean'] if clean else None
    attenuation = original_gain - shuffled['minFDE']['mean'] if clean and shuffled else None
    decays = (valid_permutation and attenuation is not None and original_gain > 0.
              and attenuation >= max(thr, .5 * original_gain))
    mechanism = dict(passes=social_dominates and decays, social_dominates=social_dominates,
                     social_n_times_gain=social_total, nonsocial_n_times_gain=nonsocial_total,
                     shuffle_drops_social_gain=decays, social_relative_gain_drop=attenuation,
                     permutation_complete=valid_permutation, groups=group_effect)
    strong = [a for a in ('C1', 'C2', 'C3') if effective(relative[a], .015)]
    effective_arms = [a for a in cs if effective(relative[a], thr)]
    eligible_c3 = effective(a0, .03) and 'C3' in strong and gate04 and mechanism['passes']
    if eligible_c3:
        raw, selected = 'DESIGN_PROMISING', ['C3']
    elif strong and (c4['minFDE']['mean'] < thr or not mechanism['passes']):
        raw, selected = 'INFO_YES_DESIGN_NO', strong
    else:
        best = max(relative[a]['minFDE']['mean'] for a in cs)
        small = [a for a in effective_arms if relative[a]['minFDE']['mean'] < .015]
        if thr <= best < .015 and small:
            raw, selected = 'INFO_SMALL', small
        elif best < thr:
            raw, selected = 'NO_INFO_GAIN', []
        else:
            raw, selected = 'UNCLASSIFIED_RULE_GAP', effective_arms
    def harm(a, comparator, key, group):
        x = [runs[a][s]['clean'][group][key] for s in ('0', '1', '2')]
        y = [baseline[group][key] if comparator == 'A0' else runs[comparator][s]['clean'][group][key]
             for s in ('0', '1', '2')]
        return float(np.mean([(v - r)/r for v, r in zip(x, y)])) if all(v is not None and r is not None and r > 0 for v, r in zip(x, y)) else None
    guards = {}
    for a in cs:
        checks = {}
        def check(key, value, limit):
            checks[key] = dict(value=value, limit=limit, passes=value is not None and value <= limit)
        check('top1_vs_A0_relative_harm', harm(a, 'A0', 'top1FDE', 'all'), .02)
        check('stable_vs_R_relative_harm', harm(a, ref, 'minFDE', 'stable'), .01)
        check('decel_stop_vs_R_relative_harm', harm(a, ref, 'minFDE', 'decel_stop'), .01)
        check('registered_parameter_fraction', runs[a]['0']['parameter_count'] / base_params, .10)
        if a == 'C3':
            times = [runs[a][s]['latency']['refiner_only']['median_ms'] -
                     runs['A1'][s]['latency']['refiner_only']['median_ms'] for s in ('0', '1', '2')]
            check('incremental_cpu_ms_vs_A1', float(np.mean(times)), 2.)
        guards[a] = dict(passes=all(x['passes'] for x in checks.values()), checks=checks,
                         examined_for_classification=a in selected)
    guard = 'NO_EFFECTIVE_ARM' if not selected else 'PASS' if all(guards[a]['passes'] for a in selected) else 'PROTECTION_FAILED'
    return dict(verdict=raw, protection_status=guard, proxy=p0, reference=ref,
                Q1=relative['C1'], Q2=relative['C2'], Q3_vs_R=relative['C3'], Q3_vs_A0=a0,
                Q4_vs_C4=c4, Q4_pass=gate04, Q5_vs_C2_descriptive=c2,
                comparisons_vs_R=relative, mechanism=mechanism, effective_arms=effective_arms,
                checked_arms=selected, guards=guards, eligible_to_advance=raw == 'DESIGN_PROMISING' and guard == 'PASS')
