"""Search predeclared shift and tilt grids and fit regularized endpoint corrections."""
import numpy as np

from bias_inputs import ALPHA, FEATURES, GRID, OBS, TAU, TILT, bins, cell_features, cells, fold_ids, quantiles
from calib_metrics import hpd_mc, log_density


def grid_pairs(grid):
    return np.array([(a, b) for a in grid for b in grid])


SHIFT, TILTS = grid_pairs(GRID), grid_pairs(TILT)


def chosen_metric(values, params):
    point = values.mean(0)
    least = point.min()
    near = np.flatnonzero(point <= least + 1e-12)
    return int(near[np.argmin(np.square(params[near]).sum(1))])


def grid_geometry(e, progress):
    curves = {k: np.empty((len(e), len(SHIFT))) for k in ("minFDE", "minADE")}
    shift = SHIFT[:, None, None, :] * TAU[None, None, :, None]
    for start in range(0, len(e), 16):
        where = slice(start, start + 16)
        lon = e[where, None, :, :, 0] + shift[..., 0]
        lat = e[where, None, :, :, 1] + shift[..., 1]
        np.square(lon, out=lon)
        np.square(lat, out=lat)
        lon += lat
        np.sqrt(lon, out=lon)
        curves["minFDE"][where] = lon[..., -1].min(2)
        curves["minADE"][where] = lon.mean(3).min(2)
        if start % 2048 == 0:
            progress(start, len(e))
    return curves


def log_weights(w):
    return np.log(np.maximum(w, np.finfo(float).tiny))


def probabilities(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def tilt_weights(w, feature, beta):
    fitted = probabilities(log_weights(w) + (feature * beta[:, None]).sum(-1))
    identity = np.all(beta == 0, axis=1)
    fitted[identity] = w[identity]
    return fitted


def grid_probability(e, w, feature, progress):
    result = {k: np.empty((len(e), len(TILTS))) for k in ("endpoint_ES", "winner_NLL")}
    lw = log_weights(w)
    d = np.linalg.norm(e, axis=2)
    best = d.argmin(1)
    pair = np.linalg.norm(e[:, :, None] - e[:, None], axis=-1)
    for i in range(0, len(e), 64):
        w_slice = slice(i, i+64)
        tilt = TILTS @ feature[w_slice].transpose(0, 2, 1) + lw[w_slice, None, :]
        update = probabilities(tilt)
        update[:, len(TILTS)//2] = w[w_slice]
        result["endpoint_ES"][w_slice] = ((update * d[w_slice, None]).sum(-1) -
                                         .5 * ((update @ pair[w_slice]) * update).sum(-1))
        result["winner_NLL"][w_slice] = -log_weights(update[np.arange(len(update)), :, best[w_slice]])
        if i % 2048 == 0:
            progress(i, len(e))
    return result


def take_model(model, mask):
    return {k: v[mask] for k,v in model.items()}


def table_fit(a, b, train, curves, params):
    c = np.zeros((len(b), 2))
    selected = {}
    for key in np.unique(a[train]):
        where = train & (a == key)
        count = int(where.sum())
        if count < 50:
            selected[key] = dict(n_train=count, correction=[0.,0.], n_eval=int((b==key).sum()), fallback=True)
            continue
        i = chosen_metric(curves[where], params)
        c[b==key] = params[i]
        selected[key] = dict(n_train=count, correction=params[i].tolist(), n_eval=int((b==key).sum()),
                             criterion=float(curves[where, i].mean()), fallback=False)
    unavailable = int((~np.isin(b, list(selected))).sum())
    return c, dict(cells=selected, unseen_eval_windows=unavailable)


def ridge_design(a, b, name, train):
    predictors = [] if name == "S" else list(OBS)
    if name == "K+N":
        predictors.append("n_nb2")

    def matrix(d):
        base = [d[k] for k in predictors]
        if predictors:
            base += [d["v_obs"]*d[k] for k in ("curv", "hist_ratio")]
        if "S" in name:
            base += [d["scene"] == level for level in np.unique(a["scene"][train])]
        return np.column_stack(base).astype(float)

    return matrix(a), matrix(b)


def regression(x, y, xx, penalty):
    mean = x.mean(0)
    scale = x.std(0)
    scale[scale < 1e-12] = 1.
    z = (x-mean)/scale
    yy = y.mean(0)
    coef = np.linalg.solve(z.T@z + penalty*np.eye(z.shape[1]), z.T@(y-yy))
    return (xx-mean)/scale @ coef + yy


def ridge_fit(a, b, train, err, name):
    x, z = ridge_design(a, b, name, train)
    endpoints = err[:, :, -1]
    index = np.square(endpoints).sum(-1).argmin(1)
    target = -endpoints[np.arange(len(index)), index]
    folds = fold_ids(a["track"][train], 3)
    x, y = x[train], target[train]
    losses = {}
    for penalty in ALPHA:
        loss = 0.
        for fold in range(3):
            keep = folds != fold
            pred = regression(x[keep], y[keep], x[~keep], penalty)
            loss += np.square(pred-y[~keep]).sum()
        losses[penalty] = float(loss/(len(x)*2))
    selected = min(losses, key=losses.get)
    return regression(x, y, z, selected), dict(penalty=selected, cv_mse=losses,
                                              n_train=len(x), tracks=len(np.unique(a["track"][train])))


def learn(a, b, error, geometry, probability, split):
    n = len(b["ids"])
    plans = {}
    for feature in FEATURES:
        for objective in ("minFDE","minADE","endpoint_ES","winner_NLL","ridge"):
            plans[feature, objective] = np.zeros((n, 2)), []
    if split == "dev_crossfit":
        f = fold_ids(a["track"], 5)
        folds = [(f != k, f == k) for k in range(5)]
    else:
        folds = [(np.ones(len(a["ids"]), bool), np.ones(n, bool))]
    for fold, (train, evaluate) in enumerate(folds):
        assert not np.intersect1d(a["track"][train], b["track"][evaluate]).size
        cuts = quantiles(a, train)
        train_bins, eval_bins = bins(a, cuts), bins(b, cuts)
        where = {key: value[evaluate] for key,value in b.items()}
        for setting, (out, fitted) in plans.items():
            feature, objective = setting
            if objective == "ridge":
                shift, details = ridge_fit(a, where, train, error, feature)
            else:
                cols = cell_features(feature)
                grid = SHIFT if objective in geometry else TILTS
                values = geometry if objective in geometry else probability
                shift, details = table_fit(cells(train_bins, cols), cells(eval_bins, cols)[evaluate],
                                           train, values[objective], grid)
            out[evaluate] = shift
            fitted.append(dict(fold=fold if split=="dev_crossfit" else None,
                               training_tracks=len(np.unique(a["track"][train])),
                               evaluation_tracks=len(np.unique(b["track"][evaluate])), cuts=cuts, **details))
    return plans


def scoring(model, err, w, width=None, draws=None):
    dist = np.linalg.norm(err, axis=-1)
    ade, fde = dist.mean(2), dist[:, :, -1]
    top, best = w.argmax(1), fde.argmin(1)
    idx = np.arange(len(top))
    x = model["paths"]
    i, j = np.triu_indices(x.shape[1], 1)
    es_path, es_end = [], []
    for start in range(0, len(x), 128):
        b = slice(start, start+128)
        squared = np.square(x[b,i]-x[b,j]).sum(-1)
        products = w[b,i]*w[b,j]
        es_path.append((np.sqrt((dist[b]**2).sum(2))*w[b]).sum(1) -
                       (np.sqrt(squared.sum(2))*products).sum(1))
        es_end.append((fde[b]*w[b]).sum(1) - (np.sqrt(squared[:,:,-1])*products).sum(1))
    value = dict(minADE=ade.min(1),minFDE=fde.min(1), J=((ade<=.2)&(fde<=.4)).any(1),
                 Top1_ADE=ade[idx,top],Top1_FDE=fde[idx,top], winner_NLL=-log_weights(w[idx,best]),
                 path_ES=np.concatenate(es_path),endpoint_ES=np.concatenate(es_end))
    if width is not None:
        d2 = np.square(x[:, :, -1] - model["y"][:, None, -1]).sum(2)
        truth = log_density(d2, log_weights(w), width)
        value["HPD90"] = hpd_mc(x[:, :, -1], w, width, truth, *draws)[0] <= .9
    return value

