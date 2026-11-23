"""Evaluate exactly ordered candidates and retain paired window metrics."""
import numpy as np
import torch

from advisor_eval import forecast_metrics, groups, metric_summary, safe_npz
from order_numpy import conf_order, kinematic_excess, risk_order, softmax
from qrank_model import MLDecoder, qhat


@torch.no_grad()
def infer(core, model, data):
    model.eval()
    device = next(model.parameters()).device
    decoder = model.decoder
    if isinstance(decoder, MLDecoder):
        decoder.eval_generator = torch.Generator(device=device).manual_seed(12345)
    saved = {}
    for start in range(0, len(data["ids"]), 64):
        sl = slice(start, start+64)
        p, n = (torch.as_tensor(data[k][sl], device=device) for k in ("past", "num_valid"))
        paths, logits, _, tf = core.prediction(model, p, n)
        if isinstance(decoder, MLDecoder):
            os = decoder.outs
            seq = torch.arange(paths.shape[2], device=device).expand(len(p), -1)
            values = dict(layer_paths=torch.stack([core.world_paths(o["paths"][:, None], tf) for o in os], 1),
                          layer_logits=torch.stack([o["logits"] for o in os], 1),
                          layer_order=torch.stack([o["order"] if o["order"] is not None else seq for o in os], 1),
                          layer_raw_quality=torch.stack([o["qraw"] for o in os], 1),
                          layer_quality=torch.stack([qhat(o["qraw"],o["paths"],decoder.past,decoder.ref,
                                                         decoder.use_kin,decoder.only_q) for o in os], 1),
                          anchor_move=torch.stack([(y["anchors"]-x["anchors"]).norm(dim=-1)
                                                   for x,y in zip(os,os[1:])], 1))
        else:
            values = dict(layer_paths=core.world_paths(paths, tf)[:, None], layer_logits=logits)
        for key, array in values.items():
            if key=="layer_paths":
                array = array.double()
            saved.setdefault(key, []).append(array.cpu().numpy())
    return {key: np.concatenate(arrays) for key, arrays in saved.items()}


def rank_correlation(a, b):
    def ranks(x):
        d = x[:, :, None]-x[:, None, :]
        return (d > 0).sum(-1)+.5*(d == 0).sum(-1)

    a, b = ranks(a), ranks(b)
    a -= a.mean(1, keepdims=True)
    b -= b.mean(1, keepdims=True)
    denom = np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)
    return np.divide((a*b).sum(1), denom, out=np.zeros(len(a)), where=denom>0)


def evaluate(core, model, data, out, fold):
    items = infer(core, model, data)
    future = data["future"]
    past = data["past"][:, 0]
    masks = groups(data)
    masks["fast"] = np.linalg.norm(np.diff(past.astype("float64"), axis=1),axis=-1).mean(1)/.4 > 1.3
    paths = items["layer_paths"]
    logits = items["layer_logits"]
    layers = [forecast_metrics(path, score, future) for path, score in zip(paths.transpose(1,0,2,3,4), logits.transpose(1,0,2))]
    last_path, last_logits = paths[:, -1], logits[:, -1]
    final = layers[-1]
    orders = dict(conf=conf_order(last_logits),
                  riskp=risk_order(last_path[:, :, -1],softmax(last_logits)))
    quality = items.get("layer_quality")
    if quality is not None:
        orders["qual"] = np.argsort(quality[:, -1],axis=1,kind="stable")
    err = np.linalg.norm(last_path-future[:, None], axis=-1)
    ade, fde = err.mean(2),err[:, :, -1]
    ordered = {}
    for name, order in orders.items():
        for k in (1,3,5,20):
            for metric, val in (("ADE",ade),("FDE",fde)):
                ordered[f"{name}_min{metric}{k}"] = np.take_along_axis(val,order[:, :k],axis=1).min(1)
    mc, gap = {}, np.zeros(len(past))
    if quality is not None:
        conf = conf_order(logits[:, 0])
        actual = items["layer_order"][:, 1]
        mc["first_diff"] = float(np.mean(actual[:, 0]!=conf[:, 0]))
        mc["top5_overlap"] = float((actual[:, :5, None]==conf[:, None, :5]).any(-1).mean())
        mc["quality_spearman"] = {}
        for i in (0,len(layers)-1):
            distance = np.linalg.norm(paths[:, i]-future[:, None],axis=-1)
            q = distance.mean(2)+.25*distance[:, :, -1]
            mc["quality_spearman"][str(i)] = dict(head=float(np.median(rank_correlation(quality[:,i],q))),
                                                    confidence=float(np.median(rank_correlation(-logits[:,i],q))))
        move = items["anchor_move"]
        mc["anchor_move_median_m"] = np.median(move,axis=(0,2)).tolist()
        mc["stationary_anchor_move_median_m"] = (np.median(move[masks["stationary"]],axis=(0,2)).tolist()
                                                 if masks["stationary"].any() else None)
        ordered_end = np.take_along_axis(last_path[:, :, -1],items["layer_order"][:, -1,:,None],axis=1)
        pair = np.linalg.norm(ordered_end[:, :,None]-ordered_end[:,None,:],axis=-1)
        gap = np.where(np.arange(20)[None,:] < np.arange(20)[:,None],pair,np.inf)[:,5:].min(2).mean(1)
    rescue = (layers[0]["minFDE"]>1.)&(final["minFDE"]<=1.)
    lost = (layers[0]["minFDE"]<=1.)&(final["minFDE"]>1.)
    reversal = final["Top1_FDE"]>layers[0]["Top1_FDE"]+.5
    mechanism = dict(rescue=rescue, loss=lost, top1_reversal=reversal,
                     gap_m=gap)
    payload = {key: value for key,value in items.items() if key not in ("layer_paths","anchor_move")}
    payload.update(paths=last_path,logits=last_logits,ids=data["ids"])
    if fold=="C0":
        payload["layer_paths"] = paths.astype("float16")
    metrics = {**final,**ordered,**mechanism}
    files = {"dev_predictions.npz":safe_npz(out/"dev_predictions.npz", **payload),
             "dev_metrics.npz":safe_npz(out/"dev_metrics.npz",ids=data["ids"],**masks,**metrics)}
    return dict(dev=metric_summary(final,masks),layers=[metric_summary(m,masks) for m in layers],
                ordered=metric_summary(ordered,masks),mechanism=metric_summary(mechanism,masks),
                manipulation=mc,artifacts=files)
