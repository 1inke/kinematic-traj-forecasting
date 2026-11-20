"""Read and verify existing C0 forecasts and recreate the fit-only quantizer."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/"NEXT_PHASE_EXEC_20261114"
CAL=ROOT.parent/"EXP_SET_CALIB_DIAG_EXEC_20261119"
PREC=ROOT.parent/"EXP_PREC_DIAG_EXEC_20261119"
SCORE=ROOT.parent/"EXP_SCORE_COUPLING_DIAG_EXEC_20261120"
PROBE=ROOT.parent/"EXP_SCORE_COUPLING_PROBE_EXEC_20261120"
sys.path[:0]=[str(CAL),str(PREC),str(SCORE)]

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from calib_data import BINS, OBS, coordinates, history_groups, observed_speed, split_halves
from calib_metrics import SEED as CAL_SEED
from run_prec_diag import arrays, load_prediction, maneuver_labels, read, sha, write
from adapt_math import errors, probabilities

SEEDS=(42,43,44,45,46)
LEGACY=("T1","T2","T3","E2_natural","E2_clamped","E3_set","E3_set_nobias","E3_self")
PROBE_ARMS=("R-couple","R-detach","R-nogeom")


def verify(p,h,files):
    v=sha(p)
    assert v==h,p
    files[str(p)]=v


def prepare_data(files):
    pinned=read(CAL/"INPUT_MANIFEST.json")["files"]
    data={}
    for name,n in (("fit",21719),("dev",4753)):
        p=OLD/f"C0/{name}.npz"
        verify(p,pinned[str(p)],files)
        d=arrays(p)
        assert len(np.unique(d["ids"]))==n
        past=d["past"][:,0].astype(np.float64)
        d["last"]=past[:,-1,None]
        d["cv"],_,d["rotation"]=coordinates(past)
        d["obs"]=history_groups(past)
        d["speed"]=observed_speed(d["past"][:,0])
        d["label"]=maneuver_labels(d["past"],d["future"])
        d["track"]=np.asarray([v.rsplit("|",1)[0] for v in d["ids"]])
        d["scene"]=np.asarray([v.split("|",1)[0] for v in d["ids"]])
        d["target_rel"]=d["future"].astype(np.float64)-d["last"]
        data[name]=d
    fit,dev=data["fit"],data["dev"]
    cuts={name:np.quantile(fit["speed"][fit["obs"]==name],[1/3,2/3]) for name in OBS[1:]}
    for d in (fit,dev):
        idx=np.zeros(len(d["ids"]),dtype=np.int8)
        for offset,name in ((1,"steady"),(4,"other")):
            p=d["obs"]==name
            idx[p]=offset+np.searchsorted(cuts[name],d["speed"][p],side="right")
        d["bin"]=np.asarray(BINS)[idx]
    assert [int((dev["obs"]==k).sum()) for k in OBS]==[1109,1991,1653]
    archive=CAL/"RESULT.json"
    verify(archive,read(CAL/"DONE.json")["files"]["RESULT.json"],files)
    reference=read(archive)
    for k,v in cuts.items():
        np.testing.assert_array_equal(v,reference["observation_speed_cuts_mps"][k])
    return data,reference,{k:v.tolist() for k,v in cuts.items()}


def groups(data):
    y,obs=data["label"],data["obs"]
    return {"overall":np.ones(len(y),bool),
            **{name:obs==name for name in OBS},
            **{f"bin_{name}":data["bin"]==name for name in BINS},
            "hard":np.isin(y,["turn","turn_and_speed"]),
            "remain":(obs=="stationary")&(y=="stationary"),
            "start":(obs=="stationary")&(y=="start")}


def old_paths(name,split,data,files):
    r=OLD/"runs"/name
    p=r/f"{split}/predictions.npz"
    verify(p,read(r/"EVALUATION_DONE.json")["files"][str(p)],files)
    paths,z=load_prediction(r,split,data)
    return paths.astype(np.float64)-data["last"][:,None],probabilities(z)


def probe_target(dev,files):
    # Read the CPU-readable extraction of the sealed local targets.
    folder=PROBE/"cache"
    p=folder/"dev.pt"
    verify(p,read(folder/"DONE.json")["files"]["dev.pt"],files)
    extract=read(ROOT/"TARGET_EXTRACT.json")
    assert extract["source_sha256"]==files[str(p)]
    q=ROOT/"PROBE_CACHED_TARGET.npz"
    verify(q,extract["output_sha256"],files)
    target=arrays(q)["target"].astype(np.float64)
    rel=np.einsum("nij,nti->ntj",dev["rotation"],dev["target_rel"])
    hist=np.einsum("nij,nti->ntj",dev["rotation"],dev["past"][:,0].astype(np.float64)-dev["last"])
    np.testing.assert_allclose(rel-hist.mean(1,keepdims=True),target,atol=1e-5,rtol=0)
    return target


def probe_paths(arm,seed,dev,y,files):
    r=PROBE/f"runs/{arm}_s{seed}"
    p=r/"dev_predictions.npz"
    seal=read(r/"DONE.json")["files"]
    verify(p,seal[p.name],files)
    q=arrays(p)
    np.testing.assert_array_equal(q["ids"],dev["ids"])
    path=q["paths"].astype(np.float64)
    w=probabilities(q["logits"])
    metrics=errors(path,y)
    derived=r/"dev_metrics.npz"
    verify(derived,seal[derived.name],files)
    evidence=arrays(derived)
    np.testing.assert_array_equal(evidence["ids"],dev["ids"])
    for key in ("minADE","minFDE"):
        np.testing.assert_allclose(metrics[key],evidence[key],atol=1e-12,rtol=0)
    np.testing.assert_array_equal(metrics["J"],evidence["J_coverage"])
    return path,w


def quantizer(data,reference,files):
    fit,dev=data["fit"],data["dev"]
    ref_bins=reference["observation_bins"]
    y=np.einsum("nij,nti->ntj",fit["rotation"],fit["future"]-fit["cv"])
    out=np.empty((len(dev["ids"]),20,12,2))
    w=np.empty(out.shape[:2])
    detail={}
    with threadpool_limits(limits=1):
        for name in BINS:
            f,d=fit["bin"]==name,dev["bin"]==name
            k=KMeans(n_clusters=20,init="k-means++",n_init=10,max_iter=300,
                     random_state=CAL_SEED,algorithm="lloyd",tol=1e-4).fit(y[f].reshape(-1,24))
            assert f.sum()==ref_bins[name]["fit_n"] and d.sum()==ref_bins[name]["dev_n"]
            count=np.bincount(k.labels_,minlength=20)
            assert np.all(count>0)
            out[d]=dev["cv"][d,None]+np.einsum("nij,ktj->nkti",dev["rotation"][d],k.cluster_centers_.reshape(20,12,2))
            w[d]=count/count.sum()
            detail[name]=dict(fit_counts=count.tolist(),n_iter=int(k.n_iter_))
    old=CAL/"windows/B0_s42.npz"
    verify(old,read(CAL/"DONE.json")["files"]["windows/B0_s42.npz"],files)
    saved=arrays(old)
    np.testing.assert_array_equal(saved["ids"],dev["ids"])
    np.testing.assert_array_equal(saved["reference_bin"],dev["bin"])
    met=errors(out,dev["future"].astype(np.float64))
    discrepancy={}
    for k in ("minADE","minFDE"):
        baseline=saved[f"t12__reference_{k}"]
        np.testing.assert_allclose(met[k],baseline,atol=1e-9,rtol=0)
        discrepancy[k]=float(np.max(abs(met[k]-baseline)))
    return out-dev["last"][:,None],w,dict(bins=detail,archive_delta=discrepancy)
