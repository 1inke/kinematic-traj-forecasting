"""Visible-only transforms. Never use the final point of an unmasked full window."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class VisibleObservation:
    positions: np.ndarray
    valid: np.ndarray

def visible_local(full_positions,visible_steps,valid=None):
    a=np.asarray(full_positions)
    if a.ndim!=3 or a.shape[-1]!=2 or not 1<=visible_steps<=a.shape[1]:raise ValueError('Expected B,T,2 and valid visible length')
    x=a[:,:visible_steps].astype(np.float64,copy=True)
    v=np.ones(x.shape[:2],bool) if valid is None else np.asarray(valid[:,:visible_steps],bool).copy()
    if v.shape!=x.shape[:2]:raise ValueError('Validity mask shape mismatch')
    if not v.all():raise ValueError('Visible prefix must be available; no hidden-value imputation')
    if not np.isfinite(x).all():raise ValueError('Nonfinite visible point')
    origin=x[:,:1].copy();delta=x[:,-1]-x[:,max(0,visible_steps-4)];magnitude=np.linalg.norm(delta,axis=-1)
    u=delta/np.maximum(magnitude[:,None],1e-8);u[magnitude<=.03]=[1.,0.]
    dx=x-origin;out=np.stack((dx[:,:,0]*u[:,None,0]+dx[:,:,1]*u[:,None,1],-dx[:,:,0]*u[:,None,1]+dx[:,:,1]*u[:,None,0]),-1)
    return VisibleObservation(out,v)

def fit_statistics(features,sample_ids,fit_ids):
    ids=np.asarray(sample_ids);allowed=set(np.asarray(fit_ids).tolist())
    if len(np.unique(ids))!=len(ids) or not set(ids.tolist()).issubset(allowed):raise ValueError('Statistics IDs must be unique FIT IDs')
    x=np.asarray(features,dtype=np.float64)
    if not len(ids) or len(x)!=len(ids) or not np.isfinite(x).all():raise ValueError('Bad FIT feature data')
    mean=x.mean(0);std=x.std(0);return {'mean':mean,'std':np.where(std<1e-6,1.,std),'training_ids':ids.copy()}
