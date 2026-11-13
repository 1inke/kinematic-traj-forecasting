"""Two preregistered bases; no inner fitting or rank search."""
import numpy as np

def knot3():
    t=np.arange(1,13,dtype=np.float64)
    return np.stack((np.maximum(0,1-np.abs(t-4)/4),np.maximum(0,1-np.abs(t-8)/4),np.maximum(0,(t-8)/4)))

def cv(past):
    v=(np.asarray(past,np.float64)[:,-1]-np.asarray(past,np.float64)[:,-2])/.4
    return v[:,None]*np.arange(1,13)[None,:,None]*.4

def svd4(gt_fit,past_fit):
    residual=np.asarray(gt_fit,np.float64)-cv(past_fit)
    r=residual.transpose(0,2,1).reshape(-1,12)
    _,singular,vh=np.linalg.svd(r,full_matrices=False)
    psi=vh[:4].copy()
    if np.any(psi[:,-1]==0):
        raise ValueError('SVD endpoint sign undefined: STOP, no implicit alternative basis')
    psi*=np.sign(psi[:,-1,None])
    return psi,singular

def projector(psi):
    psi=np.asarray(psi,np.float64)
    if psi.shape not in ((3,12),(4,12)) or np.linalg.matrix_rank(psi)!=len(psi):
        raise ValueError('Unregistered or singular basis')
    p=psi.T@np.linalg.solve(psi@psi.T,psi)
    np.testing.assert_allclose(p,p.T,atol=2e-12,rtol=0)
    np.testing.assert_allclose(p@p,p,atol=2e-12,rtol=0)
    return p

def project(paths,past,psi):
    p=projector(psi);anchor=cv(past)
    if paths.ndim==4: anchor=anchor[:,None]
    return anchor+np.einsum('tu,...ud->...td',p,np.asarray(paths,np.float64)-anchor)

def screen_gate(gt_good,projected_good,old_good,native_good,acceleration_ratio):
    gains=np.asarray(old_good)-native_good
    if np.any(gains<=0):
        return dict(passed=False,reason='CONTROL_GAIN_NOT_REPRODUCED')
    retention=(np.asarray(projected_good)-native_good)/gains
    passed=gt_good>=.98 and bool(np.all(retention>=.8)) and float(np.mean(acceleration_ratio))<=.85
    return dict(passed=passed,oracle_good=gt_good,netgain_retention=retention.tolist(),
                mean_acceleration_ratio=float(np.mean(acceleration_ratio)))
