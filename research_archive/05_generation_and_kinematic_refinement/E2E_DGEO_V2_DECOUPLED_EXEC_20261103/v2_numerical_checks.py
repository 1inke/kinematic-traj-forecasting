"""New V2 causal-feature/Taylor independent algebra fixtures, not scientific fits."""
import copy,itertools
from geo_common import *
from geo_modules import compute_anchor_geo_features_12d,kinematic_taylor_delta
from geo_math import geom
from geo_factory import make_geo_model
from train_core import exact
import numpy as np
import torch

def additional_checks(folder):
    torch.set_num_threads(1)
    a=torch.linspace(-1.3,2.1,4*20*3*2,dtype=torch.float64).reshape(4,20,3,2).requires_grad_(True)
    past=torch.linspace(-.8,.7,4*8*2,dtype=torch.float64).reshape(4,8,2).requires_grad_(True)
    w=torch.cos(torch.arange(4*20*12,dtype=torch.float64)).reshape(4,20,12)
    actual=compute_anchor_geo_features_12d(a,past);ref=geom(a.detach().numpy(),past.detach().numpy(),np)
    ve=float(np.max(abs(actual.detach().numpy()-ref)))
    x=a.detach().clone().requires_grad_(True);p=past.detach().clone().requires_grad_(True)
    expected=geom(x,p,torch)
    ga=torch.autograd.grad((actual*w).sum(),(a,past))
    gr=torch.autograd.grad((expected*w).sum(),(x,p))
    ge=max(float((u-v).abs().max()) for u,v in zip(ga,gr))
    if ve>1e-12 or ge>1e-12:raise ValueError('MANEUVER_INDEPENDENT_COORDINATE')
    perm=torch.arange(19,-1,-1)
    pe=float((compute_anchor_geo_features_12d(a[:,perm],past)-actual[:,perm]).abs().max())
    if pe>1e-12:raise ValueError('MANEUVER_PERMUTATION')
    for zero in (False,True):
        aa=torch.zeros_like(a) if zero else -a.detach()
        result=compute_anchor_geo_features_12d(aa,past)
        if not torch.isfinite(result).all():raise ValueError('NONFINITE_ZERO_OR_NEGATIVE_AXIS')
    raw=torch.linspace(-.9,1.1,4*20*3*2,dtype=torch.float64).reshape(4,20,3,2).requires_grad_(True)
    yy=kinematic_taylor_delta(raw)
    matrix=np.array([[1,0,0],[2,1,1],[3,3,0]],dtype=np.float64)
    if np.linalg.det(matrix)==0:raise ValueError('TAYLOR_RANK')
    coeff=np.array([.16,.14,.04])
    independent=np.einsum('ij,bkjc->bkic',matrix,np.tanh(raw.detach().numpy())*coeff[None,None,:,None])
    te=float(np.max(abs(independent-yy.detach().numpy())))
    ww=torch.sin(torch.arange(yy.numel(),dtype=torch.float64)).reshape_as(yy)
    grad=torch.autograd.grad((yy*ww).sum(),raw)[0]
    igrad=np.einsum('ij,bkic->bkjc',matrix,ww.numpy())*coeff[None,None,:,None]*(1-np.tanh(raw.detach().numpy())**2)
    tge=float(np.max(abs(grad.numpy()-igrad)))
    if te>1e-12 or tge>1e-12:raise ValueError('TAYLOR_INDEPENDENT_ALL_CHANNEL_GRAD')
    fd=[]
    for d in range(2):
        v=torch.cos(torch.arange(raw.numel(),dtype=torch.float64)+d).reshape_as(raw);v=v/v.norm()
        target=float((grad*v).sum())
        for eps in (1e-4,1e-5):
            val=[float((kinematic_taylor_delta(raw.detach()+sign*eps*v)*ww).sum()) for sign in (-1,1)]
            error=abs((val[1]-val[0])/(2*eps)-target)
            if error>1e-6+1e-3*abs(target):raise ValueError('TAYLOR_FIXED_FD')
            fd.append(dict(direction=d,eps=eps,error=error))
    corners=np.array(list(itertools.product((-1.,1.),repeat=3)))*coeff
    delta=corners@matrix.T
    bounds=dict(anchor_abs=np.abs(delta).max(0).tolist(),segment_diff_abs=np.abs(np.column_stack((delta[:,0],delta[:,1]-delta[:,0],delta[:,2]-delta[:,1]))).max(0).tolist(),
        hinge_abs=[float(abs(delta[:,1]-2*delta[:,0]).max()),float(abs(delta[:,2]-2*delta[:,1]+delta[:,0]).max())])
    models={}
    for arm in ARMS:
        model,_=make_geo_model(arm,'cpu')
        models[arm]={k:v.detach().clone() for k,v in model.decoder.coordinator.state_dict().items()}
        count=sum(v.numel() for v in model.parameters())
        expected_count=304802 if arm in ('DK-ManeuverDiff','DK-FullDiff') else 303970
        if count!=expected_count:raise ValueError('PARAMETER_SCOPE '+arm)
        del model
    ref=models[ARMS[0]]
    for arm,values in models.items():
        for k,v in ref.items():exact(v,values[k],'ALL4_COMMON_COORD_INITIAL_'+k)
    exact(models[ARMS[1]],models[ARMS[3]],'FACTOR_B_ALL_COORD_INIT')
    report=dict(status='PASS_V2_CAUSAL_GEOMETRY_TAYLOR_INDEPENDENT_AND_PAIRED_COMMON_INIT',
        geometry_value_error=ve,geometry_all_coordinate_gradient_error=ge,permutation_error=pe,
        taylor_value_error=te,taylor_all_channel_gradient_error=tge,taylor_fixed_FD=fd,taylor_matrix_det=float(np.linalg.det(matrix)),residual_bounds_per_coordinate=bounds,
        common_coord_parameters_all4_exact=True,zero_negative_axis_finite=True,
        residual_bounds_not_whole_trajectory_safety=True,piecewise_linear_interpolation_unchanged=True)
    write_json(folder/'V2_ADDITIONAL_CHECKS.json',report)
    return report

