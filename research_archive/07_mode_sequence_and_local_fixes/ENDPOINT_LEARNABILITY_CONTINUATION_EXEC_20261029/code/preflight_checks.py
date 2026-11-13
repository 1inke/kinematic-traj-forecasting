"""Callable preflight checks. No automatic receipt or experiment authorization.

The caller must supply real registered train/val data and the measured resource
guard. Synthetic success never produces PASS_REAL_DATA_GPU_RESUME_AND_TINY.
"""
import copy
import hashlib
import io
import numpy as np
from scipy.special import erf


def numpy_loss_gradients(parameters, x, y):
    """Independent double MLP/GELU/tanh MSE reverse pass, all parameters."""
    x, y = np.asarray(x, np.float64).reshape(-1, 40), np.asarray(y, np.float64).reshape(-1, 2)
    w0, b0, w1, b1, w2, b2 = [np.asarray(parameters['layers.'+key], np.float64)
                               for key in ('0.weight', '0.bias', '2.weight', '2.bias', '4.weight', '4.bias')]
    def gelu(z):
        return .5*z*(1+erf(z/np.sqrt(2)))
    def prime(z):
        return .5*(1+erf(z/np.sqrt(2)))+z*np.exp(-.5*z*z)/np.sqrt(2*np.pi)
    z0 = x@w0.T+b0
    a0 = gelu(z0)
    z1 = a0@w1.T+b1
    a1 = gelu(z1)
    prediction = np.tanh(a1@w2.T+b2)
    g2 = 2*(prediction-y)/y.size*(1-prediction**2)
    g1 = (g2@w2)*prime(z1)
    g0 = (g1@w1)*prime(z0)
    gradients = dict(zip(('layers.0.weight', 'layers.0.bias', 'layers.2.weight',
                         'layers.2.bias', 'layers.4.weight', 'layers.4.bias'),
                        (g0.T@x, g0.sum(0), g1.T@a0, g1.sum(0), g2.T@a1, g2.sum(0))))
    return float(((prediction-y)**2).mean()), gradients


def gradient_check(features8, targets8, device='cpu'):
    import torch
    from model import ParameterRegressor, window_mse
    from train import seed_everything
    if features8.shape != (8, 20, 40) or targets8.shape != (8, 20, 2):
        raise ValueError('exact actual8 feature/target shapes required')
    seed_everything(20261028, use_cuda=torch.device(device).type == 'cuda')
    m = ParameterRegressor().to(device=device, dtype=torch.float64)
    # Fixed nonzero final weights activate hidden connections; no scientific model saved.
    with torch.no_grad():
        m.layers[-1].weight.copy_(torch.randn_like(m.layers[-1].weight)*.01)
    x = torch.from_numpy(np.asarray(features8, np.float64)).to(device)
    y = torch.from_numpy(np.asarray(targets8, np.float64)).to(device)
    loss = window_mse(m(x), y)
    loss.backward()
    params = {name: v.detach().cpu().numpy() for name, v in m.named_parameters()}
    reference, grads = numpy_loss_gradients(params, features8, targets8)
    maximum = max(float(np.max(np.abs(v.grad.detach().cpu().numpy()-grads[name])))
                  for name, v in m.named_parameters())
    if abs(float(loss.detach().cpu())-reference) > 1e-10 or maximum > 1e-10:
        raise AssertionError('independent double scalar/all-gradient mismatch')
    # Fixed parameter/epsilons, not searching favorable finite-difference steps.
    parameter = m.layers[-1].weight
    analytic = float(parameter.grad[0, 0].detach().cpu())
    original = float(parameter[0, 0].detach().cpu())
    fd = []
    for epsilon in (1e-4, 1e-5):
        with torch.no_grad():
            parameter[0, 0] = original+epsilon
            plus = float(window_mse(m(x), y).cpu())
            parameter[0, 0] = original-epsilon
            minus = float(window_mse(m(x), y).cpu())
            parameter[0, 0] = original
        numeric = (plus-minus)/(2*epsilon)
        if abs(numeric-analytic) > 1e-8:
            raise AssertionError('fixed FD tolerance failed')
        fd.append(dict(epsilon=epsilon, numeric=numeric, analytic=analytic))
    return dict(status='PASS_ISOLATED_DOUBLE_GRADIENT_CHECK', device=str(device),
                max_gradient_error=maximum, finite_difference=fd,
                not_float32_quantization_proof=True)


def trees_equal(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(trees_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(trees_equal(x, y) for x, y in zip(a, b))
    return a == b


def real_subset_resume(features513, targets513, validation12, validation_targets12,
                       contract, device='cpu', check=lambda: None):
    import torch
    from train import Engine, infer
    if len(features513) != 513 or len(validation12) != 12:
        raise ValueError('require actual513 train and actual12 val, no synthetic substitution')
    full = Engine(features513, targets513, contract, epochs=2, device=device)
    while not full.complete:
        check()
        full.step()
    expected = full.snapshot()
    expected_val = infer(full.model, validation12, device, check)
    # Do not retain both CUDA models during the second run.
    del full
    partial = Engine(features513, targets513, contract, epochs=2, device=device)
    for _ in range(19):
        check()
        partial.step()
    stream = io.BytesIO()
    torch.save(partial.snapshot(), stream)
    del partial
    stream.seek(0)
    restored_state = torch.load(stream, map_location='cpu', weights_only=False)
    resumed = Engine(features513, targets513, contract, epochs=2, device=device)
    resumed.restore(restored_state)
    while not resumed.complete:
        check()
        resumed.step()
    if not trees_equal(expected, resumed.snapshot()):
        raise AssertionError('params/optimizer/RNG/history/order resume not exact')
    actual_val = infer(resumed.model, validation12, device, check)
    if not np.array_equal(expected_val, actual_val):
        raise AssertionError('post-resume validation predictions not exact')
    target = np.asarray(validation_targets12, float)
    return dict(status='PASS_CALLER_SUPPLIED_513_12_SUBSET_RESUME', updates=resumed.updates,
                val_mse=float(((actual_val-target)**2).mean()), device=str(device),
                full_population_resume=False, caller_must_attest_ID_and_source=True)


def tiny_fit(features, targets, sample_id, device='cpu', check=lambda: None):
    from train import Engine, infer
    active_window = np.any(targets != 0, axis=(1, 2))
    def selected(mask):
        indices = np.flatnonzero(mask)
        return sorted(indices, key=lambda i: hashlib.sha256(
            ('endpoint-temporal-tiny-v1|'+str(sample_id[i])).encode()).hexdigest())[:64]
    positive, negative = selected(active_window), selected(~active_window)
    if len(positive) != 64 or len(negative) != 64:
        return dict(status='STOP_TINY_STRATUM_INSUFFICIENT', active=len(positive), inactive=len(negative))
    ids = np.array(positive+negative)
    x, y = features[ids], targets[ids]
    active = np.any(y != 0, axis=-1)
    engine = Engine(x, y, {'tiny_fixture_ids': sample_id[ids].tolist()}, epochs=64, device=device)
    before = infer(engine.model, x, device, check)
    for _ in range(256):
        check()
        engine.step()
    after = infer(engine.model, x, device, check)
    initial = float(((before.astype(float)-y)**2).mean(-1)[active].mean())
    final = float(((after.astype(float)-y)**2).mean(-1)[active].mean())
    drop = 1-final/initial
    return dict(status='PASS_TINY_MSE_ONLY' if drop >= .9 else 'STOP_TINY_FIT',
                initial_active_mse=initial, final_active_mse=final, relative_drop=drop,
                sample_id=sample_id[ids].tolist(), updates=256,
                not_generalization=True, requires_action_hit_and_regret_supplement=True)
