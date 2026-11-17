"""One real instantiation/forward, no scientific training."""
from common import *
from baseline_factory import *
import time

if __name__ == '__main__':
    torch.set_num_threads(1)
    data = np.load(HERE / 'C0/fit.npz')
    model, cfg = make_model()
    criterion = ModeQueryLoss(cfg).cuda()
    p = torch.tensor(data['past'][:8], device='cuda')
    y = torch.tensor(data['future'][:8], device='cuda')
    n = torch.tensor(data['num_valid'][:8], device='cuda')
    pred, logits, target, tr = prediction(model, p, n, y)
    loss, _ = loss_value(pred, target, logits, criterion)
    loss.backward()
    report = dict(status='ACTUAL_MODEL8_FORWARD_BACKWARD_NOT_FULL_PREFLIGHT',
        parameters=sum(p.numel() for p in model.parameters()),
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        shape=list(pred.shape), loss=float(loss.detach()),
        nonfinite=[k for k,v in model.named_parameters() if v.grad is not None and not torch.isfinite(v.grad).all()],
        zero_or_unconnected=[k for k,v in model.named_parameters() if v.requires_grad and (v.grad is None or not v.grad.abs().any())],
        resolved_config={k:v for k,v in cfg.items() if k not in ('device',)},
        device=str(cfg.device))
    report['resolved_config']['device_list'] = [0]
    write_json(HERE / 'MODEL_INSPECTION.json', report)
    print(report['parameters'], report['trainable'], report['shape'], report['loss'], flush=True)
