"""Fixed-budget training and per-run artifacts; resume at complete epochs."""
import copy
import hashlib
import math
import random
import time
import numpy as np
import torch
from dec_io import ROOT, ARMS, read, write, save_torch, npz, seal, verify, now
from dec_model import InfoRefiner, predict, make_basis, route
from dec_analysis import aggregate, window_metrics
from probe_model import baseline_loss
from probe_train import Trainer as PriorTrainer, cpu_tree


def new_model(arm, seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return InfoRefiner(arm).to(device)


class Trainer(PriorTrainer):
    def __init__(self, arm, seed, cache, head, criterion, device, epochs=20, batch=256):
        self.arm, self.seed, self.cache, self.head, self.criterion, self.device = arm, seed, cache, head, criterion, device
        self.epochs, self.batch = epochs, batch
        self.model = new_model(arm, seed, device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=.001, weight_decay=.0001, foreach=False)
        self.steps = math.ceil(len(cache['y']) / batch)
        self.total = epochs * self.steps
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda i: .5 * (1 + math.cos(math.pi * min(i / self.total, 1))))
        self.generator = torch.Generator().manual_seed(20261117 + seed)
        self.epoch = self.updates = 0
        self.history, self.connected = [], set()

    def epoch_step(self):
        start = time.monotonic()
        order = torch.randperm(len(self.cache['y']), generator=self.generator)
        digest = hashlib.sha256(order.numpy().astype('<i8').tobytes()).hexdigest()
        self.model.train()
        total = count = 0
        for i in range(0, len(order), self.batch):
            idx = order[i:i + self.batch].to(self.device)
            paths, logits, _ = predict(self.model, self.head, self.cache, idx)
            loss, _ = baseline_loss(paths, self.cache['y'][idx], logits, self.criterion)
            if not torch.isfinite(loss):
                raise ValueError('NONFINITE_LOSS')
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            gradients = [p.grad.flatten() for p in self.model.parameters() if p.grad is not None]
            if not torch.isfinite(torch.cat(gradients)).all():
                raise ValueError('NONFINITE_GRADIENT')
            self.opt.step()
            self.scheduler.step()
            self.updates += 1
            total += float(loss.detach()) * len(idx)
            count += len(idx)
        self.epoch += 1
        for name, p in self.model.named_parameters():
            if p.grad is not None and bool(p.grad.abs().any()):
                self.connected.add(name)
        row = dict(epoch=self.epoch, updates=self.updates, seconds=time.monotonic()-start,
                   loss=total/count, order_hash=digest, lr=self.opt.param_groups[0]['lr'])
        self.history.append(row)
        return row


@torch.no_grad()
def evaluate(model, head, data, social, donor=None):
    t, meta = data['tensors'], data['meta']
    basis = make_basis().to(t['relative'].device)
    if model is not None:
        model.eval()
    parts = []
    for lo in range(0, len(social), 256):
        i = slice(lo, lo + 256)
        if model is None:
            anchors = t['relative'][i]
            out = torch.einsum('tp,nkpc->nktc', basis, anchors), t['logits'][i], torch.zeros_like(anchors)
        elif donor is None:
            out = predict(model, head, t, i)
        else:
            jj = torch.as_tensor(donor[i], device=t['relative'].device)
            nbr = ('tokens_nbr', 'nbr_pos', 'nbr_vel', 'nbr_valid')
            xb = {k: (v[jj] if k in nbr else v[i]) for k, v in t.items() if k != 'y'}
            assert torch.equal(xb['nbr_valid'], t['nbr_valid'][i])
            out = predict(model, head, xb)
        assert all(bool(torch.isfinite(v).all()) for v in out)
        parts.append(tuple(v.cpu().numpy() for v in out))
    paths, scores, shift = (np.concatenate(v) for v in zip(*parts))
    w = window_metrics(paths, scores, shift, t['y'].cpu().numpy())
    return aggregate(w, meta, social), w, dict(paths=paths, logits=scores, shift=shift)


@torch.no_grad()
def future_shuffle_check(model, head, tensors):
    sample = {k: v[:256] for k, v in tensors.items()}
    changed = dict(sample, y=sample['y'].roll(1, 0) + 5.)
    model.eval()
    original = predict(model, head, sample)
    alternative = predict(model, head, changed)
    for a, b in zip(original, alternative):
        assert torch.equal(a, b)
    if model.arm in ('C2','C3','C4'):
        def selections(x):
            paths = torch.einsum('tp,nkpc->nktc', model.basis, x['relative'])
            return route(model.arm, paths, *(x[k] for k in ('nbr_pos','nbr_vel','nbr_valid','self_vel')))[:2]
        for a, b in zip(selections(sample), selections(changed)):
            assert torch.equal(a, b)
    return dict(status='BITWISE_EQUAL', windows=len(sample['y']), geometry=True, scores=True,
                selected_neighbors=True if model.arm in ('C2','C3','C4') else None)


def latency(model, head, tensors):
    from dec_model import batch_inputs
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        m = copy.deepcopy(model).cpu().eval()
        h = copy.deepcopy(head).cpu().eval()
        counts = tensors['nbr_valid'].sum(-1).cpu().numpy()
        # Same fixed representatives spanning observed neighbor counts for every arm/seed.
        rows = np.array([np.flatnonzero(counts == k)[0] for k in np.unique(counts)])
        samples = [{k: v[i:i + 1].cpu() for k, v in tensors.items() if k != 'y'} for i in rows]
        args = [batch_inputs(m, s, None) for s in samples]
        result = dict(batch_size=1, cpu_threads=1, repeats=200, warmup=50,
                      dev_row_indices=rows.tolist(), representative_neighbor_counts=counts[rows].tolist())
        with torch.inference_mode():
            for name, fn in [('refiner_only', lambda i: m(*args[i])),
                             ('including_frozen_score_and_interpolation', lambda i: predict(m, h, samples[i]))]:
                for j in range(50):
                    fn(j % len(rows))
                values = []
                for j in range(200):
                    t = time.perf_counter()
                    fn(j % len(rows))
                    values.append((time.perf_counter()-t)*1000)
                result[name] = dict(median_ms=float(np.median(values)), p90_ms=float(np.quantile(values, .9)))
        return result
    finally:
        torch.set_num_threads(threads)


def verify_initial(trainer, initial, seed):
    p = ROOT / f'runs/A1_s{seed}/initial.pt'
    if p.exists():
        paired = torch.load(p, map_location='cpu', weights_only=False)
        for k, v in paired.items():
            actual = initial[k]
            if trainer.arm == 'A1L' and k in ('ffn.0.weight', 'ffn.0.bias'):
                actual = actual[:48]
            if trainer.arm == 'A1L' and k == 'ffn.2.weight':
                actual = actual[:, :48]
            assert torch.equal(actual, v), (trainer.arm, k)
    q = ROOT / f'runs/C1_s{seed}/initial.pt'
    if trainer.arm.startswith('C') and q.exists():
        paired = torch.load(q, map_location='cpu', weights_only=False)
        assert paired.keys() == initial.keys()
        for k in paired:
            assert torch.equal(paired[k], initial[k]), (trainer.arm, k)


def train_one(arm, seed, data, head, criterion, device, contract, social, donor):
    folder = ROOT / f'runs/{arm}_s{seed}'
    done = folder / 'DONE.json'
    if done.exists():
        verify(read(done)['files'])
        return read(folder / 'RESULT.json')
    trainer = Trainer(arm, seed, data['fit']['tensors'], head, criterion, device)
    initial = cpu_tree(trainer.model.state_dict())
    verify_initial(trainer, initial, seed)
    initial_file = folder / 'initial.pt'
    if initial_file.exists():
        saved = torch.load(initial_file, map_location='cpu', weights_only=False)
        assert saved.keys() == initial.keys() and all(torch.equal(initial[k], saved[k]) for k in initial)
    else:
        save_torch(initial_file, initial)
    receipts = sorted((folder / 'checkpoints').glob('epoch*.json'))
    if receipts:
        record = read(receipts[-1])
        verify(record['files'])
        trainer.restore(torch.load(next(iter(record['files'])), map_location='cpu', weights_only=False), contract)
        print(f'RESUME {arm} seed={seed} complete_epoch={trainer.epoch}', flush=True)
    orders = (read(ROOT / f'runs/A1_s{seed}/TRAINING.json')['history'] if arm != 'A1' else None)
    for i in range(trainer.epoch, 20):
        write(ROOT / 'CURRENT.json', dict(at=now(), phase='TRAIN', arm=arm, seed=seed, epoch=i+1, epochs=20))
        row = trainer.epoch_step()
        if orders is not None:
            assert orders[i]['order_hash'] == row['order_hash']
        checkpoint = folder / f'checkpoints/epoch{i+1:02d}.pt'
        save_torch(checkpoint, trainer.snapshot(contract))
        write(checkpoint.with_suffix('.json'), dict(epoch=i+1, files=seal([checkpoint])))
        write(folder / 'TRAINING.json', dict(arm=arm, seed=seed, history=trainer.history,
              epochs=trainer.epoch, updates=trainer.updates, connected=sorted(trainer.connected)))
        print(f'{arm} seed={seed} epoch={i+1}/20 seconds={row["seconds"]:.3f} loss={row["loss"]:.8f}', flush=True)
        if (ROOT / 'PAUSE.request').exists():
            write(ROOT / 'CURRENT.json', dict(at=now(), phase='PAUSED', arm=arm, seed=seed, epoch=i+1))
            return None
    save_torch(folder / 'final.pt', trainer.snapshot(contract))
    write(ROOT / 'CURRENT.json', dict(at=now(), phase='EVALUATE', arm=arm, seed=seed))
    clean, w, pred = evaluate(trainer.model, head, data['dev'], social)
    meta = data['dev']['meta']
    extra = dict(ids=meta['ids'], group=meta['group'], labels=meta['labels'], social=social)
    npz(folder / 'window_metrics.npz', **w, **extra)
    npz(folder / 'predictions_local.npz', **pred, ids=meta['ids'])
    result = dict(arm=arm, seed=seed, clean=clean, parameter_count=sum(p.numel() for p in trainer.model.parameters()),
                  parameters_with_nonzero_gradient=sum(p.numel() for n, p in trainer.model.named_parameters() if n in trainer.connected),
                  latency=latency(trainer.model, head, data['dev']['tensors']),
                  future_shuffle=future_shuffle_check(trainer.model, head, data['dev']['tensors']),
                  training_seconds=sum(r['seconds'] for r in trainer.history), epochs=20, updates=trainer.updates)
    more = []
    if arm in ('C2','C3','C4'):
        shuffled, w2, pred2 = evaluate(trainer.model, head, data['dev'], social, donor)
        result['shuffled'] = shuffled
        npz(folder / 'window_metrics_shuffled.npz', **w2, **extra)
        npz(folder / 'predictions_shuffled_local.npz', **pred2, ids=meta['ids'])
        more = [folder / 'window_metrics_shuffled.npz', folder / 'predictions_shuffled_local.npz']
    write(folder / 'RESULT.json', result)
    files = [folder / n for n in ('initial.pt','final.pt','TRAINING.json','RESULT.json','window_metrics.npz','predictions_local.npz')]
    write(done, dict(at=now(), files=seal(files + more)))
    return result
