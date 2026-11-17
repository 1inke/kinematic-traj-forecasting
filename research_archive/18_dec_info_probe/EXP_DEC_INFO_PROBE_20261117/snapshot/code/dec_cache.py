"""Augment a new cache with frozen h3 input tokens and observed neighbor states."""
import sys
import numpy as np
import torch
from torch.nn import functional as F
from dec_io import ROOT, OLD, PREVIOUS, HELPER, arrays, read, write, save_torch, seal, verify, sha, now
from probe_cache import criterion_from
from probe_model import score_head


def build(device):
    folder = ROOT / 'cache'
    if (folder / 'DONE.json').exists():
        verify(read(folder / 'DONE.json')['files'])
        return
    previous_cache = PREVIOUS / 'cache'
    verify(read(previous_cache / 'DONE.json')['files'])
    sys.path.insert(0, str(OLD))
    import baseline_factory as bf
    sys.path.insert(0, str(HELPER))
    from eth_speed_math import maneuver_labels
    model, cfg = bf.make_model(device, 42)
    assert float(cfg.DATA.FREQUENCY) == 2.5
    base_path = OLD / 'runs/E1C_E64_s42/final.pt'
    base = torch.load(base_path, map_location='cpu', weights_only=False)
    model.load_state_dict(base['ema'], strict=True)
    model.eval().requires_grad_(False)
    captured = {}
    def pre(m, inp):
        tokens, past = inp
        assert tokens.shape[1:] == (9, 8, 96)
        captured['tokens_self'] = tokens[:, 0].to(torch.float16)
        captured['tokens_nbr'] = tokens[:, 1:, -1].to(torch.float16)
        captured['offset'] = past[:, 0, -1]
    def mode(m, i, o): captured['mode'] = o[:, 0]
    def scale(m, i, o): captured['scale'] = F.softplus(o[:, 0]) + .25
    def delta(m, i, o): captured['delta'] = (.75 * torch.tanh(o[:, 0])).reshape(len(o), 20, 3, 2)
    hooks = [model.decoder.register_forward_pre_hook(pre),
             model.decoder.mode_mlp.register_forward_hook(mode),
             model.decoder.proposal_scale.register_forward_hook(scale),
             model.decoder.proposal_delta.register_forward_hook(delta)]
    assets = torch.load(previous_cache / 'frozen_assets.pt', map_location='cpu', weights_only=False)
    for key, value in assets['score_head'].items():
        assert torch.equal(value, base['ema']['decoder.score_head.' + key])
    save_torch(folder / 'frozen_assets.pt', assets)
    checks = []
    for split, size in (('fit', 21719), ('dev', 4753)):
        data = arrays(OLD / f'C0/{split}.npz')
        prior = torch.load(previous_cache / f'{split}.pt', map_location='cpu', weights_only=False)
        saved = arrays(OLD / f'runs/E1C_E64_s42/{split}/predictions.npz')
        ids = data['ids']
        assert len(ids) == size and len(np.unique(ids)) == size and set(ids) == set(saved['ids'])
        lookup = {v: i for i, v in enumerate(saved['ids'])}
        order = np.array([lookup[v] for v in ids])
        np.testing.assert_array_equal(prior['meta']['ids'], ids)
        np.testing.assert_array_equal(prior['meta']['group'], data['group'])
        np.testing.assert_array_equal(prior['meta']['labels'], maneuver_labels(data['past'], data['future']))
        pieces = {k: [] for k in ('tokens_self', 'tokens_nbr', 'nbr_pos', 'nbr_vel', 'nbr_valid', 'self_vel')}
        max_path = max_logit = max_pos = max_vel = max_heading = 0.
        common = {k: 0. for k in ('mode', 'relative', 'scale', 'logits')}
        for start in range(0, size, 64):
            if start % 1024 == 0:
                write(ROOT / 'CURRENT.json', dict(at=now(), phase='BUILD_CACHE', split=split, completed=start, n=size))
                print(f'CACHE {split} {start}/{size}', flush=True)
            idx = slice(start, start + 64)
            a = data['past'][idx]
            p = torch.as_tensor(a, device=device)
            counts = torch.as_tensor(data['num_valid'][idx], device=device).reshape(-1)
            assert len(counts) == len(a) and bool(((counts >= 1) & (counts <= 9)).all())
            with torch.no_grad():
                centered, _, graph, center, origin, matrix = bf.prepare(p, future=None, augment=False)
                paths = model(centered, None, None, mode='test', num_valid=counts, graph_loc=graph)[2][:, :1]
                logits = model.last_mode_logits[:, 0]
                relative = captured['scale'][:, None, None] * (model.decoder.anchor_templates[None] + captured['delta'])
                rebuilt = torch.einsum('tp,nkpc->nktc', model.decoder.interpolation, relative)
                torch.testing.assert_close(rebuilt, paths[:, 0] - captured['offset'][:, None, None], atol=2e-5, rtol=2e-5)
                world = bf.world_paths(paths, (center, origin, matrix)).cpu().numpy()
                max_path = max(max_path, float(np.max(np.abs(world - saved['paths'][order[idx]]))))
                max_logit = max(max_logit, float(np.max(np.abs(logits.cpu().numpy() - saved['logits'][order[idx]]))))
                for k, v in dict(mode=captured['mode'], relative=relative, scale=captured['scale'], logits=logits).items():
                    common[k] = max(common[k], float((v.cpu() - prior['tensors'][k][idx]).abs().max()))
                assert bool((graph[:, 0, -1] == 0).all())
                # Independent float64 rotation using the same anchor and fallback convention.
                a = a.astype(np.float64)
                disp = a[:, 0, -1] - a[:, 0, -4]
                norm = np.linalg.norm(disp, axis=-1)
                good = norm > .03
                np.testing.assert_array_equal(good, prior['meta']['heading_valid'][idx])
                basis = np.tile([1., 0.], (len(a), 1))
                basis[good] = disp[good] / norm[good, None]
                def rot(v):
                    c, s = basis[:, :1], basis[:, 1:]
                    return np.stack((v[..., 0] * c + v[..., 1] * s, -v[..., 0] * s + v[..., 1] * c), axis=-1)
                rotated_heading = rot(basis[:, None])[:, 0]
                np.testing.assert_allclose(rotated_heading, np.tile([1., 0.], (len(a), 1)), atol=1e-12, rtol=0)
                actual_heading = (graph[:, 0, -1] - graph[:, 0, -4]).cpu().numpy()
                if good.any():
                    normalized = actual_heading[good] / np.linalg.norm(actual_heading[good], axis=-1, keepdims=True)
                    max_heading = max(max_heading, float(np.abs(normalized - [1., 0.]).max()))
                pos = graph[:, 1:, -1]
                velocity = (graph[:, :, -1] - graph[:, :, -2]) * 2.5
                valid = torch.arange(1, 9, device=device)[None] < counts[:, None]
                mask = valid.cpu().numpy()
                position_error = np.abs(pos.cpu().numpy() - rot(a[:, 1:, -1] - a[:, :1, -1]))
                velocity_error = np.abs(velocity.cpu().numpy() - rot((a[:, :, -1] - a[:, :, -2]) * 2.5))
                max_pos = max(max_pos, float(position_error[mask].max(initial=0.)))
                max_vel = max(max_vel, float(velocity_error[np.concatenate((np.ones((len(a), 1), bool), mask), axis=1)].max()))
                new = dict(tokens_self=captured['tokens_self'],
                           tokens_nbr=captured['tokens_nbr'].masked_fill(~valid[..., None], 0.),
                           nbr_pos=pos.masked_fill(~valid[..., None], 0.),
                           nbr_vel=velocity[:, 1:].masked_fill(~valid[..., None], 0.),
                           nbr_valid=valid, self_vel=velocity[:, 0])
                for k, v in new.items():
                    assert bool(torch.isfinite(v).all()), k
                    pieces[k].append(v.cpu())
        assert max_path <= 1e-4 and max_logit <= 1e-4 and max(common.values()) <= 1e-4
        assert max_pos <= 1e-4 and max_vel <= 1e-4 and max_heading <= 1e-5
        # Use the identical, sealed FP32 old fields for an exact A1 protocol comparison.
        tensors = dict(prior['tensors'], **{k: torch.cat(v) for k, v in pieces.items()})
        save_torch(folder / f'{split}.pt', dict(tensors=tensors, meta=prior['meta']))
        checks.append(dict(split=split, n=size, max_saved_world_path_difference=max_path,
                           max_saved_logit_difference=max_logit, prior_fp32_field_differences=common,
                           max_position_rotation_error=max_pos, max_velocity_rotation_error=max_vel,
                           max_focal_heading_error=max_heading, focal_origin_exact_zero=True,
                           valid_neighbors=int(tensors['nbr_valid'].sum()), token_dtype='float16'))
    for hook in hooks:
        hook.remove()
    for k, v in model.state_dict().items():
        assert torch.equal(v.cpu(), base['ema'][k]), k
    write(folder / 'VALIDATION.json', dict(status='PASS', at=now(), checks=checks,
          base_checkpoint_sha256=sha(base_path), base_params=sum(p.numel() for p in model.parameters()),
          previous_fp32_cache_fields_reused=True, prior_cache_unchanged=True,
          future_passed_to_encoder=False, base_weights_unchanged=True))
    write(folder / 'DONE.json', dict(files=seal([folder / k for k in ('fit.pt', 'dev.pt', 'frozen_assets.pt', 'VALIDATION.json')])))


def load(device):
    folder = ROOT / 'cache'
    verify(read(folder / 'DONE.json')['files'])
    assets = torch.load(folder / 'frozen_assets.pt', map_location='cpu', weights_only=False)
    head = score_head().to(device)
    head.load_state_dict(assets['score_head'])
    head.eval().requires_grad_(False)
    criterion = criterion_from(assets['criterion_recipe'], device, assets['loss_buffers'])
    data = {}
    for split in ('fit', 'dev'):
        d = torch.load(folder / f'{split}.pt', map_location='cpu', weights_only=False)
        d['tensors'] = {k: v.to(device) for k, v in d['tensors'].items()}
        data[split] = d
    return data, head, criterion
