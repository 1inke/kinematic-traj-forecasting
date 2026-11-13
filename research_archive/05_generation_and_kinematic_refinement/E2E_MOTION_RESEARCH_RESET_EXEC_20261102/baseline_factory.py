"""Fresh existing F16 encoder plus existing h3 anchor proposal; no weight loads."""
import os
import sys
from types import SimpleNamespace
import math
import random
import numpy as np
from common import HERE, ROOT, ASTRA

os.environ.setdefault('TORCH_HOME', str(HERE / 'runtime_cache/torch'))
os.environ.setdefault('MPLCONFIGDIR', str(HERE / 'runtime_cache/matplotlib'))
sys.dont_write_bytecode = True
for p in (ASTRA, ROOT / 'FRONT_ENCODER_SERIES', ROOT / 'HIERARCHICAL_DECODER_SERIES'):
    sys.path.insert(0, str(p))
import torch
from experiment_config import load_locked_q20
from front_encoder_models import FrontEncoderASTRA
from hierarchical_decoder_models import replace_decoder
from locked_mode_query_loss import ModeQueryLoss
from train_front_variant import augment_trajectories, normalize_to_focal_heading, ModelEMA

CONFIG = ROOT / 'OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/motif_rbf_official_4p8.yaml'
AUGMENT = SimpleNamespace(augment_rotation=False, augment_reflection=True,
    augment_speed_min=.9, augment_speed_max=1.1, augment_probability=.5,
    augment_curvature_max=0.)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_model(device='cuda', seed=42):
    seed_all(seed)
    cfg = load_locked_q20(CONFIG)
    cfg.SUBSET = 'eth'; cfg.device = torch.device(device); cfg.device_list = [0]
    cfg.DATA.MAX_AGENTS = 9; cfg.TRAIN.BATCH_SIZE = 64; cfg.TRAIN.LR = .00025
    # Avoid loading an unrelated fitted PCA asset into a constructor module that
    # h3 immediately discards. This temporary linear decoder never runs.
    cfg.MODEL.DECODER.VARIANT = 'linear'
    model = FrontEncoderASTRA(cfg, 'f16_temporal_social_cross')
    replace_decoder(model, cfg, 'h3_anchor_proposal')
    model.decoder_variant = 'query_endpoint_pca'
    cfg.MODEL.DECODER.VARIANT = 'query_endpoint_pca'
    model.to(device)
    return model, cfg


def prepare(past, future=None, augment=False):
    # Future is transformed for the loss only, and never passed to model.forward.
    f = (past.new_zeros((len(past), 9, 12, 2)) if future is None
         else future[:, None].expand(-1, 9, -1, -1).clone())
    if augment:
        past, f = augment_trajectories(past, f, AUGMENT)
    origin = past[:, 0, -1].clone()
    disp = past[:, 0, -1] - past[:, 0, -4]
    norm = torch.linalg.vector_norm(disp, dim=-1)
    heading = torch.where((norm > .03)[:, None], disp / norm[:, None].clamp_min(1e-6),
                          disp.new_tensor([1., 0.]).expand_as(disp))
    matrix = torch.stack((heading[:, 0], -heading[:, 1], heading[:, 1], heading[:, 0]), -1).reshape(-1, 2, 2)
    p, f = normalize_to_focal_heading(past, f)
    center = p.mean(-2, keepdim=True)
    return p - center, (f - center)[:, :1], p, center, origin, matrix


def prediction(model, past, num_valid, future=None, augment=False):
    p, y, graph, center, origin, matrix = prepare(past, future, augment)
    output = model(p, None, None, mode='train' if model.training else 'test',
                   num_valid=num_valid, graph_loc=graph)[2][:, :1]
    logits = model.last_mode_logits[:, :1]
    return output, logits, y, (center, origin, matrix)


def world_paths(pred, transform):
    center, origin, matrix = transform
    local = pred[:, 0] + center[:, 0, None]
    return torch.matmul(local.unsqueeze(-2), matrix[:, None, None].transpose(-1, -2)).squeeze(-2) + origin[:, None, None]


def loss_value(pred, target, logits, criterion):
    base, detail = criterion(pred, target, logits)
    distances = torch.linalg.vector_norm(pred - target.unsqueeze(-3), dim=-1)
    quality = distances.mean(-1) + .25 * distances[..., -1]
    responsibility = torch.softmax(-quality.detach() / .10, -1)
    coverage = (responsibility * quality).sum(-1).mean()
    return base + .20 * coverage, detail


def optimizer_scheduler(model, steps_per_epoch, epochs=8):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=.00025, weight_decay=.0005, betas=(.9, .999), eps=1e-8, foreach=False)
    total, warm = epochs * steps_per_epoch, steps_per_epoch
    def schedule(step):
        if step < warm:
            return (step + 1) / warm
        return .5 * (1 + math.cos(math.pi * min((step - warm) / max(total - warm, 1), 1)))
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def rng_state(generator):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
        cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(), loader=generator.get_state())


def restore_rng(state, generator):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
    generator.set_state(state['loader'].cpu())
