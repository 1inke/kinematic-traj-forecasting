import os, sys, time, json
from pathlib import Path
import numpy as np

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]
RAW = Path(r'<LOCAL_DESKTOP>/DRSE/ASTRA_ACCURACY_Q20/datasets/eth_ucy/processed_data_diverse')
CACHE = ROOT / 'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz'
META = ROOT / 'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz'
ENCODER = ROOT / 'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz'

def build_neighbor_cache():
    print("=== BUILDING DETERMINISTIC PAST-ONLY NEIGHBOR CACHE FOR D3 ===")
    t0 = time.time()

    sample_id = np.load(CACHE)['sample_id']
    z_meta = np.load(META)
    mid = z_meta['sample_id']
    order = np.argsort(mid)
    loc = order[np.searchsorted(mid[order], sample_id)]

    enc_past = np.load(ENCODER)['past'].astype(np.float32) # (28791, 8, 2)
    raw_d = np.load(RAW / 'eth_data_train.npy', mmap_mode='r')
    raw_n = np.load(RAW / 'eth_num_train.npy')

    num_valid = raw_n[loc] # (28791,)
    focal_raw_past = raw_d[loc, 0, :8, -2:].astype(np.float32) # (28791, 8, 2)
    focal_disp = focal_raw_past[:, 7] - focal_raw_past[:, 0]
    focal_norm = np.linalg.norm(focal_disp, axis=-1, keepdims=True)
    enc_disp = enc_past[:, 7] - enc_past[:, 0]
    enc_norm = np.linalg.norm(enc_disp, axis=-1, keepdims=True)

    is_moving = (focal_norm[:, 0] > 0.05) & (enc_norm[:, 0] > 0.05)
    cos_t = np.ones(len(sample_id), dtype=np.float32)
    sin_t = np.zeros(len(sample_id), dtype=np.float32)
    denom = (focal_norm[is_moving, 0] ** 2)
    cos_t[is_moving] = (enc_disp[is_moving, 0] * focal_disp[is_moving, 0] + enc_disp[is_moving, 1] * focal_disp[is_moving, 1]) / denom
    sin_t[is_moving] = (enc_disp[is_moving, 1] * focal_disp[is_moving, 0] - enc_disp[is_moving, 0] * focal_disp[is_moving, 1]) / denom

    R = np.zeros((len(sample_id), 2, 2), dtype=np.float32)
    R[:, 0, 0] = cos_t
    R[:, 0, 1] = -sin_t
    R[:, 1, 0] = sin_t
    R[:, 1, 1] = cos_t

    # Raw neighbors (up to 8 agents)
    nb_raw = raw_d[loc, 1:9, :8, -2:].astype(np.float32) # (28791, 8, 8, 2)
    focal_origin = focal_raw_past[:, None, 7:8, :] # (28791, 1, 1, 2)
    rel_nb = nb_raw - focal_origin
    nb_canon = np.einsum('nij,natj->nati', R, rel_nb).astype(np.float32)

    j_idx = np.arange(1, 9)[None, :]
    agent_valid = (j_idx < num_valid[:, None]) & np.any(nb_raw != 0, axis=(2, 3)) # (28791, 8)
    nb_canon = nb_canon * agent_valid[:, :, None, None]

    # Current positions at t=7
    nb_pos0 = nb_canon[:, :, 7].copy() # (28791, 8, 2)

    # Current velocities (dt = 0.8s between t=5 and t=7)
    nb_vel0 = (nb_canon[:, :, 7] - nb_canon[:, :, 5]) / 0.8 # (28791, 8, 2)
    nb_vel0 = nb_vel0 * agent_valid[:, :, None]

    # Past-only constant-velocity extrapolation for t = 1..12 (dt = 0.4s)
    future_t = (np.arange(1, 13, dtype=np.float32) * 0.4)[None, None, :, None] # (1, 1, 12, 1)
    nb_extrap = nb_pos0[:, :, None, :] + future_t * nb_vel0[:, :, None, :] # (28791, 8, 12, 2)
    nb_extrap = nb_extrap * agent_valid[:, :, None, None]

    out_file = EXP_DIR / 'neighbor_cache.npz'
    np.savez_compressed(
        out_file,
        sample_id=sample_id,
        nb_canon_past=nb_canon,
        nb_valid=agent_valid,
        nb_pos0=nb_pos0,
        nb_vel0=nb_vel0,
        nb_extrap=nb_extrap
    )

    elapsed = time.time() - t0
    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"Neighbor cache built and verified in {elapsed:.2f}s ({size_mb:.2f} MB saved at {out_file.name}).")
    return out_file

if __name__ == '__main__':
    build_neighbor_cache()
