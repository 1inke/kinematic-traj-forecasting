from pathlib import Path
path = Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\visual_review_overnight\build_overnight_visual.py")
text = path.read_text(encoding="utf-8")

# 1) Insert RAW path + local_coordinates after SEED/SPLIT
needle = 'SEED = 42\nSPLIT = "outer"\n'
insert = '''SEED = 42
SPLIT = "outer"
RAW_DIR = Path(r"<LOCAL_DESKTOP>/DRSE/ASTRA_ACCURACY_Q20/datasets/eth_ucy/processed_data_diverse")

def local_coordinates(coords: np.ndarray) -> np.ndarray:
    """Ego-centric frame at last observed step; same as CANDIDATE_SELECTION_VISUAL_AUDIT_20261003."""
    focal = coords[:, 0, :8]
    displacement = focal[:, -1] - focal[:, -4]
    speed = np.linalg.norm(displacement, axis=-1)
    heading = np.zeros_like(displacement)
    heading[:, 0] = 1
    active = speed > 0.03
    heading[active] = displacement[active] / speed[active, None]
    relative = coords - focal[:, -1, None, None]
    cosine, sine = heading[:, 0, None, None], heading[:, 1, None, None]
    return np.stack(
        (relative[..., 0] * cosine + relative[..., 1] * sine,
         -relative[..., 0] * sine + relative[..., 1] * cosine),
        -1,
    ).astype(np.float32)

'''
if 'def local_coordinates' not in text:
    assert needle in text
    text = text.replace(needle, insert, 1)

# 2) After past/future loaded and past_chk, add neighbor extraction
# Find the block after past_chk assert and before deploy
marker = 'assert past_chk < 5e-5, past_chk\n'
nb_block = '''assert past_chk < 5e-5, past_chk

    # Neighbor past trajectories in the same ego-centric frame as past/gt
    raw_data = RAW_DIR / "eth_data_train.npy"
    raw_num = RAW_DIR / "eth_num_train.npy"
    raw = np.load(raw_data, mmap_mode="r")
    num = np.load(raw_num, mmap_mode="r")
    assert int(sid.min()) >= 0 and int(sid.max()) < len(raw)
    coordinates = np.asarray(raw[sid.astype(np.int64), :9, :20, -2:], dtype=np.float32)
    local = local_coordinates(coordinates)
    past_from_raw = local[:, 0, :8]
    raw_past_chk = float(np.abs(past_from_raw - past).max())
    assert raw_past_chk < 5e-5, raw_past_chk
    valid = np.minimum(np.asarray(num[sid.astype(np.int64)]).reshape(n), 9).astype(np.int64)
    assert valid.min() >= 1 and valid.max() <= 9
    neighbors = [
        np.round(local[i, 1:int(valid[i]), :8].astype(float), 5).tolist()
        for i in range(n)
    ]
'''
if 'neighbors = [' not in text or 'Neighbor past trajectories' not in text:
    # remove any previous failed inject
    if 'Neighbor past trajectories' in text:
        pass
    else:
        assert marker in text
        text = text.replace(marker, nb_block, 1)

# 3) Replace empty neighbors assignment
old_nb = 'neighbors=[[] for _ in range(n)],'
new_nb = 'neighbors=neighbors,'
assert old_nb in text, 'expected empty neighbors placeholder'
text = text.replace(old_nb, new_nb, 1)

# 4) Include neighbor arrays in npz and raw paths in sources/audit
old_savez = '''    np.savez_compressed(
        HERE / "data" / "arrays.npz",
        sample_id=sid, past=past, future=future, logits=train_logits[trows],
        change_group=change_group, **precise,
    )'''
new_savez = '''    np.savez_compressed(
        HERE / "data" / "arrays.npz",
        sample_id=sid, past=past, future=future, logits=train_logits[trows],
        change_group=change_group, neighbor_past=local[:, 1:, :8], num_valid=valid, **precise,
    )'''
if old_savez in text:
    text = text.replace(old_savez, new_savez, 1)

# add raw to sources list if present
if 'raw_data, raw_num' not in text:
    text = text.replace(
        'sources = [train_cache, labels_path, encoder_path, kin_path, bundle_path,',
        'sources = [train_cache, labels_path, encoder_path, kin_path, bundle_path, raw_data, raw_num,',
        1,
    )

# audit field for neighbor stats
if 'neighbor_nonempty' not in text:
    text = text.replace(
        'past_alignment_max_error=past_chk, native_ade_recompute_max_error=fut_chk_ade,',
        'past_alignment_max_error=past_chk, raw_past_alignment_max_error=raw_past_chk,\n'
        '        native_ade_recompute_max_error=fut_chk_ade,\n'
        '        neighbor_nonempty=int(sum(1 for x in neighbors if x)),\n'
        '        neighbor_mean_count=float(np.mean([len(x) for x in neighbors])),',
        1,
    )

path.write_text(text, encoding='utf-8')
print('patched', path)
print('has local_coordinates', 'def local_coordinates' in text)
print('has empty placeholder', 'neighbors=[[]' in text)
print('has neighbors=neighbors', 'neighbors=neighbors,' in text)
