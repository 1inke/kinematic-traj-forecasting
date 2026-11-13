"""Reuse sealed C0 arithmetic and complete independent save review, bind new source."""
from geo_common import *
import numpy as np
import evaluate_saved as base_eval


def save_population(folder,paths,logits,data,guard=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(folder/'predictions.npz',paths=paths,logits=logits,ids=data['ids'],
        future=data['future'],past=data['past'][:,0],groups=data['group'])
    with np.load(folder/'predictions.npz') as saved:
        if not np.array_equal(saved['ids'],data['ids']) or not np.array_equal(saved['paths'],paths):raise ValueError('SAVE_READBACK_NOT_EXACT')
        rows,metrics,audit=base_eval.evaluate(saved['paths'],saved['logits'],data,guard)
    for row in rows.values():
        row['LEGACY_SHADOW']={'old_native_comparator':'NOT_COMPARABLE_ON_NEW_SOURCE_IDS',
            'matched_reference':'FRESH_B0_SEED42','old_major':'NOT_CLAIMED'}
    np.savez_compressed(folder/'window_metrics.npz',**metrics)
    write_json(folder/'metrics.json',rows);write_json(folder/'audit.json',audit)
    write_json(folder/'done.json',{'status':'SUCCESS','files':seal(folder.iterdir()),
        'source':sha(RUN/'source_manifest.json') if (RUN/'source_manifest.json').exists() else 'PREFLIGHT_NOT_PRODUCTION_SOURCE_LOCKED'})
    return rows

