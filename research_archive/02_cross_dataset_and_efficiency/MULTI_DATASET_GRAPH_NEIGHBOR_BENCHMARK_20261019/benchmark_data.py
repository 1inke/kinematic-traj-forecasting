"""Benchmark Data Loader for Multi-Dataset Evaluation.
Selects representative multi-agent evaluation samples across all 5 ETH/UCY datasets:
ETH, HOTEL, UNIV, ZARA01, ZARA02.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

WORKSPACE = Path(r"<WORKSPACE_ROOT>")
PROJECT = Path(r"<LOCAL_DESKTOP>/DRSE/ASTRA_ACCURACY_Q20")
RAW = PROJECT / "datasets/eth_ucy/processed_data_diverse"
LABELS_ROOT = PROJECT / "datasets/eth_ucy/labels"

DATASETS = ["eth", "hotel", "univ", "zara01", "zara02"]

def get_multi_dataset_cohort(samples_per_dataset: int = 20, min_agents: int = 3) -> dict[str, list[dict]]:
    """Select deterministic multi-agent samples from each dataset."""
    cohort = {}
    for ds in DATASETS:
        test_d = np.load(RAW / f"{ds}_data_test.npy", mmap_mode="r")
        test_n = np.load(RAW / f"{ds}_num_test.npy")
        
        selected = []
        for idx in range(len(test_n)):
            nv = int(test_n[idx])
            if nv >= min_agents:
                seq = np.array(test_d[idx, :9, :8, -2:], dtype=np.float32)
                focal_disp = np.linalg.norm(seq[0, -1] - seq[0, 0])
                if focal_disp > 0.3:
                    selected.append({
                        "dataset": ds,
                        "sample_idx": idx,
                        "num_valid": nv,
                        "focal_disp": float(focal_disp),
                    })
            if len(selected) >= samples_per_dataset:
                break
        
        cohort[ds] = selected
        print(f"Dataset [{ds}]: selected {len(selected)} samples (nv >= {min_agents}, focal_disp > 0.3m)")
    
    return cohort

if __name__ == "__main__":
    c = get_multi_dataset_cohort(20, 3)
    total_samples = sum(len(v) for v in c.values())
    print(f"Total benchmark cohort size: {total_samples} samples across 5 datasets.")
    out_file = Path(__file__).resolve().parent / "COHORT_100.json"
    out_file.write_text(json.dumps(c, indent=2), encoding="utf-8")
    print(f"Saved cohort to {out_file}")
