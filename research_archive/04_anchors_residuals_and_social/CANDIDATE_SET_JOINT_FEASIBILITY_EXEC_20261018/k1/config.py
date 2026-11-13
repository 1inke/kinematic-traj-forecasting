"""K1 constants. Design-locked; no outer-driven retuning."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(os.environ.get("JOINT_FEAS_ROOT", r"<WORKSPACE_ROOT>"))
EXEC = Path(os.environ["JOINT_FEAS_EXEC"]) if os.environ.get("JOINT_FEAS_EXEC") else (
    ROOT / "CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018"
)
K1_DIR = Path(os.environ["JOINT_FEAS_K1"]) if os.environ.get("JOINT_FEAS_K1") else (EXEC / "k1_exec")
DESIGN = ROOT / "CANDIDATE_SET_JOINT_FEASIBILITY_DESIGN_20261018"

SEEDS = (42, 43, 44)
N_NAT, N_KIN, K_SELECT = 20, 20, 20
DT = 0.4
QUALITY = (0.2, 0.4)
MOTION_FACTOR = 1.01
GROUPS = ("overall", "stable", "transition", "change")
MOTION_KEYS = ("speed_p95", "accel_mean", "accel_p95", "jerk_mean", "jerk_p95")

# Training budget (02 §2)
N_TRAIN_WINDOWS = 4096
STATES = ("S0", "S4", "S12")
STATE_N_SWAPS = {"S0": 0, "S4": 4, "S12": 12}
N_ACTIONS_PER_STATE = 32
# 4096 * 3 * 32 = 393216

# HGB contract (02 §4)
HGB_PARAMS = dict(
    max_depth=6,
    learning_rate=0.08,
    max_iter=200,
    min_samples_leaf=40,
    l2_regularization=0.1,
    early_stopping=False,
    loss="squared_error",
)
OUTPUT_NAMES = ("dA", "dF", "dJ", "dC", "L")

# Deployment thresholds (02 §5) — locked, not recalibrated
PRED_L_MAX = 0.01
PRED_DA_MIN = 0.0
PRED_DF_MIN = 0.0
MAX_SWAPS = 20

# Bootstrap family16 (02 §6)
BOOT_SEED = 20261018
BOOT_DRAWS = 100_000
# Bonferroni family16: alpha=0.05/16 => two-sided tails 0.05/32 each side? 
# Design: [.0015625, .9984375] = 1 - 0.05/16 / 2? 
# 0.05/16 = 0.003125; half = 0.0015625. Yes.
BOOT_LO = 0.0015625
BOOT_HI = 0.9984375

FEATURE_DIM = 113
