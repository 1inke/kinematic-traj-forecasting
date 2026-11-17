from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parents[1]
PREVIOUS = WORK / 'qoder_work/EXP_SET_INTERACT_PROBE_EXEC_20261117'
OLD = WORK / 'qoder_work/NEXT_PHASE_EXEC_20261114'
HELPER = Path('C:/Users/linke/Downloads/1/EXP_ETH_SPEED_20261116/code')
sys.path.insert(0, str(PREVIOUS / 'code'))
from probe_io import arrays, now, npz, read, save_torch, seal, sha, verify, write

ARMS = ('A1', 'A1L', 'C1', 'C2', 'C3', 'C4')
SEEDS = (0, 1, 2)
INPUTS = ('mode', 'relative', 'scale', 'logits', 'tokens_self', 'tokens_nbr',
          'nbr_pos', 'nbr_vel', 'nbr_valid', 'self_vel')
