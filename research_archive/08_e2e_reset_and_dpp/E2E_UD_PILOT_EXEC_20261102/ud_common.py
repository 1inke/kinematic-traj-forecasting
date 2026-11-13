"""UD namespace. Historical B0 sources are imported read-only, never patched."""
from pathlib import Path
import os
import sys

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
BASE=ROOT/'qoder_work/E2E_MOTION_RESEARCH_RESET_EXEC_20261102'
os.environ['TORCH_HOME']=str(RUN/'runtime_cache/torch')
os.environ['MPLCONFIGDIR']=str(RUN/'runtime_cache/matplotlib')
sys.dont_write_bytecode=True
sys.path.insert(0,str(BASE))
from common import sha, seal, verify, read_json, write_json, now

DESIGN=Path('<LOCAL_DESKTOP>/新调研任务/下一阶段U与D单因素实验设计文档_20261102.md')
