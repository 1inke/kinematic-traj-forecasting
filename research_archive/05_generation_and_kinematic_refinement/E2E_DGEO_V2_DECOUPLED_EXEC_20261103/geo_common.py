"""New DGeo execution namespace; no old outputs or controls writable."""
from pathlib import Path
import os,sys
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
BASE=ROOT/'qoder_work/E2E_MOTION_RESEARCH_RESET_EXEC_20261102'
UD=ROOT/'qoder_work/E2E_UD_AUTHORIZED_EXEC_20261102'
DESIGN=Path('<USER_HOME>/.gemini/antigravity/brain/43533de8-68f1-4ef3-9b7c-526299479022/NEXT_STAGE_DGEO_V2_EXPERIMENT_DESIGN.md')
V1=ROOT/'qoder_work/E2E_DGEO_PILOT_EXEC_20261102'
ANALYSIS=V1/'RESULT_ANALYSIS_REVIEWED.md'
sys.dont_write_bytecode=True
os.environ['TORCH_HOME']=str(RUN/'runtime_cache/torch')
os.environ['MPLCONFIGDIR']=str(RUN/'runtime_cache/matplotlib')
sys.path.insert(0,str(BASE))
sys.path.insert(1,str(UD))
from common import sha,seal,verify,read_json,write_json,write_csv,now
ARMS=('DK-CenterDiff','DK-ManeuverDiff','DK-KinTaylorDiff','DK-FullDiff')

