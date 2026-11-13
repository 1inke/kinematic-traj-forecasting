"""New DGeo execution namespace; no old outputs or controls writable."""
from pathlib import Path
import os,sys
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
BASE=ROOT/'qoder_work/E2E_MOTION_RESEARCH_RESET_EXEC_20261102'
UD=ROOT/'qoder_work/E2E_UD_AUTHORIZED_EXEC_20261102'
DESIGN=Path('<LOCAL_DESKTOP>/新调研任务/下一阶段D_Geo几何感知模式协同实验设计文档_20261102.md')
ANALYSIS=Path('<LOCAL_DESKTOP>/新调研任务/E2E_UD单因素实验结果深度分析报告_20261102.md')
sys.dont_write_bytecode=True
os.environ['TORCH_HOME']=str(RUN/'runtime_cache/torch')
os.environ['MPLCONFIGDIR']=str(RUN/'runtime_cache/matplotlib')
sys.path.insert(0,str(BASE))
sys.path.insert(1,str(UD))
from common import sha,seal,verify,read_json,write_json,write_csv,now
ARMS=('DG-Base','DG-GeoGate','DG-GeoDiff','DG-GeoScore')
