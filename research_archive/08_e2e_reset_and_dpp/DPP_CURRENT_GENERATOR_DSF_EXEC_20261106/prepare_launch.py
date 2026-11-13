import ast,shutil
from run_common import *

def main():
    pre=read(RUN/'PREFLIGHT.json')
    if pre['status']!='PASS_REAL_CURRENT_DSF_PAIRED_GRAD_RESUME_SAVED_AUDIT' or pre['pre_source_sha']!=sha(RUN/'PRE_SOURCE.json'):raise ValueError('PREFLIGHT_CONTRACT')
    for path in RUN.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    source=read(RUN/'PRE_SOURCE.json');check_files(source['files'])
    files=dict(source['files']);files.update(seal([p for p in RUN.rglob('*') if p.is_file() and (p.suffix in ('.py','.md','.ps1') or p.name in ('protocol.json','PRE_SOURCE.json','CACHE_RECEIPT.json','PREFLIGHT.json','MEMORY_ESTIMATE.json','PREP_RESOURCE.json'))]));files.update(seal((RUN/'preflight').rglob('*completed.pt')))
    disk=shutil.disk_usage(RUN).free;estimated_outputs=4*1024**3
    if disk<1.3*estimated_outputs:raise ValueError('INSUFFICIENT_DISK_NO_DELETE')
    rg=rg_module();resource=rg.Guard(read(RUN/'MEMORY_ESTIMATE.json')['GPU_E']).check(mode='startup',force=True)
    write(RUN/'PRELAUNCH_REVIEW.json',dict(status='PASS_FINITE_UNIFORM_KIN_CURRENT_DSF_PRELAUNCH',at=now(),preflight_sha=sha(RUN/'PREFLIGHT.json'),planned_runs=6,epochs=120,updates=81480,scene_joint='NOT_TRIGGERED_MAP_ADMISSION_UNRESOLVED',matched_factor='ONLY_UNIFORM_vs_KIN_QUALITY_WEIGHTS',source_files=len(files),resource=resource,disk_free=disk,estimated_outputs_bytes=estimated_outputs,planning_seconds=pre['planning_seconds'],ETA_seconds=pre['ETA_seconds'],ARS_DA1='Current h3 control is non-Gaussian with fixed slot templates, not exact paper reproduction; positive latent gradient does not prove reachability/coverage. Both arms matched, no new safety fallback or objective repair.',ARS_DA2='Prior FIT large-turn559 successful candidate scores none below.1; this is descriptive and own-definition dependent, not independent safety or no-bias proof. Missing map conditions remain untested.',ARS_DA3='Own real512subset resume and saved reviews passed, not whole training repeat or external review. Shared pretrained base and repeated dev constrain inference; parameter/cost target not passed by renaming.',no_old_source_edits_or_model_training=True))
    files.update(seal([RUN/'PRELAUNCH_REVIEW.json']))
    write(RUN/'PRODUCTION_SOURCE_LOCK.json',dict(status='LOCKED_BEFORE_SCIENTIFIC_DSF_UPDATES',at=now(),files=files))
    print('PRODUCTION_PRELAUNCH_PASS',len(files),pre['planning_seconds']/60,flush=True)

if __name__=='__main__':main()
