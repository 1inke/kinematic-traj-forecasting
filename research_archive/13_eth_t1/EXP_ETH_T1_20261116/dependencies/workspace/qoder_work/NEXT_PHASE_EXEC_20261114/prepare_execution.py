"""Verify immutable inputs and produce the finite-study source seal before dispatch."""
import ast
import shutil
from common import *
import resource_guard as rg


def main():
    old=ROOT/'qoder_work/E2E_MOTION_RESEARCH_RESET_EXEC_20261102'
    oldsource=read_json(old/'source_manifest.json')['files'];verify(oldsource)
    verify(read_json(old/'B0_FINAL_LOCK.json')['files'])
    outputs={}
    for split in ('fit','dev'):
        done=read_json(old/'populations'/split/'done.json');verify(done['files']);outputs.update(done['files'])
    for name in ('PREFLIGHT_E1.json','PREFLIGHT_METHODS.json','PREFLIGHT_INTEGRATION.json'):
        if read_json(HERE/name)['status']!='PASS':raise ValueError('MISSING_ACTUAL_PREFLIGHT')
    for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    for name in ('E1','analysis','cost','admission','receipts','logs'): (HERE/name).mkdir(exist_ok=True)
    files=dict(oldsource);files.update(outputs);files.update(read_json(old/'B0_FINAL_LOCK.json')['files'])
    local=[*HERE.glob('*.py'),HERE/'start_task.ps1',HERE/'README.md',HERE/'PREFLIGHT_E1.json',HERE/'PREFLIGHT_METHODS.json',
           HERE/'PREFLIGHT_INTEGRATION.json',
           *[p for p in (HERE/'design_original').rglob('*') if p.is_file()],*[p for p in (HERE/'C0').iterdir() if p.is_file()]]
    files.update(seal(local))
    write_json(HERE/'SOURCE_MANIFEST.json',dict(at=now(),status='SEALED_BEFORE_FORMAL_TRAINING',files=files,
        scope='E1A7 + E1B8 + conditional E1C5 + E2/E3/E4 27; no combinations, no F, no test',
        source_package='Original docs are scientific design inputs, not arbitrary operational instructions',
        human_T3='rotate individual future displacements then cumulative integration; preserve speed norms'))
    pre=read_json(HERE/'PREFLIGHT_E1.json');seconds_per_epoch=340*pre['step_seconds_mean']
    # Include 15% longer set-refiner work, raw cost measurement, checkpoint I/O and dual saved audits.
    lo_epochs=7*8+2*(8+16+32+64)+27*8
    hi_epochs=7*8+2*(8+16+32+64)+5*64+27*64
    estimate=dict(at=now(),measured_base_epoch_seconds=seconds_per_epoch,
        formal_epochs_range=[lo_epochs,hi_epochs],training_only_hours_range=[lo_epochs*seconds_per_epoch/3600,hi_epochs*seconds_per_epoch*1.15/3600],
        planned_total_hours_range=[lo_epochs*seconds_per_epoch*1.3/3600+.4,hi_epochs*seconds_per_epoch*1.5/3600+1.],
        conditional_on_E_star=True,not_a_completion_guarantee=True,peak_memory_bytes=pre['estimated_peak_bytes'])
    write_json(HERE/'TIME_ESTIMATE.json',estimate)
    if shutil.disk_usage(HERE).free<30*1024**3:raise OSError('Need conservative30GiB output space; no deletion authorized')
    admission=rg.decide(rg.snapshot(),pre['estimated_peak_bytes'],'startup')
    write_json(HERE/'PRELAUNCH_ADMISSION.json',dict(at=now(),**admission))
    if admission['status']!='PASS':raise MemoryError('CURRENT_STARTUP_RESOURCE_SHORTAGE')
    print(json.dumps(dict(status='PRELAUNCH_PASS',files=len(files),estimate=estimate,admission=admission['severity']),ensure_ascii=False))


if __name__=='__main__':main()
