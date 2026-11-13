"""Validate the prepared code package only. Never imports experimental stages."""
from common import *
import ast
def main():
    control=read(RUN/'execution_control.json')
    if control['allow_experiment_start'] or control['active_worker']:raise ValueError('NOT_HELD')
    forbidden=['CALIBRATION.json','source_manifest.json','INPUT_MANIFEST.json','MAP_ADMISSION.json','PREFLIGHT_REAL.json','STAGE_A_COMPUTED.json','RESULT.json','MANUAL_PIPELINE_STARTED.json','PREPARATION_STARTED.json','stage_a.lock']
    if any((RUN/n).exists() for n in forbidden):raise ValueError('REAL_EXPERIMENT_EVIDENCE_UNEXPECTED')
    files=sorted(RUN.glob('*.py'))
    for p in files:ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    for p in RUN.glob('*.json'):read(p)
    test=sorted((RUN/'synthetic_tests').glob('*.json'))[-1];receipt=read(test)
    if receipt['status']!='PASS_SYNTHETIC_ONLY' or receipt['tests']!=15:raise ValueError('SYNTHETIC_TESTS_NOT_PASS')
    verify(receipt['source_files'])
    if set(map(str,files))!=set(receipt['source_files']):raise ValueError('UNTESTED_NEW_SCRIPT')
    package=files+list(RUN.glob('*.ps1'))+list(RUN.glob('*.md'))+[RUN/'execution_control.json',RUN/'protocol.draft.json',RUN/'map_registry.template.json',test]
    write(RUN/'CODE_VALIDATION.json',dict(status='PASS_SCRIPTS_AND_SYNTHETIC_ONLY_NOT_EXPERIMENT_PREFLIGHT',at=now(),python_AST_files=len(files),synthetic_tests=15,synthetic_receipt=str(test),
        files=seal(package),real_data_loaded=False,real_preflight_run=False,formal_training_started=False,background_task_created=False,allow_experiment_start=False,
        stageA_scripts_implemented=True,stageB_trainer_implemented=False,stageB_admission_checks_implemented=True,no_production_source_seal_claim=True))
    print('CODE_PACKAGE_PASS_NOT_STARTED',len(files),'Python scripts;15 synthetic tests')
if __name__=='__main__':main()

