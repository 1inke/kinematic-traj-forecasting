"""Authoring utility: resolve existing artifacts. Does not start experiments."""
from pathlib import Path
import hashlib
import json

package = Path(__file__).resolve().parents[1]
workspace = package.parents[1]
five = workspace/'qoder_work/FIVEFOLD_B0_EXEC_20261115'
old = workspace/'qoder_work/EXP_ETH_T1_EXEC_20261116'


def sha(file):
    h = hashlib.sha256()
    with Path(file).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


c = dict(study='EXP_ETH_SPEED_20261116', workspace=str(workspace),
         python=str(workspace/'.venv-governance/Scripts/python.exe'),
         exec_dir=str(workspace/'qoder_work/NEXT_PHASE_EXEC_20261114'),
         fivefold_code=str(five/'code'),
         data_dir='C:/Users/linke/Desktop/DRSE/ASTRA/datasets/eth_ucy/processed_data_diverse',
         astra_dir='C:/Users/linke/Desktop/DRSE/ASTRA_ACCURACY_Q20',
         video_dir='C:/Users/linke/Desktop/DRSE/ASTRA/datasets/eth_ucy/videos',
         output_dir=str(workspace/'qoder_work/EXP_ETH_SPEED_EXEC_20261116'), device='cuda:0',
         upstream_manifest=str(old/'SOURCE_MANIFEST.json'),
         upstream_manifest_sha256=sha(old/'SOURCE_MANIFEST.json'),
         recording_audit=str(five/'F0/AUDIT_RECORDING.json'),
         recording_audit_sha256=sha(five/'F0/AUDIT_RECORDING.json'),
         baselines={}, optional_saved_files={})
for fold in ('eth', 'hotel'):
    directory = five/f'FIVEFOLD_RUNS/runs/budget_{fold}_E64_s42'
    checkpoint = directory/'ckpt_epoch064.pt'; receipt = directory/'TRAIN_COMPLETE.json'
    c['baselines'][fold] = dict(checkpoint=str(checkpoint), sha256=sha(checkpoint),
                              receipt=str(receipt), receipt_sha256=sha(receipt))
for arm in ('B0', 'T1'):
    file = old/f'runs/{arm}_eth_E64_s42/test_predictions.npz'
    c[f'eth_saved_{arm}'] = str(file); c['optional_saved_files'][str(file)] = sha(file)
(package/'CONFIG.local.json').write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding='utf-8')
for name, args in [('START_DIAG', '-Phase diag'), ('START_AUG', '-Phase aug'), ('PREPARE', '-Phase prepare')]:
    (package/f'standalone/{name}.cmd').write_text('@echo off\npowershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0START.ps1" '+args+'\npause\n', encoding='ascii')
for name in ('STATUS', 'PAUSE'):
    (package/f'standalone/{name}.cmd').write_text('@echo off\npowershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0'+name+'.ps1"\npause\n', encoding='ascii')
print('Local config and manual launchers created; no scheduled task registered or launched.')
