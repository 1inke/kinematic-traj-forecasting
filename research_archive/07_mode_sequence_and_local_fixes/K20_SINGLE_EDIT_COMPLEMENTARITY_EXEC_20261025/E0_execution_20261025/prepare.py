"""Freeze new execution with all prior G0 source digests retained, no overwriting old locks."""
from common import *
from data_contract import CACHE,LABEL,ENC,SPLIT,OLD,kinpath
def main():
    if (HERE/'source_manifest.json').exists():raise FileExistsError('Current source lock already exists')
    prior=read_json(ROOT/'qoder_work/DECODER_SUPPORT_PRESERVING_UPDATE_EXEC_20261025/source_manifest.json')
    files={}
    for k,v in prior['files'].items():
        if sha256(ROOT/k)!=v['sha256']:raise ValueError('Inherited source '+k)
        files[k]=dict(path=str(ROOT/k),sha256=v['sha256'],inherited=True)
    add=[ROOT/CACHE,ROOT/LABEL,ROOT/ENC,ROOT/SPLIT,ROOT/OLD,ROOT/'GENERATION_NATIVE_DECODER_SERIES/encoder_audit.json',ROOT/'GENERATION_NATIVE_DECODER_SERIES/run_native.py',ROOT/'UNION_SELECTOR_K40_TO_K20_20261018/kdec/data.py',ROOT/'UNION_SELECTOR_K40_TO_K20_20261018/kdec/eval_selector.py',ROOT/'src/f16/eval.py',ROOT/'src/f16/resource.py',ROOT/'src/f16/io.py']
    for seed in SEEDS:add.extend([ROOT/kinpath(seed),ROOT/f'KINEMATIC_DECODER_SPEED_CLAMP_20261018/results/short/KIN_seed{seed}.pt'])
    add.extend((ROOT/'docs/K20_SINGLE_EDIT_COMPLEMENTARITY_DESIGN_20261025').glob('*.md'))
    add.extend([HERE.parent/'EXECUTION_AMENDMENT_20261025.md',HERE.parent/'PROTOCOL_MOTION_AUDIT.json',HERE.parent/'PROTOCOL_COORDINATION_PROPOSAL.md'])
    add.extend(p for p in HERE.iterdir() if p.suffix in ('.py','.ps1','.md') or p.name=='protocol.json')
    for path in sorted(set(add)):
        key=path.relative_to(ROOT).as_posix();ss=sha256(path)
        if key in files and files[key]['sha256']!=ss:raise ValueError('Inherited digest overwrite forbidden')
        files.setdefault(key,dict(path=str(path),sha256=ss,inherited=False))
    out=dict(files=files,digest=digest({k:v['sha256'] for k,v in files.items()}),inherited_count=len(prior['files']),created_at=now())
    write_json(HERE/'source_manifest.json',out);print(json.dumps(dict(count=len(files),inherited=len(prior['files']),digest=out['digest'])))
if __name__=='__main__':main()
