"""Infrastructure-only recovery: make old helper's explicit sibling import visible.

No scientific function, selected ID, core hash, numeric tolerance or optimizer
changes. The failed locked driver stays untouched. Failure happened before the
helper returned, thus before any tiny model/optimizer/run was constructed.
"""
import importlib.util
from pathlib import Path
import sys
import traceback
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'ENDPOINT_TEMPORAL_OBSERVABILITY_EXEC_20261029'
sys.path.insert(0,str(ROOT/'code'))
from io_contract import verify_manifest,json_new,sha


if __name__=='__main__':
    verify_manifest(ROOT/'namespace_adapter_source_extension.json')
    if (ROOT/'tiny_v2/initial_checkpoint.pt').exists():
        raise RuntimeError('A tiny model already exists: do not rerun; exact recovery needs review')
    sys.path.append(str(OLD))
    spec=importlib.util.spec_from_file_location('locked_v2_kernel',ROOT/'execute_tiny_v2.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    try:
        module.run_tiny()
        verify_manifest(ROOT/'namespace_adapter_source_extension.json')
        json_new(ROOT/'NAMESPACE_ADAPTER_RESULT.json',dict(status='SUCCESS_SOURCE_UNCHANGED',
            adapter_sha=sha(__file__),locked_kernel_sha=sha(ROOT/'execute_tiny_v2.py'),
            selection_sha=sha(ROOT/'tiny_v2/SELECTION.json'),no_math_change=True))
    except Exception:
        json_new(ROOT/'NAMESPACE_ADAPTER_FAILURE.json',dict(status='FAILURE',traceback=traceback.format_exc()))
        raise
