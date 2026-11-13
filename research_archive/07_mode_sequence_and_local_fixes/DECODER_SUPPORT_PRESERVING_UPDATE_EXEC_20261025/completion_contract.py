"""Explicit saved-analysis contract; complete output SHA skip is not training resume."""
from pathlib import Path
import hashlib,json

def validate(done_path,expected_source):
    path=Path(done_path);b=json.loads(path.read_text(encoding='utf-8-sig'))
    if b['source_digest']!=expected_source:raise ValueError('ANALYSIS_SOURCE_MISMATCH')
    for name,expected in b['output_SHA'].items():
        h=hashlib.sha256()
        with (path.parent/name).open('rb') as f:
            for block in iter(lambda:f.read(2**20),b''):h.update(block)
        if h.hexdigest()!=expected:raise ValueError('ANALYSIS_OUTPUT_SHA_MISMATCH')
    return dict(status='PASS_SOURCE_COMPATIBLE_COMPLETE_SHA_ANALYSIS_SKIP',new_training=False)
