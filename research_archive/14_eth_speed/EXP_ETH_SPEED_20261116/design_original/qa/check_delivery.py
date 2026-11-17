"""Read-only dependency check in a temporary output root; no formal audit/infer."""
from pathlib import Path
import ast
import re
import sys
import tempfile

package = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(package/'code'))
from speed_io import config, frozen, now, read, sha, write

c = config()
for file in package.rglob('*.py'):
    ast.parse(file.read_text(encoding='utf-8-sig'))
for file in package.glob('*.md'):
    for target in re.findall(r'\]\(([^)]+)\)', file.read_text(encoding='utf-8')):
        if '://' not in target and not (file.parent/target).is_file():
            raise ValueError('BROKEN_DOCUMENT_LINK: '+target)
with tempfile.TemporaryDirectory(prefix='eth_speed_delivery_') as temp:
    digest = frozen(c, temp)
    contract = read(Path(temp)/'INPUT_CONTRACT.json')
    # Repeat to exercise the resume contract comparison, still without a model.
    assert frozen(c, temp) == digest
    count = len(contract['files'])
    upstream_count = len(read(c['upstream_manifest'])['files'])
write(package/'qa/DEPENDENCIES_CHECK.json', dict(status='PASS', at=now(),
      old_sealed_files_verified=upstream_count, additional_contract_files_checked=count,
      formal_output_directory_created=False, formal_model_inference=False,
      real_dataset_training=False, GPU_preflight=False,
      new_python_AST='PASS', document_links='PASS', temporary_contract_resume='PASS'))
print(f'PASS: {upstream_count} upstream sealed files; {count} contract entries; no formal experiment.')
