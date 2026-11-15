"""Validate publication scope and seal its payload with portable SHA256 paths."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = {
    'private key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'GitHub token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b'),
    'OpenAI token': re.compile(r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b'),
    'AWS access key': re.compile(r'\bAKIA[A-Z0-9]{16}\b'),
    'embedded credential': re.compile(r'''(?i)(?:password|passwd|api_key|access_token|smtp_password)\s*[:=]\s*["']([^"'\s]{8,})["']'''),
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    provenance = json.loads((PACKAGE / 'COPY_PROVENANCE.json').read_text(encoding='utf-8'))
    validation = json.loads((PACKAGE / 'PUBLICATION_VALIDATION.json').read_text(encoding='utf-8'))
    assert validation['status'] == 'PASS' and validation['bootstrap_draws'] == 20000
    for row in provenance['files']:
        assert sha(PACKAGE / row['path']) == row['sha256'], 'Copied original bytes changed: ' + row['path']
    entries = {}
    failures = []
    code_files = 0
    arrays = 0
    for path in sorted(PACKAGE.rglob('*')):
        if not path.is_file() or path.name == 'ARTIFACT_MANIFEST.json':
            continue
        relative = path.relative_to(PACKAGE).as_posix()
        if path.stat().st_size >= 50 * 1024 * 1024:
            failures.append((relative, 'file exceeds 50MiB publication limit'))
        if path.suffix in {'.pt', '.pth', '.ckpt', '.onnx', '.npy', '.log', '.pyc'} or path.name == '.memory' or path.name.startswith('.env'):
            failures.append((relative, 'excluded artifact type'))
        if path.suffix == '.npz':
            arrays += 1
            if path.parent.parent.name != 'analysis_inputs':
                failures.append((relative, 'array outside compact analysis inputs'))
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) & {'paths', 'past', 'future', 'observations'}:
                    failures.append((relative, 'full coordinates unexpectedly present'))
        elif path.suffix in {'.py', '.json', '.md', '.yaml', '.yml', '.toml', '.txt', '.ps1', '.cmd', '.xml', '.csv', '.svg'}:
            raw = path.read_bytes()
            # Some original PowerShell evidence is UTF-16; preserve its bytes.
            encoding = 'utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
            text = raw.decode(encoding)
            for name, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    failures.append((relative, name))
            if path.suffix == '.py':
                ast.parse(text, filename=relative)
                code_files += 1
        entries[relative] = dict(bytes=path.stat().st_size, sha256=sha(path))
    for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)', (PACKAGE / 'README.md').read_text(encoding='utf-8')):
        if not link.startswith(('http:', 'https:', '#')) and not (PACKAGE / link.split('#')[0]).exists():
            failures.append(('README.md', 'broken local link: ' + link))
    if failures:
        print(json.dumps(dict(status='FAIL', findings=failures), ensure_ascii=False))
        raise SystemExit(1)
    manifest = dict(schema_version=1, files=entries,
                    copied_source_files=len(provenance['files']), python_files_parsed=code_files,
                    compact_analysis_arrays=arrays, payload_bytes=sum(row['bytes'] for row in entries.values()),
                    exclusions='Manifest excludes itself. Scientific seals remain unchanged historical originals.',
                    publication_checks='Copied original byte hashes; scope/file sizes; Python syntax; common credential signatures; compact-array keys; README links.')
    (PACKAGE / 'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status='SEALED', files=len(entries), compact_arrays=arrays,
                          copied_sources=len(provenance['files']), python_files=code_files,
                          payload_MiB=manifest['payload_bytes'] / 1024**2, credential_pattern_findings=0)))


if __name__ == '__main__':
    main()
