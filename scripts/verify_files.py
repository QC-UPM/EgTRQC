"""Verify the files in the integrated distribution against the recorded SHA-256 manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = root / 'provenance' / 'integration_files.sha256.json'
    if not manifest.is_file():
        parser.error(f'Missing checksum manifest: {manifest}')
    entries = json.loads(manifest.read_text(encoding='utf-8'))
    missing, changed = [], []
    for relative, expected in entries.items():
        path = root / relative
        if not path.is_file():
            missing.append(relative)
        elif sha256(path) != expected:
            changed.append(relative)
    print(json.dumps({'checked': len(entries), 'missing': missing, 'changed': changed,
                      'status': 'passed' if not missing and not changed else 'failed'}, indent=2))
    return int(bool(missing or changed))


if __name__ == '__main__':
    raise SystemExit(main())
