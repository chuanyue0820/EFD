"""Verify downloaded pretrained files against the author's published hashes."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('pretrained'))
    args = parser.parse_args()
    manifest = Path(__file__).resolve().parents[1] / 'pretrained/manifest.json'
    missing = 0
    for row in json.loads(manifest.read_text(encoding='utf-8')):
        path = args.directory / row['filename']
        if not path.is_file():
            print('MISSING %s' % row['filename'])
            missing += 1
            continue
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != row['sha256']:
            raise ValueError('SHA-256 mismatch: %s' % path)
        print('OK %s' % row['filename'])
    if missing:
        raise SystemExit('%d files missing; see pretrained/README.md.' % missing)


if __name__ == '__main__':
    main()
