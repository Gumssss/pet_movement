#!/usr/bin/env python3
"""Download a pinned InrtlWrist release and verify upstream Git/LFS hashes."""
import argparse
import hashlib
import json
import subprocess
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

REPO = 'YiquanLi/InrtlWrist'
REVISION = 'deef284b464d14145cd4b7e82e9a9f8fbe267d75'


def download_large(url, path, size):
    chunk_size = 4 * 1024 * 1024
    chunks = [(i, start, min(start + chunk_size, size) - 1)
              for i, start in enumerate(range(0, size, chunk_size))]

    def fetch(chunk):
        i, start, end = chunk
        part = path.with_name(path.name + f'.chunk{i}')
        header = part.with_name(part.name + '.headers')
        if part.exists() and header.exists() and part.stat().st_size == end - start + 1:
            prior_ranges = re.findall(r'content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)',
                                      header.read_text(), re.IGNORECASE)
            if prior_ranges and tuple(map(int, prior_ranges[-1])) == (start, end, size):
                print(f'Reusing chunk {i + 1}/{len(chunks)}', flush=True)
                return part, header
        subprocess.run(['curl', '-fL', '--retry', '2', '--connect-timeout', '15',
                        '--max-time', '60', '--silent', '--show-error',
                        '--range', f'{start}-{end}', '--dump-header', str(header),
                        url + f'?download=true&chunk={i}', '-o', str(part)], check=True)
        ranges = re.findall(r'content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)',
                            header.read_text(), re.IGNORECASE)
        assert ranges and tuple(map(int, ranges[-1])) == (start, end, size), f'Range mismatch: {i}'
        assert part.stat().st_size == end - start + 1, f'Chunk size mismatch: {i}'
        print(f'Chunk {i + 1}/{len(chunks)} verified', flush=True)
        return part, header

    with ThreadPoolExecutor(max_workers=8) as pool:
        files = list(pool.map(fetch, chunks))
    with path.open('wb') as out:
        for part, _ in files:
            with part.open('rb') as src:
                for block in iter(lambda: src.read(1024 * 1024), b''):
                    out.write(block)
    return files


def hashes(path):
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    git = hashlib.sha1(f'blob {size}\0'.encode())
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            sha256.update(block)
            git.update(block)
    return size, sha256.hexdigest(), git.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', required=True, type=Path)
    parser.add_argument('--endpoint', default='https://huggingface.co')
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    assert metadata['id'] == REPO and metadata['sha'] == REVISION
    root = Path(__file__).resolve().parents[1]
    target = root / 'models' / 'InrtlWrist'
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for entry in metadata['siblings']:
        relative = Path(entry['rfilename'])
        assert not relative.is_absolute() and '..' not in relative.parts
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f'{args.endpoint.rstrip("/")}/{REPO}/resolve/{REVISION}/{quote(relative.as_posix())}'
        if not path.exists():
            temporary = path.with_name(path.name + '.partial')
            print(f'Downloading {relative}: {entry["size"]:,} bytes', flush=True)
            chunk_files = []
            if entry['size'] > 8 * 1024 * 1024:
                chunk_files = download_large(url, temporary, entry['size'])
            else:
                subprocess.run(['curl', '-fL', '--retry', '2', '--connect-timeout', '15',
                                '--max-time', '600', '--silent', '--show-error',
                                url, '-o', str(temporary)], check=True)
            check_path = temporary
        else:
            check_path = path
        size, sha256, git = hashes(check_path)
        assert size == entry['size'], f'Size mismatch: {relative}'
        if 'lfs' in entry:
            assert sha256 == entry['lfs']['sha256'], f'LFS SHA256 mismatch: {relative}'
        else:
            assert git == entry['blobId'], f'Git blob hash mismatch: {relative}'
        if check_path != path:
            check_path.replace(path)
            for pair in chunk_files:
                for chunk_file in pair:
                    chunk_file.unlink()
        records.append({'file': relative.as_posix(), 'size': size, 'sha256': sha256,
                        'upstream': entry, 'download_url': url})
        print(f'Verified {relative}', flush=True)
    manifest = {'repository': REPO, 'revision': REVISION,
                'completed_utc': datetime.now(timezone.utc).isoformat(),
                'endpoint': args.endpoint, 'resolved_path': str(target.resolve()),
                'files': records, 'total_bytes': sum(x['size'] for x in records),
                'verification': 'All files verified against upstream Git blob SHA1 or LFS SHA256; no model code executed.'}
    (target / 'download_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    log_dir = root / 'outputs/records/model_checks'
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / 'inrtlwrist_download.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Complete: {target.resolve()} ({manifest["total_bytes"]:,} bytes)', flush=True)


if __name__ == '__main__':
    main()
