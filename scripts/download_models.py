"""Download pinned official Qwen snapshots directly into the project's models folder."""
import os
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES'] = '0'
os.environ['HF_HUB_DISABLE_XET'] = '1'
from pathlib import Path
import hashlib
import json
import shutil
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'Qwen3.5-4B': '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
    'Qwen3.5-27B': 'fc05daec18b0a78c049392ed2e771dde82bdf654',
    'Qwen3.8-27B': '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
}
def main(selected):
    api = HfApi(endpoint='https://huggingface.co')
    manifest_path = ROOT / 'models' / 'model_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    remaining = 0
    for name in selected:
        revision = MODELS[name]
        info = api.model_info('Qwen/' + name, revision=revision, files_metadata=True)
        files = []
        model_remaining = 0
        for entry in info.siblings:
            path = ROOT / 'models' / name / entry.rfilename
            size = entry.size or 0
            model_remaining += max(0, size - (path.stat().st_size if path.is_file() else 0))
            lfs = entry.lfs
            files.append({'path': entry.rfilename, 'size': size, 'sha256': lfs.sha256 if lfs else None})
        partial_dir = ROOT / 'models' / name / '.cache' / 'huggingface' / 'download'
        partial_bytes = sum(p.stat().st_size for p in partial_dir.rglob('*.incomplete') if p.is_file())
        remaining += max(0, model_remaining - partial_bytes)
        manifest[name] = {'repo_id': 'Qwen/' + name, 'revision': revision, 'files': files}
    free = shutil.disk_usage(ROOT).free
    print(f'Remaining download: {remaining / 2**30:.2f} GiB; free: {free / 2**30:.2f} GiB', flush=True)
    if free < remaining + 2 * 2**30:
        raise RuntimeError('Insufficient disk space: keep at least 2 GiB free.')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    for name in selected:
        metadata = manifest[name]
        print('Downloading', name, metadata['revision'], flush=True)
        target = ROOT / 'models' / name
        snapshot_download(metadata['repo_id'], revision=metadata['revision'], local_dir=target, max_workers=3)
        for entry in metadata['files']:
            path = target / entry['path']
            if not path.is_file() or path.stat().st_size != entry['size']:
                raise RuntimeError(f'Missing or incomplete file: {path}')
            if entry['sha256']:
                h = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
                        h.update(block)
                if h.hexdigest() != entry['sha256']:
                    raise RuntimeError(f'Checksum mismatch: {path}')
        print('VERIFIED', name, flush=True)
    print('All model files downloaded and SHA-256 verified.', flush=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=list(MODELS), help='Download/verify only this model')
    args = parser.parse_args()
    main([args.model] if args.model else list(MODELS))
