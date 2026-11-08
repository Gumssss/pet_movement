"""Add train/validation ACL images without changing any historical test assets."""
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from render_imu_acl import ROOT, BASE, figure, render, sha

OUT = BASE / 'supervised'


def init_worker():
    global X, MEAN, STD, PROTOCOL, FIG, LINES, MARKERS
    X = np.load(ROOT / 'dataset/cache/pet_windows.npz')['X']
    stats = json.loads((BASE / 'normalization.json').read_text())
    MEAN = np.array(stats['mean']); STD = np.array(stats['std'])
    PROTOCOL = json.loads((BASE / 'render_protocol.json').read_text())
    FIG, LINES, MARKERS = figure(PROTOCOL)


def draw_chunk(chunk):
    output = []
    for item in chunk:
        item = dict(item)
        i = item.pop('cache_index')
        name = 'sample_' + hashlib.sha256(item['window_id'].encode()).hexdigest()[:20] + '.webp'
        path = OUT / 'images' / name
        info = render(X[i], path, MEAN, STD, PROTOCOL, FIG, LINES, MARKERS)
        output.append({**item, 'image_file': str(path.relative_to(ROOT)), **info})
    return output


def main():
    OUT.mkdir(exist_ok=True); (OUT / 'images').mkdir(exist_ok=True)
    windows = pd.read_csv(ROOT / 'dataset/splits/windows.csv')
    z = np.load(ROOT / 'dataset/cache/pet_windows.npz')
    classes = json.loads((ROOT / 'results/protocol.json').read_text())['classes']
    assert np.array_equal(z['dogs'], windows.dog_id)
    assert np.array_equal(np.array(classes)[z['y']], windows.label)
    assert np.array_equal(z['tests'], windows.session) and np.allclose(z['starts'], windows.start_sec)
    selected = windows[windows.split.isin(['train', 'validation'])].copy()
    assert selected.split.value_counts().to_dict() == {'train': 16461, 'validation': 4727}
    assert not set(selected.dog_id) & set(windows[windows.split.eq('test')].dog_id)
    selected.insert(0, 'cache_index', selected.index)
    target = OUT / 'images.csv'
    done = {}
    if target.exists():
        for row in pd.read_csv(target).to_dict('records'):
            assert sha(ROOT / row['image_file']) == row['image_sha256']
            done[row['window_id']] = row
    jobs = [row for row in selected.to_dict('records') if row['window_id'] not in done]
    protocol = json.loads((BASE / 'render_protocol.json').read_text())
    fields = [c for c in selected.columns if c != 'cache_index'] + ['image_file', 'render_seconds', 'image_sha256']
    for c in protocol['channels']:
        fields.extend(c + s for s in ['_below', '_above', '_min', '_max'])
    mode = 'a' if target.exists() else 'w'
    chunks = [jobs[i:i + 128] for i in range(0, len(jobs), 128)]
    with target.open(mode, newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if mode == 'w': writer.writeheader()
        with ProcessPoolExecutor(max_workers=8, initializer=init_worker) as pool:
            for output in pool.map(draw_chunk, chunks):
                writer.writerows(output); stream.flush()
                done.update({r['window_id']: r for r in output})
                print(f'Supervised images: {len(done)}/21188', flush=True)
    assert len(done) == 21188
    frame = pd.read_csv(target)
    assert set(frame.window_id) == set(selected.window_id)
    check = {'windows_sha256': sha(ROOT / 'dataset/splits/windows.csv'),
             'split_sha256': sha(ROOT / 'dataset/splits/dog_splits.json'),
             'render_protocol_sha256': sha(BASE / 'render_protocol.json'),
             'normalization_sha256': sha(BASE / 'normalization.json'),
             'manifest_sha256': sha(target), 'counts': frame.split.value_counts().to_dict(),
             'all_channels': protocol['channels'], 'test_dogs_excluded': True,
             'rule': 'Exact original ACL renderer and training-only normalization/range; train/val only; original test files untouched.'}
    (OUT / 'validation.json').write_text(json.dumps(check, ensure_ascii=False, indent=2) + '\n')
    print('All train/validation images are ready.', flush=True)


if __name__ == '__main__':
    main()
