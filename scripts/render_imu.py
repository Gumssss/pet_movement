"""Render the existing IMU windows with training-only, fixed per-channel scales."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'dataset/vlm'
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#332288']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_windows():
    w = pd.read_csv(ROOT / 'dataset/splits/windows.csv')
    z = np.load(ROOT / 'dataset/cache/pet_windows.npz')
    p = json.loads((ROOT / 'results/protocol.json').read_text())
    assert len(w) == len(z['X']) == 24896
    assert np.array_equal(w.dog_id, z['dogs']) and np.array_equal(w.session, z['tests'])
    assert np.allclose(w.start_sec, z['starts'])
    assert np.array_equal(w.label, np.array(p['classes'])[z['y']])
    return w, z['X'], p

def make_protocol(w, x, p):
    train = x[w.split.eq('train')].reshape(-1, 6)
    # Fit a robust display range before viewing any test model outputs.
    q = np.quantile(train, [.001, .999], axis=0)
    pad = .05 * np.maximum(q[1] - q[0], 1e-6)
    return {
        'version': 'imu_2s_v1', 'window_manifest_sha256': sha(ROOT / 'dataset/splits/windows.csv'),
        'split_sha256': sha(ROOT / 'dataset/splits/dog_splits.json'),
        'channels': p['channels'], 'colors': COLORS, 'sampling_hz': 100,
        'samples_per_window': 200, 'input_signal': 'raw stored values; no magnitude reduction or normalization',
        'scale_fit_split': 'train', 'scale_fit_windows': int(w.split.eq('train').sum()),
        'range_rule': 'per-channel training quantiles 0.001 and 0.999 plus 5% of that span on both sides',
        'quantile_method': 'numpy linear', 'quantiles': q.tolist(),
        'y_limits': np.stack([q[0] - pad, q[1] + pad], axis=1).tolist(),
        'out_of_range': 'display clipped at fixed axes; boundary triangles; all counts/min/max logged per image',
        'width': 1280, 'height': 1152, 'dpi': 100, 'font': 'DejaVu Sans',
        'font_size_pt': 11, 'line_width_pt': 1.2, 'background': 'white',
        'time_axis_seconds': [0, 2], 'image_format': 'WebP lossless RGB; no metadata',
        'test_windows': int(w.split.eq('test').sum()),
        'filename_rule': 'sample_ + first 20 hex characters of SHA256(window_id) + .webp',
        'rendering_label_access': 'labels are used only to verify cache alignment; never drawn or encoded in image metadata',
    }

def figure(protocol):
    plt.rcParams.update({'font.family': protocol['font'], 'font.size': 11,
                         'axes.facecolor': 'white', 'figure.facecolor': 'white'})
    fig, axes = plt.subplots(6, 1, figsize=(12.8, 11.52), dpi=100, sharex=True)
    fig.subplots_adjust(left=.15, right=.98, bottom=.075, top=.95, hspace=.27)
    fig.suptitle('Collar IMU | 2.00 s | 100 Hz | raw stored values', fontsize=14, y=.985)
    fig.text(.15, .018, 'Triangles mark samples outside the training-fixed display range.', fontsize=10)
    lines, points = [], []
    for c, ax in enumerate(axes):
        lo, hi = protocol['y_limits'][c]
        ax.set_ylim(lo, hi)
        ax.set_xlim(0, 2)
        ax.set_yticks(np.linspace(lo, hi, 3))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3g'))
        ax.set_ylabel(protocol['channels'][c], rotation=0, ha='right', va='center', labelpad=16, fontsize=12)
        ax.grid(True, alpha=.22, linewidth=.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xticks(np.arange(0, 2.01, .5))
        line, = ax.plot(np.arange(200)/100, np.zeros(200), color=COLORS[c], linewidth=1.2)
        low = ax.scatter([], [], marker='v', color=COLORS[c], s=15, zorder=3)
        high = ax.scatter([], [], marker='^', color=COLORS[c], s=15, zorder=3)
        lines.append(line); points.append((low, high))
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    return fig, lines, points

def render(x, destination, protocol, fig, lines, points):
    started = time.perf_counter()
    info = {}
    t = np.arange(200)/100
    for c, channel in enumerate(protocol['channels']):
        lo, hi = protocol['y_limits'][c]
        v = x[:, c]
        below, above = v < lo, v > hi
        lines[c].set_ydata(np.clip(v, lo, hi))
        points[c][0].set_offsets(np.column_stack([t[below], np.full(below.sum(), lo)]))
        points[c][1].set_offsets(np.column_stack([t[above], np.full(above.sum(), hi)]))
        info[channel + '_below'] = int(below.sum())
        info[channel + '_above'] = int(above.sum())
        info[channel + '_min'] = float(v.min())
        info[channel + '_max'] = float(v.max())
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    assert image.size == (1280, 1152)
    image.save(destination, format='WEBP', lossless=True, method=4)
    info['render_seconds'] = time.perf_counter() - started
    info['image_sha256'] = sha(destination)
    return info

def prepare():
    BASE.mkdir(exist_ok=True)
    (BASE / 'images').mkdir(exist_ok=True)
    w, x, p = load_windows()
    protocol = make_protocol(w, x, p)
    pp = BASE / 'render_protocol.json'
    if pp.exists():
        assert json.loads(pp.read_text()) == protocol, 'Existing rendering protocol differs.'
    else:
        pp.write_text(json.dumps(protocol, indent=2) + '\n')
    fig, lines, points = figure(protocol)
    preview = BASE / 'preview.webp'
    if not preview.exists():
        i = int(np.flatnonzero(w.split.eq('train'))[0])
        render(x[i], preview, protocol, fig, lines, points)
    records_path = BASE / 'images.csv'
    existing = pd.read_csv(records_path).to_dict('records') if records_path.exists() else []
    done = {r['window_id']: r for r in existing}
    fields = ['window_id', 'image_file', 'image_sha256', 'render_seconds']
    for channel in p['channels']:
        fields += [channel + s for s in ['_below', '_above', '_min', '_max']]
    mode = 'a' if records_path.exists() else 'w'
    with records_path.open(mode, newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if mode == 'w': writer.writeheader()
        for i in np.flatnonzero(w.split.eq('test')):
            wid = w.window_id.iloc[i]
            name = 'sample_' + hashlib.sha256(wid.encode()).hexdigest()[:20] + '.webp'
            path = BASE / 'images' / name
            if wid in done:
                assert path.exists() and sha(path) == done[wid]['image_sha256']
                continue
            info = render(x[i], path, protocol, fig, lines, points)
            record = {'window_id': wid, 'image_file': str(path.relative_to(ROOT)), **info}
            writer.writerow(record); stream.flush()
            done[wid] = record
            if len(done) % 100 == 0:
                print(f'rendered {len(done)}/{protocol["test_windows"]}', flush=True)
    plt.close(fig)
    assert len(done) == protocol['test_windows'] == 3708
    print('Rendering complete: 3708 test windows; training-only preview and fixed axes.', flush=True)

def check_processor():
    from transformers import AutoProcessor
    protocol = json.loads((BASE / 'render_protocol.json').read_text())
    image = Image.open(BASE / 'preview.webp').convert('RGB')
    checks = {}
    for model in ['4B', '27B']:
        processor = AutoProcessor.from_pretrained(ROOT / 'models' / f'Qwen3.5-{model}', local_files_only=True)
        data = processor.image_processor(images=[image], return_tensors='pt')
        grid = data['image_grid_thw'][0].tolist()
        _, gh, gw = grid
        patch, merge = processor.image_processor.patch_size, processor.image_processor.merge_size
        assert (gw*patch, gh*patch) == image.size == (1280,1152)
        # Invert the documented local processor's patch layout to inspect actual pixels.
        arr = data['pixel_values'].numpy().reshape(gh//merge, gw//merge, merge, merge, 3, 2, patch, patch)
        arr = arr[:, :, :, :, :, 0].transpose(4, 0, 2, 5, 1, 3, 6).reshape(3,gh*patch,gw*patch)
        mean = np.array(processor.image_processor.image_mean)[:,None,None]
        std = np.array(processor.image_processor.image_std)[:,None,None]
        rgb = np.rint(np.clip(arr*std+mean,0,1)*255).astype('uint8').transpose(1,2,0)
        assert np.max(np.abs(rgb.astype(int)-np.asarray(image).astype(int))) <= 1
        Image.fromarray(rgb).save(BASE / f'processor_preview_{model.lower()}.webp', lossless=True)
        checks[model] = {'export_size': list(image.size), 'processed_size': [gw*patch,gh*patch],
                         'image_grid_thw': grid, 'image_tokens': gh*gw//merge**2,
                         'max_pixel_reconstruction_error': int(np.abs(rgb.astype(int)-np.asarray(image).astype(int)).max())}
    (BASE/'processor_check.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2),flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','check-processor'])
    args = parser.parse_args()
    prepare() if args.command == 'prepare' else check_processor()
