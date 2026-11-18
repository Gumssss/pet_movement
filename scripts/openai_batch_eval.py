"""Fixed test-set image evaluation via GPT-6.1 Sol Batch API; never synchronous inference."""
from pet_imu.paths import recorded_path
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image

from run_cnn import sha
from render_imu import load_windows, figure, render
from qwen_classify import parse

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'gpt-6.1-sol'
DATA = ROOT / 'data/vlm_openai_sixplot'
OUT = ROOT / 'outputs/vlm/api/openai/gpt6_1_sol_sixplot_batch'
TERMINAL = {'completed', 'failed', 'cancelled', 'expired'}


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def protocol():
    p = json.loads((ROOT / 'outputs/protocol.json').read_text())
    return {'model': MODEL, 'inference_backend': 'openai_batch_responses', 'test_windows': 3708,
            'split': 'test', 'classes': p['classes'], 'training_examples': 0,
            'prompt_sha256': sha(ROOT / 'data/vlm/prompt.txt'),
            'window_manifest_sha256': sha(ROOT / 'data/splits/windows.csv'),
            'split_sha256': sha(ROOT / 'data/splits/dog_splits.json'),
            'render_protocol_sha256': sha(ROOT / 'data/vlm/render_protocol.json'),
            'source_zip_sha256': p['audit']['csv_zip_sha256'],
            'image_size': [1280, 1152], 'image_detail': 'high', 'image_format': 'lossless WebP RGB',
            'reasoning_effort': 'low', 'max_output_tokens': 2048, 'temperature': 'provider default; omitted',
            'completion_window': '24h', 'retries': 'No semantic retries or automatic duplicate batch submissions.',
            'sampling': 'Every fixed test window once; no selection using CNN correctness.',
            'payload': 'Anonymous image bytes and unchanged English prompt only. No truth, dog/session/window ID, filename, CNN output or examples sent.',
            'parsing': 'Same exact JSON label parser as Qwen; failures remain in denominator as incorrect.',
            'comparison_note': 'Re-rendered from the same raw windows and saved training-fixed axes using current macOS libraries. Pixel hashes may differ from historical Linux images; model preprocessing, reasoning and backend also differ.',
            'pricing_usd_per_million_batch_tokens': {'input': 1, 'cached_input': .05, 'cache_write_30m': 1.25, 'output_including_reasoning': 5},
            'pricing_source': 'https://developers.openai.com/api/docs/pricing'}


def check_protocol():
    expected = protocol()
    path = OUT / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError('Frozen protocol changed; use a new versioned experiment.')
    else:
        write(path, expected)
    return expected


def prepare():
    p = check_protocol()
    windows, signals, original = load_windows()
    audit = json.loads((ROOT / 'data/cache/pet_audit.json').read_text())
    assert audit['csv_zip_sha256'] == original['audit']['csv_zip_sha256'] == p['source_zip_sha256']
    test = windows[windows.split.eq('test')]
    assert len(test) == 3708 and test.window_id.is_unique
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / 'images').mkdir(exist_ok=True)
    render_protocol = json.loads((ROOT / 'data/vlm/render_protocol.json').read_text())
    existing = {row['window_id']: row for row in json.loads((DATA / 'images.json').read_text())} if (DATA / 'images.json').exists() else {}
    records = []
    fig, lines, points = figure(render_protocol)
    for ordinal, (i, row) in enumerate(test.iterrows(), 1):
        anon = 'sample_' + hashlib.sha256(row.window_id.encode()).hexdigest()[:20]
        path = DATA / 'images' / (anon + '.webp')
        if row.window_id in existing:
            info = existing[row.window_id]
            assert path.exists() and sha(path) == info['image_sha256']
        else:
            info = render(signals[i], path, render_protocol, fig, lines, points)
            info = {'window_id': row.window_id, 'custom_id': anon, 'image_file': str(path.relative_to(ROOT)), **info}
        with Image.open(path) as image:
            assert image.size == (1280, 1152) and image.mode == 'RGB'
            assert not any(key in image.info for key in ['exif', 'xmp', 'icc_profile'])
        records.append(info)
        if ordinal % 100 == 0 or ordinal == len(test):
            write(DATA / 'images.json', records)
            write(OUT / 'progress.json', {'stage': 'rendering', 'rendered': ordinal, 'total': len(test), 'updated_utc': now()})
            print(f'Rendered {ordinal}/{len(test)} images', flush=True)
    import matplotlib.pyplot as plt
    plt.close(fig)
    # Snapshot the unchanged prompt; future submit/collect commands verify its hash.
    (OUT / 'prompt.txt').write_bytes((ROOT / 'data/vlm/prompt.txt').read_bytes())
    write(DATA / 'render_protocol.json', render_protocol)
    write(OUT / 'runtime.json', {'created_utc': now(), 'versions': {name: importlib.metadata.version(name)
          for name in ['numpy', 'pandas', 'torch', 'matplotlib', 'pillow', 'openai']},
          'source_sha256': {name: sha(ROOT / 'scripts' / name) for name in ['openai_batch_eval.py', 'render_imu.py', 'qwen_classify.py']}})
    build_inputs(records, p)


def build_inputs(records, p):
    folder = OUT / 'batch_inputs'
    folder.mkdir(exist_ok=True)
    prompt = (OUT / 'prompt.txt').read_text()
    assert hashlib.sha256(prompt.encode()).hexdigest() == p['prompt_sha256']
    manifest, ids = [], []
    stream, size, count, part = None, 0, 0, 0
    for record in records:
        path = recorded_path(record['image_file'])
        assert sha(path) == record['image_sha256']
        payload = {'model': MODEL, 'store': False, 'reasoning': {'effort': 'low'}, 'max_output_tokens': 2048,
                   'input': [{'role': 'user', 'content': [
                       {'type': 'input_image', 'detail': 'high', 'image_url': 'data:image/webp;base64,' + base64.b64encode(path.read_bytes()).decode()},
                       {'type': 'input_text', 'text': prompt}]}]}
        request = {'custom_id': record['custom_id'], 'method': 'POST', 'url': '/v1/responses', 'body': payload}
        line = (json.dumps(request, ensure_ascii=True, separators=(',', ':')) + '\n').encode()
        if stream is None or count >= 1000 or size + len(line) > 190_000_000:
            if stream:
                stream.close()
                manifest.append({'part': part, 'path': str(destination.relative_to(OUT)), 'requests': count, 'bytes': size, 'sha256': sha(destination), 'custom_ids': ids})
            part += 1
            destination = folder / f'part_{part:02d}.jsonl'
            stream = destination.open('wb')
            size, count, ids = 0, 0, []
        stream.write(line)
        size += len(line)
        count += 1
        ids.append(record['custom_id'])
    if stream:
        stream.close()
        manifest.append({'part': part, 'path': str(destination.relative_to(OUT)), 'requests': count, 'bytes': size, 'sha256': sha(destination), 'custom_ids': ids})
    all_ids = [cid for group in manifest for cid in group['custom_ids']]
    assert len(all_ids) == len(set(all_ids)) == 3708
    write(OUT / 'batch_manifest.json', manifest)
    write(OUT / 'input_validation.json', {'windows': 3708, 'all_images_dimensions_metadata_hashes_verified': True,
          'images_manifest_sha256': sha(DATA / 'images.json'), 'one_request_per_window': True,
          'parts': len(manifest), 'each_part_under_200mb': all(r['bytes'] < 200_000_000 for r in manifest),
          'same_original_prompt': True, 'no_labels_or_cnn_outputs_in_api_payload': True})
    write(OUT / 'progress.json', {'stage': 'prepared', 'requests': 3708, 'parts': len(manifest), 'updated_utc': now()})
    print('Batch inputs ready:', [(r['requests'], round(r['bytes'] / 1e6, 1)) for r in manifest], flush=True)


def client():
    from openai import OpenAI
    config = json.loads((ROOT / 'configs/openai.json').read_text())
    if config['model'] != MODEL or config['base_url'].rstrip('/') != 'https://api.openai.com/v1':
        raise ValueError('Configuration must select gpt-6.1-sol and the official OpenAI endpoint.')
    return OpenAI(api_key=config['api_key'], base_url=config['base_url'], max_retries=0, timeout=180)


def submit():
    if (OUT / 'stop_requested.json').exists():
        raise ValueError('User stopped this experiment; do not submit or resume its unfinished requests automatically.')
    check_protocol()
    manifest = json.loads((OUT / 'batch_manifest.json').read_text())
    state_path = OUT / 'batches.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else []
    api = client()
    access = api.models.retrieve(MODEL)
    write(OUT / 'model_access.json', {'id': access.id, 'checked_utc': now()})
    for part in manifest:
        known = next((s for s in state if s['part'] == part['part']), None)
        if known and known.get('batch_id'):
            continue
        # Reconcile by metadata before creating; never blindly duplicate a timed-out create.
        matches = [b for b in api.batches.list(limit=100) if b.metadata and b.metadata.get('experiment') == OUT.name
                   and b.metadata.get('input_sha256') == part['sha256']]
        if len(matches) > 1:
            raise RuntimeError('Duplicate remote batch metadata found; review before proceeding.')
        if matches:
            batch = matches[0]
            if known is None:
                known = {'part': part['part']}; state.append(known)
            known.update(batch_id=batch.id, file_id=batch.input_file_id)
            write(state_path, state)
            continue
        path = OUT / part['path']
        assert sha(path) == part['sha256']
        if known is None:
            known = {'part': part['part'], 'requests': part['requests'], 'input_sha256': part['sha256']}; state.append(known)
        if not known.get('file_id'):
            print(f'Uploading part {part["part"]}: {part["requests"]} requests...', flush=True)
            with path.open('rb') as source:
                uploaded = api.files.create(file=source, purpose='batch')
            known['file_id'] = uploaded.id
            write(state_path, state)
        batch = api.batches.create(input_file_id=known['file_id'], endpoint='/v1/responses', completion_window='24h',
            metadata={'experiment': OUT.name, 'input_sha256': part['sha256'], 'part': str(part['part'])})
        known['batch_id'] = batch.id
        known['submitted_utc'] = now()
        write(state_path, state)
        write(OUT / f'batch_{part["part"]:02d}.json', batch.model_dump(mode='json'))
        print(f'Submitted part {part["part"]}: {batch.id} ({batch.status})', flush=True)


def collect():
    check_protocol()
    api = client()
    state = json.loads((OUT / 'batches.json').read_text())
    manifest = json.loads((OUT / 'batch_manifest.json').read_text())
    if len(state) != len(manifest) or any(not row.get('batch_id') for row in state):
        raise ValueError('Not all batch parts have been submitted.')
    all_done, summary = True, []
    for row in state:
        batch = api.batches.retrieve(row['batch_id'])
        write(OUT / f'batch_{row["part"]:02d}.json', batch.model_dump(mode='json'))
        summary.append({'part': row['part'], 'batch_id': batch.id, 'status': batch.status,
                        'request_counts': batch.request_counts.model_dump() if batch.request_counts else None})
        if batch.status not in TERMINAL:
            all_done = False
            continue
        for kind in ['output', 'error']:
            file_id = getattr(batch, kind + '_file_id')
            path = OUT / f'part_{row["part"]:02d}_{kind}.jsonl'
            if file_id and not path.exists():
                content = api.files.content(file_id)
                content.write_to_file(path)
    write(OUT / 'progress.json', {'stage': 'returned' if all_done else 'batch_processing',
          'batches': summary, 'updated_utc': now()})
    print(json.dumps(summary), flush=True)
    if all_done:
        evaluate()
    return all_done


def evaluate():
    if (OUT / 'stop_requested.json').exists():
        from evaluate_partial_sol import evaluate_partial
        return evaluate_partial()
    p = check_protocol()
    manifest = json.loads((OUT / 'batch_manifest.json').read_text())
    images = json.loads((DATA / 'images.json').read_text())
    expected = {r['custom_id']: r for r in images}
    rows = {}
    for path in sorted(OUT.glob('part_*_*.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line); cid = row['custom_id']
            if cid not in expected or cid in rows:
                raise ValueError('Unexpected or duplicate output custom_id.')
            rows[cid] = row
    windows = pd.read_csv(ROOT / 'data/splits/windows.csv')
    test = windows[windows.split.eq('test')].set_index('window_id')
    results, totals = [], Counter()
    for cid, image in expected.items():
        row = rows.get(cid, {})
        response = row.get('response') or {}
        body = response.get('body') or {}
        text = ''.join(part.get('text', '') for item in body.get('output', []) if item.get('type') == 'message'
                       for part in item.get('content', []) if part.get('type') == 'output_text')
        model = body.get('model', '')
        if response.get('status_code') == 200:
            if model != MODEL and not model.startswith(MODEL + '-'):
                raise ValueError('Unexpected returned model: ' + model)
            prediction, status = parse(text, p['classes'], body.get('status') != 'completed')
        else:
            prediction, status = None, 'api_error' if row else 'missing_response'
        usage = body.get('usage') or {}
        detail = usage.get('input_tokens_details') or {}
        n_input, n_output = usage.get('input_tokens', 0), usage.get('output_tokens', 0)
        n_cached, n_write = detail.get('cached_tokens', 0), detail.get('cache_write_tokens', 0)
        assert n_cached + n_write <= n_input
        cost = ((n_input - n_cached - n_write) + .05 * n_cached + 1.25 * n_write + 5 * n_output) / 1e6
        truth = test.loc[image['window_id']]
        result = {'window_id': image['window_id'], 'custom_id': cid, 'dog_id': int(truth.dog_id),
                  'label': truth.label, 'prediction': prediction, 'status': status, 'response': text,
                  'returned_model': model, 'input_tokens': n_input, 'output_tokens': n_output,
                  'reasoning_tokens': (usage.get('output_tokens_details') or {}).get('reasoning_tokens', 0),
                  'cached_tokens': n_cached, 'cache_write_tokens': n_write, 'estimated_cost_usd': cost}
        results.append(result)
        for name in ['input_tokens', 'output_tokens', 'reasoning_tokens', 'cached_tokens', 'cache_write_tokens', 'estimated_cost_usd']:
            totals[name] += result[name]
    assert len(results) == len(test) == 3708
    frame = pd.DataFrame(results)
    frame.to_csv(OUT / 'predictions.csv', index=False)
    classes = p['classes']; cm = np.zeros((7, 8), dtype=int)
    for row in results:
        cm[classes.index(row['label']), classes.index(row['prediction']) if row['prediction'] in classes else 7] += 1
    def scores(matrix):
        tp = np.diag(matrix[:, :7]); support = matrix.sum(1); predicted = matrix[:, :7].sum(0)
        recall = np.divide(tp, support, out=np.zeros(7), where=support > 0)
        f1 = np.divide(2 * tp, support + predicted, out=np.zeros(7), where=support + predicted > 0)
        return {'macro_f1': float(f1.mean()), 'balanced_accuracy': float(recall[support > 0].mean()),
                'accuracy': float(tp.sum() / matrix.sum())}
    metrics = {**scores(cm), 'model': MODEL, 'test_windows': 3708, 'status_counts': dict(Counter(frame.status)),
               'token_usage_and_estimated_cost': dict(totals), 'confusion_matrix_columns': classes + ['FAILED'],
               'confusion_matrix': cm.tolist(), 'per_class_support': dict(zip(classes, map(int, cm.sum(1)))),
               'failures_count_as_incorrect': True, 'all_custom_ids_returned': len(rows) == 3708,
               'completed_utc': now()}
    write(OUT / 'metrics.json', metrics)
    # Independent, row-wise accuracy/recall check after export.
    exported = pd.read_csv(OUT / 'predictions.csv').fillna('')
    accuracy = (exported.label == exported.prediction).mean()
    balanced = np.mean([(g.label == g.prediction).mean() for _, g in exported.groupby('label')])
    assert abs(accuracy - metrics['accuracy']) < 1e-12 and abs(balanced - metrics['balanced_accuracy']) < 1e-12
    text = ['# GPT-6.1 Sol sixplot Batch test', '', 'All 3,708 fixed test windows; no examples, unchanged English prompt, low reasoning, high image detail.',
            'Failures remain in the denominator. Images re-rendered on macOS from saved training-fixed scales.', '',
            '| Metric | Value |', '|---|---:|']
    for key in ['macro_f1', 'balanced_accuracy', 'accuracy']:
        text.append(f'| {key} | {metrics[key]:.8f} |')
    text += ['', f'Statuses: {metrics["status_counts"]}', f'Estimated Batch token cost: ${totals["estimated_cost_usd"]:.6f}.',
             'Includes billed reasoning and cache-write tokens; excludes taxes/credits. Actual invoice is authoritative.',
             '', 'Protocol, remote batches, raw responses, per-window predictions, usage and costs are preserved in this directory.']
    (OUT / 'README.md').write_text('\n'.join(text) + '\n')
    if len(rows) == 3708:
        write(OUT / 'complete.json', {'state': 'complete', 'windows': 3708, 'metrics_verified': True,
              'completed_utc': now(), 'api_failures': int(sum(frame.status != 'ok'))})
    else:
        write(OUT / 'incomplete.json', {'returned': len(rows), 'expected': 3708, 'updated_utc': now()})
    print(json.dumps({k: v for k, v in metrics.items() if k not in ['confusion_matrix', 'confusion_matrix_columns']}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'submit', 'collect', 'evaluate', 'run'])
    args = parser.parse_args()
    try:
        if args.command == 'prepare': prepare()
        elif args.command == 'submit': submit()
        elif args.command == 'collect': collect()
        elif args.command == 'evaluate': evaluate()
        else:
            if not (OUT / 'batch_manifest.json').exists(): prepare()
            submit()
            while not collect(): time.sleep(45)
    except Exception as error:
        # Do not serialize API request objects, headers, base64 payloads or credentials.
        write(OUT / 'failure.json', {'type': type(error).__name__, 'updated_utc': now(),
              'note': 'Stopped without automatic request/batch retries. Review saved remote batch state before resuming.'})
        print('Stopped:', type(error).__name__, flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
