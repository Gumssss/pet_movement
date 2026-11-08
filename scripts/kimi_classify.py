"""Moonshot API zero-shot evaluation on the frozen, anonymous ACL test images."""
import argparse
import asyncio
import base64
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import httpx
from qwen_classify import parse

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset/vlm_acl'
DEST = ROOT / 'results/kimi/k3.1_pro_acl'
MODEL = 'kimi-k3.1-pro'
BASE = 'https://api.moonshot.ai/v1'
MAX_TOKENS = 4096
RETRIES = 5


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def records():
    with (ROOT / 'dataset/splits/windows.csv').open() as f:
        windows = [r for r in csv.DictReader(f) if r['split'] == 'test']
    with (DATA / 'images.csv').open() as f:
        images = list(csv.DictReader(f))
    assert len(windows) == len(images) == 3708
    assert [r['window_id'] for r in windows] == [r['window_id'] for r in images]
    assert len({r['window_id'] for r in windows}) == 3708
    for w, im in zip(windows, images):
        expected = 'sample_' + hashlib.sha256(w['window_id'].encode()).hexdigest()[:20] + '.webp'
        path = ROOT / im['image_file']
        assert path.parent == DATA / 'images' and path.name == expected
        assert sha(path) == im['image_sha256']
    return windows, images


def prepare():
    from PIL import Image
    DEST.mkdir(parents=True, exist_ok=True)
    windows, images = records()
    classes = json.loads((ROOT / 'results/protocol.json').read_text())['classes']
    assert (DATA / 'prompt.txt').read_text().isascii()
    for im in images:
        with Image.open(ROOT / im['image_file']) as img:
            assert img.size == (512, 409) and img.format == 'WEBP' and img.mode == 'RGB'
            assert not any(k in img.info for k in ['exif', 'xmp', 'icc_profile'])
    frozen = {
        'experiment': 'imu_2s_acl_shared_axis_zero_shot_en_kimi_api',
        'model': MODEL, 'model_revision': None,
        'revision_note': 'Provider model alias; server weights/revision and internal inference framework are not inspectable.',
        'base_url': BASE, 'inference_backend': 'moonshot_api',
        'evaluation_split': 'test', 'expected_windows': 3708, 'training_examples': 0,
        'classes': classes, 'prompt_language': 'en', 'data_dir': str(DATA.relative_to(ROOT)),
        'prompt_sha256': sha(DATA / 'prompt.txt'),
        'render_protocol_sha256': sha(DATA / 'render_protocol.json'),
        'normalization_sha256': sha(DATA / 'normalization.json'),
        'images_manifest_sha256': sha(DATA / 'images.csv'),
        'windows_sha256': sha(ROOT / 'dataset/splits/windows.csv'),
        'split_sha256': sha(ROOT / 'dataset/splits/dog_splits.json'),
        'input_image_size': [512, 409], 'image_mime_type': 'image/webp',
        'generation': {'reasoning_effort': 'low', 'max_tokens': MAX_TOKENS,
                       'thinking': 'always enabled by provider',
                       'temperature': 'provider-fixed 1.0; omitted in request',
                       'top_p': 'provider-fixed 0.95; omitted in request',
                       'n': 1, 'seed': None, 'tools': False, 'guided_decoding': False},
        'payload': 'One test image encoded as a WebP data URL and the unchanged English prompt; no filename, dog ID, Task, window ID or truth sent.',
        'parsing': 'Same exact label-only JSON parser as Qwen; only final message.content parsed; failures count as incorrect; no semantic retries.',
        'transport_retry_policy': 'At most 5 attempts for 429, 5xx and network failures; identical payload, exponential backoff. Network retries can duplicate billable server work.',
        'comparison_note': 'Same ACL images, prompt and test windows as Qwen3.8-27B ACL. Model, provider image preprocessing, reasoning, sampling, output budget and execution infrastructure differ; API server processing/resolution is not inspectable.',
        'documentation': ['https://platform.kimi.ai/docs/get-your-api-key',
                          'https://platform.kimi.com/docs/guide/kimi-k3-1-pro-quickstart',
                          'https://platform.kimi.com/docs/api/models-overview',
                          'https://platform.kimi.com/docs/guide/use-kimi-vision-model'],
    }
    path = DEST / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == frozen, 'Frozen experiment changed; use a new directory.'
    else:
        write(path, frozen)
    if (DEST / 'prompt.txt').exists():
        assert (DEST / 'prompt.txt').read_bytes() == (DATA / 'prompt.txt').read_bytes()
    else:
        (DEST / 'prompt.txt').write_bytes((DATA / 'prompt.txt').read_bytes())
    if not (DEST / 'prior_files_sha256.json').exists():
        paths = [p for p in (ROOT / 'results').rglob('*')
                 if p.is_file() and not p.is_relative_to(DEST) and p.parent != ROOT / 'results']
        paths += [ROOT / 'results/protocol.json']
        paths += [p for p in (ROOT / 'dataset').rglob('*')
                  if p.is_file() and p.suffix in ['.json', '.csv', '.txt', '.md'] and p.stat().st_size < 30_000_000]
        write(DEST / 'prior_files_sha256.json', {str(p.relative_to(ROOT)): sha(p) for p in paths})
    write(DEST / 'input_validation.json', {'test_windows': 3708, 'image_hashes_dimensions_metadata_verified': True,
                                          'same_acl_prompt_verified': True, 'only_test_images_sent': True,
                                          'test_window_order_verified': True})
    return windows, images, frozen


class StopRun(Exception):
    pass


async def infer_one(client, image, prompt, classes, key):
    image_bytes = (ROOT / image['image_file']).read_bytes()
    assert hashlib.sha256(image_bytes).hexdigest() == image['image_sha256']
    payload = {'model': MODEL, 'reasoning_effort': 'low', 'max_tokens': MAX_TOKENS,
               'messages': [{'role': 'user', 'content': [
                   {'type': 'image_url', 'image_url': {'url': 'data:image/webp;base64,' + base64.b64encode(image_bytes).decode()}},
                   {'type': 'text', 'text': prompt}]}]}
    started = time.perf_counter()
    attempts = []
    result = None
    for attempt in range(RETRIES):
        call_started = time.perf_counter()
        try:
            response = await client.post(BASE + '/chat/completions', json=payload)
            attempts.append({'attempt': attempt + 1, 'http_status': response.status_code,
                             'request_seconds': time.perf_counter() - call_started,
                             'request_id': response.headers.get('x-request-id')})
            if response.is_success:
                try:
                    result = response.json()
                except ValueError:
                    if attempt + 1 < RETRIES:
                        await asyncio.sleep(2 ** (attempt + 1))
                        continue
                break
            try:
                error = response.json().get('error', {})
            except ValueError:
                error = {'type': 'non_json_http_error'}
            # Never serialize request objects, headers, API secrets or image data URLs.
            safe_error = {k: str(v).replace(key, '[REDACTED]')[:500] for k, v in error.items() if k in ['code', 'type', 'message']}
            attempts[-1]['error'] = safe_error
            if response.status_code not in [429, 500, 502, 503, 504]:
                raise StopRun(json.dumps({'http_status': response.status_code, 'error': safe_error}))
            if attempt + 1 < RETRIES:
                retry_after = response.headers.get('retry-after', '')
                delay = float(retry_after) if retry_after.replace('.', '', 1).isdigit() else 2 ** (attempt + 1)
                await asyncio.sleep(min(delay, 60))
        except httpx.TransportError as exc:
            attempts.append({'attempt': attempt + 1, 'network_error_type': type(exc).__name__,
                             'request_seconds': time.perf_counter() - call_started})
            if attempt + 1 < RETRIES:
                await asyncio.sleep(2 ** (attempt + 1))
    if result is None:
        # Stop before recording a prediction: an unreachable provider is not a model answer.
        raise StopRun(json.dumps({'reason': 'transport_retries_exhausted', 'attempts': attempts}))
    result = json.loads(json.dumps(result, ensure_ascii=False).replace(key, '[REDACTED]'))
    if result.get('model') != MODEL:
        raise StopRun('Unexpected returned model: ' + str(result.get('model')))
    choices = result.get('choices', [])
    if len(choices) != 1:
        raise StopRun('Expected exactly one API completion.')
    choice = choices[0]
    text = choice.get('message', {}).get('content') or ''
    prediction, status = parse(text, classes, choice.get('finish_reason') == 'length')
    usage = result.get('usage', {})
    return {'window_id': image['window_id'], 'image_file': image['image_file'],
            'image_sha256': image['image_sha256'], 'prediction': prediction, 'status': status,
            'response': text, 'finish_reason': choice.get('finish_reason'),
            'inference_backend': 'moonshot_api', 'returned_model': result['model'],
            'input_tokens': usage.get('prompt_tokens'), 'generated_tokens': usage.get('completion_tokens'),
            'usage': usage, 'api_response': result, 'attempts': attempts,
            'request_wall_seconds': time.perf_counter() - started, 'timestamp_utc': now()}


def load_raw():
    path = DEST / 'responses.jsonl'
    if not path.exists():
        return []
    rows = []
    with path.open('rb+') as f:
        while line := f.readline():
            if not line.endswith(b'\n'):
                f.truncate(f.tell() - len(line))
                break
            rows.append(json.loads(line))
    assert len(rows) == len({r['window_id'] for r in rows})
    return rows


async def run(args):
    _, images, protocol = prepare()
    key = os.environ.get('MOONSHOT_API_KEY', '')
    if not key:
        raise StopRun('Set MOONSHOT_API_KEY in the environment; never pass it as a CLI argument.')
    rows = load_raw()
    done = {r['window_id']: r for r in rows}
    expected = {r['window_id']: r for r in images}
    for wid, row in done.items():
        assert wid in expected and row['image_sha256'] == expected[wid]['image_sha256']
        assert row['returned_model'] == MODEL
    pending = [r for r in images if r['window_id'] not in done]
    if args.limit:
        pending = pending[:args.limit]
    prompt = (DEST / 'prompt.txt').read_text()
    start = time.perf_counter()
    write(DEST / 'progress.json', {'completed': len(done), 'expected': 3708, 'state': 'running'})
    session = {'started_utc': now(), 'inference_backend': 'moonshot_api', 'base_url': BASE,
               'model': MODEL, 'concurrency': args.concurrency, 'completed_at_start': len(done),
               'code_sha256': sha(Path(__file__)), 'httpx_version': httpx.__version__}
    try:
        async with httpx.AsyncClient(headers={'Authorization': 'Bearer ' + key}, timeout=180,
                                     follow_redirects=False, limits=httpx.Limits(max_connections=args.concurrency)) as client:
            # Resolve the requested model before sending any test pictures.
            info = await client.get(BASE + '/models')
            if not info.is_success:
                raise StopRun('Model authentication HTTP ' + str(info.status_code))
            available = [m['id'] for m in info.json().get('data', [])]
            if MODEL not in available:
                raise StopRun('Requested model is unavailable; no substitute model used.')
            write(DEST / 'provider_info.json', {'base_url': BASE, 'checked_utc': now(),
                                               'requested_model': MODEL, 'requested_model_available': True})
            with (DEST / 'responses.jsonl').open('a') as stream:
                # Await every in-flight response before raising; already-paid answers are preserved.
                for start_idx in range(0, len(pending), args.concurrency):
                    group = pending[start_idx:start_idx + args.concurrency]
                    outputs = await asyncio.gather(*(infer_one(client, im, prompt, protocol['classes'], key) for im in group), return_exceptions=True)
                    errors = []
                    for output in outputs:
                        if isinstance(output, BaseException):
                            errors.append(output)
                            continue
                        stream.write(json.dumps(output, ensure_ascii=False) + '\n')
                        stream.flush()
                        os.fsync(stream.fileno())
                        done[output['window_id']] = output
                    elapsed = time.perf_counter() - start
                    write(DEST / 'progress.json', {'completed': len(done), 'expected': 3708, 'remaining': 3708 - len(done),
                                                   'state': 'running' if not errors else 'stopped',
                                                   'session_wall_seconds': elapsed,
                                                   'status_counts': dict(Counter(r['status'] for r in done.values()))})
                    print(f"Kimi: {len(done)}/3708; session {elapsed:.1f}s; " + str(dict(Counter(r['status'] for r in done.values()))), flush=True)
                    if errors:
                        raise errors[0]
        session['state'] = 'complete' if len(done) == 3708 else 'partial'
    except Exception as exc:
        session['state'] = 'stopped'
        session['error_type'] = type(exc).__name__
        session['error'] = str(exc).replace(key, '[REDACTED]')[:2000] if isinstance(exc, StopRun) else 'See error type; request objects/credentials are not logged.'
        write(DEST / 'stopped.json', {'timestamp_utc': now(), 'completed': len(done), 'error_type': session['error_type'], 'error': session['error']})
        raise StopRun(session['error']) from None
    finally:
        session.update(ended_utc=now(), session_wall_seconds=time.perf_counter() - start,
                       session_completed=len(done) - session['completed_at_start'])
        path = DEST / 'runtime.json'
        runtime = json.loads(path.read_text()) if path.exists() else {'sessions': []}
        runtime['sessions'].append(session)
        write(path, runtime)
    write(DEST / 'progress.json', {'completed': len(done), 'expected': 3708, 'remaining': 3708 - len(done),
                                   'state': session['state'], 'status_counts': dict(Counter(r['status'] for r in done.values()))})


def evaluate():
    import numpy as np
    import pandas as pd
    from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, recall_score
    windows, images, p = prepare()
    raw = load_raw()
    assert len(raw) == 3708, 'Full evaluation requires all 3708 responses; partial data retained without a full-test score.'
    df = pd.DataFrame(windows).merge(pd.DataFrame(raw), on='window_id', validate='one_to_one', sort=False)
    assert df.window_id.tolist() == [w['window_id'] for w in windows]
    assert df.image_sha256.tolist() == [r['image_sha256'] for r in images]
    classes = p['classes']; invalid = '__INVALID__'
    truth = df.label.to_numpy(); predicted = df.prediction.fillna(invalid).to_numpy(); dogs = df.dog_id.to_numpy()
    cm = confusion_matrix(truth, predicted, labels=classes + [invalid])[:7, :]
    def scores(c):
        support = c.sum(1); tp = np.diag(c[:, :7]); counted = c[:, :7].sum(0)
        f1 = np.divide(2 * tp, support + counted, out=np.zeros(7), where=(support + counted) > 0)
        recall = np.divide(tp, support, out=np.zeros(7), where=support > 0)
        return {'macro_f1': float(f1.mean()), 'balanced_accuracy': float(recall[support > 0].mean()),
                'accuracy': float(tp.sum() / support.sum())}
    m = scores(cm)
    assert abs(m['macro_f1'] - f1_score(truth, predicted, labels=classes, average='macro', zero_division=0)) < 1e-12
    assert abs(m['balanced_accuracy'] - recall_score(truth, predicted, labels=classes, average='macro', zero_division=0)) < 1e-12
    assert abs(m['accuracy'] - accuracy_score(truth, predicted)) < 1e-12
    unique = np.unique(dogs)
    cms = np.stack([confusion_matrix(truth[dogs == d], predicted[dogs == d], labels=classes + [invalid])[:7, :] for d in unique])
    rng = np.random.default_rng(20261101)
    boot = [scores(cms[rng.integers(0, len(unique), len(unique))].sum(0)) for _ in range(2000)]
    for r in raw:
        label, status = parse(r['api_response']['choices'][0]['message'].get('content') or '', classes, r['finish_reason'] == 'length')
        assert label == r['prediction'] and status == r['status']
        assert r['returned_model'] == MODEL and r['inference_backend'] == 'moonshot_api'
    old_files = json.loads((DEST / 'prior_files_sha256.json').read_text())
    for path, digest in old_files.items():
        assert sha(ROOT / path) == digest, path
    m.update(model=MODEL, inference_backend='moonshot_api', experiment=p['experiment'], test_windows=3708,
             status_counts=dict(Counter(df.status)), parse_success_rate=float(df.status.eq('ok').mean()), failures_counted_as_incorrect=True,
             classification_report=classification_report(truth, predicted, labels=classes, output_dict=True, zero_division=0),
             confusion_matrix=cm.tolist(), confusion_rows=classes, confusion_columns=classes + [invalid],
             per_dog={str(d): {**scores(cms[i]), 'windows': int((dogs == d).sum())} for i, d in enumerate(unique)},
             dog_bootstrap_95ci={k: np.quantile([b[k] for b in boot], [.025, .975]).tolist() for k in m},
             total_input_tokens=int(df.input_tokens.sum()), total_generated_tokens=int(df.generated_tokens.sum()),
             request_wall_seconds_p50=float(df.request_wall_seconds.median()), request_wall_seconds_p95=float(df.request_wall_seconds.quantile(.95)),
             latency_note='Observed per-request client wall time includes network, provider queueing, inference and transport retry waits; concurrent requests overlap.',
             billing_note='API-reported usage retained per response. Actual charges are provider-side; unanswered network retries may incur extra charges not visible in responses.')
    write(DEST / 'metrics.json', m)
    cols = ['window_id', 'dog_id', 'session', 'start_sec', 'end_exclusive_sec', 'label', 'prediction', 'status', 'response',
            'image_file', 'image_sha256', 'returned_model', 'finish_reason', 'input_tokens', 'generated_tokens', 'request_wall_seconds']
    df[cols].to_csv(DEST / 'predictions.csv', index=False)
    write(DEST / 'validation.json', {'test_windows': 3708, 'same_acl_images_and_prompt': True,
                                    'raw_prediction_metrics_consistent': True, 'prior_files_unchanged': len(old_files),
                                    'api_model_verified': MODEL, 'scores': {k: m[k] for k in ['macro_f1', 'balanced_accuracy', 'accuracy']}})
    print(json.dumps({k: m[k] for k in ['model', 'test_windows', 'macro_f1', 'balanced_accuracy', 'accuracy', 'status_counts']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'evaluate', 'all'])
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0, help='Optional partial transport check; responses remain formal test predictions.')
    args = parser.parse_args()
    if args.concurrency < 1 or args.limit < 0:
        parser.error('Concurrency must be positive and limit nonnegative.')
    try:
        if args.command == 'prepare':
            prepare()
        if args.command in ['run', 'all']:
            asyncio.run(run(args))
        if args.command in ['evaluate', 'all']:
            evaluate()
    except StopRun as exc:
        print('STOPPED: ' + str(exc), flush=True)
        raise SystemExit(1)
