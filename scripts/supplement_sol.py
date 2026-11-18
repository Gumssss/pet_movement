"""Complete only the 107 failed Sol Batch windows with standard API, up to 20 in flight."""
import argparse
import asyncio
import base64
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from openai_batch_eval import ROOT, OUT as BATCH, MODEL, write, now
from pet_imu.paths import recorded_hash_path
from run_cnn import sha
from qwen_classify import parse

OUT = BATCH / 'standard_supplement'
FULL = BATCH / 'full_test'
CONCURRENCY = 20
MAX_ATTEMPTS = 4


def cost(usage, multiplier=1):
    detail = usage.get('input_tokens_details') or {}
    inp, output = usage.get('input_tokens', 0), usage.get('output_tokens', 0)
    cached, writes = detail.get('cached_tokens', 0), detail.get('cache_write_tokens', 0)
    assert cached + writes <= inp
    return multiplier * (2 * (inp - cached - writes) + .1 * cached + 2.5 * writes + 10 * output) / 1e6


def prepare():
    missing = pd.read_csv(BATCH / 'unfinished_windows.csv')
    prior = pd.read_csv(BATCH / 'partial_predictions.csv')
    assert len(missing) == missing.window_id.nunique() == 107
    assert len(prior) == prior.window_id.nunique() == 3601
    assert set(missing.window_id).isdisjoint(prior.window_id)
    audit = json.loads((BATCH / 'partial_complete.json').read_text())
    for name, digest in audit['raw_file_sha256'].items():
        assert sha(BATCH / name) == digest
    target = dict(zip(missing.custom_id, missing.window_id))
    images = {r['custom_id']: r for r in json.loads((ROOT / 'data/vlm_openai_sixplot/images.json').read_text())}
    requests = {}
    for part in json.loads((BATCH / 'batch_manifest.json').read_text()):
        path = recorded_hash_path(BATCH / part['path'])
        assert sha(path) == part['sha256']
        for line in path.open():
            item = json.loads(line); cid = item['custom_id']
            if cid not in target: continue
            assert cid not in requests
            body = item['body']
            assert body['model'] == MODEL and body['reasoning'] == {'effort': 'low'} and body['max_output_tokens'] == 2048
            assert body['input'][0]['content'][0]['detail'] == 'high'
            image_bytes = base64.b64decode(body['input'][0]['content'][0]['image_url'].split(',', 1)[1], validate=True)
            assert hashlib.sha256(image_bytes).hexdigest() == images[cid]['image_sha256']
            assert body['input'][0]['content'][1]['text'] == (BATCH / 'prompt.txt').read_text()
            body['service_tier'] = 'default'
            requests[cid] = {'window_id': target[cid], 'body': body}
    assert set(requests) == set(target)
    protocol = {'model': MODEL, 'processing': 'standard Responses API supplement', 'authorized_windows': 107,
        'concurrency_limit': CONCURRENCY, 'reasoning_effort': 'low', 'max_output_tokens': 2048,
        'image_detail': 'high', 'service_tier': 'default', 'payload': 'Exact original image and prompt body, plus explicit standard service tier.',
        'max_attempts_per_window': MAX_ATTEMPTS, 'retry_policy': 'Only HTTP 429/500/502/503/504 or connection/timeout errors. No retries for model output quality or parsing.',
        'prior_partial_predictions_sha256': sha(BATCH / 'partial_predictions.csv'),
        'prior_partial_metrics_sha256': sha(BATCH / 'partial_metrics.json'),
        'target_list_sha256': sha(BATCH / 'unfinished_windows.csv'), 'base_protocol_sha256': sha(BATCH / 'protocol.json'),
        'requests': {cid: {'window_id': row['window_id'], 'body_sha256': hashlib.sha256(json.dumps(row['body'], sort_keys=True).encode()).hexdigest()}
                     for cid, row in requests.items()},
        'pricing_usd_per_million': {'input': 2, 'cached_input': .1, 'cache_write_30m': 2.5, 'output_including_reasoning': 10}}
    if (OUT / 'protocol.json').exists():
        assert json.loads((OUT / 'protocol.json').read_text()) == protocol
    else:
        write(OUT / 'protocol.json', protocol)
    return requests


def saved_responses():
    path = OUT / 'responses.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


async def run():
    from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

    requests = prepare()
    config = json.loads((ROOT / 'configs/openai.json').read_text())
    assert config['model'] == MODEL and config['base_url'].rstrip('/') == 'https://api.openai.com/v1'
    secret = config['api_key']
    previous = saved_responses()
    done = {r['custom_id'] for r in previous if r['http_status'] == 200}
    assert done <= set(requests)
    semaphore, lock = asyncio.Semaphore(CONCURRENCY), asyncio.Lock()
    completed = len(done)
    api = AsyncOpenAI(api_key=secret, base_url=config['base_url'], max_retries=0, timeout=90)
    write(OUT / 'authorization.json', {'user_instruction': 'Complete the remaining 107 windows using non-Batch API, concurrency up to 20.',
          'authorized_utc': now(), 'scope': 'Original 107 failed window IDs only; do not rerun the 3601 successes.'})
    write(OUT / 'runtime.json', {'started_utc': now(), 'source_sha256': sha(Path(__file__)), 'concurrency_limit': CONCURRENCY})

    async def one(cid, item):
        nonlocal completed
        attempts, response = [], None
        async with semaphore:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                started = time.monotonic()
                try:
                    result = await api.responses.create(**item['body'])
                    response = result.model_dump(mode='json')
                    attempts.append({'attempt': attempt, 'http_status': 200, 'request_id': getattr(result, '_request_id', None),
                                     'wall_seconds': time.monotonic() - started})
                    break
                except APIStatusError as error:
                    info = (error.body or {}).get('error', error.body or {}) if isinstance(error.body, dict) else {}
                    attempts.append({'attempt': attempt, 'http_status': error.status_code, 'request_id': error.request_id,
                                     'error': {key: str(info.get(key, '')).replace(secret, '[REDACTED]')[:300] for key in ['type', 'code', 'message']},
                                     'wall_seconds': time.monotonic() - started})
                    if error.status_code not in [429, 500, 502, 503, 504]: break
                except (APIConnectionError, APITimeoutError) as error:
                    attempts.append({'attempt': attempt, 'network_error_type': type(error).__name__,
                                     'wall_seconds': time.monotonic() - started})
                if attempt < MAX_ATTEMPTS:
                    await asyncio.sleep(2 ** attempt + int(cid[-2:], 16) / 255)
        record = {'custom_id': cid, 'window_id': item['window_id'], 'http_status': 200 if response is not None else 0,
                  'api_response': response, 'attempts': attempts, 'completed_utc': now()}
        safe_text = json.dumps(record, ensure_ascii=False).replace(secret, '[REDACTED]')
        async with lock:
            with (OUT / 'responses.jsonl').open('a') as stream:
                stream.write(safe_text + '\n'); stream.flush(); os.fsync(stream.fileno())
            completed += int(response is not None)
            write(OUT / 'progress.json', {'stage': 'standard_api_running', 'successful_responses': completed,
                  'target_windows': 107, 'concurrency_limit': CONCURRENCY, 'updated_utc': now()})
            print(f'Standard supplement: {completed}/107 successful responses', flush=True)
    try:
        await asyncio.gather(*(one(cid, item) for cid, item in requests.items() if cid not in done))
    finally:
        await api.close()
    merge()


def merge():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert sha(BATCH / 'partial_predictions.csv') == protocol['prior_partial_predictions_sha256']
    assert sha(BATCH / 'partial_metrics.json') == protocol['prior_partial_metrics_sha256']
    raw = saved_responses(); successes = {}
    for row in raw:
        assert row['custom_id'] in protocol['requests']
        if row['http_status'] == 200:
            if row['custom_id'] in successes: raise ValueError('Duplicate successful standard response.')
            successes[row['custom_id']] = row
    if len(successes) != 107:
        write(OUT / 'incomplete.json', {'successful_responses': len(successes), 'expected': 107,
              'remaining_custom_ids': sorted(set(protocol['requests']) - set(successes)), 'updated_utc': now()})
        raise ValueError('Some standard requests still have no model output; review transport attempts before continuing.')
    classes = json.loads((BATCH / 'protocol.json').read_text())['classes']
    test = pd.read_csv(ROOT / 'data/splits/windows.csv'); test = test[test.split.eq('test')].set_index('window_id')
    records = []
    for row in successes.values():
        body = row['api_response']; model = body['model']; assert model == MODEL or model.startswith(MODEL + '-')
        text = ''.join(part.get('text', '') for item in body.get('output', []) if item.get('type') == 'message'
                       for part in item.get('content', []) if part.get('type') == 'output_text')
        prediction, status = parse(text, classes, body['status'] != 'completed')
        u = body['usage']; d = u.get('input_tokens_details') or {}; truth = test.loc[row['window_id']]
        records.append({'window_id': row['window_id'], 'custom_id': row['custom_id'], 'dog_id': int(truth.dog_id),
            'label': truth.label, 'prediction': prediction, 'status': status, 'response': text, 'returned_model': model,
            'input_tokens': u['input_tokens'], 'output_tokens': u['output_tokens'],
            'reasoning_tokens': (u.get('output_tokens_details') or {}).get('reasoning_tokens', 0),
            'cached_tokens': d.get('cached_tokens', 0), 'cache_write_tokens': d.get('cache_write_tokens', 0),
            'estimated_cost_usd': cost(u), 'processing_mode': 'standard_supplement'})
    extra = pd.DataFrame(records)
    extra.to_csv(OUT / 'predictions.csv', index=False)
    prior = pd.read_csv(BATCH / 'partial_predictions.csv'); prior['processing_mode'] = 'batch'
    assert set(prior.window_id).isdisjoint(extra.window_id)
    frame = pd.concat([prior, extra], ignore_index=True).fillna('')
    assert len(frame) == frame.window_id.nunique() == len(test) == 3708
    assert set(frame.window_id) == set(test.index)
    frame = frame.set_index('window_id').loc[test.index].reset_index()
    assert frame.label.tolist() == test.label.tolist()
    cm = np.zeros((7, 8), dtype=int)
    for row in frame.to_dict('records'):
        cm[classes.index(row['label']), classes.index(row['prediction']) if row['prediction'] in classes else 7] += 1
    tp = np.diag(cm[:, :7]); support = cm.sum(1); predicted = cm[:, :7].sum(0)
    recall = tp / support
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(7), where=support + predicted > 0)
    standard_cost = float(extra.estimated_cost_usd.sum())
    batch_cost = json.loads((BATCH / 'partial_metrics.json').read_text())['token_usage_and_estimated_cost']['estimated_cost_usd']
    totals = {key: int(frame[key].sum()) for key in ['input_tokens', 'output_tokens', 'reasoning_tokens', 'cached_tokens', 'cache_write_tokens']}
    totals.update(estimated_batch_cost_usd=batch_cost, estimated_standard_cost_usd=standard_cost,
                  estimated_total_sol_test_cost_usd=batch_cost + standard_cost)
    metrics = {'model': MODEL, 'test_windows': 3708, 'balanced_accuracy': float(recall.mean()), 'macro_f1': float(f1.mean()),
        'accuracy': float(tp.sum() / 3708), 'status_counts': dict(Counter(frame.status)),
        'processing_counts': dict(Counter(frame.processing_mode)), 'token_usage_and_estimated_cost': totals,
        'confusion_matrix': cm.tolist(), 'confusion_matrix_columns': classes + ['FAILED'],
        'per_class': {label: {'support': int(support[i]), 'recall': float(recall[i]), 'f1': float(f1[i])} for i, label in enumerate(classes)},
        'evaluation_scope': 'All 3708 fixed test windows: 3601 retained Batch outputs plus 107 standard API supplements.',
        'generation_note': 'Same model alias, images, prompt, reasoning effort and output budget; processing tiers differ.',
        'original_batch_and_partial_results_unchanged': True, 'completed_utc': now()}
    FULL.mkdir(exist_ok=True)
    frame.to_csv(FULL / 'predictions.csv', index=False)
    write(FULL / 'metrics.json', metrics)
    # Independent exported-row metric check.
    saved = pd.read_csv(FULL / 'predictions.csv').fillna(''); f1_check = []; recall_check = []
    for label in classes:
        truth, pred = saved.label.eq(label), saved.prediction.eq(label); hit = int((truth & pred).sum())
        recall_check.append(hit / truth.sum()); f1_check.append(2 * hit / (truth.sum() + pred.sum()))
    assert abs(np.mean(recall_check) - metrics['balanced_accuracy']) < 1e-12
    assert abs(np.mean(f1_check) - metrics['macro_f1']) < 1e-12
    assert abs((saved.label == saved.prediction).mean() - metrics['accuracy']) < 1e-12
    assert sha(BATCH / 'partial_predictions.csv') == protocol['prior_partial_predictions_sha256']
    assert sha(BATCH / 'partial_metrics.json') == protocol['prior_partial_metrics_sha256']
    lines = ['# GPT-6.1 Sol — full 3,708-window test', '',
        '3,601 original Batch outputs retained unchanged; 107 failed requests completed through standard Responses API with concurrency up to 20.',
        'Identical model alias, image bytes, English prompt, high detail, low reasoning and 2048 output-token budget; no semantic retries.', '',
        '| Metric | Value |', '|---|---:|']
    for key in ['balanced_accuracy', 'macro_f1', 'accuracy']: lines.append(f'| {key} | {metrics[key]:.8f} |')
    lines += ['', f'Status counts: {metrics["status_counts"]}',
        f'Estimated token cost: Batch ${batch_cost:.6f}; standard supplement ${standard_cost:.6f}; total Sol test ${batch_cost + standard_cost:.6f}.',
        'Includes reasoning/cache-write usage at the applicable processing tier. Excludes the earlier Astra preflight and taxes/credits; actual invoice governs.', '',
        'Original partial results and failed Batch response files remain in the parent directory. Standard raw responses are in ../standard_supplement/responses.jsonl.',
        'Full exported per-window predictions: predictions.csv. Metrics and class support: metrics.json.']
    (FULL / 'README.md').write_text('\n'.join(lines) + '\n')
    write(FULL / 'complete.json', {'state': 'complete', 'test_windows': 3708, 'metrics_verified': True,
          'batch_windows': 3601, 'standard_windows': 107, 'original_partial_results_preserved': True,
          'predictions_sha256': sha(FULL / 'predictions.csv'), 'standard_raw_sha256': sha(OUT / 'responses.jsonl'),
          'source_sha256': sha(Path(__file__)), 'completed_utc': now()})
    write(OUT / 'complete.json', {'state': 'complete', 'standard_windows': 107, 'estimated_standard_cost_usd': standard_cost,
          'transport_attempts': sum(len(row['attempts']) for row in raw), 'updated_utc': now()})
    write(OUT / 'progress.json', {'stage': 'complete', 'successful_responses': 107, 'target_windows': 107,
          'concurrency_limit': CONCURRENCY, 'updated_utc': now()})
    print(json.dumps({k: metrics[k] for k in ['test_windows', 'balanced_accuracy', 'macro_f1', 'accuracy', 'status_counts', 'processing_counts', 'token_usage_and_estimated_cost']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('verify-inputs', 'merge', 'run'))
    args = parser.parse_args()
    if args.command == 'verify-inputs':
        print(f'Verified {len(prepare())} saved requests locally.')
    elif args.command == 'merge':
        merge()
    else:
        asyncio.run(run())
