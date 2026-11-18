"""Evaluate returned HTTP-200 outputs after the user stops the Sol Batch run."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qwen_classify import parse
from openai_batch_eval import ROOT, OUT, DATA, MODEL, check_protocol, write, now
from run_cnn import sha


def evaluate_partial():
    protocol = check_protocol()
    assert (OUT / 'stop_requested.json').exists()
    images = json.loads((DATA / 'images.json').read_text())
    expected = {row['custom_id']: row for row in images}
    windows = pd.read_csv(ROOT / 'data/splits/windows.csv')
    test = windows[windows.split.eq('test')].set_index('window_id')
    returned = {}
    for path in sorted(OUT.glob('part_*_*.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['custom_id'] not in expected or row['custom_id'] in returned:
                raise ValueError('Unexpected or duplicated custom ID in partial output.')
            returned[row['custom_id']] = row
    evaluated, unfinished, totals = [], [], Counter()
    classes = protocol['classes']
    for cid, image in expected.items():
        row = returned.get(cid, {})
        response = row.get('response') or {}
        body = response.get('body') or {}
        usage = body.get('usage') or {}
        detail = usage.get('input_tokens_details') or {}
        inp, outp = usage.get('input_tokens', 0), usage.get('output_tokens', 0)
        cached, writes = detail.get('cached_tokens', 0), detail.get('cache_write_tokens', 0)
        assert cached + writes <= inp
        cost = (inp - cached - writes + .05 * cached + 1.25 * writes + 5 * outp) / 1e6
        for key, value in {'input_tokens': inp, 'output_tokens': outp, 'cached_tokens': cached,
                           'cache_write_tokens': writes, 'reasoning_tokens': (usage.get('output_tokens_details') or {}).get('reasoning_tokens', 0),
                           'estimated_cost_usd': cost}.items():
            totals[key] += value
        truth = test.loc[image['window_id']]
        common = {'window_id': image['window_id'], 'custom_id': cid, 'dog_id': int(truth.dog_id), 'label': truth.label}
        if response.get('status_code') != 200:
            error = row.get('error') or body.get('error') or {}
            unfinished.append({**common, 'http_status': response.get('status_code'),
                               'reason': error.get('code', 'not_returned_after_stop' if not row else 'api_error')})
            continue
        model = body.get('model', '')
        if model != MODEL and not model.startswith(MODEL + '-'):
            raise ValueError('Unexpected returned model.')
        text = ''.join(part.get('text', '') for item in body.get('output', []) if item.get('type') == 'message'
                       for part in item.get('content', []) if part.get('type') == 'output_text')
        prediction, status = parse(text, classes, body.get('status') != 'completed')
        evaluated.append({**common, 'prediction': prediction, 'status': status, 'response': text,
                          'response_status': body.get('status'), 'returned_model': model,
                          'input_tokens': inp, 'output_tokens': outp,
                          'reasoning_tokens': (usage.get('output_tokens_details') or {}).get('reasoning_tokens', 0),
                          'cached_tokens': cached, 'cache_write_tokens': writes, 'estimated_cost_usd': cost})
    if not evaluated:
        raise ValueError('No model outputs have been returned yet.')
    frame = pd.DataFrame(evaluated)
    assert len(frame) + len(unfinished) == len(expected) == 3708
    cm = np.zeros((7, 8), dtype=int)
    for row in evaluated:
        cm[classes.index(row['label']), classes.index(row['prediction']) if row['prediction'] in classes else 7] += 1
    tp = np.diag(cm[:, :7]); support = cm.sum(1); pred_support = cm[:, :7].sum(0)
    recall = np.divide(tp, support, out=np.zeros(7), where=support > 0)
    precision = np.divide(tp, pred_support, out=np.zeros(7), where=pred_support > 0)
    f1 = np.divide(2 * tp, support + pred_support, out=np.zeros(7), where=support + pred_support > 0)
    metrics = {'model': MODEL, 'evaluation_scope': 'Partial test set: returned HTTP-200 model responses only after user cancellation.',
               'test_windows': 3708, 'evaluated_windows': len(frame), 'unfinished_windows': len(unfinished),
               'coverage': len(frame) / 3708, 'balanced_accuracy': float(recall[support > 0].mean()),
               'macro_f1': float(f1.mean()), 'accuracy': float(tp.sum() / len(frame)),
               'status_counts': dict(Counter(frame.status)), 'unfinished_reason_counts': dict(Counter(r['reason'] for r in unfinished)),
               'per_class': {c: {'support': int(support[i]), 'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i])}
                             for i, c in enumerate(classes)},
               'confusion_matrix': cm.tolist(), 'confusion_matrix_columns': classes + ['FAILED'],
               'token_usage_and_estimated_cost': dict(totals),
               'model_output_failures_count_as_incorrect': True, 'unexecuted_requests_excluded_from_partial_denominator': True,
               'full_test_evaluation_complete': len(unfinished) == 0,
               'limitation': 'Server completion order is not a random sample. These are partial-set results, not the full 3708-window benchmark.',
               'created_utc': now()}
    for name in ['predictions.csv', 'partial_predictions.csv']:
        frame.to_csv(OUT / name, index=False)
    pd.DataFrame(unfinished, columns=['window_id', 'custom_id', 'dog_id', 'label', 'http_status', 'reason']).to_csv(OUT / 'unfinished_windows.csv', index=False)
    for name in ['metrics.json', 'partial_metrics.json']:
        write(OUT / name, metrics)
    # Independently recompute precision/recall/F1 from exported rows, including parse failures.
    exported = pd.read_csv(OUT / 'partial_predictions.csv').fillna('')
    recalls, f1s = [], []
    for c in classes:
        true, pred = exported.label.eq(c), exported.prediction.eq(c)
        hit = int((true & pred).sum()); den = int(true.sum() + pred.sum())
        if true.any(): recalls.append(hit / true.sum())
        f1s.append(2 * hit / den if den else 0)
    assert abs(float(np.mean(recalls)) - metrics['balanced_accuracy']) < 1e-12
    assert abs(float(np.mean(f1s)) - metrics['macro_f1']) < 1e-12
    assert abs((exported.label == exported.prediction).mean() - metrics['accuracy']) < 1e-12
    lines = ['# GPT-6.1 Sol Batch — partial results after user cancellation', '',
             f'Evaluated {len(frame)} / 3708 fixed test windows; {len(unfinished)} windows were not evaluated.',
             'No automatic resubmission. Unexecuted requests are excluded from these partial-set metrics; returned but truncated/unparseable model outputs count as incorrect.', '',
             '| Metric | Partial-set value |', '|---|---:|']
    for key in ['balanced_accuracy', 'macro_f1', 'accuracy']:
        lines.append(f'| {key} | {metrics[key]:.8f} |')
    lines += ['', f'Statuses: {metrics["status_counts"]}', f'Estimated returned-token Batch cost: ${totals["estimated_cost_usd"]:.6f}.',
              'Includes reasoning/cache writes; actual invoice governs.', '', metrics['limitation'], '',
              'Files: partial_predictions.csv, partial_metrics.json, unfinished_windows.csv, raw part output/error JSONL files.', '',
              '| Class | Evaluated windows | Recall | F1 |', '|---|---:|---:|---:|']
    for c, stats in metrics['per_class'].items():
        lines.append(f'| {c} | {stats["support"]} | {stats["recall"]:.6f} | {stats["f1"]:.6f} |')
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    write(OUT / 'partial_complete.json', {'state': 'stopped_with_partial_results', 'evaluated_windows': len(frame),
          'unfinished_windows': len(unfinished), 'metrics_verified': True, 'automatic_resubmission': False,
          'source_sha256': {name: sha(ROOT / 'scripts' / name) for name in ['evaluate_partial_sol.py', 'qwen_classify.py']},
          'raw_file_sha256': {path.name: sha(path) for path in sorted(OUT.glob('part_*_*.jsonl'))}, 'created_utc': now()})
    print(json.dumps({k: metrics[k] for k in ['evaluated_windows', 'unfinished_windows', 'balanced_accuracy', 'macro_f1', 'accuracy', 'status_counts', 'token_usage_and_estimated_cost']}, indent=2), flush=True)
    return metrics


if __name__ == '__main__':
    evaluate_partial()
