"""Fit one positive temperature on all fixed validation windows; never train CNN weights."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pet_imu.paths import CNN_CHECKPOINTS
from run_cnn import ROOT, Predictor, sha
from pet_imu.data import load_windows


def fit_temperature(logits, labels):
    # NLL is convex in inverse temperature beta=1/T. Bracket its derivative,
    # then bisect to avoid optimizer learning rates or starting-point effects.
    z = torch.as_tensor(logits, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.long)
    true_z = z[torch.arange(len(y)), y]

    def derivative(beta):
        return float(((z * (z * beta).softmax(1)).sum(1) - true_z).mean())

    lo, hi = 0.0, 1.0
    if derivative(lo) >= 0:
        raise ValueError('NLL optimum is uniform probabilities (infinite T); no finite temperature fitted.')
    for _ in range(60):
        if derivative(hi) >= 0:
            break
        hi *= 2
    else:
        raise ValueError('No finite inverse-temperature optimum could be bracketed.')
    for _ in range(100):
        mid = (lo + hi) / 2
        if derivative(mid) < 0:
            lo = mid
        else:
            hi = mid
    beta = (lo + hi) / 2
    return 1 / beta, {'method': 'FP64 bisection of convex NLL derivative in beta=1/T',
                       'iterations': 100, 'inverse_temperature_gradient': derivative(beta)}


def metrics(logits, labels, temperature=1.0, bins=15):
    z = torch.as_tensor(logits, dtype=torch.float64) / temperature
    y = torch.as_tensor(labels, dtype=torch.long)
    p = z.softmax(1)
    confidence, prediction = p.max(1)
    correct = prediction.eq(y).double()
    one_hot = F.one_hot(y, z.shape[1]).double()
    reliability = []
    for i in range(bins):
        mask = (confidence >= i / bins) & ((confidence < (i + 1) / bins) if i < bins - 1 else (confidence <= 1))
        n = int(mask.sum())
        reliability.append({'lower': i / bins, 'upper': (i + 1) / bins, 'count': n,
                            'accuracy': float(correct[mask].mean()) if n else None,
                            'mean_confidence': float(confidence[mask].mean()) if n else None})
    ece = sum(row['count'] / len(y) * abs(row['accuracy'] - row['mean_confidence'])
              for row in reliability if row['count'])
    return {'windows': len(y), 'nll': float(F.cross_entropy(z, y)),
            'brier_multiclass_sum': float(((p - one_hot) ** 2).sum(1).mean()),
            'ece_top_label_15_equal_width_bins': ece, 'accuracy': float(correct.mean()),
            'mean_confidence': float(confidence.mean()), 'reliability_bins': reliability}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=CNN_CHECKPOINTS / "cnn_best.pt")
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/RQ1/baselines/cnn_temperature_calibration')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output directory exists; select a new --output-dir.')
    torch.set_num_threads(2)
    initial_sha = sha(args.checkpoint)
    predictor = Predictor(args.checkpoint, 'cpu')
    windows, signals, protocol = load_windows()
    cache_audit = json.loads((ROOT / 'data/cache/pet_audit.json').read_text())
    assert cache_audit['csv_zip_sha256'] == protocol['audit']['csv_zip_sha256']
    splits = json.loads((ROOT / 'data/splits/dog_splits.json').read_text())['splits']
    assert set(splits['validation']).isdisjoint(splits['test'])
    mask = windows.split.eq('validation').to_numpy()
    val = windows.loc[mask].reset_index(drop=True)
    assert len(val) == 4727 and set(val.dog_id) == set(splits['validation'])
    y = np.array([predictor.classes.index(label) for label in val.label])
    assert set(y) == set(range(7)) and protocol['classes'] == predictor.classes
    print(f'Computing logits for {len(val)} validation windows on CPU...', flush=True)
    logits = predictor.logits(signals[mask])
    if not np.isfinite(logits).all():
        raise ValueError('Non-finite model logits.')
    temperature, solver = fit_temperature(logits, y)
    before, after = metrics(logits, y), metrics(logits, y, temperature)
    assert after['nll'] <= before['nll'] + 1e-12
    assert np.array_equal(logits.argmax(1), (logits / temperature).argmax(1))
    assert sha(args.checkpoint) == initial_sha
    out = args.output_dir
    out.mkdir(parents=True)
    calibration = {'method': 'scalar_temperature_scaling', 'temperature': temperature,
        'formula': 'softmax(logits / temperature)', 'fit_split': 'validation', 'fit_windows': len(val),
        'fit_dogs': sorted(map(int, val.dog_id.unique())), 'objective': 'unweighted mean multiclass NLL',
        'checkpoint_sha256': initial_sha, 'checkpoint': str(args.checkpoint), 'classes': predictor.classes,
        'window_manifest_sha256': sha(ROOT / 'data/splits/windows.csv'),
        'source_zip_sha256': protocol['audit']['csv_zip_sha256'],
        'fit_window_ids_sha256': hashlib.sha256('\n'.join(val.window_id).encode()).hexdigest(),
        'created_utc': datetime.now(timezone.utc).isoformat(), 'solver': solver,
        'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'torch']},
        'limitations': 'This validation set was already used for CNN hyperparameter/epoch selection. Metrics on the temperature fitting set are descriptive, not an independent calibration evaluation. Test labels were not used for fitting.'}
    (out / 'temperature.json').write_text(json.dumps(calibration, indent=2) + '\n')
    result = {'split': 'validation (temperature fitting set)', 'before': before, 'after': after,
              'temperature': temperature, 'cnn_weights_unchanged': True, 'argmax_unchanged': True,
              'ece_definition': 'Top-label ECE, 15 equal-width confidence bins, weighted by window count.',
              'brier_definition': 'Mean sum over seven classes of (probability - one_hot_label)^2.'}
    (out / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez(out / 'validation_logits.npz', logits=logits, labels=y, window_ids=val.window_id.to_numpy(str))
    raw = torch.from_numpy(logits).double().softmax(1).numpy()
    calibrated = (torch.from_numpy(logits).double() / temperature).softmax(1).numpy()
    table = val.copy()
    table['prediction'] = [predictor.classes[i] for i in logits.argmax(1)]
    table['confidence_before'], table['confidence_after'] = raw.max(1), calibrated.max(1)
    for c, label in enumerate(predictor.classes):
        key = label.lower().replace(' ', '_')
        table['probability_before_' + key] = raw[:, c]
        table['probability_after_' + key] = calibrated[:, c]
    table.to_csv(out / 'validation_probabilities.csv', index=False)
    lines = ['# CNN temperature calibration', '', f'T = {temperature:.10g}; fitted on all 4,727 validation windows (9 dogs).',
             'CNN weights are frozen. Test data and demo predictions do not select the temperature.', '',
             '| Validation fitting-set metric | Before | After |', '|---|---:|---:|']
    for key in ['nll', 'brier_multiclass_sum', 'ece_top_label_15_equal_width_bins', 'accuracy', 'mean_confidence']:
        lines.append(f'| {key} | {before[key]:.8f} | {after[key]:.8f} |')
    lines += ['', calibration['limitations'], '', 'Method: https://proceedings.mlr.press/v70/guo17a.html',
              '', 'Artifacts: temperature.json, metrics.json, validation_logits.npz, validation_probabilities.csv.']
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'temperature': temperature, 'before': {k: v for k, v in before.items() if k != 'reliability_bins'},
                      'after': {k: v for k, v in after.items() if k != 'reliability_bins'}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
