"""Create ten fixed test-window plots and run the saved CNN without training."""
import argparse
import html
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pet_imu.paths import CNN_CHECKPOINTS
from run_cnn import CHANNELS, ROOT, Predictor, sha, read_input
from render_imu import figure, render


def select_windows(windows, classes):
    test = windows[windows.split.eq('test')]
    rng = np.random.default_rng(20261113)
    chosen = [int(rng.choice(test.index[test.label.eq(label)])) for label in classes]
    chosen.extend(map(int, rng.choice(test.index.difference(chosen), size=3, replace=False)))
    return windows.loc[chosen].reset_index(drop=True)


def extract_signals(selected, archive, classes):
    pieces = [[] for _ in range(len(selected))]
    labels = ['Behavior_1', 'Behavior_2', 'Behavior_3']
    with zipfile.ZipFile(archive) as z, z.open('DogMoveData.csv') as source:
        cols = ['DogID', 'TestNum', 't_sec'] + CHANNELS + labels
        for chunk in pd.read_csv(source, usecols=cols, chunksize=200000):
            for i, row in selected.iterrows():
                mask = (chunk.DogID.eq(row.dog_id) & chunk.TestNum.eq(row.session)
                        & chunk.t_sec.ge(row.start_sec - 1e-7)
                        & chunk.t_sec.lt(row.end_exclusive_sec - 1e-7))
                if mask.any():
                    pieces[i].append(chunk.loc[mask].copy())
    signals = []
    for i, row in selected.iterrows():
        if not pieces[i]:
            raise ValueError(f'Window not found: {row.window_id}')
        frame = pd.concat(pieces[i], ignore_index=True)
        if len(frame) != 200 or not np.allclose(frame.t_sec, row.start_sec + np.arange(200) / 100, rtol=0, atol=1e-6):
            raise ValueError(f'Window timing mismatch: {row.window_id}')
        annotations = frame[labels].fillna('').to_numpy(str)
        hit = np.stack([(annotations == c).any(1) for c in classes], axis=1)
        if not (hit.sum(1) == 1).all() or not (annotations == row.label).any(1).all():
            raise ValueError(f'Window annotation mismatch: {row.window_id}')
        signals.append(frame[CHANNELS].to_numpy(np.float32))
    return np.stack(signals)


def write_report(out, records, correct_only=False):
    """Export all seven probabilities, not just the winning class."""
    rows, cards = [], []
    for i, record in enumerate(records, 1):
        probs = record['probabilities']
        ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        flat = {k: v for k, v in record.items() if k != 'probabilities' and not k.startswith('uncalibrated_')}
        flat.update({'probability_' + label.lower().replace(' ', '_'): value for label, value in probs.items()})
        flat.update(second_prediction=ranked[1][0], second_probability=ranked[1][1],
                    top_two_margin=ranked[0][1] - ranked[1][1], true_label_probability=probs[record['label']])
        rows.append(flat)
        table = ''.join(f'<tr><td>{html.escape(label)}'
                        f'{"（预测）" if label == record["prediction"] else ""}'
                        f'{"（真实）" if label == record["label"] else ""}</td>'
                        f'<td><progress max="1" value="{value}"></progress></td>'
                        f'<td title="{value!r}">{value * 100:.6g}%</td></tr>' for label, value in ranked)
        image, signal = html.escape(record['image']), html.escape(record['signal'])
        cards.append(f'<article><a href="{image}"><img src="{image}" alt="Six IMU channels"></a>'
                     f'<h2>Sample {i:02d}</h2><p>真实标签：{html.escape(record["label"])}<br>'
                     f'CNN 预测：{html.escape(record["prediction"])}</p>'
                     f'<table><thead><tr><th>类别</th><th>概率</th><th>数值</th></tr></thead><tbody>{table}</tbody></table>'
                     f'<p>前两类概率差：{flat["top_two_margin"] * 100:.4f} 个百分点</p>'
                     f'<a href="{signal}">原始 IMU CSV</a></article>')
    pd.DataFrame(rows).to_csv(out / 'predictions.csv', index=False)
    note = ('本组专门筛选 CNN 预测正确的样本，覆盖七类行为，仅作功能展示，10/10 不代表模型整体准确率。'
            if correct_only else '样本覆盖七类行为，未按预测结果挑选。')
    calibrated = records and 'uncalibrated_probabilities' in records[0]
    probability_note = (f'以下展示校准后的概率，使用验证集 temperature scaling（T={records[0]["temperature"]:.6g}）。'
                        '温度校准保持预测类别不变，也不保证单个样本的实际正确率。' if calibrated else
                        '七类概率为 softmax 输出，未校准，不能直接理解为实际正确率。')
    (out / 'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>CNN 犬只行为 Demo</title>'
        '<style>body{font:16px system-ui;background:#f4f6fa;color:#172334;max-width:1400px;margin:30px auto;padding:20px}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:20px}'
        'article{background:white;padding:16px;border-radius:12px;min-width:0}img{width:100%}h2{font-size:20px}'
        'table{width:100%;font-size:13px;border-collapse:collapse}th,td{text-align:left;padding:6px 2px;border-bottom:1px solid #eee}'
        'progress{width:65px;accent-color:#2563eb}td:last-child{font-variant-numeric:tabular-nums;white-space:nowrap}'
        '@media(max-width:420px){main{grid-template-columns:1fr}body{padding:10px}}</style>'
        '<h1>CNN 犬只行为识别 · 10 个样本</h1><p>每张图显示 2 秒、100 Hz 的六路 IMU 信号。'
        '已有 1D CNN 使用对应的原始数值预测；图片用于展示，不直接输入 CNN。</p><p>' + note + '</p>'
        '<p>使用训练好的 cnn_best.pt；' + probability_note +
        '显示六位有效数字，CSV 保留完整精度。</p><p><a href="predictions.csv">下载全部样本和七类概率 CSV</a></p>'
        '<main>' + ''.join(cards) + '</main></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'demo/cnn')
    parser.add_argument('--checkpoint', type=Path, default=CNN_CHECKPOINTS / "cnn_best.pt")
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--correct-only', action='store_true', help='Select correctly predicted examples for presentation only')
    parser.add_argument('--refresh-report', action='store_true', help='Update HTML/CSV from an existing manifest without inference')
    parser.add_argument('--calibration', type=Path, help='Apply saved temperature to demo probabilities')
    args = parser.parse_args()
    out = args.output_dir
    if args.refresh_report:
        manifest = json.loads((out / 'manifest.json').read_text())
        if args.calibration:
            torch.set_num_threads(2)
            predictor = Predictor(args.checkpoint, args.device, args.calibration)
            if sha(args.checkpoint) != manifest['checkpoint_sha256']:
                parser.error('Demo checkpoint differs from calibration checkpoint.')
            for i, record in enumerate(manifest['samples'], 1):
                signal, _ = read_input(out / record['image'])
                prediction = predictor.predict(signal)[0]
                if prediction['prediction'] != record['prediction']:
                    raise ValueError('Temperature changed an existing predicted label.')
                record.update(prediction)
                (out / f'sample_{i:02d}.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
            manifest['calibration'] = str(args.calibration)
            manifest['temperature'] = predictor.temperature
            (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        write_report(out, manifest['samples'], manifest.get('correct_only', False))
        print(f'Updated probabilities report: {out / "index.html"}')
        return
    if out.exists():
        parser.error('Output directory already exists; choose a new --output-dir.')
    if args.calibration:
        parser.error('First create the demo, then apply --calibration together with --refresh-report.')
    torch.set_num_threads(2)
    predictor = Predictor(args.checkpoint, args.device)
    archive = ROOT / 'data/犬只运动传感器数据集/DogMoveData_csv_format.zip'
    expected = json.loads((ROOT / 'outputs/protocol.json').read_text())['audit']['csv_zip_sha256']
    print('Checking source archive...', flush=True)
    if sha(archive) != expected:
        raise ValueError('Source archive differs from historical experiment.')
    windows = pd.read_csv(ROOT / 'data/splits/windows.csv')
    selected = select_windows(windows, predictor.classes)
    target_labels = selected.label.tolist()
    if args.correct_only:
        rng = np.random.default_rng(20261113)
        extra = []
        for label in predictor.classes:
            indices = windows.index[windows.split.eq('test') & windows.label.eq(label)
                                    & ~windows.window_id.isin(selected.window_id)]
            extra.extend(map(int, rng.choice(indices, size=min(20, len(indices)), replace=False)))
        pool = pd.concat([selected, windows.loc[extra]], ignore_index=True).drop_duplicates('window_id').reset_index(drop=True)
        selected = pool
    print(f'Reading {len(selected)} candidate windows from source CSV...', flush=True)
    signals = extract_signals(selected, archive, predictor.classes)
    predictions = predictor.predict(signals)
    candidate_count = len(selected)
    if args.correct_only:
        available = [i for i, (label, pred) in enumerate(zip(selected.label, predictions)) if label == pred['prediction']]
        chosen = []
        for label in target_labels:
            match = next((i for i in available if selected.iloc[i].label == label), None)
            if match is None:
                raise ValueError(f'Not enough correctly predicted candidates for {label}; no output written.')
            chosen.append(match)
            available.remove(match)
        selected = selected.iloc[chosen].reset_index(drop=True)
        signals = signals[chosen]
        predictions = [predictions[i] for i in chosen]
    historical = pd.read_csv(ROOT / 'outputs/RQ1/baselines/cnn/predictions.csv').set_index('window_id')
    for row, prediction in zip(selected.itertuples(), predictions):
        old = historical.loc[row.window_id]
        if prediction['prediction'] != old.cnn_best_prediction or abs(prediction['confidence'] - old.cnn_best_confidence) > 1e-5:
            raise ValueError(f'Prediction differs from historical CNN: {row.window_id}')
    protocol = json.loads((ROOT / 'data/vlm/render_protocol.json').read_text())
    out.mkdir(parents=True)
    fig, lines, points = figure(protocol)
    records = []
    for i, (row, signal, prediction) in enumerate(zip(selected.to_dict('records'), signals, predictions), 1):
        stem = out / f'sample_{i:02d}'
        image, csv = stem.with_suffix('.webp'), stem.with_suffix('.csv')
        pd.DataFrame(signal, columns=CHANNELS).to_csv(csv, index=False)
        render(signal, image, protocol, fig, lines, points)
        record = {**row, **prediction, 'image': image.name, 'signal': csv.name,
                  'image_sha256': sha(image), 'signal_sha256': sha(csv)}
        stem.with_suffix('.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        # Verify the image entry resolves to the same signal and prediction.
        reloaded, _ = read_input(image)
        np.testing.assert_array_equal(reloaded, signal)
        assert predictor.predict(reloaded)[0]['prediction'] == prediction['prediction']
        records.append(record)
    plt.close(fig)
    (out / 'manifest.json').write_text(json.dumps({'device': args.device, 'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': sha(args.checkpoint), 'source_zip_sha256': expected,
        'selection': ('Seed 20261113; seeded test candidate pool; keep correct predictions with class counts matching the original ten-window demo.'
                      if args.correct_only else 'Seed 20261113; one test window per class plus three additional test windows; independent of predictions.'),
        'correct_only': args.correct_only, 'candidate_count': candidate_count,
        'inference_input': 'Original 200 x 6 IMU values, normalized using checkpoint; image pixels are display only.',
        'historical_predictions_verified': True, 'samples': records}, ensure_ascii=False, indent=2) + '\n')
    write_report(out, records, args.correct_only)
    print(f'Created {len(records)} images and verified CNN predictions: {out / "index.html"}', flush=True)


if __name__ == '__main__':
    main()
