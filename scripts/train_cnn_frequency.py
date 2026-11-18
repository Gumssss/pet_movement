"""RQ3: compare time and spectral representations on the locked dog split."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pet_imu.data import CLASSES, ROOT, load_windows
from pet_imu.evaluation import bootstrap_f1, cluster_matrices, compare
from pet_imu.models.cnn import CNN
from pet_imu.training import batches, fit, save_json

SEEDS = (20261101, 20261102, 20261103)
RECIPE = {"batch_size": 128, "lr": 0.002, "weight_decay": 0.0001, "dropout": 0.2,
          "max_epochs": 35, "patience": 7, "min_delta": 1e-6}


def spectrum(raw):
    """Return signed DC and log amplitudes at 0.5–50 Hz for each channel."""
    dc = raw.mean(axis=1, keepdims=True)
    taper = np.hanning(raw.shape[1])
    fft = np.fft.rfft((raw - dc) * taper[:, None], axis=1)
    amplitude = np.abs(fft[:, 1:]) * (2.0 / taper.sum())
    amplitude[:, -1] *= 0.5  # Nyquist has no negative-frequency partner.
    return np.concatenate((dc, np.log1p(amplitude)), axis=1).astype(np.float32)


def normalized(signal, train_mask, axes):
    train = signal[train_mask]
    mean = train.mean(axis=axes, dtype=np.float64).astype(np.float32)
    std = train.std(axis=axes, dtype=np.float64).astype(np.float32).clip(1e-6)
    inputs = np.ascontiguousarray(((signal - mean) / std).transpose(0, 2, 1))
    return inputs, {"mean": mean.tolist(), "std": std.tolist()}


def summarize(test, labels, predictions, seed):
    dogs = test.dog_id.to_numpy()
    draws = np.random.default_rng(seed).integers(len(np.unique(dogs)), size=(2000, len(np.unique(dogs))))
    results = {}
    for name, reference in [
        ("time", "time"), ("frequency", "time"), ("time_time", "time"), ("time_frequency", "time_time")
    ]:
        per_seed, bootstrap = [], []
        for run in predictions:
            per_seed.append(compare(labels, run[name], dogs, run[reference], seed))
            bootstrap.append(
                bootstrap_f1(cluster_matrices(labels, run[name], dogs), draws)
                - bootstrap_f1(cluster_matrices(labels, run[reference], dogs), draws)
            )
        deltas = np.mean(bootstrap, axis=0)
        results[name] = {
            "reference": reference,
            "mean_macro_f1": float(np.mean([r["macro_f1"] for r in per_seed])),
            "mean_delta_macro_f1": float(np.mean([r["delta_macro_f1"] for r in per_seed])),
            "paired_dog_bootstrap_95ci": np.quantile(deltas, [0.025, 0.975]).tolist(),
            "per_seed": dict(zip(map(str, SEEDS), per_seed)),
        }
    return results


def run(output, device):
    windows, raw, protocol = load_windows()
    masks = {part: windows.split.eq(part).to_numpy() for part in ("train", "validation", "test")}
    labels = windows.label.map({name: i for i, name in enumerate(CLASSES)}).to_numpy()
    y_train = torch.from_numpy(labels[masks["train"]]).to(device)
    y_val = labels[masks["validation"]]
    output.mkdir(parents=True, exist_ok=False)
    spec = {
        "device": str(device), "torch": str(torch.__version__), "precision": "fp32",
        "seeds": SEEDS, "classes": CLASSES, "splits": protocol["splits"], "training": RECIPE,
        "window_manifest_sha256": hashlib.sha256((ROOT / "data/splits/windows.csv").read_bytes()).hexdigest(),
        "class_weight": "inverse square root of training frequency",
        "checkpoint_selection": "validation macro-F1, ties favor earlier epoch",
        "input": {"time": "100 Hz, 200 samples, training-fold channel z-score",
                  "frequency": "signed mean + Hann-tapered demeaned rFFT log(1+amplitude), 101 bins, "
                               "training-fold channel-by-bin z-score",
                  "ensembles": "mean of softmax probabilities; time+time uses cyclic next seed"},
    }
    save_json(output / "protocol.json", spec)
    validation = {}
    for name in ("time", "frequency"):
        signal, axes = (raw, (0, 1)) if name == "time" else (spectrum(raw), 0)
        x, stats = normalized(signal, masks["train"], axes)
        group = output / name
        group.mkdir()
        save_json(group / "normalization.json", stats)
        x_train = torch.from_numpy(x[masks["train"]]).to(device)
        x_val = torch.from_numpy(x[masks["validation"]]).to(device)
        validation[name] = {}
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = CNN(RECIPE["dropout"]).to(device)
            validation[name][str(seed)] = fit(
                model, (x_train, y_train), (x_val, y_val), {**RECIPE, "seed": seed},
                group / str(seed), RECIPE["max_epochs"], RECIPE["lr"], amp=False,
                patience=RECIPE["patience"], min_delta=RECIPE["min_delta"],
            )
    save_json(output / "validation.json", validation)

    # Open the test fold for evaluation after all six checkpoints are selected.
    test = windows.loc[masks["test"]].copy()
    probabilities = {}
    raw_test = raw[masks["test"]]
    for name, signal in (("time", raw_test), ("frequency", spectrum(raw_test))):
        group = output / name
        stats = json.loads((group / "normalization.json").read_text())
        x = (signal - np.asarray(stats["mean"], dtype=np.float32)) / np.asarray(stats["std"], dtype=np.float32)
        inputs = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 2, 1))).to(device)
        model = CNN(RECIPE["dropout"]).to(device)
        probabilities[name] = []
        for seed in SEEDS:
            model.load_state_dict(torch.load(group / str(seed) / "best.pt", map_location="cpu", weights_only=True))
            probabilities[name].append(batches(model, inputs, RECIPE["batch_size"], amp=False).softmax(1).cpu().numpy())

    predicted, tables = [], []
    for index, seed in enumerate(SEEDS):
        time = probabilities["time"][index]
        frequency = probabilities["frequency"][index]
        time_alternate = probabilities["time"][(index + 1) % len(SEEDS)]
        predictions = {"time": time, "frequency": frequency, "time_time": (time + time_alternate) / 2,
                       "time_frequency": (time + frequency) / 2}
        predicted.append({key: scores.argmax(axis=1) for key, scores in predictions.items()})
        tables.append(test.assign(seed=seed, **{key: np.asarray(CLASSES)[pred]
                                                for key, pred in predicted[-1].items()}))
    metrics = summarize(test, labels[masks["test"]], predicted, SEEDS[0])
    pd.concat(tables).to_csv(output / "predictions.csv", index=False)
    save_json(output / "metrics.json", metrics)
    for name, metric in metrics.items():
        print(name, metric["mean_macro_f1"], metric["paired_dog_bootstrap_95ci"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/RQ3/cnn_fft_v1")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    run(args.output, torch.device(device))
