"""Train the dog-only CNN baseline at the same 50 Hz input as InrtlWrist."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pet_imu.augmentation import ORDER, Augmenter, neighbors
from pet_imu.data import CHANNELS, CLASSES, ROOT, load_windows
from pet_imu.evaluation import bootstrap_f1, cluster_matrices, compare
from pet_imu.models.cnn import CNN
from pet_imu.training import batches, fit, macro_f1, save_json
from pet_imu.transforms import (
    MODEL_SAMPLE_RATE_HZ,
    MODEL_WINDOW_SAMPLES,
    normalization,
    resample_windows,
    standardize,
)

REFERENCE = ROOT / "outputs/RQ2/cnn_augmentation_v3"
INRTLWRIST_REFERENCE = ROOT / "outputs/RQ2/inrtlwrist_augmentation_v1"


def read(path):
    return json.loads(path.read_text())


def run(output, device):
    spec = read(REFERENCE / "protocol.json")
    variant = read(REFERENCE / "selection.json")["test_comparison"]
    augmentation = read(REFERENCE / variant / "config.json")
    wrist = read(INRTLWRIST_REFERENCE / "protocol.json")
    recipe, seeds = spec["training"], spec["seeds"]
    windows, raw, protocol = load_windows()
    digest = hashlib.sha256((ROOT / "data/splits/windows.csv").read_bytes()).hexdigest()
    if (
        digest != spec["window_manifest_sha256"]
        or digest != wrist["manifest_sha256"]
        or spec["splits"] != protocol["splits"]
        or wrist["splits"] != protocol["splits"]
        or augmentation != wrist["augmentation"]
        or spec["augmentation_gate"] != wrist["augmentation_gate"]
        or seeds != wrist["seeds"]
        or protocol["audit"]["stats"]["timestamp_breaks"] != 0
    ):
        raise ValueError("The dog window manifest, augmentation and seeds must match both references.")

    operations = {op: augmentation["operations"][op] for op in ORDER if op in augmentation["operations"]}
    masks = {part: windows.split.eq(part).to_numpy() for part in ("train", "validation", "test")}
    labels = windows.label.map(CLASSES.index).to_numpy()
    raw_train = raw[masks["train"]]
    train_labels = labels[masks["train"]]
    val_labels = labels[masks["validation"]]
    sides = neighbors(windows.loc[masks["train"]])
    raw_std = np.asarray(spec["normalization"]["std"], dtype=np.float32)
    wave = resample_windows(raw_train)
    stats = normalization(wave, "train")
    if stats != wrist["normalization"]:
        raise ValueError("The 50 Hz train-only normalization must match the InrtlWrist reference.")
    x_train = torch.from_numpy(standardize(wave, stats)).to(device)
    x_val = torch.from_numpy(standardize(resample_windows(raw[masks["validation"]]), stats)).to(device)
    y_train = torch.from_numpy(train_labels).to(device)

    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "protocol.json", {
        "cnn_100hz_reference": str(REFERENCE.relative_to(ROOT)),
        "cnn_100hz_variant": variant,
        "inrtlwrist_50hz_reference": str(INRTLWRIST_REFERENCE.relative_to(ROOT)),
        "device": str(device), "torch": str(torch.__version__), "precision": "fp32",
        "optimizer": "AdamW", "lr_schedule": "constant", "training": recipe, "seeds": seeds,
        "splits": protocol["splits"], "window_manifest_sha256": digest,
        "sample_rate_hz": MODEL_SAMPLE_RATE_HZ, "window_samples": MODEL_WINDOW_SAMPLES,
        "channels": CHANNELS, "classes": CLASSES, "normalization": stats,
        "augmentation": augmentation, "augmentation_gate": spec["augmentation_gate"],
        "augmentation_stage": "train only, at native rate before anti-aliased resampling and center crop",
        "checkpoint_selection": "validation macro-F1 every epoch",
    })
    validation = {}
    for seed in seeds:
        torch.manual_seed(seed)
        model = CNN(recipe["dropout"]).to(device)
        aug = Augmenter(raw_train, train_labels, sides, raw_std, operations, augmentation["mode"], seed)

        def epoch_inputs(aug=aug):
            values = np.empty_like(wave)
            for rows, samples in aug.epoch_batches():
                values[rows] = standardize(resample_windows(samples), stats)
            return torch.from_numpy(values).to(device)

        folder = output / str(seed)
        validation[str(seed)] = fit(
            model, (x_train, y_train), (x_val, val_labels), {**recipe, "seed": seed},
            folder, recipe["max_epochs"], recipe["lr"], amp=False,
            patience=recipe["patience"], min_delta=recipe["min_delta"], epoch_inputs=epoch_inputs,
        )
        save_json(folder / "coverage.json", {k: v.tolist() for k, v in aug.counts.items()})
        save_json(output / "validation.json", validation)

    test = windows.loc[masks["test"], ["window_id", "dog_id", "label"]].reset_index(drop=True)
    x_test = torch.from_numpy(standardize(resample_windows(raw[masks["test"]]), stats)).to(device)
    y_test, dogs = labels[masks["test"]], test.dog_id.to_numpy()
    n_dogs = test.dog_id.nunique()
    draws = np.random.default_rng(0).integers(n_dogs, size=(2000, n_dogs))
    reference = pd.read_csv(REFERENCE / "predictions.csv")
    reference_scores = read(REFERENCE / "metrics.json")["per_seed"]
    per_seed, predictions, deltas = {}, [], []
    for seed in seeds:
        model.load_state_dict(torch.load(output / str(seed) / "best.pt", map_location="cpu", weights_only=True))
        pred = batches(model, x_test, recipe["batch_size"], amp=False).argmax(1).cpu().numpy()
        paired = reference.loc[reference.seed.eq(seed)].reset_index(drop=True)
        pd.testing.assert_frame_equal(paired[test.columns], test)
        old = paired[variant].map(CLASSES.index).to_numpy()
        np.testing.assert_allclose(macro_f1(y_test, old), reference_scores[str(seed)][variant]["macro_f1"])
        per_seed[str(seed)] = compare(y_test, pred, dogs, old, 0)
        deltas.append(bootstrap_f1(cluster_matrices(y_test, pred, dogs), draws)
                      - bootstrap_f1(cluster_matrices(y_test, old, dogs), draws))
        predictions.append(test.assign(seed=seed, cnn_50hz=np.asarray(CLASSES)[pred], cnn_100hz=paired[variant]))
    means = {k: float(np.mean([r[k] for r in per_seed.values()]))
             for k in ("macro_f1", "accuracy", "balanced_accuracy", "delta_macro_f1")}
    pd.concat(predictions, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    save_json(output / "metrics.json", {
        "test_dogs": n_dogs, "test_windows": len(test), "mean": means, "per_seed": per_seed,
        "cnn_100hz_paired_dog_bootstrap_95ci": np.quantile(np.mean(deltas, axis=0), [0.025, 0.975]).tolist(),
    })
    print(means, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/RQ4/cnn_50hz_baseline_v1")
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.output, torch.device(args.device))
