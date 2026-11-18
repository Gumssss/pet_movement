"""Train three RQ3 models on paired temporal and spectral views of each window."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pet_imu.augmentation import ORDER, Augmenter, neighbors
from pet_imu.data import CLASSES, ROOT, load_windows
from pet_imu.evaluation import bootstrap_f1, cluster_matrices, compare
from pet_imu.models.cnn import CNN
from pet_imu.models.inrtlwrist import SHA256, encoder
from pet_imu.training import batches, fit, macro_f1, save_json
from pet_imu.transforms import normalization, resample_windows, standardize
from torch import nn
from train_cnn_frequency import spectrum

BASELINE = "inrtlwrist"
ARMS = ("frequency_cnn", "time_fusion", "frequency_fusion")
COMPARISONS = (
    ("frequency_fusion", BASELINE),
    ("frequency_fusion", "time_fusion"),
    ("time_fusion", BASELINE),
    ("frequency_cnn", BASELINE),
)
RQ3 = ROOT / "outputs/RQ3"


class Fusion(nn.Module):
    def __init__(self, dropout, spectral_branch):
        super().__init__()
        self.backbone = encoder()
        self.cnn = CNN(dropout).net[:-1]
        self.head = nn.Linear(384 + 96, len(CLASSES))
        self.spectral_branch = spectral_branch

    def forward(self, x):
        time_view = x[:, :, :96].contiguous()
        cnn_view = x[:, :, 96:].contiguous() if self.spectral_branch else time_view
        return self.head(torch.cat((self.backbone(time_view), self.cnn(cnn_view)), dim=1))


def read(path):
    return json.loads(path.read_text())


def run(output, reference_run, device):
    spec = read(reference_run / "protocol.json")
    cnn_selection = read(ROOT / "outputs/RQ1/lr_search_seed0/selection.json")
    cnn_recipe = read(ROOT / "outputs/RQ1/cnn_cosine_decay_v2/protocol.json")
    aug_recipe = read(ROOT / "outputs/RQ2/cnn_augmentation_v3/protocol.json")
    train_recipe = spec["training"]
    cnn_lr = cnn_selection["cnn"]["lr"]
    backbone_lr = spec["lr"]
    seeds = spec["seeds"]
    windows, raw, protocol = load_windows()
    fingerprint = hashlib.sha256((ROOT / "data/splits/windows.csv").read_bytes()).hexdigest()
    if (
        fingerprint != spec["manifest_sha256"]
        or fingerprint != aug_recipe["window_manifest_sha256"]
        or spec["splits"] != protocol["splits"]
        or spec["precision"] != "fp32"
        or spec["torch"] != str(torch.__version__)
        or spec["device"] != str(device)
        or spec["lr_schedule"] != "constant"
        or not spec["encoder_trainable"]
        or spec["encoder_checkpoint_sha256"] != SHA256
        or spec["augmentation_gate"] != 0.5
        or protocol["audit"]["stats"]["timestamp_breaks"] != 0
    ):
        raise ValueError("The RQ2 reference does not match the fusion experiment protocol.")
    config = spec["augmentation"]
    operations = {op: config["operations"][op] for op in ORDER if op in config["operations"]}
    if not operations or operations.keys() != config["operations"].keys():
        raise ValueError("Expected a valid augmentation recipe from RQ2.")
    masks = {name: windows.split.eq(name).to_numpy() for name in ("train", "validation", "test")}
    labels = windows.label.map(CLASSES.index).to_numpy()
    raw_train = raw[masks["train"]]
    train_labels = labels[masks["train"]]
    y_val = labels[masks["validation"]]
    y = torch.from_numpy(train_labels).to(device)
    train_signal = resample_windows(raw_train)
    val_signal = resample_windows(raw[masks["validation"]])
    time_stats = normalization(train_signal, "train")
    if time_stats != spec["normalization"]:
        raise ValueError("Time-domain normalization must reproduce RQ2.")
    clean_spectrum = spectrum(train_signal.transpose(0, 2, 1))
    freq_mean = clean_spectrum.mean(axis=0, dtype=np.float64).astype(np.float32)
    freq_std = clean_spectrum.std(axis=0, dtype=np.float64).astype(np.float32).clip(1e-6)
    raw_std = np.asarray(aug_recipe["normalization"]["std"], dtype=np.float32)
    sides = neighbors(windows.loc[masks["train"]])

    def features(signal, arm):
        if arm != "time_fusion":
            freq = spectrum(signal.transpose(0, 2, 1))
            freq = np.ascontiguousarray(((freq - freq_mean) / freq_std).transpose(0, 2, 1))
            if arm == "frequency_cnn":
                return freq
        time = standardize(signal, time_stats)
        if arm == "frequency_fusion":
            return np.concatenate((time, freq), axis=2)
        return time

    output.mkdir(parents=True, exist_ok=False)
    save_json(
        output / "protocol.json",
        {
            "reference": str(reference_run),
            "seeds": seeds,
            "training": train_recipe,
            "device": str(device),
            "precision": "fp32",
            "torch": str(torch.__version__),
            "manifest_sha256": fingerprint,
            "splits": protocol["splits"],
            "encoder_checkpoint_sha256": SHA256,
            "augmentation": config,
            "augmentation_gate": 0.5,
            "input": "same six-axis window, resampled to 50 Hz and center cropped to 96 samples",
            "spectrum": "signed DC and 48 Hann-tapered demeaned rFFT log magnitudes through 25 Hz",
            "spectrum_hz": np.fft.rfftfreq(96, d=1 / 50).tolist(),
            "normalization": {
                "time": time_stats,
                "frequency_mean": freq_mean.tolist(),
                "frequency_std": freq_std.tolist(),
            },
            "lr": {"frequency_cnn": cnn_lr, "fusion_cnn": cnn_lr, "fusion_encoder_and_head": backbone_lr},
            "lr_schedule": "constant",
            "cnn_dropout": cnn_recipe["dropout"],
            "class_weights": spec["class_weights"],
            "comparisons": COMPARISONS,
            "checkpoints": "clean validation macro-F1 before paired test evaluation",
        },
    )
    validation = {BASELINE: read(reference_run / "validation.json")["augmented"]}
    trained = {}
    batch_size = train_recipe["batch_size"]

    def model_for(arm, seed):
        torch.manual_seed(seed)
        dropout = cnn_recipe["dropout"]
        return (CNN(dropout) if arm == "frequency_cnn" else Fusion(dropout, arm == "frequency_fusion")).to(
            device
        )

    for arm in ARMS:
        (output / arm).mkdir()
        x = torch.from_numpy(features(train_signal, arm)).to(device)
        v = torch.from_numpy(features(val_signal, arm)).to(device)
        validation[arm] = {}
        trained[arm] = v
        for seed in seeds:
            aug = Augmenter(raw_train, train_labels, sides, raw_std, operations, config["mode"], seed)
            model = model_for(arm, seed)

            def epoch_inputs(aug=aug, arm=arm, shape=tuple(x.shape)):
                transformed = np.empty(shape, dtype=np.float32)
                for rows, batch in aug.epoch_batches():
                    transformed[rows] = features(resample_windows(batch), arm)
                return torch.from_numpy(transformed).to(device)

            lr = cnn_lr if arm == "frequency_cnn" else backbone_lr
            groups = None
            if isinstance(model, Fusion):
                groups = [
                    {"params": list(model.backbone.parameters()) + list(model.head.parameters())},
                    {"params": model.cnn.parameters(), "lr": cnn_lr},
                ]
            directory = output / arm / str(seed)
            print(f"{arm}, seed={seed}", flush=True)
            validation[arm][str(seed)] = fit(
                model,
                (x, y),
                (v, y_val),
                {**train_recipe, "seed": seed},
                directory,
                train_recipe["max_epochs"],
                lr,
                False,
                patience=train_recipe["patience"],
                min_delta=train_recipe["min_delta"],
                epoch_inputs=epoch_inputs,
                parameter_groups=groups,
            )
            save_json(directory / "coverage.json", {k: values.tolist() for k, values in aug.counts.items()})
            save_json(output / "validation.json", validation)
            del model

    test_mask = masks["test"]
    test = windows.loc[test_mask, ["window_id", "dog_id", "label"]].reset_index(drop=True)
    test_signal = resample_windows(raw[test_mask])
    test_views = {arm: torch.from_numpy(features(test_signal, arm)).to(device) for arm in ARMS}
    y_test = labels[test_mask]
    dogs = test.dog_id.to_numpy()
    n_dogs = len(np.unique(dogs))
    draws = np.random.default_rng(0).integers(n_dogs, size=(2000, n_dogs))
    reference = pd.read_csv(reference_run / "predictions.csv")
    reference_scores = read(reference_run / "metrics.json")["per_seed"]
    per_seed, tables = {}, []
    differences = {pair: [] for pair in COMPARISONS}
    for seed in seeds:
        frame = reference.loc[reference.seed == seed].reset_index(drop=True)
        pd.testing.assert_frame_equal(frame[test.columns], test)
        base = frame.augmented.map(CLASSES.index).to_numpy()
        predictions = {BASELINE: base}
        np.testing.assert_allclose(
            macro_f1(y_test, base), reference_scores[str(seed)]["augmented"]["macro_f1"], atol=1e-12, rtol=0
        )
        for arm in ARMS:
            model = model_for(arm, seed)
            model.load_state_dict(
                torch.load(output / arm / str(seed) / "best.pt", map_location="cpu", weights_only=True)
            )
            pred_val = batches(model, trained[arm], batch_size, False).argmax(1).cpu().numpy()
            np.testing.assert_allclose(
                macro_f1(y_val, pred_val),
                validation[arm][str(seed)]["validation_macro_f1"],
                atol=1e-12,
                rtol=0,
            )
            predictions[arm] = batches(model, test_views[arm], batch_size, False).argmax(1).cpu().numpy()
        per_seed[str(seed)] = {arm: compare(y_test, pred, dogs, base, 0) for arm, pred in predictions.items()}
        bootstrap = {
            arm: bootstrap_f1(cluster_matrices(y_test, pred, dogs), draws)
            for arm, pred in predictions.items()
        }
        for pair in COMPARISONS:
            differences[pair].append(bootstrap[pair[0]] - bootstrap[pair[1]])
        tables.append(
            test.assign(seed=seed, **{arm: np.array(CLASSES)[pred] for arm, pred in predictions.items()})
        )
    means = {
        arm: {
            metric: float(np.mean([scores[arm][metric] for scores in per_seed.values()]))
            for metric in ("macro_f1", "accuracy", "balanced_accuracy")
        }
        for arm in (BASELINE, *ARMS)
    }
    save_json(
        output / "metrics.json",
        {
            "mean": means,
            "per_seed": per_seed,
            "test_dogs": n_dogs,
            "test_windows": len(test),
            "paired": [
                {
                    "model": a,
                    "reference": b,
                    "delta_macro_f1": means[a]["macro_f1"] - means[b]["macro_f1"],
                    "paired_dog_bootstrap_95ci": np.quantile(
                        np.mean(differences[a, b], axis=0), [0.025, 0.975]
                    ).tolist(),
                }
                for a, b in COMPARISONS
            ],
        },
    )
    pd.concat(tables).to_csv(output / "predictions.csv", index=False)
    print(means, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RQ3 / "inrtlwrist_frequency_fusion_v1")
    parser.add_argument("--reference-run", type=Path, default=ROOT / "outputs/RQ2/inrtlwrist_augmentation_v1")
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.output, args.reference_run, torch.device(args.device))
