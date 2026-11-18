"""Compare matched posture and rhythm feature branches against RQ2 InrtlWrist."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pet_imu.augmentation import ORDER, Augmenter, neighbors
from pet_imu.data import CLASSES, ROOT, load_windows
from pet_imu.descriptors import (
    AUTOCORRELATION_LAGS,
    CROSS_CORRELATION_LAGS,
    MIN_PEAK,
    POSTURE_COLUMNS,
    RHYTHM_COLUMNS,
    posture,
    rhythm,
    rhythm_floor,
)
from pet_imu.evaluation import bootstrap_f1, cluster_matrices, compare
from pet_imu.models.inrtlwrist import SHA256, encoder
from pet_imu.training import batches, fit, macro_f1, save_json
from pet_imu.transforms import normalization, resample_windows, standardize
from torch import nn

ARMS = ("posture", "rhythm")
RQ3 = ROOT / "outputs/RQ3"
WINDOW_SHAPE = (6, 96)
SIGNAL_SIZE = 6 * 96


class DescriptorFusion(nn.Module):
    def __init__(self, feature_count):
        super().__init__()
        self.backbone = encoder()
        self.side = nn.Sequential(nn.Linear(feature_count, 32), nn.ReLU())
        self.head = nn.Linear(384 + 32, len(CLASSES))

    def forward(self, inputs):
        waveforms = inputs[:, :SIGNAL_SIZE].reshape(-1, *WINDOW_SHAPE).contiguous()
        extra = self.side(inputs[:, SIGNAL_SIZE:].contiguous())
        return self.head(torch.cat((self.backbone(waveforms), extra), dim=1))


def read(path):
    return json.loads(path.read_text())


def run(output, baseline_dir, device):
    spec = read(baseline_dir / "protocol.json")
    augmentation_source = read(ROOT / "outputs/RQ2/cnn_augmentation_v3/protocol.json")
    recipe = spec["training"]
    seeds = spec["seeds"]
    lr = spec["lr"]
    side_lr = read(ROOT / "outputs/RQ1/lr_search_seed0/selection.json")["cnn"]["lr"]

    windows, raw, protocol = load_windows()
    digest = hashlib.sha256((ROOT / "data/splits/windows.csv").read_bytes()).hexdigest()
    if (
        digest != spec["manifest_sha256"]
        or digest != augmentation_source["window_manifest_sha256"]
        or spec["splits"] != protocol["splits"]
        or spec["precision"] != "fp32"
        or spec["device"] != str(device)
        or spec["torch"] != str(torch.__version__)
        or spec["lr_schedule"] != "constant"
        or not spec["encoder_trainable"]
        or spec["encoder_checkpoint_sha256"] != SHA256
        or spec["augmentation_gate"] != 0.5
        or protocol["audit"]["stats"]["timestamp_breaks"] != 0
    ):
        raise ValueError("The RQ2 baseline does not match the descriptor experiment protocol.")
    augmentation = spec["augmentation"]
    operations = {
        name: augmentation["operations"][name] for name in ORDER if name in augmentation["operations"]
    }
    if not operations or operations.keys() != augmentation["operations"].keys():
        raise ValueError("Expected a valid augmentation configuration from RQ2.")
    masks = {p: windows.split.eq(p).to_numpy() for p in ("train", "validation", "test")}
    labels = windows.label.map(CLASSES.index).to_numpy()
    raw_train = raw[masks["train"]]
    train_labels = labels[masks["train"]]
    val_labels = labels[masks["validation"]]
    train_wave = resample_windows(raw_train)
    val_wave = resample_windows(raw[masks["validation"]])
    time_stats = normalization(train_wave, "train")
    if time_stats != spec["normalization"]:
        raise ValueError("Time normalization must match the RQ2 baseline.")
    floor = rhythm_floor(train_wave)

    def describe(wave, arm):
        return posture(wave) if arm == "posture" else rhythm(wave, floor)

    columns = {"posture": POSTURE_COLUMNS, "rhythm": RHYTHM_COLUMNS}
    if len(POSTURE_COLUMNS) != len(RHYTHM_COLUMNS):
        raise ValueError("The two branches must have matched parameter counts.")
    descriptor_stats = {}
    for arm in ARMS:
        values = describe(train_wave, arm)
        descriptor_stats[arm] = (
            values.mean(0, dtype=np.float64).astype(np.float32),
            values.std(0, dtype=np.float64).astype(np.float32).clip(1e-6),
        )

    def features(wave, arm):
        mean, std = descriptor_stats[arm]
        coefficient = (describe(wave, arm) - mean) / std
        time = standardize(wave, time_stats).reshape(len(wave), SIGNAL_SIZE)
        return np.concatenate((time, coefficient), axis=1)

    def make_model(seed):
        torch.manual_seed(seed)
        return DescriptorFusion(len(POSTURE_COLUMNS)).to(device)

    output.mkdir(parents=True, exist_ok=False)
    save_json(
        output / "protocol.json",
        {
            "reference": str(baseline_dir),
            "manifest_sha256": digest,
            "splits": spec["splits"],
            "training": recipe,
            "seeds": seeds,
            "optimizer": "AdamW",
            "precision": "fp32",
            "torch": str(torch.__version__),
            "device": str(device),
            "encoder_checkpoint_sha256": SHA256,
            "augmentation": augmentation,
            "augmentation_gate": spec["augmentation_gate"],
            "input": "same resampled and centered 50 Hz, 96-sample six-axis window for both branches",
            "lr": {"encoder_and_head": lr, "side": side_lr},
            "lr_schedule": "constant",
            "columns": columns,
            "rhythm_min_peak": MIN_PEAK,
            "rhythm_floor_training_percentile": 10,
            "rhythm_floor": floor.tolist(),
            "autocorrelation_lags": AUTOCORRELATION_LAGS.tolist(),
            "cross_correlation_lags": CROSS_CORRELATION_LAGS.tolist(),
            "normalization": {
                "time": time_stats,
                **{a: {"mean": m.tolist(), "std": s.tolist()} for a, (m, s) in descriptor_stats.items()},
            },
        },
    )
    y = torch.from_numpy(train_labels).to(device)
    validation = {"baseline": read(baseline_dir / "validation.json")["augmented"]}
    val_views = {}
    sides = neighbors(windows.loc[masks["train"]])
    raw_std = np.asarray(augmentation_source["normalization"]["std"], dtype=np.float32)
    for arm in ARMS:
        (output / arm).mkdir()
        x = torch.from_numpy(features(train_wave, arm)).to(device)
        v = torch.from_numpy(features(val_wave, arm)).to(device)
        val_views[arm] = v
        validation[arm] = {}
        for seed in seeds:
            aug = Augmenter(raw_train, train_labels, sides, raw_std, operations, augmentation["mode"], seed)

            def epoch_inputs(aug=aug, arm=arm, shape=tuple(x.shape)):
                result = np.empty(shape, dtype=np.float32)
                for rows, windows_batch in aug.epoch_batches():
                    result[rows] = features(resample_windows(windows_batch), arm)
                return torch.from_numpy(result).to(device)

            model = make_model(seed)
            groups = [
                {"params": list(model.backbone.parameters()) + list(model.head.parameters())},
                {"params": model.side.parameters(), "lr": side_lr},
            ]
            destination = output / arm / str(seed)
            print(f"{arm}, seed={seed}", flush=True)
            validation[arm][str(seed)] = fit(
                model,
                (x, y),
                (v, val_labels),
                {**recipe, "seed": seed},
                destination,
                recipe["max_epochs"],
                lr,
                False,
                patience=recipe["patience"],
                min_delta=recipe["min_delta"],
                epoch_inputs=epoch_inputs,
                parameter_groups=groups,
            )
            save_json(destination / "coverage.json", {k: v.tolist() for k, v in aug.counts.items()})
            save_json(output / "validation.json", validation)
            del model

    test_mask = masks["test"]
    test = windows.loc[test_mask, ["window_id", "dog_id", "label"]].reset_index(drop=True)
    test_wave = resample_windows(raw[test_mask])
    test_views = {arm: torch.from_numpy(features(test_wave, arm)).to(device) for arm in ARMS}
    test_labels = labels[test_mask]
    dogs = test.dog_id.to_numpy()
    dog_count = test.dog_id.nunique()
    draws = np.random.default_rng(0).integers(dog_count, size=(2000, dog_count))
    reference = pd.read_csv(baseline_dir / "predictions.csv")
    reference_scores = read(baseline_dir / "metrics.json")["per_seed"]
    per_seed, tables = {}, []
    differences = {arm: [] for arm in ARMS}
    for seed in seeds:
        frame = reference.loc[reference.seed == seed].reset_index(drop=True)
        pd.testing.assert_frame_equal(frame[test.columns], test)
        baseline = frame.augmented.map(CLASSES.index).to_numpy()
        np.testing.assert_allclose(
            macro_f1(test_labels, baseline),
            reference_scores[str(seed)]["augmented"]["macro_f1"],
            atol=1e-12,
            rtol=0,
        )
        preds = {"baseline": baseline}
        base_boot = bootstrap_f1(cluster_matrices(test_labels, baseline, dogs), draws)
        for arm in ARMS:
            model = make_model(seed)
            model.load_state_dict(
                torch.load(output / arm / str(seed) / "best.pt", map_location="cpu", weights_only=True)
            )
            val_preds = batches(model, val_views[arm], recipe["batch_size"], False).argmax(1).cpu().numpy()
            np.testing.assert_allclose(
                macro_f1(val_labels, val_preds),
                validation[arm][str(seed)]["validation_macro_f1"],
                atol=1e-12,
                rtol=0,
            )
            preds[arm] = batches(model, test_views[arm], recipe["batch_size"], False).argmax(1).cpu().numpy()
            differences[arm].append(
                bootstrap_f1(cluster_matrices(test_labels, preds[arm], dogs), draws) - base_boot
            )
        per_seed[str(seed)] = {
            arm: compare(test_labels, pred, dogs, baseline, 0) for arm, pred in preds.items()
        }
        tables.append(test.assign(seed=seed, **{a: np.asarray(CLASSES)[p] for a, p in preds.items()}))

    means = {
        arm: {
            metric: float(np.mean([r[arm][metric] for r in per_seed.values()]))
            for metric in ("macro_f1", "accuracy", "balanced_accuracy")
        }
        for arm in ("baseline", *ARMS)
    }
    pd.concat(tables).to_csv(output / "predictions.csv", index=False)
    save_json(
        output / "metrics.json",
        {
            "test_dogs": dog_count,
            "test_windows": len(test),
            "mean": means,
            "per_seed": per_seed,
            "paired_vs_baseline": {
                arm: {
                    "delta_macro_f1": means[arm]["macro_f1"] - means["baseline"]["macro_f1"],
                    "paired_dog_bootstrap_95ci": np.quantile(np.mean(diff, axis=0), [0.025, 0.975]).tolist(),
                }
                for arm, diff in differences.items()
            },
        },
    )
    print(means, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RQ3 / "inrtlwrist_descriptor_fusion_v1")
    parser.add_argument("--baseline", type=Path, default=ROOT / "outputs/RQ2/inrtlwrist_augmentation_v1")
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.output, args.baseline, torch.device(args.device))
