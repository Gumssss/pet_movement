"""Test transfer of the validation-selected RQ2 augmentation to full InrtlWrist fine-tuning."""

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
from pet_imu.models.inrtlwrist import SHA256, classifier, encoder
from pet_imu.training import batches, fit, macro_f1, save_json
from pet_imu.transforms import normalization, resample_windows, standardize

RQ2 = ROOT / "outputs/RQ2"
BASELINE = "inrtlwrist_finetune"


def read(path):
    return json.loads(path.read_text())


def run(output, cnn_run, reference_run, device):
    selection = read(cnn_run / "selection.json")
    name = selection["test_comparison"]
    config = read(cnn_run / name / "config.json")
    cnn_spec = read(cnn_run / "protocol.json")
    operations = {op: config["operations"][op] for op in ORDER if op in config["operations"]}
    if not operations or operations.keys() != config["operations"].keys():
        raise ValueError("Expected a validation-selected CNN augmentation.")
    if cnn_spec["augmentation_gate"] != 0.5:
        raise ValueError("The CNN augmentation probability differs from the common implementation.")
    spec = read(reference_run / "protocol.json")
    recipe = spec["training"]
    seeds = spec["seeds"]
    lr = read(reference_run / "selection.json")["lr"]
    windows, raw, protocol = load_windows()
    fingerprint = hashlib.sha256((ROOT / "data/splits/windows.csv").read_bytes()).hexdigest()
    if (
        fingerprint != spec["manifest_sha256"]
        or fingerprint != cnn_spec["window_manifest_sha256"]
        or spec["splits"] != protocol["splits"]
        or cnn_spec["splits"] != protocol["splits"]
        or seeds != cnn_spec["seeds"]
        or spec["device"] != str(device)
        or spec["torch"] != str(torch.__version__)
        or spec["precision"] != "fp32"
        or spec["augmentation"]
        or spec["lr_schedule"] != "constant"
        or spec["warmup"]
        or spec["encoder_checkpoint_sha256"] != SHA256
        or not spec["encoder_trainable"]
    ):
        raise ValueError("CNN selection and InrtlWrist baseline must match the current experiment protocol.")
    if protocol["audit"]["stats"]["timestamp_breaks"] != 0:
        raise ValueError("Augmentation requires timestamp-contiguous source windows.")
    masks = {part: windows.split.eq(part).to_numpy() for part in ("train", "validation", "test")}
    labels = windows.label.map(CLASSES.index).to_numpy()
    train_raw = raw[masks["train"]]
    train_signal = resample_windows(train_raw)
    stats = normalization(train_signal, "train")
    if stats != spec["normalization"]:
        raise ValueError("Recomputed training normalization differs from RQ1.")
    sides = neighbors(windows.loc[masks["train"]])
    raw_std = np.asarray(cnn_spec["normalization"]["std"], dtype=np.float32)
    train_labels = labels[masks["train"]]
    validation_labels = labels[masks["validation"]]
    x = torch.from_numpy(standardize(train_signal, stats)).to(device)
    y = torch.from_numpy(train_labels).to(device)
    v = torch.from_numpy(standardize(resample_windows(raw[masks["validation"]]), stats)).to(device)

    output.mkdir(parents=True, exist_ok=False)
    save_json(
        output / "protocol.json",
        {
            "reference_run": str(reference_run),
            "cnn_selection_run": str(cnn_run),
            "cnn_selected": selection["selected"],
            "augmentation_name": name,
            "augmentation": config,
            "augmentation_gate": 0.5,
            "augmentation_stage": "raw 100 Hz, before resampling",
            "training": recipe,
            "seeds": seeds,
            "lr": lr,
            "lr_schedule": "constant",
            "encoder_trainable": True,
            "encoder_checkpoint_sha256": SHA256,
            "precision": spec["precision"],
            "device": str(device),
            "torch": str(torch.__version__),
            "manifest_sha256": fingerprint,
            "splits": protocol["splits"],
            "normalization": stats,
            "input": spec["input"],
            "class_weights": spec["class_weights"],
        },
    )

    def make_model(seed):
        torch.manual_seed(seed)
        return torch.nn.Sequential(encoder(), classifier(seed)).to(device)

    comparison = read(reference_run / "validation.json")[BASELINE]
    validation = {"baseline": comparison, "augmented": {}}
    for seed in seeds:
        aug = Augmenter(train_raw, train_labels, sides, raw_std, operations, config["mode"], seed)

        def epoch_inputs(aug=aug):
            inputs = np.empty_like(train_signal)
            for rows, batch in aug.epoch_batches():
                inputs[rows] = standardize(resample_windows(batch), stats)
            return torch.from_numpy(inputs).to(device)

        model = make_model(seed)
        validation["augmented"][str(seed)] = fit(
            model,
            (x, y),
            (v, validation_labels),
            {**recipe, "seed": seed},
            output / str(seed),
            recipe["max_epochs"],
            lr,
            False,
            patience=recipe["patience"],
            min_delta=recipe["min_delta"],
            epoch_inputs=epoch_inputs,
        )
        save_json(
            output / str(seed) / "coverage.json", {k: values.tolist() for k, values in aug.counts.items()}
        )
        save_json(output / "validation.json", validation)
        del model

    test_mask = masks["test"]
    test = windows.loc[test_mask, ["window_id", "dog_id", "label"]].reset_index(drop=True)
    test_x = torch.from_numpy(standardize(resample_windows(raw[test_mask]), stats)).to(device)
    test_y = labels[test_mask]
    dogs = test.dog_id.to_numpy()
    n_dogs = len(np.unique(dogs))
    draws = np.random.default_rng(0).integers(n_dogs, size=(2000, n_dogs))
    reference = pd.read_csv(reference_run / "predictions.csv")
    original_metrics = read(reference_run / "metrics.json")["per_seed"]
    paired, boot, frames = {}, [], []
    for seed in seeds:
        frame = reference.loc[reference.seed == seed].reset_index(drop=True)
        pd.testing.assert_frame_equal(frame[test.columns], test)
        baseline = frame[BASELINE].map(CLASSES.index).to_numpy()
        np.testing.assert_allclose(
            macro_f1(test_y, baseline), original_metrics[str(seed)][BASELINE]["macro_f1"], atol=1e-12, rtol=0
        )
        model = make_model(seed)
        model.load_state_dict(
            torch.load(output / str(seed) / "best.pt", map_location="cpu", weights_only=True)
        )
        val_pred = batches(model, v, recipe["batch_size"], False).argmax(1).cpu().numpy()
        np.testing.assert_allclose(
            macro_f1(validation_labels, val_pred),
            validation["augmented"][str(seed)]["validation_macro_f1"],
            atol=1e-12,
            rtol=0,
        )
        pred = batches(model, test_x, recipe["batch_size"], False).argmax(1).cpu().numpy()
        paired[str(seed)] = {
            arm: compare(test_y, p, dogs, baseline, 0)
            for arm, p in (("baseline", baseline), ("augmented", pred))
        }
        boot.append(
            bootstrap_f1(cluster_matrices(test_y, pred, dogs), draws)
            - bootstrap_f1(cluster_matrices(test_y, baseline, dogs), draws)
        )
        frames.append(test.assign(seed=seed, baseline=frame[BASELINE], augmented=np.array(CLASSES)[pred]))
    means = {
        arm: {
            metric: float(np.mean([r[arm][metric] for r in paired.values()]))
            for metric in ("macro_f1", "accuracy", "balanced_accuracy")
        }
        for arm in ("baseline", "augmented")
    }
    pd.concat(frames).to_csv(output / "predictions.csv", index=False)
    save_json(
        output / "metrics.json",
        {
            "test_dogs": n_dogs,
            "test_windows": len(test),
            "mean": means,
            "per_seed": paired,
            "delta_macro_f1": means["augmented"]["macro_f1"] - means["baseline"]["macro_f1"],
            "paired_dog_bootstrap_95ci": np.quantile(np.mean(boot, axis=0), [0.025, 0.975]).tolist(),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RQ2 / "inrtlwrist_augmentation_v1")
    parser.add_argument("--cnn-run", type=Path, default=RQ2 / "cnn_augmentation_v3")
    parser.add_argument("--reference-run", type=Path, default=ROOT / "outputs/RQ1/inrtlwrist_finetune_fp32")
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.output, args.cnn_run, args.reference_run, torch.device(args.device))
