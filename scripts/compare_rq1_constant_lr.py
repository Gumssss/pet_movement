"""Compare the frozen InrtlWrist linear probe with the matched constant-LR CNN."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pet_imu.data import CLASSES, ROOT, load_windows
from pet_imu.evaluation import bootstrap_f1, cluster_matrices, compare
from pet_imu.models.inrtlwrist import SHA256, classifier, encoder
from pet_imu.training import batches, fit, macro_f1, save_json
from pet_imu.transforms import normalization, resample_windows, standardize

RQ1 = ROOT / "outputs/RQ1"


def run(output, cnn_run, lr_selection, device):
    spec = json.loads((cnn_run / "protocol.json").read_text())
    cnn_validation = json.loads((cnn_run / "validation.json").read_text())["constant"]
    selected = json.loads(lr_selection.read_text())
    windows, raw, protocol = load_windows()
    digest = hashlib.sha256((ROOT / "data/splits/windows.csv").read_bytes()).hexdigest()
    if (
        spec["manifest_sha256"] != digest
        or spec["splits"] != protocol["splits"]
        or spec["augmentation"]
        or spec["warmup"]
        or spec["precision"] != "fp32"
        or spec["device"] != str(device)
        or spec["torch"] != str(torch.__version__)
        or spec["initial_lr"] != selected["cnn"]["lr"]
    ):
        raise ValueError("CNN run does not match the constant-LR comparison protocol.")

    recipe = spec["training"]
    seeds = spec["seeds"]
    lr = selected["inrtlwrist_head"]["lr"]
    train = windows.split.eq("train").to_numpy()
    val = windows.split.eq("validation").to_numpy()
    labels = windows.label.map(CLASSES.index).to_numpy()
    train_signal = resample_windows(raw[train])
    stats = normalization(train_signal, "train")
    batch_size = recipe["batch_size"]
    backbone = encoder().requires_grad_(False).to(device).eval()

    def features(signal):
        inputs = torch.from_numpy(standardize(signal, stats)).to(device)
        return batches(backbone, inputs, batch_size, False)

    x = features(train_signal)
    v = features(resample_windows(raw[val]))
    y = torch.from_numpy(labels[train]).to(device)
    output.mkdir(parents=True, exist_ok=False)
    save_json(
        output / "protocol.json",
        {
            "cnn_run": str(cnn_run),
            "lr_selection": str(lr_selection),
            "training": recipe,
            "seeds": seeds,
            "precision": spec["precision"],
            "device": str(device),
            "torch": spec["torch"],
            "manifest_sha256": digest,
            "splits": spec["splits"],
            "constant_lr": {"cnn": selected["cnn"]["lr"], "inrtlwrist_head": lr},
            "encoder_checkpoint_sha256": SHA256,
            "normalization": stats,
            "input": "Anti-aliased 50 Hz, center 96 samples",
            "class_weights": spec["class_weights"],
            "augmentation": False,
        },
    )
    validation = {}
    for seed in seeds:
        validation[str(seed)] = fit(
            classifier(seed).to(device),
            (x, y),
            (v, labels[val]),
            {**recipe, "seed": seed},
            output / str(seed),
            recipe["max_epochs"],
            lr,
            False,
            patience=recipe["patience"],
            min_delta=recipe["min_delta"],
        )
    save_json(output / "validation.json", {"cnn": cnn_validation, "inrtlwrist_head": validation})

    test_mask = windows.split.eq("test").to_numpy()
    test = windows.loc[test_mask, ["window_id", "dog_id", "label"]].reset_index(drop=True)
    test_x = features(resample_windows(raw[test_mask]))
    y_test = labels[test_mask]
    dogs = test.dog_id.to_numpy()
    cnn_predictions = pd.read_csv(cnn_run / "predictions.csv")
    cnn_metrics = json.loads((cnn_run / "metrics.json").read_text())["per_seed"]["constant"]
    n_dogs = len(np.unique(dogs))
    draws = np.random.default_rng(0).integers(n_dogs, size=(2000, n_dogs))
    per_seed = {}
    boot_deltas, tables = [], []
    for seed in seeds:
        frame = cnn_predictions.loc[cnn_predictions.seed == seed].reset_index(drop=True)
        pd.testing.assert_frame_equal(frame[test.columns], test)
        baseline = frame.constant.map(CLASSES.index).to_numpy()
        np.testing.assert_allclose(
            macro_f1(y_test, baseline), cnn_metrics[str(seed)]["macro_f1"], atol=1e-12, rtol=0
        )
        head = classifier(seed).to(device)
        head.load_state_dict(
            torch.load(output / str(seed) / "best.pt", map_location="cpu", weights_only=True)
        )
        val_prediction = batches(head, v, batch_size, False).argmax(1).cpu().numpy()
        np.testing.assert_allclose(
            macro_f1(labels[val], val_prediction),
            validation[str(seed)]["validation_macro_f1"],
            atol=1e-12,
            rtol=0,
        )
        pred = batches(head, test_x, batch_size, False).argmax(1).cpu().numpy()
        per_seed[str(seed)] = {
            "cnn": cnn_metrics[str(seed)],
            "inrtlwrist_head": compare(y_test, pred, dogs, baseline, 0),
        }
        head_boot = bootstrap_f1(cluster_matrices(y_test, pred, dogs), draws)
        cnn_boot = bootstrap_f1(cluster_matrices(y_test, baseline, dogs), draws)
        boot_deltas.append(head_boot - cnn_boot)
        tables.append(test.assign(seed=seed, cnn=frame.constant, inrtlwrist_head=np.asarray(CLASSES)[pred]))
    means = {
        name: {
            metric: float(np.mean([r[name][metric] for r in per_seed.values()]))
            for metric in ("macro_f1", "accuracy", "balanced_accuracy")
        }
        for name in ("cnn", "inrtlwrist_head")
    }
    delta = means["inrtlwrist_head"]["macro_f1"] - means["cnn"]["macro_f1"]
    interval = np.quantile(np.mean(boot_deltas, axis=0), [0.025, 0.975]).tolist()
    pd.concat(tables).to_csv(output / "predictions.csv", index=False)
    save_json(
        output / "metrics.json",
        {
            "test_dogs": n_dogs,
            "test_windows": len(test),
            "mean": means,
            "head_minus_cnn": delta,
            "paired_dog_bootstrap_95ci": interval,
            "per_seed": per_seed,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RQ1 / "constant_lr_comparison")
    parser.add_argument("--cnn-run", type=Path, default=RQ1 / "cnn_cosine_decay_v2")
    parser.add_argument("--lr-selection", type=Path, default=RQ1 / "lr_search_seed0/selection.json")
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.output, args.cnn_run, args.lr_selection, torch.device(args.device))
