"""Compare fixed and cosine learning rates for the RQ1 time-domain CNN."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pet_imu.data import CLASSES, ROOT, load_windows
from pet_imu.evaluation import bootstrap_f1, cluster_matrices, compare
from pet_imu.models.cnn import CNN
from pet_imu.training import batches, fit, macro_f1, save_json
from pet_imu.transforms import normalization, standardize
from search_rq1_lr import CNN_DROPOUT, CONFIG
from train_cnn_frequency import SEEDS

ARMS = ("constant", "cosine")
DEFAULT_SELECTION = ROOT / "outputs/RQ1/lr_search_seed0/selection.json"
MIN_LR_FACTOR = 0.01


def cosine_schedule(horizon_steps):
    hold_steps = horizon_steps // 10

    def factor(completed_steps):
        fraction = (completed_steps + 1 - hold_steps) / (horizon_steps - hold_steps)
        progress = min(max(fraction, 0), 1)
        return MIN_LR_FACTOR + (1 - MIN_LR_FACTOR) * 0.5 * (1 + math.cos(math.pi * progress))

    return factor


def run(output, selection, device):
    lr = json.loads(selection.read_text())["cnn"]["lr"]
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("The selected CNN learning rate must be finite and positive.")
    windows, raw, protocol = load_windows()
    masks = {part: windows.split.eq(part).to_numpy() for part in ("train", "validation", "test")}
    labels = windows.label.map(CLASSES.index).to_numpy()
    signals = raw.transpose(0, 2, 1)
    stats = normalization(signals[masks["train"]], "train")
    x, v = [
        torch.from_numpy(standardize(signals[masks[part]], stats)).to(device)
        for part in ("train", "validation")
    ]
    batch_size = CONFIG["batch_size"]
    steps_per_epoch = math.ceil(len(x) / batch_size)
    output.mkdir(parents=True, exist_ok=False)
    manifest = ROOT / "data/splits/windows.csv"
    save_json(
        output / "protocol.json",
        {
            "device": str(device),
            "torch": str(torch.__version__),
            "precision": "fp32",
            "splits": protocol["splits"],
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "seeds": SEEDS,
            "training": {k: v for k, v in CONFIG.items() if k != "seed"},
            "initial_lr": lr,
            "lr_selection_source": str(selection),
            "dropout": CNN_DROPOUT,
            "optimizer": "AdamW",
            "class_weights": "inverse square root of training frequency",
            "normalization": stats,
            "augmentation": False,
            "warmup": False,
            "steps_per_epoch": steps_per_epoch,
            "cosine": {
                "horizon": "optimizer steps completed by constant LR under the same seed",
                "hold_fraction": 0.1,
                "min_lr": lr * MIN_LR_FACTOR,
            },
            "checkpoint_selection": "clean validation macro-F1, evaluated every epoch",
            "test_policy": "evaluate both arms once after all best checkpoints are fixed",
        },
    )
    y = torch.from_numpy(labels[masks["train"]]).to(device)
    y_val = labels[masks["validation"]]
    validation = {arm: {} for arm in ARMS}
    schedules = {}
    for arm in ARMS:
        (output / arm).mkdir()
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = CNN(CNN_DROPOUT).to(device)
            directory = output / arm / str(seed)
            print(f"{arm} seed={seed}", flush=True)
            lr_factor = None
            if arm == "cosine":
                horizon_steps = validation["constant"][str(seed)]["epochs"] * steps_per_epoch
                lr_factor = cosine_schedule(horizon_steps)
                schedules[str(seed)] = {"hold_steps": horizon_steps // 10, "horizon_steps": horizon_steps}
                save_json(output / "schedules.json", schedules)
            result = fit(
                model,
                (x, y),
                (v, y_val),
                {**CONFIG, "seed": seed},
                directory,
                CONFIG["max_epochs"],
                lr,
                False,
                patience=CONFIG["patience"],
                min_delta=CONFIG["min_delta"],
                lr_factor=lr_factor,
            )
            model.load_state_dict(torch.load(directory / "best.pt", map_location="cpu", weights_only=True))
            predicted = batches(model, v, batch_size, False).argmax(1).cpu().numpy()
            np.testing.assert_allclose(
                macro_f1(y_val, predicted), result["validation_macro_f1"], rtol=0, atol=1e-12
            )
            validation[arm][str(seed)] = result
    save_json(output / "validation.json", validation)

    test = windows.loc[masks["test"]]
    test_x = torch.from_numpy(standardize(signals[masks["test"]], stats)).to(device)
    test_y = labels[masks["test"]]
    dogs = test.dog_id.to_numpy()
    n_dogs = len(np.unique(dogs))
    draws = np.random.default_rng(0).integers(n_dogs, size=(2000, n_dogs))
    binned_deltas, prediction_tables = [], []
    test_results = {arm: {} for arm in ARMS}
    for seed in SEEDS:
        predicted = {}
        for arm in ARMS:
            model.load_state_dict(
                torch.load(output / arm / str(seed) / "best.pt", map_location="cpu", weights_only=True)
            )
            predicted[arm] = batches(model, test_x, batch_size, False).argmax(1).cpu().numpy()
            test_results[arm][str(seed)] = compare(test_y, predicted[arm], dogs, predicted["constant"], 0)
        boot = {arm: bootstrap_f1(cluster_matrices(test_y, p, dogs), draws) for arm, p in predicted.items()}
        binned_deltas.append(boot["cosine"] - boot["constant"])
        prediction_tables.append(
            test.assign(seed=seed, **{arm: np.array(CLASSES)[p] for arm, p in predicted.items()})
        )
    scores = ("macro_f1", "accuracy", "balanced_accuracy")
    means = {
        arm: {metric: float(np.mean([r[metric] for r in test_results[arm].values()])) for metric in scores}
        for arm in ARMS
    }
    paired_interval = np.quantile(np.mean(binned_deltas, axis=0), [0.025, 0.975]).tolist()
    metrics = {
        "test_dogs": n_dogs,
        "test_windows": len(test),
        "mean_by_arm": means,
        "per_seed": test_results,
        "delta_macro_f1": means["cosine"]["macro_f1"] - means["constant"]["macro_f1"],
        "paired_dog_bootstrap_95ci": paired_interval,
    }
    pd.concat(prediction_tables).to_csv(output / "predictions.csv", index=False)
    save_json(output / "metrics.json", metrics)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_seed"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/RQ1/cnn_cosine_decay_v2")
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.output, args.selection, torch.device(args.device))
