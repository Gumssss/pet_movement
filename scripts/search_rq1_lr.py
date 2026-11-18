"""Validation-only learning-rate search for the CNN and frozen InrtlWrist linear probe."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from pet_imu.data import CLASSES, ROOT, load_windows
from pet_imu.models.cnn import CNN
from pet_imu.models.inrtlwrist import SHA256, classifier, encoder
from pet_imu.training import batches, fit, macro_f1, save_json
from pet_imu.transforms import normalization, resample_windows, standardize

GRID = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2)
CONFIG = {
    "seed": 0,
    "batch_size": 128,
    "weight_decay": 1e-4,
    "max_epochs": 200,
    "patience": 10,
    "min_delta": 1e-6,
}
CNN_DROPOUT = 0.2


def outside(lr, upper):
    decade = 10 ** np.floor(np.log10(lr))
    if upper:
        return float(f"{(3 if lr < 2 * decade else 10) * decade:.3g}")
    return float(f"{(0.3 if lr < 2 * decade else 1) * decade:.3g}")


def search(directory, make_model, inputs, labels, device, *, grid=GRID, config=None):
    config = CONFIG if config is None else config
    x, v = inputs
    y, val_labels = labels
    pending = list(grid)
    results = []
    while pending:
        for lr in pending:
            torch.manual_seed(config["seed"])
            model = make_model().to(device)
            trial = directory / f"lr_{lr:.0e}"
            result = fit(
                model,
                (x, y),
                (v, val_labels),
                config,
                trial,
                config["max_epochs"],
                lr,
                False,
                patience=config["patience"],
                min_delta=config["min_delta"],
            )
            model.load_state_dict(
                torch.load(trial / "best.pt", map_location="cpu", weights_only=True)
            )
            predicted = (
                batches(model, v, config["batch_size"], False).argmax(1).cpu().numpy()
            )
            np.testing.assert_allclose(
                macro_f1(val_labels, predicted),
                result["validation_macro_f1"],
                rtol=0,
                atol=1e-12,
            )
            results.append(result)
            save_json(directory / "trials.json", results)
        winner = min(results, key=lambda r: (-r["validation_macro_f1"], r["lr"]))
        rates = [r["lr"] for r in results]
        if winner["lr"] == max(rates):
            pending = [outside(max(rates), upper=True)]
        elif winner["lr"] == min(rates):
            pending = [outside(min(rates), upper=False)]
        else:
            pending = []
    return winner


def run(output, device):
    windows, raw, _ = load_windows()
    folds = [windows.split.eq(s).to_numpy() for s in ("train", "validation")]
    raw_folds = [raw[m] for m in folds]
    labels = [windows.loc[m, "label"].map(CLASSES.index).to_numpy() for m in folds]
    labels[0] = torch.from_numpy(labels[0]).to(device)
    output.mkdir(parents=True, exist_ok=False)
    save_json(
        output / "protocol.json",
        {
            **CONFIG,
            "initial_lr": GRID,
            "adam": "AdamW, constant lr",
            "device": str(device),
            "torch": str(torch.__version__),
            "precision": "fp32",
            "augmentation": False,
            "cnn_dropout": CNN_DROPOUT,
            "classifier": "frozen pretrained InrtlWrist + Linear(384,7)",
            "checkpoint_sha256": SHA256,
            "class_weight": "inverse square root of train frequency",
            "selection": "best validation macro-F1 per model, ties choose smaller lr",
            "expansion": "extend winning edge on 1-3-10 log grid until winner is interior",
            "manifest_sha256": hashlib.sha256(
                (ROOT / "data/splits/windows.csv").read_bytes()
            ).hexdigest(),
            "dogs": [sorted(windows.loc[m, "dog_id"].unique().tolist()) for m in folds],
        },
    )
    winners = {}
    for name in ("cnn", "inrtlwrist_head"):
        directory = output / name
        directory.mkdir()
        if name == "cnn":
            signals = [x.transpose(0, 2, 1) for x in raw_folds]
        else:
            signals = [resample_windows(x) for x in raw_folds]
        stats = normalization(signals[0], "train")
        save_json(directory / "normalization.json", stats)
        inputs = [torch.from_numpy(standardize(x, stats)).to(device) for x in signals]
        if name == "cnn":

            def make_model():
                return CNN(CNN_DROPOUT)
        else:
            backbone = encoder().requires_grad_(False).to(device).eval()
            inputs = [batches(backbone, x, CONFIG["batch_size"], False) for x in inputs]
            del backbone

            def make_model():
                return classifier(CONFIG["seed"])

        winners[name] = search(directory, make_model, inputs, labels, device)
        save_json(output / "selection.json", winners)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/RQ1/lr_search_seed0"
    )
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default="mps")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.output, torch.device(args.device))
