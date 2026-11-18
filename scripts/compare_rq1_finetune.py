"""Select a full InrtlWrist fine-tuning LR, then compare on the fixed test dogs."""

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
from search_rq1_lr import search

RQ1 = ROOT / "outputs/RQ1"
GRID = (1e-5, 3e-5, 1e-4, 3e-4)
MODEL = "inrtlwrist_finetune"
REFERENCES = ("cnn", "inrtlwrist_head")


def run(output, reference_run, device):
    spec = json.loads((reference_run / "protocol.json").read_text())
    windows, raw, protocol = load_windows()
    digest = hashlib.sha256((ROOT / "data/splits/windows.csv").read_bytes()).hexdigest()
    if (
        spec["manifest_sha256"] != digest
        or spec["splits"] != protocol["splits"]
        or spec["augmentation"]
        or spec["precision"] != "fp32"
        or spec["device"] != str(device)
        or spec["torch"] != str(torch.__version__)
        or spec["encoder_checkpoint_sha256"] != SHA256
    ):
        raise ValueError("Reference predictions do not match this experiment protocol.")
    recipe = spec["training"]
    seeds = spec["seeds"]
    labels = windows.label.map(CLASSES.index).to_numpy()
    train = windows.split.eq("train").to_numpy()
    val = windows.split.eq("validation").to_numpy()
    train_signal = resample_windows(raw[train])
    stats = normalization(train_signal, "train")
    if stats != spec["normalization"]:
        raise ValueError("Fine-tuning and the frozen encoder must use the same input normalization.")

    def inputs(signal):
        return torch.from_numpy(standardize(signal, stats)).to(device)

    x = inputs(train_signal)
    v = inputs(resample_windows(raw[val]))
    y = torch.from_numpy(labels[train]).to(device)
    y_val = labels[val]
    batch_size = recipe["batch_size"]

    def model_for(seed):
        torch.manual_seed(seed)
        return torch.nn.Sequential(encoder(), classifier(seed)).to(device)

    output.mkdir(parents=True, exist_ok=False)
    save_json(
        output / "protocol.json",
        {
            **spec,
            "reference_run": str(reference_run),
            "encoder_trainable": True,
            "optimizer": "AdamW",
            "lr_schedule": "constant",
            "warmup": False,
            "lr_search": {
                "seed": 0,
                "initial_grid": GRID,
                "criterion": "validation macro-F1",
                "expand_winning_edge": True,
                "ties": "lower lr",
            },
        },
    )
    sweep = output / "lr_search"
    sweep.mkdir()
    winner = search(
        sweep, lambda: model_for(0), (x, v), (y, y_val), device, grid=GRID, config={**recipe, "seed": 0}
    )
    save_json(output / "selection.json", winner)
    lr = winner["lr"]
    print(f"Selected LR {lr:.0e}, validation macro-F1 {winner['validation_macro_f1']:.4f}", flush=True)
    validation = json.loads((reference_run / "validation.json").read_text())
    validation[MODEL] = {}
    for seed in seeds:
        validation[MODEL][str(seed)] = fit(
            model_for(seed),
            (x, y),
            (v, y_val),
            {**recipe, "seed": seed},
            output / str(seed),
            recipe["max_epochs"],
            lr,
            False,
            patience=recipe["patience"],
            min_delta=recipe["min_delta"],
        )
    save_json(output / "validation.json", validation)

    mask = windows.split.eq("test").to_numpy()
    test = windows.loc[mask, ["window_id", "dog_id", "label"]].reset_index(drop=True)
    test_x = inputs(resample_windows(raw[mask]))
    y_test = labels[mask]
    dogs = test.dog_id.to_numpy()
    n_dogs = len(np.unique(dogs))
    draws = np.random.default_rng(0).integers(n_dogs, size=(2000, n_dogs))
    reference = pd.read_csv(reference_run / "predictions.csv")
    per_seed = json.loads((reference_run / "metrics.json").read_text())["per_seed"]
    deltas = {name: [] for name in REFERENCES}
    tables = []
    for seed in seeds:
        frame = reference.loc[reference.seed == seed].reset_index(drop=True)
        pd.testing.assert_frame_equal(frame[test.columns], test)
        model = model_for(seed)
        model.load_state_dict(
            torch.load(output / str(seed) / "best.pt", map_location="cpu", weights_only=True)
        )
        val_pred = batches(model, v, batch_size, False).argmax(1).cpu().numpy()
        np.testing.assert_allclose(
            macro_f1(y_val, val_pred), validation[MODEL][str(seed)]["validation_macro_f1"], atol=1e-12, rtol=0
        )
        pred = batches(model, test_x, batch_size, False).argmax(1).cpu().numpy()
        head = frame.inrtlwrist_head.map(CLASSES.index).to_numpy()
        per_seed[str(seed)][MODEL] = compare(y_test, pred, dogs, head, 0)
        finetune_boot = bootstrap_f1(cluster_matrices(y_test, pred, dogs), draws)
        for name in REFERENCES:
            baseline = frame[name].map(CLASSES.index).to_numpy()
            np.testing.assert_allclose(
                macro_f1(y_test, baseline), per_seed[str(seed)][name]["macro_f1"], atol=1e-12, rtol=0
            )
            deltas[name].append(finetune_boot - bootstrap_f1(cluster_matrices(y_test, baseline, dogs), draws))
        tables.append(frame.assign(**{MODEL: np.asarray(CLASSES)[pred]}))
    means = {
        name: {
            metric: float(np.mean([r[name][metric] for r in per_seed.values()]))
            for metric in ("macro_f1", "accuracy", "balanced_accuracy")
        }
        for name in (*REFERENCES, MODEL)
    }
    comparisons = {
        name: {
            "delta_macro_f1": means[MODEL]["macro_f1"] - means[name]["macro_f1"],
            "paired_dog_bootstrap_95ci": np.quantile(np.mean(delta, axis=0), [0.025, 0.975]).tolist(),
        }
        for name, delta in deltas.items()
    }
    pd.concat(tables).to_csv(output / "predictions.csv", index=False)
    save_json(
        output / "metrics.json",
        {
            "test_dogs": n_dogs,
            "test_windows": len(test),
            "mean": means,
            "finetune_vs": comparisons,
            "per_seed": per_seed,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RQ1 / "inrtlwrist_finetune_fp32")
    parser.add_argument("--reference-run", type=Path, default=RQ1 / "constant_lr_comparison")
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.output, args.reference_run, torch.device(args.device))
