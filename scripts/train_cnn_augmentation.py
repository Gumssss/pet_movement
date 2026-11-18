"""Train and select RQ2 CNN augmentations on validation dogs, then compare on test dogs."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pet_imu.augmentation import (
    DIAGNOSTICS,
    GAIT,
    ORDER,
    STRENGTHS,
    Augmenter,
    neighbors,
    rotate,
    scale,
)
from pet_imu.data import CHANNELS, CLASSES, ROOT, load_windows
from pet_imu.evaluation import bootstrap_f1, cluster_matrices, compare
from pet_imu.models.cnn import CNN
from pet_imu.training import batches, fit, save_json
from train_cnn_frequency import RECIPE as SPECTRAL_COMPARISON_RECIPE
from train_cnn_frequency import SEEDS, normalized


def check_inputs(raw, labels, sides, std):
    rows = np.concatenate([np.flatnonzero(labels == i)[:20] for i in range(len(CLASSES))])
    x = raw[rows]
    rng = np.random.default_rng(572)

    def magnitude(v):
        return np.linalg.norm(v.reshape(-1, 200, 2, 3), axis=-1)

    np.testing.assert_allclose(magnitude(rotate(x, 10, rng)), magnitude(x), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(scale(x, 0.1, rng).mean(1), x.mean(1), rtol=2e-5, atol=2e-5)
    eligible = (sides[0] >= 0) | (sides[1] >= 0)
    both = np.flatnonzero((sides[0] >= 0) & (sides[1] >= 0))[:20]
    sampler = Augmenter(raw, labels, sides, std, {"recrop": 20}, "single", 572)
    for shift in (-20, 20):
        got = sampler.sample(both, np.tile(np.arange(200) + shift, (len(both), 1)))
        context = np.concatenate([raw[sides[0][both]], raw[both], raw[sides[1][both]]], axis=1)
        np.testing.assert_array_equal(got, context[:, 200 + shift : 400 + shift])
    qc = {}
    for op in ORDER:
        value = STRENGTHS[op][1] if op in STRENGTHS else None
        aug = Augmenter(raw, labels, sides, std, {op: value}, "single", 572)
        got = aug.batch(rows)
        assert got.shape == x.shape and np.isfinite(got).all()
        qc[op] = int(aug.counts[op + "_applied"].sum())
        if op == "speed":
            gait = np.isin(labels[rows], GAIT)

            def peak(v):
                motion = np.linalg.norm(v[:, :, 3:], axis=2)
                power = abs(np.fft.rfft((motion - motion.mean(1, keepdims=True)) * np.hanning(200), axis=1))
                return (1 + power[:, 1:21].argmax(1)) * 0.5

            qc["gait_peak_shift_hz_abs_max_on_sample"] = float(abs(peak(got[gait]) - peak(x[gait])).max())
            np.testing.assert_array_equal(got[~gait], x[~gait])
    return {
        "sampled_per_class": 20,
        "sample_checks": qc,
        "timestamp_breaks_in_raw_audit": 0,
        "classes": CLASSES,
        "train_counts": np.bincount(labels).tolist(),
        "contiguous_neighbor_counts": np.bincount(labels[eligible], minlength=len(CLASSES)).tolist(),
    }


def run(output, device, max_epochs, patience, lr):
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must both be positive")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be finite and positive.")
    recipe = {**SPECTRAL_COMPARISON_RECIPE, "max_epochs": max_epochs, "patience": patience, "lr": lr}
    windows, raw, protocol = load_windows()
    masks = {s: windows.split.eq(s).to_numpy() for s in ("train", "validation", "test")}
    train = windows.loc[masks["train"]]
    labels = windows.label.map(CLASSES.index).to_numpy()
    y = labels[masks["train"]]
    y_train = torch.from_numpy(y).to(device)
    val = windows.loc[masks["validation"]]
    y_val = labels[masks["validation"]]
    sides = neighbors(train)
    assert protocol["audit"]["stats"]["timestamp_breaks"] == 0
    x, stats = normalized(raw, masks["train"], (0, 1))
    x_train, x_val = [torch.from_numpy(x[masks[s]]).to(device) for s in ("train", "validation")]
    mean, std = [np.array(stats[key], dtype=np.float32) for key in ("mean", "std")]
    train_raw = raw[masks["train"]]
    checks = check_inputs(train_raw, y, sides, std)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "preflight.json", checks)
    save_json(
        output / "protocol.json",
        {
            "seeds": SEEDS,
            "training": recipe,
            "lr_schedule": "constant",
            "validation_schedule": "every epoch, clean validation windows, macro-F1",
            "splits": protocol["splits"],
            "device": str(device),
            "precision": "fp32",
            "torch_version": str(torch.__version__),
            "classes": CLASSES,
            "channels": CHANNELS,
            "window_manifest_sha256": hashlib.sha256(
                (ROOT / "data/splits/windows.csv").read_bytes()
            ).hexdigest(),
            "augmentation_gate": 0.5,
            "strengths_weak_default_strong": STRENGTHS,
            "diagnostics": DIAGNOSTICS,
            "normalization": stats,
            "plan": (ROOT / "docs/rq2-augmentation.md").read_text(),
        },
    )
    scores, configs, val_tables = {}, {}, []

    def trial(name, ops, mode="single"):
        folder = output / name
        folder.mkdir()
        ops = {op: ops[op] for op in ORDER if op in ops}
        configs[name] = {"operations": ops, "mode": mode}
        save_json(folder / "config.json", configs[name])
        per_seed = []
        for seed in SEEDS:
            print(f"Training {name} seed {seed}", flush=True)
            torch.manual_seed(seed)
            model = CNN(recipe["dropout"]).to(device)
            aug = Augmenter(train_raw, y, sides, std, ops, mode, seed)
            directory = folder / str(seed)
            result = fit(
                model,
                (x_train, y_train),
                (x_val, y_val),
                {**recipe, "seed": seed},
                directory,
                recipe["max_epochs"],
                recipe["lr"],
                False,
                patience=recipe["patience"],
                min_delta=recipe["min_delta"],
                epoch_inputs=(lambda aug=aug: torch.from_numpy(aug.epoch(mean)).to(device)) if ops else None,
            )
            save_json(directory / "coverage.json", {k: v.tolist() for k, v in aug.counts.items()})
            model.load_state_dict(torch.load(directory / "best.pt", map_location="cpu", weights_only=True))
            pred = batches(model, x_val, recipe["batch_size"], False).argmax(1).cpu().numpy()
            val_tables.append(
                pd.DataFrame(
                    {
                        "window_id": val.window_id,
                        "dog_id": val.dog_id,
                        "label": val.label,
                        "trial": name,
                        "seed": seed,
                        "prediction": np.array(CLASSES)[pred],
                    }
                )
            )
            per_seed.append(result["validation_macro_f1"])
        scores[name] = float(np.mean(per_seed))
        print(f"Validation {name} {scores[name]:.6f}", flush=True)
        save_json(output / "validation.json", scores)
        return name

    trial("baseline", {})
    for op, levels in STRENGTHS.items():
        trial(op, {op: levels[1]})
    for op in DIAGNOSTICS:
        trial(op, {op: None})
    finalists = sorted(STRENGTHS, key=lambda op: -scores[op])[:2]
    best = {}
    for op in finalists:
        weak, _default, strong = STRENGTHS[op]
        candidates = [trial(op + "_weak", {op: weak}), op, trial(op + "_strong", {op: strong})]
        winner = max(candidates, key=scores.get)
        best[op] = configs[winner]["operations"][op]
    for mode in ("choice", "both"):
        trial("combo_" + mode, best, mode)

    def rank(name):
        ops = configs[name]["operations"]
        levels = tuple(STRENGTHS[op].index(v) for op, v in ops.items())
        return (-scores[name], len(ops), levels, tuple(list(STRENGTHS).index(op) for op in ops))

    ranked = sorted((name for name in configs if name not in DIAGNOSTICS), key=rank)
    selected = ranked[0]
    comparison = selected if selected != "baseline" else ranked[1]
    save_json(
        output / "selection.json",
        {
            "top_two": finalists,
            "best_strengths": best,
            "ranking": ranked,
            "selected": selected,
            "test_comparison": comparison,
        },
    )
    pd.concat(val_tables).to_csv(output / "validation_predictions.csv.gz", index=False)
    test = windows.loc[masks["test"]]
    dogs = test.dog_id.to_numpy()
    y_test = labels[masks["test"]]
    x_test = torch.from_numpy(x[masks["test"]]).to(device)
    draws = np.random.default_rng(SEEDS[0]).integers(
        test.dog_id.nunique(), size=(2000, test.dog_id.nunique())
    )
    tables, results, boot = [], [], []
    for seed in SEEDS:
        preds = {}
        for name in ("baseline", comparison):
            model = CNN(recipe["dropout"]).to(device)
            model.load_state_dict(
                torch.load(output / name / str(seed) / "best.pt", map_location="cpu", weights_only=True)
            )
            preds[name] = batches(model, x_test, recipe["batch_size"], False).argmax(1).cpu().numpy()
        base, alt = preds.values()
        results.append({name: compare(y_test, p, dogs, base, SEEDS[0]) for name, p in preds.items()})
        boot.append(
            bootstrap_f1(cluster_matrices(y_test, alt, dogs), draws)
            - bootstrap_f1(cluster_matrices(y_test, base, dogs), draws)
        )
        tables.append(test.assign(seed=seed, **{name: np.array(CLASSES)[p] for name, p in preds.items()}))
    pd.concat(tables).to_csv(output / "predictions.csv", index=False)
    metric = {
        "per_seed": dict(zip(map(str, SEEDS), results)),
        "mean_macro_f1": {name: float(np.mean([r[name]["macro_f1"] for r in results])) for name in preds},
        "mean_delta": float(np.mean([r[comparison]["delta_macro_f1"] for r in results])),
        "paired_dog_bootstrap_95ci": np.quantile(np.mean(boot, axis=0), [0.025, 0.975]).tolist(),
    }
    save_json(output / "metrics.json", metric)
    print(json.dumps(metric["mean_macro_f1"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/RQ2/cnn_augmentation_v3_repeat")
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default="mps")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.output, torch.device(args.device), args.max_epochs, args.patience, args.lr)
