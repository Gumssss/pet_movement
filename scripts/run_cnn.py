"""Inference for the saved 1D CNN on 200 x 6 IMU samples (CPU or Apple MPS)."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__:
    from .pet_imu.cnn_inference import CNN, Predictor, sha
    from .pet_imu.paths import CNN_CHECKPOINTS
    from .pet_imu.data import ROOT, CHANNELS
else:
    from pet_imu.cnn_inference import CNN, Predictor, sha
    from pet_imu.paths import CNN_CHECKPOINTS
    from pet_imu.data import ROOT, CHANNELS


__all__ = ["CHANNELS", "ROOT", "CNN", "Predictor", "read_input", "sha"]


def read_input(path):
    """Demo images identify a verified paired signal; pixels are not model inputs."""
    path = Path(path)
    kind = "imu_csv"
    if path.suffix.lower() in {".webp", ".png", ".jpg", ".jpeg"}:
        metadata = json.loads(path.with_suffix(".json").read_text())
        if sha(path) != metadata["image_sha256"]:
            raise ValueError("Image does not match its paired demo metadata.")
        path = path.with_suffix(".csv")
        if sha(path) != metadata["signal_sha256"]:
            raise ValueError("Signal does not match its paired demo metadata.")
        kind = "demo_image_with_paired_imu_csv"
    elif path.suffix.lower() != ".csv":
        raise ValueError("Use an IMU CSV or a generated demo image with its paired CSV/JSON.")
    frame = pd.read_csv(path)
    missing = set(CHANNELS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing channels: {sorted(missing)}")
    return frame[CHANNELS].to_numpy(dtype=np.float32), kind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="200-row IMU CSV, or demo image with paired CSV/JSON"
    )
    parser.add_argument("--checkpoint", type=Path, default=CNN_CHECKPOINTS / "cnn_best.pt")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument(
        "--calibration", type=Path, help="Saved temperature.json; absent means original probabilities"
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        samples, kind = read_input(args.input)
        result = Predictor(args.checkpoint, args.device, args.calibration).predict(samples)[0]
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "input": str(args.input),
                "input_kind": kind,
                "device": args.device,
                "checkpoint": str(args.checkpoint),
                **result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
