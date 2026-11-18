"""Inference with the historical CNN checkpoint and optional temperature calibration."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .paths import CNN_CHECKPOINTS
from .data import CHANNELS, ROOT
from .models.cnn import CNN


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Predictor:
    def __init__(self, checkpoint=CNN_CHECKPOINTS / "cnn_best.pt", device="cpu", calibration=None):
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("Apple MPS is unavailable; use --device cpu.")
        self.device = device
        self.checkpoint = Path(checkpoint)
        ck = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        if ck["channels"] != CHANNELS or ck["window_samples"] != 200 or len(ck["classes"]) != 7:
            raise ValueError("Checkpoint input schema is incompatible.")
        self.classes = ck["classes"]
        # Training and historical evaluation normalized float32 source windows.
        self.mean = np.array(ck["mean"], dtype=np.float32)
        self.std = np.array(ck["std"], dtype=np.float32)
        if (
            self.mean.shape != (6,)
            or self.std.shape != (6,)
            or not np.isfinite([self.mean, self.std]).all()
            or (self.std <= 0).any()
        ):
            raise ValueError("Invalid checkpoint normalization.")
        self.model = CNN(ck.get("config", {}).get("dropout", 0.2))
        self.model.load_state_dict(ck["state_dict"], strict=True)
        self.model.to(device).eval()
        self.temperature = 1.0
        self.calibration = None
        if calibration is not None:
            config = json.loads(Path(calibration).read_text())
            if config["checkpoint_sha256"] != sha(self.checkpoint) or config["classes"] != self.classes:
                raise ValueError("Temperature calibration does not match this checkpoint/classes.")
            self.temperature = float(config["temperature"])
            if not np.isfinite(self.temperature) or self.temperature <= 0:
                raise ValueError("Temperature must be finite and positive.")
            self.calibration = str(calibration)

    def logits(self, samples, batch_size=256):
        x = np.asarray(samples, dtype=np.float32)
        if x.ndim == 2:
            x = x[None]
        if x.ndim != 3 or x.shape[1:] != (200, 6) or not len(x) or not np.isfinite(x).all():
            raise ValueError("Expected finite IMU data with shape (200, 6) or (N, 200, 6).")
        x = ((x - self.mean) / self.std).transpose(0, 2, 1).astype(np.float32)
        with torch.inference_mode():
            return torch.cat(
                [
                    self.model(batch.to(self.device)).cpu()
                    for batch in torch.from_numpy(x.copy()).split(batch_size)
                ]
            ).numpy()

    def predict(self, samples):
        logits = torch.from_numpy(self.logits(samples))
        raw = logits.softmax(1).numpy()
        probs = (logits / self.temperature).softmax(1).numpy()
        records = []
        for p, original in zip(probs, raw):
            record = {
                "prediction": self.classes[int(p.argmax())],
                "confidence": float(p.max()),
                "probabilities": dict(zip(self.classes, map(float, p))),
            }
            if self.calibration is not None:
                record.update(
                    temperature=self.temperature,
                    calibration=self.calibration,
                    uncalibrated_confidence=float(original.max()),
                    uncalibrated_probabilities=dict(zip(self.classes, map(float, original))),
                )
            records.append(record)
        return records
