"""Convert DogMove windows to the shared model input and standardize six channels."""

import numpy as np
from scipy.signal import resample_poly

MODEL_SAMPLE_RATE_HZ = 50
MODEL_WINDOW_SAMPLES = 96


def resample_windows(raw):
    """Reduce 100 Hz DogMove windows to the centered 50 Hz model input."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1:] != (200, 6) or not np.isfinite(raw).all():
        raise ValueError("Expected finite (N, 200, 6) collar samples.")
    filtered = resample_poly(raw, up=1, down=2, axis=1, padtype="line")
    start = (filtered.shape[1] - MODEL_WINDOW_SAMPLES) // 2
    return np.ascontiguousarray(filtered[:, start : start + MODEL_WINDOW_SAMPLES].transpose(0, 2, 1))


def normalization(x_train, kind):
    if kind == "window":
        return {"kind": kind}
    if kind != "train":
        raise ValueError(f"Unknown normalization {kind}")
    return {
        "kind": kind,
        "mean": x_train.mean((0, 2), dtype=np.float64).tolist(),
        "std": x_train.std((0, 2), dtype=np.float64).tolist(),
    }


def standardize(x, stats):
    if stats["kind"] == "window":
        mean, std = x.mean(-1, keepdims=True), x.std(-1, keepdims=True, ddof=1)
        return np.ascontiguousarray((x - mean) / (std + 1e-6))
    else:
        mean = np.asarray(stats["mean"], np.float32).reshape(1, 6, 1)
        std = np.asarray(stats["std"], np.float32).reshape(1, 6, 1)
    return np.ascontiguousarray((x - mean) / np.maximum(std, 1e-6))
