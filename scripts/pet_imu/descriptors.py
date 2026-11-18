"""Small posture and rhythm feature vectors for a 50 Hz collar window."""

import numpy as np

SAMPLE_RATE = 50
AUTOCORRELATION_LAGS = np.arange(5, 46)
CROSS_CORRELATION_LAGS = np.arange(-10, 11)
MIN_PEAK = 0.2
POSTURE_COLUMNS = (
    "mean_acc_dir_x",
    "mean_acc_dir_y",
    "mean_acc_dir_z",
    "dynamic_acc_rms",
    "dynamic_acc_median",
    "dynamic_acc_p90",
    "gyro_rms",
    "gyro_median",
    "gyro_p90",
)
RHYTHM_COLUMNS = tuple(
    f"{signal}_{feature}"
    for signal in ("dynamic_acc", "gyro", "cross")
    for feature in ("correlation", "lag_seconds", "valid")
)


def magnitudes(windows):
    """Acceleration around its window mean and angular velocity, in original units."""
    acceleration = windows[:, :3]
    dynamic = acceleration - acceleration.mean(axis=2, keepdims=True)
    return np.linalg.norm(dynamic, axis=1), np.linalg.norm(windows[:, 3:], axis=1)


def posture(windows):
    mean = windows[:, :3].mean(axis=2)
    direction = mean / np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-6)
    columns = [direction]
    for signal in magnitudes(windows):
        rms = np.sqrt(np.mean(signal**2, axis=1))
        quantiles = np.quantile(signal, [0.5, 0.9], axis=1).T
        columns.append(np.column_stack((rms, quantiles)))
    return np.concatenate(columns, axis=1).astype(np.float32)


def rhythm_floor(windows):
    """Training-fold activity threshold for the reliability flags."""
    return np.maximum([np.quantile(x.std(axis=1), 0.1) for x in magnitudes(windows)], 1e-6)


def _lagged_correlation(left, right, lag):
    # Both signals are centered over the full window. Scale by overlap energies.
    if lag < 0:
        left, right = left[:, -lag:], right[:, :lag]
    elif lag > 0:
        left, right = left[:, :-lag], right[:, lag:]
    scale = np.sqrt(np.sum(left**2, axis=1) * np.sum(right**2, axis=1))
    return np.sum(left * right, axis=1) / np.maximum(scale, 1e-12)


def rhythm(windows, floor):
    signals = magnitudes(windows)
    active = [x.std(axis=1) >= minimum for x, minimum in zip(signals, floor, strict=True)]
    centered = [x - x.mean(axis=1, keepdims=True) for x in signals]
    columns = []
    for signal, is_active in zip(centered, active, strict=True):
        correlations = np.stack(
            [
                _lagged_correlation(signal, signal, k)
                for k in range(AUTOCORRELATION_LAGS[0] - 1, AUTOCORRELATION_LAGS[-1] + 2)
            ],
            axis=1,
        )
        center = correlations[:, 1:-1]
        peaks = (center > correlations[:, :-2]) & (center >= correlations[:, 2:])
        index = np.where(peaks, center, -np.inf).argmax(axis=1)
        value = center[np.arange(len(center)), index]
        valid = is_active & peaks[np.arange(len(center)), index] & (value >= MIN_PEAK)
        lag = AUTOCORRELATION_LAGS[index] / SAMPLE_RATE
        columns.append(np.column_stack((np.where(valid, value, 0), np.where(valid, lag, 0), valid)))
    cross = np.stack([_lagged_correlation(*centered, k) for k in CROSS_CORRELATION_LAGS], axis=1)
    index = np.abs(cross).argmax(axis=1)
    value = cross[np.arange(len(cross)), index]
    valid = active[0] & active[1] & (np.abs(value) >= MIN_PEAK)
    lag = CROSS_CORRELATION_LAGS[index] / SAMPLE_RATE
    columns.append(np.column_stack((np.where(valid, value, 0), np.where(valid, lag, 0), valid)))
    return np.concatenate(columns, axis=1).astype(np.float32)
