"""Label-preserving collar augmentations and diagnostic perturbations, on raw windows."""

from itertools import permutations

import numpy as np

from .data import CLASSES

STRENGTHS = {
    "recrop": (10, 20, 30),
    "rotate": (5, 10, 20),
    "scale": (0.05, 0.1, 0.15),
    "noise": (0.005, 0.01, 0.02),
    "speed": (0.025, 0.05, 0.075),
}
DIAGNOSTICS = ("band", "flip", "permute")
ORDER = ("recrop", "speed", "rotate", "scale", "noise", *DIAGNOSTICS)
PERMUTATIONS = np.asarray(list(permutations(range(4)))[1:])
GAIT = [CLASSES.index(label) for label in ("Walking", "Trotting", "Galloping")]


def neighbors(windows):
    """Local indices of verified adjacent, pure windows from the same training session."""
    if set(windows.split) != {"train"}:
        raise ValueError("Context must come from training dogs only.")
    times = windows.start_sec.to_numpy()
    keys = list(zip(windows.dog_id, windows.session, windows.label, np.rint(times * 100).astype(int)))
    lookup = {key: i for i, key in enumerate(keys)}
    assert len(lookup) == len(keys)
    sides = []
    for step in (-200, 200):
        side = np.array([lookup.get((d, s, c, t + step), -1) for d, s, c, t in keys])
        valid = side >= 0
        side[valid & ~np.isclose(times[side] - times, step / 100, rtol=0, atol=1e-5)] = -1
        sides.append(side)
    return sides


def rotate(x, degrees, rng):
    axis = rng.normal(size=(len(x), 3)).astype("f")
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    angle = np.deg2rad(rng.uniform(0, degrees, len(x))).astype("f")[:, None, None, None]
    axis = axis[:, None, None, :]
    triads = x.reshape(len(x), 200, 2, 3)
    projected = axis * (triads * axis).sum(axis=-1, keepdims=True)
    return (
        triads * np.cos(angle) + np.cross(axis, triads) * np.sin(angle) + projected * (1 - np.cos(angle))
    ).reshape(x.shape)


def scale(x, halfwidth, rng):
    gains = rng.uniform(1 - halfwidth, 1 + halfwidth, size=(len(x), 1, 2, 1)).astype("f")
    triads = x.reshape(len(x), 200, 2, 3)
    center = triads.mean(axis=1, keepdims=True)
    return ((triads - center) * gains + center).reshape(x.shape)


class Augmenter:
    def __init__(self, raw, labels, sides, std, operations, mode, seed):
        self.raw, self.labels = raw, labels
        self.left, self.right = sides
        self.std, self.operations, self.mode = std, operations, mode
        self.rng = np.random.default_rng(seed)
        self.counts = {}

    def count(self, event, indices):
        counts = self.counts.setdefault(event, np.zeros(len(CLASSES), dtype=int))
        counts += np.bincount(self.labels[indices], minlength=len(CLASSES))

    def sample(self, ids, points):
        # All points lie within verified same-label context (at most one neighbor per side).
        rows = np.where(
            points < 0, self.left[ids, None], np.where(points >= 200, self.right[ids, None], ids[:, None])
        )
        if np.any(rows < 0):
            raise ValueError("Sampling outside contiguous labeled windows")
        return self.raw[rows, points % 200]

    def batch(self, ids):
        rng = self.rng
        x = self.raw[ids].copy()
        n = len(x)
        trigger = rng.random(n) < 0.5
        choices = rng.integers(len(self.operations), size=n) if self.mode == "choice" else None
        offsets = np.zeros(n, dtype=int)
        n_applied = np.zeros(n, dtype=int)
        timeline = np.arange(200)
        for k, (op, strength) in enumerate(self.operations.items()):
            pos = np.flatnonzero(trigger if choices is None else trigger & (choices == k))
            rows = ids[pos]
            self.count(op + "_attempt", rows)
            if op == "recrop":
                left, right = self.left[rows] >= 0, self.right[rows] >= 0
                sign = np.where(left & right, rng.choice([-1, 1], len(rows)), np.where(left, -1, 1))
                offsets[pos] = sign * rng.integers(1, strength + 1, len(rows))
                offsets[pos[~(left | right)]] = 0
                pos = pos[left | right]
                x[pos] = self.sample(ids[pos], timeline + offsets[pos, None])
            elif op == "speed":
                pos = pos[np.isin(self.labels[rows], GAIT)]
                rows = ids[pos]
                rates = rng.uniform(1 - strength, 1 + strength, len(rows))
                t = 99.5 + offsets[pos, None] + rates[:, None] * (timeline - 99.5)
                valid = ((t[:, 0] >= 0) | (self.left[rows] >= 0)) & (
                    (t[:, -1] <= 199) | (self.right[rows] >= 0)
                )
                pos, t, rates = pos[valid], t[valid], rates[valid]
                rows = ids[pos]
                lo, hi = np.floor(t).astype(int), np.ceil(t).astype(int)
                a, b = self.sample(rows, lo), self.sample(rows, hi)
                x[pos] = a + (b - a) * (t - lo)[..., None]
                self.count("speed_faster", rows[rates > 1])
                self.count("speed_slower", rows[rates < 1])
            elif op == "rotate":
                x[pos] = rotate(x[pos], strength, rng)
            elif op == "scale":
                x[pos] = scale(x[pos], strength, rng)
            elif op == "noise":
                x[pos] += rng.standard_normal(x[pos].shape, dtype=np.float32) * self.std * strength
            elif op == "band":
                f = np.fft.rfft(x[pos], axis=1)
                first = rng.integers(1, 100, size=len(pos))
                for offset in (0, 1):
                    f[np.arange(len(pos)), first + offset] = 0
                x[pos] = np.fft.irfft(f, n=200, axis=1)
            elif op == "flip":
                sign = np.full((len(pos), 3), -1)
                sign[np.arange(len(pos)), rng.integers(3, size=len(pos))] = 1
                x[pos] *= np.tile(sign, (1, 2))[:, None, :]
            elif op == "permute":
                blocks = PERMUTATIONS[rng.integers(len(PERMUTATIONS), size=len(pos))]
                points = (blocks[..., None] * 50 + np.arange(50)).reshape(len(pos), 200)
                x[pos] = x[pos[:, None], points]
            else:
                raise ValueError(op)
            self.count(op + "_applied", ids[pos])
            n_applied[pos] += 1
        for i in range(len(self.operations) + 1):
            self.count("operations_" + str(i), ids[n_applied == i])
        return x

    def epoch_batches(self):
        """Yield augmented raw windows in training-manifest order."""
        for start in range(0, len(self.raw), 512):
            ids = np.arange(start, min(start + 512, len(self.raw)))
            yield slice(start, start + len(ids)), self.batch(ids)

    def epoch(self, mean):
        normalized = np.empty((len(self.raw), 6, 200), dtype=np.float32)
        for rows, raw in self.epoch_batches():
            normalized[rows] = ((raw - mean) / self.std).transpose(0, 2, 1)
        return normalized
