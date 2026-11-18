"""Verified DogMove windows in manifest order. Inputs are raw (N, 200, 6)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CHANNELS = ["ANeck_x", "ANeck_y", "ANeck_z", "GNeck_x", "GNeck_y", "GNeck_z"]
CLASSES = ["Galloping", "Lying chest", "Sitting", "Sniffing", "Standing", "Trotting", "Walking"]


def load_windows():
    directory = ROOT / "data/splits"
    windows = pd.read_csv(directory / "windows.csv")
    splits = json.loads((directory / "dog_splits.json").read_text())["splits"]
    protocol = json.loads((ROOT / "outputs/protocol.json").read_text())
    assignments = [(dog, part) for part, dogs in splits.items() for dog in dogs]
    dog_split = dict(assignments)
    if len(dog_split) != len(assignments) or splits != protocol["splits"]:
        raise ValueError("Dog split definitions disagree or overlap.")
    if protocol["channels"] != CHANNELS or protocol["classes"] != CLASSES:
        raise ValueError("Historical channel or label order changed.")
    with np.load(ROOT / "data/cache/pet_windows.npz", allow_pickle=False) as cache:
        x, y, dogs, sessions, starts = (cache[k] for k in ("X", "y", "dogs", "tests", "starts"))
    ids = [f"dog{d}_session{s}_start{t:.12g}" for d, s, t in zip(dogs, sessions, starts)]
    valid = (
        x.shape == (len(windows), 200, 6)
        and np.isfinite(x).all()
        and windows.window_id.is_unique
        and windows.window_id.tolist() == ids
        and np.array_equal(windows.dog_id, dogs)
        and np.array_equal(windows.session, sessions)
        and np.allclose(windows.start_sec, starts, rtol=0, atol=1e-8)
        and np.allclose(windows.end_exclusive_sec, starts + 2, rtol=0, atol=1e-8)
        and np.all((0 <= y) & (y < len(CLASSES)))
        and np.array_equal(windows.label, np.array(CLASSES)[y])
        and set(dogs) == set(dog_split)
        and windows.dog_id.map(dog_split).equals(windows.split)
    )
    if not valid:
        raise ValueError("Signal cache and fixed window manifest do not match.")
    return windows, x, protocol
