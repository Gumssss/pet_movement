"""Audit raw DogMove annotations and build non-overlapping labeled windows."""

import hashlib
import json
import zipfile
from collections import Counter

import numpy as np
import pandas as pd

from .data import CHANNELS, CLASSES, ROOT

DATA = ROOT / "data/犬只运动传感器数据集"
CACHE_DIR = ROOT / "data/cache"
CACHE = CACHE_DIR / "pet_windows.npz"
AUDIT_CACHE = CACHE_DIR / "pet_audit.json"


def audit():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    rawlabels = Counter()
    support = Counter()
    sessions = []
    xx = []
    yy = []
    dd = []
    tt = []
    ss = []

    def process(df):
        dog = int(df.DogID.iloc[0])
        test = int(df.TestNum.iloc[0])
        t = df.t_sec.to_numpy()
        x = df[CHANNELS].to_numpy(np.float32)
        a = df[["Behavior_1", "Behavior_2", "Behavior_3"]].fillna("<undefined>").to_numpy(str)
        for row in a.T:
            rawlabels.update(row)
        hit = np.stack([(a == c).any(axis=1) for c in CLASSES], axis=1)
        for k, c in enumerate(CLASSES):
            support[(dog, c)] += int(hit[:, k].sum())
        stats["sole_galloping_rows"] += int(
            ((a == "Galloping").any(1) & ((a != "<undefined>").sum(1) == 1)).sum()
        )
        stats["rows"] += len(df)
        stats["no_target_rows"] += int((hit.sum(1) == 0).sum())
        stats["multiple_target_rows"] += int((hit.sum(1) > 1).sum())
        stats["unannotated_rows"] += int((a == "<undefined>").all(1).sum())
        stats["nonfinite_signal_rows"] += int((~np.isfinite(x).all(1)).sum())
        delta = np.diff(t)
        bad = np.where(~np.isclose(delta, 0.01, atol=1e-5))[0] + 1
        stats["timestamp_breaks"] += len(bad)
        sessions.append({"dog": dog, "test": test, "rows": len(df), "breaks": len(bad)})
        for seg in np.split(np.arange(len(df)), bad):
            n = len(seg) // 200
            stats["tail_rows"] += len(seg) % 200
            if not n:
                continue
            inds = seg[: n * 200].reshape(n, 200)
            h = hit[inds]
            v = x[inds]
            # Exactly one main behavior throughout 2 seconds. Auxiliary behaviors may coexist.
            valid = (h.sum(2) == 1).all(1) & (h == h[:, :1, :]).all((1, 2)) & np.isfinite(v).all((1, 2))
            stats["candidate_windows"] += n
            stats["rejected_windows"] += int((~valid).sum())
            if valid.any():
                xx.append(v[valid])
                yy.extend(h[valid, 0].argmax(1))
                dd.extend([dog] * int(valid.sum()))
                tt.extend([test] * int(valid.sum()))
                ss.extend(t[inds[valid, 0]])

    with zipfile.ZipFile(DATA / "DogMoveData_csv_format.zip") as z:
        with z.open("DogMoveData.csv") as f:
            carry = None
            cols = ["DogID", "TestNum", "t_sec"] + CHANNELS + ["Behavior_1", "Behavior_2", "Behavior_3"]
            for chunk in pd.read_csv(f, usecols=cols, chunksize=400000):
                if carry is not None:
                    chunk = pd.concat([carry, chunk], ignore_index=True)
                groups = list(chunk.groupby(["DogID", "TestNum"], sort=False))
                for _, g in groups[:-1]:
                    process(g)
                carry = groups[-1][1].copy()
            if carry is not None:
                process(carry)
    X = np.concatenate(xx)
    y = np.array(yy)
    dogs = np.array(dd)
    tests = np.array(tt)
    starts = np.array(ss)
    np.savez(CACHE, X=X, y=y, dogs=dogs, tests=tests, starts=starts)
    classes = {
        c: {
            "rows_with_label": sum(n for (d, b), n in support.items() if b == c),
            "windows": int((y == k).sum()),
            "dogs": len(np.unique(dogs[y == k])),
        }
        for k, c in enumerate(CLASSES)
    }
    with (DATA / "DogMoveData_csv_format.zip").open("rb") as archive:
        archive_hash = hashlib.file_digest(archive, "sha256").hexdigest()
    result = {
        "stats": dict(stats),
        "classes": classes,
        "raw_labels": dict(rawlabels),
        "sessions": sessions,
        "csv_zip_sha256": archive_hash,
    }
    AUDIT_CACHE.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "stats": dict(stats),
                "classes": classes,
                "dogs": len(np.unique(dogs)),
                "sessions": len(sessions),
            },
            indent=2,
        ),
        flush=True,
    )
