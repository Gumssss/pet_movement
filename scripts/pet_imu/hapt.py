"""Read labeled six-axis HAPT recordings from the UCI 341 archive."""

import re
from pathlib import PurePosixPath
from zipfile import ZipFile

import numpy as np

from .transforms import MODEL_WINDOW_SAMPLES as SAMPLES

STRIDE = 48


def load_hapt(archive):
    """Return training and validation windows in (N, 6, 96), separated by participant."""
    with ZipFile(archive) as source:
        members = {PurePosixPath(name).name: name for name in source.namelist() if not name.endswith("/")}

        def array(name, dtype):
            with source.open(members[name]) as file:
                return np.loadtxt(file, dtype=dtype, ndmin=2)

        train_subjects = np.unique(array("subject_id_train.txt", int))
        val_subjects = np.unique(array("subject_id_test.txt", int))
        splits = {"train": train_subjects.tolist(), "validation": val_subjects.tolist()}
        if np.intersect1d(train_subjects, val_subjects).size:
            raise ValueError("HAPT subject assignment overlaps across source folds.")
        classes = {}
        with source.open(members["activity_labels.txt"]) as file:
            for line in file:
                if line.strip():
                    idx, label = line.decode().strip().split(maxsplit=1)
                    classes[int(idx)] = label
        class_ids = sorted(classes)
        encode = {idx: j for j, idx in enumerate(class_ids)}
        intervals = array("labels.txt", int)
        if intervals.shape[1] != 5:
            raise ValueError("Expected experiment, subject, activity, first and final sample.")
        # Labels in the UCI archive index samples starting at 1 and include the final sample.
        if np.any(intervals[:, 3] < 1) or np.any(intervals[:, 4] < intervals[:, 3]):
            raise ValueError("Unexpected HAPT sample bounds.")
        person_to_fold = {subject: part for part, subjects in splits.items() for subject in subjects}
        data = {part: [] for part in splits}
        labels = {part: [] for part in splits}
        for name in sorted(members):
            match = re.fullmatch(r"acc_exp(\d+)_user(\d+)\.txt", name)
            if not match:
                continue
            experiment, subject = map(int, match.groups())
            fold = person_to_fold[subject]
            acc = array(name, np.float32)
            gyro = array(name.replace("acc_", "gyro_", 1), np.float32)
            if acc.shape != gyro.shape or acc.shape[1] != 3:
                raise ValueError(f"Misaligned acceleration and gyroscope in {name}.")
            signal = np.concatenate((acc, gyro), axis=1)
            rows = intervals[(intervals[:, 0] == experiment) & (intervals[:, 1] == subject)]
            if len(rows) == 0 or rows[:, 4].max() > len(signal):
                raise ValueError(f"Missing or out-of-bounds labels for {name}.")
            for _, _, label, first, last in rows:
                starts = np.arange(first - 1, last - SAMPLES + 1, STRIDE)
                if len(starts):
                    examples = signal[starts[:, None] + np.arange(SAMPLES)]
                    data[fold].append(examples.transpose(0, 2, 1))
                    labels[fold].extend([encode[label]] * len(starts))
        datasets = {}
        for part in splits:
            x = np.ascontiguousarray(np.concatenate(data[part]))
            y = np.asarray(labels[part], dtype=np.int64)
            if len(x) != len(y) or not np.isfinite(x).all() or len(np.unique(y)) != len(classes):
                raise ValueError(f"HAPT {part} data are malformed or a class has no full window.")
            datasets[part] = (x, y)
    return datasets, [classes[idx] for idx in class_ids], splits
