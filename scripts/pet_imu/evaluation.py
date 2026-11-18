"""Window metrics and confidence intervals paired across the same held-out dogs."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from .data import CLASSES
from .training import macro_f1


def cluster_matrices(y, predicted, dogs):
    return np.stack(
        [
            confusion_matrix(y[dogs == dog], predicted[dogs == dog], labels=range(len(CLASSES)))
            for dog in np.unique(dogs)
        ]
    )


def bootstrap_f1(matrices, samples):
    cm = matrices[samples].sum(axis=1)
    numerator = 2 * np.diagonal(cm, axis1=1, axis2=2)
    denom = cm.sum(axis=1) + cm.sum(axis=2)
    return np.divide(numerator, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0).mean(axis=1)


def compare(y, predicted, dogs, cnn, seed):
    matrices = cluster_matrices(y, predicted, dogs)
    reference = cluster_matrices(y, cnn, dogs)
    n = len(matrices)
    sample = np.random.default_rng(seed).integers(n, size=(2000, n))
    delta = bootstrap_f1(matrices, sample) - bootstrap_f1(reference, sample)
    return {
        "macro_f1": macro_f1(y, predicted),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "accuracy": float(accuracy_score(y, predicted)),
        "delta_macro_f1": macro_f1(y, predicted) - macro_f1(y, cnn),
        "paired_dog_bootstrap_95ci": np.quantile(delta, [0.025, 0.975]).tolist(),
        "classes": classification_report(
            y, predicted, labels=range(len(CLASSES)), target_names=CLASSES, output_dict=True, zero_division=0
        ),
        "confusion_matrix": matrices.sum(axis=0).tolist(),
        "per_dog_confusion_matrix": dict(zip(map(str, np.unique(dogs)), matrices.tolist())),
    }
