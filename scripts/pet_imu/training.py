"""PyTorch training primitives for the locked dog-holdout experiment."""

import json
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from .data import CLASSES

DOG_CLASS_COUNT = len(CLASSES)


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def macro_f1(labels, predicted, class_count=DOG_CLASS_COUNT):
    return float(f1_score(labels, predicted, labels=range(class_count), average="macro", zero_division=0))


def batches(model, inputs, batch_size, amp):
    model.eval()
    with torch.no_grad(), torch.autocast(inputs.device.type, dtype=torch.bfloat16, enabled=amp):
        return torch.cat([model(batch).float() for batch in inputs.split(batch_size)])


def fit(
    model, train, val, config, directory, epochs, lr, amp, *,
    patience=None, min_delta=0.0, epoch_inputs=None, lr_factor=None, parameter_groups=None,
    class_count=DOG_CLASS_COUNT,
):
    """Select a checkpoint using validation labels only. Inputs already reside on the model device."""
    directory.mkdir()
    x, y = train
    x_val, y_val = val
    device = x.device
    freq = torch.bincount(y.cpu(), minlength=class_count).float()
    if freq.min() == 0:
        raise ValueError("A training class is absent.")
    weight = freq.rsqrt().to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    parameters = model.parameters() if parameter_groups is None else parameter_groups
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=config["weight_decay"])
    scheduler = None if lr_factor is None else torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    batch_size = config["batch_size"]
    rng = torch.Generator().manual_seed(config["seed"])
    history, best = [], -1.0
    stale = 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        x_epoch = x if epoch_inputs is None else epoch_inputs()
        order = torch.randperm(len(x), generator=rng).to(device)
        accumulated = torch.zeros((), device=device)
        first_lr = optimizer.param_groups[0]["lr"]
        for indices in order.split(batch_size):
            last_lr = optimizer.param_groups[0]["lr"]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss = criterion(model(x_epoch[indices]).float(), y[indices])
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            accumulated += loss.detach() * len(indices)
        logits = batches(model, x_val, batch_size, amp)
        score = macro_f1(y_val, logits.argmax(1).cpu().numpy(), class_count)
        train_loss = float(accumulated.item() / len(x))
        if not np.isfinite(train_loss) or not torch.isfinite(logits).all().item():
            raise ValueError("Nonfinite loss or validation logits.")
        history.append(
            {
                "epoch": epoch,
                "validation_macro_f1": score,
                "train_loss": train_loss,
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
        if scheduler is not None:
            history[-1].update(first_lr=first_lr, last_lr=last_lr)
        if score > best + min_delta:
            best, best_epoch = score, epoch
            stale = 0
            tmp = directory / "best.tmp"
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, tmp)
            tmp.replace(directory / "best.pt")
        else:
            stale += 1
        save_json(directory / "history.json", history)
        print(f"{directory.name} epoch {epoch}/{epochs} val={score:.4f}", flush=True)
        if patience is not None and stale >= patience:
            break
    result = {
        "best_epoch": best_epoch,
        "validation_macro_f1": best,
        "epochs": len(history),
        "lr": lr,
        "seconds": time.perf_counter() - start,
    }
    save_json(directory / "validation.json", result)
    return result
