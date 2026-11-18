"""Independently check RQ2 saved predictions, selection and coverage, and write the run report."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pet_imu.data import CLASSES, ROOT
from sklearn.metrics import confusion_matrix, f1_score

NAMES = {
    "baseline": "原窗 CNN",
    "recrop": "同标签重切窗",
    "rotate": "小角度旋转",
    "scale": "动态幅值缩放",
    "noise": "微弱加性噪声",
    "speed": "步态变速",
    "band": "频带遮挡",
    "flip": "大角度旋转",
    "permute": "时间段乱序",
    "combo_choice": "组合（随机择一）",
    "combo_both": "组合（同时施加）",
}
CLASS_NAMES = ["疾跑", "胸卧", "坐", "嗅闻", "站", "小跑", "步行"]


def describe(name):
    for suffix, text in (("_weak", "弱"), ("_strong", "强")):
        if name.endswith(suffix):
            return NAMES[name.removesuffix(suffix)] + f"（{text}）"
    return NAMES[name]


def read(path):
    return json.loads(path.read_text())


def check_stopping(history, run, recipe):
    """Replay the validation-based checkpoint and stopping decisions from the learning curve."""
    best, best_epoch = -1.0, 0
    assert len(history) == run["epochs"] <= recipe["max_epochs"]
    for epoch, record in enumerate(history, 1):
        assert epoch == record["epoch"]
        score = record["validation_macro_f1"]
        if score > best + recipe["min_delta"]:
            best, best_epoch = score, epoch
        if epoch - best_epoch >= recipe["patience"]:
            assert epoch == len(history)
    assert best_epoch == run["best_epoch"] and best == run["validation_macro_f1"]
    assert len(history) == recipe["max_epochs"] or len(history) - best_epoch == recipe["patience"]


def report(root):
    protocol = read(root / "protocol.json")
    seeds = protocol["seeds"]
    metrics = read(root / "metrics.json")
    selection = read(root / "selection.json")
    val_scores = read(root / "validation.json")
    coverage = read(root / "preflight.json")
    manifest = pd.read_csv(ROOT / "data/splits/windows.csv").set_index("window_id")
    val = pd.read_csv(root / "validation_predictions.csv.gz")
    test = pd.read_csv(root / "predictions.csv")
    candidate = selection["test_comparison"]
    assert len(val_scores) == 15
    assert len(val) == 45 * manifest.split.eq("validation").sum()
    assert len(test) == len(seeds) * manifest.split.eq("test").sum()
    assert set(val.trial) == set(val_scores) and set(test.seed) == set(seeds)
    counts = np.array(coverage["train_counts"])
    table = []
    val_f1 = {}
    times, lengths = [], []
    for name, trial in val.groupby("trial"):
        config = read(root / name / "config.json")
        per_seed, gallop_counts = [], []
        class_f1 = []
        n_seen = np.zeros(len(CLASSES))
        applied = {op: np.zeros(len(CLASSES)) for op in config["operations"]}
        assert set(trial.seed) == set(seeds)
        for seed, frame in trial.groupby("seed"):
            check_frame(frame, manifest, "validation")
            score = f1_score(frame.label, frame.prediction, labels=CLASSES, average="macro")
            per_seed.append(score)
            class_f1.append(
                f1_score(frame.label, frame.prediction, labels=CLASSES, average=None, zero_division=0)
            )
            gallop = frame.label.eq("Galloping")
            gallop_counts.append(int(frame.prediction[gallop].eq("Galloping").sum()))
            directory = root / name / str(seed)
            run = read(directory / "validation.json")
            np.testing.assert_allclose(score, run["validation_macro_f1"], atol=1e-12, rtol=0)
            assert (directory / "best.pt").is_file()
            check_stopping(read(directory / "history.json"), run, protocol["training"])
            times.append(run["seconds"])
            lengths.append(run["epochs"])
            seen = counts * run["epochs"]
            n_seen += seen
            raw_counts = read(directory / "coverage.json")
            if applied:
                np.testing.assert_array_equal(
                    sum(np.array(v) for k, v in raw_counts.items() if k.startswith("operations_")), seen
                )
                for op in applied:
                    used = np.array(raw_counts[op + "_applied"])
                    assert np.all(used <= raw_counts[op + "_attempt"])
                    applied[op] += used
        np.testing.assert_allclose(np.mean(per_seed), val_scores[name], atol=1e-12, rtol=0)
        table.append((name, per_seed, gallop_counts))
        val_f1[name] = np.mean(class_f1, axis=0)
        for op, totals in applied.items():
            coverage.setdefault("realized_rates", {}).setdefault(name, {})[op] = (totals / n_seen).tolist()

    valid_candidates = {n for n in val_scores if n not in protocol["diagnostics"]}
    assert val_scores[selection["selected"]] == max(val_scores[n] for n in valid_candidates)
    selected_ops = sorted(protocol["strengths_weak_default_strong"], key=val_scores.get, reverse=True)[:2]
    assert selected_ops == selection["top_two"]
    matrix = {}
    for seed, frame in test.groupby("seed"):
        check_frame(frame, manifest, "test")
        per_dog = {}
        for name in ("baseline", candidate):
            mats = np.stack(
                [
                    confusion_matrix(d.label, d[name], labels=CLASSES)
                    for _, d in frame.groupby("dog_id", sort=True)
                ]
            )
            np.testing.assert_array_equal(
                mats.sum(0), metrics["per_seed"][str(seed)][name]["confusion_matrix"]
            )
            per_dog[name] = mats
        matrix[seed] = per_dog
    dogs = np.sort(test.dog_id.unique())
    draws = np.random.default_rng(seeds[0]).integers(len(dogs), size=(2000, len(dogs)))

    def scores(m):
        denom = m.sum(-1) + m.sum(-2)
        return np.divide(
            2 * m.diagonal(axis1=-1, axis2=-2), denom, where=denom > 0, out=np.zeros_like(denom, dtype=float)
        )

    def aggregate(name):
        m = np.stack([matrix[s][name] for s in seeds])
        all_dogs = m.sum(1)
        return {
            "f1": scores(all_dogs).mean(0),
            "per_seed": scores(all_dogs).mean(-1),
            "recall": (all_dogs.diagonal(axis1=1, axis2=2) / all_dogs.sum(2)).mean(0),
            "per_dog": scores(m).mean(axis=(0, 2)),
            "matrix": all_dogs.mean(0),
            "boot": scores(m[:, draws].sum(2)).mean(axis=(0, 2)),
        }

    base, other = [aggregate(name) for name in ("baseline", candidate)]
    gap = other["per_seed"] - base["per_seed"]
    ci = np.quantile(other["boot"] - base["boot"], [0.025, 0.975])
    np.testing.assert_allclose(gap.mean(), metrics["mean_delta"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(ci, metrics["paired_dog_bootstrap_95ci"], atol=1e-12, rtol=0)
    gallop_val = manifest[(manifest.split == "validation") & (manifest.label == "Galloping")]
    lines = [
        "# RQ2 · CNN 数据增强结果",
        "",
        (
            f"共核验 45 次训练，每个设置三个匹配种子。原样验证犬选中 **{describe(selection['selected'])}**。"
            f"测试比较覆盖 {len(dogs)} 只犬、{len(test) // len(seeds):,} 个原始窗口。"
        ),
        "",
        (
            f"恒定学习率 {protocol['training']['lr']:.0e}，FP32，batch size {protocol['training']['batch_size']}。"
            f"上限 {protocol['training']['max_epochs']} 轮，每轮验证，早停耐心 {protocol['training']['patience']} 轮。"
            f"实际训练 {min(lengths)}–{max(lengths)} 轮，中位数 {np.median(lengths):.0f} 轮。"
            f"一次训练耗时 {min(times):.1f}–{max(times):.1f} 秒，中位数 {np.median(times):.1f} 秒。"
            f"45 次训练累计 {sum(times) / 60:.1f} 分钟（含逐轮验证与保存权重）。"
        ),
        "",
        "## 原样测试犬",
        "",
        "| 设置 | 三种子平均 macro-F1 | 三个种子分别得分 |",
        "|---|---:|---|",
    ]
    for name, results in (("baseline", base), (candidate, other)):
        values = results["per_seed"]
        lines.append(
            f"| {describe(name)} | {values.mean():.4f} | " + " / ".join(f"{v:.4f}" for v in values) + " |"
        )
    lines += [
        "",
        f"增强相对基线的差值为 **{100 * gap.mean():+.2f} 个百分点**，"
        f"按犬配对的 95% bootstrap 区间为 [{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]。"
        f"逐种子差值为 " + " / ".join(f"{100 * v:+.2f}" for v in gap) + " 个百分点。"
        "区间使用 2,000 次犬级重抽样，三个种子固定。测试犬沿用 RQ1 的固定划分。",
        "",
        "| 行为 | 测试窗数 | 基线 F1 | 增强 F1 | 基线召回 | 增强召回 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for i, c in enumerate(CLASSES):
        support = int(test.label.eq(c).sum() // len(seeds))
        lines.append(
            f"| {CLASS_NAMES[i]} | {support} | {base['f1'][i]:.3f} | {other['f1'][i]:.3f} | "
            f"{base['recall'][i]:.3f} | {other['recall'][i]:.3f} |"
        )
    lines += [
        "",
        "## 验证集与强度选择",
        "",
        (
            f"首轮物理增强的前两项为{'、'.join(describe(n) for n in selected_ops)}。"
            f"各自最佳强度为 `{selection['best_strengths']}`。三个诊断条件只参与单项验证。"
            f"验证集的疾跑来自 {gallop_val.dog_id.nunique()} 只犬，共 {len(gallop_val)} 窗。"
        ),
        "",
        "| 配置 | 验证 macro-F1 均值 | 各种子验证 macro-F1 | 各种子正确疾跑窗数 |",
        "|---|---:|---|---|",
    ]
    for name, values, correct in sorted(table, key=lambda t: -val_scores[t[0]]):
        lines.append(
            f"| {describe(name)} | {np.mean(values):.4f} | "
            + " / ".join(f"{v:.4f}" for v in values)
            + " | "
            + " / ".join(map(str, correct))
            + " |"
        )
    lines += [
        "",
        "各配置在原样验证窗上的分行为 F1 为三个种子的均值。",
        "",
        "| 设置 | " + " | ".join(CLASS_NAMES) + " |",
        "|---|" + "---:|" * len(CLASSES),
    ]
    for name in sorted(val_scores, key=val_scores.get, reverse=True):
        lines.append(f"| {describe(name)} | " + " | ".join(f"{f:.3f}" for f in val_f1[name]) + " |")
    lines += [
        "",
        "## 各犬与误判方向",
        "",
        "每犬 macro-F1 采用固定七类，缺失且未预测的类别按 0 计。表中差值为三种子平均。",
        "",
        "| 犬 ID | 基线 | 增强 | 差值（百分点） |",
        "|---|---:|---:|---:|",
    ]
    for d, a, b in zip(dogs, base["per_dog"], other["per_dog"]):
        lines.append(f"| {d} | {a:.3f} | {b:.3f} | {(b - a) * 100:+.2f} |")
    lines += [
        "",
        "下表为每个种子的误判窗口数取平均，方向为真实行为到预测行为。",
        "",
        "| 误判方向 | 基线 | 增强 |",
        "|---|---:|---:|",
    ]
    for family in (("Sitting", "Standing", "Lying chest"), ("Walking", "Trotting", "Galloping")):
        for a in family:
            for b in family:
                if a != b:
                    i, j = CLASSES.index(a), CLASSES.index(b)
                    lines.append(
                        f"| {CLASS_NAMES[i]} → {CLASS_NAMES[j]} | {base['matrix'][i, j]:.1f} | "
                        f"{other['matrix'][i, j]:.1f} |"
                    )
    lines += [
        "",
        "## 训练覆盖率",
        "",
        "以各类窗口实际经历的训练轮次为分母。下表汇总默认强度的三个种子；组合与其他强度见覆盖率文件。",
        "",
        "| 操作 | " + " | ".join(CLASS_NAMES) + " |",
        "|---|" + "---:|" * len(CLASSES),
    ]
    for name in [*protocol["strengths_weak_default_strong"], *protocol["diagnostics"]]:
        lines.append(
            f"| {describe(name)} | "
            + " | ".join(f"{v:.1%}" for v in coverage["realized_rates"][name][name])
            + " |"
        )
    lines += [
        "",
        (
            "[完整指标](metrics.json) · [逐窗测试预测](predictions.csv) · "
            "[验证选择](selection.json) · [实际覆盖率](coverage.json) · [训练协议](protocol.json)"
        ),
        "",
    ]
    (root / "coverage.json").write_text(json.dumps(coverage, indent=2, ensure_ascii=False))
    (root / "README.md").write_text("\n".join(lines))


def check_frame(frame, manifest, split):
    original = manifest[manifest.split.eq(split)]
    pd.testing.assert_index_equal(pd.Index(frame.window_id), original.index)
    np.testing.assert_array_equal(frame.dog_id, original.dog_id)
    np.testing.assert_array_equal(frame.label, original.label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    report(parser.parse_args().run)
