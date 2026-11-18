# 犬只 IMU 项目交接

## 当前工作

后续模型输入统一为 **50 Hz**。RQ4 使用六轴、96 点窗口，固定 27/9/9 只犬的划分。50 Hz CNN 已完成三种子训练，平均验证和测试 macro-F1 为 0.7857 和 0.7860。相同输入下，现有 InrtlWrist 直接犬端微调的测试 macro-F1 为 0.8184，作为域适应对照。[统一输入协议](docs/imu-datasets.md#后续实验的统一输入) · [RQ4 报告](outputs/RQ4/README.md)

跨域对齐与山羊／绵羊迁移仍属于后续实验。旧 HAPT 顺序微调实验已归档。RealWorld HAR 的六轴原始文件和山羊／绵羊数据均已核验。RealWorld HAR 与山羊／绵羊的项目划分待定。[数据集及划分](docs/imu-datasets.md) · [迁移下载清单](data/transfer/README.md)

## 现行实验入口

| 工作 | 入口 |
|---|---|
| 原始数据核验及缓存重建 | `scripts/baseline.py audit` |
| RQ1 学习率和模型对照 | `scripts/search_rq1_lr.py`、`scripts/compare_rq1_*.py` |
| RQ2 数据增强 | `scripts/train_cnn_augmentation.py`、`scripts/train_inrtlwrist_augmentation.py` |
| RQ3 频域及辅助特征 | `scripts/train_cnn_frequency.py`、`scripts/train_rq3_*_fusion.py` |
| RQ4 50 Hz 犬端 CNN 基线 | `scripts/train_rq4_cnn_baseline.py` |

macOS 实验使用 `collar`、PyTorch MPS 和 FP32。训练使用验证 macro-F1 选最佳检查点，测试只使用固定测试犬。复现旧跑次按对应运行协议；另跑实验使用新输出目录。[结果索引](outputs/README.md)

## VLM 与归档

已保存 Kimi、Sol 完整测试和 Qwen 推理结果。Sol 的原 Batch 曾按用户要求停止，获授权的 107 窗补跑已完成。原 Batch 的停止记录仍在，对应提交入口拒绝恢复这个旧任务。[VLM 结果](outputs/vlm/README.md) · [Sol 完整测试](outputs/vlm/api/openai/gpt6_1_sol_sixplot_batch/full_test/README.md)

训练产物保存在各 IMU 实验目录，`models/` 用于下载的预训练权重。历史代码、旧跑次、VLM 训练记录以及整理前的 README 和交接全文保存在 `~/PlayGround/Archive/imu-pet/2026-11-18-project-cleanup/`，目录层级对应原项目。恢复时查对归档清单里的原路径，并使用新的实验输出目录。[归档清单](outputs/records/cleanup_manifest.json) · [早期路径迁移记录](outputs/records/output_layout.json)
