# 犬只 IMU 行为识别

项目使用六轴项圈信号预测犬只的七类行为。当前研究比较 CNN、Random Forest 与 InrtlWrist，并探索数据增强、辅助特征和跨物种迁移。

## 从这里开始

| 内容 | 入口 |
|---|---|
| 数据集、样本数和划分 | [IMU 数据集](docs/imu-datasets.md) |
| 实验结果 | [RQ1–RQ4 与 VLM 输出索引](outputs/README.md) |
| 研究进度与归档 | [HANDOFF.md](HANDOFF.md) |

犬只划分固定为 27 只训练、9 只验证、9 只测试。新实验的模型输入统一为 **50 Hz**。RQ4 的 CNN 和 InrtlWrist 共用六轴、96 点的窗口，并在同一 3,708 个测试窗上比较。[犬只划分](data/splits/dog_splits.json) · [窗口清单](data/splits/windows.csv)

## 本地目录

| 入口 | 内容 |
|---|---|
| `scripts/` | 实验入口、数据核验、历史推理评估和复现脚本 |
| `scripts/pet_imu/` | 加载、增强、预处理、模型、训练和评估 |
| `docs/` | 数据和实验协议 |
| `data/` | 原始数据、划分和缓存，指向 `~/PlayGround/Data/imu-pet_data/` |
| `models/` | 下载的预训练权重，指向 `~/PlayGround/Models/` |
| `outputs/` | 指标、预测、协议和 IMU 训练权重，指向 `~/PlayGround/Outputs/imu-pet_outputs/` |

## 运行当前 50 Hz 实验

训练依赖放在 macOS 的 `collar` conda 环境。原始 DogMove 数据已就位但数值缓存缺失时，先运行 `scripts/baseline.py audit`。

```bash
conda activate collar
python -m pip install -r scripts/requirements_inrtlwrist.txt
python scripts/train_rq4_cnn_baseline.py --device mps --output outputs/RQ4/cnn_50hz_baseline_repeat
```

基线复跑使用新的输出目录。模型恢复入口为 `scripts/download_inrtlwrist.py`。[迁移实验协议](docs/inrtlwrist-experiments.md) · [数据来源](data/transfer/README.md)

## 推理与历史记录

- `scripts/run_cnn.py` 和 `scripts/demo_cnn.py` 用于既有 100 Hz CNN 演示和校准结果。新 50 Hz CNN 的权重与输入参数位于对应 RQ4 运行目录。[CNN 校准](outputs/RQ1/baselines/cnn_temperature_calibration/README.md) · [50 Hz CNN](outputs/RQ4/cnn_50hz_baseline_v1/README.md)
- VLM 保存 API 与模型推理结果。本地 Qwen 入口是 `scripts/qwen`，运行在配置了 vLLM 和模型权重的 CUDA 主机上。macOS 训练环境和 CUDA 推理依赖分别见 `scripts/requirements_inrtlwrist.txt` 与 `requirements.txt`。[VLM 输出](outputs/vlm/README.md)
- 归档按原目录结构保存，可在 `~/PlayGround/Archive/imu-pet/2026-11-18-project-cleanup/` 查找已停用代码、旧输出和整理前的说明。[归档清单](outputs/records/cleanup_manifest.json)
