# 犬只 IMU 小模型与 VLM 研究

研究同一段犬只项圈六路 IMU 信号：直接处理时序的小模型，与信号图像化后输入 VLM 的识别能力、标注需求和计算效率。

已完成固定数据划分、随机森林与 CNN 基线、20 组 CNN 超参数搜索。Qwen3.5-4B/27B 已下载并通过本地文本和图片推理验证；已完成两份 Qwen3.5 中文提示的历史实验；本轮已完成 Qwen3.8-27B 英文提示实验，所有后续VLM推理入口统一使用vLLM，旧结果保留。研究结论和指标见 [results/README.md](results/README.md)，研究进度和后续边界见 [HANDOFF.md](HANDOFF.md)。

## 目录

```text
pet_movement/
├── README.md                     # 项目入口、目录和运行说明
├── HANDOFF.md                    # 研究交接、结果及边界
├── requirements.txt              # 当前 Linux 运行环境依赖版本
├── .venv/                        # 当前机器的 Python 环境
├── dataset/
│   ├── 数据处理与任务定义.md     # 任务、标签、窗口筛选和犬只划分说明
│   ├── 犬只运动传感器数据集/     # 原始发行文件，保留原样
│   ├── vlm/                      # 固定提示、文献记录、绘图配置和匿名WebP
│   └── splits/
│       ├── dog_splits.json        # 固定犬只划分及随机种子
│       └── windows.csv           # 固定窗口、真实标签、split、window_id
├── models/
│   ├── model_manifest.json        # Qwen 官方来源、固定版本及校验值
│   ├── random_forest.joblib
│   ├── cnn.pt
│   ├── cnn_best.pt
│   ├── Qwen3.5-4B/
│   ├── Qwen3.5-27B/
│   └── Qwen3.8-27B/
├── scripts/
│   ├── baseline.py               # 原始数据审计和基线训练
│   ├── cnn_search.py             # 20 组 CNN 参数搜索
│   ├── download_models.py        # Qwen 下载和完整性校验
│   ├── render_imu.py             # 训练集固定尺度的六通道WebP绘图
│   ├── qwen_classify.py          # 完整测试集图像分类及评价
│   ├── run_qwen.py               # 离线文本/单图推理
│   ├── qwen                      # 使用项目环境的快捷入口
│   └── assets/smoke_image.png     # 基础推理验证用图
├── results/
│   ├── README.md                 # 实验报告
│   ├── README.pdf                # 整理前报告的四页历史导出
│   ├── protocol.json             # 原实验协议、版本、参数和划分快照
│   ├── cnn/
│   │   ├── metrics.json          # 原始CNN与验证最优CNN指标
│   │   ├── predictions.csv       # 两份CNN的预测和置信度
│   │   ├── training_history.csv
│   │   ├── cnn_search.csv
│   │   └── cnn_search_history.csv
│   ├── random_forest/
│   │   ├── metrics.json          # 随机森林指标与特征提取耗时
│   │   └── predictions.csv
│   ├── qwen_4b/                  # 文献先验零样本图像分类结果
│   ├── qwen_27b/                 # 文献先验零样本图像分类结果
│   ├── qwen3_8_27b/               # vLLM＋英文提示的Qwen3.8实验
│   ├── kimi/                     # k3.1_pro_acl/：官方API行为分类结果
│   └── model_checks/             # 下载日志、模型基础验证记录
└── 文档/                         # 研究方案等材料
```

`dataset/cache/` 仅在需要重建数值缓存时由审计脚本生成，保存 `pet_windows.npz` 和 `pet_audit.json`。缓存可从原始数据重建，不是划分的依据。

## 固定数据划分

处理流程、混合行为窗口的处理及任务适用范围见 [数据处理与任务定义](dataset/数据处理与任务定义.md)。

以 [dataset/splits/dog_splits.json](dataset/splits/dog_splits.json) 和 [dataset/splits/windows.csv](dataset/splits/windows.csv) 为准。随机种子为 20261101，按犬只隔离，同犬的所有场次属于同一集合。

|集合|犬只|窗口|
|---|---:|---:|
|train|27|16,461|
|validation|9|4,727|
|test|9|3,708|

原始数据不复制成三份。`windows.csv` 用犬只、场次和时间定位原始信号，包含标签和 split；`results/cnn/predictions.csv` 和 `results/random_forest/predictions.csv` 分别保存对应模型的预测及置信度，通过唯一的 `window_id` 与窗口表一对一关联。窗口表及两个模型预测表各保留全部 24,896 行，未做推理的训练/验证行预测为空。

`results/protocol.json` 保留原实验使用的犬只划分快照，并记录当前清单的相对路径。训练脚本读取数据集中的固定划分；搜索脚本同时核对协议快照，避免两份记录不一致。

## 本地 Qwen 推理

在项目根目录运行：

```bash
./scripts/qwen --model 4B --device cuda:0 --prompt '请用中文介绍你自己。'
./scripts/qwen --model 27B --device cuda:1 --prompt '请用中文介绍你自己。'
./scripts/qwen --model 4B --prompt '描述这张图片。' --image scripts/assets/smoke_image.png
```

当前入口使用 vLLM，仅离线读取 `models/` 中的权重。每次命令加载所选模型，回答后退出；没有常驻 HTTP 服务。默认 BF16、关闭思考模式、最多生成 256 token；可使用 `--thinking`、`--max-new-tokens`。需要保存本次输出时使用 `--report results/model_checks/自定义名称.json`，避免覆盖已有验证记录。

三份权重来自 Qwen 官方仓库，未量化，包含视觉编码器。来源、固定提交及文件校验值见 `models/model_manifest.json`：Qwen3.5-4B 约 8.70 GiB，两份27B分别约 51.77 GiB。下载恢复入口为 `.venv/bin/python scripts/download_models.py`，只在需要下载/修复时执行。

`results/model_checks/` 保留已完成的文本和红色图片测试输出、载入耗时、显存记录与下载日志。这些是功能验证，不是犬只行为分类成绩。历史 JSON/日志中的 `deployment/smoke_image.png` 是当时的路径；对应图片现位于 `scripts/assets/smoke_image.png`，历史记录原文保留。

## Python 环境与实验复现

当前 `.venv/` 继承 `/home/exet6233/miniconda3/envs/ecg` 的系统包，额外安装 accelerate 1.15.0。整理时已在新位置离线重建环境入口并修正命令路径；它依赖本机基础环境，不能直接复制到其他机器使用。当前版本见 `requirements.txt`，该文件不包含所有传递依赖，CUDA 版 PyTorch 仍依赖对应安装源和驱动。

本机有两张约 96 GiB 显存的 NVIDIA RTX PRO 6000 Blackwell。现有 Qwen 验证使用 Transformers 的 PyTorch 实现，未安装额外 flash-linear-attention/causal-conv1d 内核。短输入验证耗时不能代表最大吞吐。

原始基线是在 Mac 的另一套环境训练的，精确历史版本在 `results/protocol.json`。当前 Linux 环境版本不同；保留的指标和耗时仍是历史实验记录，不应当作在本机重新训练的成绩。

需要重建数值缓存时，可在项目根目录执行：

```bash
.venv/bin/python scripts/baseline.py audit
```

该命令只生成 `dataset/cache/` 下的数值与审计缓存，不修改固定清单或训练模型。本次目录整理没有执行该命令，没有重训或重新下载权重。

`baseline.py train` 会覆盖已有基线权重和部分结果；`cnn_search.py` 会写入搜索与评估结果。只有明确安排新实验并保护当前结果后才运行。模型 `.joblib` 和 `.pt` 只加载可信来源的文件。

## 本次目录整理

2026-11-02：代码统一到 `scripts/`，模型保留在 `models/`，原 `deployment/` 的验证输出归入 `results/model_checks/`；固定犬只划分和窗口清单归入 `dataset/splits/`，原混合窗口表的预测部分拆到 `results/cnn/predictions.csv` 和 `results/random_forest/predictions.csv`。新增的 `window_id` 由犬只、场次和起始时间确定。

模型权重、原始数据、指标、训练记录和历史验证输出均保留。`results/README.pdf` 保留原导出文件，里面的旧目录路径仅代表导出时的状态；当前路径以本文件及更新后的 Markdown 报告为准。

整理验证已通过：24,896 行窗口与预测可逐字段还原原表；犬只划分互斥、窗口不重叠、七类覆盖及数量一致；三个模型的测试指标由拆分后的表格重算一致。前次整理验证了原有数据/模型文件的完整性；本轮仅将数据目录改名，文件内容保持不变；指标、训练记录、PDF及历史验证输出保持原内容。虚拟环境激活和所有脚本的帮助入口均可从项目外运行。本轮未加载模型重新推理。

2026-11-02：项目文件夹已改名为 `pet_movement`。历史验证 JSON/日志中的原项目根目录 `/home/exet6233/pet/宠物医疗` 对应现在的 `/home/exet6233/pet/pet_movement`；这些历史输出保留原文，活动脚本及虚拟环境已适配新目录。

2026-11-02：数据目录改名为 `dataset/`；指标和预测按模型拆分到 `results/cnn/` 与 `results/random_forest/`，所有 CNN 训练和搜索记录移入 `results/cnn/`。`protocol.json`、对比报告及历史 PDF 保留在 `results/` 根目录。`qwen_4b/`、`qwen_27b/`、`kimi/` 当时按要求创建为空，随后经授权填入实验结果；已有 Qwen 功能验证仍保留在 `model_checks/`。

## Qwen图像分类实验

固定输入及文献依据见 [dataset/vlm/research.md](dataset/vlm/research.md)，模型实际文本见 [prompt.txt](dataset/vlm/prompt.txt)。每张图片保留原窗口200点、六通道，采用训练集确定的固定尺度；边界外点在images.csv中记录。两份模型使用相同3,708个测试窗口、相同提示和相同图片，关闭思考，贪心输出七类之一。

结果分别位于 results/qwen_4b/ 和 results/qwen_27b/。本轮两模型均已完成3,708窗推理与评价，查看各目录的README.md、metrics.json和predictions.csv；progress.json保留最终完成计数。responses.jsonl保留逐窗原始输出；protocol.json记录固定配置与指纹。每窗独立对话，不传入犬只ID、Task、真实标签或原始文件名。

本轮全量结果已完成核对，详见 [Qwen 4B报告](results/qwen_4b/README.md)、[Qwen 27B报告](results/qwen_27b/README.md)及[跨模型比较](results/README.md)。原始回答、预测CSV和重算指标一致；冻结输入与原基线文件的指纹未变。

## 2026-11-03：Qwen3.8与英文提示

按用户要求，`dataset/vlm/prompt.txt`现为英文；原中文保留在`dataset/vlm/prompt_zh.txt`和两份旧结果目录内。旧中文实验的实际推理代码原样保存在`scripts/archive/qwen_classify_20261102.py`，其指纹与旧runtime记录一致。历史结果不重跑或覆盖。

新模型为官方固定版本Qwen/Qwen3.8-27B，模型文件全部核对大小，18个权重分片通过官方SHA-256校验。当前项目环境已有vLLM 0.26.0，不需要升级基础环境。通用`scripts/qwen`与完整实验`scripts/qwen_classify.py`的实际推理均已切换vLLM；Transformers仅用于处理器和聊天模板。

新实验沿用原3,708张匿名WebP、七类任务和训练集尺度；英文说明见[research_en.md](dataset/vlm/research_en.md)。结果位于`results/qwen3_8_27b/`，运行时查看progress.json。模型版本、提示语言和框架均发生变化，不能将与旧结果的差异单独归因于其中一项。

通用离线入口：`./scripts/qwen --model 3.8-27B --device cuda:0 --prompt 'Briefly introduce yourself.'`。完整任务入口为`.venv/bin/python scripts/qwen_classify.py run --model 3.8-27B --device cuda:0 --batch-size 32`；当前任务运行时不要重复启动同一结果目录。原结果目录有固定协议保护，若为旧模型安排新vLLM实验，需显式指定新的--output-dir。

Qwen3.8的3708窗vLLM英文实验已完成并验证，见[完整报告](results/qwen3_8_27b/README.md)。旧中文结果保持原样，当前共享prompt.txt为英文。

## ACL论文绘图适配

已完成同一3708窗的Qwen3.8-27B/vLLM零样本新实验。新图片与[方法说明](dataset/vlm_acl/method.md)在dataset/vlm_acl/；[新报告](results/qwen3.8_27B_acl/README.md)在results/qwen3.8_27B_acl/。旧图片及全部历史结果保留原样。

## Kimi K3.1 Pro API测试

2026-11-04已完成同一3708个ACL图像窗口，结果保存在[results/kimi/k3.1_pro_acl](results/kimi/k3.1_pro_acl/README.md)。使用官方国际API和相同英文输入；脚本scripts/kimi_classify.py读取MOONSHOT_API_KEY环境变量。密钥未写入项目，旧结果保持原样。

## Qwen3.5-4B LoRA

已完成r=4/alpha=8的ViT、视觉投影与注意力LoRA训练及同3708窗vLLM测试。[模型](models/Qwen3.5-4B-lora/README.md)、[完整报告](results/qwen3.5_4B_lora_acl/README.md)、[匹配未微调基线](results/qwen3.5_4B_acl/README.md)。通用推理使用--model 4B-lora；旧实验结果保留。

## 普通六子图与IMU2IMG

2026-11-04已完成同一未微调Qwen3.5-4B＋vLLM＋英文的两组3708窗测试：[普通六子图](results/qwen3.5_4B_sixplot_en/README.md)、[IMU2IMG适配](results/qwen3.5_4B_imu2img_en/README.md)。[图片与方法](dataset/vlm_imu2img/method.md)、[转换器训练](results/imu2img_converter/README.md)。Qwen均保持零样本，IMU2IMG转换器按用户确认使用训练标签。旧结果保留。

## 冻结Qwen的IMU2IMG对齐

已完成简化英文提示对照和仅训练835584转换器参数的冻结Qwen监督对齐。原Qwen权重不变；[完整测试报告](results/qwen3.5_4B_imu2img_aligned_en/README.md)、[匹配旧转换器基线](results/qwen3.5_4B_imu2img_simple_en/README.md)、[训练与验证](results/imu2img_qwen_alignment/README.md)、[模型](models/IMU2IMG-Qwen3.5-4B-aligned/README.md)。

## 三组相同LoRA配置实验

已完成普通六子图、冻结IMU2IMG与联合IMU2IMG三组LoRA训练及相同3708窗测试。[完整比较](results/lora_comparison/README.md)。
