# Qwen3.5-4B-lora：ViT＋视觉投影＋混合注意力 LoRA

2026-11-04，已完成 **3个epoch的监督微调**，并用vLLM完成相同 **3,708个测试窗口**。模型：[Qwen3.5-4B-lora](../../models/Qwen3.5-4B-lora/)，选中第 **1** 个epoch。

## 测试结果

|同一ACL图片、英文提示、vLLM|Macro-F1|Balanced accuracy|Accuracy|
|---|---:|---:|---:|
|未微调Qwen3.5-4B|0.075361|0.209059|0.162352|
|Qwen3.5-4B-lora|0.673548|0.676463|0.623247|

LoRA解析成功率100.00%，状态`{"ok": 3708}`。基线状态`{"ok": 3708}`。解析失败/拒绝/截断一律计为错误，不按真值重试。

本轮新增了严格匹配的[未微调4B基线](../qwen3.5_4B_acl/README.md)，模型基础提交、像素、英文文本、处理器、采样和vLLM设置均一致，改变项为选中的LoRA。旧中文六子图结果仍在results/qwen_4b/，不作为本轮LoRA直接对照。微调使用16,461个标注训练窗，基线为零样本，比较时应同时报告这个标注预算。

## 训练范围与参数

- r=4、alpha=8、缩放2、LoRA dropout0.05。只训练A/B矩阵，基础权重冻结；没有把ViT整塔全参数解冻。
- 96个ViT线性层（各block的qkv、proj、MLP两层）；2个视觉merger投影；152个语言投影（完整注意力Q/K/V/O，以及线性注意力qkv、z、b、a和out），共250模块。
- 可训练参数 **5,226,496**，占含适配器总参数4,544,492,032的 **0.1150%**。卷积、归一化、嵌入和语言MLP冻结，精确名称见target_modules.json。
- 3epoch、AdamW、LR2e-4、betas(0.9,0.999)、epsilon1e-8、weight decay0.01、5%warmup后线性衰减、梯度裁剪1。BF16基础权重、FP32适配器和优化器。双卡每卡microbatch8、累积2，有效batch32，1545个优化步。
- 仅监督最终JSON答案与EOS，屏蔽全部图片/问题/非思考前缀和padding；只计算答案位置的词表logits，监督目标不变。未按类别加权或重采样，无图像增强。
- 所有原16,461训练窗参与，DDP每epoch为对齐两rank额外复制1个窗；总样本呈现49,386次。4,727验证窗完整且不重复地分配至两卡，只用于回答token NLL选checkpoint；测试不参与训练、早停或选择。
- 选中epoch=1，验证回答NLL=0.080381；训练成功会话墙钟3954.84秒。初始化失败、数据渲染和后续vLLM测试计时另列，不混入此值。

|epoch|验证回答NLL|本epoch墙钟秒|
|---|---:|---:|
|1|0.080381|1332.55|
|2|0.093438|1310.31|
|3|0.114947|1311.80|

## 每类结果

|类别|Precision|Recall|F1|窗口数|
|---|---:|---:|---:|---:|
|Galloping|1.0000|0.6897|0.8163|29|
|Lying chest|0.4375|0.6640|0.5274|869|
|Sitting|0.4163|0.2509|0.3131|813|
|Sniffing|0.9343|0.9955|0.9639|671|
|Standing|0.3149|0.1882|0.2356|574|
|Trotting|0.9788|0.9976|0.9881|416|
|Walking|0.8035|0.9494|0.8704|336|

按测试犬整体bootstrap 2,000次的95%区间：

- macro_f1：[0.637087, 0.704737]
- balanced_accuracy：[0.635986, 0.722707]
- accuracy：[0.561417, 0.667084]

只有9只测试犬，区间不覆盖训练种子/超参数选择方差。本轮仅一个训练种子。

## 数据、合并与测试

保持2秒、200点、六通道和原犬只划分；训练尺度、标准化和固定纵轴全部沿用dataset/vlm_acl/。新增训练/验证图片位于[supervised](../../dataset/vlm_acl/supervised/)，测试图片保持原样，匿名文件名和像素不含标签、犬只ID或Task。

训练使用Transformers＋PEFT，所有生成式测试均用vLLM0.26.0。选择后的视觉、投影、注意力适配器全部合并成完整BF16权重，逐个检查250个合并矩阵与基础权重＋rank4更新完全相等；保留adapter/与3份epoch适配器用于复现。不是仅加载语言侧LoRA。基础模型原权重的官方SHA-256全部未变。

本轮和基线均关闭思考、温度0、最多32个输出token，不使用标签约束解码；每窗输入1022 token，图像512×409处理为512×416、208视觉token。模型只接收像素与固定英文文本，不传入外部文件名/真值/犬只ID。测试仍只覆盖真值筛选后整窗一致的七种行为，不覆盖窗内切换。

## 文件

- [training_protocol.json](training_protocol.json)、[target_modules.json](target_modules.json)：冻结配置、精确范围与参数。
- [training_history.jsonl](training_history.jsonl)、[validation_history.jsonl](validation_history.jsonl)、[checkpoint_selection.json](checkpoint_selection.json)：训练、验证及选择记录。
- [backward_check.json](backward_check.json)：训练窗技术检查，三个范围都有非零梯度且未执行优化。
- [export_validation.json](export_validation.json)、[validation.json](validation.json)：合并指纹、250矩阵检查、原回答/预测/独立重算一致性和历史数据保护。
- [protocol.json](protocol.json)、[prompt.txt](prompt.txt)、[predictions.csv](predictions.csv)、[responses.jsonl](responses.jsonl)、[metrics.json](metrics.json)：实际测试配置、全部预测、原回答与完整指标。
- [runtime.json](runtime.json)：vLLM批计时与NVML进程树显存观测，包含缓存预留，不是精确分配器峰值。

私有训练依赖在.venv/lora-packages；未升级原torch/transformers/vLLM。默认NCCL P2P启动曾在优化前停留，归档startup_attempts/nccl_default.log；正式运行使用NCCL_P2P_DISABLE=1、NCCL_IB_DISABLE=1。训练技术检查还补齐了新版处理器要求的多模态位置类型标记。所有技术失败发生在正式优化前，不作为模型分类失败。

## 复现

```sh
.venv/bin/python -m pip install --no-deps --target .venv/lora-packages -r scripts/requirements_lora.txt
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 .venv/bin/python -m torch.distributed.run --standalone --nnodes 1 --nproc_per_node 2 scripts/train_qwen35_lora.py train
.venv/bin/python scripts/train_qwen35_lora.py export
.venv/bin/python scripts/eval_qwen35_lora.py run --variant base
.venv/bin/python scripts/eval_qwen35_lora.py evaluate --variant base
.venv/bin/python scripts/eval_qwen35_lora.py run --variant lora
.venv/bin/python scripts/eval_qwen35_lora.py evaluate --variant lora
```

实验已完成，无需重训或重复生成。仅重算evaluate不加载GPU模型。通用推理入口：`./scripts/qwen --model 4B-lora --device cuda:0 --prompt 'Briefly introduce yourself.'`。

独立合并校验采用PEFT安全合并的实际顺序：先将FP32 LoRA更新转为BF16，再与BF16基础权重相加。250个矩阵逐位相等，最大误差0；首次校验公式使用不同舍入顺序的诊断保存在final_validation_rounding_check.log，修正仅涉及校验器，未重训或重跑测试。
