# Qwen3.5-4B：IMU2IMG学习图片适配＋vLLM＋英文

2026-11-04，已完成同一批 **3,708个固定测试窗口**。Qwen采用官方基础权重、未微调、无LoRA、无带标签提示示例；温度0、关闭思考、最多32输出token、vLLM0.26.0。两组基础模型、生成参数和核心引擎配置已核对一致；图片前处理下限的差异见下文。

## 对比结果

|未微调Qwen3.5-4B＋vLLM＋英文|Macro-F1|Balanced accuracy|Accuracy|
|---|---:|---:|---:|
|普通六子图|0.098390|0.255631|0.216828|
|IMU2IMG学习图片适配|0.023739|0.142857|0.090615|

本目录解析成功率 **100.00%**，状态`{"ok": 3708}`。失败/拒绝/截断按错误计入完整3708窗，不根据真值重试。

普通组直接复用原六通道曲线，图像1280×1152，实际输入2230token，1440视觉token。IMU2IMG组使用训练得到的224×224 RGB纹理，实际输入930token，49视觉token；图片最小像素预算设为224²以保持7×7布局，普通大图沿用既有256²下限。其他模型与生成设置相同。两组动作定义和JSON规则逐字一致，图像说明及观察汇总句根据实际输入调整。

## IMU2IMG的来源与适配

来源是 [IMU2IMG（UbiComp Companion 2025）](https://doi.org/10.1145/3714394.3756208)，[作者公开PDF](https://gw-jung.github.io/pdf/lee2025imu2img.pdf)。方法使用学习到的编码器/解码器，而非固定曲线图。详细来源、公式与适配在[method.md](../../dataset/vlm_imu2img/method.md)。

本轮将每时刻加速度XYZ和陀螺仪XYZ分别合成RMS，沿用原200点窗口清单，编码器内部中心裁剪为196点、49个4点patch。原Mantis-v1 TGU产生256维token，线性层解码为每块32×32，按时间行优先拼成224×224，经tanh固定映射成RGB。R来自加速度、G来自陀螺仪；无磁力计，B固定中灰128。

按用户确认的方案，16,461个训练窗联合训练图片转换器、论文指定ViT和7类头；4,727验证窗仅用于选模，测试不参与训练。AdamW、LR1e-4、weight decay0.01、batch512、10epoch。选中epoch=7，配套ViT的**验证**Macro-F1=0.748002。此验证分数不是Qwen的测试分数。

**Qwen保持zero-shot；整套IMU2IMG流水线使用了训练标签。** 不能将其称为零标签系统，也不是原论文SHL三模态ViT系统的完整复现。原论文最终使用验证集进一步训练的步骤未采用，以保持本项目原划分。

本轮还同时改变了RMS压缩、编码器内部裁剪、学习到的表示、图像尺寸/视觉预算与图像说明。因此分数差异反映这套转换方案向冻结Qwen的迁移，不能单独归因于“画图方式”，也不能代替原论文端到端ViT分类器的结果。

## 每类测试结果

|类别|Precision|Recall|F1|窗口数|
|---|---:|---:|---:|---:|
|Galloping|0.0000|0.0000|0.0000|29|
|Lying chest|0.0000|0.0000|0.0000|869|
|Sitting|0.0000|0.0000|0.0000|813|
|Sniffing|0.0000|0.0000|0.0000|671|
|Standing|0.0000|0.0000|0.0000|574|
|Trotting|0.0000|0.0000|0.0000|416|
|Walking|0.0906|1.0000|0.1662|336|

9只测试犬整体bootstrap2,000次，95%区间：

- macro_f1：[0.013305, 0.033090]
- balanced_accuracy：[0.142857, 0.142857]
- accuracy：[0.048841, 0.130985]

## 计算记录与文件

引擎初始化累计26.73秒，完整模型运行墙钟150.55秒，正式vLLM调用累计106.08秒。引擎调用包括内部处理、排队、GPU计算与解码，不是纯GPU核函数时间。NVML进程树逐批观测显存峰值81.99GiB，包含缓存预留，不是最低显存需求或精确分配器峰值。

- [predictions.csv](predictions.csv)、[responses.jsonl](responses.jsonl)：3708窗的真值、预测和原回答。
- [metrics.json](metrics.json)：总体、每类、每犬、混淆矩阵和区间；[validation.json](validation.json)是独立重算及原输入/历史文件保护校验。
- [protocol.json](protocol.json)、[prompt.txt](prompt.txt)、[runtime.json](runtime.json)：冻结配置、实际英文文本、引擎版本与计时。
- [转换器训练记录](../imu2img_converter/README.md)、[生成图片](../../dataset/vlm_imu2img/images/)、[转换器模型](../../models/IMU2IMG-pet/README.md)。

仅测试原真值筛选后整窗一致七类行为，不覆盖窗内切换；已查看过此前固定测试成绩，本轮未根据本轮输出调提示、编码或阈值。只有一个训练/推理种子，区间不覆盖训练方差。

复算本目录指标：`.venv/bin/python scripts/eval_qwen35_images.py evaluate --method imu2img`。本轮已全部完成，不必重跑推理或训练。普通组实际源码归档scripts/archive/eval_qwen35_images_sixplot_20261104.py；第2组源码scripts/eval_qwen35_images.py，转换器scripts/train_imu2img.py。

逐类预测数量：`{"Walking": 3708}`。
本轮冻结Qwen对全部3708张学习图像均输出Walking，故balanced accuracy=1/7。这是该学习表示向零样本Qwen迁移失败的观测，不能替代论文原配ViT分类器的结果；未据此重训、改提示或重复测试。
