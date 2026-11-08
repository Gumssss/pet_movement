# Qwen3.5-4B：旧转换器＋简化英文提示

2026-11-04，已完成同一批 **3,708个测试窗口**，全部生成均使用vLLM0.26.0。Qwen视觉编码器、投影层、语言模型的参数全部冻结，没有加载LoRA或带标签提示示例。对齐系统使用训练标签，所以称“冻结VLM的有监督适配”。

## 测试对照

|同一冻结Qwen3.5-4B、相同3708测试窗|Macro-F1|Balanced accuracy|Accuracy|
|---|---:|---:|---:|
|旧转换器＋原波形先验提示|0.023739|0.142857|0.090615|
|旧转换器＋简化提示|0.023739|0.142857|0.090615|
|Qwen对齐转换器＋同一简化提示|0.125809|0.275129|0.297465|

本目录解析成功率100.00%，状态`{"ok": 3708}`。预测分布`{"Walking": 3708}`。失败/拒绝/截断按错误纳入3708窗分母，未按真值重试。

## 第一步：只更改文本

旧转换器生成的同一4,727验证窗，原提示macro-F1=0.026537，简化提示=0.026537。简化提示在本轮输出前固定，仅保留图像模态、时间patch顺序、任务、七类标签和JSON规则；删除波形幅度/基线/频率等观察先验。没有提示搜索或按测试成绩修改文本。

## 第二步：只训练转换器，通过冻结Qwen对齐

从原ViT监督训练选中的转换器初始化，丢弃原ViT和分类头。16,461训练窗只更新 **835,584个图像转换器参数**；Qwen保持eval、requires_grad=False，但答案梯度可经过其视觉与语言激活回传到像素。没有新增Qwen参数、LoRA或分类头。

保持原两类三轴RMS、196点中心裁剪、49个4点patch、224×224图像和B128。参数/优化器FP32，转换器前向BF16，与旧图精度一致；8位round使用直通估计近似梯度，前向仍是实际8位值。按官方融合归一化`(byte-127.5)/127.5`和Qwen patch排列，训练路径不经过PIL或detach截断。导出仍为相同无损WebP，模型推理49视觉token。

双GPU每卡microbatch16、有效batch32；3epoch、AdamW LR1e-4、wd0.01、5%warmup线性衰减、梯度裁剪1。每epoch完整验证答案NLL只作监控；最终对三个checkpoint分别用vLLM在全部4,727验证窗生成答案，选macro-F1最高者，完全相同时保留较早epoch。

选中 **epoch3**，验证macro-F1=0.128750。正式3,708测试窗只在选择完成后处理，没有用于选checkpoint。原图转换器使用过同一批训练标签，本轮没有新增标注；这是对原转换器的额外监督适配，不能将系统叫作严格zero-shot。

|转换器checkpoint|验证macro-F1|验证balanced accuracy|验证accuracy|
|---|---:|---:|---:|
|epoch1|0.080802|0.186405|0.170721|
|epoch2|0.116907|0.244151|0.278401|
|epoch3|0.128750|0.267882|0.294902|

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

按9只测试犬整体bootstrap2,000次的95%区间：

- macro_f1：[0.013305, 0.033090]
- balanced_accuracy：[0.142857, 0.142857]
- accuracy：[0.048841, 0.130985]

## 验证与计算

Qwen的完整参数内存SHA在训练前后完全相同，原始基础权重文件的官方SHA也未变。优化器严格只包含转换器参数，Qwen梯度始终为空。转换器编码器/解码器都有实际更新。外部官方处理器对照覆盖全部256种8位像素值，BF16 patch逐位相等，梯度路径和答案掩码通过独立检查；200个历史输入/结果文件未变。

训练成功会话墙钟629.01秒；训练前单批16样本峰值分配25.45GiB，正式rank0峰值见training_progress.json。显存与速度对应本轮较短提示和49视觉token配置，不能仅据此归因为可训练参数少。vLLM各数据集runtime.json记录模型请求批计时，不等于纯GPU核函数或单请求延迟。

## 文件

- [predictions.csv](predictions.csv)、[responses.jsonl](responses.jsonl)、[metrics.json](metrics.json)、[validation.json](validation.json)：全量预测、原回答、总体/每类/每犬/区间与独立校验。
- [protocol.json](protocol.json)、[prompt.txt](prompt.txt)、[runtime.json](runtime.json)：固定测试配置、实际英文文本和vLLM计时/代码指纹。
- [训练与选模记录](../imu2img_qwen_alignment/README.md)、[方法说明](../../dataset/vlm_imu2img_qwen/method.md)、[选中转换器](../../models/IMU2IMG-Qwen3.5-4B-aligned/README.md)。

本轮保持原犬只划分、整窗一致七分类和同一测试集；只有一个训练种子，区间不覆盖训练或超参数选择方差。该固定测试集此前已查看，本轮没有根据新测试输出调整方案。图片人类可读性与跨VLM迁移不在本轮优化目标之内。

复算本目录指标：`.venv/bin/python scripts/eval_imu2img_alignment.py evaluate --result-dir results/qwen3.5_4B_imu2img_simple_en`。实验已完成，不必再次训练或生成。
