# Qwen3.8-27B：ACL 论文绘图适配实验

2026-11-03，已用本地 **vLLM 0.26.0** 完成全部 **3,708 个固定测试窗口**。原窗口、犬只划分及旧实验均未改动；每窗仍为2秒、200点、六通道。本轮图片在[dataset/vlm_acl/images](../../dataset/vlm_acl/images/)。

## 方法与比较

参考 [By My Eyes（EMNLP 2024）](https://aclanthology.org/2024.emnlp-main.133/) 和作者固定提交的波形实现，将六路曲线叠加到同一坐标轴，带固定颜色和通道图例，导出512×409无损WebP。仅由训练集拟合每通道均值/标准差及固定共享纵轴；不按目标犬或窗口重新缩放。详情和与作者方法的差异见[method.md](../../dataset/vlm_acl/method.md)。

这是共享坐标波形的**零样本适配**，没有运行论文的自动可视化搜索或加入带标签示例。相比上一轮，改变了标准化、布局、图像尺寸和首段图像说明；Qwen权重、英文动作特征说明、JSON规则、vLLM参数、随机种子及测试窗口相同。分数差异不能单独归因于某一项。

|输入表示|Macro-F1|Balanced accuracy|Accuracy|
|---|---:|---:|---:|
|原六子图，英文，vLLM|0.157967|0.329281|0.205502|
|本轮共享坐标，英文，vLLM|0.123266|0.294250|0.196602|

解析成功率：100.00%；状态：`{"ok": 3708}`。解析失败、拒绝和截断均按错误计算，不按真值重试。预测分布：`{"Galloping": 821, "Standing": 1282, "Lying chest": 1316, "Walking": 251, "Sniffing": 31, "Sitting": 7}`。

按测试犬整体bootstrap 2,000次的95%区间：

- macro_f1：[0.099515, 0.145288]
- balanced_accuracy：[0.276868, 0.308760]
- accuracy：[0.166858, 0.223083]

只有9只测试犬；区间不覆盖提示选择或训练方差。

## 每类结果

|类别|Precision|Recall|F1|窗口数|
|---|---:|---:|---:|---:|
|Galloping|0.0353|1.0000|0.0682|29|
|Lying chest|0.2667|0.4039|0.3213|869|
|Sitting|0.4286|0.0037|0.0073|813|
|Sniffing|0.0000|0.0000|0.0000|671|
|Standing|0.2387|0.5331|0.3297|574|
|Trotting|0.0000|0.0000|0.0000|416|
|Walking|0.1594|0.1190|0.1363|336|

## 固定输入与运行记录

- 模型：Qwen/Qwen3.8-27B，提交`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`；复用已校验的本地完整BF16权重，无量化、微调或新下载。
- 单GPU、TP=1、上下文4096、并发序列32、每批token上限8192、显存预算0.85、关闭前缀缓存。逐窗独立请求，温度0、关闭思考、输出上限32 token；不使用标签约束或推测解码。沿用原vLLM原生采样器设置。
- 图像512×409由Qwen处理器对齐为512×416，208个视觉token；实际每窗输入1022 token，累计输入3,789,576、输出31,561 token。模型仅收到RGB像素及固定英文文本，未传入文件名、犬只ID、Task或真值。
- 新图片渲染累计133.57秒；有343窗存在超出训练固定纵轴的点，共3,536/4,449,600个通道采样点。越界显示截断并加边界标记，逐通道统计在images.csv，原信号仍保留。
- 正式引擎调用累计713.20秒，包括内部预处理、调度、GPU计算和解码；父进程图片/模板准备累计3.23秒。
- 引擎初始化累计35.50秒；模型进程实测墙钟771.24秒，包含初始化、训练窗技术预检及正式推理。批成本摊销P50=0.1941秒/窗、P95=0.1972秒/窗，这不是单请求延迟。
- NVML逐批观测的进程树显存峰值82.35GiB，包含vLLM缓存预留，不是精确分配器峰值或模型最低显存需求。

## 文件与复现

- [predictions.csv](predictions.csv)：3708窗的真值、预测及原回答；[responses.jsonl](responses.jsonl)为推理时原始记录。
- [metrics.json](metrics.json)：总体、每类、每犬、混淆矩阵与计算记录；[validation.json](validation.json)为独立重算校验。
- [protocol.json](protocol.json)、[prompt.txt](prompt.txt)、[runtime.json](runtime.json)：固定输入指纹、实际文本、代码指纹、版本和批计时。
- [preflight.json](preflight.json)：仅训练窗技术检查；run.log、progress.json记录正式进度。
- [图片清单](../../dataset/vlm_acl/images.csv)、[训练标准化参数](../../dataset/vlm_acl/normalization.json)、[绘图协议](../../dataset/vlm_acl/render_protocol.json)、[处理器检查](../../dataset/vlm_acl/processor_check_qwen38.json)、[输入校验](../../dataset/vlm_acl/input_validation.json)。
- [历史文件指纹](prior_files_sha256.json)：57个已有文件均未改变；旧结果见[原英文vLLM报告](../qwen3_8_27b/README.md)。

任务只覆盖真值筛选后整窗一致的七种目标行为，不覆盖窗内行为切换。当前固定测试集此前已被查看；本轮方案在新测试输出前冻结，未根据本轮分数调参。

```sh
.venv/bin/python scripts/render_imu_acl.py prepare
.venv/bin/python scripts/render_imu_acl.py check-processor
.venv/bin/python scripts/qwen_classify.py run --model 3.8-27B --device cuda:0 --batch-size 32 --data-dir dataset/vlm_acl --output-dir results/qwen3.8_27B_acl
.venv/bin/python scripts/qwen_classify.py evaluate --model 3.8-27B --data-dir dataset/vlm_acl --output-dir results/qwen3.8_27B_acl
```

已完成的同目录运行会校验冻结协议并跳过已有响应；方法有变化时使用新的数据/结果目录。
