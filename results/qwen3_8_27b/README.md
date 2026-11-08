# Qwen3.8-27B：vLLM＋英文提示的 IMU 图像七分类

日期：2026-11-03。已使用全部 **3,708 个固定测试窗口**完成本地推理。每窗 2 秒、200 点、六通道，复用已有匿名 WebP，保持相同训练集尺度。实际推理为 **vLLM 0.26.0**，英文文本使用 [prompt.txt](prompt.txt)，没有样例图片或微调。

## 测试结果

|指标|数值|
|---|---:|
|Macro-F1|0.157967|
|Balanced accuracy|0.329281|
|Accuracy|0.205502|
|解析成功率|100.00%|

状态分布：`{"ok": 3708}`。失败按错误计入总体分母，不根据真值重试。预测类别分布：`{"Galloping": 212, "Trotting": 100, "Standing": 2658, "Lying chest": 369, "Walking": 369}`。

测试犬整体 bootstrap 2,000 次，95% 区间：

- macro_f1：[0.094022, 0.245740]
- balanced_accuracy：[0.302155, 0.377148]
- accuracy：[0.137957, 0.292317]

仅9只测试犬；区间不涵盖提示选择或训练方差。

## 每类结果

|类别|Precision|Recall|F1|Support|
|---|---:|---:|---:|---:|
|Galloping|0.1368|1.0000|0.2407|29|
|Lying chest|0.2060|0.0875|0.1228|869|
|Sitting|0.0000|0.0000|0.0000|813|
|Sniffing|0.0000|0.0000|0.0000|671|
|Standing|0.2077|0.9617|0.3416|574|
|Trotting|0.9900|0.2380|0.3837|416|
|Walking|0.0163|0.0179|0.0170|336|

## 推理设置与计算记录

- 官方仓库：[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)，固定提交 `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`。32个文件已核对大小，18个权重分片通过官方SHA-256；下载与校验日志为download.log。
- 原始BF16权重、视觉编码器保留；不量化。单GPU，tensor parallel 1，上下文上限4096，最多32个并发序列，批token上限8192，显存预算0.85；关闭前缀缓存，不使用推测或标签约束解码。
- 每窗独立请求；关闭思考，温度0，最多32个新token。模型只收到RGB图像和英文文本，未传入文件名、犬只ID、Task或真值。
- 实际每窗输入 2230 token，含1440个视觉token；累计输入 8,268,840 token，累计输出 27,200 token。处理器保持1280×1152，像素重建误差0。
- 正式vLLM引擎调用累计 1608.71 秒，包含内部预处理、调度、GPU计算与文本解码，不是纯GPU核函数时间。父进程图像与模板准备累计 15.69 秒。
- 正式图像准备＋引擎请求按窗口摊销，P50 0.4388 秒，P95 0.4414 秒；这是批处理成本，不是单请求延迟。
- 本次成功引擎载入、编译和预热累计 36.45 秒；完整模型进程实测墙钟 1698.24 秒，包括初始化、训练窗技术预检和正式推理。首次形状相关JIT开销保留在实测值中。
- NVML逐批观测的本进程及vLLM子进程显存峰值 82.44 GiB，包含引擎缓存预留；不等于模型最低显存需求，也不等于PyTorch分配器精确峰值。
- 复用的图像历史渲染时间 555.08 秒，本轮未重绘。范围外点仍是468窗、6,751/4,449,600个通道采样点。metrics.json中的“渲染＋推理摊销”是把历史渲染组件加到新推理组件上的估计，不是本轮重新渲染的墙钟时间；均不含2秒信号采集等待。

### 启动兼容性记录

正式样本前有两次技术启动失败：首次FlashInfer采样器JIT缺少nvcc，改为`VLLM_USE_FLASHINFER_SAMPLER=0`；第二次显存统计自定义RPC无法序列化，改为直接NVML统计。两次均未处理正式测试样本，日志与初始协议保存在startup_attempts/。正式运行仍使用vLLM，未启用额外pickle反序列化选项。上面的成功运行计时不包含下载或这两次先前失败的启动。

## 比较范围

旧Qwen3.5实验使用中文提示和Transformers。本轮同时改变模型版本、提示语言及推理框架，分数差异不能单独归因于某一项。旧CNN、随机森林和Qwen3.5结果均保持原指纹，未重跑。

任务仍是对未参与本项目训练的测试犬，在真值筛选后的整窗单一目标行为中做七分类；不覆盖行为切换或全天连续监测。固定测试集成绩此前已被查看；本轮输入在正式推理前冻结，未按本轮测试成绩修改。

## 文件与复现

- [protocol.json](protocol.json)：模型、框架、固定配置与输入指纹。
- [prompt.txt](prompt.txt)：本轮实际英文文本；共享入口为[dataset/vlm/prompt.txt](../../dataset/vlm/prompt.txt)，历史中文为[prompt_zh.txt](../../dataset/vlm/prompt_zh.txt)。
- [文献与英文实验说明](../../dataset/vlm/research_en.md)、[绘图配置](../../dataset/vlm/render_protocol.json)、[处理器检查](../../dataset/vlm/processor_check_qwen38.json)。
- [predictions.csv](predictions.csv)：全部3708窗的真值、预测、原始回答与状态。
- [responses.jsonl](responses.jsonl)：推理时追加的原始记录，逐条注明vLLM后端。
- [metrics.json](metrics.json)：总体、每类、每犬、区间及计算记录。
- [runtime.json](runtime.json)：引擎版本、GPU、逐批时间、NVML显存观测及代码指纹。
- [preflight.json](preflight.json)：训练窗技术检查，不计入正式成绩。
- [validation.json](validation.json)：原始回答、预测与独立重算指标一致性检查。
- progress.json、run.log、download.log及startup_attempts/保留进度和运行历史。

本地通用入口为`./scripts/qwen --model 3.8-27B --device cuda:0 --prompt 'Briefly introduce yourself.'`。全部当前VLM入口已统一vLLM；两份旧Qwen3.5结果仅作历史保留。完整实验脚本为scripts/qwen_classify.py，评价命令为`.venv/bin/python scripts/qwen_classify.py evaluate --model 3.8-27B`。
