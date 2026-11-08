# 旧IMU2IMG图片的英文提示对照

同一冻结Qwen3.5-4B、vLLM和4,727个验证窗口，分别使用原波形特征提示与简化提示。简化文本在本轮新验证输出前固定，仅保留学习图像的模态、时间顺序、七类任务和JSON规则。

|提示|验证Macro-F1|Balanced accuracy|Accuracy|
|---|---:|---:|---:|
|原波形特征提示|0.026537|0.142857|0.102391|
|简化提示|0.026537|0.142857|0.102391|

两组均全部预测Walking，解析成功率100%。没有进行提示搜索。`old_prompt/`与`simple_prompt/`分别保存实际文本、全部原回答、预测表、指标、运行协议及计时；`comparison.json`汇总对照。

简化提示在同一3,708测试窗口的[匹配旧转换器基线](../qwen3.5_4B_imu2img_simple_en/README.md)仍全部Walking。随后只训练图片转换器、保持Qwen全部权重冻结的结果见[对齐实验](../qwen3.5_4B_imu2img_aligned_en/README.md)。测试未用于选择提示或转换器checkpoint。
