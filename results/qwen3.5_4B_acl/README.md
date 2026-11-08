# Qwen3.5-4B：同ACL图片与英文提示的匹配基线

2026-11-04，已用vLLM完成3708测试窗，未微调。Macro-F1=0.075361，balanced accuracy=0.209059，accuracy=0.162352，解析成功率=100.00%。

本目录为LoRA实验配套的新基线，完整比较与设置见[LoRA报告](../qwen3.5_4B_lora_acl/README.md)。原中文六子图历史结果保留在results/qwen_4b/。输入、提示、处理器、采样、vLLM与LoRA测试一致，基础权重未更新。原回答、CSV和独立重算一致，见validation.json。全部响应已存在，请勿重复生成。
