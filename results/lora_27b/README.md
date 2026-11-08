# 27B IMU2IMG + LoRA 实验

状态：两组双卡检查已通过，Qwen3.5-27B 的100步学习率筛选已启动；完成其正式训练与测试后自动执行 Qwen3.8-27B。

- 两组均从各自原始 BF16 模型创建新 LoRA；ViT、视觉投影、语言混合注意力，r=4、alpha=8、dropout=0.05，共 414 个线性模块、13,881,216 个 LoRA 参数。
- 图像转换器从原 Qwen3.5-4B 对齐版本初始化（835,584 参数），按联合训练准备；Qwen 原基础参数全部冻结。转换器 LR 为 LoRA LR 的 1/20。
- 同一犬只互斥划分：训练 16,461、验证 4,727、测试 3,708 个窗口；同一简短英文提示、224×224 lossless WebP、49 视觉 token，B 通道固定 128。转换器曾使用训练标签，因此整套方法属于监督适配。
- 两个 96GB GPU，容量测试后共同采用每卡 microbatch=96、有效 batch=192、无额外梯度累积。开启 non-reentrant gradient checkpointing。单卡 batch128 的 Qwen3.8 显存余量不足2GiB；更大 batch 不采用。
- 每个候选 LoRA LR=1e-4 / 3e-5 / 1e-5 分别训练100个优化器更新；相同初始化、种子及窗口顺序，训练遍历结束后循环，不使用验证标签训练。完整验证集 vLLM 贪心生成的 Macro-F1 最高者胜出，同分选更小 LR。
- 选定 LR 后重新初始化，正式训练3epoch；每个候选试跑及正式训练均单独使用10%线性warmup＋90%cosine衰减到零，AdamW、weight decay=0.01、clip=1。
- 三个正式 epoch 按完整验证集 vLLM Macro-F1 选择，同分选最早。仅选定 checkpoint 后执行固定测试集；历史上已看过该测试集，不能声称从未接触测试集。
- 与之前4B实验的训练调度、batch和LR选择流程不同，跨实验分数只能作背景比较，不能当作只改变模型大小的严格消融。

完整预设见 [protocol.json](protocol.json)，容量选择见 [batch_selection.json](batch_selection.json)。

|模型|结果|权重|生成图片|
|---|---|---|---|
|Qwen3.5-27B|../qwen3.5_27B_lora_imu2img_joint/|../../models/Qwen3.5-27B-lora-imu2img-joint/|../../dataset/vlm_imu2img_qwen35_27b_lora_joint/|
|Qwen3.8-27B|../qwen3.8_27B_lora_imu2img_joint/|../../models/Qwen3.8-27B-lora-imu2img-joint/|../../dataset/vlm_imu2img_qwen38_27b_lora_joint/|

每组 lr_search/ 保存三个试跑的训练记录及验证结果；checkpoints/ 保存正式训练各轮 adapter＋converter；最终模型保存合并权重，结果根目录保存测试原回答、预测、混淆矩阵及犬只bootstrap区间。所有验证/测试推理均使用vLLM，训练使用原生 Transformers＋PEFT。

入口：scripts/lora_27b.py；公共实现：scripts/imu2img_lora27_common.py。最终独立校验通过后才生成 complete.json；generation_complete.json 仅表示生成结束，不代表审核结束。先前三组4B结果保留原样。
