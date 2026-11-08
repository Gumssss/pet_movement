# Qwen3.8-27B 单GPU联合LoRA

状态：用户要求停止，单卡实验队列已停止，保留已产生记录，不自动恢复。

用户指定只运行Qwen3.8-27B，IMU2IMG转换器与LoRA联合训练，保留一张GPU供别人使用。

- 仅使用物理GPU0（UUID GPU-0ded27d3-e266-f20b-2476-f2980be2980e）。通过CUDA_VISIBLE_DEVICES固定隔离；训练、绘图、vLLM验证及测试全部限制同一卡，物理GPU1不使用。
- microbatch96，梯度累积2，有效batch192；每个更新按全部监督答案token数归一化，正确处理epoch末尾短batch。不开DDP，不复制或丢弃训练样本。
- ViT、视觉投影和语言混合注意力LoRA r4 alpha8 dropout0.05，414模块13,881,216参数；转换器835,584参数联合训练，基础Qwen全冻结。转换器LR=LoRA LR/20；开启non-reentrant gradient checkpointing。
- 三个LoRA LR（1e-4、3e-5、1e-5）从相同原始Qwen、相同全新LoRA、原对齐4B converter分别开始，均100优化器更新；每个候选数据顺序一致，完整验证Macro-F1选LR，同分选小LR。
- 选LR后重新初始化正式训练3epoch，每epoch86更新、共258更新，训练样本呈现49,383次；10% linear warmup＋90% cosine decay，各试跑100步单独调度。正式每轮完整验证，以vLLM生成Macro-F1选epoch，同分最早，再执行固定3708窗测试。
- train16461/val4727/test3708，原犬只互斥划分；同一英文简短提示；原200点→RMS中心196点→49个patch→224RGB losslessWebP，蓝色128。所有推理用vLLM。已有训练标签用于转换器及LoRA，整套不是zero-shot。
- 单卡新实验从头执行，双卡旧结果保留在原目录。有效batch相同不能保证与双卡结果逐位一致（随机数、梯度归并及末尾采样均有差别）。旧Qwen3.5-27B实验已完成，旧Qwen3.8-27B正式训练已完成但验证中断，不继续使用双卡流程。

协议 protocol.json；进度 progress.json；入口 scripts/lora_27b_single_gpu.py；独立审核 scripts/validate_lora_27b_single_gpu.py。complete.json只在全流程生成和独立检查均通过后写入，failure.json表示流程异常停止。

结果：../qwen3.8_27B_lora_imu2img_joint_single_gpu/

权重：../../models/Qwen3.8-27B-lora-imu2img-joint-single-gpu/

图片：../../dataset/vlm_imu2img_qwen38_27b_lora_joint_single_gpu/
