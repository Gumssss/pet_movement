# 三组相同LoRA配置实验

已完成全部三组训练、完整验证选模、3708窗测试及独立校验。固定配置见protocol.json；运行日志pipeline.log。实验已完成，不必重跑。

|组别|图像及提示|转换器|结果目录|
|---|---|---|---|
|①普通方法＋LoRA|1280×1152六通道波形图，原英文波形特征描述|固定绘图|../qwen3.5_4B_lora_sixplot/|
|②IMU2IMG_qwen冻结＋LoRA|224×224学习纹理，简短英文提示|原对齐epoch3固定|../qwen3.5_4B_lora_imu2img_frozen/|
|③IMU2IMG_qwen＋LoRA|同②初始化和提示|与LoRA联合训练|../qwen3.5_4B_lora_imu2img_joint/|

三组均从同一原始Qwen3.5-4B初始化新LoRA，seed20261101。ViT＋投影＋语言混合注意力250个线性模块，r4/alpha8/dropout0.05，共5226496个LoRA参数；不加载旧ACL适配器。Qwen基础权重冻结。联合组额外训练835584个转换器参数，LR1e-5；三组LoRA LR均2e-4。

3epoch、双GPU每卡microbatch4、累积4次、有效batch32，AdamW/5%warmup/线性衰减/wd0.01/clip1。普通图因高分辨率启用非重入梯度检查点，初次无检查点反向OOM发生在正式训练前，日志保留；两组IMU2IMG不开检查点，效率比较需考虑这一差异。所有正式生成均vLLM，训练使用Transformers＋PEFT。

相同16461训练、4727验证、3708测试窗口，按犬只隔离。每组训练三轮后，用各checkpoint在完整验证集生成答案，按macro-F1选epoch，完全相同时选较早者。答案JSON＋EOS参与损失，输入和padding屏蔽；无重采样、增强、提示搜索或测试调参。

①②比较图像表示及配套提示整体，不是单独图像消融；②③转换器初始参数、LoRA初始参数、提示、样本、轮数与优化步数相同。IMU2IMG转换器此前使用同一批训练标签进行过额外监督训练，应另计预训练计算。此前固定测试集已在其他实验被查看，本轮方案在本轮测试输出前固定，单种子结果不覆盖训练方差。

执行诊断：固定转换器组首次训练在第1轮结束后的冻结参数检查处停止，原因是DDP初始化广播增加了张量版本计数；没有验证生成或测试。改为检查广播前后完整参数值相同、在广播后记录版本基线，从原始初始化重新完成正式3轮。首次未保存的权重已丢弃，日志保留在该组startup_attempts/ddp_version_check；这次失败的训练计算额外发生，不计入成功训练会话墙钟。相关源码归档scripts/archive/lora_comparison_20261104_before_ddp_version_fix.py。

## 完整测试结果

|实验|测试Macro-F1|Balanced accuracy|Accuracy|选中epoch|
|---|---:|---:|---:|---:|
|①普通六子图＋LoRA|0.755951|0.757997|0.685275|3|
|②固定IMU2IMG_qwen＋LoRA|0.703040|0.711713|0.640237|3|
|③联合IMU2IMG_qwen＋LoRA|0.712871|0.721404|0.648058|3|

三组LoRA初始参数与Qwen基础参数指纹相同，原始基础权重官方SHA未变，290个历史文件指纹未变。

|配对差值|Macro-F1差值|按犬只配对bootstrap95%区间|
|---|---:|---|
|imu2img_frozen_minus_sixplot|-0.052911|[-0.131772, -0.001537]|
|imu2img_joint_minus_imu2img_frozen|0.009831|[-0.002046, 0.043125]|

[①普通图报告](../qwen3.5_4B_lora_sixplot/README.md)、[②冻结转换器报告](../qwen3.5_4B_lora_imu2img_frozen/README.md)、[③联合报告](../qwen3.5_4B_lora_imu2img_joint/README.md)。comparison.json保存完整汇总，单组目录保留原回答、预测、每类/每犬指标和三轮验证输出。
