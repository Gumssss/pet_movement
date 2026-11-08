# ②固定IMU2IMG_qwen＋LoRA

本轮始于2026-11-04，现已完成3epoch、1545步、完整验证选模与全部3708测试窗口。正式生成使用vLLM0.26.0，全部测试预测/原回答/指标已独立核验。

## 三组对照

|实验|测试Macro-F1|Balanced accuracy|Accuracy|选中epoch|
|---|---:|---:|---:|---:|
|①普通六子图＋LoRA|0.755951|0.757997|0.685275|3|
|②固定IMU2IMG_qwen＋LoRA|0.703040|0.711713|0.640237|3|
|③联合IMU2IMG_qwen＋LoRA|0.712871|0.721404|0.648058|3|

本组状态{"ok": 3708}，解析成功率100.00%，预测分布{"Trotting": 426, "Sitting": 829, "Standing": 488, "Lying chest": 850, "Walking": 461, "Sniffing": 628, "Galloping": 26}。所有解析失败均计入3708窗分母，没有按标签重试。

## 训练与输入

三组均从同一原始Qwen3.5-4B初始化相同seed的新LoRA，不加载旧ACL LoRA。250个ViT/视觉投影/语言混合注意力线性模块，r4、alpha8、dropout0.05，5226496个LoRA参数；Qwen所有基础权重冻结。仅联合组额外更新835584个转换器参数。两组IMU2IMG从同一已对齐epoch3转换器热启动，固定组参数全程不变。普通组是原六轴1280×1152波形图/1440视觉token＋原英文波形特征提示；IMU两组224×224/49视觉token＋完全相同简短英文提示。

相同16461训练、4727验证、3708测试窗，犬只隔离。3epoch，有效batch32（2GPU×micro4×累积4），AdamW、LoRA LR2e-4、联合转换器LR1e-5、wd0.01、5%warmup线性衰减、clip1。BF16基础模型、FP32LoRA/优化器；答案JSON＋EOS损失，输入和padding屏蔽，无类别重采样或增强。普通图开启非重入梯度检查点；IMU两组不开。训练前普通图不启用检查点时OOM，发生在任何优化前，诊断日志保留。

IMU训练图片从RMS196点实时生成，两组前向处理一致。联合图像使用可微8位round直通估计及官方融合归一化/patch顺序，转换器BF16前向、FP32参数，保存无损WebP；冻结组没有转换器梯度。Qwen基础参数完整内存SHA前后相同，两个IMU组初始化、三组LoRA初始化指纹分别一致。选中adapter全部250处合并矩阵逐位等于base＋LoRA，其余参与生成的基础权重与原权重按BF16导出精度逐位一致；未使用的15个MTP张量不在HF导出中。原官方权重文件SHA未变。

每epoch验证答案NLL仅作监控，最后对三个checkpoint分别进行4727窗完整vLLM生成，以macro-F1最高选epoch3；同分选较早者。测试没有参与选择。温度0、关闭思考、最多32输出token，使用相同解析函数。训练墙钟923.03秒；输入token和梯度检查点不同，不能将训练速度差异只归因于可训练参数量。各阶段计时保存在runtime及训练日志中。

|epoch|验证Macro-F1|验证Balanced accuracy|验证Accuracy|
|---|---:|---:|---:|
|1|0.642911|0.662395|0.555744|
|2|0.641423|0.666739|0.583880|
|3|0.699271|0.709161|0.609689|

## 每类测试结果

|类别|Precision|Recall|F1|窗口数|
|---|---:|---:|---:|---:|
|Galloping|0.9615|0.8621|0.9091|29|
|Lying chest|0.5682|0.5558|0.5620|869|
|Sitting|0.4885|0.4982|0.4933|813|
|Sniffing|0.9092|0.8510|0.8791|671|
|Standing|0.3279|0.2787|0.3013|574|
|Trotting|0.9695|0.9928|0.9810|416|
|Walking|0.6876|0.9435|0.7955|336|

按9只测试犬整体bootstrap2000次的95%区间：

- macro_f1：[0.625390, 0.741539]
- balanced_accuracy：[0.627805, 0.753601]
- accuracy：[0.603371, 0.673663]

## 文件与边界

- [predictions.csv](predictions.csv)、[responses.jsonl](responses.jsonl)、[metrics.json](metrics.json)、[validation.json](validation.json)：完整预测、原输出、各类/犬指标/区间与独立校验。
- [training_protocol.json](training_protocol.json)、[protocol.json](protocol.json)、[prompt.txt](prompt.txt)：固定训练/生成配置与实际英文输入。
- validation/：三个epoch各4727条完整验证输出；checkpoint_selection.json：选模；training_complete.json：冻结基础权重证明。
- [三组汇总](../lora_comparison/README.md)：配对犬只bootstrap差值、初始化公平性和计算边界。

①②同时改变图片和对应提示，是整体方案比较；②③相同初始化、提示和新增训练预算，仅转换器是否更新不同。IMU转换器此前使用同一批训练标签做过额外监督预训练，标签种类/窗口预算不增加，但总训练计算更多。固定测试集此前已被查看，本轮没有按新测试输出调参。仅一个训练随机种子，犬只bootstrap不覆盖训练随机性。

[模型与选中adapter](../../models/Qwen3.5-4B-lora-imu2img-frozen/README.md)。
