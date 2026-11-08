# 与LoRA联合训练的IMU2IMG图片

此目录只保存第三组联合转换器的验证和测试图片。转换器从原冻结Qwen对齐epoch3初始化，与全新r4/alpha8 LoRA联合训练；Qwen基础权重冻结。输入仍为原200点窗口经两模态三轴RMS、中心196点、49×4点patch，输出224×224 RGB无损WebP，B固定128；BF16前向、固定8位round，与训练可微桥接的前向一致。

validation/epoch_1至epoch_3分别对应三个checkpoint的全部4727验证窗。按完整验证macro-F1选epoch3，test/images/保存其3708测试图片。各images.csv记录窗口来源与SHA，各render_protocol.json记录转换器SHA。匿名文件名不含标签、犬只或Task，元数据及清单不会传入VLM。prompt.txt是实际使用的简短英文文本，与第二组完全一致。

[训练和完整结果](../../results/qwen3.5_4B_lora_imu2img_joint/README.md)、[三组比较](../../results/lora_comparison/README.md)、[最终模型](../../models/Qwen3.5-4B-lora-imu2img-joint/README.md)。训练图片实时生成，不另存全部训练图；验证和测试图完整保存。此表示仍为学习纹理，图片亮度不能直接当作原波形幅度。
