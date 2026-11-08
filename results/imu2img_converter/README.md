# IMU2IMG犬只转换器训练

已完成10epoch，按完整验证集macro-F1选epoch7，分数0.748002。16,461训练窗、4,727验证窗，未用测试标签训练或选模；Qwen从未加载/更新。配套ViT的验证分数不等于Qwen测试成绩。

模型采用原Mantis-v1 TGU、两个线性patch解码器、预训练Google ViT-base-patch32-224-in21k及新7类头；转换器835,584参数，整个训练系统88,886,791参数。FP32权重/优化器与BF16 autocast。AdamW、LR1e-4、weight decay0.01、batch512、10epoch；成功训练会话实测墙钟43.97秒，不含下载、技术反向检查、后续导图和Qwen推理。

training_protocol.json固定全部设置与来源；history.jsonl保存每轮训练和验证；checkpoint_selection.json/ training_complete.json记录选择；backward_check.json验证四部分非零梯度。render.log记录图片导出。原始TGU源码许可证和NOTICE在scripts/vendor/mantis_tgu/，来源指纹在dataset/vlm_imu2img/source/。

[完整适配说明](../../dataset/vlm_imu2img/method.md)；[最终Qwen测试报告](../qwen3.5_4B_imu2img_en/README.md)。
