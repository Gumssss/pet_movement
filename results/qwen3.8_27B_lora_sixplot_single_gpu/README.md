# Qwen3.8-27B：sixplot + LoRA，单GPU

状态：学习率筛选、正式训练、测试及独立检查全部完成。

仅用物理GPU0（UUID GPU-0ded27d3-e266-f20b-2476-f2980be2980e），训练、验证、测试全部通过CUDA_VISIBLE_DEVICES隔离。GPU1保留给其他人。此前IMU2IMG实验保持停止。

## 固定方法

- 原始Qwen3.8-27B创建全新LoRA；基础模型冻结，ViT/视觉投影/语言混合注意力414模块、13,881,216个LoRA参数，r4、alpha8、dropout0.05；开启non-reentrant gradient checkpointing。
- 复用dataset/vlm现有普通六子图：1280×1152、六路/200点/2秒、训练集固定轴限、匿名WebP；保持英文波形特征提示。训练与vLLM推理均1440视觉token、总前缀2230token，不降低图像分辨率。
- 同一犬只互斥划分：训练16461、验证4727、测试3708。无类别重采样、增强或丢弃样本；只监督JSON答案与EOS，不监督输入/图片/填充。
- 初筛batch12能跑但真实连续训练显存不足或实际剩余显存不足2GiB；batch11同样余量不足，13/14 OOM。最终batch10连续三步通过，峰值reserved88.080GiB、实际free5.317GiB。采用microbatch10、累积1、有效batch10。
- LoRA LR=1e-4/3e-5/1e-5各训练100个优化器更新；每组从同一原始模型、相同初始LoRA、相同训练顺序开始。以完整验证集vLLM贪心Macro-F1选最佳LR，同分选更小LR。
- 选LR后重新初始化正式训练3epoch。每轮1647更新、总4941更新、样本呈现49383次。每个100步试跑单独10步warmup，正式训练495步warmup；之后cosine衰减到零。AdamW、weight decay0.01、clip1、BF16基础权重/FP32 LoRA与优化器。
- 正式各轮checkpoint以完整验证集生成Macro-F1选优，同分选最早轮次，再测固定3708窗。历史已查看该测试集，不称从未见过的测试集；本轮不按测试反馈调参。

## 查看loss

用户确认W&B离线记录，不等待登录。W&B SDK0.31.0安装在项目私有.venv/wandb-packages中，不修改共享Python环境。

- 第一个试跑曲线：[lr1e-4 loss](lr_search/lr_1e-4/tracking/loss_curve.png)。每20步及阶段结束刷新；第一步就生成预览。
- 其余LR：lr_search/lr_3e-5/tracking/、lr_search/lr_1e-5/tracking/。
- 正式训练：tracking/loss_curve.png及tracking/metrics.jsonl。
- 每步记录train/loss（答案token CE）、train/running_loss、train/learning_rate、train/grad_norm；正式每轮还记validation/answer_nll。完整验证生成F1另存每组validation/metrics.json及evaluation_tracking离线run。
- 每个阶段有独立wandb_run.json、wandb/offline-run-*/run-*.wandb。Project=pet-movement，Group=qwen38-27b-sixplot-lora；仅记录指标和配置，不上传图片/权重。

登录并同步历史离线记录可使用：

```bash
cd /home/exet6233/pet/pet_movement
PYTHONPATH=.venv/wandb-packages .venv/bin/python -m wandb login
PYTHONPATH=.venv/wandb-packages .venv/bin/python -m wandb sync results/qwen3.8_27B_lora_sixplot_single_gpu/lr_search/lr_1e-4/tracking/wandb/offline-run-*
```

这两步是后续需要网页展示时的操作，本轮不自动登录或同步。[W&B官方文档](https://docs.wandb.ai/support/)说明离线保存与后续同步。

## 文件与完成标记

结果/协议/日志/预测均在本目录；权重：models/Qwen3.8-27B-lora-sixplot-single-gpu/（三个试跑adapter、三个正式epoch adapter、最终合并模型及独立adapter）。图片保留在dataset/vlm，不再复制。

入口scripts/sixplot_qwen38_lora.py；模型/数据scripts/sixplot_qwen38_common.py；日志scripts/sixplot_tracking.py；最终独立核验scripts/validate_sixplot_qwen38.py。

experiment_protocol.json保存实验预设，根目录protocol.json在最终测试时保存生成协议；progress.json显示当前阶段，failure.json表示异常；generation_complete.json只表示生成结束，complete.json仅在独立检查通过后出现。最终检查覆盖原回答/CSV/指标、414合并矩阵及非目标权重、官方原文件SHA、历史文件、相同试跑初始化与顺序、每步LR和loss记录。

与此前4B实验的batch、LR筛选和调度不同，跨实验比较需要披露，不能作为只改变模型规模的严格消融。

<!-- final-results -->
# Qwen3.8-27B sixplot + LoRA 结果

最佳LR 0.0001；选中epoch 3。

|测试指标|结果|
|---|---:|
|Macro-F1|0.761949|
|Balanced accuracy|0.751806|
|Accuracy|0.692017|

3,708测试窗，vLLM英文贪心生成；完整验证选LR及epoch，测试不参与选优。单物理GPU0，batch10，梯度累积1，3epoch，r4/alpha8/dropout0.05，ViT/投影/语言混合注意力LoRA，基础模型冻结，GC开启，10%linear warmup+90%cosine。W&B离线记录及各阶段loss_curve.png均保留。普通图保持原1280×1152六子图与长英文波形提示。

独立验证全部414目标矩阵合并、非目标权重、官方基础文件SHA、原回答/CSV/指标、历史文件、候选初始状态与输入顺序、每步LR/损失日志。详细 comparison.json。
