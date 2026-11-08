# 冻结Qwen监督对齐IMU2IMG转换器

已完成3epoch、1545步，仅835584转换器参数参与优化；Qwen可训练参数0。完整Qwen参数内存哈希前后相同：83798a04335da0c1e533fca5a2bd595ddf75782fe2d12af353478bd3d4879bb0。详细方法在[method.md](../../dataset/vlm_imu2img_qwen/method.md)。

## 验证与选模

旧图＋原提示验证macro-F1=0.026537，旧图＋简化提示=0.026537。三轮对齐模型的完整验证macro-F1：[{"epoch": 1, "macro_f1": 0.08080202904989371, "balanced_accuracy": 0.1864051422271537, "accuracy": 0.17072138777237147}, {"epoch": 2, "macro_f1": 0.11690684359241835, "balanced_accuracy": 0.24415098897637424, "accuracy": 0.27840067696213244}, {"epoch": 3, "macro_f1": 0.12874986111263353, "balanced_accuracy": 0.26788249178935863, "accuracy": 0.29490162894013117}]。选epoch3，测试不参与。validation/保存各轮全部4727条原回答和预测。

## 记录

- training_protocol.json：冻结超参数、初始化来源、答案损失和量化桥接。
- training_history.jsonl、validation_nll_history.jsonl、training_progress.json：优化与验证NLL。
- training_complete.json：所有Qwen参数训练前后的SHA、版本计数及用时。
- backward_check.json：转换器与像素非零梯度、Qwen梯度空、官方BF16输入逐位一致。
- checkpoint_selection.json：vLLM验证宏F1选择；evaluation_complete.json：最终测试成绩。
- validation.json：独立重算、全部8位输入对照与历史文件保护。
- baseline.log、train.log、pipeline.log、aligned_evaluation.log、render_validation_epochN.log：运行日志。

首次检查发现官方处理器将rescale融合至mean/std，已在正式更新前修正差异；诊断保留backward_normalization_check.log。正式转换器前向/导出使用BF16以匹配旧图片，FP32检查另存backward_fp32_converter.log。所有调整均在正式训练前完成。

首次测试图片导出因vLLM加载后CUDA_VISIBLE_DEVICES收窄、子进程cuda:1不可见而停止，尚未进行测试预测。修正为给导出子进程保留调用者原设备环境后继续，训练、提示、生成与完整验证成绩均未改变，验证原回答直接复用。错误保留render_test_device_mapping_error.log和pipeline.log；运行前源码按runtime指纹归档scripts/archive/eval_imu2img_alignment_20261104_before_render_env_fix.py，完成日志为aligned_evaluation_resume.log。

[匹配旧转换器测试](../qwen3.5_4B_imu2img_simple_en/README.md)、[对齐版本测试](../qwen3.5_4B_imu2img_aligned_en/README.md)。
