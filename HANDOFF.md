# 犬只 IMU 小模型与 VLM 研究：聊天交接

## 2026-11-05 Qwen3.8-27B sixplot单卡LoRA（已完成）

用户新授权：普通sixplot＋Qwen3.8-27B新LoRA，单卡、基础冻结r4alpha8、三LR各100步验证F1选、10%linear warmup＋90%cosine、GC开、batch尽量大、正式3epoch。用户明确选择W&B离线记录，训练不等登录；不要重启先前已停止IMU2IMG实验。

仅GPU0 UUID GPU-0ded27d3-e266-f20b-2476-f2980be2980e，GPU1保持空闲。复用dataset/vlm/supervised的21188训练验证图及dataset/vlm的3708测试图；所有24896图SHA检查通过；1280×1152六子图、1440视觉token/2230前缀、原长英文波形提示不变。新LoRA414模块13,881,216参数，ViT/projection/hybrid language attention、dropout.05，BF16原模型、FP32 LoRA/AdamW，non-reentrant GC。

初筛batch12能跑、13/14 OOM；连续真实训练12/11显存或实际余量不达标，最终batch10连续3步通过（reserved88.080GiB、实际free5.317GiB）。固定batch10、累积1、有效10，运行3个100步试跑；相同原始模型及新LoRA初始化/顺序，各自10步warmup，完整4727验证窗vLLM Macro-F1选LR，同分小LR。正式重新初始化3epoch=4941更新、warmup495，验证选epoch同分最早，再测试3708窗；不按测试反馈调参。label预算同原train16461，试跑额外优化计算单列。

脚本sixplot_qwen38_lora.py、sixplot_qwen38_common.py、sixplot_tracking.py、validate_sixplot_qwen38.py；结果/协议/进度/源归档results/qwen3.8_27B_lora_sixplot_single_gpu（experiment_protocol.json为实验预设，最终protocol.json为测试生成协议），模型models/Qwen3.8-27B-lora-sixplot-single-gpu。W&B SDK0.31.0私有安装.venv/wandb-packages，仅指标/配置离线保存，不上传图/权重。每LR和正式训练均有tracking/metrics.jsonl、loss_curve.png（每20步刷新）、wandb离线文件。离线技术写入测试通过。CLI需CUDA_VISIBLE_DEVICES=上述UUID；所有推理vLLM，训练HF/PEFT。最后自动独立校验，complete.json出现才算完成，failure.json表示异常停止；不自动重试。

后台pipeline PID4171648，日志results/qwen3.8_27B_lora_sixplot_single_gpu/pipeline.log；首个LR1e-4真实优化已开始，tracking/metrics.jsonl及loss_curve.png已生成，单卡batch10稳定运行。完成标记仍以complete.json为准。


## 2026-11-05 Qwen3.8-27B单卡联合LoRA（用户要求停止）

用户确认只训练Qwen3.8-27B，IMU2IMG转换器联合训练，释放一张卡。原后台双卡组PID3339140已SIGTERM停止；切换时Qwen3.5-27B已完成（测试macro-F1=.5109779294/BA=.5258741521/acc=.5636461704，LR1e-4、epoch2），Qwen3.8-27B旧双卡3epoch训练已结束，第三轮验证中断。历史结果不删除不混用，不再启动Qwen3.5-27B。

新代码scripts/lora_27b_single_gpu.py、imu2img_lora27_single_common.py、validate_lora_27b_single_gpu.py；新结果results/qwen3.8_27B_lora_imu2img_joint_single_gpu；协议/状态results/lora_27b_single_gpu；模型models/Qwen3.8-27B-lora-imu2img-joint-single-gpu；图dataset/vlm_imu2img_qwen38_27b_lora_joint_single_gpu。仅GPU0 UUID GPU-0ded27d3-e266-f20b-2476-f2980be2980e，CUDA_VISIBLE_DEVICES传给全部子进程，GPU1留空。每卡96、累积2、有效192，无DDP。三步完整梯度检查通过，峰值reserved83.71875GiB，ViT/projection/attention/converter梯度均正，基础梯度空/版本不变。

新运行所有LR和正式训练均从头初始化，原Qwen基础冻结，r4alpha8dropout.05，converterLR=LoRA/20，GCnon-reentrant；三LR各100步完整验证Macro-F1选，再重置正式3epoch(258updates)，10%linear warmup+90%cosine。按监督token数加权累积，epoch末短batch不丢不复制；每轮86步，无训练样本重复填充。选轮再测试，所有推理vLLM。未改动原双卡实现，保留原来源SHA。完整流程最后自动独立核验，complete.json出现才算完成。

用户于2026-11-05T11:13:10.172310+00:00明确要求停止3.8实验。已终止单卡pipeline进程组4081265，保留记录，不自动重启或继续训练/验证/测试。单卡3步技术检查通过，LR试跑启动后被停止。


## 2026-11-05 两组27B联合LoRA（双卡历史流程已停止）

用户授权前述三组4B完成后开始IMU2IMG转换器＋Qwen3.5-27B、＋Qwen3.8-27B。同样ViT/visual_projection/language_attention LoRA r4 alpha8 dropout.05；414线性模块13,881,216LoRA参数，转换器835,584参数，Qwen基础权重冻结。暂按联合训练解释并已告知用户；可选澄清尚未回复。原对齐4B converter热启动，LoRA全新，简短英文提示。

新要求：1e-4/3e-5/1e-5各100个优化器更新，完整4727验证窗vLLM贪心Macro-F1选LR，同分选小LR；正式重新初始化训练3epoch，再按完整验证宏F1选checkpoint，同分最早。10%linear warmup+90%cosine，转换器LR=LoRA/20，GC开启non-reentrant。双96GB GPU容量检查共同micro96、有效batch192，无额外累积；模型3.5单卡128通过但3.8的128显存余量不足2GiB，160OOM。先两组都通过DDP3步技术检查，再启动试跑；技术更新不保存不参与选优。

代码scripts/lora_27b.py、imu2img_lora27_common.py；protocol/batch/prior-SHA见results/lora_27b/，README含完整方案。输出results/qwen3.5_27B_lora_imu2img_joint与qwen3.8_27B_lora_imu2img_joint；模型models/Qwen3.5-27B-lora-imu2img-joint与3.8同后缀；图片dataset/vlm_imu2img_qwen35_27b_lora_joint、qwen38同后缀。两组DDP3步技术检查已全部通过（每卡峰值reserved约80.9GiB）；后台pipeline PID3339140，日志results/lora_27b/pipeline.log，当前Qwen3.5-27B的100步LR筛选已启动，随后自动正式训练/验证/测试再处理Qwen3.8-27B；最终自动运行validate_lora_27b.py，complete.json出现才算全部完成。当前状态results/lora_27b/progress.json。所有历史结果不重跑，不修改；原deleted simple目录不重建。默认shell bwrap出错，必要工具用require_escalated。DDP需NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1。


## 2026-11-04 三组相同LoRA配置实验（已完成）

|实验|测试Macro-F1|Balanced accuracy|Accuracy|选中epoch|
|---|---:|---:|---:|---:|
|①普通六子图＋LoRA|0.755951|0.757997|0.685275|3|
|②固定IMU2IMG_qwen＋LoRA|0.703040|0.711713|0.640237|3|
|③联合IMU2IMG_qwen＋LoRA|0.712871|0.721404|0.648058|3|

三组从原始Qwen3.5-4B初始化相同新LoRA（r4/alpha8/dropout0.05、ViT/投影/语言混合注意力250模块5226496参数），Qwen基础参数全程冻结且内存SHA前后不变。两组IMU从原对齐epoch3转换器初始化，固定组converter不变；联合组追加835584转换器参数LR1e-5，LoRA LR均2e-4。3epoch、有效batch32、双卡micro4累积4，按完整4727验证窗vLLM宏F1选epoch再测试3708窗。普通波形长英文提示，IMU两组相同简短提示。普通图启用梯度检查点，IMU不开；效率需披露差异。

结果与协议results/lora_comparison/及三份results/qwen3.5_4B_lora_sixplot、_imu2img_frozen、_imu2img_joint；模型models/Qwen3.5-4B-lora-sixplot、-imu2img-frozen、-imu2img-joint，各保留三个adaptercheckpoint及最终合并模型，IMU还存选中converter。普通新训练/验证图dataset/vlm/supervised/，联合图dataset/vlm_imu2img_qwen_lora_joint/；固定组复用原已对齐验证及测试图。用户已删除dataset/vlm_imu2img_simple，不再使用或重建。

脚本lora_comparison.py和validate_lora_comparison.py；已逐条核验原回答/预测/指标，全部250矩阵合并正确且非目标权重逐位不变、基础官方SHA不变、历史文件未变、相同LoRA初始化已验证。comparison.json含配对犬只bootstrap差值。全部工作完成，不自动重新训练、测试或调参。

## 2026-11-04 冻结Qwen的IMU2IMG对齐（已完成）

用户同意简化提示验证对照＋冻结Qwen只训练图像转换器。两种提示的旧图验证macro-F1均=0.0265372701。旧转换器＋简化提示测试macro-F1=0.0237388724，仍全部Walking，结果results/qwen3.5_4B_imu2img_simple_en/。

仅835584转换器参数通过冻结Qwen答案损失训练3epoch、1545步；Qwen全参数requires_grad=False/eval，梯度经过激活回传，所有Qwen梯度空；训练前后完整内存SHA相同83798a04335da0c1e533fca5a2bd595ddf75782fe2d12af353478bd3d4879bb0，原基础权重文件SHA未变。转换器热启动原ViT版epoch7，原2模态RMS/196点/49patch/224RGB/B128保持。8位round用STE，官方融合归一化及patch排列与部署一致，BF16输入逐位检查通过。没有训练或加载Qwen LoRA。

三个checkpoint由全部4727验证窗的vLLM宏F1选epoch3（score=0.1287498611），随后才导出3708测试图片。对齐版本测试macro-F1=0.1258094880、BA=0.2751286994、accuracy=0.2974649407，状态={'ok': 3708}。结果results/qwen3.5_4B_imu2img_aligned_en/；模型models/IMU2IMG-Qwen3.5-4B-aligned/仅转换器；数据dataset/vlm_imu2img_qwen/；训练与选择results/imu2img_qwen_alignment/；原提示验证对照results/qwen3.5_4B_imu2img_promptval/。200个历史文件未变，原回答/CSV/重算一致。整体是冻结VLM的监督适配，不称严格zero-shot。

脚本align_imu2img_qwen.py prepare/smoke/train/render与eval_imu2img_alignment.py baseline-all/aligned-all/evaluate；双GPU训练需NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1。全部任务已完成，不自动重训或重跑；复算evaluate --result-dir路径不加载模型。


## 2026-11-04 普通与IMU2IMG英文vLLM测试（已完成）

用户要求未微调Qwen3.5-4B分别测试普通六子图和DOI10.1145/3714394.3756208（IMU2IMG），均vLLM＋英文；并明确“qwen保持zero-shot，剩下都按照论文思路来”，已允许RMS、监督训练转换器和缺失磁力计B中灰适配。没有训练或加载Qwen LoRA。

两组各3708窗已完成，全部状态ok。普通结果results/qwen3.5_4B_sixplot_en/：macro-F1=0.0983898619、BA=0.2556309109、accuracy=0.2168284790；IMU2IMG结果results/qwen3.5_4B_imu2img_en/：macro-F1=0.0237388724、BA=0.1428571429、accuracy=0.0906148867。图片dataset/vlm_imu2img/images/、来源与差异method.md；普通复用dataset/vlm原图。

转换器训练scripts/train_imu2img.py，Mantis原TGU版本23ecfe493830db177546ef159947de7d79a573bd，Google ViT版本98cb6f7099d372694c8547bf7002289b36857dc7；16461训练/4727验证、10epoch、验证macro-F1选epoch7=0.7480024243。该分数仅为配套ViT验证分数，不是Qwen结果。模型models/IMU2IMG-pet/，训练记录results/imu2img_converter/。R/G分别是两类三轴RMS，内部196点（两端各丢2点）、49个4点patch、224RGB、B128；Qwen图片下限224²，实际49视觉token/930总输入，普通1440视觉token/2230总输入。转换器使用了标签，整套不是零标签系统，也不是SHL原论文完整复现。

两组基础权重/生成/引擎一致，英文七类定义/JSON规则一致，图像说明及观察汇总句按格式调整。源Qwen SHA未变，156个历史输入/结果文件未变，原回答/CSV/独立重算一致。不要重跑。通用测试入口scripts/eval_qwen35_images.py，复算evaluate --method ordinary或imu2img不加载模型。普通实际源码已按运行SHA归档scripts/archive/eval_qwen35_images_sixplot_20261104.py；当前入口增加保持224布局的IMU分支。


## 2026-11-04 Qwen3.5-4B LoRA（已完成）

用户指定ViT＋Visual Projection＋Attention，r=4/alpha=8。已完成3epoch双GPU监督微调，250个线性LoRA模块、5,226,496可训练参数；基础权重冻结。采用dataset/vlm_acl/supervised/的16461训练窗、4727验证窗，最低验证回答NLL选epoch=1，3708测试窗完全隔离。

模型models/Qwen3.5-4B-lora/保留选中adapter/及3份epoch checkpoint，模型根目录是全部视觉/投影/注意力LoRA合并后的BF16权重。250个合并矩阵逐一检查等于base＋LoRA。通用vLLM入口支持--model 4B-lora。训练依赖仅在.venv/lora-packages，NCCL需NCCL_P2P_DISABLE=1和NCCL_IB_DISABLE=1；默认NCCL初始化失败发生在优化前，日志归档。

已完成匹配未微调4B基线和LoRA的同ACL/英文/vLLM测试。结果分别results/qwen3.5_4B_acl/及results/qwen3.5_4B_lora_acl/。基线Macro-F1=0.0753607741、BA=0.2090592334、accuracy=0.1623516721；LoRA Macro-F1=0.6735484200、BA=0.6764632224、accuracy=0.6232470334；LoRA状态={'ok': 3708}。原回答/预测/重算一致，旧数据/结果及基础权重SHA未变，不要自动重训或重跑。


## 2026-11-04 Kimi API测试（已完成）

用户明确授权调用官方Kimi K3.1 Pro完成相同测试，覆盖“不自动调用付费API”的默认限制。通过https://api.moonshot.ai/v1完成全部3708测试窗，逐条返回模型kimi-k3.1-pro。新结果results/kimi/k3.1_pro_acl/，复用dataset/vlm_acl/图片及相同英文文本，无训练图上传。已生成完整报告、原始API返回、预测、指标及独立校验。Macro-F1=0.2650917229，balanced accuracy=0.4080362397，accuracy=0.2961165049；状态={'ok': 3708}。

Kimi始终思考，low effort、温度固定1、输出上限4096，与Qwen的关闭思考/温度0/上限32不同；服务端框架未知，不声称vLLM。全部92个历史文件指纹未变。密钥不在项目中，临时文件已删除，不应重新调用已完成测试。复算评价：.venv/bin/python scripts/kimi_classify.py evaluate（不需要密钥）。


更新：2026-11-04（Europe/London）。已按统一目录结构整理并更新当前Linux路径；原实验结果保留。请先读此文件，再按用户在新聊天中的具体要求继续；不要自动重训或调用付费 API。

## 1. 用户目标和偏好

研究同一段犬只项圈运动数据：小模型直接处理六路时序，与图像化后输入视觉语言模型（VLM）相比，在识别能力、标注需求和计算效率上有什么差异。

用户已明确收窄范围：不要讨论创业计划的其他板块，不扩展硬件采购、招募或商业计划；不要把研究搞复杂，不要追问预算/GPU额度。中文沟通，解释必要术语，文件尽量少且整洁。

最简研究计划包含：两个方向的 background、一个核心问题、三个候选 contributions。
- 方法背景：多模态大模型垂域适配（LLaVA-Med、SensorLLM、By My Eyes）。
- 应用背景：宠物运动/健康相关行为监测。
- 核心问题：标注有限、测试犬未参与训练时，图像化及领域适配 VLM 能否以可接受计算成本超过直接处理 IMU 的小模型？
- 候选贡献：①公平的能力/效率比较；②低标注领域适配；③有错误互补证据后再研究大小模型协同。先完成①，再选择②或③，不承诺全部做。

当前公开数据只支持行为分类，不能据此声称疾病诊断、疼痛识别或医学有效性。不得预设 VLM 更好或声称“首次”。Jev不是主线，也不等于JEPA。

## 2026-11-03 本轮已完成的任务

用户要求本地部署Qwen/Qwen3.8-27B，用现有图片完成同一任务，并将dataset/vlm/prompt.txt改为英文。随后要求所有VLM统一使用vLLM，并明确选择“只跑新Qwen3.8，旧结果保留为历史”。不要重跑两份旧Qwen3.5。

新模型固定提交1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0，已下载至models/Qwen3.8-27B，18个权重分片均通过SHA-256校验。项目环境已有vLLM 0.26.0，未升级原环境。所有当前VLM入口已切换vLLM；旧实验代码按原指纹归档到scripts/archive/qwen_classify_20261102.py。

原中文提示另存dataset/vlm/prompt_zh.txt，旧模型目录中的提示与结果均未改。新英文提示及证据记录是dataset/vlm/prompt.txt、research_en.md，处理器检查为processor_check_qwen38.json。复用同一3,708张图片、相同尺度和标签；新模型实际处理器保持1280×1152、1440视觉token，总输入2230token。

Qwen3.8已通过vLLM完成全部3708测试窗：Macro-F1=0.1579669916，balanced accuracy=0.3292810333，accuracy=0.2055016181，状态={'ok': 3708}。结果、指标、原始回答、运行和验证记录均在results/qwen3_8_27b/，旧结果和固定图片指纹未变。勿自动重跑。

与旧中文结果相比，模型、语言和框架三项同时变化；不单独归因于模型升级。新协议在正式测试前冻结，不根据测试输出修改。

## 2. 当前状态

已完成用户明确要求的工作：
1. 原始数据审计、固定七类/犬只划分/窗口。
2. 随机森林和轻量CNN基线训练。
3. 补算balanced accuracy，完成20组CNN超参数训练并记录全部验证结果，保存验证集选出的最佳模型。
4. 将results/README.md转为4页中文PDF，检查过四页的字体、表格和分页。

2026-11-02已完成犬只行为任务的Qwen图像分类：两份模型均使用全部3,708个固定测试窗、同一份文献先验提示和匿名WebP，没有少样本示例或微调，没有付费API或自采实验。结果、原始回答、预测、协议和运行记录分别位于results/qwen_4b/及results/qwen_27b/；汇总见results/README.md。

- qwen_4b：Macro-F1=0.0954456031，balanced accuracy=0.2439024390，accuracy=0.2138619202，状态={'ok': 3708}
- qwen_27b：Macro-F1=0.2755248617，balanced accuracy=0.3222238163，accuracy=0.2464940669，状态={'ok': 3708}

图片与文献依据保存在dataset/vlm/：prompt.txt是模型实际文本，research.md列出文献和推断边界，render_protocol.json保存训练集拟合的固定范围。实际模型输入为1280×1152、1440视觉token，处理后预览已检查。共有468窗包含范围外点，6,751/4,449,600个通道采样点被显示截断；逐窗记录见images.csv。input_validation.json与result_validation.json保存完整性检查。

两份模型的输入、输出与重算指标核对一致，原CNN/RF、划分和原协议文件保持原指纹。正式提示和图像规则未根据测试成绩修改。当前推理进程已结束，后续会话仍应按实际进程检查状态，不要自动重跑或根据本次测试成绩反复调提示。


## 3. 当前文件位置

项目根目录：`/home/exet6233/pet/pet_movement`。完整目录与运行说明见根目录README.md；脚本从自身位置确定项目根目录，不依赖旧Mac绝对路径。

|内容|相对项目根目录的路径|
|---|---|
|原始数据|dataset/犬只运动传感器数据集/|
|固定犬只划分及种子|dataset/splits/dog_splits.json|
|固定窗口、标签、split和window_id|dataset/splits/windows.csv|
|各模型逐窗口预测和置信度|results/cnn/predictions.csv 和 results/random_forest/predictions.csv|
|随机森林、原始CNN、搜索最佳CNN|models/random_forest.joblib、models/cnn.pt、models/cnn_best.pt|
|Qwen预训练模型|models/Qwen3.5-4B/、models/Qwen3.5-27B/|
|Qwen来源、固定版本和文件校验清单|models/model_manifest.json|
|实验报告及历史PDF导出|results/README.md、results/README.pdf|
|协议、原始依赖版本、搜索参数及划分快照|results/protocol.json|
|指标、每犬结果、混淆矩阵|results/cnn/metrics.json 和 results/random_forest/metrics.json|
|Qwen图像分类完整结果|results/qwen_4b/、results/qwen_27b/（各有README、metrics、predictions及原始回答）|
|基线、搜索、下载、离线推理脚本|scripts/baseline.py、scripts/cnn_search.py、scripts/download_models.py、scripts/run_qwen.py|
|Qwen快捷入口|scripts/qwen|
|原始CNN逐轮记录|results/cnn/training_history.csv|
|20组搜索汇总与逐轮记录|results/cnn/cnn_search.csv、results/cnn/cnn_search_history.csv|
|模型下载日志及基础验证记录|results/model_checks/|
|基础验证图片|scripts/assets/smoke_image.png|
|当前机器Python环境与版本|.venv/、requirements.txt|

原results/windows.csv已拆分：窗口元数据在dataset/splits/windows.csv，预测列在results/cnn/predictions.csv 和 results/random_forest/predictions.csv，使用window_id一对一关联，窗口表及两个模型预测表各为24,896行。protocol.json中的splits是原实验快照；数据集中的dog_splits.json是当前加载依据，搜索脚本会核对两者一致。

模型、原始数据和研究结果不按每次操作复制新文件夹。原deployment目录已取消：代码、环境和验证记录分别移入上述位置。历史验证JSON/日志保留原文，其中旧图片路径指向当时位置。

原实验来自Mac桌面项目；早期调研笔记曾在另一聊天目录researchwrite/pet-imu-vlm/，当前Linux项目未包含这些笔记。研究状态以此交接和实际results为准，勿把历史“尚未审计数据”等记录当作当前状态。

## 4. 数据与固定协议

本地输入包含DogMoveData_csv_format.zip、DogMoveData.mat、DogInfo.csv、Data_description.txt、AnalysisCode.zip。只使用ZIP内CSV。

- 实测45犬、62场、10,611,068行，100Hz；未发现六路信号非有限值或0.01秒采样间隔异常。
- 六路顺序：ANeck_x、ANeck_y、ANeck_z、GNeck_x、GNeck_y、GNeck_z。
- 数据说明对ANeck的x/y有对调描述；当前保留CSV列名，不推断解剖方向。物理单位未充分确证，使用原始存储值。
- 固定2秒/200点窗口，步长2秒，不重叠，不跨场，剩余尾部舍弃。
- 目标类别及索引顺序：0 Galloping；1 Lying chest；2 Sitting；3 Sniffing；4 Standing；5 Trotting；6 Walking。
- 遍历Behavior_1/2/3：每个采样点恰好有一个上述目标行为，整窗必须同一目标。允许Playing、Panting等附加行为共存。不能只看Behavior_1，不能把Task当真值，也不能把undefined训练为其他类。
- 53,023候选窗，保留24,896（46.95%），拒绝28,127；另有6,468尾部采样点。
- 疾跑仅无附加标注时36.68秒；跨三列允许附加标注后有更多疾跑，严格纯窗口最终140个。因此七类全部保留，没有静默删疾跑。
- 当前是清晰窗口的封闭七分类；筛选依赖真值，不代表全天连续识别成绩或实际部署覆盖率。

种子20261101。先从DogInfo固定犬只分组，再在场内独立构窗。全部场次随犬隔离：

```text
train (27): 16,18,19,23,26,27,29,30,33,34,36,43,44,46,47,48,49,55,57,58,60,63,66,67,70,72,74
validation (9): 22,39,41,45,53,54,56,59,73
test (9): 20,21,25,28,51,52,61,65,68
```

|类别|train|validation|test|
|---|---:|---:|---:|
|Galloping|99|12|29|
|Lying chest|3458|1050|869|
|Sitting|3374|997|813|
|Sniffing|2897|695|671|
|Standing|2967|859|574|
|Trotting|1837|630|416|
|Walking|1829|484|336|

用户最初指定仓库V2，但本地文件发行版本尚未证实。网络发现另有V3、V4，不能直接称本地V2或最新版本。ZIP SHA256：
`47f9e41428c0591d3c980aff1b59981eac14567a02019cc044f69ef4648a3b46`

## 5. 已完成的模型与结果

|模型|测试Macro-F1|测试Balanced accuracy|测试Accuracy|
|---|---:|---:|---:|
|随机森林|0.7107474354|0.7077814383|0.6326860841|
|原始CNN|0.7774075506|0.7877743912|0.7273462783|
|20组验证最优CNN|0.7942443910|0.7982030840|0.7270765912|

Balanced accuracy为真实存在类别的召回率平均；总体测试包含全部七类。它不是按犬平均的准确率。原始模型每犬指标字段为balanced_accuracy_present_classes；最佳CNN的每犬balanced_accuracy也遵循出现类别平均。每犬固定七类Macro-F1会把缺失且未预测类记0，不要忽略类别支持数直接排名犬只。

随机森林：每路9统计量，共54特征；200树，min_samples_leaf=2、max_features=sqrt、balanced_subsample。没有用目标犬站立标签校准。训练约2.8秒；本机CPU单窗P50约13.17ms（含特征提取）。

CNN固定结构：Conv1d(6,32,7)+BN+ReLU+MaxPool2；Conv1d(32,64,5)+BN+ReLU+MaxPool2；Conv1d(64,96,3)+BN+ReLU；全局平均池化、Dropout、Linear(96,7)。31,271参数。标准化仅拟合训练犬，均值/标准差存checkpoint。

原始CNN：AdamW，lr=.001、weight_decay=.0001、dropout=.2、batch128，训练频率平方根倒数加权，最多35轮/早停7轮，验证Macro-F1最佳第32轮。CPU训练约232秒，CPU单窗P50约0.18ms（不含2秒采集等待）。

20组搜索：固定配置预声明，从头训练每组而非续训权重；调整lr、weight_decay、dropout、batch_size、class_weight_power、scheduler；每组最多35轮，耐心7轮。使用Apple MPS、2 CPU线程，总训练约5.1分钟，完整参数在protocol.json和CSV。与原始CPU训练使用的设备及批次随机顺序实现有差异，不能将全部变化因果归于某个单独超参数。MPS不承诺逐比特复现。

最佳：第4组，第22轮：
```json
{"lr":0.002,"weight_decay":0.0001,"dropout":0.2,"batch_size":128,"class_weight_power":0.5,"scheduler":"none"}
```
验证Macro-F1=0.7880728931，验证balanced accuracy=0.7968016165。按验证Macro-F1选，不是按测试集或balanced accuracy选。

20组均保存验证结果；只保留搜索过程的全局最佳权重，没有20份模型，也没有对每组计算测试结果。最终选定后仅测试最佳组，原始CNN权重保持不变。

测试犬整体bootstrap 2,000次，最佳CNN的95%区间：Macro-F1 [0.7387263,0.8506760]；balanced accuracy [0.7429393,0.8566049]。仅9测试犬、单划分单种子；区间不涵盖重新训练方差。测试集此前已经报告过基线，正式论文不能称它从未被查看。

已经通过：犬只集合互斥、窗口不重复/不重叠、每集合七类覆盖、模型重载预测一致；20组/510轮完整性、逐轮最佳值与汇总一致、导出预测重新计算三模型Macro-F1与balanced accuracy一致。

## 6. 环境和继续工作注意事项

当前Linux运行环境为项目根目录.venv/，继承/home/exet6233/miniconda3/envs/ecg的系统包，补充accelerate 1.15.0。整理时保留依赖并在新位置离线重建环境入口。当前版本见requirements.txt，依赖当前基础环境，不能直接跨机器复制。

原基线环境为Python 3.12.14、numpy2.5.4、pandas3.1.0、scikit-learn1.9.1、torch2.15.0；历史版本以results/protocol.json为准。当前环境不同，原指标和耗时不代表在Linux上重新训练的结果。原Mac /private/tmp下的环境和缓存不能假定在当前机器可用。

数值缓存现在统一写入dataset/cache/pet_windows.npz（X、y、dogs、tests、starts），审计缓存写入同目录pet_audit.json。本轮已从原ZIP按既定协议重建缓存，并验证与24,896个固定窗口的犬只、场次、时间和标签逐项对应；后续需要重建时在项目根目录执行：

```sh
.venv/bin/python scripts/baseline.py audit
```

该命令从原始ZIP重建缓存，不重训、不改固定清单。固定划分不再随运行重新随机生成，脚本直接读取dataset/splits/dog_splits.json。

**不要为了查看结果运行train或搜索脚本。** baseline.py train会覆盖部分原模型、protocol、metrics、predictions和训练历史，可能丢掉已合并的调参结果；cnn_search.py按已有CSV跳过完成组，但仍会写入评价结果。需要新实验时先保护已有结果并明确版本策略。

离线推理示例：`./scripts/qwen --model 4B --device cuda:0 --prompt '请用中文介绍你自己。'`。脚本读取models/下本地权重，无常驻服务。两份Qwen模型原有基础文本和图片验证已通过，不等于完成犬只行为识别实验。

results/README.pdf是整理前的历史导出，保留原文件；当前路径以根README.md及更新后的results/README.md为准。

## 7. 背景来源及后续边界

- 数据仓库V2：https://data.mendeley.com/datasets/vxhx934tbn/2
- V4：https://data.mendeley.com/datasets/vxhx934tbn/4
- 数据说明：https://doi.org/10.1016/j.dib.2022.107822
- 原犬分类论文：https://doi.org/10.1016/j.applanim.2021.105393
- By My Eyes：https://aclanthology.org/2024.emnlp-main.133/
- SensorLLM：https://aclanthology.org/2025.emnlp-main.19/
- LLaVA-Med：https://arxiv.org/abs/2306.00890
- 犬真实环境行为识别：https://doi.org/10.3390/ani11061549
- 骨关节炎活动监测：https://doi.org/10.2460/javma.237.1.66

若用户要求进入VLM比较：严格复用dataset/splits/windows.csv，固定绘图、不得显示标签/Task/犬ID，训练示例池与测试犬隔离；零样本与少样本分开；记录拒绝/解析失败、标注预算、端到端耗时。新方案不应根据测试集反复修改。眼下并未授权具体付费服务或预算。

本轮用户授权将数据目录改名为dataset，并按模型归类results。CNN（原始与调参）在results/cnn/，随机森林在results/random_forest/；results/qwen_4b/、results/qwen_27b/用于VLM历史结果，results/kimi/当时为空，目前已有2026-11-04完成的k3.1_pro_acl/结果。共用协议和对比报告保留在results根目录。已迁移文件并修正路径；未重训、未重新下载模型；本轮随后经用户授权启动犬只VLM图像实验。后续工作以用户新指令为准。

本轮vLLM设置关闭可选FlashInfer采样器（缺少nvcc），显存使用NVML过程观测；两次正式测试前的兼容性启动失败记录保存在results/qwen3_8_27b/startup_attempts/。未处理正式样本的失败启动不计为分类失败。

## 2026-11-03 ACL绘图实验（已完成）

用户授权学习By My Eyes的时序转图方法并重跑Qwen3.8-27B。全部3708测试窗已完成，图片在dataset/vlm_acl/images/，精确结果路径results/qwen3.8_27B_acl/。采用六通道共享坐标叠加、训练集全局每通道z-score、训练固定共享轴、512×409无损WebP；不是论文few-shot/VisGen完整复现，详见dataset/vlm_acl/method.md。

本轮仍用vLLM0.26.0、相同BF16权重、英文动作说明与JSON规则。实际输入1022 token（视觉208），处理器尺寸512×416。Macro-F1=0.1232659160，balanced accuracy=0.2942501778，accuracy=0.1966019417；状态={'ok': 3708}。57个历史文件指纹未变。无需重新启动或重跑。

scripts/qwen_classify.py新增--data-dir，与--output-dir搭配用于独立绘图版本。当前代码仍可复用旧输入，但旧历史目录禁止用新协议覆盖；精确旧vLLM源码另存scripts/archive/qwen_classify_20261103_vllm.py。新绘图入口scripts/render_imu_acl.py。复核此结果的evaluate必须同时传--data-dir dataset/vlm_acl和--output-dir results/qwen3.8_27B_acl。
