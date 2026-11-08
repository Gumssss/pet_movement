# 犬只IMU基线结果

基线实验日期：2026-11-01；调参与目录说明更新：2026-11-02。已完成本地原始数据处理、固定犬只划分、随机森林与轻量CNN训练及测试集评估。前半部分为历史基线记录，完整VLM结果见文末按日期追加的章节；基础推理验证另见model_checks/。

## 固定协议

- 45只犬、62场、10,611,068行。CSV实际通道名称为准。仅用项圈六路，顺序见protocol.json。
- 100Hz；2秒（200点）窗口；步长2秒，不重叠、不跨场。时间戳连续性已检查，无非有限信号；尾部不足200点舍弃。
- 七类全部保留：Galloping、Lying chest、Sitting、Sniffing、Standing、Trotting、Walking。
- 三个行为列中每个采样点必须有且只有一个七类目标，整窗目标一致；允许Panting、Playing等附加标签共存。未标注、非目标行为与过渡窗不用于本次监督任务。Task不是标签也不是模型输入。
- 53,023个候选窗中保留24,896个（46.95%），剔除28,127个；另有6,468个尾部采样点不足成窗。这是清晰行为窗口的封闭七分类，不是所有自然连续行为的识别成绩。
- 根据DogInfo中的犬ID、随机种子20261101确定划分，再在各场内独立切窗。27训练犬、9验证犬、9测试犬；每犬所有场次固定在同一集合。固定ID见../dataset/splits/dog_splits.json（protocol.json保留原实验快照）；逐窗归属见../dataset/splits/windows.csv，预测分别见cnn/predictions.csv和random_forest/predictions.csv，通过window_id对应。
- 疾跑没有静默删除：无附加标注仅36.68秒，允许附加标签后有更多记录；严格整窗筛选保留140窗。不能直接拿本协议与原论文75%占比、50%重叠窗口的成绩比较。
- 本地文件发行版本尚未确证，不冒称V2；记录压缩包SHA256。说明文本对ANeck_x/y存在对调描述，本实验保留CSV列名，不推断解剖方向。单位未得到充分文件依据，使用原始存储值，不作物理单位转换。

## 窗口数量

|类别|训练|验证|测试|
|---|---:|---:|---:|
|Galloping|99|12|29|
|Lying chest|3458|1050|869|
|Sitting|3374|997|813|
|Sniffing|2897|695|671|
|Standing|2967|859|574|
|Trotting|1837|630|416|
|Walking|1829|484|336|

## 测试结果

|模型|Macro-F1|Accuracy|Macro-F1犬级bootstrap 95%区间|训练秒数|单窗耗时P50/P95（毫秒）|
|---|---:|---:|---|---:|---:|
|随机森林|0.7107|0.6327|[0.6542, 0.7706]|2.8|13.17 / 13.52|
|轻量CNN|0.7774|0.7273|[0.7138, 0.8325]|231.9|0.18 / 0.19|

Macro-F1对七类等权。区间基于测试犬整体重采样2,000次，包含每犬所有窗；只有9只测试犬，区间精度有限，不包含重新训练的不确定性。一次划分、一次随机种子不能证明某模型普遍更好。

耗时是本机CPU测量：随机森林包含特征提取，CNN包含标准化和推理；10次预热后测100次。未计2秒信号采集等待、磁盘读取或网络传输。没有云推理费用；没有测量耗电成本。RF全数据特征提取另耗时0.8秒。

## 每类召回率

|类别|随机森林|CNN|
|---|---:|---:|
|Galloping|0.7931|0.9310|
|Lying chest|0.4200|0.7181|
|Sitting|0.3432|0.4945|
|Sniffing|1.0000|0.9985|
|Standing|0.5192|0.4111|
|Trotting|0.9832|0.9880|
|Walking|0.8958|0.9732|

每犬结果、各类precision/F1/support及混淆矩阵分别见cnn/metrics.json和random_forest/metrics.json。每犬Macro-F1采用固定七类，完全缺失且未预测的类记0，并附每类支持数；不宜把不同类别覆盖犬的该指标直接排名。混淆矩阵行是真实标签，列是预测，按protocol.json类别顺序。

## 模型与选择规则

随机森林：每路9个统计量，共54特征；200棵树，最小叶子2，训练集类别平衡。没有用测试犬的站立标签校准。

CNN：3层一维卷积（32/64/96通道），全局平均池化；共31,271个参数。每通道均值/标准差只拟合训练犬。AdamW，学习率0.001，batch128，最多35轮，验证Macro-F1连续7轮不改善即停止；训练损失使用训练频率平方根倒数加权。实际训练35轮，选择第32轮。测试数据只在模型固定后评价，没有根据测试结果改参数。

## 文件与复现

已训练模型保存在../models/：random_forest.joblib包含模型、特征顺序与类别；cnn.pt和cnn_best.pt包含权重、训练标准化统计量、类别与窗口配置。CNN结构在../scripts/baseline.py和../scripts/cnn_search.py中。Qwen预训练模型也保存在../models/，尚未产生本任务的分类结果。

当前目录按模型保存实验输出：cnn/含原始与最优CNN的metrics.json、predictions.csv、training_history.csv、cnn_search.csv及cnn_search_history.csv；random_forest/含随机森林metrics.json和predictions.csv。protocol.json与本对比报告保留在根目录。qwen_4b/、qwen_27b/、kimi/为空的预留结果目录。model_checks/另存模型下载和基础运行验证记录。原始数据及固定窗口/犬只划分统一在../dataset/；不另存原始CSV副本或大量窗口图片。

当前运行环境、目录和离线推理命令见[项目README](../README.md)。需要重建窗口缓存时，在项目根目录执行：

```sh
.venv/bin/python scripts/baseline.py audit
```

生成的数值与审计缓存位于dataset/cache/，可以删除后重建。目录整理未重训；baseline.py train会覆盖部分现有模型/结果，cnn_search.py也会更新搜索和评价结果，运行新实验前需保护现有记录。

本报告的原始基线环境为Python 3.12.14，numpy/pandas/scikit-learn/torch历史精确版本见protocol.json；当前Linux运行环境版本不同，见项目根目录requirements.txt。已保存权重、指标和耗时仍代表原实验。模型重新加载预测一致性、犬只互斥、窗口不重叠及各集合七类覆盖均在原实验中验证；当前目录整理另做文件与表格一致性检查。

README.pdf保留整理前的四页导出，旧路径不再代表当前布局；当前目录说明以本Markdown和项目README为准。

本轮边界：没有完整交叉验证、多随机种子、真实设备或医学验证；没有进行VLM比较。疾跑样本少，静态类别和每犬差异应结合各类召回与犬级区间判断。后续VLM必须使用../dataset/splits/windows.csv中同一批测试窗口，不能另挑容易样本。

## Balanced accuracy 与20次CNN超参数搜索

原始随机森林 balanced accuracy：0.7078；原始CNN：0.7878。定义为真实标签中各类别召回率的平均值；总体测试集包含全部七类。每犬仅平均该犬真实出现的类别。

20组预声明超参数均使用相同七类、窗口、犬只划分和种子。仅按验证Macro-F1选择checkpoint及最终模型，balanced accuracy同时记录；每次试验结果在cnn/cnn_search.csv，每轮记录在cnn/cnn_search_history.csv。最佳权重在models/cnn_best.pt，原始cnn.pt保留。

### 20次搜索完成结果

|模型|测试Macro-F1|测试Balanced accuracy|测试Accuracy|
|---|---:|---:|---:|
|随机森林|0.7107|0.7078|0.6327|
|原始CNN|0.7774|0.7878|0.7273|
|调参CNN（验证集选择）|0.7942|0.7982|0.7271|

20/20组完成，最佳为第4组、第22轮。参数：`{"lr": 0.002, "weight_decay": 0.0001, "dropout": 0.2, "batch_size": 128, "class_weight_power": 0.5, "scheduler": "none"}`。按验证Macro-F1选择，并非按测试集挑最好者。每组从头训练而非连续续训，架构与数据任务保持一致；固定种子，MPS可能存在数值非确定性。设备为mps，因此训练耗时不可直接与此前CPU训练相比。所有组有验证结果，只有最终选中组额外做测试评价。搜索总训练时间5.1分钟。

搜索脚本现位于../scripts/cnn_search.py，记录为cnn/cnn_search.csv和cnn/cnn_search_history.csv，最佳模型为../models/cnn_best.pt；CNN指标与预测在cnn/metrics.json和cnn/predictions.csv，随机森林对应文件在random_forest/。20次搜索可能过拟合9只验证犬，这不是20次独立泛化验证。测试集已在先前基线中查看过，正式研究仍需额外独立验证。

## Qwen图像输入：固定测试集结果

2026-11-02：两份本地模型均完成相同3,708个测试窗口的文献先验辅助零样本分类。

|模型|测试Macro-F1|测试Balanced accuracy|测试Accuracy|
|---|---:|---:|---:|
|随机森林|0.710747|0.707781|0.632686|
|原始CNN|0.777408|0.787774|0.727346|
|调参CNN（验证集选择）|0.794244|0.798203|0.727077|
|Qwen3.5-4B（文献先验零样本）|0.095446|0.243902|0.213862|
|Qwen3.5-27B（文献先验零样本）|0.275525|0.322224|0.246494|

图片保留六通道、200点、2秒，统一1280×1152无损WebP；纵轴只由训练集稳健分位数拟合，范围外点记录并显示截断。输入不包含窗口真值、犬只ID或Task。实际提示见[共享prompt](../dataset/vlm/prompt.txt)，证据边界见[文献记录](../dataset/vlm/research.md)。

完整结果分别见[qwen_4b/README.md](qwen_4b/README.md)和[qwen_27b/README.md](qwen_27b/README.md)。原CNN/RF指标、预测、训练记录及固定划分均经指纹检查保持原内容。VLM失败输出按错误计分。VLM批处理摊销耗时与原CPU单窗延迟的口径不同，请结合各模型报告中的硬件和组件时间解释。历史PDF保持原导出，不含本轮VLM结果。

## Qwen3.8：vLLM与英文提示

2026-11-03：仅新增Qwen3.8，复用相同3,708个测试窗口及图片；两份旧Qwen3.5保留历史，不重跑。

|模型|提示语言|推理框架|Macro-F1|Balanced accuracy|Accuracy|
|---|---|---|---:|---:|---:|
|随机森林|—|scikit-learn|0.710747|0.707781|0.632686|
|验证最优CNN|—|PyTorch|0.794244|0.798203|0.727077|
|Qwen3.5-4B|中文|Transformers|0.095446|0.243902|0.213862|
|Qwen3.5-27B|中文|Transformers|0.275525|0.322224|0.246494|
|Qwen3.8-27B|英文|vLLM|0.157967|0.329281|0.205502|

完整记录见[qwen3_8_27b/README.md](qwen3_8_27b/README.md)。本轮模型、语言、框架三项同时改变，差异不能单独归因。原始回答、预测表与重算指标一致，旧结果及固定数据指纹未变。

## ACL共享坐标波形实验

2026-11-03：同一Qwen3.8-27B、英文动作先验、vLLM及3,708窗，采用By My Eyes启发的训练集标准化共享坐标波形，完成零样本推理。

|输入表示|Macro-F1|Balanced accuracy|Accuracy|
|---|---:|---:|---:|
|原六子图|0.157967|0.329281|0.205502|
|ACL共享坐标适配|0.123266|0.294250|0.196602|

新结果见[qwen3.8_27B_acl/README.md](qwen3.8_27B_acl/README.md)，图片与方法在dataset/vlm_acl/。这组标准化、布局、尺寸与图像描述改动是共同变化，未复现论文的few-shot与自动可视化搜索。旧结果保持原样，原始回答、预测表和独立重算一致。

## Kimi K3.1 Pro API实验

2026-11-04，已通过官方国际API完成同一3708个ACL图像窗口，输入英文提示逐字相同。

|模型|Macro-F1|Balanced accuracy|Accuracy|
|---|---:|---:|---:|
|Qwen3.8-27B ACL，本地vLLM|0.123266|0.294250|0.196602|
|Kimi K3.1 Pro ACL，官方API|0.265092|0.408036|0.296117|

完整报告见[kimi/k3.1_pro_acl/README.md](kimi/k3.1_pro_acl/README.md)。Kimi必须开启思考、固定温度1，与Qwen推理预算及采样不同；不单独归因于模型。其他模型结果与数据均保留原样。

## Qwen3.5-4B LoRA实验

2026-11-04，完成ViT＋视觉投影＋混合注意力LoRA（r4/alpha8）3epoch，选验证NLL最低epoch=1。两者使用相同3708个ACL图像、英文提示与vLLM。

|模型|训练标签窗数|Macro-F1|Balanced accuracy|Accuracy|
|---|---:|---:|---:|---:|
|未微调4B匹配基线|0|0.075361|0.209059|0.162352|
|Qwen3.5-4B-lora|16461|0.673548|0.676463|0.623247|

完整记录见[qwen3.5_4B_lora_acl/README.md](qwen3.5_4B_lora_acl/README.md)。未覆盖历史中文Qwen结果，基础权重和所有既有结果指纹未变；验证选模、测试仅评价。

## Qwen4B英文vLLM：普通与IMU2IMG

2026-11-04，同一未微调基础模型和3708窗口：

|未微调Qwen3.5-4B＋vLLM＋英文|Macro-F1|Balanced accuracy|Accuracy|
|---|---:|---:|---:|
|普通六子图|0.098390|0.255631|0.216828|
|IMU2IMG学习图片适配|0.023739|0.142857|0.090615|

完整报告：[普通六子图](qwen3.5_4B_sixplot_en/README.md)、[IMU2IMG适配](qwen3.5_4B_imu2img_en/README.md)。两组Qwen均零样本，但IMU2IMG转换器已使用16461训练标签，且模态压缩/裁剪/分辨率不同；不将整个流水线称为零标签，也不单独归因于绘图。旧结果保持原样。

## 冻结Qwen的IMU2IMG对齐

2026-11-04，完成简化提示对照与仅训练转换器的监督对齐。Qwen所有参数冻结、同一3708窗、同一简化文本/图像格式/vLLM设置：

|同一冻结Qwen3.5-4B、相同3708测试窗|Macro-F1|Balanced accuracy|Accuracy|
|---|---:|---:|---:|
|旧转换器＋原波形先验提示|0.023739|0.142857|0.090615|
|旧转换器＋简化提示|0.023739|0.142857|0.090615|
|Qwen对齐转换器＋同一简化提示|0.125809|0.275129|0.297465|

原提示与简化提示验证成绩相同；新转换器按验证宏F1选epoch3，测试不参与选择。完整结果见[qwen3.5_4B_imu2img_aligned_en/README.md](qwen3.5_4B_imu2img_aligned_en/README.md)。835584转换器参数监督训练，整套不称严格zero-shot。旧结果保持原样。

## 三组相同LoRA配置实验

|实验|测试Macro-F1|Balanced accuracy|Accuracy|选中epoch|
|---|---:|---:|---:|---:|
|①普通六子图＋LoRA|0.755951|0.757997|0.685275|3|
|②固定IMU2IMG_qwen＋LoRA|0.703040|0.711713|0.640237|3|
|③联合IMU2IMG_qwen＋LoRA|0.712871|0.721404|0.648058|3|

[完整比较与配对区间](lora_comparison/README.md)。三组新LoRA配置相同，原始Qwen权重冻结，旧结果保留。
