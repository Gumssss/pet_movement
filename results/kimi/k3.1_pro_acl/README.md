# Kimi K3.1 Pro：ACL图片 API 测试结果

2026-11-04，已通过 **Moonshot官方国际API** 完成全部 **3,708个固定测试窗口**。结果使用同一dataset/vlm_acl中的WebP和完全相同的英文提示；窗口仍为2秒、200点、六通道、9只测试犬，没有重新划分或筛选。

## 总体结果

|模型与输入|Macro-F1|Balanced accuracy|Accuracy|
|---|---:|---:|---:|
|Qwen3.8-27B，ACL图，本地vLLM|0.123266|0.294250|0.196602|
|Kimi K3.1 Pro，相同ACL图，官方API|0.265092|0.408036|0.296117|

解析成功率 **100.00%**；状态分布：`{"ok": 3708}`。解析失败、拒绝及截断按错误纳入3708窗分母，不按真值重试。预测分布：`{"Galloping": 246, "Trotting": 674, "Standing": 1388, "Sitting": 165, "Lying chest": 118, "Sniffing": 866, "Walking": 251}`。

测试犬整体bootstrap 2,000次的95%区间：

- macro_f1：[0.237769, 0.283774]
- balanced_accuracy：[0.374847, 0.432613]
- accuracy：[0.279269, 0.311112]

只有9只测试犬，区间不覆盖提示选择或模型训练方差。

## 每类结果

|类别|Precision|Recall|F1|窗口数|
|---|---:|---:|---:|---:|
|Galloping|0.1016|0.8621|0.1818|29|
|Lying chest|0.3390|0.0460|0.0811|869|
|Sitting|0.4000|0.0812|0.1350|813|
|Sniffing|0.3522|0.4545|0.3969|671|
|Standing|0.2428|0.5871|0.3435|574|
|Trotting|0.3680|0.5962|0.4550|416|
|Walking|0.3068|0.2292|0.2624|336|

## 输入、API与比较范围

- 端点：`https://api.moonshot.ai/v1`，请求与每条响应的模型名均为`kimi-k3.1-pro`；未替换成其他型号。API是可变别名，服务端权重修订与推理框架不可检查，不宣称此服务使用vLLM。
- 同一匿名512×409无损WebP和[英文prompt.txt](prompt.txt)，只发送RGB图片的base64数据与固定问题。没有上传训练图、标签、文件名、犬只ID、Task或窗口标识；不调用工具或联网搜索。
- [官方参数说明](https://platform.kimi.com/docs/api/models-overview)规定K3.1 Pro始终思考，温度固定1、top_p固定0.95。本轮固定reasoning_effort=low、max_tokens=4096（包含推理预算）；没有强制JSON/schema解码，沿用原标签JSON解析器，仅解析最终message.content。完整API返回保存在原始记录中。
- Qwen上一轮关闭思考、温度0、最大输出32 token；Kimi的服务端视觉预处理、推理预算和采样不同。因此本轮是相同输入与任务的服务比较，不能把成绩差异单独归因于模型权重。Kimi实际内部图像尺寸/视觉token不可观测，不套用Qwen的208视觉token。
- 没有微调或少样本示例，绘图方法及原数据说明见[ACL作图适配](../../../dataset/vlm_acl/method.md)。只覆盖真值筛选后的整窗一致七种行为，不覆盖窗内行为切换。此前已查看过固定测试成绩，本轮不据输出调图或改提示。

## 用量与时间

- API返回输入累计 **4,308,696 token**，输出累计 **701,025 token**；逐响应usage保留全部缓存/推理细项，汇总见[usage_summary.json](usage_summary.json)。细项与总量存在包含关系，不重复相加。
- 本地各API运行会话累计墙钟 **4267.79秒**；本轮正式响应的客户端请求耗时P50=9.082秒、P95=16.902秒，包含网络、排队、服务推理及可能的退避等待。并发请求彼此重叠，耗时相加不等于总墙钟，也不能直接对比本地GPU核函数时间。
- 仅网络故障、HTTP429/5xx按相同载荷退避重试，已有响应不重跑；共记录0次额外传输尝试。实际费用以平台账单为准，未返回的重试可能包含未观测计费；这里不把估算标成实际扣费。
- 图像直接复用，没有新渲染或本地GPU模型载入。服务端GPU显存和耗电不可观测。

## 文件与复现

- [predictions.csv](predictions.csv)：3708窗真值、预测、最终文本、状态、token及耗时。
- [responses.jsonl](responses.jsonl)：完整API返回、最终回答、解析状态、用量、传输尝试、匿名图像指纹；不包含密钥或请求图片base64。
- [metrics.json](metrics.json)：总体、每类、每犬、混淆矩阵和bootstrap区间。
- [protocol.json](protocol.json)、[prompt.txt](prompt.txt)、[provider_info.json](provider_info.json)：冻结配置、实际文本和模型可用性检查。
- [input_validation.json](input_validation.json)、[validation.json](validation.json)、[prior_files_sha256.json](prior_files_sha256.json)：输入校验、原回答/CSV/独立重算一致性，以及92个历史文件未变的证据。
- [runtime.json](runtime.json)、progress.json、run.log：运行会话、代码指纹、进度与日志。first_request.log记录一次本地启动器导入失败（未发送图片），修正后首个正式响应计入全量结果。

API密钥仅从MOONSHOT_API_KEY环境变量读取，未写入项目、报告或命令参数；临时密钥文件已删除。不同协议或模型应使用新的独立目录，禁止覆盖历史实验。

```sh
.venv/bin/python scripts/kimi_classify.py all --concurrency 16
.venv/bin/python scripts/kimi_classify.py evaluate
```

同目录续跑校验冻结协议并跳过已有响应；全部结果已完成，无需重新调用API。仅重算评价无需密钥，先前3708个响应均保持原样。

官方依据：[国际站API](https://platform.kimi.ai/docs/get-your-api-key)、[K3.1 Pro](https://platform.kimi.com/docs/guide/kimi-k3-1-pro-quickstart)、[视觉输入](https://platform.kimi.com/docs/guide/use-kimi-vision-model)。
