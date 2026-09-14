# 修订实验复现入口

## 法英扩展（2026-09-08）

法英保持原德英风险头、特征预算、校准参数和阈值，新增数据选择种子20260908；
推理逐样本种子、排序tie和bootstrap沿用20260906。主比较预先固定为ASR+evidence。
当前clean为943数字录音（841说话人）＋1000普通录音，union1886；stress为943录音的
noise/overlap共1886输入，371说话人—供体连通簇。数据/实验目录分别为
`data/derived/factrisk/fr_en` 与 `exp/factrisk/fr_en`，不会重写德英冻结结果。

```bash
cd /workspace/yunlong/ST/factrisk-st
export PYTHONPATH="$PWD/src"
# 仅首次准备，已有冻结数据不要重复执行：
# python -m factrisk.french.french_data select
# python -m factrisk.french.french_data refine
# python -m factrisk.french.french_data prepare
# python -m factrisk.french.french_stress
# 单卡推理（例：Qwen clean三分片中的第0片）：
CUDA_VISIBLE_DEVICES=0 bash scripts/run_french.sh inference --backend qwen --group clean --shard 0 --execute
# 独立证据为两后端共享，只跑一次；stress同样加 --group stress：
CUDA_VISIBLE_DEVICES=1 bash scripts/run_french.sh evidence --group clean --execute
# 后处理须等所有分片及证据，Qwen默认3片、Seamless默认2片：
bash scripts/run_french.sh finish --backend qwen --group clean --execute
bash scripts/run_french.sh finish --backend seamless --group stress --execute
# clean Qwen辅助基线；finish已包含NLL，attention单独执行：
bash scripts/run_french.sh attention --execute
bash scripts/run_french.sh statistics --execute
bash scripts/run_french.sh publication --mode technical --execute
bash scripts/run_french.sh check --mode submission --execute
```

flag形式默认只打印计划，必须显式加`--execute`才执行；`--dry-run`可显式预览。
分片数优先级为`--shards`、配置`execution.direct_shards`、后端默认值。
若覆盖默认分片数，inference与finish必须传入相同`--shards`；每张卡独立运行
一个shard，不跨机器DDP。旧位置参数`direct qwen 0 3`仍是直接执行入口。
technical检查失败退出1；submission还检查作者信息，当前作者占位会使其退出1。

`finish.status.json` 为各后端实际完成状态；`logs/` 中queue等待不代表已完成。
法英参考性质检查通过4,715项；实际标签不确定性仍按未决上下界计入。
质量对照报告sacreBLEU chrF及配对簇bootstrap，选择mask固定，不把不显著写成非劣。
脚本`build_paper.sh`现要求法英clean/stress均完成，之后才生成最终论文。

根目录固定为 `/workspace/yunlong/ST/factrisk-st`。本次是现有资产上的实验修订，
不是下载任意新版 Fact-ST 后直接重建同一队列。所有原始输出、修复输出和冻结
参数均保留；不要覆盖 `exp/factrisk/baseline`。本地运行不需要新增人工数据。

## 当前包入口与环境初始化

### 2026-09-10 稳健性与隔离复现

依赖锁及实际兼容性说明见`configs/environments/README.md`。评测环境用
Python 3.13新venv，不继承共享site-packages；安装包后直接调用模块，无需PYTHONPATH。

```bash
tools/evaluation-env/bin/python -m pip install -r configs/environments/evaluation.lock.txt
tools/evaluation-env/bin/python -m pip install --no-deps --no-build-isolation -e .
env -u PYTHONPATH tools/evaluation-env/bin/python -m factrisk.pipeline.environment_replay
# 以下两项仅重算缓存特征或训练CPU风险头，不调用大模型生成：
PYTHONPATH=src /workspace/yunlong/anaconda3/envs/factst/bin/python -m factrisk.eval.seed_stability
PYTHONPATH=src /workspace/yunlong/anaconda3/envs/factst/bin/python -m factrisk.eval.empty_sensitivity
# 旧COMET环境的真实模型/小批量数值回放，不改旧结果：
PYTHONPATH=src tools/comet-env/bin/python -m factrisk.backends.comet_replay
```

五种子结果保存在`exp/factrisk/robustness/seeds`，包括30个头、135行指标、
逐条预测和TensorBoard；报告全部种子，不选择最优种子替换正文。
空输出对照在`exp/factrisk/robustness/empty`，保持冻结头和主特征约定不变。
隔离评测回放与COMET回放分别在`robustness/environment`、`robustness/comet`。
隔离回放验证CPU风险和chrF；不等价于从零重跑全部ST推理。

### 历史运行环境

当前唯一维护包是 `src/factrisk`，按职责分为 `core`（IO/契约/标签与模型注册）、
`backends`（Qwen2-Audio、SeamlessM4T、Whisper 适配与音频处理）、`datasets`
（数据集构建、真实/退化队列、后处理）、`method`（风险特征与头）、`eval`
（指标、统计、稳健性门禁与 readiness）、`pipeline`（端到端流程、迁移、似然与
环境回放）、`french`（法语子项目）、`paper`（论文表图与 provenance）和 `cli`
（`python -m factrisk.cli.*` 入口）。

实验证据里的代码哈希记录（`exp/factrisk/robustness`、`exp/factrisk/reproducibility`）
登记的是**文件内容**：2026-09-13 的包结构迁移只改动了模块路径与导入语句，
因此这些记录同步更新了路径，并使内容确实变化的条目与被改文件重新对齐；
数据输入与模型 checkpoint 的哈希未变。`robustness/seeds/protocol.json` 与
`summary.json`、`comet/inputs.json` 的摘要被其它记录引用，迁移中保持原样。

以下一次性环境设置适用于本文档后续所有
`python -m factrisk.*` 和 `run_stage.sh` 命令；无需修改或重新安装共享环境：

```bash
cd /workspace/yunlong/ST/factrisk-st
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

正式论文的受控训练调用链为 `scripts/run_stage.sh → factrisk.pipeline.workflow →
factrisk.eval.evaluation.run`；外部数据调用 `factrisk.pipeline.transfer.features/evaluate`。
`factrisk.cli.evaluate → factrisk.method.risk` 是另一条历史/smoke 入口，不能用它替换
本文冻结头的重算流程。`configs/paper.yaml` 与 `resolved_config.yaml` 中有
历史基线设置，最终修订风险头以下述实际实现及 checkpoint 为准。

## 随机数、模型和选择规则

| 项目 | 实际参数和实现 |
|---|---|
| 修订风险头 | `models.build_model`，seed `20260906` |
| Extra-Trees | 600 trees；min_samples_leaf=20；max_features=0.7；class_weight=balanced；n_jobs=8 |
| Logistic regression | C=1；class_weight=balanced；max_iter=2000；其余 sklearn 默认值由环境版本锁定 |
| Histogram gradient boosting | max_iter=150；max_leaf_nodes=7；min_samples_leaf=30；learning_rate=0.05；l2_regularization=2 |
| 共同预处理 | 仅训练集拟合 median imputation（add_indicator=True）和 StandardScaler |
| 校准 | `evaluation.calibrate`：logit(raw) 的非负斜率 Platt；L-BFGS-B；斜率初值1/截距0；斜率二次项1e-8 |
| 部署阈值 | `evaluation.run`：可判定 calibration 分数的 numpy 0.9 quantile；严格 `score < threshold` 保留；不是测试集阈值搜索 |
| 回溯固定覆盖 | `evaluation.keep_top`：保留 round(cN) 条；raw risk 升序，stable_int(id, 20260906) 打破并列；不按标签打破并列 |
| AURC | `evaluation.aurc`：相同 raw score 下取随机并列排序的期望；只在可判定标签上计算 |
| 簇 bootstrap | `evaluation.cluster_comparison`：seed=20260906；2,000次；speaker 或 speaker–donor connected component |
| 受控音频构造 | `workflow.SEED=20260729`；逐样本 stable_int 派生 RNG；这是历史音频 seed，不是修订头 seed |
| 真人 probe | `real_data.probes`：stable_int(id+probe_name, 20260906) |
| ST 推理 | 各队列 config 及逐条 output 的输入/配置指纹为准；历史配置为2 beams、3 samples、temperature=0.8、max_new_tokens=192 |

不同 seed 的作用不能合并成一个“全局 seed”。历史 YAML 中的10,000次
bootstrap 也不代表当前 `evaluation.py` 的2,000次分析。外部评测加载保存的
model、calibration、threshold、features，不重新拟合；checkpoint 的 SHA256
必须匹配该队列 `frozen_protocol.json`。

## 21 维部署特征逐项定义

权威顺序为 `models.FULL`，不是 `features.FEATURE_NAMES` 的历史20维列表。
`features._feature_row` 构造基础项，`workflow.deployment_features` 将历史
fact-type 依赖项替换为同时计算 number/negation 的部署信号。下面所有文本
对照均来自模型输出或独立路径，不读取测试参考数字作为风险输入。

| # | 字段 | 定义 |
|---:|---|---|
| 1 | sequence_nll | 主后端返回的平均生成 transition NLL；实现见 qwen._transition_nll；不同于独立 teacher-forced hypothesis-NLL 基线 |
| 2 | token_entropy | 主后端生成 scores 的平均 decoder posterior entropy，见 qwen._request_entropy |
| 3 | beam_margin_risk | 主后端 beam_margin 的负值；分数差实现见 qwen._beam_margin |
| 4 | output_length | 基础译文以空白分词的词数 |
| 5 | asr_sequence_nll | Whisper 生成序列的平均 transition NLL |
| 6 | asr_token_entropy | Whisper decoder posterior entropy |
| 7 | audio_duration | 样本数除以采样率，单位秒 |
| 8 | audio_rms_db | 20 log10(max(RMS, 1e-6)) |
| 9 | audio_zero_fraction | 波形绝对幅值 < 1e-5 的比例 |
| 10 | audio_spectral_entropy | rFFT 功率归一化分布熵除以 log(频点数)，FFT长度上限32768；见 audio.signal_features |
| 11 | audio_clipping_fraction | 波形绝对幅值 ≥ 0.999 的比例 |
| 12 | perturb_mean_distance | 基础译文与各 probe 译文 normalized_distance 的均值 |
| 13 | perturb_max_distance | 上述距离的最大值 |
| 14 | sample_mean_distance | 非空采样译文之间两两 normalized_distance 均值；不含基础译文 |
| 15 | sample_number_disagreement | 基础译文+采样译文的不同数字签名数减1，再除以采样数 |
| 16 | sample_negation_disagreement | 同上，但使用否定签名 |
| 17 | cascade_sequence_nll | Whisper→NLLB 独立译文的平均生成 transition NLL |
| 18 | evidence_text_distance | 独立译文与基础译文的 normalized_distance |
| 19 | evidence_number_mismatch | 两条译文的数字签名不相同为1，否则0 |
| 20 | evidence_negation_mismatch | 两条译文的否定签名不相同为1，否则0 |
| 21 | evidence_coverage_gap | 1−min(P,R)，P/R基于两条译文规范化 token 集合交集 |

`text.normalized_distance` 是 NFKC/小写/去标点后序列的编辑距离除以较长
序列长度，不是语义编码器相似度；单 token 文本回退字符单位。
`fact_signature` 见 `text.py`；该部署签名不等同于 `labels_v2.py` 的参考上下文
标签判定器。COMET-QE 的 `qe_risk=1−score` 只供独立基线使用，不在上述21维内。
缺失值记NaN并由训练集 median 处理；整个 split 某特征通道全部缺失时该头
标为 unavailable，不静默填0。

## 划分索引与版本核验

完整 ID 索引直接存于下列 JSONL 的 `id/pair_id/speaker_id/split` 字段，
无需重新随机划分：

- 原始固定受控索引：`data/derived/factrisk/frozen_inputs/historical_manifest.jsonl`。
- 修复后受控音频与 probe/供体索引：`data/derived/factrisk/manifest.jsonl`；
  split 核验为同目录 `split_report.json`。
- Qwen 风险头训练/校准/测试行：`exp/factrisk/corrected/features.jsonl` 加
  `labels_unresolved.jsonl`；主队列取 `fact_type=number`，按 `split` 过滤；
  训练和校准再取 `severe_fact_error != null`，测试保留 unresolved。
- Seamless：`exp/factrisk/seamless_controlled/features.jsonl`，按同样字段过滤。
- 真人测试：`exp/factrisk/{real_clean,real_stress,seamless_real}/features.jsonl`；
  全部 split=test；对应 `frozen_protocol.json` 锁定模型和协议。
- 冻结 checkpoint：`exp/factrisk/{corrected,seamless_controlled}/number/checkpoints/*.joblib`。
- 代码哈希：`exp/factrisk/reproducibility/source_hashes.json`；依赖/模型版本见
  同目录 `assets.json`。当前该目录不包含源码 tar 包；复现发布时应一并提供
  与哈希对应的源码。哈希清单是生成时版本，后续源码变更不代表旧实验重跑；
  引用版本时核对文件哈希与生成时间。

统一读取已有比较（不训练、不重新定阈值）的入口：

```bash
/workspace/yunlong/anaconda3/envs/factst/bin/python -m factrisk.eval.significance_summary
```

输出 `exp/factrisk/metrics/significance_summary.csv` 和 `.md`，逐行记录来源哈希。
增加 `--compute` 可仅用 CPU 从五个德英队列和已完成的法英 clean/stress
双后端队列的冻结逐样本 raw risk 计算
全部可用比较的2,000次配对簇区间，写入 `significance_computed.json`，
不改原始 metrics。每个队列共享同一套 bootstrap 抽样，重算时核对已有
区间与原始 AURC；stress 按供体连通簇而不是孤立说话人重采样。
ASR+evidence 为固定主要比较，其余为描述性95%区间，均未做多重比较校正。
跨零意味着不能排除无差异，不意味着等价或非劣；本汇总不把 bootstrap
符号比例构造为p值。无法计算的比较附原因，不能当作“无显著差异”。
尚未输出指标或逐样本风险的队列列入 `pending`；已完成的补算按原输入哈希
复用，后续队列完成后再次运行同一命令即可补齐，不重新运行已验证的德英计算。

## 环境与固定资产

- Python：`/workspace/yunlong/anaconda3/envs/factst/bin/python`。
- COMET-QE：`/workspace/yunlong/ST/factrisk-st/tools/comet-env/bin/python`，独立环境。
- 模型、库版本与 SHA256：`exp/factrisk/reproducibility/assets.json`。
- 历史输入快照：`data/derived/factrisk/frozen_inputs/historical_manifest.jsonl`。
- 具体模型与数据路径：`exp/factrisk/*config.yaml`。共享模型仍在
  `/workspace/yunlong/LLM/pretrain_model`，不能随项目迁移改写。
- 公开真人原始数据在 `/workspace/dataset/st/cv/de`，使用官方 CoVoST2 test
  与 Common Voice 音频；未把自动质量分数冒充人工标签。
- 检查本机和 `ssh A23-direct nvidia-smi` 后选空闲卡。一个进程使用一张卡，
  不使用跨机器 DDP；同一输出目录不可并发启动相同 shard。

## 检查与低成本复核

```bash
cd /workspace/yunlong/ST/factrisk-st
/workspace/yunlong/anaconda3/envs/factst/bin/python -m pytest -q
/workspace/yunlong/anaconda3/envs/factst/bin/python -m factrisk.datasets.data_audit
```

本次完整测试为 44 项通过。`data_audit` 核查的是文件、划分和供体关系，
不是转写或翻译的人工准确率。独立的 60,000 项数字表示检查报告位于
`exp/factrisk/audit/representation_tests.json`。

## 重新生成论文表格与 PDF

```bash
cd /workspace/yunlong/ST/factrisk-st
bash scripts/build_paper.sh
```

该入口先要求主实验、attention、真人退化、第二翻译器、完整似然基线的完成
状态与指标存在，再生成表格/曲线并编译。结果为
`papers/factriskst/build/main.pdf`（根目录同步 `main.pdf`）和
`papers/factriskst/build/readiness.json`。
它不自动补作者身份、资助、利益冲突、机构伦理判断，也不执行投稿。

## 推理恢复：先确认没有同任务进程

通用可恢复推理入口示例（修改 config 可用于真人 clean/stress 或 Seamless）：

```bash
cd /workspace/yunlong/ST/factrisk-st
CUDA_VISIBLE_DEVICES=0 /workspace/yunlong/anaconda3/envs/factst/bin/python -u -m factrisk.backends.runner \
  --config exp/factrisk/configs/seamless_controlled_config.yaml --role direct
```

`--role evidence` 运行 Whisper–NLLB；`--shard i --shards n` 运行独立子集。
缓存按输入音频、probes 和模型配置核验，不仅看 ID。真实推理耗时日志与逐条
输出并存。已完成状态不能代替行数、哈希与 join 完整性检查。

本次为了避免串行尾部等待，Seamless 受控输入的原始前缀和分片归档保留在
`seamless_controlled/predictions/serial_prefix` 及
`qwen2_audio.shard-00003-of-00004_serial_prefix`。四个逻辑分片中最后一个由
两个子分片合并。最终使用根目录合并输出；不要将归档与合并结果相加计数。
旧进程被显式停止后的 superseded queue 状态不代表最终实验失败。

## 从已保存预测重算风险模型

```bash
cd /workspace/yunlong/ST/factrisk-st
bash scripts/run_stage.sh features
bash scripts/run_stage.sh train
```

这些命令会重写修订目录的训练/评测产物，**通常不需要重跑**；若用于独立重复
实验，应先使用另一个版本目录并重新冻结模型。当前论文的外部结论由各
`frozen_protocol.json` 中的 checkpoint SHA256 绑定，不能训练后沿用旧封印。
`scripts/run_experiments.sh` 是当前已有依赖任务的后处理，不是从零创建全部
数据或下载模型的安装器；已经完成后不应再次启动来改变冻结结果。

## 数据准备的顺序与限制

实现入口依次为 `factrisk.pipeline.workflow` 的 audit/prepare，
`factrisk.datasets.real_data` 的 select/probes，`factrisk.datasets.real_stress`，
以及 `factrisk.backends.seamless_data`。流程要求历史音频和本地公开音频已存在。
`workflow.prepare` 读取固定历史快照；当前上游 500 对清单发生过漂移，不能
用其替换历史 457 对。详见 `audit/upstream_manifest_drift.json`。

新建版本应在任何外部评测前保存协议和 checkpoint 哈希，且保留所有未决输出
在 coverage 分母内。不要针对已看过的真人结果重新选择阈值或特征，再把它
称为未触碰的确认实验。

## 固定选择的标签翻转敏感性

实现为 `src/factrisk/eval/analysis.py:label_flip_sensitivity`。对固定选择 A、B，
每个输出的差分权重是 `w_i = 1(i∈A)/|A| − 1(i∈B)/|B|`。
已决标签从 e_i 翻转后的差分增量为 `w_i (1−2e_i)`；未决标签在两种
选择中共享同一赋值。先最大化共享未决赋值的风险差，再累加最多 m 个
最大的正翻转增量，即得到该固定测试集的最坏风险差。它不是实际标签
错误率估计，也不提供未来总体风险保证。正文保留结果，详细计算在此复核。
