# 实际运行依赖锁

三个lock文件分别由对应实际Python的distribution metadata递归导出，仅包括该任务的
运行依赖闭包。JSON记录Python/平台、包版本及已安装依赖约束冲突。导出命令：

```bash
PYTHONPATH=src /workspace/yunlong/anaconda3/envs/factst/bin/python -m factrisk.pipeline.environment_lock evaluation
PYTHONPATH=src /workspace/yunlong/anaconda3/envs/factst/bin/python -m factrisk.pipeline.environment_lock inference
PYTHONPATH=src tools/comet-env/bin/python -m factrisk.pipeline.environment_lock comet
```

## 评测环境

```bash
python3.13 -m venv tools/evaluation-env
tools/evaluation-env/bin/python -m pip install -r configs/environments/evaluation.lock.txt
tools/evaluation-env/bin/python -m pip install --no-deps --no-build-isolation -e .
tools/evaluation-env/bin/python -m pip check
```

本轮使用不继承system-site-packages的新venv。隔离重算报告记录在
`exp/factrisk/robustness/environment/`。2026-09-10安装、pip check和27组风险指标
及两个1000条general chrF回放已通过；最大风险差1.11e-15。锁文件存在本身
不代表未运行的其它环境也已通过安装/复现。
推理环境同样可用新venv安装inference.lock.txt，模型与数据由原资产记录定位，
Tectonic二进制另见项目tools目录，不属于pip依赖。

## COMET历史环境

现有tools/comet-env通过system-site-packages继承factst包。unbabel-comet 2.2.7
声明的numpy<2、transformers<5和huggingface-hub<1与实跑版本不一致，详情见comet.json。
comet.lock.txt记录的是**观测到的可执行版本组合**，不是满足所有上游约束的环境。
完整闭包可用`pip install --no-deps -r configs/environments/comet.lock.txt`精确回放，
但安装后pip check仍将报告上述冲突。不得将其宣称为依赖约束全绿，也不得为消除
警告而静默升级/降级再混用旧QE缓存。

2026-09-10：已在原环境加载冻结COMET-QE模型，并回放16条缓存输入。
最大绝对差为1.10e-6，小于预设1e-5；模型/数据/缓存均未修改。
实际报告见`exp/factrisk/robustness/comet/report.json`。这验证旧环境的执行和
数值回放，不等价于全新安装通过依赖解析器；inference锁也未执行完整新环境GPU重跑。
