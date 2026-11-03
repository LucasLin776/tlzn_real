# Wind Audio · 风机音频异常检测

一个仅使用 WAV 音频的风机声音异常定位原型，支持正常基线、有监督分类、事件级评估和人工审核。

> **项目状态：研究原型，非生产故障诊断系统。** L1 检测声音异常；L2 故障分类尚未实现。合成数据只能验证程序流程，不能证明真实风机召回率或误报率。已知限制见 [KNOWN_ISSUES](docs/KNOWN_ISSUES.md)。

## 功能

- 从 PCM WAV 提取 log-mel、能量和多尺度上下文特征。
- L1 二选一：正常帧 median/MAD 基线，或 XGBoost / sklearn / NumPy 有监督分类。
- 支持精确区间标注，保留未审核区域，按原始录音或批次分组。
- 在验证集选择阈值和后处理，测试集只用于独立评估。
- 导出异常时间、音频片段、HTML 试听报告和人工审核 CSV。
- 保存模型配置、后端和版本；预测分数**不是经过校准的故障概率**。

## 算法结构

```text
WAV → 单声道 / 重采样 → STFT / log-mel / 能量与上下文特征
                            │
                 ┌──────────┴───────────┐
                 │                      │
          baseline：正常分布      supervised：正常/异常分类
                 │                      │
                 └──────────┬───────────┘
                            ▼
                异常分数 → 双阈值 / 合并 / 时长筛选
                            ▼
                异常区间 + 音频片段 + 试听报告
                            ▼
                  人工复核 → L1 反馈 / L2 数据积累
```

默认分析采样率 44.1 kHz，STFT 窗长 2048、帧移 441；有监督模型每 0.1 秒使用 282 维特征。特征包括当前帧、前后约 1 秒统计，以及 0.25/0.5/1/2 秒左右差异。当前实现需要整段录音和未来上下文，**不是即时流式算法**。

## 安装

要求 Python **3.10 或以上**。下面命令在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[sklearn]'
python -m wind_audio --help
```

Windows PowerShell 激活环境使用 `.\.venv\Scripts\Activate.ps1`，其余命令使用 `python`。

安装选择：

| 用途 | 安装命令 |
|---|---|
| 仅基线 / NumPy 后端 | `python -m pip install -e .` |
| sklearn CPU 训练（推荐起步） | `python -m pip install -e '.[sklearn]'` |
| XGBoost CPU 训练 | `python -m pip install -e '.[xgboost]'` |
| 开发与测试 | `python -m pip install -r requirements-dev.txt` |

也支持 `python -m pip install -r requirements.txt`，默认安装 sklearn 版本。安装后可用 `wind-audio` 代替 `python -m wind_audio`。元数据的依赖范围不是锁定环境；训练交付时请另存实际依赖版本。

## 快速开始：只用合成数据跑通流程

不依赖仓库中的真实音频或历史模型。以下目录是新产物位置；若已存在重要内容，请使用新的版本名。

```bash
python -m wind_audio.synthetic --out data/synthetic_demo --seed 0

python -m wind_audio check \
  --manifest data/synthetic_demo/manifest.csv \
  --annotations data/synthetic_demo/annotations.csv

python -m wind_audio split \
  --manifest data/synthetic_demo/manifest.csv \
  --annotations data/synthetic_demo/annotations.csv \
  --out data/synthetic_demo/splits.json --group-by source --seed 7

python -m wind_audio train \
  --manifest data/synthetic_demo/manifest.csv \
  --annotations data/synthetic_demo/annotations.csv \
  --splits data/synthetic_demo/splits.json \
  --detector supervised --algo sklearn --no-fallback \
  --out models/demo_sklearn --version demo_sklearn --target-recall 0.90

python -m wind_audio evaluate \
  --model models/demo_sklearn \
  --manifest data/synthetic_demo/manifest.csv \
  --annotations data/synthetic_demo/annotations.csv \
  --splits data/synthetic_demo/splits.json \
  --split test --out reports/demo_sklearn

python -m wind_audio run \
  --input data/synthetic_demo/audio/syn_mixed.wav \
  --model models/demo_sklearn --out runs/demo_sklearn
```

打开生成的 `runs/demo_sklearn/report.html` 查看试听报告。模型保存成功不代表校准成功：请检查 `meta.json` 的 `calibrated` 和 `calibration.json` 的 `target_reached`。

> 合成数据包含共享背景，测试集很小；上述步骤只是功能演示，不能作为独立泛化证据。XGBoost 有专门的可选测试，只有安装该依赖后才会执行。

## 使用真实音频训练

### 1. 准备音频与标注

建议新建独立数据版本，例如 `data/project_v1/audio/`，将原始 WAV 放入其中。简单标注格式见 [示例](examples/labels.example.csv)：

```csv
file,start_s,end_s,label,group
sample_a.wav,0,12,normal,recording_001
sample_a.wav,12,14,anomaly,recording_001
sample_a.wav,14,30,normal,recording_001
```

- `normal`：人工确认正常；`anomaly`：需要关注的异常波动；`uncertain`：不确定。
- 未标注区域默认是 unknown，不进入训练。
- 同一连续录音的切片必须共用 `group`，否则无法保证独立划分。
- 粗标“整段包含异常”不等于整段都是异常，不能作为精确区间导入。
- 时间必须以实际 WAV 为准；示例文件名和时长是虚构的。
- 设备编号、时间和分组不进入模型特征，只用于划分与追溯。

### 2. 导入、检查、划分

```bash
python -m wind_audio import-simple \
  --audio-dir data/project_v1/audio --labels data/project_v1/labels.csv \
  --manifest data/project_v1/manifest.csv \
  --annotations data/project_v1/annotations.csv

python -m wind_audio check \
  --manifest data/project_v1/manifest.csv \
  --annotations data/project_v1/annotations.csv

python -m wind_audio split \
  --manifest data/project_v1/manifest.csv \
  --annotations data/project_v1/annotations.csv \
  --out data/project_v1/splits.json --group-by source --seed 42
```

至少需要 3 个独立分组才能运行划分，这不是可信性能的样本量标准。检查 train 是否有足够正常和异常样本，val/test 是否都有异常事件和正常时长。仅有文件级粗标不能训练真实监督模型。

### 3. 训练与校准

```bash
python -m wind_audio train \
  --manifest data/project_v1/manifest.csv \
  --annotations data/project_v1/annotations.csv \
  --splits data/project_v1/splits.json \
  --detector supervised --algo sklearn --no-fallback \
  --max-samples-per-file 5000 --max-total-samples 50000 \
  --cache-dir cache/project_v1 \
  --out models/l1_v1 --version l1_v1 --no-calibrate

python -m wind_audio calibrate \
  --model models/l1_v1 \
  --manifest data/project_v1/manifest.csv \
  --annotations data/project_v1/annotations.csv \
  --splits data/project_v1/splits.json --split val --target-recall 0.95
```

先单独训练可将模型拟合与阈值搜索分开诊断。目标召回未达到时，校准会优先保留较高召回，但不代表模型已经达标。大数据集建议先用代表性小样本试跑；样本上限只限制拟合样本，不是整个预处理过程的内存上限。

对照基线：把训练命令改为 `--detector baseline`，移除 `--algo` / `--no-fallback`，并使用不同输出目录。基线只用确认正常的帧拟合。

### 4. 测试与人工复核

使用快速开始中的 `evaluate` / `run` 命令，替换为真实数据和模型路径。冻结参数后才能评估 test，不能拿 test 重新校准。评估指标仍有已知限制，见 [KNOWN_ISSUES](docs/KNOWN_ISSUES.md)；正式效果结论需要先修复指标并用独立真实数据复核。

`run` 输出报告、事件 JSON/CSV、音频副本、片段和审核模板。审核中 `anomaly + nonfault` 仍是 L1 正例；只有明确的 `normal` 才是 L1 正常反馈。跨模型版本的审核暂应分开保存。L2 仅预留数据接口，不会自动训练。

## 目录

```text
wind_audio/       算法、训练和命令行
tests/           unittest 单元与集成测试
docs/            标注、工作流、模型说明及已知问题
examples/        无真实数据的标注模板
scripts/         发布包检查
.github/         CI、Issue 和 PR 模板
pyproject.toml   标准 Python 安装、依赖及工具配置
data/            本地数据，不应提交（保留使用说明）
models/          本地模型，不应提交
cache/           特征缓存，不应提交
runs/ reports/   推理与评估产物，不应提交
```

## 开发与验证

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -t . -v
python -m ruff check wind_audio tests
python -m build
python scripts/check_distribution.py dist
```

测试生成临时合成音频，不依赖生产数据。CI 配置覆盖 NumPy 核心、sklearn 和 XGBoost 环境；是否通过以实际 Actions 结果为准。

## 上传 GitHub 与数据安全

先阅读 [上传检查清单](docs/GITHUB.md)。`.gitignore` **不会自动移除已经被 Git 跟踪或写进历史的音频、模型、缓存**；上传前必须检查已有索引与历史。不要直接用 `git add .` 代替检查。

仅加载可信来源的模型和缓存：sklearn 模型使用 pickle，部分缓存允许对象反序列化。不可信文件可能执行代码。详见 [安全说明](SECURITY.md)。

## 文档与许可证

- [标注规范](docs/ANNOTATION.md)
- [完整工作流](docs/WORKFLOW.md)
- [模型格式](docs/MODEL.md)
- [已知限制](docs/KNOWN_ISSUES.md)
- [贡献说明](CONTRIBUTING.md)
- [变更记录](CHANGELOG.md)

当前没有指定开源许可证，本次整理不自动授予开源使用权。仓库所有者应确认代码和数据授权后选择许可证；在此之前不要将本项目标记为 MIT / Apache-2.0 等已授权开源项目。
