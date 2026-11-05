# Wind Audio：风机音频两级检测

新增独立 L1 模型类型 **`panns_temporal_dae_v1`**（仅完成代码，尚未训练/校准）：
冻结 PANNs 中浅层时序特征 → 多尺度动态表示 → 小型时序去噪自编码器 → 网络残差事件。
使用 `l1-panns-*` 命令，或在 `l1-detect --config` 中显式选择该模型；旧默认行为和 L2 不变。
数据必须显式指定配对、标注语义及来源划分，unknown 不推导为正常。
新训练入口支持 `--audio-dir`、`--labels` 和 `--check-only`；完整流程在正常训练、独立正常校准后，
自动生成 `val/comparison.pdf`、`test/comparison.pdf` 及配套 CSV/JSON/简报（本次只实现代码，未运行训练）。
详见 [L1 PANNs 结构、约束与占位路径命令](docs/L1_PANNS.md)。新 L1 需要已有 PyTorch 和本地 CNN14 权重，不自动安装或下载。

当前工程版本 **0.11.0**。**normal_envelope_v1 正常样本学习模型**支持多批次目录一次训练
（`--dataset-dir`）和跨批次混合打散划分（`--split shuffle`）；原有
`spectral_stability_v3`、`spectral_event_v2` 配置保留，可复现历史结果。
不传 `--config` 仍使用旧有 v3 默认规则；使用新路线必须传训练得到的 `model.json`。
当前模型在 `runs/normal_mixed_v1/model.json`（B05+B06+B07 共 196 条）。

| 层 | 输入与作用 | 输出 | 是否训练 |
|---|---|---|---|
| L1 正常模型 | 仅从标注正常区间学习允许的频谱时间波动，过滤轻微短噪音 | 正常区间、候选异常窗口、可试听切片 | 需要正常样本；独立正常片段校准阈值 |
| L1 | 原始 WAV → 多尺度稳定性（快变化/慢漂移/局部噪声容差）+ 事件状态机 | 候选事件 `[start_s, end_s]`，只关心有没有事件及其范围 | 无需训练，使用规则与稳健统计量 |
| L2 | 原始 WAV + L1 窗口 → 冻结 PANNs CNN14 特征 → 分类头 | 已知异常类型 / 疑似正常 / unknown、相似案例 | 需要人工审核数据；支持回放续训 |

L1 的业务前提是：**在足够长的观察时间内频谱基本稳定即正常**，不要求固定音色、频率位置或响度；
明显超过局部噪声容差的变化才是事件，工况切换后重新稳定即结束并接受新背景。
L1 不区分事件类型，也不直接判断故障。L2 负责类型判断，保留原 L1 告警；
人工审核后更新分类头。L2 无法补回 L1 未提供的候选事件。

## 运行环境

```bash
conda activate post_training
cd /Users/shuoqinlin/Desktop/wind_audio
```

L1 检测只用 NumPy；测试版默认生成图表/PDF，另需 Matplotlib 和 ReportLab；L2 的 CNN14 还需 PyTorch。
依赖见 `requirements.txt`；当前 post_training 已安装图表/PDF 依赖。
从工程目录可直接运行 `python -m wind_audio`，无需安装本项目。
WAV 支持 8/16/32 位整数 PCM；24 位及浮点 WAV 需先转换。多声道平均为单声道。

## L1 的两个入口

| 入口 | 数据输入 | 用途与输出 |
|---|---|---|
| `l1-detect` 检测版 | `--input` 原始 WAV 文件或目录 | 实际工作：输出事件窗口、切片、可试听报告；不接收标签 |
| `l1-test` 测试版 | `--audio-dir` 音频目录 + `--labels` 标签文件或目录 | 测试与改进：调用同一 L1，接入检验器，输出准确率、召回率、误报和逐条对比 |

两个入口共享算法及 `config/l1_v3.json`。标签仅在测试版中参与结果比较，不进入检测算法。
旧命令 `run`、`evaluate` 分别作为两者的别名保留；`run` 也不再接受标签输入。
实际检测仍是离线 WAV 分析，不是实时流式服务。

### 多批次目录：一次训练全部批次（混合打散）

数据按「批次文件夹」交付：一个根目录下放多批，每批固定为 `音频/` + `标注/`。
标注区间是**正常/平稳区间**，录音内其余时间按异常处理，编号顺序与音频时间顺序一致。

```text
新一批/
  B05/音频/*.wav   B05/标注/*.csv
  B06/音频/*.wav   B06/标注/*.csv
  B07/音频/*.wav   B07/标注/*.csv
```

`--dataset-dir` 自动发现全部批次、合并成一次训练。批次目录默认
`--pairing sorted`（编号顺序=音频时间顺序）与 `--split shuffle`（跨批次全局打散后再按
60/20/20 划分）；单批 `--audio-dir/--labels` 保持旧行为（同名配对、按时间顺序切分）。

```bash
# 1) 训练
python -m wind_audio l1-train-normal \
  --dataset-dir /Users/shuoqinlin/Desktop/数据/新一批 \
  --fully-reviewed-normal \
  --out runs/normal_mixed_v1

# 2) 全量对照报告：逐条图片 + comparison.pdf + 全部 CSV/JSON/简报
python -m wind_audio l1-test \
  --dataset-dir /Users/shuoqinlin/Desktop/数据/新一批 \
  --config runs/normal_mixed_v1/model.json \
  --out runs/normal_mixed_v1_report

# 3) 实际检测（只要候选窗口和切片，不含人工对比）
python -m wind_audio l1-detect \
  --input /Users/shuoqinlin/Desktop/数据/新一批 \
  --config runs/normal_mixed_v1/model.json \
  --out runs/normal_mixed_v1_detection
```

第 2 条命令的输出与 `l1-test` 单批口径完全一致，共 9 个文件加 `plots/`：

| 文件 | 内容 |
|---|---|
| `comparison.pdf` | 全部录音按 ID 自然顺序，一条一页（196 页），上行为人工正常区域、下行为模型正常区域 |
| `plots/0001_B05-001.png` | 每条录音的四联图：波形、偏离曲线、原始区间对比、0.3 s 合并辅助对比 |
| `plots/index.csv` | 页码 → audio_id → 原音频路径 → 图片，四列 |
| `metrics.json` | 全部指标、逐文件明细、实际参数、IoU 匹配、PDF 元信息 |
| `per_file.csv` | 每条录音的准确率/精确率/召回率与 TP/TN/FP/FN |
| `normal_regions.csv` | 人工与模型正常区间逐段对照（含匹配序号与 IoU） |
| `labels.csv` | 交付标签：每条录音一行，整段减正常段，>5 段则整段返回 |
| `brief.md` | 中文摘要 |
| `manifest.csv` / `annotations.csv` / `pairing.csv` | 合并后的输入快照（含批次归属） |

图片和 PDF 默认开启；`--no-plots` 只出数值。合并评估目前只接受批次目录与正常/平稳口径，
`--target anomaly` 仍走单批 `--audio-dir` 通路。

本次已训练的模型为 `runs/normal_mixed_v1/model.json`，训练记录、逐文件批次归属和逐集合指标
在同目录 `training_report.json`；`manifest.csv/annotations.csv/pairing.csv` 是合并后的输入快照。
训练输出目录必须是新的；`--seed` 固定打散结果，可复现同一划分。
正常模型学习的是**允许多大时间波动**，不是固定音色；轻微短偏离先过滤，强短事件仍保留，
相邻异常可以合并，新工况稳定后结束报警。正常片段边界留出完整上下文，防止邻接异常污染训练。

B05 96 条 + B06 50 条 + B07 50 条 = **196 条 / 3 台设备**，`seed=0` 打散为 **118 训练 / 39 校准 / 39 留出检查**，
只有前两组的正常内部帧用于学习（104,744 / 41,081 帧）：

| 集合 | 条数 | 时长准确率 | 异常时长召回 | 时长精确率 | 事件召回 | 正常误报时长 |
|---|---:|---:|---:|---:|---:|---:|
| 训练 | 118 | 0.837 | 0.481 | 0.751 | 0.717 | 136.3 s（正常时长的 5.07%） |
| 校准 | 39 | 0.879 | 0.453 | 0.724 | 0.684 | 34.1 s（3.50%） |
| **留出检查** | 39 | **0.817** | **0.480** | **0.717** | **0.681** | **57.4 s（6.61%）** |

留出集按批次看差别很大：B05 误报 1.4%、异常覆盖 21.9%；B06 误报 8.3%、覆盖 72.2%；
B07 误报 14.1%、覆盖 68.1%。打散让同一设备的录音同时进入训练与留出检查，
**留出指标是内插估计，不是独立设备/跨会话测试**；要跨设备结论必须留出一整台设备。

第 2 条命令的全量回放（196 条，含训练数据，不是盲测）：正常时长精确率 85.84%、
召回率 94.97%、时长准确率 84.07%，人工正常 713 段 / 模型 734 段 / 匹配 565 段；
异常错当正常 709.3 s、正常被排除 227.7 s。`brief.md` 与 `metrics.json` 里有完整口径，
`comparison.pdf` 是 196 页逐条对照，用来人工复核漏检和误报的位置。
详见 [正常样本学习说明](docs/NORMAL_MODEL.md)。

### 检测版：仅指定原始音频

```bash
python -m wind_audio l1-detect \
  --input /Users/shuoqinlin/Desktop/数据/新一批 \
  --config runs/normal_mixed_v1/model.json \
  --out runs/l1_detect_latest
```

替换 `--input` 就能切换实际工作数据；也可以直接指定单条 `.wav` 的完整路径。
目录输入生成 `index.html` 和逐文件报告；单文件输入直接生成 `report.html`。
`--input` 递归扫描 WAV，因此直接传批次根目录会一次性处理全部批次。
`review.csv` 是新生成的空审核模板，不是检测输入。

### 测试版：指定音频和标签

```bash
python -m wind_audio l1-test \
  --audio-dir /Users/shuoqinlin/Desktop/数据/新一批/B05/音频 \
  --labels /Users/shuoqinlin/Desktop/数据/新一批/B05/标注 \
  --pairing sorted --labels-are-stable \
  --config runs/normal_mixed_v1/model.json \
  --no-plots --out runs/check_B05
```

`l1-test` 仍按**单个批次**评估（多批次合并评估尚未提供，用 `training_report.json` 的
`metrics.<集合>.per_file` 看合并后的逐文件结果）。
本数据集已确认：B05-001～096 对应音频按时间排序；所有音频完整审核；
标注区间是正常/平稳区间，其余时间为异常。
`--pairing sorted` 仅用于已确认顺序对应的数据；默认按同名配对。
输出目录必须为空或不存在，避免覆盖既有标注和评估结果。
调整 `--threshold` 等参数或编辑配置后，用新的输出目录对比结果；测试命令不会自动训练 L1。
确定参数后，检测版使用相同配置即可。正式测试集不应反复用于调参。

测试版默认生成 `plots/0001_B02-001.png` 等逐录音对照图，并按音频 ID 的自然顺序
汇总到 **`comparison.pdf`**，每条音频一页；`plots/index.csv` 保存页码、图片与原音频路径。
每页有四块内容：波形、实际 L1 分数、原始事件对比、事件簇辅助对比。
下面两张图均为上行标签、下行预测，共用同一秒数坐标；未匹配标签为红色，未匹配预测带橙边。

事件簇默认将两侧间隔不超过 0.3 秒的区间分别合并，可用 `--plot-merge-gap 0.3` 调整，
仅影响辅助图，不修改标签、预测和正式指标。原始事件图与指标采用相同的区间真值
（重叠或相接标签取并集）；分数是实际 dB 值，不是概率。
未审核区域用灰底表示；没有精确区间标签时会明确提示，不把整段粗标签画成异常区间。
可用 `--no-plots` 只输出数值结果；中文字体默认使用系统字体，缺少时用 `--plot-font /path/to/font.ttf` 指定。

三批开发数据的新旧对比（完整命令与说明见 [L1 v3 设计与结果](docs/L1_V3.md)）：

| 批次 | 版本 | 事件召回 | 事件精确率 | 事件 F1 | 时长准确率 | 时长精确率 | 覆盖率 | 报警拖尾 s | 最长窗口 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B02 | v2 | 0.667 | 0.606 | 0.635 | 0.977 | 0.309 | 0.530 | 22.9 | 21.4 s |
| B02 | **v3** | **0.733** | 0.333 | 0.458 | **0.982** | **0.364** | **0.624** | **12.9** | **6.2 s** |
| B03 | v2 | 0.863 | 0.942 | 0.901 | 0.857 | 0.552 | 0.596 | 282.0 | 17.4 s |
| B03 | **v3** | **0.951** | **0.946** | **0.948** | **0.868** | **0.670** | 0.354 | **96.7** | **2.5 s** |
| B04 | v2 | 0.531 | 0.938 | 0.678 | 0.921 | 0.702 | 0.367 | 48.8 | 20.1 s |
| B04 | **v3** | **0.675** | 0.860 | **0.757** | 0.921 | **0.737** | 0.333 | **40.7** | **4.3 s** |

事件召回三批均上升；最长窗口从 20 s 级降到 2–6 s，“工况改变后一直报警到文件末尾”的问题基本消失；
报警拖尾大幅下降；B03/B04 的事件 F1 与时长精确率上升；B02 的时长口径全部优于旧版。
**B02 的事件精确率仍低于旧版**（0.606→0.333）：新版会多报约 45 个短事件（2–16 kHz 对比 +2~4 dB 的弱瞬态，
以及 20–300 Hz 的低频起伏），旧版靠“整条录音最强声”过滤压掉了它们，新业务定义不再使用这种过滤。
B03 覆盖率下降属于标注口径冲突：标注比实测声学核心前后各宽约 0.2 s（详见文档第 5 节）。
三批均用于开发与对照，**不代表独立测试集上的泛化性能**。

[检验器实现步骤、标签格式与指标定义](docs/L1_EVALUATION.md) · [完整标注规范](docs/ANNOTATION.md)

检验器也支持简易 CSV/JSON（`file,label` 或 `file,start_s,end_s,label`）和现有清单：

```bash
python -m wind_audio l1-test \
  --audio-dir /path/to/audio --labels /path/to/labels.csv \
  --config config/l1_v3.json --out reports/l1_check

python -m wind_audio l1-test \
  --manifest /path/to/manifest.csv --annotations /path/to/annotations.csv \
  --config config/l1_v3.json --out reports/l1_check_manifest
```

## L1 检测与审核

```bash
python -m wind_audio l1-detect --input /path/to/audio \
  --config config/l1_v3.json --out runs/l1_latest
```

`--input` 支持单条 WAV 或目录；目录输入生成总览和逐文件报告。
`events.json/csv` 保存检测窗口；`windows/` 是窗口音频，`clips/` 额外添加试听上下文；
`report.html` 供试听，`review.csv` 供人工填写。
`trigger_start_s/end_s` 是变化种子，L2 和标签比较使用完整窗口的 `start_s/end_s`。
`right_censored=true` 表示录音结束时尚未确认事件结束。

L1 默认 44.1 kHz、2048 点 FFT、64 Mel、20 ms 输出步长，在**多个时间尺度**上判断稳定性：
0.1 s 快速变化、1 s 慢漂移、2 s 局部噪声尺度、0.6 s 稳定观察窗。
每个频带用自身波动与局部噪声比较，不再使用固定 dB 门槛，也不与整条录音的最强声比较。

| 参数 | 默认 | 作用 |
|---|---:|---|
| `--trigger-change-z` | 3.0 | 触发所需变化强度（局部噪声 σ，不是 dB） |
| `--trigger-exceed` | 0.15 | 触发所需的超容差频带比例 |
| `--band-z` / `--sigma-floor-db` / `--sigma-cap-db` | 3 / 0.35 / 6 dB | 单频带门槛与噪声尺度上下限 |
| `--noise-s` / `--stable-s` / `--fast-s` | 2 / 0.6 / 0.1 秒 | 噪声、稳定、快变化的观察尺度 |
| `--confirm-s` / `--hold-s` | 0.8 / 0.2 秒 | 结束事件所需的稳定确认与“无触发”保持时间 |
| `--slow-min-duration` | 0.5 秒 | 仅由慢变化触发的事件最短时长，防止块边界伪事件 |
| `--min-dur` / `--padding` | 0.06 / 0 秒 | 最短事件 / 边界补充（默认不扩张） |

完整参数与调节方向见 `python -m wind_audio l1-detect --help`、`config/l1_v3.json`
和 [L1 v3 文档](docs/L1_V3.md)。`change_z` 的单位是局部噪声 σ，**不是 dB、不是概率**；
`exceed` 与 `stable_frac` 是 0–1 的比例。报告与图表使用相同阈值，不再标注为 dB。
旧版配置 `config/l1.json` 仍可用，旧参数（`--threshold`、`--release-db` 等）只能配旧配置，
与新配置混用会直接报错而不是静默解释。
全文件离线分析不是流式服务；20 ms 网格不等于真实边界精度；削顶录音只做质量诊断，不修改波形。

人工填写 `l1_label`、修订边界及 `positive_types/negative_types` 后导入：

```bash
python -m wind_audio review-import \
  --review runs/l1_latest/review.csv --out data/reviews/reviews.csv
```

预测不能直接当人工真值；未审核部分不默认正常。还需抽查无告警录音以发现漏检。

## 交付标签规则：每条录音一个标签

检测与正常区域测试都会为每条录音产出一条交付标签（`labels.csv`）：

- 标签内容 = 整条录音 **减去模型判定的正常段**，剩下的异常区间；
- 若异常区间 **超过 5 段**，直接返回整条录音（0–时长）；
- 0 段表示整条正常（`rule=none`），≤5 段按实际输出（`rule=as_detected`）。

实现见 `wind_audio/labeling.py`，阈值 `MAX_INTERVALS = 5` 可改。输出位置：

| 命令 | 文件 | 字段 |
|---|---|---|
| `l1-detect` | `<out>/labels.csv`、每条 `<out>/<文件>/label.json` | `audio_id, n_intervals, rule, intervals_s` |
| `l1-test-normal` | `<out>/labels.csv` | 同上（模型侧标签，不影响指标） |

B05 上 96 条的分布：36 条整条正常、51 条按实际段输出、9 条因超过 5 段整条返回。

## 平稳区域拟合（B05 口径）

如果标注标的是**平稳区间**而不是异常事件，用 `--labels-are-stable` 评估，
用 `l1-fit` 在该批数据上重新标定阈值并选择分段方式：

```bash
python -m wind_audio l1-test \
  --audio-dir /Users/shuoqinlin/Desktop/数据/新一批/B05/音频 \
  --labels    /Users/shuoqinlin/Desktop/数据/新一批/B05/标注 \
  --pairing sorted --labels-are-stable \
  --config config/l1_v3.json --no-plots --out runs/l1_v3_stable_base

python -m wind_audio l1-fit \
  --audio-dir /Users/shuoqinlin/Desktop/数据/新一批/B05/音频 \
  --labels    /Users/shuoqinlin/Desktop/数据/新一批/B05/标注 \
  --out config/l1_fit_stable.json --report runs/l1_fit/report.json
```

`l1-fit` 在平稳帧上估计每个判据的分布，把阈值放到其尾部（`--alpha`、`--alpha-stable` 控制目标误判率），
再按时间顺序留出尾部做验证，最后选择 `segment_mode`/`confirm_s` 等结构项。
B05 结果：验证集时长 F1 0.419 → **0.688**、覆盖率 0.158 → **0.473**。
详见 [平稳区域拟合](docs/L1_FIT.md)。

## L2 训练与使用

当前仓库保留最新版 L2 实现和类型配置，没有已训练的 L2 模型或 CNN14 权重。
请准备可信来源的官方 `Cnn14_mAP=0.431.pth`；程序不会自动下载。
[PANNs 上游](https://github.com/qiuqiangkong/audioset_tagging_cnn)。

L2 在 32 kHz 下提取 2048 维特征，默认 4 秒上下文、2 秒滑动步长。
CNN14 冻结，仅训练正常/异常与多标签分类头。掩码 BCE 使用已确认标签，
旧样本回放与可选蒸馏支持增量训练；原型相似度用于拒识和案例检索。
每个输出在 train/val 的正负例中至少各覆盖两个来源才启用决策。

```bash
python -m wind_audio split --manifest data/manifest.csv \
  --annotations data/annotations.csv --group-by source --out data/splits.json
python -m wind_audio l2-prepare --reviews data/reviews/reviews.csv \
  --manifest data/manifest.csv --splits data/splits.json \
  --taxonomy config/taxonomy.json --out data/l2/v1.json
python -m wind_audio l2-check --dataset data/l2/v1.json
python -m wind_audio l2-embed --dataset data/l2/v1.json \
  --checkpoint weights/Cnn14_mAP=0.431.pth --out cache/l2_v1.npz --device cpu
python -m wind_audio l2-train --dataset data/l2/v1.json --features cache/l2_v1.npz \
  --version l2_v1 --out models/l2_v1
python -m wind_audio l2-evaluate --model models/l2_v1 \
  --dataset data/l2/v1.json --features cache/l2_v1.npz --split test --out reports/l2_latest.json
```

续训时 `l2-prepare --previous data/l2/v1.json` 生成累计 v2 快照，重新提取特征，
`l2-train --resume models/l2_v1 --version l2_v2 --out models/l2_v2`。
新类型只在 taxonomy 末尾追加；保留历史人工审核、原始音频、固定来源划分与回放样本。
不能把旧测试来源移入训练。这里的文件来源 ID 不等于独立设备，划分前应核实采集来源。

```bash
python -m wind_audio pipeline --input /path/to/audio.wav \
  --l1-config config/l1_v3.json --l2-model models/l2_v2 \
  --checkpoint weights/Cnn14_mAP=0.431.pth --device cpu --out runs/two_stage
```

## 工程布局与验证

```text
wind_audio/
  stability.py / state.py      v3 多尺度稳定性指标与事件状态机
  l1.py / windowing.py         旧版 v2 评分与窗口逻辑（保留复现）
  detectors.py                 新旧检测器与配置分发
  evaluate.py / markers.py     准确率、事件指标、B02 标注导入
  eval_plots.py               逐录音四联对照图与有序 PDF
  data.py / splits.py          数据校验与来源划分
  audio.py / events.py         WAV 处理、事件特征
  cli.py                      统一命令入口
  review.py / report.py / render.py
  l2/                         PANNs、分类头、回放训练
config/                       当前 L1 参数与 L2 类型表
docs/                         标注规范、检验步骤、清理记录
tests/                        自动化测试；数据在临时目录生成并自动清除
runs/normal_mixed_v1/          当前正常模型（多批次合并 + 打散划分）
```

旧版代码、演示模型、旧测试音频和结果已移除，清单及恢复位置见 [清理记录](docs/CLEANUP.md)。
业务数据保留在外部 B02 目录；运行产物和本地权重默认忽略，不修改 Git 历史。

```bash
python -m unittest discover -s tests -t . -v
# 可选：需要本地 PyTorch，不下载权重
WIND_AUDIO_TEST_PANNS=1 python -m unittest tests.test_l2.PannsStructureTests -v
```
