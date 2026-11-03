# 风机音频异常检测（L1 原型）

风力发电机轮毂气传导音频，输入是一段或多段 WAV。L1 检测异常并定位到秒级区间；L2 用人工审核结果学习“故障 / 正常工况变化”，当前不训练 L2。

> 当前真实数据只有 1 条 30 s 异常录音，且只有**文件级粗标**（0–30 s anomaly，无秒级区间）。因此本仓库**没有真实召回率/误报率**；真实数据现在只能做读取和演示推理。合成数据上的指标只用于验证程序逻辑，见“验证结果”。

## 目录

```text
wind_audio/
├── wind_audio/
│   ├── audio.py          # WAV 读取、真实时长、居中 STFT、log-mel、RMS
│   ├── data.py           # manifest / 区间标注 / 校验 / 标签掩码 / 旧标签迁移
│   ├── features.py       # 上下文特征、逐文件增益归一化、特征缓存
│   ├── events.py         # 双阈值滞回、合并、最短时长、事件特征
│   ├── l1.py             # 旧 NormalModel（保留，作为 baseline 模型）
│   ├── detectors.py      # baseline + supervised 统一接口、保存/加载
│   ├── backends.py       # XGBoost / sklearn / numpy 逻辑回归后端
│   ├── splits.py         # 按录音/批次/设备/时间分组划分
│   ├── calibrate.py      # 验证集阈值 + 后处理校准
│   ├── evaluate.py       # 一对一事件匹配、召回/误报/覆盖/边界误差
│   ├── render.py         # 波形 / 时频图 / 分数图（真实时间轴）
│   ├── report.py         # HTML 报告（含试听、标注叠加、审核链接）
│   ├── review.py         # 人工审核导出/导入、L1/L2 反馈语义
│   ├── synthetic.py      # 合成数据（仅逻辑验证）
│   └── cli.py            # 全部命令
├── data/
│   ├── anomaly/          # 现有真实异常录音
│   ├── normal/           # 正常录音（目前为空）
│   ├── labels.csv        # 旧标签，保持原样
│   ├── manifest.csv      # 新：文件级元数据
│   ├── annotations.csv   # 新：区间标注（当前只有 file 级粗标）
│   ├── synthetic/        # 合成数据
│   └── reviews/          # 人工审核结果
├── models/
│   ├── l1_demo/          # 旧演示模型（仅用于兼容/演示）
│   ├── syn_baseline/     # 合成数据 baseline
│   └── syn_supervised/   # 合成数据 supervised（sklearn 后端）
├── runs/                 # 每次推理的报告/事件/片段/审核模板
├── reports/              # 评估指标与简报
├── tests/
├── docs/
└── requirements.txt
```

## 安装

需要 Python ≥ 3.10，核心依赖只有 numpy。有监督模型需要二选一：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` 里：

```text
numpy>=1.24
scikit-learn>=1.3      # 备用 CPU 梯度提升（本机已验证）
xgboost>=2.0           # 首选，CPU
```

本机默认 `python3` 没有 sklearn/xgboost，且当前无法联网安装。已存在的 `~/miniconda3/envs/rag_learn/bin/python` 带 sklearn 1.9.1，可用来跑有监督流程：

```bash
~/miniconda3/envs/rag_learn/bin/python -m wind_audio check ...
```

XGBoost 未安装，**本机没有执行过 XGBoost 路径**。`--algo xgboost` 缺依赖会报错并给出安装命令；默认 `--algo auto` 会回退到 sklearn/numpy，并把实际算法写进 `meta.json` 和报告，不会静默替换。禁止回退用 `--no-fallback`。

## 快速开始

### 0. 只有音频 + 标注？

不需要手工写 manifest。把 WAV 放 `data/audio/`，标注写成：

```csv
file,start_s,end_s,label,group
WT01_1200.wav,12.40,13.10,anomaly,WT01_20261103
WT01_1200.wav,0,12.40,normal,WT01_20261103
WT01_1200.wav,13.10,30.00,normal,WT01_20261103
```

```bash
python3 -m wind_audio import-simple \
  --audio-dir data/audio --labels data/labels.csv \
  --manifest data/manifest.csv --annotations data/annotations.csv
```

采样率/声道/时长从 WAV 头自动读取；`group` 可选，同一条连续录音的切片填同一个值。只标异常、又确认整段都审过时，加 `--reviewed-list reviewed.txt`（逐文件）或 `--assume-unlabeled-normal`（全局）；否则未标注区域保持 unknown。完整说明见 [docs/WORKFLOW.md](docs/WORKFLOW.md)。

### 0. 看一条音频

```bash
python3 -m wind_audio probe --input data/anomaly/A01_01_20261030_120227.wav
```

### 1. 迁移旧标签（只做一次）

```bash
python3 -m wind_audio migrate-labels \
  --labels data/labels.csv \
  --audio-dir data/anomaly \
  --manifest data/manifest.csv \
  --annotations data/annotations.csv
```

旧 `labels.csv` 不改。0–30 s 的 anomaly 记录会变成**file 级粗标**，`fully_reviewed=false`，不会展开成逐帧真值。详见 [docs/MIGRATION.md](docs/MIGRATION.md)。

### 2. 数据检查

```bash
python3 -m wind_audio check \
  --manifest data/manifest.csv \
  --annotations data/annotations.csv
```

输出可用正常/异常/未审核时长、异常事件数、粗标文件数，以及不能训练/评估的具体原因。当前真实数据的输出是：

```text
已审核正常 0.0s · 已审核异常 0.0s · 未审核 30.0s
异常事件 0 个 · 完全审核文件 0 · 含粗标文件 1
```

这正是正确行为：粗标不能当区间真值。

### 3. 划分（需要 ≥3 个独立录音/批次）

```bash
python3 -m wind_audio split \
  --manifest data/manifest.csv \
  --annotations data/annotations.csv \
  --out data/splits.json \
  --group-by source --seed 42
```

`--group-by` 可选 `source`（默认，录音/批次）、`device`（跨机组评估）、`time`、`audio`。同一分组不会跨 train/val/test。

### 4. 训练

有监督（推荐先确认后端）：

```bash
python3 -m wind_audio train \
  --manifest data/manifest.csv \
  --annotations data/annotations.csv \
  --splits data/splits.json \
  --detector supervised \
  --algo auto \
  --out models/l1_supervised \
  --version l1_v2 \
  --target-recall 0.95
```

- `--algo auto|xgboost|sklearn|numpy`；`auto` 依次尝试 xgboost → sklearn → numpy，并记录实际使用。
- `--no-fallback`：缺 XGBoost 时直接失败。
- `--max-samples-per-file` / `--max-total-samples`：限制单文件/总样本，避免长录音支配训练。
- 没有足够已审核正常/异常样本时，有监督训练会停止并说明缺口。

基线（对照）：

```bash
python3 -m wind_audio train \
  --manifest data/manifest.csv \
  --annotations data/annotations.csv \
  --splits data/splits.json \
  --detector baseline \
  --out models/l1_baseline
```

旧入口仍可用（无 manifest，仅正常 wav，或演示拟合）：

```bash
python3 -m wind_audio train --normal-dir data/normal --out models/l1_baseline_legacy
python3 -m wind_audio train --demo-from data/anomaly/A01_01_20261030_120227.wav --out models/l1_demo2 --threshold 8
```

`--demo-from` 只用待检测录音的安静段拟合，**不是正常基线**，不能用于评估。

### 5. 校准与评估

`train` 默认在 val 上校准阈值和后处理，并写 `calibration.json`。也可以单独重跑：

```bash
python3 -m wind_audio calibrate \
  --model models/l1_supervised \
  --manifest data/manifest.csv \
  --annotations data/annotations.csv \
  --splits data/splits.json \
  --split val --target-recall 0.95
```

测试集评估（阈值冻结后）：

```bash
python3 -m wind_audio evaluate \
  --model models/l1_supervised \
  --manifest data/manifest.csv \
  --annotations data/annotations.csv \
  --splits data/splits.json \
  --split test \
  --out reports/l1_supervised_test
```

输出 `metrics.json` + `brief.md`。指标包括事件召回、事件精确率、每小时误报次数、异常覆盖率、报警时长占比、起止边界误差。未审核区域不计入任何分母；文件级粗标不参与评估。

### 6. 单条 / 批量推理

```bash
python3 -m wind_audio run \
  --input data/anomaly/A01_01_20261030_120227.wav \
  --model models/l1_supervised \
  --manifest data/manifest.csv \
  --annotations data/annotations.csv \
  --review data/reviews/reviews.csv \
  --out runs/real_case_01
```

`--review` 可选；传了就把已导入的人工 L1/L2 结论显示在报告事件表里。批量：

```bash
python3 -m wind_audio run --input data/anomaly --model models/l1_supervised --out runs/batch_01
```

每次输出：

```text
runs/<name>/
  report.html       # 波形 + 时频图 + 分数 + 标注 + 试听 + 事件表
  waveform.png
  spectrogram.png   # 低频在下
  score.png
  events.json       # 含音频 ID、真实时长、模型版本、阈值来源、事件特征、clip 时间对应
  events.csv
  audio/            # 原始音频副本，供 HTML 播放
  review.csv/json   # 审核模板
  clips/            # 每个事件 ±3 s 片段
```

`--threshold` 显式覆盖模型阈值时会写入 `events.json` 的 `threshold_source: "cli"`，报告显示实际生效值；未传则用模型/校准阈值。推理不接受与模型不一致的 `--n-mels/--n-fft/--sr/...`，会直接报错。

### 7. 人工审核

```bash
python3 -m wind_audio review-export \
  --events runs/real_case_01/events.json \
  --out runs/real_case_01/review.csv
```

编辑 `review.csv` 后导入：

```bash
python3 -m wind_audio review-import \
  --review runs/real_case_01/review.csv \
  --out data/reviews/reviews.csv \
  --json data/reviews/reviews.json \
  --emit-annotations data/reviews/annotations_from_review.csv
```

审核字段：`event_id, audio_id, source_audio_path, predicted_start_s, predicted_end_s, corrected_start_s, corrected_end_s, l1_label, l2_label, fault_type, note, reviewed_at, source_model_version, clip_path`。

语义：

- `l1_label=anomaly` + `l2_label=nonfault` **仍然是 L1 正例**；
- 只有 `l1_label=normal` 才是确认的正常误报，可用于改进 L1；
- `uncertain`/空白不进入训练；
- 当前没有 L2 模型，`l2_label` 默认 `not_trained`，报告统一显示“待审核/未训练”。

## 数据与标注

### manifest.csv

```text
audio_id,path,source_id,device_id,recorded_at,duration_s,sample_rate,channels,fully_reviewed,notes
```

- `audio_id` 唯一；`path` 相对 manifest 文件。
- `source_id` 是原始录音/采集批次，用于默认分组；`device_id`/`recorded_at` 只用于划分和追溯。**三者都不作为模型特征**；模型输入只有音频。`duration_s`/`sample_rate` 只用于预处理和报告，也不是特征。
- `fully_reviewed=true` 仅当整段已完整审核且所有异常都已标出；此时区间外才可推导为 normal。

### annotations.csv

```text
annotation_id,audio_id,start_s,end_s,label,precision,fault_type,note,reviewed_at,annotator,version
```

- `label`：`normal` / `anomaly` / `uncertain`。
- `precision`：`file`（只能整段粗标，不展开）或 `interval`（精确区间）。
- 规则：
  1. 未审核、未标注、`uncertain` 一律 unknown，不能当正常。
  2. 只有 `fully_reviewed=true` 且异常是 interval 时，区间外才推导 normal。
  3. file 级粗标永远不逐帧展开；`fully_reviewed=true` + 只有 file 级 anomaly 会被校验拒绝。
  4. 越界、起止倒置、区间冲突、未知 `audio_id`、缺失音频都会报错。
  5. 目标网格默认 0.1 s；标签取目标时间点所在区间。上下文只用于特征，不会因为窗里沾到异常就把整窗标异常。

详见 [docs/ANNOTATION.md](docs/ANNOTATION.md)。

## L1 方法

### baseline

保留原 `NormalModel`：每个 log-mel 频带学 median/MAD，帧分数为 `0.5×平均|z| + 0.5×top10%|z|`，再做中值平滑。只用确认 normal 帧拟合；旧 `models/l1_demo` 可直接加载。`score_type="baseline_z"`，是标准化偏差，**不是概率**。

### supervised

- 特征：64 维 log-mel（20–16 kHz）+ 10 个帧级描述量（增益归一化 RMS、质心、平坦度、频谱通量、能量变化率、5 个频带占比）。
- 每个目标时刻（默认 0.1 s）：当前帧特征 + 上下文（默认 ±1 s）的 mean/std/min/max + log-mel 上下文 mean/std + 0.25/0.5/1/2 s 左右差异，共 282 维，名称/顺序写入模型。
- 增益鲁棒：逐文件减中位 RMS 和中位频带电平，保留录音内部变化，降低整条录音固定增益差异。不是逐窗自适应，避免持续异常被吸收。
- 后端：XGBoost `hist` CPU 优先；缺依赖时 sklearn `HistGradientBoostingClassifier`；再缺时 numpy 逻辑回归。实际算法、请求算法、是否回退全部写入 `meta.json`。
- `score_type="probability_uncalibrated"`；阈值在 val 上按目标召回选取，**不把输出叫可靠置信度**。

### 事件后处理

- 双阈值滞回：高于 `threshold` 进入，低于 `threshold × low_ratio` 退出。
- `merge_gap` 内的两段合并；短于 `min_duration` 的事件丢弃。
- 事件起止按真实时间轴 `times ± target_step/2` 计算并夹到 `[0, duration]`；贴首尾标 `at_edge`。
- 这些参数会写进模型，并可在校准中搜索。短事件可能被 `min_duration` 丢掉，近邻事件可能被 `merge_gap` 合并，这是明确的取舍。

## 划分、校准、评估口径

- **分组划分**：默认按 `source_id`，绝不把同一录音的切片/相邻窗口分到不同集合；跨机组用 `--group-by device`。分组不足 3 个直接报错，不拿同一条录音切碎冒充测试集。
- **校准**：只在 val 上搜索阈值 + `min_duration/merge_gap/low_ratio`；目标召回优先，再最小化 FA/h。达不到目标会写 `target_reached=false`，不伪造。
- **匹配**：GT 是 interval 级 anomaly（相接的合并）。预测与 GT 按 IoU 贪心一对一匹配；一个覆盖全文件的报警最多命中一个 GT，重复报警不会都算对。
- **未审核**：只与 unknown 重叠的预测被排除；normal 时长分母只算已审核 normal。
- **指标**：事件召回、事件精确率、FA/h、FP 报警时长/小时、覆盖率、报警时长占比、起止边界中位误差；零样本情形输出 `null`，不产生 NaN 或假满分。

## 模型产物与服务器加载

```text
models/l1_supervised/
  meta.json          # 版本、detector、score_type、阈值、后处理、特征配置、
                     # 特征名/顺序、后端、训练清单、split、seed、依赖版本、校准
  model/             # xgboost model.json 或 sklearn model.pkl 或 numpy model.npz
  calibration.json
```

加载：

```python
from wind_audio.detectors import load_detector
from wind_audio.features import extract_features
from wind_audio.audio import load_wav

det = load_detector("models/l1_supervised")
x, sr = load_wav("case.wav", target_sr=det.feature_config.sr)
bundle = extract_features(x, sr, det.feature_config, "case")
events = det.predict_events(bundle)
```

加载时会校验 `feature_config`、特征名/顺序；不一致直接报错。服务器需要同一份代码和相同的 numpy/sklearn/xgboost 版本；详见 [docs/MODEL.md](docs/MODEL.md)。

## 验证结果

### 自动化测试

```bash
python3 -m unittest discover -s tests -t . -v
```

- 默认 `python3`（numpy only）：**30 个测试，29 通过，1 跳过**（sklearn 有监督集成测试跳过）。
- `~/miniconda3/envs/rag_learn/bin/python`（sklearn 1.9.1）：**30 个测试全部通过**。

覆盖：标签掩码与粗标、未审核屏蔽、划分无泄漏、一对一事件匹配、阈值覆盖、保存/加载一致性、特征配置不匹配、时间边界、异常但非故障反馈、合成全链路（check/split/train/calibrate/evaluate/run/review）。

### 合成数据（仅逻辑验证，不代表真实风机效果）

12 条 12 s 合成录音：正常背景、正常背景切换、短冲击、持续异常、整条增益变化、首尾事件。按 source 分 7/2/3 文件。

| 模型 | 测试集召回 | 精确率 | FA/h | 覆盖率 |
|---|---|---|---|---|
| baseline（校准后） | 1.000 | 1.000 | 0.000 | 0.952 |
| supervised（sklearn，校准后） | 1.000 | 1.000 | 0.000 | 0.952 |

这些数字只证明标签、划分、训练、校准、匹配和报告链路能跑通；合成异常是人为加的高频爆发，比真实风机故障容易得多。

### 真实音频

- `probe` / `run` 已用 `data/anomaly/A01_01_20261030_120227.wav` 验证：真实时长 `30.002721 s`（不再用帧数×帧移）。
- 旧 `models/l1_demo` 可加载并出报告（9 个事件）；**这是演示模型，不是真实正常基线**。
- 真实数据没有正常样本、没有秒级异常区间，因此**没有真实召回率/误报率**可报。

## 已知限制

1. 真实数据不足：只有 1 条异常录音 + file 级粗标；无法训练/评估真实模型。
2. 无 SCADA/转速/风速：工况变化只能从声音推断；变转速会让频谱搬移，正式模型需要更多工况数据。
3. 逐文件中位归一化在异常占据整条录音 >50% 时可能吸收持续异常；当前设计假设异常是局部/少数。
4. 单次 `run` 对整条文件做 STFT；30 s 很轻，几分钟也可用，但数小时连续录音建议先按 30–60 s（保留 2×context 重叠）切段。
5. XGBoost 本机未安装、未执行；sklearn/numpy 路径已验证。XGBoost 代码路径存在但未实跑。
6. 事件后处理可能漏短事件、合并近邻事件；参数需在 val 上校准。
7. 50 Hz 工频及谐波很强（真实录音），当前未陷波；是否陷波要先用多文件确认稳定性。
8. 报告中的分数不是故障概率；L2 未训练，所有事件统一“待审核/未训练”。

## 文档

- [docs/WORKFLOW.md](docs/WORKFLOW.md)：数据标注 → 导入 → 训练 → 推理完整 SOP
- [docs/ANNOTATION.md](docs/ANNOTATION.md)：标注规范、CSV/JSON、标签掩码、目标窗
- [docs/MIGRATION.md](docs/MIGRATION.md)：旧 `labels.csv` 迁移与为什么不补造区间
- [docs/MODEL.md](docs/MODEL.md)：模型产物字段、导出、服务器加载、版本兼容
- `data/synthetic/`：可复现的合成数据与划分
- [reports/VALIDATION.md](reports/VALIDATION.md)：本轮实际执行的验证记录与未执行项
- `reports/`：已有合成评估指标
