# 数据标注 → 导入 → 训练 → 推理：完整流程

本文是 `~/Desktop/wind_audio` 的可执行 SOP。所有命令都在项目根目录执行：

```bash
cd ~/Desktop/wind_audio
```

---

## 0. 最简模式：只有音频 + 标注

如果你没有机组编号、采集时间、批次等元数据，不需要手工写 manifest。只要：

```text
data/
├── audio/
│   ├── WT01_1200.wav
│   └── WT01_1210.wav
└── labels.csv
```

`labels.csv` 最少四列：

```csv
file,start_s,end_s,label,group
WT01_1200.wav,12.40,13.10,anomaly,WT01_20261103
WT01_1200.wav,0,12.40,normal,WT01_20261103
WT01_1200.wav,13.10,30.00,normal,WT01_20261103
WT01_1210.wav,5.00,5.30,anomaly,WT01_20261103
```

- `file`：文件名或相对 `audio/` 的路径；重名文件必须写相对路径。
- `start_s/end_s`：秒；必须是真实区间，不能只写“整段异常”。
- `label`：`normal` / `anomaly` / `uncertain`。
- `group`：可选；同一条连续录音切出的多个文件填同一个 group，划分时不会泄漏。不填则每个文件自己一组。
- 采样率、声道、时长、位深由程序读 WAV 头自动补全，**不需要你提供**。
- 机组/日期/批次不会进入模型特征；只用于划分和追溯。

导入：

```bash
python3 -m wind_audio import-simple \
  --audio-dir data/audio \
  --labels data/labels.csv \
  --manifest data/manifest.csv \
  --annotations data/annotations.csv
```

三种“未标注区域”处理方式，必须显式选一种：

1. **显式写 normal 区间**（推荐）：像上面的例子，normal/anomaly 覆盖整段；不需要额外开关。
2. **只标异常 + 逐文件声明整段已审核**：写一个 `reviewed.txt`，每行一个已完整审核的文件名：

   ```text
   WT01_1200.wav
   WT01_1210.wav
   ```

   ```bash
   python3 -m wind_audio import-simple \
     --audio-dir data/audio --labels data/labels.csv \
     --reviewed-list reviewed.txt \
     --manifest data/manifest.csv --annotations data/annotations.csv
   ```

   这些文件里，异常区间外推导为 normal；不在 `reviewed.txt` 的文件，未标注区域仍是 unknown。

3. **全部文件都完整审核**：`--assume-unlabeled-normal`。这是全局假设，只在你确认“每个文件都听完了、所有异常都标了”时用。

如果只标了异常、又没给 `reviewed.txt` / `--assume-unlabeled-normal`，异常区间外保持 unknown，不会被偷当成正常。

两个容易踩的点：

- **正常文件也要有 normal 时长**：如果某个文件完全没有异常，建议写一行 `normal,0,duration`，或把它放进 `reviewed.txt`；否则它不会贡献正常时长。
- **同一段连续录音的多个切片要共用 group**：在 `labels.csv` 里给每个文件至少写一行带 `group` 的标注（正常文件写 `normal,0,duration`）。不写 group 时每个文件各自成组，划分时可能把同一段录音的切片分到 train/test 两边。

导入后照常：

```bash
python3 -m wind_audio check --manifest data/manifest.csv --annotations data/annotations.csv
python3 -m wind_audio split --manifest data/manifest.csv --annotations data/annotations.csv \
  --out data/splits.json --group-by source --seed 42
```

之后训练/推理命令与下面完全相同。

## 0.1 环境和数据现状

```bash
python3 -m wind_audio probe --input data/anomaly/A01_01_20261030_120227.wav
```

输出里的 `duration_s` 是真实采样点数算出的时长（示例 `30.002721 s`），不是“帧数 × 帧移”。

当前仓库真实数据只有：

- 1 条 30 s 异常录音；
- 只有 file 级粗标（0–30 s anomaly，无秒级区间）；
- 没有正常录音。

所以现在**只能跑演示推理，不能训练真实模型**。下面流程从“准备数据”开始，按顺序做即可。

---

## 1. 准备目录和两份 CSV

推荐目录：

```text
data/
├── audio/                 # 所有 WAV
├── manifest.csv           # 文件级元数据
├── annotations.csv        # 区间级标注
├── splits.json            # 分组划分
└── reviews/               # 人工审核结果
```

WAV 要求：

- PCM WAV；默认按 44.1 kHz 分析，其他采样率会重采样并记录；
- 单声道/多声道都可（多声道会混成单声道）；
- 单条建议 30 s；其他长度也支持；很长的连续录音先按 30–60 s 切段，保留 2×context（默认 2 s）重叠。

### 1.1 `manifest.csv`

```csv
audio_id,path,source_id,device_id,recorded_at,duration_s,sample_rate,channels,fully_reviewed,notes
WT01_20261103_1200,audio/WT01_20261103_1200.wav,WT01_20261103,WT01,2026-11-03T12:00:00,30.002721,44100,1,true,
WT01_20261103_1210,audio/WT01_20261103_1210.wav,WT01_20261103,WT01,2026-11-03T12:10:00,30.002721,44100,1,true,
```

字段含义：

| 字段 | 说明 |
|---|---|
| `audio_id` | 唯一 ID，事件/审核/报告都用它关联 |
| `path` | WAV 路径，**相对 manifest 文件所在目录**；也可绝对路径 |
| `source_id` | 原始录音/采集批次；默认按它分组划分 |
| `device_id` | 风机/机组编号；只用于划分和追溯，**不作模型特征** |
| `recorded_at` | 采集时间；只用于划分和追溯 |
| `duration_s` | 可留空；`check` 会用 WAV 头补齐 |
| `sample_rate`/`channels` | 可留空；`check` 会补齐 |
| `fully_reviewed` | **只有整段都审核完、所有异常都标出**才写 `true` |
| `notes` | 备注 |

**关键**：如果一条连续录音被切成多个 30 s 文件，它们必须共用同一个 `source_id`，否则会被分到 train/test 不同集合，造成泄漏。

### 1.2 `annotations.csv`

```csv
annotation_id,audio_id,start_s,end_s,label,precision,fault_type,note,reviewed_at,annotator,version
ann_0001,WT01_20261103_1200,12.40,13.10,anomaly,interval,,轴承异响候选,2026-11-03,lin,1
ann_0002,WT01_20261103_1200,25.00,25.30,uncertain,interval,,听不清,2026-11-03,lin,1
```

字段：

| 字段 | 取值 |
|---|---|
| `label` | `normal` / `anomaly` / `uncertain` |
| `precision` | `interval`（秒级区间）或 `file`（整段粗标） |
| `start_s` / `end_s` | 秒，`end_s > start_s` |
| `fault_type` | 可选；没有 L2 结论就留空，不要猜 |
| `reviewed_at`/`annotator`/`version` | 可选，便于追溯 |

规则：

1. 未审核、未标注、`uncertain` 都是 **unknown**，不能当正常。
2. 只有 `fully_reviewed=true` 且异常是 `interval` 时，异常区间外才自动推导为 normal。
3. `precision=file` 永远不展开成逐帧真值。
4. `fully_reviewed=true` + 只有 file 级 anomaly 会被 `check` 拒绝。
5. 同一文件不同 label 的区间重叠、越界、起止倒置都会报错。

---

## 2. 怎么标（人工 SOP）

### 2.1 正常文件

如果整段都正常：

- `manifest.csv` 里 `fully_reviewed=true`；
- `annotations.csv` 不写任何行；
- 程序会把整段推导为 normal。

### 2.2 有异常的文件

1. 用 Audacity、Logic、或报告里的音频播放器听一遍；
2. 找到异常起止秒（例如 `12.40–13.10`）；
3. 写一行 `label=anomaly, precision=interval`；
4. 如果整段其余部分都确认正常，`fully_reviewed=true`，不需要逐段写 normal；
5. 如果有听不准的地方，写 `label=uncertain, precision=interval`，程序会屏蔽这块，不参与训练/评估。

### 2.3 只审了一部分

- `fully_reviewed=false`；
- 只写你实际审过的区间：`normal` 或 `anomaly`；
- 没写的部分自动 unknown，不会被当正常，也不会被当异常。

### 2.4 只有整段粗标

例如现有 `data/labels.csv` 的 `0–30 s anomaly，无秒级标注`：

```csv
annotation_id,audio_id,start_s,end_s,label,precision,...
legacy_0001,A01_01_20261030_120227,0,30.002721,anomaly,file,...
```

这种记录只用于追溯，**不能训练/评估**。要变成可训练数据，必须人工补出具体异常的 `start_s/end_s`，并把 `precision` 改成 `interval`。

### 2.5 标注粒度

- 标的是**事件区间**，不是逐帧标签；程序按目标网格（默认 0.1 s）栅格化。
- 上下文只用于提特征，不会因为窗里沾到异常就把整窗标异常。
- 事件可以很短（0.1 s 级），也可以持续几十秒；后处理参数在验证集上校准，短事件可能被 `min_duration` 丢掉，这是明确取舍。

---

## 3. 导入与校验

把 CSV 放到 `data/` 后：

```bash
python3 -m wind_audio check \
  --manifest data/manifest.csv \
  --annotations data/annotations.csv
```

输出示例：

```text
文件 12 个 · 来源 12 个 · 设备 4 个
已审核正常 131.2s · 已审核异常 12.8s · 未审核 0.0s
异常事件 9 个 · 完整审核文件 12 · 含粗标文件 0
```

如果数据不足，会列出具体原因，例如：

```text
- 已审核正常音频 0.0s < 60s，不足以训练基线/有监督模型
- 至少需要 3 个独立录音/批次，当前不足以做 train/val/test 分组划分
- 已审核异常音频 0.0s < 10s，不足以训练有监督模型
- 独立异常事件 0 < 3，不足以做事件级评估
```

常见错误：

| 报错 | 原因 |
|---|---|
| `音频不存在` | `path` 不是相对 manifest 目录，或文件没放对 |
| `end_s 超出音频时长` | 标注超过真实时长 |
| `end_s 必须大于 start_s` | 起止写反/相等 |
| `区间冲突` | normal 和 anomaly 重叠 |
| `fully_reviewed=true 但只有 file 级 anomaly` | 必须补 interval，或改 false |
| `未知 audio_id` | annotations 里的 ID 不在 manifest |

## 3.1 旧 `labels.csv` 迁移

如果只有旧格式：

```bash
python3 -m wind_audio migrate-labels \
  --labels data/labels.csv \
  --audio-dir data/anomaly \
  --manifest data/manifest.csv \
  --annotations data/annotations.csv
```

旧文件不改；0–30 s 的粗标会变成 `precision=file`、`fully_reviewed=false`，30 s 全部 unknown。详见 [MIGRATION.md](MIGRATION.md)。

---

## 4. 划分 train / val / test

```bash
python3 -m wind_audio split \
  --manifest data/manifest.csv \
  --annotations data/annotations.csv \
  --out data/splits.json \
  --group-by source \
  --ratios 0.6,0.2,0.2 \
  --seed 42
```

输出：

```text
train: 7 文件, 6 个异常事件
val  : 2 文件, 1 个异常事件
test : 3 文件, 2 个异常事件
```

规则：

- 默认按 `source_id` 分组；同一录音/批次的切片不会跨集合。
- 跨机组评估用 `--group-by device`。
- 至少需要 **3 个独立分组**，否则直接报错，不拿一条录音切碎冒充测试集。
- 测试集要尽量同时有 normal 和 anomaly；只有一类时指标会是 `null`。

---

## 5. 训练

### 5.1 基线（只用确认 normal）

适合“先有正常数据、异常很少”的阶段：

```bash
python3 -m wind_audio train \
  --manifest data/manifest.csv \
  --annotations data/annotations.csv \
  --splits data/splits.json \
  --detector baseline \
  --out models/l1_baseline \
  --version l1_baseline \
  --threshold 5.0 \
  --target-recall 0.95
```

- 只用 `label=normal` 的帧拟合 `NormalModel`。
- 异常标注不参与训练，只用于 val 阈值校准和 test 评估。
- `score_type=baseline_z`，是标准化偏差，不是概率。
- 如果 val 里没有异常事件，阈值校准会失败（stderr 提示），模型会保留未校准状态；这时它只能当异常打分器，不能报召回/FA。

### 5.2 有监督（normal + anomaly）

```bash
~/miniconda3/envs/rag_learn/bin/python -m wind_audio train \
  --manifest data/manifest.csv \
  --annotations data/annotations.csv \
  --splits data/splits.json \
  --detector supervised \
  --algo sklearn \
  --out models/l1_supervised \
  --version l1_supervised \
  --target-recall 0.95
```

后端选择：

| `--algo` | 行为 |
|---|---|
| `auto` | 依次尝试 xgboost → sklearn → numpy，记录实际使用 |
| `xgboost` | 首选；缺依赖时默认回退并记录，`--no-fallback` 则直接报错 |
| `sklearn` | `HistGradientBoostingClassifier`，CPU |
| `numpy` | 逻辑回归兜底，不建议生产 |

本机默认 `python3` 没有 sklearn/xgboost；当前用 `rag_learn` 环境跑 sklearn。XGBoost 未安装，`--algo xgboost --no-fallback` 会明确给出 `pip install xgboost`。

其他参数：

```text
--max-samples-per-file 5000     # 单文件最多样本，防止长录音支配
--max-total-samples 50000       # 总样本上限
--seed 42
--no-calibrate                  # 暂不校准（不建议）
```

训练门槛（代码会检查）：

- 总样本 ≥ 50；
- normal ≥ 10，anomaly ≥ 10；
- 划分至少 3 个分组。

不满足会停止并说明缺口，不会用 demo 数据冒充训练。
如果 val 没有异常事件，校准会失败（模型保留未校准）；需要补充带区间标注的异常数据，或把有异常的录音分到 val。

### 5.3 训练产物

```text
models/l1_supervised/
  meta.json          # 特征配置、特征名顺序、阈值、后处理、后端、训练清单、split、seed、依赖
  model/             # xgboost model.json / sklearn model.pkl / numpy model.npz
  calibration.json   # 验证集校准结果
```

`meta.json` 会记录实际后端。例如：

```json
"backend": {
  "requested": "sklearn",
  "used": "sklearn",
  "fallback": false,
  "versions": {"numpy": "2.5.3", "sklearn": "1.9.1", "xgboost": null}
}
```

---

## 6. 校准与评估

### 6.1 校准（train 默认已做）

```bash
python3 -m wind_audio calibrate \
  --model models/l1_supervised \
  --manifest data/manifest.csv \
  --annotations data/annotations.csv \
  --splits data/splits.json \
  --split val \
  --target-recall 0.95
```

只在 val 上搜索：

- 阈值；
- `min_duration`（最短事件）；
- `merge_gap`（合并间隔）；
- `low_ratio`（滞回退出比例）。

选择规则：先满足目标召回，再最小化 FA/h，再比精确率。达不到目标会写 `target_reached=false`，不会伪造。

### 6.2 测试集评估

```bash
python3 -m wind_audio evaluate \
  --model models/l1_supervised \
  --manifest data/manifest.csv \
  --annotations data/annotations.csv \
  --splits data/splits.json \
  --split test \
  --out reports/l1_supervised_test
```

输出：

```text
reports/l1_supervised_test/
  metrics.json
  brief.md
```

关键指标：

- 事件召回：命中 GT 事件 / GT 事件数；
- 事件精确率：命中 / 已审核预测；
- FA/h：未命中且落在已审核 normal 的预测 / normal 小时数；
- 覆盖率：命中预测覆盖 GT 异常时长的比例；
- 报警时长占比；
- 起止边界中位误差。

规则：

- 一对一 IoU 贪心匹配（默认 `--match-iou 0.1`）；
- 覆盖全文件的一个报警最多命中一个 GT，不能命中所有；
- 只落在 unknown 的预测被排除；
- 文件级粗标不参与评估；
- 零样本情形输出 `null`，不产生 NaN 或假满分。

不要在 test 上调阈值。`evaluate --threshold` 只用于分析，会打印警告。

---

## 7. 推理

### 7.1 单条

```bash
python3 -m wind_audio run \
  --input data/audio/WT01_20261103_1200.wav \
  --model models/l1_supervised \
  --manifest data/manifest.csv \
  --annotations data/annotations.csv \
  --out runs/WT01_20261103_1200
```

可选：

```text
--review data/reviews/reviews.csv   # 报告显示已有人工 L1/L2 结论
--threshold 0.8                     # 显式覆盖阈值，报告标 threshold_source=cli
--min-dur 0.2 --merge-gap 0.3 --low-ratio 0.6
--fmax-plot 16000
--clip-context 3.0
```

推理时不允许用 `--n-mels/--n-fft/--sr/--fmin/--fmax` 改模型特征；和模型不一致会直接报错。

### 7.2 批量

```bash
python3 -m wind_audio run \
  --input data/audio \
  --model models/l1_supervised \
  --out runs/batch_20261103
```

批量为每个文件建独立子目录，同名文件用路径哈希区分，不会互相覆盖。

### 7.3 输出

```text
runs/<name>/
  report.html       # 波形 + 时频图 + 分数 + 标注 + 试听 + 事件表
  waveform.png
  spectrogram.png   # 低频在下
  score.png
  events.json       # 音频 ID、真实时长、模型版本、阈值来源、事件特征、clip 起止
  events.csv
  audio/            # 原始音频副本
  clips/            # 每个事件 ±3 s
  review.csv/json   # 审核模板
```

报告会明确显示：

- 真实时长、采样率、分析频段；
- 模型版本、检测器、分数类型、阈值及来源；
- demo / 未校准 / 后端回退警告；
- 人工标注 normal/anomaly/unknown 色带；
- 每个事件的试听片段。

---

## 8. 人工审核回流

### 8.1 导出

```bash
python3 -m wind_audio review-export \
  --events runs/WT01_20261103_1200/events.json \
  --out runs/WT01_20261103_1200/review.csv
```

字段：

```text
event_id,audio_id,source_audio_path,predicted_start_s,predicted_end_s,
corrected_start_s,corrected_end_s,l1_label,l2_label,fault_type,note,
reviewed_at,source_model_version,clip_path
```

### 8.2 填写

| 字段 | 取值 |
|---|---|
| `l1_label` | `normal` / `anomaly` / `uncertain` / 空白 |
| `l2_label` | `fault` / `nonfault` / `uncertain` / `not_trained` |
| `corrected_start_s/end_s` | 人工修正区间，留空用预测区间 |
| `fault_type` | `l2_label=fault` 时填具体类型 |

语义：

- `l1_label=anomaly` + `l2_label=nonfault`：**L1 正例**（异常但非故障）；
- `l1_label=normal`：确认误报，才是 L1 normal 反馈；
- `uncertain`/空白：不进入训练/评估；
- 当前没有 L2 模型，`l2_label` 默认 `not_trained`。

### 8.3 导入

```bash
python3 -m wind_audio review-import \
  --review runs/WT01_20261103_1200/review.csv \
  --out data/reviews/reviews.csv \
  --json data/reviews/reviews.json \
  --emit-annotations data/reviews/annotations_from_review.csv
```

`--emit-annotations` 生成独立反馈文件，不会自动改 `data/annotations.csv`；人工确认后再合并。

---

## 9. 服务器部署与加载

```bash
rsync -a models/l1_supervised/ server:/opt/wind_audio/models/l1_supervised/
```

服务器安装：

```bash
python3 -m venv /opt/wind_audio/.venv
/opt/wind_audio/.venv/bin/pip install numpy scikit-learn xgboost
```

加载：

```python
from wind_audio.audio import load_wav
from wind_audio.detectors import load_detector
from wind_audio.features import extract_features

det = load_detector("models/l1_supervised")
x, sr = load_wav("case.wav", target_sr=det.feature_config.sr)
bundle = extract_features(x, sr, det.feature_config, "case")
events = det.predict_events(bundle)
```

加载会校验特征配置、特征名/顺序和后端依赖；不一致直接报错。详见 [MODEL.md](MODEL.md)。

---

## 10. 用合成数据跑一遍完整链路

没有真实数据时，可以先用合成数据确认命令都通：

```bash
python3 -m wind_audio.synthetic --out data/synthetic --seed 0

python3 -m wind_audio check \
  --manifest data/synthetic/manifest.csv \
  --annotations data/synthetic/annotations.csv

python3 -m wind_audio split \
  --manifest data/synthetic/manifest.csv \
  --annotations data/synthetic/annotations.csv \
  --out data/synthetic/splits.json --seed 7

python3 -m wind_audio train \
  --manifest data/synthetic/manifest.csv \
  --annotations data/synthetic/annotations.csv \
  --splits data/synthetic/splits.json \
  --detector baseline --out models/syn_baseline --target-recall 0.9

~/miniconda3/envs/rag_learn/bin/python -m wind_audio train \
  --manifest data/synthetic/manifest.csv \
  --annotations data/synthetic/annotations.csv \
  --splits data/synthetic/splits.json \
  --detector supervised --algo sklearn \
  --out models/syn_supervised --target-recall 0.9

python3 -m wind_audio evaluate \
  --model models/syn_baseline \
  --manifest data/synthetic/manifest.csv \
  --annotations data/synthetic/annotations.csv \
  --splits data/synthetic/splits.json \
  --split test --out reports/syn_baseline_test

~/miniconda3/envs/rag_learn/bin/python -m wind_audio run \
  --input data/synthetic/audio/syn_mixed.wav \
  --model models/syn_supervised \
  --manifest data/synthetic/manifest.csv \
  --annotations data/synthetic/annotations.csv \
  --out runs/syn_mixed_demo
```

注意：合成指标只验证程序逻辑，不代表真实风机效果。

---

## 11. 数据量建议

代码门槛（能跑）：

- 已审核 normal ≥ 60 s；
- 已审核 anomaly ≥ 10 s；
- ≥ 3 个独立分组；
- 有监督：normal/anomaly 各 ≥ 10 个目标样本。

可信评估建议：

- 正常：20–40 小时，覆盖不同机组、季节、风速、启停；
- 异常：至少 50–60 个独立事件，才比较有把握说“95% 召回”；
- 测试集按机组/时间分组，不能随机切 30 s 窗口。

---

## 12. 当前真实数据的正确结论

```bash
python3 -m wind_audio check \
  --manifest data/manifest.csv \
  --annotations data/annotations.csv
```

会得到：

```text
已审核正常 0.0s · 已审核异常 0.0s · 未审核 30.0s
异常事件 0 个 · 完全审核文件 0 · 含粗标文件 1
```

这不是程序故障，而是数据状态：现有 A01_01 只有 file 级粗标，不能变成逐帧真值。补正常录音和秒级异常区间后，按第 3–7 步即可训练和评估真实模型。
