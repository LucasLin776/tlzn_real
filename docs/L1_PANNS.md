# L1 PANNs 正常时序学习（代码实现，尚未训练/校准）

显式模型类型：`panns_temporal_dae_v1`，模型清单 `schema_version: 4`。
这是独立的新 L1；旧 `normal_envelope_v1`、v2/v3 配置和默认行为保持不变。
没有修改 `wind_audio/l2/`，新网络不使用旧 L1 的分数、候选、背景参考或整段报警规则。

本次交付没有启动训练、微调或阈值拟合，没有对参考批次进行特征提取/评估。
没有训练结果，也不主张准确率提高。以下命令均需用户审核后自行执行。

## 数据流与模型

完整 WAV → 有限支持重采样 → 冻结 CNN14 时序特征 → 固定动态表示 →
正常训练集 RMS 标准化 → 小型时序 DAE → 逐时刻网络重建残差 → 正常/候选/unknown。

| 阶段 | 实现与约束 |
|---|---|
| 输入 | PCM WAV（8/16/32-bit 整型），多通道算术平均；整段输入，不接收旧候选 |
| 重采样 | 32 kHz；局部窗化 sinc，原生采样点左右各 16 点支持；不是会全段混合的 FFT 重采样 |
| 前端 | 本地官方 CNN14 32k 权重的 FFT1024/hop320/mel64 核；`center=False` |
| 响度处理 | 每帧 log-mel 去掉 mel 均值并加固定 -40 dB；不以绝对响度作判断。这是显式记录的 AudioSet 前处理改动 |
| 编码器 | 独立实现 CNN14 B1/B2/B3，64/128/256 通道；只做频率平均，保留时间轴；参数、BN 统计量永久冻结 |
| 多尺度对齐 | B1/B2 分别时间平均池化 4/2 倍，B3 原生时间轴；各尺度逐时间点通道 L2 归一化；合计 448 维，无整段池化 |
| 动态目标 | 短差分 1 步、长差分 4 步、半径 4 步的局部均值残差；拼成固定 1344 维；不拼入静态音色或响度 |
| 标准化 | 仅正常训练核心帧估计逐通道 RMS，最小值 0.01；无全文件自适应、无裁剪、无校准集参与 |
| DAE | Conv1D 1344→96→24→96→1344，kernel=3，dilation=1/2/2/1；中间 GELU；无 bias、无 skip/输入直通 |
| 损失 | 正常核心帧 MSE，重建干净固定特征；编码器/目标无梯度；AdamW 仅更新 DAE |
| 可选增强 | 标准化动态特征上添加 std≤0.05 的小高斯噪声，默认关闭；必须显式确认符合该业务的正常定义 |
| 评分 | 每个原生时间格的特征残差 RMS；不使用旧频谱规则，不做会擦掉单帧冲击的中值滤波 |

对时间恒定的编码特征，动态目标趋近零；无 bias 网络把零映射为零。
因此不会因为一个稳定工况的音色与旧背景不同而持续报警，也不要求工况切换后回到原背景。
**真实宽带噪声仍有随机波动**，其容忍程度、短冲击灵敏度和实际泛化必须用后续正常训练、独立校准及盲测验证；不能由合成测试保证。

## 时间轴、完整支持与边缘

- STFT 窗口为 32 ms，步长 10 ms；三层池化后真实特征步长为 **80 ms**。
- 第一个特征中心为 **0.051 s**，后续为 `0.051 + i × 0.080`。
- 对齐后的三个尺度感受野分别约 142/222/382 ms；最深使用到 B3，不使用 2048 维整段向量。
- 动态表示的默认最大半径 320 ms；DAE 感受野 13 格，左右各 480 ms。
- 评分依赖范围的保守界为左右各 **0.991 s**，全宽 **1.982 s**；非 32k 输入还加左右各 `16 / 原生采样率`。
- **80 ms 只是原生输出格距，不是经过验证的定位精度**；未插值成更细网格。事件边界由格子的半步边界给出，可能受约 2 秒上下文影响。
- 整段推理按对齐的块处理，每块带 13 格 halo，丢弃人工填充影响。上下文不完整的分数为无效值，JSON 中为 `null`。
- 文件开头/末尾、过短音频、未确认的中间状态明确进入 `unknown_regions`，不会拿正常区间的补集充当所有异常，也不会把 unknown 当正常。

训练时先选明确正常的原始区间，再形成不重叠的原始音频裁片。
裁片包含所有前端、重采样、动态差分和网络上下文；只在去掉 halo 的核心处计算损失。
默认核心最多 32 格；不足时可用更短核心，但至少需要 27 格的输入（约 2.182 s，加网格对齐与重采样余量）。
太短的正常区间不生成样本；所有正常区间都不足时明确报错。
模型保存实际使用的裁片和未产生样本的录音 ID。不会把异常/未审核帧拿来补上下文。

## 数据导入：用户明确指定配对、语义和来源划分

只读取了 `/Users/shuoqinlin/Desktop/数据/` 的目录及 B02–B07 少量标注样例作为格式参考。
该目录当时未见 B01。参考格式为 `音频/*.wav` 和 `标注/*.csv`；CSV 可能是带 BOM 的制表符分隔 Audition 导出：

```text
Name    Start       Duration    Time Format    Type    Description
任意名  0:01.000    0:04.000     decimal        Cue
```

没有把任何参考批次写入配置/训练集；也没有采纳历史的排序配对、标注语义、完整审核假设或数据划分。
有的参考标注编号和音频长文件名不同，因此需要**显式配对**。
可以先使用下面的配对 CSV 导入快照，也可以用下文的目录训练入口显式指定 `stem / sorted / table`；程序不会暗中选择排序配对：

```csv
audio_id,audio_path,label_path,source_id,split
rec_a,/PLACEHOLDER/audio/recording_a.wav,/PLACEHOLDER/labels/recording_a.csv,original_session_a,train
rec_b,/PLACEHOLDER/audio/recording_b.wav,/PLACEHOLDER/labels/recording_b.csv,original_session_b,calibration
rec_c,/PLACEHOLDER/audio/recording_c.wav,/PLACEHOLDER/labels/recording_c.csv,original_session_c,test
```

路径可为绝对路径，或相对配对 CSV 的路径。用户须审核每一行对应关系。
无论原来是同名配对还是已确认的排序配对，都将最终配对显式列入该表。程序不自动扫描批次加入样本。

`source_id` 必须是**原始录音/采集会话来源**，不是裁片文件名；同一原始录音的裁片、重命名副本、重叠采样窗口必须共用来源。
`split` 必须逐条指定为 `train / calibration / val / test`。
不进行自动比例划分，训练不使用 val/test，校准只用 calibration，评估只用 val/test。
同来源不得跨集合；文件 SHA256 和解码后单声道 PCM SHA256 检查重复，连同来源记录进入模型血缘，跨数据快照也检查训练/校准/评估隔离。
PCM 身份检查能识别同采样率下改头/改名的相同音频，**不是任意重编码、裁切、重采样的声学查重**；那些关联必须由真实 `source_id` 保证。

另提供语义 JSON，二选一：

```json
{"all_markers": "normal"}
```

或逐个精确匹配名字（名字本身不决定语义）：

```json
{"name_map": {"用户定义的正常标记": "normal", "用户定义的异常标记": "anomaly", "待审核": "unknown"}}
```

允许的语义只有 `normal/anomaly/unknown`；未映射名字报错。
**空标注、未标记区间始终 unknown**，不从正常或异常标注推导其补集。
如果只有异常标注，必须另补明确正常标注才能训练。
如果全文件已审核正常，也应提供覆盖全文件的明确正常区间。
重叠的冲突语义、点标记、非 decimal 时间、越界区间报错，不自动裁剪或修补原始标注。
导入产生只读语义快照；训练/评估读取该快照并验证音频身份，不重新解释标记名字。

## 校准与事件

训练产物标记为未校准，不能检测。
校准只统计**完整评分上下文都落在明确正常区间内**的残差，且来源/音频与训练集不重叠。
默认 low/high/impulse 分别取正常残差的 0.95/0.995/0.9995 `higher` 分位数，最小 1e-6。
默认至少 1000 个有效正常校准帧；这不是 1000 个独立样本，也不保证事件级误报率。
校准输出到新目录，不改训练产物，后续检测不改变网络、标准化、阈值或事件参数。

- 普通触发需 high 阈值连续 2 格；单格超过 impulse 可保留明显短冲击。
- low 以下连续 0.32 s 确认为正常；事件在新稳定状态处结束，不参考旧背景。
- 0.16 s 内短间隔可合并，但**任何已确认正常区间或无效上下文都是硬阻断**。
- 未达到持续/强触发条件的轻微孤立波动不产生候选；尚未确认正常的部分仍为 unknown。
- 候选再多也不把正常间隔吞并为整段告警。

## 手动命令示例（本次未执行）

在项目根目录、已有依赖的 Python 环境中使用。全部为占位路径；程序不安装依赖或下载权重。
可用 `--network-config` / `--event-config` 指定结构和事件参数 JSON；未指定时使用上文版本化默认值。

### 推荐：直接指定音频、标签路径，完成后输出验证/测试 PDF

`--audio-dir` 指向 WAV 目录，`--labels`（别名 `--labels-dir`）指向每条录音对应的 Audition CSV/TSV 标注目录。
路径、格式、标注起止、配对数量、重复音频、来源泄漏、有效正常训练/校准上下文均在训练前校验。
报告依赖及中文字体也提前校验；缺少依赖不会自动安装。

`--pairing` 必须显式选择：

- `stem`：音频和标注文件 stem 完全一致。
- `sorted`：**用户已确认**按相对路径自然排序后一一对应；不能仅因数量相同就选择此项。
- `table`：另外提供 `--pairing-file` CSV，列为 `audio_id,audio_path,label_path`。音频路径相对音频目录、标注路径相对标注目录（或用绝对路径）；必须完整一一覆盖指定目录，不接收目录外文件。

语义必须通过 `--marker-label normal` 明确声明，或用 `--semantics /PLACEHOLDER/semantics.json` 指定逐名字映射。
空标注及未标注部分仍然 unknown。不能仅因目录名是 B05 就自动使用正常语义。

另提供 `--splits` CSV，不复用历史划分、不自动切分窗口：

```csv
audio_id,source_id,split
recording_a,original_session_a,train
recording_b,original_session_b,calibration
recording_c,original_session_c,val
recording_d,original_session_d,test
```

`stem/sorted` 模式的 `audio_id` 为标注文件 stem；`table` 模式使用配对表的 ID。
同一原始录音/会话的重叠裁片必须共用 source_id，不能横跨集合。
目录模式必须逐条覆盖全部配对，任何遗漏、重复 ID 或多余划分条目都报错。
本次提供的 B05 两个目录只作为输入/报告格式参考，不自动指定为实际训练数据。

```bash
# 仅验证输入；不提特征、不加载网络、不训练/校准/评估，不创建输出目录
python -m wind_audio l1-panns-train \
  --audio-dir /PLACEHOLDER/audio \
  --labels /PLACEHOLDER/labels \
  --pairing stem --marker-label normal \
  --splits /PLACEHOLDER/splits.csv \
  --checkpoint /PLACEHOLDER/Cnn14.pth \
  --out /PLACEHOLDER/new_run --check-only

# 用户审核后手动执行：正常训练 -> 独立正常校准 -> val/test PDF
python -m wind_audio l1-panns-train \
  --audio-dir /PLACEHOLDER/audio \
  --labels /PLACEHOLDER/labels \
  --pairing stem --marker-label normal \
  --splits /PLACEHOLDER/splits.csv \
  --checkpoint /PLACEHOLDER/Cnn14.pth \
  --out /PLACEHOLDER/new_run --epochs 20 --device cpu
```

`--check-only` 对权重仅验证存在且非空；实际训练加载时还会检查官方结构兼容性。
完整模式要求四个非空集合，train 有有效正常上下文，calibration 有至少 `--min-frames`（默认 1000）个有效正常帧，
val/test 分别有明确审核区间。校准分位数使用 `--quantiles` 指定；val/test 不参与拟合或参数选择。
无法满足时在训练前报错，不用验证集/测试集顶替校准集。
保留 `--train-only`：只写未校准模型，不做校准或报告；之后可显式运行校准及评估命令。

默认完整流程输出：

```text
new_run/
  dataset.json, input_validation.json
  manifest.csv, annotations.csv, pairing.csv
  training.json, training.md
  metrics.json, brief.md, run.json
  trained/model.json, trained/temporal.pt       # 未校准
  calibrated/model.json, calibrated/temporal.pt # 检测使用
  val/
    comparison.pdf, metrics.json, brief.md
    per_file.csv, normal_regions.csv, events.csv, unknown_regions.csv, labels.csv
    manifest.csv, annotations.csv, pairing.csv
    plots/*.png, plots/index.csv
  test/                                      # 与 val 相同格式
```

每条录音一页横向 A4 PDF，按 ID 自然顺序排列：原始波形、网络残差与固定阈值、人工/模型正常区间、人工异常/模型候选区间。
借鉴参考报告版式，但不调用旧 L1 的评分、补集标签或跨正常间隔的视觉合并规则。
人工未审核和模型拒判在各自时间线上分别显示；准确范围同时保存在 CSV。
正常召回率及异常召回率保留拒判部分的分母，另列已审核区间覆盖率；未标注区域不能据此当异常。
仅训练模式仍将模型保存到 `new_run/trained/`。只有完整成功后才写 `run.json` 的 complete 状态；
中途失败会保留已产生的模型和明确标为 partial 的报告，不伪装成成功，也不覆盖旧结果。

### 兼容快照输入与分步执行

```bash
# 1. 显式配对、语义与划分；不提取 PANNs 特征
python -m wind_audio l1-panns-prepare \
  --pairing /PLACEHOLDER/reviewed_pairing.csv \
  --semantics /PLACEHOLDER/marker_semantics.json \
  --out /PLACEHOLDER/new_dataset.json

python -m wind_audio l1-panns-check --dataset /PLACEHOLDER/new_dataset.json

# 2. 显式选择仅训练模式；不自动校准或生成 PDF
python -m wind_audio l1-panns-train \
  --dataset /PLACEHOLDER/new_dataset.json \
  --checkpoint /PLACEHOLDER/Cnn14_mAP=0.431.pth \
  --out /PLACEHOLDER/trained_uncalibrated \
  --epochs 20 --device cpu --train-only

# 3. 用 calibration 中独立正常上下文校准到新的模型目录
python -m wind_audio l1-panns-calibrate \
  --dataset /PLACEHOLDER/new_dataset.json \
  --model /PLACEHOLDER/trained_uncalibrated/trained/model.json \
  --out /PLACEHOLDER/new_calibrated_model --device cpu

# 4. 直接读取完整音频；不接受任何标注或阈值覆盖参数
python -m wind_audio l1-panns-detect \
  --model /PLACEHOLDER/new_calibrated_model/model.json \
  --input /PLACEHOLDER/unseen_recording.wav \
  --out /PLACEHOLDER/new_detection --device cpu

# 等价的原检测入口，供已有 L2 pipeline --l1-config 使用
python -m wind_audio l1-detect \
  --config /PLACEHOLDER/new_calibrated_model/model.json \
  --input /PLACEHOLDER/unseen_recording.wav \
  --out /PLACEHOLDER/another_new_detection

# 5. 独立测试；内部调用与检测完全相同的 Model.predict(audio_path)
python -m wind_audio l1-panns-evaluate \
  --model /PLACEHOLDER/new_calibrated_model/model.json \
  --dataset /PLACEHOLDER/new_dataset.json --split test \
  --out /PLACEHOLDER/new_evaluation_report --device cpu
```

可选增强必须同时设置 `--noise-std 0.01 --augmentation-reviewed`，不是默认训练操作。
模型迁移后可以在专用校准/检测/评估入口用 `--checkpoint` 指定新本地路径，但必须匹配保存的权重哈希。
原 `l1-test` 不接收新模型，请用 `l1-panns-evaluate`，避免旧标签补集及报告规则污染新输出。
`l1-panns-evaluate` 默认 `--out` 为新报告目录，生成 PDF/CSV/JSON；需要原来的单 JSON 输出时，
显式加 `--json-only --out /PLACEHOLDER/evaluation.json`。

## 保存、输出与接口

模型目录包含：

- `model.json`：模型类型、结构、时间支持、PANNs 路径/SHA256、完整预处理、RMS 标准化、训练来源/裁片/状态、校准阈值/来源、事件参数。
- `temporal.pt`：仅新增网络权重；清单保存其 SHA256。PANNs 大权重不复制、不下载。

`torch.load` 使用 `weights_only=True`；不使用不安全回退。
未训练、未校准、权重缺失/哈希不符、预处理版本不兼容均失败，不会退回旧 L1。
输出路径必须是新路径，不覆盖旧模型、人工审核或历史结果。

检测输出 `events.json` 和空白的 `review.csv/review.json`；保留 L2 消费字段：
`audio.id/path/sha256/duration_s`、`events[].event_id/start_s/end_s`。
附加 `normal_regions`（与 `stable_segments` 同义）、`unknown_regions`、原生时间轴分数/有效掩码/状态及物理支持信息。
检测入口不生成旧 HTML/试听切片，L2 可以直接按原始音频路径与候选区间使用这些结果。
验证/测试报告入口生成独立 PDF，不影响检测或 L2 接口。
评估只在明确审核区间做时间格统计，单独报告 unknown/边缘拒判时长与覆盖率、含拒判的保守异常召回；不把拒判当正常，也不把条件指标称为事件准确率。

## 无训练验证

新增 `tests/test_l1_panns.py` 只做合成数据测试、接口检查与前向验证。
测试的 checkpoint 是形状兼容但随机参数的临时夹具，不是官方预训练权重；训练/校准状态标记也只是序列化测试夹具。
测试对 `backward`、优化器 `step`、分位数拟合设有禁止调用保护。
没有实测训练收敛、阈值质量、真实音频检出效果或官方权重端到端表现。

首版验证结果：21 项合成/接口/前向测试通过；78 项不含训练/校准的现有回归测试通过；
新模块及新增测试的 Ruff E9/F 检查通过，`git diff --check` 通过；`wind_audio/l2/` 无改动。

目录/报告扩展新增 `tests/test_l1_panns_reports.py`，覆盖输入有效性、check-only、四集合预检、
自然顺序 PDF/CSV、检测预测与报告一致性，以及带测试桩的完整流程。
报告测试中的训练/校准均被替换为测试桩，真实 train/calibrate/quantile 调用被禁止。
PDF 用合成音频生成，并通过 Poppler 渲染人工检查排版；不代表真实模型训练或评估结果。
本次扩展验证：15 项目录/报告测试、21 项原 PANNs 测试和 81 项安全回归测试，共 117 项通过；
合成验证集 1 页、测试集 2 页的 PDF 均完成逐页渲染检查。未执行真实训练、校准或 B05 评估。

```bash
python -m unittest tests.test_l1_panns tests.test_l1_panns_reports -v
```

不要在本次“禁止训练/校准”的限制下直接运行全量测试发现：旧 `test_normal.py`、`test_l2.py`、`test_stable_labels.py` 中包含实际拟合调用。
