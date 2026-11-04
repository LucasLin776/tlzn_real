# Wind Audio：频谱波动窗口检测＋PANNs 异常识别

本项目用于风机声音的两级分析：

- **L1 不需要训练**：寻找超过阈值的局部频谱波动，输出候选异常的完整时间窗口。
- **L2 才需要训练**：使用 PANNs 特征和人工审核记录，学习多种异常类型，并通过回放续训。

L1 的目标不是识别所有异常声音，而是定位“频谱从稳定到明显波动”的片段。
波动前后可以是不同音量、不同频率结构的稳定工况；不要求后半段恢复成前半段的样子。
候选窗口不等于已确认故障，正常工况切换也可能成为候选。

## 1. 架构与算法

```text
WAV → STFT / Mel 功率谱
        ↓
     相邻局部窗口比较 → 频谱变化分数 → 触发候选
        ↓
     冻结事件前背景 → 跟踪持续活动与衰减尾部 → 背景恢复/新工况稳定
        ↓
     微弱噪声过滤 → 合并短间隔 → 候选完整事件窗口 [start_s, end_s]
        ├─ 独立窗口 WAV、试听上下文、HTML、审核表
        └─ L2：原音频＋候选窗口 → 冻结 PANNs CNN14 → 多标签分类
                                                      ↓
                           已知类型 / 疑似正常 / unknown＋相似案例
                                                      ↓
                           人工审核 → 累计数据 → 分类头回放续训
```

### L1：频谱波动，而不是固定正常模板

1. 默认 44.1 kHz、2048 点 FFT、10 ms 帧移、64 Mel，分析 20 Hz–16 kHz。
2. 每 20 ms 计算一次分数：比较该时刻前后各 0.1 秒的平均 Mel 功率谱。
3. 对每个频带计算两侧功率的 dB 差，取按局部能量加权的均方根作为变化分数。
4. 使用共同的局部频谱底限（默认相对峰值 40 dB），降低极弱频带数值抖动；默认约 60 ms 中值平滑抑制孤立噪声。
5. 双阈值只产生“变化触发种子”，不再把它直接当成完整事件。
6. 取触发前约 0.6 秒的局部谱中位数作为临时背景，事件期间不更新它。
   使用相对该背景的正向谱能量差持续跟踪：即使前后谱变化变小，中间持续段和衰减尾部仍可保持活动。
7. 活动降到释放阈值以下，且连续 0.25 秒没有明显变化，才确认背景恢复；
   或持续 0.6 秒能量和谱形稳定、并呈现低平坦度谱线/近似原谱形，确认为新稳定工况。
   结束边界回溯到这段稳定时间的起点，不额外把整段确认延迟加进窗口。
8. 过滤不显著的独立候选，再合并短间隔，最后仅补各 0.05 秒边界。
   没有固定“异常最多持续几秒”的截断；直到录音结束仍未恢复时保留到文件尾，并标记结束尚未确认。

分数单位是加权 dB 变化，不是概率。稳定的大声与稳定的小声都可得低分；
整条录音乘固定增益时，局部比值基本不变；突然的能量跃变仍可触发候选，恢复稳定后分数回落。
不拟合训练模型，但会测量当前录音的局部背景和相对强度，用于规则判断。
默认参数是工程起点，尚未通过独立精确标注验证。

### 微弱噪声如何过滤

- 局部显著性：事件期间 Mel 功率谱相对背景的差异比例
  `sum(abs(P-B)) / sum(P+B)`，峰值需达到默认 0.3，防止微弱频带变化单独报警。
- 相对强度：在整条录音的最强若干帧（总计约 0.1 秒，不要求连续）中取能量级中位数；
  候选峰值低于该参考 15 dB 以上时过滤。不是绝对声压标定，整体录音增益变化基本不改变结果。
- 上述门槛用于事件是否入选，**不会用于切掉已入选事件的弱尾部**；
  尾部由较低的持续活动释放阈值决定。

调大 `--min-prominence` 或调小 `--relative-floor-db` 可加强过滤；
反向调整可保留更弱的真实事件。强过滤也会漏掉弱异常，不能把所有被过滤项都认定为噪声。
整段相对强度参考意味着：同一声音放在不同响度组成的录音里，入选结果可能不同；
它适用于当前离线筛查，不是跨设备已标定的故障阈值。

默认忽略依赖边缘补齐的 FFT 帧，避免文件首尾伪变化。极短音频没有完整 FFT 帧时输出零分；
录音开始就已经存在且持续稳定的异音，不能靠本方法判断其是否异常。
这是离线分析，使用右侧窗口和居中滤波，不是已实现的实时流式服务。

### 窗口是 L1 的核心输出

每个事件明确区分：

| 字段/文件 | 含义 |
|---|---|
| triggers、trigger_start_s / trigger_end_s | 窗口内各变化种子，以及首尾触发范围；不是完整事件边界 |
| start_s / end_s / duration_s | 跟踪持续段和尾部后的窗口，L2 和审核模板使用此范围 |
| boundary_reason | background_return / new_stable_state / end_of_file，解释结束依据 |
| right_censored | true 表示到文件末尾仍未确认结束，不代表事件恰好在那里结束 |
| windows/*.wav | 检测窗口本身，不带额外试听上下文 |
| clips/*.wav、clip.start_s/end_s | 默认额外前后各 1 秒，便于听取，不是异常真值 |
| scores.csv、activity.png | 分开记录/绘制变化触发分数、持续活动分数、显著性、能量和最终窗口状态 |
| diagnostics.rejected_candidates | 被过滤候选及原因；不进入 L2 或审核表，不是假定正常标签 |
| l1_config.json | 本次实际使用的全部 L1 参数 |

所有时间相对于原录音；窗口裁剪到文件范围内，补充后重叠的窗口会合并。
连续活动期间多个触发属于同一事件；两个完整事件间安静间隔不超过 merge_gap 时再合并。
不会用一个峰值代替异常窗口；只有没找到结束依据时才保留到文件末尾并标记。

20 ms 是输出网格，不代表 20 ms 的真实边界精度：FFT、比较窗口、平滑和边界补充都会带来时间扩展。
边界需要精确时，人工修订 corrected_start_s/end_s；不能声称默认参数能保证覆盖每一个异常。

### L2：保留人工审核增量学习

- 本地官方 32 kHz PANNs CNN14：STFT 1024 / hop 320 / 64 Mel，六个双卷积块，输出 2048 维特征。
- 默认 4 秒上下文、2 秒滑动步长；读取原音频，短事件补真实上下文，边缘不足补零；长事件滑窗平均后归一化。
- 冻结 CNN14，训练线性 sigmoid 分类头：一个正常/异常输出＋K 个可并存的异常类型。
- 掩码 BCE 只使用已确认正负标签；旧类别可回放，新类别末尾追加；可选旧输出蒸馏、原型相似度拒识和相似案例检索。
- 主干不随审核更新；参数按批次更新，不是每审核一条就自动训练或上线。
- 每个输出在 train/val 正负例中至少各覆盖两个来源才启用决策。该门槛不等于统计充分。
- 输出 known_anomaly / likely_normal / unknown，均保留原 L1 告警供人工审核。
- L2 内部的 4 秒编码上下文不会改写 L1 的检测窗口或自动变成人工标签。

## 2. 环境与直接运行

使用你的环境：

```bash
conda activate post_training
cd /Users/shuoqinlin/Desktop/wind_audio

# 不需要模型目录、不需要标注、不需要先训练
python -m wind_audio run \
  --input data/anomaly/A01_01_20261030_120227.wav \
  --out runs/l1_spectral_check
```

默认参数来自内置配置，仓库提供同值的 config/l1.json 便于编辑：

```bash
python -m wind_audio run \
  --input data/anomaly/A01_01_20261030_120227.wav \
  --config config/l1.json --threshold 6 \
  --out runs/l1_threshold6
```

input 可为一个 WAV 或目录。每次请选择新输出目录，已有审核记录的目录不覆盖。
目录输入生成 `index.html` 总览和 `summary.json`，各录音分别保存报告、窗口和审核表：

```bash
python -m wind_audio run \
  --input /Users/shuoqinlin/Desktop/异常 \
  --config config/l1.json --out runs/batch_event_windows
```

| 参数 | 默认值 | 作用 |
|---|---:|---|
| --threshold | 6 | 变化分数进入阈值，越高越不敏感 |
| --low-ratio | 0.6 | 仅变化触发种子的低阈值比例 |
| --compare-s | 0.1 秒 | 前后各比较多长的谱窗口 |
| --smooth-s | 0.06 秒 | 分数中值平滑时间 |
| --target-step | 0.02 秒 | 输出网格间隔 |
| --min-dur | 0.06 秒 | 合并后跟踪事件的最短持续时间，在边界补充前检查 |
| --merge-gap | 0.35 秒 | 完整事件之间的合并间隔 |
| --padding | 0.05 秒 | 检测窗口每端额外覆盖范围 |
| --clip-context | 1 秒 | 仅试听切片每端额外上下文 |
| --reference-s | 0.6 秒 | 事件前背景谱长度 |
| --release-db | 3 | 相对背景的正向谱差释放阈值，越低尾部越长 |
| --quiet-s | 0.25 秒 | 确认背景恢复所需连续安静时间 |
| --settle-s | 0.6 秒 | 确认新稳定工况所需时间 |
| --stability-db | 1.5 | 新工况能量/归一化谱形稳定容差 |
| --min-prominence | 0.3 | 与背景的能量差异比例下限，越高过滤越强 |
| --relative-floor-db | 15 | 容许比录音强声参考低多少 dB，越小过滤越强 |

配置版本为 `spectral_event_v2`；0.4 的 `spectral_change_v1` 配置拒绝加载，避免静默混用。
旧报告仍可查看；需用新配置重新生成新目录，旧人工审核不会自动转成新边界。

L1 只需 NumPy；完整 L2 还需 PyTorch。requirements.txt：
```text
numpy>=1.24,<3
torch>=2.6,<3
```

本次没有安装任何依赖或下载权重。你可自行安装；从仓库目录用 python -m wind_audio 无需安装本项目。
L2 请自行准备可信来源的官方 CNN14 检查点，例如 weights/Cnn14_mAP=0.431.pth；
不会自动下载，也不需要 xgboost、sklearn、torchaudio、librosa 或 torchlibrosa。
上游：[PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) · [论文](https://arxiv.org/abs/1912.10211)。

## 3. 用你的标注检验 L1 窗口

标注数据的主要用途是检查漏检、误检和窗口边界，不是训练 L1。
精确标注格式见 [标注规范](docs/ANNOTATION.md)；完整审核的录音可把未标异常的部分视为正常，
未完整审核则保留 unknown。整段“有异常”的粗标不能当成整段都是精确异常窗口。

```bash
python -m wind_audio evaluate \
  --manifest data/manifest.csv --annotations data/annotations.csv \
  --config config/l1.json --out reports/l1_check
```

不需要 train/val/test 划分，一条精确标注录音也能检验。
有固定独立评估集时可加 --splits data/splits.json --split test。
检查输出的事件召回、精确率、每小时误报、异常覆盖率、边界误差和正常区报警时长。

评估使用连续标注区间，不再把短事件压到 0.1 秒网格；
只计入已审核区间，IoU 分母排除未知时长，并按一对一事件匹配。
已匹配窗口中超出真实异常、落在正常区的部分也计入正常区报警时长。
默认匹配 IoU=0.1 较宽松，可用 --match-iou 提高；不能仅看召回忽略边界过宽。

阈值可手动调整：先在开发录音中检查，再冻结配置，在未参与调参的录音上报告正式指标。
没有自动阈值训练，也不会把待检录音拟合成正常参考。
当前现存录音只有整段粗标，能直接检测窗口，但不能据此给出真实召回率或边界精度。

## 4. 人工审核

打开 run 输出的 report.html，对每个候选窗口听取并修订 review.csv：

- l1_label：normal / anomaly / uncertain。
- corrected_start_s / corrected_end_s：必要时修订检测边界。
- positive_types / negative_types：已确认/排除的类型 ID，多个用分号分隔。
- types_complete：只有已审查当前全部类型时才填 true。
- l2_label / fault_type：独立的人工故障结论，不把异常声音直接等同于故障。

```bash
python -m wind_audio review-import \
  --review runs/l1_spectral_check/review.csv --out data/reviews/reviews.csv
```

预测不自动写成人工真值。anomaly + nonfault 仍是异常；uncertain/待审不进入 L2 训练。
导入会补审核时间，已有事件的训练标签发生改变时递增 review_revision。
检查无告警区域也很重要：L1 漏掉的事件不会自动出现在 L2 数据中。

## 5. L2 首次训练与续训

以下只适用于 L2；L1 无此训练步骤。
准备 manifest、累计 reviews 和 config/taxonomy.json，按原始录音/采集来源固定划分：

```bash
python -m wind_audio split \
  --manifest data/manifest.csv --annotations data/annotations.csv \
  --group-by source --out data/splits.json

python -m wind_audio l2-prepare \
  --reviews data/reviews/reviews.csv --manifest data/manifest.csv --splits data/splits.json \
  --taxonomy config/taxonomy.json --out data/l2/v1.json

python -m wind_audio l2-check --dataset data/l2/v1.json

python -m wind_audio l2-embed \
  --dataset data/l2/v1.json --checkpoint weights/Cnn14_mAP=0.431.pth \
  --out cache/l2_v1.npz --device cpu

python -m wind_audio l2-train \
  --dataset data/l2/v1.json --features cache/l2_v1.npz \
  --version l2_v1 --out models/l2_v1

python -m wind_audio l2-evaluate \
  --model models/l2_v1 --dataset data/l2/v1.json --features cache/l2_v1.npz \
  --split test --out reports/l2_v1_test.json
```

新审核导入后，保持旧来源划分，新类别仅在 taxonomy 末尾追加：

```bash
python -m wind_audio l2-prepare \
  --reviews data/reviews/reviews.csv --manifest data/manifest.csv --splits data/splits.json \
  --taxonomy config/taxonomy.json --previous data/l2/v1.json --out data/l2/v2.json

python -m wind_audio l2-embed \
  --dataset data/l2/v2.json --checkpoint weights/Cnn14_mAP=0.431.pth \
  --out cache/l2_v2.npz --device cpu

python -m wind_audio l2-train \
  --dataset data/l2/v2.json --features cache/l2_v2.npz --resume models/l2_v1 \
  --version l2_v2 --out models/l2_v2
```

--previous 保留旧审核时覆盖的类别范围；旧完整审核不会自动把新类型当负例。
模型保存分类头、类别顺序、训练来源、原型、阈值和记忆库；默认不覆盖旧输出。
默认续训用新/修订样本＋每类最多 32 个回放代表样本，支持正负标签掩码和旧输出蒸馏。
仍需累计快照和历史音频；不允许把旧测试来源移入训练。
旧边界/保留集标签需修订时应重建明确的数据基准，从头训练分类头，不能静默续训。
用独立保留集比较后显式选择新模型，不自动替换上线。

## 6. 两级推理

```bash
python -m wind_audio pipeline \
  --input data/anomaly/A01_01_20261030_120227.wav \
  --l1-config config/l1.json --l2-model models/l2_v2 \
  --checkpoint weights/Cnn14_mAP=0.431.pth --device cpu --out runs/two_stage_v2
```

--l1-config 可省略，--l1-threshold 可覆盖阈值；不再需要 --l1-model。
pipeline 接受单条 WAV；批量先 run，再逐份 l2-run：

```bash
python -m wind_audio l2-run \
  --events runs/l1_spectral_check/events.json --model models/l2_v2 \
  --checkpoint weights/Cnn14_mAP=0.431.pth --out runs/l1_spectral_check/events_l2.json
```

L2 使用 events 的 start_s/end_s，不使用带额外试听上下文的 clip.start_s/end_s。
联合结果为 JSON；HTML 展示 L1 窗口，审核表仍由人工填写。

## 7. 对照与验证

L1 不再比较逻辑回归/XGBoost等训练模型；可在同一检验集比较：
固定阈值、比较窗宽、释放阈值、稳定确认时间、噪声过滤和合并间隔的影响。
对照阈值必须在开发集选定，不能把反复调整后的测试结果当独立成绩。

L2 保留：静态模型、朴素增量（--no-replay --distill 0）、仅回放（--distill 0）、
默认回放＋蒸馏、累计从头重训（--full-retrain）。
这些消融都冻结 PANNs；无回放只消融分类头历史训练，原型仍由累计 train 统计。
L2 指标是已审核候选上的条件指标，不是整个系统的端到端召回。

```bash
python -m unittest discover -s tests -t . -v
WIND_AUDIO_TEST_PANNS=1 python -m unittest tests.test_l2.PannsStructureTests -v
```

必要回归包括：水平谱线＋背景噪声不持续触发、幅值切换后恢复、频率变化定位、
冲击窗口覆盖、长持续段不退化成两条边缘、尾部覆盖、弱扰动过滤、相邻与独立事件、
窗口 WAV 时长、长文件名切片不碰撞、批量报告、L2 接收正确边界。
测试临时信号不代表真实风机性能；CNN14 结构测试不下载预训练权重。

## 8. 工程布局和边界

```text
wind_audio/
  l1.py                    # 无需训练的局部频谱变化评分
  features.py              # 频谱与配置，不再生成 282 维分类特征
  windowing.py             # 局部背景冻结、事件跟踪、稳定结束、微弱噪声过滤
  detectors.py             # 跟踪窗口、合并、边界补充、配置加载
  cli.py                   # run / evaluate / review / L2 命令入口
  audio.py / events.py      # 音频处理、事件后处理与描述
  data.py / splits.py       # 标注检验与 L2 来源划分
  evaluate.py               # 连续时间窗口评估
  review.py / report.py / render.py
  l2/                      # PANNs、审核快照、回放分类训练与推理
config/l1.json             # L1 参数，不是训练模型
config/taxonomy.json       # L2 类型表，配置项不表示已训练成功
docs/ANNOTATION.md
tests/
requirements.txt / pyproject.toml / .github/workflows/ci.yml
```

旧 L1 train/calibrate、baseline/supervised 和 --model/--l1-model 入口已移除，避免混用不同目标；
历史模型、原始录音、审核记录和已有报告保留在原处，不修改 Git 历史。

仍需注意：

- 稳定异常音调、过慢变化、过短冲击、文件边缘缺乏上下文的事件可能漏检。
- “新稳定工况”是启发式判断，持续稳定的异常谱线也可能被当成新工况；需要 L2/人工区分。
- 相对强度过滤可能压掉弱真实异常；批量窗口数不等于准确率，过滤数量不等于已证实误报。
- 工况切换和环境冲击也可触发候选，是否异常需 L2/人工确认。
- 本版全文件离线分析，未实现流式处理；大录音需注意内存。
- WAV 支持 8/16/32 位整数 PCM，不支持 24 位或浮点 WAV；多声道平均可能抵消信号。
- 数据、权重、审核隐私和产物不应上传；.gitignore 不会取消过去的跟踪。
- 没有选择许可证、安装库、下载权重、提交或推送。真实效果需独立标注验证。
