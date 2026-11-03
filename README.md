# Wind Audio：风机声音两级异常检测

本项目面向离线 WAV：先定位可疑声音，再识别异常类型，并用人工审核结果持续改进识别。
当前实现是**监督式增量学习，不是强化学习**。声音异常不等于机械故障；故障结论仍需人工和工况证据。

## 1. 算法架构

```text
原始 WAV
  │
  ├─ L1：高召回候选检测
  │     声学特征 → 正常参考模型 / 有监督对照 → 阈值与事件合并
  │     输出：起止时间、分数、音频切片、HTML 报告、审核 CSV
  │
  └─ L2：候选区间 + 真实上下文
        32 kHz → 冻结 PANNs CNN14 → 2048 维归一化特征
        ├─ 正常/异常二分类头
        ├─ 可扩展的多标签异常类型头
        ├─ 验证集阈值 + 类原型余弦相似度 → 已知类型 / 疑似正常 / unknown
        └─ 训练样本记忆库 → 相似案例
                   │
             人工确认/修订
                   │
         累计审核快照 + 新样本 + 旧样本回放
                   │
         批次更新分类头 → 独立评估 → 人工选择新版本
```

### L1：定位异常候选

- 主模型 `baseline`：已审核正常声音的 log-mel 中位数/MAD 参考分布，输出异常分数。
- 对照模型 `supervised`：默认 282 维声学特征 + NumPy 逻辑回归。特征包括当前 mel、谱描述量、上下文统计、多时间尺度左右差。
- 默认 L1 配置：44.1 kHz、64 mel、0.1 秒目标网格、逐文件归一化；实际配置写入模型。
- 只使用 train 拟合，val 选择阈值和后处理，test 仅做冻结评估。默认目标事件召回 0.95，不保证实际达到。
- 未审核区间不当正常，整段粗标不展开为逐帧真值。L1 分数不是故障概率。
- 已移除“拿当前待测音频安静部分直接拟合”的演示模式，以及 XGBoost/sklearn 自动回退。

### L2：多类型识别与增量学习

编码器采用官方 **32 kHz CNN14** 结构：1024 点 STFT、320 点 hop、64 mel；六个双卷积块的通道为 64/128/256/512/1024/2048，频率平均池化、时间 max+mean 池化、2048 维全连接。
STFT 和 mel 矩阵直接读取本地检查点里的冻结参数。AudioSet 的 527 类输出层仅为兼容权重保留，不拿它直接当风机故障分类器。

- 默认 4 秒窗口、2 秒步长；短事件读取原音频上下文，边界不足补零；长事件滑窗后平均，再进行 L2 范数归一化。
- 分类头为 `2048 → 1+K` 的线性层，独立 sigmoid；一段声音可同时属于多种类型。
- 掩码 BCE：只学习人工确认的正/负标签；未标注类型不自动变成负例。
- 首次训练使用全部 train。续训默认使用新增/修订的 train 样本和上一版记忆库；旧分类列继承，新类别在类别表末尾追加。
- 回放按类型保留代表样本，优先覆盖不同来源；默认每类 32 个，另保留正常及无明确类型的异常。记忆库和原型只来源于 train。
- 默认以 0.5 权重约束回放样本的旧分类输出，减轻遗忘；修订样本不套旧教师标签。训练采用 NumPy Adam，默认 400 轮。
- 类原型用累计 train 统计，阈值仅从 val 选择。每个输出在 train/val 中的正、负样本均至少覆盖两个独立来源，才启用其决策；这是最低检查门槛，不代表统计充分。
- 类型判定同时要求分类分数和原型相似度通过门槛；样本不足或不匹配输出 `unknown`。未知拒识是启发式方法，不能保证识别所有新型异常。
- 返回 `known_anomaly`、`likely_normal` 或 `unknown`，附相似案例；**均保留原 L1 告警供人工审核，不自动删除告警**。

当前只更新分类头、记忆库和原型，**不会更新 PANNs 主干参数**。每次审核可立即存档，但参数按批次更新；不是每审核一条就自动上线。主干微调、在线服务、自动模型发布不在本版范围内。

## 2. 项目布局

```text
wind_audio/
  cli.py                 # 统一命令入口、L1 工作流
  audio.py               # WAV、重采样与频谱
  data.py / splits.py    # 清单、标注、来源划分
  features.py            # L1 特征
  l1.py / detectors.py   # L1 模型
  backends.py            # NumPy 有监督对照
  calibrate.py / evaluate.py
  events.py              # 事件后处理
  review.py              # 人工审核、标签修订
  render.py / report.py   # L1 图表与报告
  l2/
    common.py            # JSON、音频/权重身份
    data.py              # 累计审核快照与标签掩码
    encoder.py           # PANNs CNN14 与特征缓存
    model.py             # 分类、回放、蒸馏、校准、评估
    cli.py               # L2 命令与两级推理
config/taxonomy.json      # 可编辑的异常类型配置
docs/ANNOTATION.md        # 数据和人工审核字段规范
tests/                   # 必要回归测试；夹具仅在临时目录生成
requirements.txt         # 运行依赖
pyproject.toml           # 标准 Python 打包与命令入口
.github/workflows/ci.yml  # 测试、静态检查、构建
```

`data/`、`weights/`、`models/`、`cache/`、`runs/`、`outputs/`、`reports/` 是本地数据/产物，不随源码发布。
现有真实录音、人工记录、历史真实模型保留；演示与合成产物不再放在主项目中。

## 3. 环境与依赖

使用你的环境：

```bash
conda activate post_training
cd /Users/shuoqinlin/Desktop/wind_audio
python -m wind_audio --help
```

`requirements.txt`：

```text
numpy>=1.24,<3
torch>=2.6,<3
```

本次没有安装任何库或下载权重。如需安装，由你手动执行：

```bash
python -m pip install -r requirements.txt
# 可选：安装项目命令 wind-audio；在项目目录中用 python -m wind_audio 则无需此步
python -m pip install -e . --no-deps
```

不依赖 xgboost、scikit-learn、torchaudio、librosa、torchlibrosa 或 panns-inference。
本机检查到 `post_training` 为 Python 3.11，NumPy 和 PyTorch 可用；当前无 CUDA/MPS，可先用 `--device cpu`。

请自行准备可信来源的官方 CNN14 检查点，放在例如 `weights/Cnn14_mAP=0.431.pth`。
只支持本地、32 kHz CNN14，支持官方 `{"model": state_dict}` 格式；不支持随意替换 CNN6、Wavegram 或不同前端配置。
程序不联网下载，加载使用 `weights_only=True`。更换权重、窗口或步长后须重提特征，不能直接继续旧模型训练。

上游：[PANNs 实现](https://github.com/qiuqiangkong/audioset_tagging_cnn) · [论文](https://arxiv.org/abs/1912.10211)。
上游权重及实现的使用条件按其原项目执行；本仓库尚未替你选择开源许可证。

## 4. 数据准备

先看 [标注规范](docs/ANNOTATION.md)。至少准备：

1. `data/manifest.csv`：音频 ID、路径、原录音/批次 `source_id`。
2. `data/annotations.csv`：L1 已审核正常/异常区间。
3. `data/splits.json`：按来源固定 train/val/test。
4. `config/taxonomy.json`：真实需要识别的声音类型。当前提供冲击、摩擦、窄带异音三个可修改配置项，不是已训练类别，也不表示故障原因。

同一原录音裁剪出的片段必须属于同一来源、同一集合。若评估跨设备泛化，用 `--group-by device`，并在清单中填写设备 ID。L2 还会检查同一音频内容不能跨集合。

```bash
python -m wind_audio check --manifest data/manifest.csv --annotations data/annotations.csv

# 首次建立划分，增量阶段不要重跑随机划分
python -m wind_audio split \
  --manifest data/manifest.csv --annotations data/annotations.csv \
  --group-by source --out data/splits.json
```

目前仓库中的真实数据只有一条约 30 秒、整段粗标异常的录音，没有足够独立来源、精确异常区间和已审核正常样本，**不能据此完成正式训练/校准/测试**。下列命令是在补齐数据之后的完整运行方式，不是已取得的模型成绩。

## 5. L1 训练、推理与审核

```bash
python -m wind_audio train \
  --manifest data/manifest.csv --annotations data/annotations.csv --splits data/splits.json \
  --detector baseline --version l1_v1 --out models/l1_v1 --target-recall 0.95

python -m wind_audio evaluate \
  --model models/l1_v1 --manifest data/manifest.csv --annotations data/annotations.csv \
  --splits data/splits.json --split test --out reports/l1_v1_test

python -m wind_audio run \
  --input data/anomaly/A01_01_20261030_120227.wav --model models/l1_v1 \
  --manifest data/manifest.csv --annotations data/annotations.csv --out runs/recording_v1
```

运行目录输出 `events.json`、`report.html`、图表、音频切片和 `review.csv`。
请每次使用新输出目录；已有审核 CSV 的目录不会被推理覆盖。
检查模型的 `calibrated` 和 `calibration.json` 的 `target_reached`；校准失败或召回目标未达到不能当作验收通过。

人工听取后填写 `review.csv`：

- `l1_label`：normal / anomaly / uncertain。
- `positive_types`：已确认类型 ID，多个用分号，例如 `impact;friction`。
- `negative_types`：明确排除的类型；不知道就留空。
- `types_complete`：只有完整审核当前类型表后才填 true，否则 false。
- `corrected_start_s/end_s`：必要时修正边界。
- `l2_label`、`fault_type`：保留人工故障结论，不由类型模型擅自填真值。

```bash
python -m wind_audio review-import \
  --review runs/recording_v1/review.csv --out data/reviews/reviews.csv
```

导入会记录审核时间，并在已有事件的训练标签改变时递增 `review_revision`。
模型预测不会自动转成人工标签。待审/不确定记录不进入训练。
`anomaly + nonfault` 依然是异常正例；只有人工确认 normal 才是正常样本。

## 6. L2 首次训练

先收集 train、val、test 都有覆盖的人工审核记录，再运行：

```bash
python -m wind_audio l2-prepare \
  --reviews data/reviews/reviews.csv --manifest data/manifest.csv --splits data/splits.json \
  --taxonomy config/taxonomy.json --out data/l2/v1.json

python -m wind_audio l2-check --dataset data/l2/v1.json

python -m wind_audio l2-embed \
  --dataset data/l2/v1.json --checkpoint weights/Cnn14_mAP=0.431.pth \
  --out cache/l2_v1.npz --device cpu

python -m wind_audio l2-train \
  --dataset data/l2/v1.json --features cache/l2_v1.npz \
  --out models/l2_v1 --version l2_v1

python -m wind_audio l2-evaluate \
  --model models/l2_v1 --dataset data/l2/v1.json --features cache/l2_v1.npz \
  --split test --out reports/l2_v1_test.json
```

模型目录保存 `meta.json`、`model.npz`、`validation.json`：类别顺序、编码器身份、分类参数、阈值、训练来源与回放样本均随版本记录。
数据快照、特征、评估文件和 L2 模型输出默认不覆盖旧文件；请选择新版本路径。

## 7. 增量学习与新增类型

1. 新事件先人工审核，合并进入累计 `reviews.csv`。
2. 新音频加入 manifest 和 splits；保持旧来源的集合归属，不把旧 val/test 移到 train。
3. 新类型只在 taxonomy 的末尾追加稳定 ID，不删除、不重排旧 ID。
4. 使用 `--previous` 保留旧审核当时覆盖的类别范围；旧记录的“完整审核”不会自动否定后来新增的类型。
5. 重新提取累计特征，再用 `--resume` 续训。

```bash
python -m wind_audio l2-prepare \
  --reviews data/reviews/reviews.csv --manifest data/manifest.csv --splits data/splits.json \
  --taxonomy config/taxonomy.json --previous data/l2/v1.json --out data/l2/v2.json

python -m wind_audio l2-embed \
  --dataset data/l2/v2.json --checkpoint weights/Cnn14_mAP=0.431.pth \
  --out cache/l2_v2.npz --device cpu

python -m wind_audio l2-train \
  --dataset data/l2/v2.json --features cache/l2_v2.npz \
  --resume models/l2_v1 --out models/l2_v2 --version l2_v2
```

本版要求保留累计数据快照和历史原音频，尚未实现仅依赖固定记忆库的流式存储。
旧训练标签可修订，但版本必须递增；已用于训练的事件边界变化、删除历史样本、改保留集标签，需要重建明确的数据基准并从头训练，不能静默续训。

新类型需要独立 val 正/负例，否则该输出保持 inactive/unknown。
在相同冻结保留集上比较旧类型 F1、异常保留率与新增类型识别；不能仅看总体准确率。
开发阶段优先比较 val，正式 test 不用于反复挑版本。上线由你通过 `--model` / `--l2-model` 显式指定，不自动替换旧模型。

## 8. 两级推理

```bash
python -m wind_audio pipeline \
  --input data/anomaly/A01_01_20261030_120227.wav \
  --l1-model models/l1_v1 --l2-model models/l2_v2 \
  --checkpoint weights/Cnn14_mAP=0.431.pth --device cpu --out runs/two_stage_v2
```

- `runs/two_stage_v2/l1/report.html`：L1 时间定位、波形、频谱与切片。
- `runs/two_stage_v2/events.json`：两级联合结果，每个事件的 `l2` 含状态、类型、分数、相似案例。
- `runs/two_stage_v2/l1/review.csv`：继续人工审核，预测不会代填真值。

对已经生成的 L1 结果也可单独运行：

```bash
python -m wind_audio l2-run \
  --events runs/recording_v1/events.json --model models/l2_v2 \
  --checkpoint weights/Cnn14_mAP=0.431.pth --device cpu \
  --out runs/recording_v1/events_l2.json
```

`run` 支持目录批量处理；`pipeline` 当前接收单条 WAV。联合结果目前为 JSON，HTML 仍是 L1 报告。

## 9. 已实现的对照实验

所有对照应共享来源划分、标签和评估口径。以下是可运行的配置，不是已测出的结论。

| 对照 | 设置 |
|---|---|
| L1 正常参考模型 | `train --detector baseline` |
| L1 有监督基线 | `train --detector supervised`，同一数据参数 |
| L2 静态模型 | 冻结首次训练的 `l2_v1`，不执行续训 |
| L2 朴素增量 | `l2-train --resume ... --no-replay --distill 0` |
| L2 回放、不蒸馏 | `l2-train --resume ... --distill 0` |
| L2 回放＋蒸馏 | 默认续训配置 |
| L2 累计全量重训 | `l2-train --resume ... --full-retrain`，分类头重新初始化 |

完整命令均需提供 dataset/features/version/out；每个实验使用独立输出目录。
“无回放”只消融分类头的历史样本训练，类原型仍从累计 train 统计。上述对照都冻结 PANNs，不包含主干微调对照。

L1 输出事件召回、精确率、每小时误报；L2 输出逐类型 P/R/F1、macro-F1、疑似正常拒绝率、建议异常保留率和未知拒识召回。
L2 指标只针对已审核候选，**不是端到端召回**。L1 没报出的异常不会自动进入 L2，必须定期抽查未告警音频并补充漏检标注。
未知拒识指标只覆盖已确认类型全部未被启用的异常；无类型真值时不能据此估计未知检出能力。

## 10. 验证、工程边界与发布

```bash
python -m unittest discover -s tests -t . -v
# 可选的大模型结构测试：只生成本地临时随机/零权重，不下载预训练权重
WIND_AUDIO_TEST_PANNS=1 python -m unittest tests.test_l2.PannsStructureTests -v
```

保留回归测试，移除了独立合成演示工程和产物。CI 使用 Python 3.11，执行测试、静态检查和打包。
提交前检查 `git status`；忽略规则不会自动清除过去已跟踪的数据。不要提交真实录音、审核隐私、训练权重或本地缓存。没有自动提交、推送或选择许可证。

仍需注意：

- 当前尚无真实 PANNs 预训练权重下的风机效果验证，也没有足够真实标注支撑性能结论。
- L1 评估在部分审核区间、跨 unknown 预测、极短事件和正常区误报时长上仍有旧版计量限制；正式验收优先使用完整精确审核录音，不能将现有报表当最终工业指标。
- L1 逐文件归一化可能削弱持续异常；完整录音上下文不是低延迟流式实现。
- 目前全文件读取、特征矩阵驻留内存，长录音/大数据需按独立录音合理分批，不能打乱来源分组。
- WAV 支持 8/16/32 位整数 PCM，不支持 24 位/浮点 WAV；多声道平均可能抵消反相信号。
- 增量学习不保证每次性能上升；新版本若旧类退化，应保留旧版并补数据/调整训练，而不是自动部署。
