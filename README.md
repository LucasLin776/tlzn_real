# Wind Audio：风机音频两级检测

当前工程版本 **0.6.0**，L1 配置版本 **spectral_event_v2**。

| 层 | 输入与作用 | 输出 | 是否训练 |
|---|---|---|---|
| L1 | 原始 WAV → 比较相邻局部频谱 → 跟踪持续活动和衰减尾部 | 候选事件 `[start_s, end_s]`，只关心有没有事件及其范围 | 无需训练，使用规则和阈值 |
| L2 | 原始 WAV + L1 窗口 → 冻结 PANNs CNN14 特征 → 分类头 | 已知异常类型 / 疑似正常 / unknown、相似案例 | 需要人工审核数据；支持回放续训 |

L1 不区分胶粒掉落、启动、异响等类型，也不直接判断故障。L2 负责类型判断，
保留原 L1 告警；人工审核后更新分类头。L2 无法补回 L1 未提供的候选事件。

## 运行环境

```bash
conda activate post_training
cd /Users/shuoqinlin/Desktop/wind_audio
```

L1 只用 NumPy；L2 的 CNN14 还需 PyTorch。依赖见 `requirements.txt`。
从工程目录可直接运行 `python -m wind_audio`，无需安装本项目。
WAV 支持 8/16/32 位整数 PCM；24 位及浮点 WAV 需先转换。多声道平均为单声道。

## 用 B02 标注检验 L1

```bash
python -m wind_audio evaluate \
  --audio-dir /Users/shuoqinlin/Desktop/B02/音频 \
  --labels /Users/shuoqinlin/Desktop/B02/标注 \
  --pairing sorted --assume-unlabeled-normal \
  --config config/l1.json --out reports/l1_B02_new
```

本数据集已确认：B02-001～100 对应音频按时间排序；所有音频完整审核；
空标注和未标注部分为正常；全部标记均为 L1 目标事件，不区分名称。
`--pairing sorted` 仅用于已确认顺序对应的数据；默认按同名配对。
输出目录必须为空或不存在，避免覆盖既有标注和评估结果。

现有默认配置的 B02 结果在 [评估简报](reports/l1_B02/brief.md)：

| 指标 | 结果 |
|---|---:|
| 文件级准确率 | 96.00%（11 TP、85 TN、4 FP、0 FN） |
| 文件级召回率 / 精确率 | 100.00% / 73.33% |
| 按时长准确率 | 97.71% |
| 异常时长召回率 / 时长精确率 | 79.18% / 30.90% |
| 事件召回率 / 精确率 | 66.67% / 60.61%（IoU ≥ 0.1，一对一匹配） |

文件级全检出仅表示 11 条带标记录音均出现过告警；30 个事件中只有 20 个匹配成功。
正常时长占比很高，需结合异常召回率、误报时长与 F1 判断效果。
这是一批同一设备录音上的结果，不代表跨设备泛化性能；本次未调参。

[检验器实现步骤、标签格式与指标定义](docs/L1_EVALUATION.md) · [完整标注规范](docs/ANNOTATION.md)

检验器也支持简易 CSV/JSON（`file,label` 或 `file,start_s,end_s,label`）和现有清单：

```bash
python -m wind_audio evaluate \
  --audio-dir /path/to/audio --labels /path/to/labels.csv \
  --config config/l1.json --out reports/l1_check

python -m wind_audio evaluate \
  --manifest /path/to/manifest.csv --annotations /path/to/annotations.csv \
  --config config/l1.json --out reports/l1_check_manifest
```

## L1 检测与审核

```bash
python -m wind_audio run --input /path/to/audio \
  --config config/l1.json --out runs/l1_latest
```

`--input` 支持单条 WAV 或目录；目录输入生成总览和逐文件报告。
`events.json/csv` 保存检测窗口；`windows/` 是窗口音频，`clips/` 额外添加试听上下文；
`report.html` 供试听，`review.csv` 供人工填写。
`trigger_start_s/end_s` 是变化种子，L2 和标签比较使用完整窗口的 `start_s/end_s`。
`right_censored=true` 表示录音结束时尚未确认事件结束。

L1 默认 44.1 kHz、2048 点 FFT、64 Mel、20 ms 输出步长，比较前后各 0.1 秒频谱，
计算能量加权的 dB 变化。变化触发后冻结前 0.6 秒背景，持续跟踪正向谱差；
背景恢复或新工况稳定时结束窗口，过滤弱候选、合并短间隔并补边界。

| 参数 | 默认 | 作用 |
|---|---:|---|
| `--threshold` | 6 dB | 变化进入阈值，越高越不敏感 |
| `--release-db` | 3 dB | 持续活动释放阈值，越低尾部越长 |
| `--quiet-s` / `--settle-s` | 0.25 / 0.6 秒 | 背景恢复 / 新工况稳定确认时间 |
| `--min-prominence` | 0.3 | 与局部背景的差异比例下限 |
| `--relative-floor-db` | 15 dB | 相对整条录音强声参考的筛选范围 |
| `--merge-gap` / `--padding` | 0.35 / 0.05 秒 | 窗口合并间隔 / 每侧边界补充 |

完整参数见 `python -m wind_audio run --help` 和 `config/l1.json`。
分数不是概率；稳定异音、缓慢变化、短促弱事件和录音边缘可能漏检，工况切换可能误报。
全文件离线分析使用未来上下文，不是流式服务；20 ms 网格不等于真实边界精度。

人工填写 `l1_label`、修订边界及 `positive_types/negative_types` 后导入：

```bash
python -m wind_audio review-import \
  --review runs/l1_latest/review.csv --out data/reviews/reviews.csv
```

预测不能直接当人工真值；未审核部分不默认正常。还需抽查无告警录音以发现漏检。

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
  --l1-config config/l1.json --l2-model models/l2_v2 \
  --checkpoint weights/Cnn14_mAP=0.431.pth --device cpu --out runs/two_stage
```

## 工程布局与验证

```text
wind_audio/
  l1.py / features.py          局部频谱评分与特征
  windowing.py / detectors.py  持续活动跟踪与最终窗口
  evaluate.py / markers.py     准确率、事件指标、B02 标注导入
  data.py / splits.py          数据校验与来源划分
  audio.py / events.py         WAV 处理、事件特征
  cli.py                      统一命令入口
  review.py / report.py / render.py
  l2/                         PANNs、分类头、回放训练
config/                       当前 L1 参数与 L2 类型表
docs/                         标注规范、检验步骤、清理记录
tests/                        自动化测试；数据在临时目录生成并自动清除
reports/l1_B02/                本次最新 B02 结果，不复制原始音频
```

旧版代码、演示模型、旧测试音频和结果已移除，清单及恢复位置见 [清理记录](docs/CLEANUP.md)。
业务数据保留在外部 B02 目录；运行产物和本地权重默认忽略，不修改 Git 历史。

```bash
python -m unittest discover -s tests -t . -v
# 可选：需要本地 PyTorch，不下载权重
WIND_AUDIO_TEST_PANNS=1 python -m unittest tests.test_l2.PannsStructureTests -v
```
