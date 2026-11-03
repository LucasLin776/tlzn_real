# 验证记录

日期：2026-11-03 ｜ 项目：`/Users/shuoqinlin/Desktop/wind_audio`

## 1. 环境

- 默认解释器：Python 3.14.7，numpy 2.5.3，**无 sklearn/xgboost**（本机无法联网安装）。
- 备用环境：`~/miniconda3/envs/rag_learn/bin/python`，Python 3.12，numpy 2.5.3，scipy 1.18.1，sklearn 1.9.1，**无 xgboost**。
- 因此：
  - `baseline` 在默认解释器验证；
  - `supervised` 在 rag_learn 环境用 sklearn 后端验证；
  - **XGBoost 路径未执行**，只验证了缺依赖时的显式报错和回退记录。

## 2. 自动化测试

```bash
python3 -m unittest discover -s tests -t .
~/miniconda3/envs/rag_learn/bin/python -m unittest discover -s tests -t .
```

结果：

```text
默认 python3: Ran 30 tests — OK (skipped=1)
rag_learn:    Ran 30 tests — OK
```

覆盖：

- 标签掩码：fully_reviewed 推导 normal、uncertain 保持 unknown、file 级粗标不展开
- 校验：越界、顺序、冲突、未知 audio_id、fully_reviewed + 仅 file 级 anomaly
- 划分：按 source/device 分组无泄漏、确定性、分组不足报错
- 事件：滞回、合并、最短时长、首尾边界
- 匹配：一对一、巨大预测不能命中所有 GT、重复预测不重复计数
- 模型：baseline/supervised 保存加载一致、阈值覆盖、特征配置不匹配报错、缺 XGBoost 显式回退
- 审核：anomaly+nonfault 仍是 L1 正例、normal 才是负例、uncertain 跳过、CSV 导入导出
- 特征：增益不变性、短音频、静音、特征名与矩阵维度一致
- 最简导入：只有音频+标注时自动读 WAV 头、reviewed-list、未标注区域 unknown
- 合成全链路：check → split → train → calibrate → evaluate → run（baseline 与 sklearn supervised）

## 3. 合成数据全链路

```bash
python3 -m wind_audio.synthetic --out data/synthetic --seed 0
python3 -m wind_audio check --manifest data/synthetic/manifest.csv \
  --annotations data/synthetic/annotations.csv
python3 -m wind_audio split --manifest data/synthetic/manifest.csv \
  --annotations data/synthetic/annotations.csv --out data/synthetic/splits.json --seed 7
```

合成集：12 条 12 s，正常 131.2 s，异常 12.8 s，9 个异常事件；train/val/test = 7/2/3 文件。

训练/评估：

```bash
python3 -m wind_audio train --manifest data/synthetic/manifest.csv \
  --annotations data/synthetic/annotations.csv --splits data/synthetic/splits.json \
  --detector baseline --out models/syn_baseline --version syn_baseline --target-recall 0.9
python3 -m wind_audio evaluate --model models/syn_baseline \
  --manifest data/synthetic/manifest.csv --annotations data/synthetic/annotations.csv \
  --splits data/synthetic/splits.json --split test --out reports/syn_baseline_test

~/miniconda3/envs/rag_learn/bin/python -m wind_audio train \
  --manifest data/synthetic/manifest.csv --annotations data/synthetic/annotations.csv \
  --splits data/synthetic/splits.json --detector supervised --algo sklearn \
  --out models/syn_supervised --version syn_supervised --target-recall 0.9
~/miniconda3/envs/rag_learn/bin/python -m wind_audio evaluate \
  --model models/syn_supervised --manifest data/synthetic/manifest.csv \
  --annotations data/synthetic/annotations.csv --splits data/synthetic/splits.json \
  --split test --out reports/syn_supervised_test
```

测试集结果（**仅合成逻辑验证**）：

| 模型 | 召回 | 精确率 | FA/h | 覆盖率 | 边界误差中位 |
|---|---|---|---|---|---|
| baseline | 1.000 | 1.000 | 0.000 | 0.952 | 起点 0.00 s / 终点 0.01 s |
| supervised（sklearn） | 1.000 | 1.000 | 0.000 | 0.952 | 起点 0.00 s / 终点 0.01 s |

测试集为 35.6 s 已审核正常、0.4 s 已审核异常、2 个 GT 事件、0 s 未审核。

## 4. 真实音频

```bash
python3 -m wind_audio probe --input data/anomaly/A01_01_20261030_120227.wav
python3 -m wind_audio migrate-labels --labels data/labels.csv --audio-dir data/anomaly \
  --manifest data/manifest.csv --annotations data/annotations.csv
python3 -m wind_audio check --manifest data/manifest.csv --annotations data/annotations.csv
python3 -m wind_audio run --input data/anomaly/A01_01_20261030_120227.wav \
  --model models/l1_demo --manifest data/manifest.csv --annotations data/annotations.csv \
  --out runs/real_final
```

结果：

- `probe`：`duration_s=30.002721`，44100 Hz，单声道，峰值 0.4898，RMS −28.2 dBFS。
- `check`：1 文件，已审核正常 0 s，已审核异常 0 s，未审核 30 s；0 个区间事件；1 个粗标文件；明确列出不能训练/评估的原因。
- `run`：旧 `l1_demo` 演示模型输出 9 个事件；报告标记“演示模型”，区域全部为“未审核”。
- **没有真实召回率/误报率**：没有正常样本，异常只有 file 级粗标，没有秒级真值。

## 5. 审核接口

```bash
python3 -m wind_audio review-export --events runs/real_final/events.json \
  --out runs/real_final/review.csv
python3 -m wind_audio review-import --review runs/real_final/review.csv \
  --out data/reviews/reviews.csv --json data/reviews/reviews.json \
  --emit-annotations data/reviews/annotations_from_review.csv
```

已验证：`anomaly + nonfault` 生成 anomaly 正例；`normal` 生成 normal；`uncertain` 不生成；导入后可再次读取并合并。

## 6. 未执行/未声称

- 未执行 XGBoost 训练/推理（未安装，且本机无法联网安装）。
- 未在真实风机数据上训练或评估。
- 未实现 L2 故障分类器（本轮范围只做审核数据接口与特征保留）。
- 未实现服务器平台、在线标注前端、自动增量训练调度。
- 合成指标不能解释为真实风机性能。
