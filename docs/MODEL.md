# 模型产物与服务器加载

## 1. 目录结构

基线：

```text
models/l1_baseline/
  meta.json
  model.npz            # mu, sigma, score_mu, score_sigma
  calibration.json     # 若已校准
```

有监督：

```text
models/l1_supervised/
  meta.json
  model/
    model.json         # xgboost
    或 model.pkl       # sklearn
    或 model.npz       # numpy 逻辑回归
  calibration.json
```

## 2. `meta.json` 关键字段

| 字段 | 说明 |
|---|---|
| `schema_version` | 当前 `2.0` |
| `version` | 模型版本名 |
| `detector` | `baseline` / `supervised` |
| `score_type` | `baseline_z` / `probability_uncalibrated` |
| `threshold` | 实际生效阈值 |
| `postprocess` | `min_duration`, `merge_gap`, `low_ratio` |
| `calibrated` | 是否在 val 上校准过 |
| `demo` | 是否是演示拟合（不能用真值/评估） |
| `feature_config` | sr / n_fft / hop / n_mels / fmin / fmax / target_step / context_s / scales / normalize |
| `feature_fingerprint` | 特征配置哈希 |
| `feature_names` | 有监督特征名和顺序（推理时强校验） |
| `backend` | 请求/实际后端、是否回退、原因、依赖版本 |
| `training` | 训练文件、样本数、normal/anomaly 数、seed、上限 |
| `split` | train/val/test 的 audio_id 列表、seed、group_by |
| `calibration` | 目标召回、是否达到、val 指标、选中阈值/后处理 |
| `dependencies` | numpy/sklearn/xgboost 版本 |

## 3. 分数语义

- `baseline_z`：相对正常基线的标准化偏差，**不是概率**，不要用“90% 故障概率”描述。
- `probability_uncalibrated`：分类器原始输出，**未做概率校准**；只用于按阈值出事件。
- 阈值在验证集上按目标召回选取；报告和 `meta.json` 都记录阈值来源（model/calibration/cli）。

## 4. 加载与推理

```python
from wind_audio.audio import load_wav
from wind_audio.detectors import load_detector
from wind_audio.features import extract_features

det = load_detector("models/l1_supervised")
x, sr = load_wav("case.wav", target_sr=det.feature_config.sr)
bundle = extract_features(x, sr, det.feature_config, "case")
events = det.predict_events(bundle)          # 用模型阈值/后处理
events = det.predict_events(bundle, threshold=0.8)  # 显式覆盖
```

加载时：

- 校验 `feature_config` 与特征名/顺序；
- 校验后端依赖（xgboost/sklearn）；
- 不一致/缺依赖直接报错，不静默改配置。

## 5. 导出到服务器

```bash
# 本地训练目录整体拷贝即可（meta + model + calibration）
rsync -a models/l1_supervised/ server:/opt/wind_audio/models/l1_supervised/
```

服务器需要：

```bash
python3 -m venv /opt/wind_audio/.venv
/opt/wind_audio/.venv/bin/pip install numpy scikit-learn xgboost
```

- 首选 CPU；`xgboost` 的 `tree_method=hist`，不需要 CUDA。
- 如果服务器没有 xgboost，可先在本地用 `--algo sklearn` 训练并导出 sklearn 模型，服务器只装 sklearn；`meta.json` 会记录实际后端。
- 不建议在服务器用 `--algo numpy` 作为正式模型；它只是无依赖时的逻辑回归兜底。

## 6. 版本、回滚、再训练

建议每个版本一个独立目录，不要覆盖：

```text
models/
  l1_supervised_20261103/
  l1_supervised_20261201/
  current -> l1_supervised_20261201
```

回滚 = 把 `current` 指回旧目录；`meta.json` 自带特征配置、阈值、后处理，旧目录可独立加载。

服务器定期再训练时：

1. 用人工审核结果更新 `annotations.csv`（`l1_label=normal` 才是 normal 反馈；`anomaly+nonfault` 仍是 anomaly 正例）。
2. 按新录音/批次重新 `split`，不要随机切窗口。
3. 在 val 上校准阈值，冻结后跑 test。
4. 与当前线上模型对比事件召回、FA/h、覆盖率；达标再切换 `current`。
5. 保留旧版本用于回滚。

## 7. 已知限制

- 当前没有真实正常样本，真实模型未训练。
- XGBoost 路径未在本机执行（未安装）；sklearn/numpy 已验证。
- 逐文件中位归一化假设异常不占整条录音的绝大多数；持续异常占 >50% 时可能被部分吸收。
- 单次 `run` 对整条文件做 STFT；数小时录音建议先切段（保留 2×context 重叠）。
- 事件后处理会漏短事件/合并近邻事件，参数需按业务校准并在 `meta.json` 记录。
- L2 未训练：事件特征和音频引用已保存，但故障类型统一“待审核/未训练”。
