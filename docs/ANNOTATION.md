# 标注规范

## 1. 两个文件

### `manifest.csv`（文件级元数据）

```text
audio_id,path,source_id,device_id,recorded_at,duration_s,sample_rate,channels,fully_reviewed,notes
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `audio_id` | 是 | 唯一 ID，事件/审核/报告都用它关联 |
| `path` | 是 | WAV 路径，**相对 manifest 文件所在目录**；也可用绝对路径 |
| `source_id` | 建议 | 原始录音/采集批次；默认按它分组划分 |
| `device_id` | 可选 | 风机/机组编号；只用于划分、跨设备评估和追溯 |
| `recorded_at` | 可选 | 采集时间；只用于划分和追溯 |
| `duration_s` | 建议 | 可留空，校验时用 WAV 头补齐；不一致以 WAV 为准并警告 |
| `sample_rate` / `channels` | 可选 | 校验时补齐 |
| `fully_reviewed` | 是 | `true` 仅当整段已完整审核、且所有异常都已标出 |
| `notes` | 可选 | 备注 |

`device_id`、`recorded_at` **不会进入模型特征**。

### `annotations.csv`（区间标注）

```text
annotation_id,audio_id,start_s,end_s,label,precision,fault_type,note,reviewed_at,annotator,version
```

| 字段 | 说明 |
|---|---|
| `label` | `normal` / `anomaly` / `uncertain` |
| `precision` | `file`（整段粗标）或 `interval`（精确区间） |
| `fault_type` | 可选；没有 L2 标签时留空，不要猜 |
| `reviewed_at` / `annotator` / `version` | 可选，用于审核追溯 |

当前主格式是 CSV；人工审核记录同时支持 CSV 和 JSON（`review-export` / `review-import`）。

## 2. 硬规则

1. **未审核 = unknown**。没有 normal 标注、也没有被 `fully_reviewed` 覆盖的区域，不能当正常。
2. **`uncertain` 永远是 unknown**，即使文件 `fully_reviewed=true`。
3. **只有 `fully_reviewed=true` 且异常是 interval 标注**，异常区间以外才能推导为 normal。
4. **file 级粗标不展开**。`precision=file` 只记录“整段被标成 anomaly/normal/uncertain”，不产生逐帧/逐区间真值。
5. `fully_reviewed=true` + 只有 file 级 anomaly 会被校验拒绝；必须补 interval 或把 `fully_reviewed` 改成 false。
6. 校验会拒绝：`start_s < 0`、`end_s <= start_s`、`end_s > duration`、未知 `audio_id`、重复 ID、不同 label 的区间重叠、缺失音频。
7. 同一 label 的重叠区间允许，评估时按合并后的区间处理。

## 3. 标签掩码怎么生成

目标网格默认 0.1 s：

```text
times[i] = (i + 0.5) * 0.1
```

- 先把 interval 标注画到网格：`normal→0`、`anomaly→1`、`uncertain→-1`。
- 若 `fully_reviewed=true`，把**没有被任何标注覆盖、也不是 uncertain** 的格子填成 0（normal）。
- 否则未覆盖格子保持 -1（unknown）。
- 文件级标注不画格子，只设置 `coarse=True`。

推理/训练时，目标时刻 `t` 的标签取它所在网格单元；上下文窗口只用于提特征，不会因为窗里沾到一点异常就把整窗标成异常。

## 4. 审核 CSV

```text
event_id,audio_id,source_audio_path,predicted_start_s,predicted_end_s,
corrected_start_s,corrected_end_s,l1_label,l2_label,fault_type,note,
reviewed_at,source_model_version,clip_path
```

| 字段 | 取值 |
|---|---|
| `l1_label` | `normal` / `anomaly` / `uncertain` / 空白（待审） |
| `l2_label` | `fault` / `nonfault` / `uncertain` / `not_trained` / 空白 |
| `corrected_start_s` / `corrected_end_s` | 人工修正区间；留空则用预测区间 |
| `fault_type` | 只有 `l2_label=fault` 时有意义 |

语义：

- `l1_label=anomaly` + `l2_label=nonfault`：**L1 正例**（异常，但不是故障）。
- `l1_label=normal`：确认是正常误报，才能作为 L1 normal 反馈。
- `l1_label=uncertain` 或空白：不进入训练/评估。
- 当前无 L2 模型，`l2_label` 默认 `not_trained`；报告显示“待审核/未训练”。

## 5. 命令

```bash
# 检查
python3 -m wind_audio check --manifest data/manifest.csv --annotations data/annotations.csv

# 导出审核模板
python3 -m wind_audio review-export --events runs/case/events.json --out runs/case/review.csv

# 导入（校验 + 合并 + 可选生成 L1 反馈区间）
python3 -m wind_audio review-import \
  --review runs/case/review.csv \
  --out data/reviews/reviews.csv \
  --json data/reviews/reviews.json \
  --emit-annotations data/reviews/annotations_from_review.csv
```

`--emit-annotations` 生成的是**独立反馈文件**，不会自动改 `data/annotations.csv`；确认后再人工合并，避免审核记录和原始真值混在一起。
