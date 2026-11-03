# 旧标签迁移说明

## 原文件

`data/labels.csv`（**未修改**）：

```csv
file,label,fault_type,start_s,end_s,note
A01_01_20261030_120227.wav,anomaly,unknown,0,30,用户确认整段为异常；含多次宽带爆发；无秒级标注
```

这条记录表示“用户确认整段是异常”，但**没有指出异常的具体起止秒**。

## 为什么不能展开成 0–30 s 全异常

- `start=0, end=30` 是文件范围，不是人工标注的异常区间。
- 备注明确写了“无秒级标注”。
- 把整段 30 s 当逐帧 anomaly，会制造 300 个假正样本，并让模型把正常部分也学成异常。
- 把整段当 anomaly 还会让事件匹配、边界误差、覆盖率全部失真。

所以迁移只保留**文件级粗标**。

## 迁移结果

命令：

```bash
python3 -m wind_audio migrate-labels \
  --labels data/labels.csv \
  --audio-dir data/anomaly \
  --manifest data/manifest.csv \
  --annotations data/annotations.csv
```

产物：

`data/manifest.csv`

```csv
audio_id,path,source_id,device_id,recorded_at,duration_s,sample_rate,channels,fully_reviewed,notes
A01_01_20261030_120227,anomaly/A01_01_20261030_120227.wav,A01_01_20261030_120227,,,30.002721,44100,1,false,迁移自旧 labels.csv；未声明整段完整审核
```

`data/annotations.csv`

```csv
annotation_id,audio_id,start_s,end_s,label,precision,fault_type,note,reviewed_at,annotator,version
legacy_0001,A01_01_20261030_120227,0.0,30.002721,anomaly,file,unknown,用户确认整段为异常；含多次宽带爆发；无秒级标注,,,legacy
```

`fully_reviewed=false` + `precision=file` 的含义：

- 这文件是“已知异常文件”，用于追溯和挑选；
- 它不是训练/校准/评估的正样本；
- 30 s 全部是 unknown；
- `check` 会显示 `已审核异常 0.0s / 未审核 30.0s`，并说明不能训练/评估。

## 后续怎么补成真正可训练的数据

1. 找人听/看时频图，标出每个异常事件的 `start_s/end_s`，写成 `precision=interval` 的 `anomaly`。
2. 确认整段其余部分都正常后，把该文件 `fully_reviewed` 改为 `true`；此时区间外会自动推导 normal。
3. 补充正常录音（建议至少 20–30 段 30 s，覆盖不同时间/风速/机组），以及更多异常事件。
4. 运行 `check`，直到 normal/异常时长和事件数满足训练门槛。
5. 重新 `split`（按录音/批次/设备）后训练；不要复用同一录音的相邻切片做测试。

## 注意

- 不要根据“声音大”“模型预测”“备注里的用户确认”自动补造区间真值。
- 现有 `data/labels.csv` 是历史记录，保留不动；新数据写 `manifest.csv` / `annotations.csv`。
- `data/annotations_from_review.csv` 是审核反馈文件，合并进正式标注前要人工确认。
