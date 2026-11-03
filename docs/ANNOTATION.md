# 数据与人工审核规范

## L1 文件

`manifest.csv` 表头：

```text
audio_id,path,source_id,device_id,recorded_at,duration_s,sample_rate,channels,fully_reviewed,notes
```

- audio_id 唯一；path 相对清单所在目录，不是工作目录。
- source_id 是原始录音/采集批次；同一原录音的所有裁剪片段必须同组。L2 要求显式填写。
- device_id 用于跨设备划分，不作为输入特征。
- fully_reviewed=true 仅限整段完整审核、所有异常都有精确标注。不要把“听了一段”当完整审核。

`annotations.csv` 表头：

```text
annotation_id,audio_id,start_s,end_s,label,precision,fault_type,note,reviewed_at,annotator,version
```

label 是 normal/anomaly/uncertain。precision=interval 才提供区间真值；file 是整段粗标，不自动展开。
没有标签的区间为 unknown。完整审核且异常为 interval 时才可把未标注区间推为 normal；
uncertain 始终未知。重叠冲突、倒序/越界区间和未知 audio_id 不允许。

## 审核记录（L2 训练输入）

先用 L1 `run` 生成 review.csv，再人工编辑，最后 `review-import` 导入累计文件。

| 字段 | 含义 |
|---|---|
| event_id / audio_id | 事件与原音频身份，保持稳定 |
| predicted_start_s / predicted_end_s | L1 候选边界 |
| corrected_start_s / corrected_end_s | 人工修订边界；空白沿用预测 |
| l1_label | normal / anomaly / uncertain / 空白 |
| positive_types | 已确认的声音类型 ID，分号分隔 |
| negative_types | 明确排除的类型 ID，分号分隔 |
| types_complete | true/false 或 1/0；是否已审查当前所有类型 |
| review_revision | 正整数，标签修订递增；导入同一事件时可自动递增 |
| reviewed_at | 审核时间，导入时可补齐 |
| l2_label | fault / nonfault / uncertain / not_trained，人工业务结论 |
| fault_type / note | 人工故障说明、备注；不自动映射成类型标签 |
| source_audio_path / clip_path / source_model_version | 追溯信息 |

训练标签规则：

- l1_label 为空或 uncertain：不进入 L2 数据。
- anomaly：二分类正例；多类型只使用 positive_types 和明确的 negative_types。
- normal：二分类负例，并否定审核时已注册的所有异常类型。
- types_complete=true：否定当前类型表中未出现的类型；只能在确实全部审核后使用。
- anomaly + nonfault 仍是异常，不能转为 normal。
- positive_types 与 negative_types 不能交叉；正常不能有异常正标签。
- 不从 fault_type 自动猜测声学类型；旧记录未填类型仍可用于二分类。
- 预测值不是人工标签；不得把所有 L2 预测直接导回训练真值。

## 新类别与历史数据

taxonomy 中 ID 用小写英文、数字、下划线，例 impact；name 是显示名。ID 不能改名或重排，新增类别末尾追加。
续训准备数据时传 `l2-prepare --previous 上一版快照`，保留旧审核的类型覆盖范围。
旧样本对新类型默认未知；如果重新审核了新类型，更新标签并递增 review_revision。

同一原音频内容和区间跨检测版本会去重；不同模型产生冲突结论时须先人工合并。
事件 ID 不同但引用同一音频区间时，不把它们当独立训练样本。
L2 的 sample_id 由音频内容和区间确定，不依赖检测模型。

累计快照保存固定的 train/val/test。已训练样本不能换集合；同源音频不能跨集合。
新数据追加到与来源一致的集合。已用于模型的边界/保留集标签需要修改时，重新建立有版本的数据基准并全量训练。

## 人工抽查漏检

除审核 L1 告警外，也要抽查无告警音频。确认的正常/异常区间可手工补充审核 CSV，
赋予唯一 event_id，引用真实 audio_id、精确时间和人工标签，再统一导入。
不能只审核告警后宣称系统整体召回率已经提高。
