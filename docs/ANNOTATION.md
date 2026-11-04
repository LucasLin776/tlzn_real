# 数据与人工审核规范

## L1 检验数据（不是训练数据）

L1 直接检测频谱波动，不拟合正常/异常分类器，也不需要训练集。
这些文件用于检验预测窗口与人工标注是否匹配；单条录音也能运行和检验。
同一批数据调过阈值后，只能视为开发集结果；正式效果应在未参与调参的录音上检验。


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

label 是 normal/anomaly/uncertain。precision=interval 才提供区间真值；file 是整段粗标，
只用于文件级准确率，不自动展开为区间真值。冲突的文件标签或 file normal 与异常区间不能并存。
没有标签的区间为 unknown。完整审核且异常为 interval 时才可把未标注区间推为 normal；
uncertain 始终未知。重叠冲突、倒序/越界区间和未知 audio_id 不允许。

也可直接传入简易 CSV/JSON 或 B02 的 Name/Start/Duration 标注目录，
由检验器生成标准清单；格式、配对规则和准确率口径见 [L1 检验器](L1_EVALUATION.md)。

## 审核记录（L2 训练输入）

先用 L1 `run` 生成 review.csv，再人工编辑，最后 `review-import` 导入累计文件。

`predicted_start_s/end_s` 是跟踪持续段与尾部、再补充边界的检测窗口。
`triggers` 列出窗口内触发种子；`trigger_start_s/end_s` 只是首尾触发范围，不能当完整异常边界。
`right_censored=true` 表示文件结束时仍未确认事件结束，不代表真实事件恰好在文件尾停止。
被过滤候选在 `events.json → diagnostics.rejected_candidates` 中供诊断，不自动标为正常。
`windows/*.wav` 是检测窗口本身；`clips/*.wav` 为额外试听上下文，不能把整个试听切片当异常标注。
L2 使用原音频和检测窗口（审核训练时优先使用人工修订边界），不把切片内的相对时间误当原音频时间。

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
