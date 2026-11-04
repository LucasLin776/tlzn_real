# 工程清理记录

日期：2026-11-04。

## 保留

- 当前 L1 `spectral_event_v2` 实现及 `config/l1.json`。
- 当前 L2 的 PANNs 编码器、分类头、回放训练逻辑及 `config/taxonomy.json`。
- 最新 B02 评估：`reports/l1_B02/`，只保存标签快照与结果，不复制 WAV。
- 自动化测试源码；运行中生成的临时音频与结果自动清除。
- 用户原始 B02 音频/标注位于 `/Users/shuoqinlin/Desktop/B02`，未修改。

工程内没有已训练的 L2 权重、累计审核训练集或历史 L2 模型，因此没有删减 L2 训练资产。

## 已移除

- `backup/v1/`：历史实现与旧标签。
- `models/l1_demo/`：已停用的旧 L1 演示模型；当前 L1 不需要训练模型。
- `outputs/A01_01_demo/`、`outputs/A01_01_model/`：旧版测试报告。
- `runs/real_final/`、`runs/l1_spectral_v04_20261103/`、
  `runs/l1_events_v05_original_20261103/`、`runs/l1_events_v05_batch_20261103/`：旧测试音频副本、切片和结果。
  清理前核对所有 review.csv，均无人工审核标签或审核时间。
- `reports/VALIDATION.md`：引用已停用 baseline/supervised 流程的旧验证结果。
- `data/anomaly/`、旧 `data/labels.csv`、`data/manifest.csv`、`data/annotations.csv`：单条 A01 演示数据和粗标签。
- 空目录 `cache/`、`data/normal/`，以及清理后为空的 backup/models/outputs/runs。
- 旧 `migrate-labels` 命令及迁移实现；当前使用统一 `import-simple` 或 `evaluate` 导入。

移出工程的旧资料约 60 MB，源码仍由 Git 历史保存；未修改 Git 历史，也未提交或推送。

## 恢复

清理前已归档并检查可读取：

`/private/tmp/wind_audio_cleanup_20261104.A4wtJZ/legacy.tar.gz`

压缩包约 50 MB，包含上述被删除的旧资料。它位于系统临时目录，可能被系统清理；
如需长期保留请另行存放。已跟踪文件还可从清理前的 Git 版本恢复。
建议解压到新的目录核对，避免覆盖当前工程。
