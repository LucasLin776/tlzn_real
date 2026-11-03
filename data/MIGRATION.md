# 旧标签迁移说明

- 原文件：`data/labels.csv`（未修改）
- 新 manifest：`data/manifest.csv`
- 新区间标注：`data/annotations.csv`

## 规则

1. 旧 `labels.csv` 中 0–30s、备注“无秒级标注”的记录迁移为 **file 级粗标**，`fully_reviewed=false`，不生成逐帧/逐区间异常真值。
2. 只有明确写出的区间标注才保留为 interval；file 级标签只用于审计和追溯。
3. 未审核区域一律 unknown，不参与训练、阈值校准和评估。
4. 原 `data/labels.csv` 保持原样，作为历史记录。
