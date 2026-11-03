# 本地数据目录

保留真实 WAV、文件清单、精确标注、固定来源划分和累计人工审核。
除本文外，默认不纳入新的源码提交；忽略规则不会取消已有 Git 跟踪。

- manifest.csv：音频路径相对本文件所在目录，source_id 表示原始录音/批次。
- annotations.csv：已审核正常/异常区间；未审核不当正常。
- splits.json：固定 train/val/test，增量阶段保持旧来源归属。
- reviews/reviews.csv：人工审核主表。
- l2/v*.json：不可覆盖的累计审核快照。

格式见 [标注规范](../docs/ANNOTATION.md)，命令见 [README](../README.md)。
本目录不再提供合成演示数据。请勿把真实录音和审核隐私上传到公开仓库。
