# 本地数据目录

此目录用于存放本地 WAV、标签、manifest 和分组文件。除本文外，默认不提交。

使用 `python -m wind_audio.synthetic --out data/synthetic_demo --seed 0` 可以重新生成演示数据。

真实数据按数据版本组织，例如 `data/project_v1/audio/` 与 `data/project_v1/labels.csv`。标注格式参考 `examples/labels.example.csv` 和 `docs/ANNOTATION.md`。

生产录音和审核结果可能包含敏感信息。已经写入 Git 的文件不会因忽略规则而自动取消跟踪，上传前请阅读 `docs/GITHUB.md`。
