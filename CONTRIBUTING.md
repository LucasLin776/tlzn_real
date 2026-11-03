# 贡献指南

## 开发环境

使用 Python >=3.10，在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Windows 使用 `.\.venv\Scripts\Activate.ps1` 激活。

## 修改约定

- 算法变更请提供最小复现和回归测试；不要只更新合成指标。
- 未审核、不确定区域不能作为正常真值；L1 异常与 L2 故障标签保持独立。
- 按原始录音/批次划分数据；test 不用于调参。
- 修改特征、模型格式或事件 ID 时说明兼容性，不能无提示复用旧缓存。
- 使用临时目录生成测试音频和模型，不依赖个人绝对路径或真实录音。
- 提交前检查差异，避免包含音频、模型、生产标签、缓存、访问凭证。

## 验证

```bash
python -m unittest discover -s tests -t . -v
python -m ruff check wind_audio tests
python -m build
python scripts/check_distribution.py dist
```

XGBoost 可选测试需另外安装 `python -m pip install '.[xgboost]'`。依赖不存在时的跳过不代表对应后端已经验证。

在 PR 中写明改动理由、执行过的检查、未执行项及剩余限制。当前未确定开源许可证；提交和分发之前先由仓库所有者确认授权范围。
