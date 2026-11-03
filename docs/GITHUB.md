# 上传 GitHub 前的检查清单

本指南不会自动提交、推送、取消跟踪文件或重写历史。以下命令均需你在仓库根目录确认后自行执行。

## 1. 确认哪些内容可以共享

建议共享：源代码、测试、文档、无真实数据的示例、依赖与 CI 配置。

默认不共享：真实或合成 WAV、生产标签、模型权重、特征缓存、试听报告、审核记录、历史备份、个人编辑器配置和凭证。合成音频可用代码重新生成。

## 2. 检查已经被跟踪的文件

```bash
git status --short
git ls-files data models runs cache outputs reports backup .vscode
git ls-files '*.pyc' '*.wav' '*.npz' '*.pkl'
git ls-files -ci --exclude-standard
```

最后一条列出“已跟踪但符合忽略规则”的文件。**增加 `.gitignore` 不会让这些文件自动退出 Git。** 此项目本次整理时已存在被跟踪的音频、模型和缓存，请先检查，不要直接推送。

如果决定让某个文件仅保留本地，可针对核对过的单个路径执行 `git rm --cached -- 路径`；这只改变后续提交的跟踪状态，不删除工作区文件。目录需要明确审核后再处理，不要批量清空整个索引。

已进入历史的文件仍可从旧提交访问。若含机密数据，需另行制定历史清理或建立干净发布仓库的方案；不要擅自 force push。`git rm --cached` 和 `.gitignore` 都不能清除历史。

## 3. 先验证可安装和可复现

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -t . -v
python -m ruff check wind_audio tests
python -m build
python scripts/check_distribution.py dist
```

发布包检查用于确认 wheel / sdist 不携带数据和权重；它不是源码秘密扫描器，也不会检查 Git 历史。使用 README 的合成快速开始验证安装，不要依赖本机已有音频。

## 4. 审阅并提交标准项目文件

解决上面的已有跟踪问题后，可以显式暂存：

```bash
git add README.md pyproject.toml requirements.txt requirements-dev.txt MANIFEST.in
git add .gitignore .gitattributes .editorconfig .github
git add CONTRIBUTING.md SECURITY.md CHANGELOG.md
git add wind_audio tests docs examples scripts data/README.md
git diff --cached --stat
git diff --cached --check
git status --short
```

在提交前再次检查音频、权重、标签和绝对路径。确认无误后自行创建提交，并核对远程地址与仓库可见性后再推送。

## 5. 许可证

GitHub 不强制必须有 LICENSE。当前没有指定开源许可证，本次没有添加默认 MIT/Apache 授权。公开前请确认代码、音频和第三方依赖的权利范围，再选择适合的许可证。

## 6. 不应作出的声明

- 不要把合成集高召回称为真实风机性能。
- 不要写“生产可用”“自动故障诊断”或“所有异常都能检测”。
- 不要使用未实际跑过的 CI 通过徽章或后端测试结论。
