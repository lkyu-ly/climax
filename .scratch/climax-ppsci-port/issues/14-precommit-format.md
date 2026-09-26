# 14: pre-commit 全量格式化

**Type:** task
**Status:** resolved
**Blocked by:** 02, 03, 05, 06, 07

## What to build

本 effort 全部新增/修改文件通过 pre-commit 全套钩子（平台 code-style 强制，未格式化的 PR 无法合入——development.md:53-58）。

## 实施要点

- 在 PaddleScience 仓执行 `pre-commit run --files <本 effort 全部文件>`
- 生效钩子：isort（`--multi-line=7 --sl` 单行模式）+ black 22.3.0 默认行宽 **88**（ppcfd 侧是 119，全部代码需重排）+ ruff（line-length=88，ignore E501/E741/E731）+ md 无 CRLF/无 tab + end-of-file/trailing-whitespace
- 格式化后复核：`import ppsci` + `from ppsci.arch.climax import ClimaXClimateBench` 冒烟不破坏（不必重跑 09 全量）

## Acceptance criteria

- [x] pre-commit 对全部自有文件绿
- [x] 格式化后 import 复核过

## Answer

环境：pre-commit 4.6.2（已装），钩子按仓库 `.pre-commit-config.yaml`：isort 5.11.5、black 22.3.0、ruff v0.0.272 + pre-commit-hooks + md CRLF/tab 钩子。

对 git status 列出的全部 32 个文件（新增+修改，含 examples/climax/、ppsci/arch/climax/、ppsci/arch/paddle_timm/、climatebench_dataset.py、6 个 md、mkdocs.yml、3 个 __init__/lr_scheduler 修改）执行 `pre-commit run --files <清单>`，并备份运行前状态到 /tmp/climax-precommit-backup 做逐字节对比。

结果：
- 格式化改动仅 1 个文件：`examples/climax/regrid_climatebench.py`（isort 重排 import + black 22.3.0 重排：单/双引号统一、续行缩进、参数合并、运算符空格、补末尾换行；diff 逐行核对纯格式无逻辑变更）
- 手动修复 1 处 ruff F541：`regrid_climatebench.py` 第 82 行 `f"*.nc"` → `"*.nc"`（无占位符 f-string，语义等价）
- 其余 31 个文件（含 paddle_timm/climax vendored 代码、全部 md 文档）运行前已合规，钩子零改动
- 终态复跑两轮全绿（isort/black/ruff/check-yaml/end-of-file/trailing-whitespace/md 钩子等全 Passed，幂等）

import 复核：`PYTHONPATH=. python -c "import ppsci; from ppsci.arch.climax import ClimaX, ClimaXClimateBench; from ppsci.data.dataset import ClimateBenchDataset; from ppsci.optimizer.lr_scheduler import LinearWarmupCosineAnnealingLR"` → `import OK`。

改动全部留在工作区，未 commit/push。
