# 12: docs/zh/examples/climax.md（中文文档）

**Type:** task
**Status:** resolved
**Blocked by:** 05

## What to build

ppsci 中文主文档就位：以 en 版信息为底稿、用 /humanizer-zh 重新措辞的自然中文（**不直译**）。

## 实施要点

- 先调 Skill 工具加载 humanizer-zh，按其去 AI 痕迹指南行文
- 信息与 en 版等价（命令、数字、口径一字不差），行文独立组织
- 中文技术文档惯例：公认英文缩写保留（CINN/RMSE/patch embedding 等），模型/论文术语用官方译名或保留原文，禁止自造缩写（项目术语规范）
- ppsci zh 文档语感参照 `docs/zh/examples/climateformer.md`（zh 是主文档，en 是镜像）

## Acceptance criteria

- [x] zh 独立成文、无翻译腔（humanizer-zh 检查项过）
- [x] 与 en 版信息一致（数字/命令逐项核对）

## Answer

已创建 /home/lkyu/baidu/PaddleScience/docs/zh/examples/climax.md（283 行），并在 mkdocs.yml 地球科学分类注册 `- ClimaX: examples/climax.md`。

结构照 climateformer.md 惯例：标题 → 数据/权重准备段（zenodo 7064308、regrid、xesmf conda-forge、historical 软链、requirements.txt、HF 权重转换说明）→ 命令 tab（训练/CINN/评估）→ `## 1. 背景简介` → `## 2. 模型原理`（arch.py 与 climatebench.py 的 --8<-- 嵌入）→ `## 3. 模型训练`（3.1 数据集、3.2.1~3.2.5 装配序列、3.3 评估）→ `## 4. 完整代码` → `## 5. 结果展示`（三组数字带口径）。

验收执行（真实命令，均通过）：
1. 独立成文：未读 en 版（票 11 产出前完成），按 canonical facts 独立组织行文；已加载 humanizer-zh 技能并按其清单自查——全文无"此外/然而/值得注意的是"类连接词、无夸大意义词、无三段式修辞、无粗体滥用、无模糊归因，句长错落，公认缩写（CINN/RMSE/ViT/patch embedding/checkpoint/token）保留原文。
2. 数字/命令逐项核对：脚本逐条 grep 34 项 canonical facts（arXiv 2301.10343、839 MB/74 MB、412 MB、2.24e-08、0.13394/0.13900、0.36180/0.36871、5.1%、5.2%、793 步、90 s、seed 42、三条命令等）全部命中，输出 ALL FACTS PRESENT。en/zh 两票共用同一 canonical facts 基准，en 版就位后的两文对照由终审票 15 执行。
3. 附加检查：22 处 --8<-- 切片的 linenums 与文件实际行号、区间边界校验 ALL OK；CRLF 计数 0、tab 计数 0。环境无 mkdocs，未跑全站 build，嵌入语法与 climateformer.md 逐字符同构。
