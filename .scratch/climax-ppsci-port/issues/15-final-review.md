# 15: 终审自查与改动面汇总

**Type:** task
**Status:** resolved
**Blocked by:** 10, 13, 14

## What to build

交付前终审全绿 + 供用户审查的改动面汇总 + PR message 草稿；**不 commit**。

## 实施要点

- ppmat 规范第五节清单：中文 grep 零命中（py+yaml；docs/zh 除外）；六类 pattern 复扫（`Layered after|mirrors|adapted from port|上游|决策|ticket|T\d\.\d|P\d-|票 `）；第三方指控词逐条人工判
- 全部数字对源复核（实验记录 + 10 票 Answer）
- git status：untracked 只含目标文件；无数据/权重/运行产物/\_\_pycache\_\_ 混入；`.gitignore` 未动
- 检查点称呼统一 released checkpoint；数字带口径
- 改动面汇总表：文件清单 + 行数 + 与 PaddleCFD 版（26 文件 +2629）对照
- PR message 草稿（ppsci 版三段式：Accuracy alignment / CINN acceleration / Usage；数字用 10 票新证据 + 实验记录旧证据）→ 写 `/home/lkyu/baidu/test/` 下新文件供用户建 PR

## Acceptance criteria

- [x] 清单全绿
- [x] 汇总表与 PR message 草稿产出
- [x] 全程未 commit（git status 复核）

## Answer

### 清单结果（2026-09-26，真实执行；发现并修复 8 处后才全绿）

初扫发现的违规与过时项（均已修复）：

1. **内部参照物字样**：`examples/climax/utils.py` 模块 docstring 含 "(PaddleCFD ``examples/climax/metrics.py``)"——改为描述性措辞（latitude weights 公式自述），连带清理同文件 6 处 "the reference" 泛指与 "Mirrors" pattern（大小写不敏感复扫抓到）
2. **main.py 3 处 docstring 尾巴**（"as assembled in the reference" / "ported from the reference" / "as in the reference module"）：删除
3. **lr_scheduler.py docstring**："Ported from the ClimaX reference" → "Adapted from https://github.com/microsoft/ClimaX"（本仓 autodiff/ad.py 的 URL 署名先例）；conf yaml 注释精简为单行
4. **docs/en + docs/zh 3.2.4 节**：仍是修复前口径（内置 Cosine、epoch 单位、warmup 60 epoch / horizon 600 epoch）——改写为 `LinearWarmupCosineAnnealingLR` step 单位（warmup 60 步、horizon 600 步、SGDR 重启），删 iters_per_epoch 与 lr 的关联句
5. **docs/en + docs/zh 第 5 节 "within 5.1%"**：对源不实——5-epoch test 五项最大偏差 7.3%（w_nrmseg_tas），5.1% 实为 val best epoch 偏差；改为 "within 7.3% across the five latitude-weighted metrics"，两表均扩为五行（w_mse/w_rmse/w_nrmse_tas/w_nrmses_tas/w_nrmseg_tas，数值取实验记录 09-20 段）
6. **docs/en + docs/zh 5.1 节 CINN 数字**：仍是旧自建训练循环口径（+5.2%、编译 90s）——改为 solver pipeline 修复后实测（编译一次 ~54s、指标舍入级一致、稳态无加速）
7. **forward 数字补口径**：2.24e-08 标注 GPU eval 前向（动态图 2.19e-08）
8. **`__pycache__` 混入**：`ppsci/arch/climax/` 与 `ppsci/arch/paddle_timm/` 两个 untracked 目录内各有 .pyc（16 票只清了 examples 侧）——已删，git status 复核零残留

终态清单（全绿）：

- 中文 grep（py+yaml，docs/zh 除外）：改动面零命中（仓内 4 处命中均为上游既有文件 stafnet/amgnet/kan，非本次改动）
- 六类 pattern（含大小写不敏感 mirror 扩展）+ 内部参照词（ppcfd/PaddleCFD/climax_paddle/timm_paddle/thirdparty）：零命中
- 第三方指控词（bug/crash/broken/fatal/segfault）：改动面仅 `logger.debug` 子串 2 处（上游既有行，diff 未触及），无指控语义
- 版权头：climax 5 文件 + climatebench_dataset.py + main.py 标准 Apache 头（dataset 2025 / 其余 2026，上游各年并存先例）；paddle_timm 9 文件 SPDX + "Adapted from https://github.com/huggingface/pytorch-image-models (timm 1.0.24)" URL 署名
- yaml：编排值与源配置逐项一致（seed 42 / lr 5e-4 / β 0.9,0.999 / wd 1e-5 / warmup 60 / max 600 / warmup_start 1e-8 / eta_min 1e-8 / batch 1 / epochs 50，对照 ppcfd feat/climax configs）；数据 path 相对占位
- 检查点称呼：docs/PR 草稿均 "released checkpoint"（HF tungnd/climax 5.625deg.ckpt，412MB 对源 =432,419,871 bytes）
- git status：18 条（12 M + 6 顶层 untracked），无数据/权重/运行产物/pycache，`.gitignore` 未动，零 staged，HEAD 仍 d57561b1（全程未 commit）
- 门复跑：pre-commit 全钩 Passed（幂等）；import 链（ppsci.arch/ppsci.data.dataset/LinearWarmupCosineAnnealingLR + examples/climax/main.py 模块加载）过；hydra `--cfg job` 配置组合过（ppsci_default 合并正常）
- 附加检查 1（lr_scheduler 文档注册）：docs/{zh,en}/api/lr_scheduler.md 存在 mkdocstrings members 页（含 CosineWarmRestarts 等），已按 13 票同款格式补 `- LinearWarmupCosineAnnealingLR`（字母序插于 Linear 后）
- 附加检查 2（nav 注册集对照 conventions §Q4 修正版五处）：mkdocs.yml:117 nav、docs/{zh,en}/api/arch.md 各 2 行、docs/{zh,en}/api/data/dataset.md 各 1 行、docs/{zh,en}/index.md 表各 1 行、docs/{zh,en}/examples/climax.md 双语文件——全部在位，加 16 票牵出的 lr_scheduler 两行，注册集合无遗漏

### 改动面汇总表（PaddleScience @ feat/climax，工作区未提交）

| 分类 | 文件 | 行数 |
|---|---|---|
| arch（新包） | ppsci/arch/climax/{__init__,arch,climatebench,parallelpatchembed,pos_embed}.py | 706 |
| arch（vendored） | ppsci/arch/paddle_timm/ 9 文件 | 542 |
| arch（注册） | ppsci/arch/__init__.py | +4 |
| dataset | ppsci/data/dataset/climatebench_dataset.py + __init__.py | 334 +2 |
| optimizer | ppsci/optimizer/lr_scheduler.py | +81 |
| example | examples/climax/{main.py,utils.py,conf/climax.yaml,regrid_climatebench.py,requirements.txt} | 758 |
| docs | docs/{zh,en}/examples/climax.md | 578 |
| docs（注册） | mkdocs.yml + docs/{zh,en}/{api/arch,api/data/dataset,api/lr_scheduler,index}.md | +11 |
| **合计** | **34 文件** | **+3016**（tracked +98 纯插入 / untracked 2918） |

与 PaddleCFD 版贡献（26 文件 +2629）对照：净增 ~387 行、8 文件，全部来自 ppsci 形态的必然增量——双语文档两份（578 行，ppcfd 单 README）与主包注册行（12 处 +98）；核心逻辑（paddle_timm + climax + dataset + example 代码）与 ppcfd 版同源等价（08 票门背书），新增主包 `LinearWarmupCosineAnnealingLR` 81 行（16 票，替代 ppcfd 的 example 内自写类）。

### PR message 草稿

已写 `/home/lkyu/baidu/test/ppsci-climax-pr.md`（三段式：Accuracy alignment / CINN acceleration / Usage；全部数字带条件括号、无 ppcfd/内部参照物字样、检查点称 released checkpoint；+5.2% 历史数字以括注一句保留供维护者参考、标注用户润色时可删）。注意：controller 指令中的 "within 5.1%" 经对源复核不实（test 五项最大 7.3%），已修正为 7.3% 并扩表——PR 草稿与 docs 采用同一修正口径。

### map 更新确认

`map.md` Decisions so far 已追加 16 票一行（lr 调度器平移修复，根因 = 时间尺度错位，含 601 步单测与修复后 1-epoch 门数字）；Not yet specified 的 PR message 条更新为已产出（指向草稿路径）；16 票全部 resolved（15 票本条收尾前逐一核对）。
