# PaddleCFD vs PaddleScience 结构与惯例对照（ClimaX 移植调研）

- 调研日期：2026-09-26
- 调研对象：
  - `/home/lkyu/baidu/PaddleCFD`（fork of github.com/PaddlePaddle/PaddleCFD，分支 `feat/climax` @ `5df8c0a`，基线官方 develop `f366aae`）
  - `/home/lkyu/baidu/PaddleScience`（fork of github.com/PaddlePaddle/PaddleScience，HEAD `d57561b1`，submodule 未检出、不影响本调研）
- 所有行号引用均基于上述本地仓库当前检出状态。

## 执行摘要

**移植可行**：PaddleScience 对"论文复现型 + 气象领域"贡献有充分先例（Climateformer/FuXi/Pangu-Weather/GraphCast 等约 20 个气象 example，官方复现验收标准文档 `docs/zh/reproduction.md`），且全仓库无任何 `climax/ClimaX` 命名冲突（grep 全仓库 py/md/yaml 无命中）、已内置 `LatitudeWeightedRMSE/ACC` 指标可复用。ClimaX 的移植路径清晰：模型进 `ppsci/arch/`、数据集进 `ppsci/data/dataset/`、示例进 `examples/`、文档进 `docs/zh+en/examples/` 并注册 mkdocs nav 与 README 大表。

**最大的 3 个差异点（也是主要工作量来源）**：

1. **框架接管训练循环**：ppcfd 的 ClimaX 用 Lightning 式 `module.py/datamodule.py` 自管训练+训练毕自动 test（`examples/climax/train.py:237-256`）；PaddleScience 惯例是模型类继承 `base.Arch`（dict 进 dict 出的 `input_keys/output_keys` 接口，`ppsci/arch/climateformer.py:347`），训练/评估交给 `ppsci.solver.Solver` + hydra 配置（`examples/climateformer/main.py`）。ClimaX 的 backbone+head+loss 一体封装（`ppcfd/models/climax/climatebench.py`）必须拆解开构。
2. **文档形态完全不同**：ppcfd 是 example 目录内英文 README（`examples/climax/README.md`）；PaddleScience 是 mkdocs 双语文档树 `docs/zh/examples/<name>.md` + `docs/en/examples/<name>.md`（各 86 篇，example 目录内不放 README，仅 `yinglong/adr` 两个例外），且需同步注册 `mkdocs.yml` nav、根 README 案例大表、`docs/zh/api/arch.md` API 文档共 4 处。
3. **贡献流程更正式**：PaddleScience 有 PR 模板（`.github/PULL_REQUEST_TEMPLATE.md`）、pre-commit（isort+black，**black 行宽 88**，ppcfd 是 119）、squash merge（commit 带 `(#PR号)` 后缀，如 `b2e35d87 Add Climateformer for climate prediction (#1211)`）、以及复现任务前置 RFC 要求（`docs/zh/development.md:5`）。代码需全面重新格式化并重写提交叙事。

---

## 1. 模型包/架构层

| 维度 | PaddleCFD | PaddleScience | 对 ClimaX 移植的影响 |
|---|---|---|---|
| 对应目录 | `ppcfd/models/` | `ppsci/arch/`（63 个 py 文件） | 路径迁移 |
| 组织方式 | **一模型一目录**：`ppcfd/models/climax/` 含 `arch.py / climatebench.py / parallelpatchembed.py / pos_embed.py / __init__.py / timm/`（vendored）；同类如 `multiple_physics_pretraining/`、`poseidon/` | **一模型一文件**为主（`afno.py`、`climateformer.py`…）；"一族多文件"例外：`cuboid_transformer*.py`（4 文件）、`moflow_*.py`（3 文件）、`extformer_moe_*.py`（5 文件）；**目录例外**：`ppsci/arch/paddle_harmonics/`（vendored torch-harmonics 的 paddle 移植，保留上游 SPDX BSD-3 头，`ppsci/arch/paddle_harmonics/sht.py:3-4`） | 两种可行形态：(a) 单文件 `climax.py` 内联 timm Block/PatchEmbed（主流）；(b) `climax/` 目录 vendored timm（`paddle_harmonics` 先例）。参考 Climateformer PR #1211 是单文件 |
| 基类 | `paddle.nn.Module` 直接继承（`ppcfd/models/climax/arch.py:20` `class ClimaX(paddle.nn.Module)`） | `base.Arch`（`ppsci/arch/climateformer.py:347` `class Climateformer(base.Arch)`）；官方指南 `docs/zh/development.md:113-128` 明确要求"从 `base.Arch` 派生" | ClimaX/ClimaXClimateBench 需改造成 `input_keys/output_keys` + dict 式 forward（见 `ppsci/arch/base.py:30-32` 的 keys 声明） |
| 命名 | 文件名=模型名（目录），类名 `ClimaX` / `ClimaXClimateBench` | 文件名小写模型名（`climateformer.py`），类名 PascalCase（`Climateformer`），首个公开类不导出辅助类 | `climax.py` + `class ClimaX(base.Arch)`（无冲突） |
| 注册机制 | `ppcfd/models/__init__.py:19-124`：**try-import 块** + `__all__.append("climax")`，缺依赖时 warning 降级 | `ppsci/arch/__init__.py:20-77`：**无条件直接 import**（无 try/except）+ `__all__` 列表（:83-145）+ `build_model()` 用 `eval(arch_cls)` 实例化（:148-163） | 删掉 try-import 包装；ClimaX 不依赖 timm 包（vendored），无缺依赖风险，直接 import 即可 |
| 模型内部封装 | backbone（`arch.py` ClimaX）+ 下游任务封装（`climatebench.py` ClimaXClimateBench，含 loss 逻辑）分层 | 模型只做 forward，loss 在 `ppsci.loss.*`、指标在 `ppsci.metric.*`，由 Solver 组装（`examples/climateformer/main.py` 的 train() 内 MSELoss + MAE/MSE metric） | ClimaXClimateBench 的 loss/w_mse 逻辑要移出到 example/自定义 metric |

## 2. example 组织

实际目录名：两仓库都是平级 `examples/<name>/`（PaddleScience 共 89 个，PaddleCFD 17 个）。

**PaddleScience 现有气象 example 实测文件清单**（均无 example 内 README）：

| example | 文件清单 | 备注 |
|---|---|---|
| `examples/climateformer/`（最接近 ClimaX 形态：ERA5+ViT 型时空 Transformer） | `main.py`、`utils.py`、`conf/climateformer.yaml` | 单入口 train/eval 合一；数据集类在 `ppsci/data/dataset/era5climate_dataset.py` |
| `examples/fourcastnet/`（AFNO 气象预报，复现型） | `train_pretrain.py`、`train_finetune.py`、`train_precip.py`、`utils.py`、`sample_data.py`、`conf/fourcastnet_{pretrain,finetune,precip}.yaml` | 多阶段=多脚本多 yaml |
| `examples/earthformer/`（ViT/cuboid Transformer 复现） | `earthformer_enso_train.py`、`earthformer_sevir_train.py`、`predictor.py`、`enso_metric.py`、`sevir_metric.py`、`sevir_vis_seq.py`、`sevir_cmap.py`、`conf/*.yaml` ×2 | 自定义 metric/可视化放 example 内 |
| `examples/fuxi/` | `predict.py`、`util.py`、`visualize.py`、`conf/*.yaml` ×4、**`requirements.txt`** | example 级额外依赖先例（bottleneck/cartopy/dask/netCDF4/xarray） |

**PaddleCFD ClimaX 现状**（`examples/climax/`）：`train.py`、`module.py`、`datamodule.py`、`dataset.py`、`metrics.py`、`lr_scheduler.py`、`normalize.py`、`regrid_climatebench.py`、`configs/climate_projection.yaml`、`README.md`（在 example 内）。

**映射建议（影响）**：
- `dataset.py/datamodule.py` → `ppsci/data/dataset/climatebench_dataset.py`（Climateformer 先例：`era5climate_dataset.py` 随 PR #1211 加入主包，commit body 明确 "add era5climate_dataset for Climateformer"，见 `b2e35d87`）；`__getitem__` 需改为返回 `(input_dict, label_dict, weight_dict)` 三元组形态（`ppsci/data/dataset/era5climate_dataset.py:168`）。
- `module.py`（训练/优化/日志逻辑）→ 解体进 `main.py` 的 train(cfg) + Solver。
- `metrics.py` → 部分复用 `ppsci.metric.LatitudeWeightedRMSE/ACC`，变量维/气候态归一化的 metric（`metrics.py:154 lat_weighted_nrmses` 等）可做自定义 metric 类放 example 内（earthformer 的 `enso_metric.py` 先例）。
- `regrid_climatebench.py`/`normalize.py` → 保留在 example 内（earthformer 的 `sevir_*.py` 辅助脚本先例）。

## 3. 入口与配置形态

| 维度 | PaddleCFD | PaddleScience | 影响 |
|---|---|---|---|
| CLI/入口 | `python train.py`，无 CLI，无 hydra；`OmegaConf.load` 固定路径 `configs/climate_projection.yaml`（`train.py:33-41, 79`），docstring 明言 "there is no CLI surface"（`train.py:6`） | `python main.py [hydra overrides]`；`@hydra.main(version_base=None, config_path="./conf", config_name="climateformer.yaml")` 装饰（`examples/climateformer/main.py` 尾部）；`mode: train/eval` 单入口双模式 | 改造成 hydra：conf 目录名是 `conf/`（不是 ppcfd 的 `configs/`） |
| 配置结构 | 扁平业务 yaml（`configs/climate_projection.yaml`：seed/model.net.class_path+init_args/data 三段，`class_path` 指向模型类——ppsci 无此机制） | `defaults:` 组合链 `ppsci_default / TRAIN:train_default / TRAIN/ema / TRAIN/swa / EVAL:eval_default / INFER:infer_default / _self_`（`examples/climateformer/conf/climateformer.yaml:1-9`）；默认值由 pydantic 模型在 `ppsci/utils/config.py:380` `cs.store(name="ppsci_default",...)` 代码内注册（仓库中无独立 default yaml 文件） | ClimaX yaml 需重写为 defaults 组合结构；超参禁止写死（`docs/zh/reproduction.md:196`） |
| 训练毕自动 test | 有：训练结束自动加载 best checkpoint 跑 test 并打印 lat-weighted 表（`train.py:237-256`） | **无"训练毕自动 test"先例**；对应惯例是 `TRAIN.eval_during_train: true` + `TRAIN.eval_freq`（验证集训练中评估，`climateformer.yaml` TRAIN 段），测试集用 `python main.py mode=eval EVAL.pretrained_model_path=<bcebos url>` 独立跑（`docs/zh/examples/climateformer.md:18-22`） | 放弃内建 test，改 eval_during_train + mode=eval；或与 reviewer 沟通保留 |
| to_static/CINN | 环境变量 `CLIMAX_USE_CINN=1` 在 import paddle 前设 FLAGS + `paddle.jit.to_static(full_graph=True)`（`train.py:17-31, 110-112`） | Solver 有 `to_static` 参数（`ppsci/solver/solver.py:159, 517-525`，yaml 字段 `to_static`，`ppsci/utils/config.py:316`），但**只包 forward helper，无 CINN FLAGS/env 设置先例** | CINN 支持在 ppsci 无先例：可对接 yaml `to_static: true`，CINN FLAGS 方案需与 reviewer 对齐呈现方式（env var 保留 or 文档说明） |

## 4. README 骨架

**ppcfd（`examples/climax/README.md`，英文，132 行）**：`# ClimaX — A Weather and Climate Foundation Model` → Model architecture → Pretrained checkpoint → Quick start（Data preparation / Training / Evaluation）→ CINN acceleration → Accuracy alignment → Additional dependencies → Citing（bibtex）。

**PaddleScience（`docs/zh/examples/climateformer.md`，中文；`docs/en/examples/climateformer.md` 同构英文镜像）**标准章节：

1. `# <ModelName>` 标题
2. 数据/预训练模型下载段（bcebos 链接直链 `.h5/.nc/.pdparams`，如 `climateformer.md:7-10`）
3. 命令段：`=== "模型训练命令"` / `=== "模型评估命令"` mkdocs-material tab（`climateformer.md:12-22`）
4. `## 1. 背景简介`
5. `## 2. 模型原理`（嵌源码片段 `--8<-- ppsci/arch/climateformer.py:243:277` 行级引用 + 架构图 figure）
6. `## 3. 模型训练`（3.1 数据集介绍 → 3.2 逐步讲解：模型构建/约束器构建/评估器构建/学习率与优化器/模型训练/训练时评估 → 3.3 评估模型；fourcastnet/earthformer 还有模型导出+推理小节）
7. `## 4. 完整代码`（把涉及的 dataset/arch/example/yaml 全文嵌入）
8. `## 5. 结果展示`（预测 vs 真值图）
9. 部分有 `## 6/7. 参考资料`（`pangu_weather.md:110`）

**口径要点**：
- 结果表格化：ACC/RMSE 按预报时效列（`docs/zh/examples/fourcastnet.md:59-65`），模型名即 pdparams 下载链接。
- checkpoint 措辞：统一"预训练模型"、`EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/...`（bcebos 托管，ClimaX 的 best.pdparams 需上传）。
- 复现文档规范（`docs/zh/reproduction.md:188-197`）：开头须有论文题目/地址/参考代码链接并致谢原作者；末尾附参考论文、参考代码、复现模型参数下载链接；额外依赖（如 pandas）在文档开头说明。bib 引用非强制（`transformer4sr.md:275` 有 @inproceedings 先例）。
- 语言：中文为主文档 + 英文全量镜像（`docs/en/examples/` 86 篇与 zh 一一对应）。

**影响**：ClimaX 英文 README 需整体重写为中英双份 mkdocs 文档，Accuracy alignment/CINN 章节按 ppsci 口径融入"结果展示"与命令段。

## 5. 注册/打包

| 维度 | PaddleCFD | PaddleScience | 影响 |
|---|---|---|---|
| 打包范围 | `find_packages(exclude=("doc","examples"))`（`setup.py:22-25`）；`ppcfd` 包整体入 wheel | exclude `docs/examples/jointContribution/test_tipc/test/tools/ppsci/externals*`（`setup.py:39-49`、`pyproject.toml [tool.setuptools.packages.find]`）；**`ppsci/arch/` 下内容（含 vendored `paddle_harmonics/`）会打进 wheel**，仅 `ppsci/externals/` submodule 不打包 | vendored `timm/` 若放 `ppsci/arch/climax/` 会随包发布（paddle_harmonics 同样如此，可接受） |
| example 与主包 import | example 内平级 import（`train.py:35-36` `from datamodule import ...`）+ 绝对 `from ppcfd.models.climax.climatebench import ...`（`train.py:37`）；根目录 `pip install -e .` 运行 | example 内平级 import（`examples/climateformer/main.py:15` `import utils as utils`）+ `import ppsci`；官方用 `export PYTHONPATH=$PWD:$PYTHONPATH`（`docs/zh/development.md:31-35`）或 editable 安装 | 基本同构，迁移成本低 |
| 数据集注册 | 无集中注册（example 自带 dataset.py） | `ppsci/data/dataset/__init__.py` 直接 import 全部 dataset 类 + `build_dataset` 的 `eval()`（`:122-142`）；新 dataset 须加入此处 | ClimateBench dataset 要登记进主包 `__init__.py` |

## 6. 贡献流程

| 维度 | PaddleCFD | PaddleScience | 影响 |
|---|---|---|---|
| 贡献指南 | 无 CONTRIBUTING.md、无 .github/；README 无贡献章节（grep 仅 license 行 `README.md:119`） | 无根级 CONTRIBUTING.md，但有 `docs/zh/development.md`（开发指南全文 972 行）+ `docs/zh/reproduction.md`（复现验收标准）；README 含案例大表 | 按后者流程走 |
| 分支命名 | `feat/<model>`（`feat/climax`、merge 记录 `f366aae` "Merge pull request #141 from lkyu-ly/feat/G-FNO"、`b2e1655` #137 feat/mpp 等） | 官方示例用 `dev_model` 泛化名（`development.md:25-29`）；实测贡献分支名不一（`add_p`、`new` 等，git log merge 记录） | 分支名无硬约束 |
| 合并方式 | **merge commit** 保留分支史（`git log --merges develop`：#136-#141 全是 "Merge pull request ..."） | **squash merge**：commit 标题带 `(#PR号)`（`a961fab4` "[Example] Add transolver(shapenet_car) (#1238)"、`b2e35d87` "Add Climateformer for climate prediction (#1211)"，body 保留分支 commit 列表） | ppcfd 侧的多 commit 结构在 ppsci 会被 squash，commit message 要写成完整叙事 |
| commit message | 自由格式（`5df8c0a` "add climax model (climate-projection downstream task only)"） | 惯例前缀：`[Example]` / `[Doc]` / `[BUG]` / `[Refine]` / `[Doc&Fix]` / 【Hackathon Nth No.X】任务号（git log --grep 实测） | 用 `[Example] Add ClimaX ...` 形态 |
| PR 模板 | 无 | `.github/PULL_REQUEST_TEMPLATE.md`：PR types（New features/Bug fixes/...）+ PR changes（OPs/APIs/Docs/Others）+ Describe，demo 指向 PR #96 | 照模板填写 |
| 前置要求 | 无 | **RFC**：论文复现、API 开发任务前需提交 RFC 文档（`docs/zh/development.md:5`，模板在 PaddlePaddle/community repo）；pre-commit 必须（isort+black+ruff，`development.md:45-58`） | RFC 是否适用于本次内部移植——**未确认，需与 reviewer 核实** |
| 格式化 | black line-length **119** + ruff（`.pre-commit-config.yaml`，pyproject `[tool.black] line-length=119`） | isort(single-line)+black **22.3.0 默认 88**+ruff 0.0.272（`.pre-commit-config.yaml`） | 全部 ClimaX 代码需按 88 列重新格式化，import 单行化 |

## 7. 复现型贡献先例

- **PaddleCFD**：`b2e1655` Merge PR #137 feat/mpp（Multiple Physics Pretraining，NeurIPS 2024 复现）、`5df8c0a` feat/climax 本身；example README 明写 "integrates the MPP model into PaddleCFD"（`examples/multiple_physics_pretraining/README.md`）。
- **PaddleScience**：明确接受且成体系：
  - 官方复现流程+验收标准文档 `docs/zh/reproduction.md`：指标相对误差 ≤10%（:86），推荐 PaConvert/PaDiff 工具链（:54）。
  - 论文复现 merge 实录：`ba9f019f` "【Hackathon 8th No.17】FuXi 论文复现 (#1145)"、`96eb5ba6` "【Hackathon 8th No.19】Pangu-Weather 论文复现 (#1089)"、`2eb2639b` "【Hackathon 6th No.37】GraphCastNet 代码迁移至 PaddleScience (#897)"、`ba0bfce7` "Add Meteoformer for meteorological forecasting (#1126)"、`b2e35d87` "Add Climateformer for climate prediction (#1211)"（单 PR 同时带 arch+dataset+example+docs 四件套，是 ClimaX 的最佳模板）。
  - `jointContribution/` 目录自述 "mainly used for sample libraries and model reproduction"（`jointContribution/README.md`），内含 graphcast/gencast/CFDGCN 等 17 个子项目（较大型/联合贡献走此目录，ClimaX 规模更适合标准 examples 路径）。

## 8. 版权头与注释

| 维度 | PaddleCFD | PaddleScience | 影响 |
|---|---|---|---|
| 覆盖率 | 204 个 py 中 26 个有 PaddlePaddle 头（13%，grep "Copyright (c) .* PaddlePaddle"）；climax 各模型文件均**无头**（`arch.py` 直接 `from functools import lru_cache` 开头），仅 `__init__.py` 有 2025 头 | 218 个 py 中 177 个有头（**81%**）；arch 下 35/63（56%），无头者多为外部移植文件（`climateformer.py`、`meteoformer.py`、`preformer.py` 无头；`transolver.py` 用 `"""Reference: https://github.com/thuml/Transolver"""`） | 新文件应加头，格式见下 |
| 格式 | `# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.` + Apache 正文（`ppcfd/models/climax/__init__.py:1-13`） | 同款格式，**年份=写文件的当年**：现存 2023（20 处，arch 内最多）/2024（11 处）/2025（2 处，如 `examples/climateformer/main.py:1`、`ppsci/data/dataset/__init__.py:1`） | ClimaX 新文件用 `# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.`（是否强制加头未在 development.md 中明文规定——未确认，但主流是加） |
| vendored 代码头 | `timm/block.py:1-2` 保留 "Hacked together by / Copyright 2020 Ross Wightman" + timm 1.0.24 来源说明 | `ppsci/arch/paddle_harmonics/sht.py:3-4` 保留上游 `SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors` + BSD-3 SPDX 头 | vendored timm 保留上游声明即可，有直接先例 |
| 注释/docstring 语言 | 英文（`arch.py` 类 docstring 英文） | **docstring 英文、文档中文**（`climateformer.py:348-353` 英文 Google-style docstring；example 内注释英文为主，docs 全中文） | 代码注释保持英文即可 |

## 9. 依赖与环境

| 维度 | PaddleCFD | PaddleScience | 影响 |
|---|---|---|---|
| python | `>=3.10`（`pyproject.toml requires-python`，setup.py `python_requires=">=3.10"`） | `>=3.8`（pyproject classifiers 3.8/3.9/3.10；docs 推荐 conda py310 环境 `docs/zh/install_setup.md:44-48`） | 若严格 3.8 兼容需检查 ClimaX 代码是否用了 3.9+/3.10+ 语法——**未确认 reviewer 是否强制 3.8** |
| paddle | `paddlepaddle-gpu==3.0.0`（`README.md:54`） | "3.0 以上稳定版或最新 develop"（`README.md:191` paddle_install 片段） | 兼容，CINN 需 paddle 3.x 亦满足 |
| requirements | 36 项，含 xarray/xskillscore/zarr/pandas/scienceplots/fastapi 等（`requirements.txt`） | 21 项（`requirements.txt`），**无 timm、无 xarray/netCDF4/pandas** | ClimaX 的 xarray/netCDF 栈不进主 requirements；先例：`examples/fuxi/requirements.txt`（bottleneck/cartopy/dask/netCDF4/xarray）+ 文档开头说明安装（`reproduction.md:194`）。ppsci 无 timm pip 依赖，vendored 方案正好规避 |
| 打包名 | `ppcfd` 0.3.0 | `paddlesci`（setuptools_scm，fallback 1.4.0） | — |

## 10. ClimaX 特有检查

- **命名冲突**：`grep -ri climax` PaddleScience 全仓库（py/md/yaml）**零命中** → `climax.py`/`class ClimaX`/`examples/climax/` 均可用。
- **已有 ViT/pos_embed 实现重叠**（需避让或复用）：
  - `ppsci/arch/cvit.py:35-81` 已有 **同名同签名** 的 `get_1d_sincos_pos_embed_from_grid` / `get_1d_sincos_pos_embed` / `get_2d_sincos_pos_embed`——与 `ppcfd/models/climax/pos_embed.py` 导出的两个函数完全同名。注意：cvit 的这些函数是模块私有（未从 `ppsci/arch/__init__.py` 导出），ClimaX 自带 pos_embed 不构成公共 API 冲突，但 review 时可能被要求复用/合并。
  - `ppsci/arch/afno.py:371` 有 `PatchEmbed` 类 + 可学习 `pos_embed` 参数（:478-529）；`ppsci/arch/transformer.py:38-273` 有 MultiHeadAttention/Encoder/Decoder；`ppsci/arch/physx_transformer.py:187` 有 `Block`。这些均为各模型私有辅助类，无跨模型共享的 ViT 组件——ClimaX 内联自己的实现符合现状。
- **气象领域 example**：约 20 个（fourcastnet/graphcast/gencast/pangu_weather/fuxi/fengwu/earthformer/climateformer/dgmr/nowcastnet/meteoformer/kmcast/stafnet/preformer/iops/tgcn/velocitygan/utae…），mkdocs nav 归入"地球科学(AI for Earth Science)"分组（`mkdocs.yml:106-130`），README 大表归"天气预报"等行（`README.md:135-144`）。ClimaX README 口径直接参照 `docs/zh/examples/climateformer.md` + `fourcastnet.md`。
- **可复用组件**：`ppsci.metric.LatitudeWeightedRMSE`（`ppsci/metric/rmse.py:73`）、`LatitudeWeightedACC`（`ppsci/metric/anomaly_coef.py:28`）已存在；但 ClimaX 的 metric 是按 vars 分组、带 clim 归一化的批函数（`examples/climax/metrics.py:154 lat_weighted_nrmses`），形态不同，预计仍需 example 内自定义（earthformer `enso_metric.py` 先例）。
- **arch 无单测惯例**：`test/` 下无 test/arch（仅 constraint/data/equation/experimental/geometry/loss/probability/utils）→ 新增 arch 不强制单测。

## 11. 两仓库顶层结构总览（深度 2，标注对应关系）

```text
PaddleCFD                                PaddleScience
──────────────────────────               ──────────────────────────
ppcfd/                                   ppsci/
├── data/        (loader/downloader)     ├── arch/        ← 对应 ppcfd/models（一文件一模型）
│   └── parser/                          ├── data/        （dataset/ 有 33 个数据集类）
├── models/      (一目录一模型)           │   └── dataset/
│   └── climax/                          ├── constraint/   ← ppcfd 无此层（自管训练）
│       └── timm/ (vendored)             ├── validate/     ← 同上
└── utils/       (logger/parallel)       ├── solver/       ← 训练循环接管（ppcfd 由 module.py 承担）
                                         ├── loss/ metric/ optimizer/ equation/ geometry/
examples/<model>/  (内含 README)          ├── autodiff/ probability/ visualize/ experimental/
├── climax/                              ├── utils/        (logger/config…)
│   └── configs/*.yaml                   └── externals/    ← submodule 的第三方 fork（paddle_harmonics 等 9 个）
doc/ docs/  (轻量)                       examples/<name>/  (无 README，conf/*.yaml)
source/ppfno_op/ (C++ op)                docs/ zh+en/      ← mkdocs 双语文档树（README 的去处）
checkpoint/ (本地产物)                    competition/ deploy/ docker/ jointContribution/
.spec-workflow/ .claude/ (本地)           recipe/ test/ test_tipc/ .github/ mkdocs.yml
```

要点：PaddleScience 多出 solver/constraint/validate/loss/metric 框架层与 docs 双语树、`.github/` CI 与模板、`jointContribution/` 复现专区；PaddleCFD 的模型目录制（models/climax/）在 ppsci 收敛为 arch 单文件制（vendored 依赖可例外成目录）。

## 12. License 对照

- 两仓库 LICENSE 均为 **Apache-2.0 全文**（同为 11438 字节），仅附录版权行不同：PaddleCFD "Copyright (c) 2016 PaddlePaddle Authors"（`LICENSE:1`）vs PaddleScience "Copyright (c) 2022 PaddlePaddle Authors"（`LICENSE:1`，diff 实测仅此一行）。代码头部声明各自为 Apache-2.0（见第 8 节）。
- vendored timm 上游为 Apache-2.0（Ross Wightman，`ppcfd/models/climax/timm/block.py:1-2`），与两仓库 license 兼容；ppsci 的 paddle_harmonics vendored 先例保留上游 BSD-3 SPDX 头共存于 Apache-2.0 仓库中。

---

## 移植风险与开放问题

1. **`base.Arch` 接口改造（最大技术风险）**：ClimaX forward 需要 `lead_times / variables / out_variables` 等非张量元数据（`examples/climax/datamodule.py:12-18` collate 形态），而 ppsci 的 Arch/Constraint/Validator 管线是 `input_dict → output_dict`。需要设计：把这些元数据编码进 input dict 键、用 `output_transform`、或自定义 constraint（`development.md:438-477` 允许新 constraint 类）。参照系 Climateformer 的 forward 只吃单输入张量，复杂度低于 ClimaX——**无完全同形的先例**。
2. **ClimaXClimateBench 拆解**：backbone+head+loss 一体类（`ppcfd/models/climax/climatebench.py`）需拆为 arch（纯 forward）+ loss/metric（example/主包）；torch 基线权重转换脚本与 init_state_path strict-load 逻辑（`train.py:101-109`）要在 ppsci 的 `pretrained_model_path` 机制下重演。
3. **CINN 支持的呈现**：ppsci Solver 有 `to_static` yaml 开关（`solver/solver.py:517-525`）但无 CINN FLAGS 先例；`CLIMAX_USE_CINN` env 方案 + "训练毕自动 test" 均无 ppsci 先例，PR 中如何保留 5.2% 加速证据需与 reviewer 对齐（可能以文档/实验记录形式呈现）。
4. **pos_embed 同名函数**：与 `ppsci/arch/cvit.py:35-81` 的 `get_*_sincos_pos_embed*` 同名——私有实现无技术冲突，但 review 可能要求去重/复用。
5. **数据与 checkpoint 托管**：ppsci 惯例全部走 `paddle-org.bj.bcebos.com/paddlescience/...` 直链（数据 + pdparams）；ClimateBench（Zenodo）+ regrid（xesmf，conda-forge）流程在 ppsci 文档中如何呈现待定（climateformer 先例是把 ERA5 预处理为 .h5 上传 bcebos）。
6. **依赖边界**：xarray/netCDF4 栈走 example 级 requirements.txt（fuxi 先例）还是文档说明（reproduction.md:194 允许）待选；主包 requirements 不动。
7. **RFC 前置**：`development.md:5` 要求论文复现任务先提交 RFC——本次从 PaddleCFD 平移是否豁免**未确认**。
8. **格式化重做**：black 88（ppsci）vs 119（ppcfd）+ isort single-line，全部文件需重排；PR 将被 squash，commit message 需按 `[Example] Add ClimaX ...` 重写。
9. **python 3.8 兼容声明**：ppsci classifiers 到 3.8/3.9/3.10；ClimaX 现代码按 3.10+ 编写，若需降级语法（如 `X | Y` 类型注解）要排查——**未确认是否强制**。
10. **文档四点注册缺一不可**：`docs/zh/examples/climax.md`、`docs/en/examples/climax.md`、`mkdocs.yml` nav 地球科学分组（:106-130）、根 `README.md` 案例大表（:135-144 气象区），另 `docs/zh/api/arch.md` 若导出新类需登记（Climateformer 在 `docs/zh/api/arch.md:12`）。
