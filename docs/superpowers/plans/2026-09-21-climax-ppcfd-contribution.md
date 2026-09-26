# ClimaX 向 PaddleCFD 贡献实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 climax_paddle（已对齐验证：前向 2.24e-08、5 epoch 训练偏差<10%、CINN +5.2%）按 PaddleCFD 惯例安置到 `feat/climax` 分支（已基于官方 develop f366aae 创建），产出可审查的贡献改动（**不 commit**，等用户审查）。

**Architecture:** 双区安置（mpp/poseidon 先例）：`ppcfd/models/climax/` 放模型构建代码（arch/parallelpatchembed/climatebench/pos_embed + timm 闭包子包），`examples/climax/` 放外围（一键 train.py + 数据/训练脚手架 + configs + README）。train.py 去 CLI 化（配置驱动 + 原模型默认参数），README 参考 mpp/poseidon 骨架与原版官方内容。

**Tech Stack:** paddle 3.4（paddletorch 环境）、omegaconf、PaddleCFD 仓库（/home/lkyu/baidu/PaddleCFD，分支 feat/climax）。

## Global Constraints

- **全程不 commit**：改动留在 feat/climax 工作区等用户审查；不 push、不建 PR
- 源码来源：/home/lkyu/baidu/CLIMAX/climax_paddle/（已通过三轮对齐验证与清理的最终形态）
- 安置纪律：tools/ 脚本（export/convert/compare_forward）与 tests/ 不进 ppcfd；数据/权重不入库；examples 运行产物不留
- 注释与措辞遵守 ppmat 贡献交付规范六规则（英文简练事实、无内部用语、无设计史、无第三方指控）；中文零残留
- 版权头：跟随 ppcfd 实际先例——`ppcfd/models/climax/__init__.py` 加 Apache 2.0 PaddlePaddle 头（同 ppcfd/models/__init__.py 形态），其余文件不加（poseidon/g_fno 先例均无）
- train.py 一键化：无 CLI 参数（删 --key=value 覆盖与 limit_batches 调试参数）；配置由 configs/climate_projection.yaml 驱动；yaml 编排值对照原版（climax_torch/configs/climate_projection.yaml）逐项恢复原模型默认（max_epochs 恢复 50）；数据路径占位化 `./data/climatebench/5.625deg`；pretrained_path 默认 ""（paddle 版读不了 torch HF ckpt，README 说明权重转换）
- CINN 开关（CLIMAX_USE_CINN 环境变量 + FLAGS + to_static）保留为功能（README 单独章节介绍）
- README 英文；结构参考 poseidon/mpp（Model/Checkpoint/Quick start(含 CINN 命令)/CINN acceleration/Alignment/Datasets/Dependencies/Citing）；模型介绍参考原版 README+docs/usage.md 口径；数字带口径不外推；检查点称 released checkpoint
- 修改涉及的接口改名最小化；climax_paddle 原仓不动（对照基线）
- 验证（最小努力）：import 链冒烟 + examples 单批跑通（动态图与 CINN 各一次，临时把 yaml root_dir 指向 /home/lkyu/baidu/CLIMAX/dataset/climatebench/5.625deg 跑完改回占位）+ 副产物清理
- 环境变量名 CLIMAX_USE_CINN 保持

## 已锁定的接口契约（跨任务）

- `ppcfd.models.climax` 包导出：`ClimaX`（arch.py）、`ClimaXClimateBench`（climatebench.py）；timm 闭包 `ppcfd.models.climax.timm`（导出 Block/PatchEmbed/trunc_normal_/to_2tuple）；pos_embed 在 `ppcfd.models.climax.pos_embed`
- examples/climax 内模块相互 import 用相对导入（`from dataset import ...` 不行——ppcfd 无包结构……先例：examples/poseidon/train.py 直接 `import` 同目录模块（examples 不打包，cwd 运行）。按此：examples/climax 内平级 import（`from dataset import ClimateBenchDataset`），模型 import 用 `from ppcfd.models.climax import ...`
- models 侧不 import examples 侧任何东西

---

### Task 1: ppcfd/models/climax 包落位

**Files:**
- Create: `/home/lkyu/baidu/PaddleCFD/ppcfd/models/climax/{__init__.py, arch.py, parallelpatchembed.py, climatebench.py, pos_embed.py}`
- Create: `/home/lkyu/baidu/PaddleCFD/ppcfd/models/climax/timm/`（整目录 10 文件自 climax_paddle/thirdparty/timm/）
- Modify: `/home/lkyu/baidu/PaddleCFD/ppcfd/models/__init__.py`（注册块）

**Interfaces:**
- Produces（Task 2/3 依赖）：契约区的导出与路径

- [ ] **Step 1: 复制与改名映射**：`climax_paddle/climax/arch.py → arch.py`（改 import：`climax.utils.pos_embed` → `.pos_embed`；`thirdparty.timm` → `.timm`）；`parallelpatchembed.py` 同理（`from .timm import to_2tuple`）；`climax/climate_projection/arch.py → climatebench.py`（`from climax.arch import ClimaX` → `from ppcfd.models.climax.arch import ClimaX`；pos_embed 同改）；`climax/utils/pos_embed.py → pos_embed.py`（无内部依赖，原样）；`thirdparty/timm/ → timm/`（内部相对 import 不变）
- [ ] **Step 2: `__init__.py`**：Apache 2.0 PaddlePaddle 头（照抄 ppcfd/models/__init__.py 头）+ `from ppcfd.models.climax.arch import ClimaX` / `from ppcfd.models.climax.climatebench import ClimaXClimateBench` + `__all__`
- [ ] **Step 3: 注册**：ppcfd/models/__init__.py 在合适位置加 try-import 块（照 confild 块形态，模型名 climax，说明注释一行 Climate model）
- [ ] **Step 4: 冒烟**：`cd /home/lkyu/baidu/PaddleCFD && PYTHONPATH=. /home/lkyu/miniconda3/envs/paddletorch/bin/python -c "from ppcfd.models.climax import ClimaX, ClimaXClimateBench; from ppcfd.models.climax.timm import Block; m = ClimaXClimateBench(default_vars=['CO2','SO2','CH4','BC'], out_vars='tas'); print(sum(p.numel() for p in m.parameters()))"` 期望 111831040（注意 paddletorch 环境需 paddle 可用；模型的 CLIMAX_USE_CINN 开关不在模型侧，构造无需环境变量）
- [ ] **Step 5: 规范自查**：中文 grep 零、无内部用语；timm 包 docstring 已合规（上轮清理后形态直接复制）

### Task 2: examples/climax 落位（一键 train）

**Files:**
- Create: `/home/lkyu/baidu/PaddleCFD/examples/climax/{train.py, module.py, datamodule.py, dataset.py, lr_scheduler.py, metrics.py, normalize.py, configs/climate_projection.yaml, README.md(占位，Task 3 填)}`

**Interfaces:**
- Consumes: Task 1 的 ppcfd.models.climax 契约
- Produces: 一键训练入口 `python train.py`（cwd=examples/climax）

- [ ] **Step 1: 外围模块落位与 import 改造**：`module.py`（`from ppcfd.models.climax.climatebench import ClimaXClimateBench`；`climax.utils.lr_scheduler` → 同目录 `from lr_scheduler import ...`；metrics/pos_embed 同理——pos_embed 用 `from ppcfd.models.climax.pos_embed import ...`）；`datamodule.py/dataset.py`（`from dataset import ...`、normalize 同目录）；`lr_scheduler.py/metrics.py/normalize.py` 原样（内部无跨依赖）
- [ ] **Step 2: train.py 一键化**（基于 climax_paddle 的 train.py）：
  - 删 argparse/--key=value 深覆盖、--train.limit_batches 调试参数及其 `_iter_limited` 逻辑（全量迭代）
  - 保留：CINN 环境变量开关块（FLAGS + to_static）、init_state_path 加载、seed 三件套、装配序列、训练循环（含融合计时的 50 步 loss 行）、验证/检查点/早停/测试、指标表打印
  - 配置：`OmegaConf.load(os.path.join(os.path.dirname(__file__), "configs/climate_projection.yaml"))`；train 段 max_epochs 等从 yaml
  - yaml 编排值对照原版逐项恢复：`max_epochs: 50`（原版 trainer 默认）、lr/beta/wd/warmup 60/600 等已是原值（核对）；`root_dir: ./data/climatebench/5.625deg`（占位）；`pretrained_path: ""`；`default_root_dir: ./exps/climate_projection`（占位）；init_state_path 默认 ""（yaml 加注释键或 train.py 常量——以 yaml 键存在、默认空为准）
- [ ] **Step 3: 单批验证（临时数据路径）**：临时把 yaml root_dir 改为 /home/lkyu/baidu/CLIMAX/dataset/climatebench/5.625deg、init_state_path 指向 models/climax_paddle/climax_initial.pdparams、max_epochs 临时 1；分别跑 `CLIMAX_USE_CINN=0/1 python train.py` 各 1 个 epoch × 1 批（临时再改 max_epochs？无 limit_batches 了——验证用 max_epochs 临时 1 + batch_size 临时大？不：临时把 max_epochs 设 1、跑完 793 步约 4.5 分钟可接受；或临时 batch_size 不变直接 1 epoch。选择：动态图 1 epoch（约 5 分钟）+ CINN 1 epoch（约 6 分钟含编译）——最小可接受验证）→ 跑完恢复占位值；删除运行产物目录
- [ ] **Step 4: 规范自查**（同 Task 1 Step 5）

### Task 3: examples/climax/README.md

**Files:** Create: `/home/lkyu/baidu/PaddleCFD/examples/climax/README.md`

- [ ] **Step 1: 撰写**（英文，结构参考 poseidon/mpp 先例）：
  1. `# ClimaX — A Weather and Climate Foundation Model`：一段简介（模型是什么、arXiv 2301.10343、本实现为 PaddlePaddle 复现）
  2. `## Model architecture`：简述（per-variable patch embedding → variable aggregation → ViT backbone → linear head；ClimateBench 子类的时间聚合）
  3. `## Pretrained checkpoint`：released checkpoint（HF tungnd/climax 5.625deg.ckpt，412MB）+ 转换为 .pdparams 的说明（一句话指向权重转换属移植仓库工作、本仓只加载 pdparams；"released"口径）
  4. `## Quick start`：`### Data preparation`（ClimateBench zenodo 下载、5.625° 重采样命令、目录结构）、`### Training`（`cd examples/climax && python train.py`；CINN 命令 `CLIMAX_USE_CINN=1 python train.py`；说明配置在 configs/ 编辑数据路径与权重路径）、`### Evaluation`（训练结束自动以 best checkpoint 运行 test，输出纬度加权指标表）
  5. `## CINN acceleration`（单独章节）：开启方式（CLIMAX_USE_CINN=1 环境变量，内部 FLAGS+to_static(full_graph=True)）+ 实测数据（**+5.2% steady-state**，口径括号：RTX 4060 Ti / FP32 / batch_size=1 / 793 steps per epoch / second-epoch pure-training wall，one-off compile ~90 s；注明不外推）
  6. `## Accuracy alignment`（简洁）：forward 与训练对齐结论各一句带口径（forward mean_abs 2.24e-08 vs torch；5-epoch final test metrics within 5.1% of the torch baseline）
  7. `## Additional dependencies`：xarray、netCDF4、scipy（pip 一行）
  8. `## Citing`：原版 bibtex 原文
- [ ] **Step 2: 规范自查**：无内部用语/无本地路径/数字全带口径/citation 与原版一致

### Task 4: 整体验证与交付前检查

- [ ] **Step 1: 全量自查清单**（ppmat 规范第五节 adapted）：中文 grep 零命中（py+md+yaml）；六类 pattern 复扫（Task/票号/上游/参照物/第三方指控）；git status 确认无数据/权重/运行产物混入（.gitignore 不动 ppcfd 的，手动确认 untracked 只有目标文件）
- [ ] **Step 2: 最终验证运行**（若 Task 2 Step 3 已跑通则仅复核 log）；清理一切临时产物（恢复 yaml 占位值）
- [ ] **Step 3: 汇总改动面**（文件清单+行数）供用户审查，**不 commit**

## 执行波次（subagent-driven-development 调度）

Wave 1 并行：Task 1 / Task 2（接口契约已锁定，文件不交叉——T1 在 ppcfd/models/，T2 在 examples/；T2 的 Step 3 验证依赖 T1 完成，等待策略同前）→ Wave 2：Task 3 → Wave 3：Task 4（主会话执行或单代理）

## Self-Review 结论

- 用户要求覆盖：develop 更新检查+分支 ✓（已办：upstream 添加、官方=fork 无前进、feat/climax 已建）；借鉴安置 ✓（mpp/poseidon/prose_fd 三先例）；model/example 划分 ✓；一键 train 无 CLI+原默认参数 ✓（max_epochs 50 恢复、limit_batches 删）；README 规范+原版参考+CINN 章节 ✓；tools 不进 ✓；最小验证+无副产物 ✓；不 commit ✓
- 遗留决策点（已按证据裁定，如有异议用户审查时可改）：climatebench.py 扁平命名（ppcfd 惯例）；tests 不带入（ppcfd 无测试惯例）；requirements.txt 不动（README Additional dependencies 章节，poseidon 先例）
