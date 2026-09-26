# 01: paddle_timm vendored 落位

**Type:** task
**Status:** resolved
**Blocked by:** None (can start immediately)

## What to build

PaddleScience 主包内出现可独立 import 的 `ppsci/arch/paddle_timm/`（timm 1.0.24 最小闭包的 vendored 平移），Block/PatchEmbed 等符号与 state_dict 键名契约和已验证形态完全一致。

## 实施要点

- 源：PaddleCFD feat/climax 的 `ppcfd/models/climax/timm/`（10 文件约 627 行；读法：`git -C /home/lkyu/baidu/PaddleCFD show 'feat/climax:ppcfd/models/climax/timm/<file>'`，或确认该仓工作区分支后直读）
- 文件头参考 `ppsci/arch/paddle_harmonics/sht.py:3-4` 的 SPDX/上游署名**风格**（选择性参考不照抄）：保留 timm 上游署名（Ross Wightman, Apache-2.0）与来源说明；目录不得叫 `timm` 或 `vendor`（Q3）
- 包内相对 import 不变；对外仅依赖 paddle + 标准库（deep-dive-climax-side.md §10 已核实）
- Q7 复用：`weight_init.py` 删除，`trunc_normal_` 改从 `ppsci.utils.initializer` 导出（签名逐参一致）；`paddle_timm/__init__.py` 导出面维持 Block/PatchEmbed/trunc_normal_/to_2tuple 等
- 全仓 grep 确认 `paddle_timm` 无同名冲突

## Acceptance criteria

- [x] `PYTHONPATH=仓库根` 下 `from ppsci.arch.paddle_timm import Block, PatchEmbed` 冒烟通过
- [x] Block/PatchEmbed 的 state_dict 键名契约不变（平移不改键名）
- [x] 全仓无 `paddle_timm` 同名冲突；文件头署名合规

## Answer

### 做了什么

`/home/lkyu/baidu/PaddleScience/ppsci/arch/paddle_timm/` 落位 9 个文件（共 544 行），自 PaddleCFD `feat/climax:ppcfd/models/climax/timm/` 平移：

- 平移：`__init__.py, attention.py, block.py, drop.py, format.py, helpers.py, mlp.py, patch_embed.py, trace_utils.py`
- 删除：`weight_init.py`（Q7 复用落地，见下）
- 每个文件头新增 5 行：`# coding=utf-8` + `SPDX-FileCopyrightText: Copyright (c) 2020 Ross Wightman. All rights reserved.` + `SPDX-License-Identifier: Apache-2.0`（仿 `ppsci/arch/paddle_harmonics/sht.py:3-4` 风格）；原 docstring 中的 "Hacked together by / Copyright 2020 Ross Wightman"、"# Adapted from .../pytorch-image-models (timm 1.0.24)" 及 patch_embed 的 google-research/ViT/big_vision 来源说明全部保留
- 仅有的非头部改动（diff 逐行核实）：包内相对 import 拆单行 + block.py 一处 assert 收拢为单行（均为仓库 pre-commit 强制的 black 22.x / isort `--sl --multi-line=7 --profile=black` 格式）；`__init__.py` 中 `from .weight_init import trunc_normal_` → `from ppsci.utils.initializer import trunc_normal_`。零逻辑行改动

### Q7 复用核实（trunc_normal_）

签名逐参一致：ppcfd `trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0)` vs `ppsci/utils/initializer.py:156` 同名同序同默认值（ppsci 版仅多类型注解）。数值链路一致：两侧均为 no_grad 下 uniform(min=2l-1, max=2u-1) → erfinv → `*std*sqrt(2)+mean` → clip(a,b) → set_value。唯一差异是超界告警通道（warnings.warn vs logger.warning），无数值影响。实测同种子 3 组参数（含 ClimaX 典型 std=0.02, a/b=±0.04）maxdiff=0.000e+00，in-place 语义两侧均成立（返回原 tensor 引用）。

### 验收证据（真实执行）

1. 冒烟（`PYTHONPATH=.` 真跑）：`from ppsci.arch.paddle_timm import Block, PatchEmbed, trunc_normal_, to_2tuple, ...` 12 个导出符号全部可导入；Block/PatchEmbed 构造成功；`trunc_normal_(t, std=0.02, a=-0.04, b=0.04)` 后 100% 落界内，std=0.0176（截断正态预期）。
2. state_dict 键名契约：ppcfd 原包（git show 抽取为独立参照包）与 ppsci 版子进程同构对照——Block 12 键有序全等 `['norm1.weight', 'norm1.bias', 'attn.qkv.weight', 'attn.qkv.bias', 'attn.proj.weight', 'attn.proj.bias', 'norm2.weight', 'norm2.bias', 'mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight', 'mlp.fc2.bias']`；PatchEmbed 2 键 `['proj.weight', 'proj.bias']`；qkv_bias=False（11 键）与 drop_path>0 变体键集亦全等。附加：state_dict 互转后前向 maxdiff=0.0（Block 与 PatchEmbed 均）。
3. 冲突与合规：grep 全仓 `paddle_timm`（py/toml/cfg/md/yml）文件内容零命中（目录名不进内容，后续票 import 前无任何引用）；`__all__` 12 项与 ppcfd 版完全一致；black --check 与 isort --check-only（pre-commit 同参）均 PASS。

### 遗留事项

- 下游票注意：`__init__.py` 的 `trunc_normal_` 现为 `ppsci.utils.initializer` 的再导出（函数对象即 `ppsci.utils.initializer.trunc_normal_`），arch.py/parallelpatchembed.py 平移时 `from ppsci.arch.paddle_timm import Block, PatchEmbed, trunc_normal_` / `to_2tuple` 可直接照抄。
- 本票未动 PaddleScience 其他文件；工作区所见 `examples/climax/`、`ppsci/data/dataset/{__init__.py, climatebench_dataset.py}` 为并行票产物。
- 临时验证物（/tmp/ppcfd_timm_ref、/tmp/verify_paddle_timm.py、paddle_timm/__pycache__）已全部清理。
