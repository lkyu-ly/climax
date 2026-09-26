# 11: docs/en/examples/climax.md（英文文档）

**Type:** task
**Status:** resolved
**Blocked by:** 05

## What to build

ppsci 骨架的英文 mkdocs 文档就位：以 ppcfd README 内容为底稿改写（非翻译式平移，按 ppsci 268 行骨架组织）。

## 实施要点

- 骨架（deep-dive-conventions.md §Q3，模板 climateformer.md）：标题 → 数据/权重下载段 → 命令 tab（模型训练命令 / 模型评估命令）→ `## 1. 背景简介` → `## 2. 模型原理`（`--8<--` 源码行级嵌入）→ `## 3. 模型训练`（数据集介绍 / 逐步讲解：模型构建-约束器-评估器-优化器-训练-训练时评估）→ `## 4. 完整代码`（四件嵌入）→ `## 5. 结果展示`
- 数据段：zenodo 下载直链（UTAE 先例）+ regrid 命令 + xesmf conda 安装说明；**权重只写转换出处**（HF tungnd/climax 的 5.625deg.ckpt，说明转换为 pdparams 属本地工作），不放任何直链（Q9 零上传）
- 数字全部带口径不外推；checkpoint 称 released checkpoint；CINN 单独小节（+5.2%，RTX 4060 Ti / FP32 / bs=1 / 793 steps per epoch，one-off compile ~90 s）
- Accuracy alignment 段引实验记录旧证据（2.24e-08 / within 5.1%）+ 10 票 ppsci 路径新证据
- ppmat 规范：英文简练事实、无内部用语、无设计史

## Acceptance criteria

- [x] 骨架全节齐备（对照 climateformer.md 逐节）
- [x] ppmat 六类 pattern grep 零命中
- [x] 全部数字对源复核（实验记录 + 10 票 Answer）

## Answer

### 做了什么

新增 `/home/lkyu/baidu/PaddleScience/docs/en/examples/climax.md`（290 行，英文 mkdocs），骨架逐节对照 climateformer.md：标题 → 数据/权重下载段（zenodo record 7064308 两文件直链 + regrid 命令 + xesmf conda-forge 说明 + requirements.txt + test 目录软链 historical 说明 + released checkpoint 仅写转换出处：HF tungnd/climax 5.625deg.ckpt 412 MB → 本地转 pdparams，`{"state_dict": {net. 前缀}}` 经 TRAIN.init_state_path 加载）→ 命令 tab（=== "Model Training Command" / "Model Evaluation Command"，与模板同语法）→ `## 1. 背景简介`（气候投影 emulation 动机 + ClimaX 2301.10343 + 本 case 范围）→ `## 2. 模型原理`（阶段表 + 4 小节行级嵌入：climatebench.py:137:154 per-variable patch embedding / arch.py:220:231 variable aggregation / climatebench.py:155:165 ViT encoder + temporal/lead-time conditioning / climatebench.py:166:169+191:194 temporal aggregation + head）→ `## 3. 模型训练`（3.1 数据集介绍；3.2.1–3.2.6 模型构建/约束器/评估器/学习率与优化器/训练/训练时评估全部贴 main.py 对应代码块；3.3 评估模型两小节）→ `## 4. 完整代码`（dataset/arch.py/climatebench.py/main.py/conf 四类五块嵌入）→ `## 5. 结果展示`（forward 2.24e-08、5-epoch within 5.1% + w_mse/w_rmse 双列指标表）→ `### 5.1 CINN Acceleration`（单独小节：开启命令 + +5.2% 全口径 + one-off ~90 s）。

### 验收实录

1. 骨架：heading 清单逐项对照 climateformer.md（命令 tab、## 1–5、3.2 六小节、3.3 两小节）全齐，另加 5.1 CINN 小节（票面允许）。
2. ppmat grep：`grep -nE 'Layered after|mirrors|adapted from port|上游|决策|ticket|T[0-9]\.[0-9]|P[0-9]-|票'` 零命中（初版 "CMIP6-based" 命中 `P[0-9]-`，已改写为 "built on CMIP6 scenario simulations"）。
3. 数字复核：全部对 canonical facts 一字不差（2.24e-08 / 5.1% / 0.13394 vs 0.13900 / 0.36180 vs 0.36871 / +5.2% / RTX 4060 Ti / FP32 / batch_size=1 / 793 steps per epoch / 第二 epoch 纯训练段墙钟 / ~90 s / 839 MB / 74 MB / 412 MB / seed 42 / 5-epoch），并与实验记录（ClimaX迁移实验记录.md 2026-09-20/21 段）核对来源一致；超参数均取自 conf/climax.yaml（epochs 50、lr 5e-4、warmup 60、horizon 600、wd 1e-5、train_ratio 0.9、history 10、img_size 32x64 等）。
4. 嵌入语法：21 块 `--8<--` 全部通过脚本校验（文件存在、行区间不越界、linenums 与区间起点一致：climatebench.py 198 行 / arch.py 289 行 / main.py 311 行 / dataset 334 行 / conf 93 行）。
5. 格式：无 CRLF（grep -cP '\r' = 0）、无 tab（grep -cP '\t' = 0）。

### 备注

- 10 票（1-epoch 双模式验证）当时仍 ready-for-agent，本文数字全部取自 canonical facts 清单（其口径为 5-epoch / 2.24e-08 / +5.2%，与实验记录一致），未引用未落盘的 10 票新数字；10 票完成后如需补充 ppsci 路径 1-epoch 证据，属其票内动作。
- utils.py 未纳入 §4 四件嵌入（票面四件 = arch/dataset/example/conf）；§3 讲解按票面只贴 main.py 代码块，loss/metric 以文字描述。
- 未改 mkdocs.yml / mkdocs_zh.yml（注册属 13 票）。
