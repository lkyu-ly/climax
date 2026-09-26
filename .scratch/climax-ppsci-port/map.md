# Wayfinder map: climax-ppsci-port

Label: wayfinder:map

## Destination

ClimaX 以 ppsci 惯例完整落地 PaddleScience fork 的 `feat/climax` 分支：`ppsci/arch/paddle_timm/` + `ppsci/arch/climax/` + 主包 dataset + `examples/climax/`（hydra/Solver 形态）+ 双语文档与四处注册 + pre-commit 格式化，全部改动**不 commit** 等用户审查；验证 = 1 epoch 全量双模式（动态/CINN）与既往运行无差异即判等价。

## Notes

- **本图携带执行**（task 票即工作块）：允许 subagent 波次并行多票（superpowers:subagent-driven-development），突破 wayfinder 默认的"单会话单票"。
- 执行仓库：`/home/lkyu/baidu/PaddleScience`（fork，分支 `feat/climax` 自 develop 新建；执行前 controller 先核对官方 upstream 是否前进）。对照源（只读）：`/home/lkyu/baidu/PaddleCFD` feat/climax @ `5df8c0a`。
- 环境：conda `paddletorch`（`/home/lkyu/miniconda3/envs/paddletorch/bin/python`，paddle 3.4 + torch 2.11）；数据 `/home/lkyu/baidu/CLIMAX/dataset/climatebench/5.625deg`（test 目录 historical 软链接已在位）；初始权重 `/home/lkyu/baidu/CLIMAX/models/climax_paddle/climax_initial.pdparams`。
- 关键工件：调研 `docs/superpowers/research/2026-09-26-ppcfd-vs-ppsci-structure.md` + 同日 `deep-dive-{solver-pipeline,climax-side,conventions}.md`；实验记录 `ClimaX迁移实验记录.md`；措辞规范 `/home/lkyu/baidu/工作总结/ppmat贡献交付注释与措辞规范.md`；旧 PR message `/home/lkyu/baidu/test/tmp.md`（ppsci 版待终数字）。
- 技能：执行用 superpowers:subagent-driven-development；等价门 TDD（08 票）；zh 文档用 humanizer-zh（12 票）；每票收尾 superpowers:verification-before-completion。
- 纪律：不 commit/不 push（用户建 PR）；最小努力、敏捷、疑难先询问；禁 SHA 标记文件；tools/ 与 tests/ 不进上游；数据/权重/运行产物不入库（含 \_\_pycache\_\_）。
- 复用优先（Q7）：ppsci 已有实现一律优先（内置 Cosine lr、`ppsci.utils.initializer.trunc_normal_`、既有 transform）；Functional\* 仅用于数值不等价处（Q5）。

## Decisions so far

（导入自 2026-09-26 grilling 会话：Q1–Q9 全拍板 + 授权裁定）

- Q1 ppcfd 侧不动：PR 用户自行关闭，feat/climax 留本地作参照
- Q2 验证深度 = B：1 epoch 全量双模式，与既往无差异即判等价；核心逻辑改动走端到端宽范围 TDD 式等价重构（08 票承担）
- Q3 安置：`ppsci/arch/climax/{__init__,arch,climatebench,pos_embed}.py` 目录 + `ppsci/arch/paddle_timm/`（对标 paddle_harmonics 先例的位置与命名法；文件头 SPDX/上游署名风格选择性参考）
- Q4 RFC 搁置，用户需要时另起
- Q5 loss/metric = FunctionalLoss + FunctionalMetric 平移原公式（数值精确；内置 LatitudeWeightedRMSE 的 linspace 权重数值不等价，不用）
- Q6 CINN 完整保留在 example 实现内（FLAGS 块 + to_static 联动 + 文档数字），不推框架层
- Q7 复用优先全局生效（内置 Cosine 接受 warmup off-by-one；trunc_normal_ 换 `ppsci.utils.initializer`）
- Q8 纯 Solver 惯例（eval_during_train + mode=eval 两段式），放弃训练毕自动 test
- Q9 零上传：文档留 zenodo 下载说明 + regrid 脚本放 example + 权重只写转换出处（HF tungnd/climax）
- 文档语言：en 按 ppsci 骨架改写 ppcfd README；zh 不直译、用 humanizer-zh 重措辞
- pos_embed 保 ppcfd 数值布局（两仓 2D 版实测不一致，权重绑定 ppcfd 版）
- 16 票：内置 Cosine 换平移版 `LinearWarmupCosineAnnealingLR`（入主包 lr_scheduler.py + `__all__`），根因 = warmup/max_epochs 是 step 单位而被内置 Cosine 按 epoch 解释的时间尺度错位；601 步逐点 abs_err=0.0，修复后 1-epoch val/w_mse 0.26845（vs 锚点 0.2601 +3.21% 过门）

## Not yet specified

- PR message 已产出草稿（15 票，`/home/lkyu/baidu/test/ppsci-climax-pr.md`），用户润色后建 PR
- reviewer 反馈应对（paddle_timm 命名、pos_embed 合并建议等）：无法预判；出现时按 Q3/Q5 决策的实测证据回应

## Out of scope

- ppcfd 仓任何改动（Q1）；其 PR 关闭动作（用户手动）
- 数据/权重上传 bcebos 或任何托管（Q9）
- RFC 文档（Q4）
- 5 epoch 全量复验（Q2 选 B；reviewer 要求时另起 effort）
- tools/、tests/ 进上游；数据/权重入库；jointContribution 路径（调研已裁 examples）
