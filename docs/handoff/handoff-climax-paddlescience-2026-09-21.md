# 交接：ClimaX 贡献收尾 → PaddleScience 移植调研

交接日期 2026-09-21。交接方：ClimaX paddle 迁移与 PaddleCFD 贡献会话（工作区 /home/lkyu/baidu/CLIMAX）。

## 一、会话纪要（已完成，勿重做）

ClimaX（ClimateBench 气候投影任务）torch→paddle 全链路已完成并交付 PaddleCFD：

1. **迁移与对齐**：paconvert 转换 + 手动修复（timm 1.0.24 闭包本地化、Lightning 舍弃自建训练循环）；前向对齐 mean_abs **2.24e-08**；同源初始权重 5-epoch 训练最终 test 指标 **within 5.1%** of torch baseline；CINN（`CLIMAX_USE_CINN=1` 环境变量 + `to_static(full_graph=True)`）steady-state 加速 **+5.2%**（RTX 4060 Ti / FP32 / bs=1 / 793 steps/epoch，编译 ~90s 一次）。三轮测试数字全在实验记录。
2. **PaddleCFD 贡献**：`feat/climax` 分支（基于官方 develop `f366aae`），**用户已 commit `5df8c0a`**，PR message 已写好（/home/lkyu/baidu/test/tmp.md，待用户自行创建 PR）。
3. **状态确认**：climax→paddlecfd 工作宣告完成。

## 二、本地工作区状态

| 位置                                | 状态                                                                                                                                                                                                                                                                                                                                                                   |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/home/lkyu/baidu/CLIMAX`（开发仓） | main @ `3d07e9d`（已推送至 github.com/lkyu-ly/climax 至 `0c18518`，其后 4 个 commit 未推送，用户自行决定）。未跟踪：`docs/superpowers/plans/2026-09-21-climax-ppcfd-contribution.md`（贡献计划文档，可随手 commit）。`climax_torch/`=对照基线（勿动），`climax_paddle/`=对齐验证形态（贡献的源）。实验日志 `ClimaX迁移实验记录.md`（全部过程与数字，含三次测试与踩坑） |
| `/home/lkyu/baidu/PaddleCFD`        | `feat/climax` @ `5df8c0a`（用户已 commit）。upstream=PaddlePaddle/PaddleCFD（fetch-only），origin=lkyu-ly fork。`.spec-workflow/` 已通过 `.git/info/exclude` 本地排除。5 个历史模型贡献分支（feat/mpp、feat/phye2e、feat/poseidon、feat/prose_fd、feat/G-FNO）保留在本地可参照                                                                                         |
| 数据                                | `/home/lkyu/baidu/CLIMAX/dataset/climatebench/{raw,5.625deg}`（ClimateBench v1.0 + 5.625° 重采样版）；test 目录含指向 train_val 的 historical 软链接（`load_x_y` 的 ssp 拼接历史期所需，usage.md 未写的隐含要求）                                                                                                                                                      |
| 权重                                | `/home/lkyu/baidu/CLIMAX/models/climax_torch/5.625deg.ckpt`（官方 412MB，seed 42 下 bit 级可重建导出）；`models/climax_paddle/climax_initial.pdparams`（含随机新头的完整初始态，对齐测试用）；torch 侧导出 `climax_initial.pt` 同目录                                                                                                                                  |
| 环境                                | conda `paddletorch`（torch 2.11+cu128 与 paddle 3.4 同环境，uv pip 装包；paddle sdpa 在本机 libcuda 加载异常——attention 恒走手工分支）；conda `climdat`（xesmf 重采样专用）                                                                                                                                                                                            |

## 三、PaddleCFD 贡献经验（下一仓直接复用）

**安置模式（mpp 先例）**：`ppcfd/models/<model>/` 放模型构建代码（本例：arch/climatebench/pos_embed + timm 闭包子包 `timm/`，`__init__.py` 带 Apache 2.0 PaddlePaddle 头并导出）；`examples/<model>/` 放外围（一键 train.py、数据管线 module/datamodule/dataset、lr_scheduler/metrics/normalize、configs、regrid 数据脚本、README）。`ppcfd/models/__init__.py` 加 try-import 注册块（仿 confild）。examples 不打包：平级 import + 模型绝对 import `from ppcfd.models.<model>...`，`PYTHONPATH=仓库根` 运行。

**入口规范**：一键 `python train.py` 无 CLI（配置全在 configs/\*.yaml）；编排值逐项对照原版 yaml 恢复原作者默认（本例 max_epochs 恢复 50）；数据/输出路径占位化（`./data/...`）；eval 内建（训练毕自动 best checkpoint 跑 test）。

**注释与措辞**（`/home/lkyu/baidu/工作总结/ppmat贡献交付注释与措辞规范.md`，六规则已全量执行过三轮）：英文简练事实、无内部用语（Task/票号/上游）、无设计史（dropped/removed 叙事）、无第三方 bug 指控、来源标注统一 `# Adapted from <url>` 署名式；README 英文官方口径、数字带口径不外推、检查点称 released checkpoint、CINN 单独章节。

**流程经验**：官方 develop 与 fork 同步核对（本例 `f366aae` 无前进）→ 从官方 develop 分 feat 分支 → 改动全程不 commit 等用户审查（用户亲自润色 README 后自行 commit）→ PR message 只写 Alignment/CINN/Usage 三部分 → 用户手动建 PR（对官方仓库的一切写操作归用户）。tools/（权重转换/对齐工具）与 tests/ 不进上游。

**改动面参考**：本例 = `M ppcfd/models/__init__.py`（+10）+ `ppcfd/models/climax/`（约 1520 行）+ `examples/climax/`（约 1100 行）；验证 = import 链冒烟 + 双模式（动态/CINN）各 1 epoch 全量 exit 0 + 副产物零残留（**pycache** 也要清）。

## 四、下一步任务（本会话只定义，不做）

**背景（用户原话要点）**：由于 reviewer 意见，该模型要转入 **PaddleScience** 仓库。用户发现 PaddleScience 与 PaddleCFD 在 "pp\*\*\*"（模型包）与 example 组织方式上高度相似。

1. **先调研**：PaddleScience 与 PaddleCFD 的相似结构与不同之处（重点：模型包目录惯例与命名、example 组织、入口/配置形态、README 骨架、注册方式、贡献流程与分支规范、是否同样接受复现型贡献、版权头与注释要求）。产出对照结论后再定移植方案。
2. **后移植**：根据调研结果把已完成的 paddlecfd 贡献（`feat/climax` @ `5df8c0a`）转（复制）到 PaddleScience 对应结构；预期大部分文件可平移，重点在包名（ppcfd→PaddleScience 的对应命名空间）、import 路径、注册点、README 口径的适配。
3. PaddleScience 本地仓库位置未知（`/home/lkyu/baidu/` 下未见，需先 clone 或询问用户）；fork/分支/PR 流程待调研后与用户确认。

## 五、关键工件索引

- 实验日志（全过程+三次测试数字）：`/home/lkyu/baidu/CLIMAX/ClimaX迁移实验记录.md`
- 贡献计划：`/home/lkyu/baidu/CLIMAX/docs/superpowers/plans/2026-09-21-climax-ppcfd-contribution.md`
- 前序计划：`docs/superpowers/plans/2026-09-20-climax-paddle-port.md`（迁移）、`2026-09-21-climax-cinn.md`（CINN）
- CINN 踩坑手册：`/home/lkyu/baidu/工作总结/CINN动转静训练踩坑手册.md`
- 注释与措辞规范：`/home/lkyu/baidu/工作总结/ppmat贡献交付注释与措辞规范.md`
- ppcfd 贡献 diff：`/home/lkyu/baidu/PaddleCFD` feat/climax @ `5df8c0a`（对照 `f366aae`）
- PR message：`/home/lkyu/baidu/test/tmp.md`
- sdd 账本（三份，过程细节）：`/home/lkyu/baidu/CLIMAX/.superpowers/sdd/{2026-09-20-climax-paddle-port,2026-09-21-climax-cinn,2026-09-21-climax-ppcfd-contribution}/progress.md`

## 六、建议技能（下一会话按触发条件调用）

| 技能                                       | 何时                                                    |
| ------------------------------------------ | ------------------------------------------------------- |
| superpowers:brainstorming                  | 移植方案设计前（调研结论 → 结构映射方案）               |
| superpowers:writing-plans                  | 调研完成、移植方案确定后展开实施计划                    |
| superpowers:subagent-driven-development    | 按计划执行（本项目一贯并行/串行混合模式）               |
| grilling                                   | 有需用户拍板的结构性决策时（用户有 /grill-me 使用习惯） |
| superpowers:systematic-debugging           | 移植后冒烟报错时                                        |
| superpowers:verification-before-completion | 每轮声称完成前（副产物/规范自查）                       |

## 七、纪律（沿用）

- 对官方源仓库（PaddlePaddle/\*）的一切写操作归用户手动执行；agent 只做本地与 fork 侧
- 贡献改动全程不 commit，等用户审查；禁 SHA 标记文件；最小努力、敏捷、疑难先询问
