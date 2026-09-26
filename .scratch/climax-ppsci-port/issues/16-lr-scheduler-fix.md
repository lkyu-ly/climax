# 16: lr 调度器修复（F1 平移方案）

**Type:** task
**Status:** resolved
**Blocked by:** None (can start immediately)

## What to build

用 ppcfd 版 LinearWarmupCosineAnnealingLR 的等价实现替换内置 Cosine，修复 step 单位时间尺度错位（根因见下），使 epoch-0 训练动态恢复与基线一致。

## 根因（已闭环诊断，2026-09-26）

ClimaX 原版 warmup_epochs=60/max_epochs=600 配 interval="step" 是 **step 单位**（60 步升满 lr、600 步 cosine 周期、SGDR 重启）；内置 Cosine 两种 by_epoch 模式均按 epoch 单位解释（true: epoch-0 lr 恒 1e-8 实测；false: 47580 步升满），epoch-0 有效 lr 与基线差 400~8000 倍 → 头未训练 → val 停在初始权重水平 1.7675。基线实测 step-50 loss=0.66（已训），ppsci 同位置 3.09（未训）。08 门三层 diff=0.0 证明模型/数据/loss 无关。

## 实施要点

- 源：`git -C /home/lkyu/baidu/PaddleCFD show 'feat/climax:examples/climax/lr_scheduler.py'`（110 行，与 torch 逐点一致的历史已验实现：warmup 每步 +(base-warmup_start)/(warmup_epochs-1)、last_epoch==warmup_epochs 返回 base、SGDR 重启特判）
- 接线方式二选一（先查再选，Answer 记录理由）：
  - **a) paddle LRScheduler 子类化**：get_lr() 复刻递推公式（注意 paddle LRScheduler 构造/step 时序与 torch 差异——ppcfd 版曾刻意不继承；子类化后需逐点单测背书），main.py 以 learning_rate=<scheduler> 传 AdamW，paddle optimizer.step() 自动递推
  - **b) 保自写类**：若 ppsci Solver 训练循环有每步钩子（查 solver.py 的 callbacks/lambda 机制）则手动 step；无钩子则不可行
- **601 步逐点单测（必须）**：新实现 vs ppcfd 版（子进程隔离），步进 0,1,2,…,601（覆盖 warmup 60、边界 last_epoch==warmup_epochs、cosine、重启点）全部 abs_err=0.0（ppcfd 版已知与 torch 逐点一致，等价于 vs torch）
- conf 调整：`TRAIN.lr_scheduler` 段键改为平移版签名（warmup_epochs: 60, max_epochs: 600, learning_rate: 5e-4, warmup_start_lr: 1e-8, eta_min: 1e-8；去掉 by_epoch/iters 依赖）
- main.py：构造 `LinearWarmupCosineAnnealingLR`（或子类化形态）替换内置 Cosine；iters_per_epoch 不再用于 lr
- 复验（Q2 口径）：动态 1 epoch 全量 → epoch-0 val/w_mse 应 ≈0.2601（<10% 偏差门）；CINN 1 epoch → 与动态一致（逐位或舍入级）；日志落 CLIMAX/exps/，跑完清理 PaddleScience 侧产物

## Acceptance criteria

- [x] 601 步逐点单测 abs_err=0.0 全过
- [x] 动态 1 epoch：epoch-0 val/w_mse ≈0.2601（<10% 偏差）——实测 0.26845（+3.21%）
- [x] CINN 1 epoch 与动态一致（舍入级）；编译仍 1 次
- [x] 票 10 回填 resolved（对照表更新）、票 04 Answer 追加 lr 映射勘误一行

## Answer

### 做了什么（2026-09-26，真实执行）

改动三文件（PaddleScience 工作区，未提交）：

- `ppsci/optimizer/lr_scheduler.py`（+81 行）：新增 `LinearWarmupCosineAnnealingLR(paddle.optimizer.lr.LRScheduler)` 子类并导出 `__all__`。递推公式逐字平移自 ppcfd `examples/climax/lr_scheduler.py`，映射：`base_lrs→base_lr`、`group["lr"]（上一步值）→self.last_lr`、list→标量；四个分支（warmup 线性、last_epoch==0、last_epoch==warmup_epochs 返回 base、SGDR 重启特判、cosine 比率式）表达式树保持一致以保证浮点逐位等价
- `examples/climax/main.py`：`lr_scheduler = ppsci.optimizer.lr_scheduler.LinearWarmupCosineAnnealingLR(**cfg.TRAIN.lr_scheduler)` 直接作为 AdamW 的 `learning_rate`（不再 `Cosine(...)()`，iters_per_epoch 不再喂给 lr）
- `examples/climax/conf/climax.yaml`：`TRAIN.lr_scheduler` 段改为 `learning_rate: 5e-4 / warmup_epochs: 60 / max_epochs: 600 / warmup_start_lr: 1e-8 / eta_min: 1e-8`（step 单位；去掉 epochs/warmup_epoch/by_epoch，iters_per_epoch 依赖移除）

### 接线方式：a（paddle LRScheduler 子类化），理由

1. ppsci 既有管线零改动即达 torch 时序：`Solver.__init__` 自动从 `optimizer._learning_rate` 认领调度器（solver.py:187），`train.py:189-190` 对 `by_epoch=False` 的调度器在每次 `optimizer.step()` **之后**调用 `scheduler.step()`——与 ppcfd train.py:177（optimizer 更新后手动 step）、torch 默认时序完全一致。子类只需 `self.by_epoch = False` 即走 per-iter 分支
2. 经验探针实证（paddle 3.4 动态图，/tmp/lr_probe.py）：`optimizer.step()` 不会自动步进调度器；每次 `optimizer.step()` 开始时从调度器当前 `last_lr` 刷新全局 lr 变量并用于本次更新——第 k 次更新用的正是调度器第 k-1 次的值，torch 同构（冒烟：iter-1 applied lr=1e-8 → 8.484e-6 → 1.696e-5，与单测 lr[0..2] 一致）
3. 方式 b（保自写类 + `register_callback_on_iter_end`）可行但多两层活动件：回调挂在 iter 末（晚于日志打印）、须 optimizer 持 float lr 并 `set_lr` 传播（paddle 禁止 optimizer 持 LRScheduler 时调 set_lr）、printer 的 `optimizer.get_lr()` 显示与 checkpoint 的 `LR_Scheduler` state_dict 往返（已验证可用）都要另接；a 全部白得

### 601 步逐点单测（必须项）

`/tmp/lr601_test.py`：双子进程隔离（CUDA_VISIBLE_DEVICES=""，CPU），新实现 vs ppcfd 源（`git show feat/climax:examples/climax/lr_scheduler.py` 落 /tmp，子进程内真 AdamW+param group 包裹）。步进 n=0..601：

- **602/602 点 abs_err == 0.0（含格式化后复跑）**，覆盖：lr[0]=1e-8、warmup 区（lr[30]=2.5424e-4）、边界 lr[59]=5.0000000000000002e-4 / lr[60]=5e-4（恰为 base）、cosine 段（lr[300]=2.9342e-4）、谷底 lr[600]=1e-8（=eta_min）、SGDR 重启点 lr[601]=1.4230700948229906e-08（重启跳升分支）

### 复验（1 epoch 全量双模式）

命令同票 10（`CLIMAX_USE_CINN={0,1}`，seed 42、FP32、batch_size=1、同源初始权重）；日志 `/home/lkyu/baidu/CLIMAX/exps/ppsci_fix_{dynamic,cinn}.log`。

| 指标 | 锚点（ppcfd 动态基线） | ppsci 动态（修复后） | ppsci CINN（修复后） | 修复前（票 10） |
|---|---|---|---|---|
| exit code | — | 0 | 0 | 0 |
| epoch-0 val/w_mse | **0.2601** | **0.26845（+3.21%）** | **0.26845** | 1.76750（+580%） |
| best metric | — | 0.2684498429298401 | 0.2684507966041565 | 1.7674980163574219 |
| epoch-0 val/w_rmse | 0.4767 | 0.51812 | 0.51812 | 1.32947 |
| test/w_mse（1 ep 后） | 0.1339（5 ep 后） | 0.12991 | 0.12991 | 6.11169 |
| iter-1 train loss | — | 0.26611 | 0.26611 | 0.26611 |
| iter-793 段均 loss | — | 0.70920 | 0.70920 | ~2.6–3.2（不降） |
| lr 轨迹 | 60 步升满 5e-4 | iter-1 1e-8 → iter-60 0.00050 → cosine 回落 → iter-793 0.00014 | 同左 | 恒 0.00000 |
| 训练段墙钟（iter1→793） | — | ≈267 s | ≈258 s | 253 s |
| CINN 编译 | — | 0 次 | **1 次**（13:50:11；iter-1 batch_cost 53.69 s 含编译） | 1 次 ≈58 s |
| 稳态 batch_cost | — | ~0.32 s（ips 3.13） | ~0.32 s | ~0.319 s |

- 动态 ↔ CINN：793 个日志轨迹点仅 iter-160 差末位（2.10931 vs 2.10932）；指标打印精度内一致，仅 Val/loss 0.26005/0.26006、Test w_nrmse_tas 0.44925/0.44924、best metric 第 6 位小数差——**舍入级一致**（lr 真实生效后权重被训练，CINN 核与动态在 fp 舍入层发散，属预期；票 10 时代码逐位一致只因 lr≈0 权重未动）
- 40 个日志点的 loss 曲线单调下降（4.41@20 → 1.49@260 → 0.71@793），修复前全程平在 2.6–3.2
- 门：0.26845 vs 0.2601 偏差 +3.21% < 10% **通过**

### 清理

`examples/climax/exps/`（hydra run.dir 产物）与 `examples/climax/__pycache__` 已删；单测/探针脚本留 /tmp（未进两仓）；日志保留于 CLIMAX/exps/。
