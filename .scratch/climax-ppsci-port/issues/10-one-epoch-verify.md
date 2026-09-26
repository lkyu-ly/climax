# 10: 1 epoch 全量双模式验证

**Type:** task
**Status:** resolved（票 16 平移 LinearWarmupCosineAnnealingLR 修复 lr 时间尺度错位后复验通过：epoch-0 val/w_mse 0.26845 vs 锚点 0.2601，偏差 +3.21% < 10% 门；对照表已更新为修复后数字）
**Blocked by:** 09

## What to build

Q2 决策 B 的正式验证：动态图与 CINN 各 1 epoch 全量（793 训练 + 89 验证批 + test），与既往运行无差异即判等价。

## 实施要点

- 同源初始权重（climax_initial.pdparams）、seed 42、FP32、batch_size=1，与既往口径一致
- 对照锚点（既往 paddle 侧）：epoch-0 val/w_mse = 0.2601（动态）；初始 test/w_mse ≈ 4.5 量级
- 量级一致即判等价（Q2：不做精确对齐复验、不做 5 epoch）；差异大则记录现象、按疑难流程停下询问
- CINN 模式同时记录：编译一次的耗时、epoch-0 训练段墙钟（文档数字口径备料）
- 日志落盘 `/home/lkyu/baidu/CLIMAX/exps/` 下；跑完清理 PaddleScience 侧运行产物（保留 log 于 CLIMAX 侧）

## Acceptance criteria

- [x] 双模式 exit 0
- [x] 指标量级对照表记录于本票 Answer（供 11/15 与 PR message 引用；**2026-09-26 已更新为票 16 修复后数字**）

## Answer

### 执行实录（2026-09-26，真实执行）

命令（两跑同构，仅 CLIMAX_USE_CINN 0/1）：`cd /home/lkyu/baidu/PaddleScience/examples/climax && CLIMAX_USE_CINN={0,1} PYTHONPATH=/home/lkyu/baidu/PaddleScience python main.py mode=train DATA.root_dir=.../5.625deg TRAIN.init_state_path=.../climax_initial.pdparams TRAIN.epochs=1`（python = paddletorch env，paddle 3.4，单卡）。日志：`/home/lkyu/baidu/CLIMAX/exps/ppsci_1epoch_dynamic.log`、`ppsci_1epoch_cinn.log`。

### 指标量级对照表（供 15 与 PR message 引用）

| 指标 | 既往 paddle 动态基线（PaddleCFD climax_paddle） | ppsci 动态图（CLIMAX_USE_CINN=0） | ppsci CINN（CLIMAX_USE_CINN=1） |
|---|---|---|---|
| exit code | — | 0 | 0 |
| epoch-0 val/w_mse | **0.2601** | **1.76750** | **1.76750** |
| epoch-0 val/w_rmse | 0.4767 | 1.32947 | 1.32947 |
| test/w_mse（该 epoch 后） | 0.1339（5 ep 后）；票载初始 ≈4.5 量级 | 6.11169 | 6.11169 |
| epoch-0 训练 loss 量级 | 0.15–1.0 | iter1=0.26611 → 稳定 ~2.6–3.2 | 与动态逐位同（iter1=0.26611） |
| 训练段墙钟（iter1→793） | — | 253 s（11:42:11→11:46:24） | 252 s（11:49:46→11:53:58） |
| 全程墙钟 | — | ≈287 s | ≈350 s |
| CINN 编译 | — | — | 1 次，≈58 s（"Compiling subgraph with CINN backend" 仅出现 1 次，11:48:52；iter-1 batch_cost 57.74 s 含编译） |
| 稳态 batch_cost | — | ~0.319 s（ips 3.13） | ~0.318 s（ips 3.14） |

- 动态 ↔ CINN：**全部指标逐位一致**（Val/w_mse 1.76750、Val/w_rmse 1.32947、Test/w_mse 6.11169、三个 w_nrmse 及各 iter train loss 全同，best metric 同为 1.7674980163574219）。Q2 决策 B 的 CINN↔动态等价成立；本工作负载下 CINN 无加速（稳态吞吐相同，编译为纯开销）。
- best_model 监控确认正确：`[Eval][Epoch 1][best metric: 1.7674980163574219]` == Val/w_mse（Val validator 在最后，监控对象无误）。

### 异常现象（按票规停下，未自行调试）

**ppsci 两侧 vs 既往基线锚点不通过**：epoch-0 val/w_mse = 1.76750 vs 锚点 0.2601，偏差 +580%，远超 <10% 门限；train loss 量级也差 ~5–10 倍（~2.6–3.2 vs 0.15–1.0）。test 侧 6.11 vs "初始 ≈4.5 量级"（同量级但 +36%）。

客观事实（非调试结论，供决策）：

1. warmup 60 epoch（by_epoch）下首 epoch lr≈0（两侧日志 lr 均显示 0.00000），权重经 1 epoch 几乎不动，故两侧 epoch-0 val 差异体现的是"初始权重下的指标口径"差异，而非训练动态差异。
2. ppsci Test/loss 首个 batch = 4.55758，与 08 票逐位对照工具给出的 w_mse = 4.55973（identity denorm、test 首样本，两侧 abs_diff=0.0）基本重合；ppsci 训练 iter-1 loss 0.26611 与基线量级一致，随后样本升至 ~3–4.5。
3. 08 票已证：数据三分区（793/89/21）、前向输出、单样本 loss 路径（identity denorm 口径）两侧全 diff=0.0。
4. 综上，分歧点大概率在 val/test 指标的反归一化或聚合口径（08 门为简化条件 identity denorm，实际运行两侧各自的 y_normalization/climatology 路径），以及逐样本 loss 量级随样本变化的分布——需用户定夺下一步（疑难流程）。

### 清理

PaddleScience 侧 `examples/climax/exps/`（hydra run.dir 产物）与 `__pycache__` 已删除；工作树仅余移植代码本身；日志保留于 `/home/lkyu/baidu/CLIMAX/exps/ppsci_1epoch_{dynamic,cinn}.log`。

### 修复后复验（2026-09-26，票 16 平移 LinearWarmupCosineAnnealingLR；上表保留为修复前实录）

根因（票 16 已闭环）：内置 Cosine 两种 by_epoch 模式均按 epoch 单位解释 warmup，epoch-0 有效 lr 与基线差 400~8000 倍，头未训练。修复后同命令复跑（日志 `/home/lkyu/baidu/CLIMAX/exps/ppsci_fix_{dynamic,cinn}.log`），**对照表以本表为准（供 11/15 与 PR message 引用）**：

| 指标 | 既往 paddle 动态基线 | ppsci 动态（修复后） | ppsci CINN（修复后） |
|---|---|---|---|
| exit code | — | 0 | 0 |
| epoch-0 val/w_mse | **0.2601** | **0.26845（+3.21%，<10% 门过）** | **0.26845** |
| epoch-0 val/w_rmse | 0.4767 | 0.51812 | 0.51812 |
| test/w_mse（1 ep 后） | 0.1339（5 ep 后）；初始 ≈4.5 | 0.12991 | 0.12991 |
| iter-1 train loss | — | 0.26611 | 0.26611 |
| iter-793 段均 loss | — | 0.70920（曲线 4.41→1.49→0.71 单调降） | 0.70920（793 点仅 iter-160 差末位 1e-5） |
| lr 轨迹 | 60 步升满 5e-4 后 cosine | iter-1 1e-8 → iter-60 5e-4 → iter-793 1.4e-4 | 同左 |
| 训练段墙钟（iter1→793） | — | ≈267 s | ≈258 s |
| CINN 编译 | — | 0 | 1 次（iter-1 batch_cost 53.69 s 含编译） |
| 稳态 batch_cost | — | ~0.32 s | ~0.32 s |

- 动态 ↔ CINN 为**舍入级一致**（Val/w_mse、Test/w_mse、w_rmse、w_nrmse 打印精度内全同；Val/loss 0.26005/0.26006、best metric 第 6 位小数差）——lr 真实生效后权重被训练，CINN 核与动态在 fp 舍入层发散，属预期。
- 修复前"异常现象"第 1 条的猜测（lr≈0 导致权重不动）即被证实为根因；val/test 指标口径本身两侧一致，无需进一步追查。
