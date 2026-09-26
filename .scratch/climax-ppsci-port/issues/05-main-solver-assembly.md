# 05: examples/climax/main.py Solver 装配

**Type:** task
**Status:** resolved
**Blocked by:** 02, 03, 04

## What to build

`python main.py mode=train / mode=eval` 一键训练与评估入口跑通单批 dryrun（动态图），完全走 ppsci.solver.Solver 管线。

## 实施要点

- 装配照 `examples/climateformer/main.py:61-125`：dataloader cfg（dataset 子 dict）→ SupervisedConstraint → SupervisedValidator（val + test 两套，test 供 mode=eval）→ `arch.ClimaXClimateBench(**cfg.MODEL)` → Solver 逐参数传入（deep-dive-solver-pipeline.md §Q1 模板）
- lr 用内置 `Cosine`（Q7；warmup off-by-one 已接受）
- optimizer：AdamW 双参数组（decay/no_decay 按参数名 var_embed/pos_embed/time_pos_embed 分组 + freeze 剔除 stop_gradient 参数）——手动构造 optimizer 传入 Solver（逐参数传入是先例）；`ppsci.optimizer` 包装若不支持参数组则 paddle 直构
- loss = `ppsci.loss.FunctionalLoss(w_mse)`、metric = FunctionalMetric 全族（06 票产出；本票可先接桩函数联调，06 完成后替换）
- **权重加载在 Solver 构造前手动完成**：`load_pretrain` 不支持 `{"state_dict": ...}` 包装结构且静默失败（deep-dive-solver-pipeline.md §Q7）——init_state_path strict 直载（`paddle.load` → 拆 `net.` 前缀 → `set_state_dict` 检查 missing/unexpected）+ load_mae_weights 清洗逻辑（interpolate_pos_embed/channel→var 改名/删键）自 PaddleCFD `examples/climax/module.py` 平移进 main.py
- CINN（Q6）：入口 `import paddle` 之前 FLAGS 块（`CLIMAX_USE_CINN`，照 ppcfd feat/climax train.py:17-31 模式）+ env 开时 Solver 传 `to_static=True`；生效判据日志 `Compiling subgraph with CINN backend`
- `mode: train/eval` 分派；evaluate(cfg) 用 `EVAL.pretrained_model_path`

## Acceptance criteria

- [x] 单批 dryrun（动态图，临时 conf 指向本地数据与 init_state，跑完恢复占位）EXIT=0
- [x] 08 等价门单批 loss 对照 ppcfd 版一致
- [x] CINN FLAGS 块在 import paddle 之前（静态检查）

## Answer

### 做了什么

新增 `/home/lkyu/baidu/PaddleScience/examples/climax/main.py`（black 22 / isort 过，88 列内）：hydra `mode=train/eval` 双入口。train：SupervisedConstraint（FunctionalLoss(utils.mse_loss)）→ ITERS_PER_EPOCH → Val/Test 双 SupervisedValidator（FunctionalLoss(make_w_mse_loss) + make_lat_weighted_metrics(lat, denorm, clim=test y_normalization)）→ ClimaXClimateBench(\*\*cfg.MODEL) → Solver 逐参数（to_static=CLIMAX_USE_CINN）→ train()+eval()。evaluate：TEST validator + 手动结构检测加载 EVAL.pretrained_model_path → eval()。

### 验收实录

1. 动态图 dryrun EXIT=0（`DATA.list_train_simu='["ssp126"]'` 子集 77 iters、TRAIN.epochs=1、真实数据+init_state，纯 CLI override 无需恢复）：train loss 4.06407→3.43650；eval 行 `[Eval][Avg] Val/loss 2.50272, Val/w_mse 3.20920, …, Test/w_mse 6.11389, Test/w_rmse 2.47263, Test/w_nrmse 6.43140`。mode=eval 用 best_model.pdparams 复跑 EXIT=0，Test 三指标与训练尾部 eval 完全一致。
2. 指标键集：`w_mse / w_mse_tas_None / w_rmse / w_rmse_tas_None / w_nrmses_tas / w_nrmseg_tas / w_nrmse_tas`，与既往 ppcfd 键名一致。08 票（已 resolved）等价门：数据分区/前向/loss 三对照全部 diff=0.0（train_mse=6.516786、w_mse=4.559730 两侧逐位一致），本票引用其结论。
3. CINN：静态检查 FLAGS 块 char 1124 < import paddle char 1507；CLIMAX_USE_CINN=1 同 dryrun EXIT=0，日志含 `Compiling subgraph with CINN backend ...`，末步 loss 3.43650 及全部 eval 指标与动态图逐字相同。

### 决策记录

- `build_dataset` 强制 `cfg.pop("name")` → train dataset 走 dict 形式带 `"name": "ClimateBenchDataset"`；val/test 需 `set_normalize` 复用 train 的 transform，故直接构造实例传入（build_dataset 对 io.Dataset 实例直通）。train 实例经 `sup_constraint.data_loader.dataset` 取回以拟合 denorm/clim。
- collate_fn 传递路径确认：`ppsci/data/__init__.py:104` 从 dataloader cfg pop "collate_fn" 后直传 io.DataLoader，三个 cfg 均显式 `"collate_fn": batch_transform.default_collate_fn`。
- optimizer：ppsci.optimizer 包装只收 model_list 拍平 parameters（无参数组）→ `paddle.optimizer.AdamW` 直构双参数组（var_embed/pos_embed/time_pos_embed → wd=0，其余 wd=cfg.TRAIN.weight_decay；stop_gradient 冻结参数剔除；beta/epsilon 照 ppcfd）。
- denorm 按 ppcfd train.py 实际用 **out_transform**（非 brief 所写 inp_transform）构造 `Normalize(-mean/std, 1/std)`，此处恒等，与原版装配一致。
- 权重在 Solver 前手动加载：init_state_path strict（unwrap `{"state_dict":...}`、仅当键带 "net." 才剥前缀——修复了纯 state_dict 被剥空的实际 bug，eval 冒烟抓到）；load_mae_weights 自 ppcfd module.py 逐行平移（interpolate_pos_embed 在剥前缀前、channel→var、删 token_embeds/head、shape 过滤、宽松 set_state_dict）。evaluate() 不把 pretrained_model_path 传 Solver（load_pretrain 不支持 wrapper 结构，静默失配）。
- Solver 未用键：TRAIN.pretrained_model_path / TRAIN.checkpoint_path（票面 Solver 参数清单不含；断点续训待后续票如需再加）。
- 已知偏差（留 10 票定夺）：Solver best_model 的 target metric 取**最后一个** validator 的首个指标 → {"Val","Test"} 顺序下监控 Test/w_mse.w_mse_tas_None，而 torch 原版 monitor=val/w_mse；若需对齐可交换 dict 顺序。另 Val/Test 同名 metric 有 cosmetic duplicate 警告（日志键带 validator 前缀，无冲突）。
- dryrun 方式备注：`list_train_simu='["historical"]'` 会得到空 train 集（skip_historical 把纯 historical 序列全部跳过，参考实现的语义，非 bug），改用 '["ssp126"]'（86 窗口→77 iters）。

### 遗留

- 09 票将做正式 CINN 冒烟（本票快速版已过）；10 票处理 best-metric 监控口径。运行产物（/tmp/climax_dryrun_\*、examples/climax/\_\_pycache\_\_）已清理。
