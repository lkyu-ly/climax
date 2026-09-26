# 09: 冒烟——import 链 + 单批双模式

**Type:** task
**Status:** resolved
**Blocked by:** 05, 06

## What to build

ppsci 侧完整链路（arch + dataset + example + Solver）首次端到端跑通：动态图与 CINN 各一次单批 dryrun。

## 实施要点

- import 链：`import ppsci`（新注册不破坏主包）、`from ppsci.arch.climax import ...`、`from ppsci.data.dataset.climatebench_dataset import ...`；examples/climax 内无 ppcfd 残留（grep）
- 完整构造：cfg.MODEL 全参数构造参数量 111,831,040
- 单批 dryrun 双模式：`CLIMAX_USE_CINN=0` 与 `=1` 各跑 1 批（临时 conf 指向本地数据与 init_state，跑完恢复占位）
- **CINN 在 Solver 路径下的首次实测**（Q6 风险点）：确认日志出现 `Compiling subgraph with CINN backend`；若不生效，按踩坑手册排查（to_static 包的是 forward_helper，FLAGS 分层理论上可共存——deep-dive-solver-pipeline.md §Q6）；疑难即停走询问
- 初始指标量级：随机头 test/w_mse ≈ 4.5 量级（既往 4.545/4.560 同量级）
- 副产物清理（含 \_\_pycache\_\_）

## Acceptance criteria

- [x] 双模式 exit 0、指标键集与既往一致
- [x] CINN 判据日志在位（或如实记录阻塞）
- [x] conf 占位值恢复、无副产物残留

## Answer

### 验收实录（2026-09-26，RTX 4060 Ti，paddle 3.4.0，PYTHONPATH=/home/lkyu/baidu/PaddleScience，cwd=examples/climax）

1. **import 链三件全过**：`import ppsci`（本地仓副本）、`from ppsci.arch.climax import ClimaX, ClimaXClimateBench` + `pos_embed.interpolate_pos_embed`、`from ppsci.data.dataset.climatebench_dataset import ClimateBenchDataset, Normalize`；主包注册不破坏——`ppsci.arch.__all__` 含 ClimaX/ClimaXClimateBench 且与子模块同一对象，无关注册（MLP/ResNet/Climateformer/CSVDataset/ERA5SQDataset）正常导入。`grep -rn ppcfd examples/climax` 零命中。
2. **完整构造参数量精确匹配**：hydra compose 取 cfg.MODEL 全 15 个 kwarg 构造 ClimaXClimateBench，total params = **111,831,040**（assert 通过；freeze_encoder=true 下 trainable 11,094,016 / frozen 100,737,024）。
3. **双模式 dryrun**（纯 CLI override：`mode=train DATA.root_dir=<真实数据> TRAIN.init_state_path=<climax_initial.pdparams> TRAIN.epochs=1 DATA.list_train_simu='["ssp126"]'`，77 iters，无 conf 文件改动）：
   - `CLIMAX_USE_CINN=0`：**EXIT=0**。train loss 4.06407→3.43650；`[Eval][Avg]` Test/w_mse 6.11389、Test/w_rmse 2.47263、Test/w_nrmse 6.43140、Val/loss 2.50272、Val/w_mse 3.20920——与 05 票记录逐位一致。eval 输出顺序 Test 在前 Val 在后（main.py 更新后顺序），best_model 正常保存。
   - `CLIMAX_USE_CINN=1`：**EXIT=0**。日志含 `add_cinn_pass.cc:334] Compiling subgraph with CINN backend ...`（Solver 路径 CINN 首次实测生效，Q6 关闭）；末步 loss 3.43650 与全部 eval 指标与动态图逐位相同。
   - 指标键集两模式一致且与既往相同：`{Test,Val}/loss、w_mse、w_mse_tas_None、w_rmse、w_rmse_tas_None、w_nrmses_tas、w_nrmseg_tas、w_nrmse_tas`。
4. **初始指标量级**：`mode=eval EVAL.pretrained_model_path=<init_state>` EXIT=0（strict 加载 missing=[] unexpected=[]）。单批（首个 test 样本，出库组件 dataset+utils.make_lat_weighted_metrics 直接计算）初始 **test/w_mse = 4.559772**，≈4.5 量级，与既往 4.545（paddle_dryrun_task5）/4.560（ppcfd cinn_smoke limit_batches=1 口径 4.559733）同量级。注意：既往 4.5x 均为**单批**口径；init_state 全 test 集（21 批）平均 Test/w_mse=6.11435（w_rmse 2.47272、w_nrmses 1.19295、w_nrmseg 1.04774、w_nrmse 6.43166），10 票对照时勿混口径。
5. **清理**：examples/climax/exps/{climate_projection_dyn,cinn,eval_init} 与全仓 `__pycache__` 已删，/tmp 脚本已删；conf 占位值未动（root_dir=./data/... init_state_path="" epochs=50 全量 simu 列表）。冒烟日志留存：`/home/lkyu/baidu/CLIMAX/exps/ppsci_smoke_{dyn_train,cinn_train,eval_init}.log`。

### 备注

- 手动 hydra compose 需先 `import ppsci`：ppsci_default/train_default/eval_default/exclude_keys_default 是 `ppsci/utils/config.py` 在 import 时注册进 ConfigStore 的，不装包直接 compose 会报 MissingConfigException。
- 手动 default_collate_fn 出来是 numpy（paddle.io.DataLoader 才做 to_tensor），脚本对照时需自行 to_tensor。
